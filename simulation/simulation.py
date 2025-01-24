# Logic the runs the simulation
from datetime import datetime, timedelta
import json
import heapq
from simulation.utils import is_work_time, advance_to_work_time, get_task_duration, get_condition_probability, advance_time_in_seconds
from simulation.reporting import print_processing_times_and_utilization, save_simulation_report
from simulation.event import Event
import logging
import random
import re
import pandas as pd

# Configure logging
logging.basicConfig(
    filename='simulation_log.txt',
    filemode='w',  # Overwrite the log file
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Helper functions for resource management
def assign_resource(resource_name, active_resources, available_resources):
    if active_resources[resource_name] < available_resources.get(resource_name, 0):
        active_resources[resource_name] += 1
        return True
    return False

def release_resource(resource_name, active_resources):
    if active_resources[resource_name] > 0:
        active_resources[resource_name] -= 1

# Function to schedule all tokens
def schedule_tokens(json_file_path, max_arrival_count, arrival_interval_minutes):
    """
    Schedules tokens to start the process based on the "Start" node and provided parameters.

    Parameters:
        json_file_path (str): Path to the JSON file.
        max_arrival_count (int): The maximum number of tokens to schedule.
        arrival_interval_minutes (int): The interval in minutes between token arrivals.

    Returns:
        list: A list of dictionaries containing token ID and start time.
    """
    # Load the JSON file
    with open(json_file_path, "r") as file:
        process_model_data = json.load(file)

    # Extract the "Start" node
    nodes = process_model_data.get("nodes", [])
    start_node = None
    for node in nodes:
        if node.get("type") == "Start":
            start_node = node["id"]
            break

    if not start_node:
        raise ValueError("No 'Start' node found in the JSON file.")

    # Schedule tokens
    schedule = []
    current_time = datetime.now()
    for token_id in range(1, max_arrival_count + 1):
        schedule.append({
            "token_id": f"Token-{token_id}",
            "start_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_node": start_node
        })
        current_time += timedelta(minutes=arrival_interval_minutes)

    return schedule

# Function to start token processing
def start_token_processing(token_id, task_name, current_time, simulation_metrics, active_resources, available_resources, 
                           resource_busy_periods, activity_processing_times, event_queue, active_tokens, resource_wait_queue):
    # Determine task duration
    task_duration = get_task_duration(task_name, simulation_metrics)

    # If duration is None or zero, skip processing and schedule end event immediately
    if task_duration is None or task_duration == 0:
        heapq.heappush(event_queue, Event(current_time, token_id, task_name, 'end'))
        logging.info(f"Token {token_id} skipped processing for task '{task_name}' due to no duration or zero processing time.")
        return True

    # Determine the resource required for the task
    resource_name = simulation_metrics.loc[simulation_metrics['name'] == task_name, 'resource'].values
    if resource_name.size > 0 and pd.notna(resource_name[0]):
        resource_name = resource_name[0]

        # Check if resource is available
        if not assign_resource(resource_name, active_resources, available_resources):
            # If resource unavailable, add token to the wait queue
            if token_id not in [t[0] for t in resource_wait_queue[resource_name]]:
                resource_wait_queue[resource_name].append((token_id, task_name))
                logging.info(f"Token {token_id} added to wait queue for resource '{resource_name}'.")

            # Mark the token as waiting if not already marked
            if active_tokens[token_id]['wait_start_time'] is None:
                active_tokens[token_id]['wait_start_time'] = current_time
                logging.info(f"Token {token_id} wait start time set to {current_time}.")
            return False  # Resource unavailable

        # If the token was waiting, calculate and log the wait time
        if active_tokens[token_id]['wait_start_time'] is not None:
            wait_time = (current_time - active_tokens[token_id]['wait_start_time']).total_seconds() / 60  # Convert to minutes
            active_tokens[token_id]['total_wait_time'] += wait_time
            active_tokens[token_id]['wait_start_time'] = None  # Reset wait start time
            if task_name not in activity_processing_times:
                activity_processing_times[task_name] = {"durations": [], "wait_times": [], "tokens_started": 0, "tokens_completed": 0}
            activity_processing_times[task_name]["wait_times"].append(wait_time)
            logging.info(f"Token {token_id} waited {wait_time:.2f} minutes for task '{task_name}'.")

        # Mark the resource as busy
        resource_busy_periods[resource_name].append([current_time, None])

    # Log the start time of the task
    active_tokens[token_id]['task_start_time'] = current_time

    # Increment the number of tokens started for this activity
    if task_name not in activity_processing_times:
        activity_processing_times[task_name] = {"durations": [], "wait_times": [], "tokens_started": 0, "tokens_completed": 0}
    activity_processing_times[task_name]["tokens_started"] += 1

    # Schedule the end event for the task
    heapq.heappush(event_queue, Event(current_time + timedelta(minutes=task_duration), token_id, task_name, 'end'))
    logging.info(f"Token {token_id} started processing task '{task_name}' at {current_time} for {task_duration} minutes.")

    return True

# Function to complete a token's activity
def complete_activity(token_id, task_name, current_time, simulation_metrics, active_tokens, active_resources, 
                      resource_busy_periods, transitions_df, event_queue, activity_processing_times, completed_tokens, paths):
    logging.info(f"Token {token_id} completed task '{task_name}' at {current_time}")

    # Fetch task duration
    task_duration = get_task_duration(task_name, simulation_metrics)

    # Calculate and log processing time if task had a valid duration
    start_time = active_tokens[token_id].get('task_start_time', None)
    if start_time and task_duration is not None:
        process_time = (current_time - start_time).total_seconds() / 60  # Convert to minutes
        if task_name not in activity_processing_times:
            activity_processing_times[task_name] = {"durations": [], "wait_times": [], "tokens_started": 0, "tokens_completed": 0}
        activity_processing_times[task_name]["durations"].append(process_time)
        activity_processing_times[task_name]["tokens_completed"] += 1
        logging.info(f"Token {token_id}: Task '{task_name}' processing time = {process_time:.2f} minutes.")

    # Release resources if applicable
    resource_name = simulation_metrics.loc[simulation_metrics['name'] == task_name, 'resource'].values
    if task_duration is not None and resource_name.size > 0 and pd.notna(resource_name[0]):
        resource_name = resource_name[0]
        release_resource(resource_name, active_resources)
        for period in resource_busy_periods[resource_name]:
            if period[1] is None:
                period[1] = current_time
                break

    # Handle "Stop" activity
    activity_type = simulation_metrics.loc[simulation_metrics['name'] == task_name, 'type'].values
    activity_type = activity_type[0] if activity_type.size > 0 else "unknown"

    # Check if this is the last activity in the path
    if task_name not in transitions_df['from'].values:  # task_name does not lead to any other task
        active_tokens[token_id]['end_time'] = current_time
        if token_id not in [token['token_id'] for token in completed_tokens]:  # Ensure no duplicates
            completed_tokens.append({'token_id': token_id, **active_tokens[token_id]})  # Add completed token
        logging.info(f"Token {token_id} completed the process at {current_time}.")
        return

    # Schedule next tasks
    next_tasks = transitions_df[transitions_df['from'] == task_name]
    if activity_type == "gateway":
        # Fetch probabilities for each outgoing conditional path
        probabilities = []
        tasks = []
        for _, row in next_tasks.iterrows():
            condition_type = row['type']  # Example: "condition-yes" or "condition-no"
            probability = get_condition_probability(simulation_metrics, task_name, condition_type)
            probabilities.append(probability)
            tasks.append(row['to'])

        # Normalize probabilities to cumulative distribution
        cumulative_probabilities = []
        cumulative_sum = 0
        for prob in probabilities:
            cumulative_sum += prob
            cumulative_probabilities.append(cumulative_sum)

        # Generate a random number to determine the path
        rand = random.random()
        chosen_index = next(i for i, cp in enumerate(cumulative_probabilities) if rand <= cp)
        chosen_task = tasks[chosen_index]

        heapq.heappush(event_queue, Event(current_time, token_id, chosen_task, 'start'))
        logging.info(f"Token {token_id} chose path to '{chosen_task}' at Gateway '{task_name}' with random value {rand:.4f}")
    else:
        # Non-Gateway: Schedule all outgoing transitions
        for _, row in next_tasks.iterrows():
            next_task = row['to']
            if next_task not in active_tokens[token_id]['completed_tasks']:
                heapq.heappush(event_queue, Event(current_time, token_id, next_task, 'start'))
                logging.info(f"Token {token_id} scheduled for next task '{next_task}' at {current_time}")

# Function to process tokens
def process_tokens(event_queue, active_tokens, active_resources, available_resources, resource_busy_periods, 
                   activity_processing_times, transitions_df, simulation_metrics, resource_wait_queue, completed_tokens, paths):
    while event_queue:
        event = heapq.heappop(event_queue)
        current_time = event.time
        token_id = event.token_id
        task_name = event.task_name

        if event.event_type == 'start':
            if start_token_processing(token_id, task_name, current_time, simulation_metrics, active_resources, 
                                       available_resources, resource_busy_periods, activity_processing_times, 
                                       event_queue, active_tokens, resource_wait_queue):
                active_tokens[token_id]['current_task'] = task_name
        elif event.event_type == 'end':
            complete_activity(token_id, task_name, current_time, simulation_metrics, active_tokens, active_resources, 
                              resource_busy_periods, transitions_df, event_queue, activity_processing_times, completed_tokens, paths)

# Main simulation function
def discrete_event_simulation(max_arrival_count, arrival_interval_minutes, simulation_days, paths, simulation_metrics, 
                              start_time, xpdl_file_path, transitions_df, start_tasks):
    simulation_end_date = start_time + timedelta(days=simulation_days)

    # Normalize column names to lowercase for consistency
    simulation_metrics.columns = map(str.lower, simulation_metrics.columns)

    # Prepare resource-related dictionaries with lowercase keys
    resource_busy_periods = {resource: [] for resource in simulation_metrics['resource'].dropna().unique()}
    activity_processing_times = {}
    event_queue = []
    active_tokens = {}
    completed_tokens = []  # New list for completed tokens
    active_resources = {resource: 0 for resource in resource_busy_periods.keys()}
    available_resources = simulation_metrics.set_index('resource')['available resources'].to_dict()
    resource_wait_queue = {resource: [] for resource in resource_busy_periods.keys()}

    # Schedule tokens
    schedule_tokens(max_arrival_count, arrival_interval_minutes, start_time, start_tasks, paths, 
                    event_queue, active_tokens, transitions_df, simulation_end_date)

    # Wait until all tokens are scheduled before processing
    while len(event_queue) < max_arrival_count:
        pass

    # Process tokens
    process_tokens(event_queue, active_tokens, active_resources, available_resources, resource_busy_periods, 
                   activity_processing_times, transitions_df, simulation_metrics, resource_wait_queue, completed_tokens, paths)

    resource_utilization = print_processing_times_and_utilization(
        activity_processing_times,
        resource_busy_periods,
        simulation_end_date,
        start_time,
        available_resources,
        transitions_df
    )

    save_simulation_report(activity_processing_times, resource_utilization, len(active_tokens), xpdl_file_path, transitions_df, completed_tokens)


