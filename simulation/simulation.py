# Logic the runs the simulation
from datetime import timedelta
import heapq
from simulation.utils import is_work_time, advance_to_work_time, get_task_duration, get_condition_probability
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
def schedule_tokens(max_arrival_count, arrival_interval_minutes, start_time, start_tasks, paths, event_queue, active_tokens, transitions_df, simulation_end_date):
    token_count = 0
    for start_task in start_tasks:
        next_tasks = transitions_df[transitions_df['From'] == start_task]['To'].tolist()
        for next_task in next_tasks:
            while token_count < max_arrival_count:
                arrival_time = start_time + timedelta(minutes=arrival_interval_minutes * token_count)
                if arrival_time >= simulation_end_date:
                    break
                heapq.heappush(event_queue, Event(arrival_time, token_count, next_task, 'start'))
                active_tokens[token_count] = {
                    'current_task': None,
                    'completed_tasks': {start_task},  # Mark "Start" as completed
                    'start_time': arrival_time,
                    'paths': paths[:],
                    'wait_start_time': None
                }
                logging.info(f"Token {token_count} scheduled to start at {arrival_time} for task '{next_task}'")
                token_count += 1


# Function to start token processing
def start_token_processing(token_id, task_name, current_time, simulation_metrics, active_resources, available_resources, 
    resource_busy_periods, activity_processing_times, event_queue, active_tokens):
    # Determine the resource required for the task
    resource_name = simulation_metrics.loc[simulation_metrics['Name'] == task_name, 'Resource'].values
    if resource_name.size > 0 and pd.notna(resource_name[0]):
        resource_name = resource_name[0]
        # Check if resource is available
        if not assign_resource(resource_name, active_resources, available_resources):
            # Token needs to wait; mark waiting start time
            if active_tokens[token_id].get('wait_start_time') is None:
                active_tokens[token_id]['wait_start_time'] = current_time
            return False  # Resource unavailable

        # If waiting, calculate and log the wait time
        if active_tokens[token_id]['wait_start_time'] is not None:
            wait_time = (current_time - active_tokens[token_id]['wait_start_time']).total_seconds() / 60  # Convert to minutes
            if task_name not in activity_processing_times:
                activity_processing_times[task_name] = {"durations": [], "wait_times": [], "tokens_started": 0, "tokens_completed": 0}
            activity_processing_times[task_name]["wait_times"].append(wait_time)
            active_tokens[token_id]['wait_start_time'] = None  # Reset wait start time
            logging.info(f"Token {token_id} waited {wait_time:.2f} minutes for task '{task_name}'")

        resource_busy_periods[resource_name].append([current_time, None])

    # Log the start time of the task
    active_tokens[token_id]['task_start_time'] = current_time

    # Increment the number of tokens started for this activity
    if task_name not in activity_processing_times:
        activity_processing_times[task_name] = {"durations": [], "wait_times": [], "tokens_started": 0, "tokens_completed": 0}
    activity_processing_times[task_name]["tokens_started"] += 1

    # Schedule the end event for the task
    task_duration = timedelta(minutes=get_task_duration(task_name, simulation_metrics))
    heapq.heappush(event_queue, Event(current_time + task_duration, token_id, task_name, 'end'))
    logging.info(f"Token {token_id} started processing task '{task_name}' at {current_time} for {task_duration}")

    return True


# Function to complete a token's activity
def complete_activity(token_id, task_name, current_time, simulation_metrics, active_tokens, active_resources, resource_busy_periods, transitions_df, event_queue, activity_processing_times):
    logging.info(f"Token {token_id} completed task '{task_name}' at {current_time}")

    # Calculate and log processing time
    start_time = active_tokens[token_id].get('task_start_time', None)
    if start_time:
        process_time = (current_time - start_time).total_seconds() / 60  # Convert to minutes
        if task_name not in activity_processing_times:
            activity_processing_times[task_name] = {"durations": [], "wait_times": [], "tokens_started": 0, "tokens_completed": 0}
        activity_processing_times[task_name]["durations"].append(process_time)
        activity_processing_times[task_name]["tokens_completed"] += 1
        logging.info(f"Token {token_id}: Task '{task_name}' processing time = {process_time:.2f} minutes.")

    # Release resources
    resource_name = simulation_metrics.loc[simulation_metrics['Name'] == task_name, 'Resource'].values
    if resource_name.size > 0 and pd.notna(resource_name[0]):
        resource_name = resource_name[0]
        release_resource(resource_name, active_resources)
        for period in resource_busy_periods[resource_name]:
            if period[1] is None:
                period[1] = current_time
                break

    # Handle "Stop" activity
    activity_type = simulation_metrics.loc[simulation_metrics['Name'] == task_name, 'Type'].values
    activity_type = activity_type[0] if activity_type.size > 0 else "Unknown"
    if activity_type == "Stop":
        logging.info(f"Token {token_id} reached 'Stop' at {current_time}. Process completed.")
        return

    # Schedule next tasks
    next_tasks = transitions_df[transitions_df['From'] == task_name]['To'].tolist()
    for next_task in next_tasks:
        if next_task not in active_tokens[token_id]['completed_tasks']:
            heapq.heappush(event_queue, Event(current_time, token_id, next_task, 'start'))
            logging.info(f"Token {token_id} scheduled for next task '{next_task}' at {current_time}")

# Function to process tokens
def process_tokens(event_queue, active_tokens, active_resources, available_resources, resource_busy_periods, activity_processing_times, transitions_df, simulation_metrics):
    while event_queue:
        event = heapq.heappop(event_queue)
        current_time = event.time
        token_id = event.token_id
        task_name = event.task_name

        if event.event_type == 'start':
            if start_token_processing(token_id, task_name, current_time, simulation_metrics, active_resources, available_resources, resource_busy_periods, activity_processing_times, event_queue, active_tokens):
                active_tokens[token_id]['current_task'] = task_name
        elif event.event_type == 'end':
            complete_activity(token_id, task_name, current_time, simulation_metrics, active_tokens, active_resources, resource_busy_periods, transitions_df, event_queue, activity_processing_times)


# Main simulation function
def discrete_event_simulation(max_arrival_count, arrival_interval_minutes, simulation_days, paths, 
                              simulation_metrics, start_time, xpdl_file_path, transitions_df, start_tasks):

    simulation_end_date = start_time + timedelta(days=simulation_days)
    print('Simulation End Date:', simulation_end_date)

    resource_busy_periods = {resource: [] for resource in simulation_metrics['Resource'].dropna().unique()}
    activity_processing_times = {}
    event_queue = []
    active_tokens = {}
    active_resources = {resource: 0 for resource in resource_busy_periods.keys()}
    available_resources = simulation_metrics.set_index('Resource')['Available Resources'].to_dict()

    # Schedule tokens
    schedule_tokens(max_arrival_count, arrival_interval_minutes, start_time, start_tasks, paths, event_queue, active_tokens, transitions_df, simulation_end_date)

    # Process tokens
    process_tokens(event_queue, active_tokens, active_resources, available_resources, resource_busy_periods, activity_processing_times, transitions_df, simulation_metrics)

    resource_utilization = print_processing_times_and_utilization(
        activity_processing_times,
        resource_busy_periods,
        simulation_end_date,
        start_time,
        available_resources,
        transitions_df
    )

    save_simulation_report(activity_processing_times, resource_utilization, len(active_tokens), xpdl_file_path, transitions_df)
    print(f"Tokens started: {len(active_tokens)}")
