# Logic the runs the simulation
from datetime import datetime, timedelta
import json
import heapq
from simulation.utils import is_work_time, advance_to_work_time, get_condition_probability, advance_time_in_seconds, choose_node
from simulation.reporting import print_processing_times_and_utilization, save_simulation_report
from simulation.data_handler import extract_start_tasks_from_json
import logging
import random
from collections import defaultdict

# Function to schedule all tokens
def schedule_tokens(json_file_path, max_arrival_count, arrival_interval_minutes, start_time, simulation_end_date):
    """
    Schedules tokens to start the process based on the "Start" node and provided parameters.
    """
    with open(json_file_path, "r") as file:
        process_model_data = json.load(file)

    # Extract the "Start" node attributes
    nodes = process_model_data.get("nodes", [])
    start_node = next((node for node in nodes if node.get("type") == "Start"), None)

    if not start_node:
        raise ValueError("No 'Start' node found in the JSON file.")

    schedule = []
    current_time = start_time
    token_count = 0

    while token_count < max_arrival_count and current_time <= simulation_end_date:
        token_id = f"Token-{token_count + 1}"
        token_start_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        schedule.append({
            "token_id": token_id,
            "start_time": token_start_time,
            "start_node": start_node["id"]
        })
        logging.info(f"Scheduled {token_id} to start at {token_start_time}.")
        token_count += 1
        current_time += timedelta(minutes=arrival_interval_minutes)

    return schedule

def release_resources(task_name, active_resources, resource_wait_queue, event_queue, activity_processing_times):
    """
    Release resources when a token completes an activity and handle the next token in queue.
    """
    with open('process_model.json', 'r') as f:
        process_model = json.load(f)

    nodes = {node["id"]: node for node in process_model["nodes"]}
    task_node = nodes.get(task_name)

    if not task_node:
        raise ValueError(f"Task '{task_name}' not found in the process model.")

    resource = task_node.get("resource")

    if resource:
        active_resources[resource] -= 1

        if resource_wait_queue[resource]:
            next_token_id, next_task_name, queued_time = resource_wait_queue[resource].pop(0)
            wait_duration = (datetime.now() - queued_time).total_seconds() / 60

            if next_task_name not in activity_processing_times:
                activity_processing_times[next_task_name] = {"wait_times": [], "process_times": [], "tokens_started": 0}

            activity_processing_times[next_task_name]["wait_times"].append(wait_duration)

            heapq.heappush(event_queue, (datetime.now(), next_token_id, next_task_name, "start"))
            logging.info(f"Token {next_token_id} started from wait queue for '{next_task_name}' after waiting {wait_duration:.2f} minutes.")

# Helper function to process events
def process_events(event_queue, active_tokens, active_resources, resource_wait_queue, activity_processing_times, start_time,
                   number_workdays, number_work_hours_per_day):
    """
    Process all events in the simulation.
    """
    with open('process_model.json', 'r') as f:
        process_model = json.load(f)

    nodes = {node["id"]: node for node in process_model["nodes"]}
    links = process_model["links"]

    tokens_completed = 0  # Track tokens completing the entire process

    while event_queue:
        event_time, token_id, task_name, event_type = heapq.heappop(event_queue)

        if event_type == 'start':
            start_token_processing(
                token_id=token_id,
                task_name=task_name,
                start_time = start_time,
                token_start_time=event_time,  # Fixed: pass `start_time`
                active_resources=active_resources,
                resource_wait_queue=resource_wait_queue,
                activity_processing_times=activity_processing_times,
                event_queue=event_queue,
                active_tokens=active_tokens,
                number_workdays=number_workdays,
                number_work_hours_per_day=number_work_hours_per_day  # Fixed: pass missing argument
            )

        elif event_type == 'end':
            logging.info(f"Token {token_id} completed task '{task_name}' at {event_time}.")
            release_resources(
                task_name=task_name,
                active_resources=active_resources,
                resource_wait_queue=resource_wait_queue,
                event_queue=event_queue,
                activity_processing_times=activity_processing_times
            )

            source_node = nodes.get(task_name)
            if not source_node:
                raise ValueError(f"Node '{task_name}' not found in the process model.")

            # Determine next node(s) using choose_node
            next_nodes = choose_node(source_node, links)
            if not next_nodes:
                # Token has completed the process
                tokens_completed += 1
                logging.info(f"Token {token_id} has completed the entire process.")
                continue

            for next_task_name in next_nodes:
                task_node = nodes.get(next_task_name, {})

                # Determine processing time for the next node
                avg_time = task_node.get("avg time", 0)
                min_time = task_node.get("min time", 0)
                max_time = task_node.get("max time", 0)

                if avg_time > 0 and min_time > 0 and max_time > 0:
                    task_duration = random.triangular(min_time, avg_time, max_time)
                    next_start_time = event_time + timedelta(minutes=task_duration)
                else:
                    # No processing time, move immediately
                    task_duration = 0
                    next_start_time = event_time

                # Log token movement
                logging.info(
                    f"Token {token_id} scheduled to start task '{next_task_name}' "
                    f"after {task_duration:.2f} minutes at {next_start_time}."
                )

                # Add next task to the event queue
                heapq.heappush(event_queue, (next_start_time, token_id, next_task_name, "start"))

# Function to start token processing
def start_token_processing(token_id, task_name, start_time, token_start_time, active_resources, resource_wait_queue,
                           activity_processing_times, event_queue, active_tokens, number_workdays, number_work_hours_per_day):
    """
    Start processing a token for a given task, considering resource availability, wait times, and work hours.
    """
    with open('process_model.json', 'r') as f:
        process_model = json.load(f)

    nodes = {node["id"]: node for node in process_model["nodes"]}
    task_node = nodes.get(task_name)

    if not task_node:
        raise ValueError(f"Task '{task_name}' not found in the process model.")

    resource = task_node.get("resource")
    available_resources = task_node.get("available resources", 0)

    # Ensure the start time is within work hours
    if not is_work_time(token_start_time, start_time, number_workdays, number_work_hours_per_day):
        logging.info(f"Token {token_id} cannot start task '{task_name}' at {token_start_time} (outside work hours).")
        next_work_time = advance_to_work_time(token_start_time, start_time, number_workdays, number_work_hours_per_day)
        heapq.heappush(event_queue, (next_work_time, token_id, task_name, "start"))
        return False

    # Check resource availability
    if resource and active_resources[resource] >= available_resources:
        resource_wait_queue[resource].append((token_id, task_name, token_start_time))
        logging.info(f"Token {token_id} added to wait queue for '{task_name}' at {token_start_time}.")
        return False

    if resource:
        active_resources[resource] += 1

    active_tokens[token_id]["current_task"] = task_name
    active_tokens[token_id]["start_time"] = token_start_time
    logging.info(f"Token {token_id} started task '{task_name}' at {token_start_time}.")

    if task_name not in activity_processing_times:
        activity_processing_times[task_name] = {"wait_times": [], "process_times": [], "tokens_started": 0}

    activity_processing_times[task_name]["tokens_started"] += 1

    # Determine task duration
    task_duration = random.triangular(
        task_node.get("min time", 0),
        task_node.get("avg time", 0),
        task_node.get("max time", 0)
    )
    end_time = token_start_time + timedelta(minutes=task_duration)

    # Ensure the end time is within work hours
    if not is_work_time(end_time, start_time, number_workdays, number_work_hours_per_day):
        logging.info(f"Token {token_id} cannot end task '{task_name}' at {end_time} (outside work hours).")
        end_time = advance_to_work_time(end_time, start_time, number_workdays, number_work_hours_per_day)

    # Track processing time
    activity_processing_times[task_name]["process_times"].append(task_duration)

    heapq.heappush(event_queue, (end_time, token_id, task_name, "end"))
    logging.info(f"Token {token_id} scheduled to end task '{task_name}' at {end_time}.")

    return True


def run_simulation(json_file_path, simulation_days, start_time, number_workdays, number_work_hours_per_day):
    """
    Run the simulation process using parameters from the JSON file.
    """
    with open(json_file_path, "r") as file:
        process_model_data = json.load(file)

    nodes = {node["id"]: node for node in process_model_data.get("nodes", [])}
    start_event = next((node for node in nodes.values() if node.get("type") == "Start"), None)

    if not start_event:
        raise ValueError("No 'Start' event found in the JSON file.")

    max_arrival_count = int(start_event.get("max arrival count", 10))
    arrival_interval_minutes = int(start_event.get("arrival interval", 2))
    simulation_end_date = start_time + timedelta(days=simulation_days)

    active_resources = defaultdict(int)
    resource_wait_queue = defaultdict(list)
    event_queue = []
    activity_processing_times = {}
    active_tokens = {}

    scheduled_tokens = schedule_tokens(json_file_path, max_arrival_count, arrival_interval_minutes, start_time, simulation_end_date)

    for token in scheduled_tokens:
        token_id = token["token_id"]
        token_start_time = datetime.strptime(token["start_time"], "%Y-%m-%d %H:%M:%S")
        heapq.heappush(event_queue, (token_start_time, token_id, token["start_node"], "start"))
        active_tokens[token_id] = {"current_task": None, "start_time": None, "wait_start_time": None, "total_wait_time": 0, "completed_tasks": []}

    # Corrected process_events call
    process_events(event_queue, active_tokens, active_resources, resource_wait_queue, activity_processing_times, start_time, number_workdays, number_work_hours_per_day)

    logging.info(f"Simulation complete. Processing times: {activity_processing_times}")
