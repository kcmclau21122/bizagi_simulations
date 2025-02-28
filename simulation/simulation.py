import random
import heapq
import json
import logging
import time
from datetime import datetime, timedelta
from collections import defaultdict
from utils import is_work_time, advance_to_work_time, choose_node, format_duration
import copy

def run_resource_optimization(json_file_path, simulation_days, start_time, number_workdays, 
                              number_work_hours_per_day, target_avg_time, max_iterations=10, progress_callback=None):
    """
    Optimize resources to meet target average process time.
    
    Args:
        json_file_path: Path to the process model JSON file
        simulation_days: Number of days to simulate
        start_time: Start time for the simulation
        number_workdays: Number of workdays per week
        number_work_hours_per_day: Number of work hours per day
        target_avg_time: Target average process time in minutes
        max_iterations: Maximum number of optimization iterations
        progress_callback: Optional callback function to report progress
        
    Returns:
        Tuple of (optimized_model_path, final_avg_time, resource_counts)
    """
    logging.info(f"Starting resource optimization to meet target average time of {target_avg_time} minutes")
    
    # Load original model
    with open(json_file_path, "r") as file:
        original_model = json.load(file)
    
    # Identify all resources used in the model
    resources = set()
    for node in original_model.get("nodes", []):
        resource = node.get("resource")
        if resource:
            resources.add(resource)
    
    # Initialize resource multiplication factors
    resource_multipliers = {resource: 1 for resource in resources}
    best_model = None
    best_avg_time = float('inf')
    best_resource_config = None
    
    for iteration in range(max_iterations):
        # Update progress if callback provided
        if progress_callback:
            progress_callback(f"Resource optimization - iteration {iteration+1}/{max_iterations}")
        
        # Create a copy of the original model to modify
        current_model = copy.deepcopy(original_model)
        
        # Update resource counts in the model
        for node in current_model.get("nodes", []):
            resource = node.get("resource")
            if resource and "available resources" in node:
                node["available resources"] = max(1, int(node["available resources"]) * resource_multipliers[resource])
        
        # Save the modified model temporarily
        temp_model_path = f"temp_model_iteration_{iteration}.json"
        with open(temp_model_path, "w") as file:
            json.dump(current_model, file, indent=4)
        
        # Run simulation with current resource configuration
        avg_time, completed_tokens = run_simulation_for_optimization(
            temp_model_path, simulation_days, start_time, 
            number_workdays, number_work_hours_per_day
        )
        
        logging.info(f"Iteration {iteration}: Average processing time = {avg_time:.2f} min with resource config: {resource_multipliers}")
        
        # Track best configuration
        if avg_time <= target_avg_time and (best_avg_time > target_avg_time or sum(resource_multipliers.values()) < sum(best_resource_config.values() if best_resource_config else [float('inf')])):
            best_model = current_model
            best_avg_time = avg_time
            best_resource_config = resource_multipliers.copy()
            logging.info(f"Found better configuration: {best_resource_config}")
        
        # Stop if we've met or exceeded target
        if best_avg_time <= target_avg_time:
            break
        
        # Adjust resource multipliers based on results
        if avg_time > target_avg_time:
            # Identify bottleneck resources (highest utilization)
            bottlenecks = identify_bottleneck_resources(temp_model_path)
            
            # Increase resources for bottlenecks
            for resource in bottlenecks:
                if resource in resource_multipliers:
                    resource_multipliers[resource] += 1
        else:
            # We're under target time, try to reduce resources
            for resource in sorted(resource_multipliers.keys()):
                if resource_multipliers[resource] > 1:
                    resource_multipliers[resource] -= 1
                    break
    
    # If we found a valid configuration, save it as the final model
    if best_model:
        optimized_model_path = "optimized_process_model.json"
        with open(optimized_model_path, "w") as file:
            json.dump(best_model, file, indent=4)
        
        logging.info(f"Resource optimization complete. Final average time: {best_avg_time:.2f} min")
        logging.info(f"Optimized resource configuration: {best_resource_config}")
        
        return optimized_model_path, best_avg_time, best_resource_config
    else:
        logging.warning("Could not find a resource configuration that meets the target time")
        return json_file_path, avg_time, resource_multipliers
    
def identify_bottleneck_resources(model_path):
    """
    Identify the resources with highest utilization (bottlenecks)
    
    Args:
        model_path: Path to the process model with simulation results
        
    Returns:
        List of resource names sorted by utilization (highest first)
    """
    # This is a simplified version - in production you'd want to
    # read the actual simulation results to determine bottlenecks
    with open(model_path, "r") as file:
        model = json.load(file)
    
    resources = {}
    for node in model.get("nodes", []):
        resource = node.get("resource")
        if resource:
            if resource not in resources:
                resources[resource] = {
                    "load": node.get("tokens_started", 0) * node.get("avg time", 0),
                    "capacity": node.get("available resources", 1)
                }
            else:
                resources[resource]["load"] += node.get("tokens_started", 0) * node.get("avg time", 0)
    
    # Calculate utilization (load/capacity)
    resource_utilization = {
        res: data["load"] / data["capacity"] for res, data in resources.items() if data["capacity"] > 0
    }
    
    # Return resources sorted by utilization (highest first)
    return [res for res, _ in sorted(resource_utilization.items(), key=lambda x: x[1], reverse=True)]

def run_simulation_for_optimization(json_file_path, simulation_days, start_time, number_workdays, number_work_hours_per_day):
    """
    Run a simulation for optimization purposes, with a focus on collecting token processing times.
    
    Args:
        json_file_path: Path to the process model JSON file
        simulation_days: Number of days to simulate
        start_time: Start time for the simulation
        number_workdays: Number of workdays per week
        number_work_hours_per_day: Number of work hours per day
        
    Returns:
        Tuple of (average_processing_time, completed_tokens)
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
    resource_busy_periods = defaultdict(list)
    event_queue = []
    activity_processing_times = {}
    active_tokens = {}
    completed_tokens = []
    total_tokens_started = 0

    # Schedule tokens
    scheduled_tokens = schedule_tokens(json_file_path, max_arrival_count, arrival_interval_minutes, start_time, simulation_end_date)
    total_tokens_started = len(scheduled_tokens)

    for token in scheduled_tokens:
        token_id = token["token_id"]
        token_start_time = datetime.strptime(token["start_time"], "%Y-%m-%d %H:%M:%S") if isinstance(token["start_time"], str) else token["start_time"]
        heapq.heappush(event_queue, (token_start_time, token_id, token["start_node"], "start"))
        active_tokens[token_id] = {
            "current_task": None, 
            "start_time": token_start_time,  # Process start time
            "wait_start_time": None, 
            "total_wait_time": 0, 
            "completed_tasks": [],
            "path": []  # Track the path taken
        }

    # Process events
    process_events_for_optimization(
        event_queue, active_tokens, active_resources, resource_wait_queue, 
        resource_busy_periods, activity_processing_times, start_time, 
        number_workdays, number_work_hours_per_day, completed_tokens
    )

    # Calculate average processing time from completed tokens
    if completed_tokens:
        process_durations = [
            (token["end_time"] - token["start_time"]).total_seconds() / 60 
            for token in completed_tokens
        ]
        avg_time = sum(process_durations) / len(process_durations)
    else:
        avg_time = float('inf')  # No tokens completed
    
    return avg_time, completed_tokens

def process_events_for_optimization(event_queue, active_tokens, active_resources, resource_wait_queue, 
                                  resource_busy_periods, activity_processing_times, start_time,
                                  number_workdays, number_work_hours_per_day, completed_tokens):
    """
    Process all events in the simulation with a focus on optimization.
    """
    with open('process_model.json', 'r') as f:
        process_model = json.load(f)

    nodes = {node["id"]: node for node in process_model["nodes"]}
    links = process_model["links"]
    
    # Track tokens that have already completed the process to prevent double counting
    completed_token_ids = set()

    while event_queue:
        event_time, token_id, task_name, event_type = heapq.heappop(event_queue)
        
        # Skip processing for tokens that have already completed
        if token_id in completed_token_ids:
            continue

        if event_type == 'start':
            resource = nodes.get(task_name, {}).get("resource")
            
            # Track resource busy periods
            if resource:
                resource_busy_periods[resource].append([event_time, None])  # Start time, end time will be filled later
            
            # Update token path
            if token_id in active_tokens:
                active_tokens[token_id]["path"].append(task_name)
            
            start_token_processing(
                token_id=token_id,
                task_name=task_name,
                start_time=start_time,
                token_start_time=event_time,
                active_resources=active_resources,
                resource_wait_queue=resource_wait_queue,
                activity_processing_times=activity_processing_times,
                event_queue=event_queue,
                active_tokens=active_tokens,
                number_workdays=number_workdays,
                number_work_hours_per_day=number_work_hours_per_day
            )

        elif event_type == 'end':
            logging.info(f"Token {token_id} completed task '{task_name}' at {event_time}.")
            
            # Update resource busy period end time
            resource = nodes.get(task_name, {}).get("resource")
            if resource:
                # Find the most recent busy period with no end time for this resource
                for period in reversed(resource_busy_periods[resource]):
                    if period[1] is None:
                        period[1] = event_time
                        break
            
            # Track activity completion
            if task_name in activity_processing_times:
                activity_processing_times[task_name]["tokens_completed"] = activity_processing_times[task_name].get("tokens_completed", 0) + 1
            
            # Release resources
            release_resources(
                task_name=task_name,
                active_resources=active_resources,
                resource_wait_queue=resource_wait_queue,
                event_queue=event_queue,
                activity_processing_times=activity_processing_times,
                current_time=event_time  # Pass the current event time
            )

            source_node = nodes.get(task_name)
            if not source_node:
                raise ValueError(f"Node '{task_name}' not found in the process model.")

            # Check if this is an end node (Stop type) or no next nodes available
            # or if the node name is "Unknown" which seems to be used as an end node
            if not source_node or task_name == "Unknown" or source_node.get("type") == "Stop":
                # Token has completed the process
                logging.info(f"Token {token_id} has completed the entire process at {event_time}.")
                if token_id in active_tokens and token_id not in completed_token_ids:
                    token_data = active_tokens[token_id]
                    token_data["end_time"] = event_time
                    process_duration = (event_time - token_data["start_time"]).total_seconds() / 60
                    formatted_duration = format_duration(process_duration)
                    logging.info(f"Token {token_id} completed in {process_duration:.2f} minutes ({formatted_duration}) with path: {' -> '.join(token_data['path'])}")
                    
                    # Add to completed tokens list for reporting (use a copy to prevent further changes)
                    completed_tokens.append(token_data.copy())
                    
                    # Mark this token as completed to prevent double counting
                    completed_token_ids.add(token_id)
                    
                    # Remove token from active tokens to prevent further processing
                    if token_id in active_tokens:
                        del active_tokens[token_id]
                continue

            # Determine next node(s) using choose_node with triangular distribution
            next_nodes = choose_node(source_node, links)
            if not next_nodes:
                # Token has completed the process
                logging.info(f"Token {token_id} has completed the entire process at {event_time}.")
                if token_id in active_tokens and token_id not in completed_token_ids:
                    token_data = active_tokens[token_id]
                    token_data["end_time"] = event_time
                    process_duration = (event_time - token_data["start_time"]).total_seconds() / 60
                    formatted_duration = format_duration(process_duration)
                    logging.info(f"Token {token_id} completed in {process_duration:.2f} minutes ({formatted_duration}) with path: {' -> '.join(token_data['path'])}")
                    
                    # Add to completed tokens list for reporting (use a copy to prevent further changes)
                    completed_tokens.append(token_data.copy())
                    
                    # Mark this token as completed to prevent double counting
                    completed_token_ids.add(token_id)
                    
                    # Remove token from active tokens to prevent further processing
                    if token_id in active_tokens:
                        del active_tokens[token_id]
                continue

            for next_task_name in next_nodes:
                task_node = nodes.get(next_task_name, {})

                # Determine processing time using triangular distribution
                min_time = task_node.get("min time", 0)
                avg_time = task_node.get("avg time", 0)  # Mode of the triangular distribution
                max_time = task_node.get("max time", 0)

                if avg_time > 0 and min_time > 0 and max_time > 0:
                    task_duration = random.triangular(min_time, avg_time, max_time)
                    next_start_time = event_time + timedelta(minutes=task_duration)
                else:
                    # No processing time, move immediately
                    task_duration = 0
                    next_start_time = event_time

                # Add next task to the event queue
                heapq.heappush(event_queue, (next_start_time, token_id, next_task_name, "start"))

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
    available_resources = int(task_node.get("available resources", 0))  # Ensure this is an integer

    # Ensure the start time is within work hours
    if not is_work_time(token_start_time, start_time, number_workdays, number_work_hours_per_day):
        logging.info(f"Token {token_id} cannot start task '{task_name}' at {token_start_time} (outside work hours).")
        next_work_time = advance_to_work_time(token_start_time, start_time, number_workdays, number_work_hours_per_day)
        heapq.heappush(event_queue, (next_work_time, token_id, task_name, "start"))
        return False

    # Check resource availability
    if resource and active_resources[resource] >= available_resources:
        # Start tracking wait time if not already waiting
        if token_id in active_tokens and active_tokens[token_id]["wait_start_time"] is None:
            active_tokens[token_id]["wait_start_time"] = token_start_time
        
        resource_wait_queue[resource].append((token_id, task_name, token_start_time))
        logging.info(f"Token {token_id} added to wait queue for '{resource}' resource at {token_start_time}.")
        return False

    # Calculate wait time if the token was waiting
    if token_id in active_tokens and active_tokens[token_id]["wait_start_time"]:
        wait_duration = (token_start_time - active_tokens[token_id]["wait_start_time"]).total_seconds() / 60
        active_tokens[token_id]["total_wait_time"] += wait_duration
        active_tokens[token_id]["wait_start_time"] = None
        
        if task_name not in activity_processing_times:
            activity_processing_times[task_name] = {"wait_times": [], "durations": [], "tokens_started": 0, "tokens_completed": 0}
        
        activity_processing_times[task_name]["wait_times"].append(wait_duration)

    if resource:
        active_resources[resource] += 1

    active_tokens[token_id]["current_task"] = task_name
    logging.info(f"Token {token_id} started task '{task_name}' at {token_start_time}.")

    if task_name not in activity_processing_times:
        activity_processing_times[task_name] = {"wait_times": [], "durations": [], "tokens_started": 0, "tokens_completed": 0}

    activity_processing_times[task_name]["tokens_started"] += 1

    # Determine task duration using triangular distribution
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
    activity_processing_times[task_name]["durations"].append(task_duration)

    heapq.heappush(event_queue, (end_time, token_id, task_name, "end"))
    logging.info(f"Token {token_id} scheduled to end task '{task_name}' at {end_time}.")

    return True

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
        # Store datetime object directly to avoid conversion issues
        schedule.append({
            "token_id": token_id,
            "start_time": current_time,
            "start_node": start_node["id"]
        })
        logging.info(f"Scheduled {token_id} to start at {current_time}.")
        token_count += 1
        current_time += timedelta(minutes=arrival_interval_minutes)

    return schedule

def release_resources(task_name, active_resources, resource_wait_queue, event_queue, activity_processing_times, current_time):
    """
    Release resources when a token completes an activity and handle the next token in queue.
    Now with proper simulation time tracking.
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
            # Use simulation time for wait duration calculation
            wait_duration = (current_time - queued_time).total_seconds() / 60

            if next_task_name not in activity_processing_times:
                activity_processing_times[next_task_name] = {"wait_times": [], "durations": [], "tokens_started": 0}

            activity_processing_times[next_task_name]["wait_times"].append(wait_duration)

            # Use simulation time for the next event
            heapq.heappush(event_queue, (current_time, next_token_id, next_task_name, "start"))
            logging.info(f"Token {next_token_id} started from wait queue for '{next_task_name}' after waiting {wait_duration:.2f} minutes.")

def run_simulation(json_file_path, simulation_days, start_time, number_workdays, number_work_hours_per_day, 
                  target_avg_time=None, progress_callback=None):
    """
    Run the simulation process with optional optimization for target average time.
    
    Args:
        json_file_path: Path to the process model JSON file
        simulation_days: Number of days to simulate
        start_time: Start time for the simulation
        number_workdays: Number of workdays per week
        number_work_hours_per_day: Number of work hours per day
        target_avg_time: Optional target average process time in minutes
        progress_callback: Optional callback function to report progress
        
    Returns:
        Dictionary containing simulation results and completed tokens
    """
    # If target_avg_time is specified, run optimization
    if target_avg_time is not None:
        if progress_callback:
            progress_callback("Starting resource optimization...")
        
        logging.info(f"Target average time specified: {target_avg_time} minutes. Running resource optimization.")
        optimized_model_path, achieved_avg_time, resource_config = run_resource_optimization(
            json_file_path, simulation_days, start_time, number_workdays, 
            number_work_hours_per_day, target_avg_time, progress_callback=progress_callback
        )
        json_file_path = optimized_model_path
        logging.info(f"Using optimized model with achieved average time of {achieved_avg_time:.2f} minutes")
    
    if progress_callback:
        progress_callback("Loading process model...")
        
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
    resource_busy_periods = defaultdict(list)
    event_queue = []
    activity_processing_times = {}
    active_tokens = {}
    completed_tokens = []

    if progress_callback:
        progress_callback("Scheduling tokens...")
        
    # Schedule tokens
    scheduled_tokens = schedule_tokens(json_file_path, max_arrival_count, arrival_interval_minutes, start_time, simulation_end_date)
    total_tokens_started = len(scheduled_tokens)

    for token in scheduled_tokens:
        token_id = token["token_id"]
        token_start_time = token["start_time"]
        heapq.heappush(event_queue, (token_start_time, token_id, token["start_node"], "start"))
        active_tokens[token_id] = {
            "current_task": None, 
            "start_time": token_start_time,  # Process start time
            "wait_start_time": None, 
            "total_wait_time": 0, 
            "completed_tasks": [],
            "path": []  # Track the path taken through the process
        }

    if progress_callback:
        progress_callback("Processing events...")
        
    # Process events with enhanced tracking for completed tokens
    process_events_for_optimization(
        event_queue, active_tokens, active_resources, resource_wait_queue, 
        resource_busy_periods, activity_processing_times, start_time, 
        number_workdays, number_work_hours_per_day, completed_tokens
    )
    
    if progress_callback:
        progress_callback("Calculating statistics...")
    
    # Calculate and log average processing time
    if completed_tokens:
        process_durations = [
            (token["end_time"] - token["start_time"]).total_seconds() / 60 
            for token in completed_tokens
        ]
        avg_time = sum(process_durations) / len(process_durations)
        min_time = min(process_durations)
        max_time = max(process_durations)
        
        logging.info(f"Process Statistics:")
        logging.info(f"  Total tokens started: {total_tokens_started}")
        logging.info(f"  Total tokens completed: {len(completed_tokens)}")
        logging.info(f"  Average processing time: {avg_time:.2f} minutes ({format_duration(avg_time)})")
        logging.info(f"  Minimum processing time: {min_time:.2f} minutes ({format_duration(min_time)})")
        logging.info(f"  Maximum processing time: {max_time:.2f} minutes ({format_duration(max_time)})")
        
        # Log individual token completion times
        logging.info("Individual Token Completion Times:")
        for token in completed_tokens:
            token_duration = (token["end_time"] - token["start_time"]).total_seconds() / 60
            logging.info(f"  Token {token['current_task']}: {token_duration:.2f} minutes ({format_duration(token_duration)})")
    
    # Calculate resource utilization
    available_resources = {node["resource"]: node.get("available resources", 1) 
                          for node in process_model_data.get("nodes", []) 
                          if "resource" in node}
    
    resource_utilization = {}
    for resource, periods in resource_busy_periods.items():
        total_busy_time = sum(
            (end - start).total_seconds() for start, end in periods if start and end
        )
        total_simulation_time = (simulation_end_date - start_time).total_seconds()
        num_resources = available_resources.get(resource, 1)
        
        utilization = (total_busy_time / (total_simulation_time * num_resources)) * 100 if total_simulation_time > 0 else 0
        resource_utilization[resource] = min(utilization, 100)
    
    if progress_callback:
        progress_callback("Simulation completed")
        
    return {
        "activity_processing_times": activity_processing_times,
        "resource_utilization": resource_utilization,
        "total_tokens_started": total_tokens_started,
        "completed_tokens": completed_tokens
    }
