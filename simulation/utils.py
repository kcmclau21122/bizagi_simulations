# Utilities used by the simulation.py file
from datetime import datetime, timedelta
import logging
import random
import pandas as pd
import heapq
from simulation.event import Event

# Read simulation parameters
def get_simulation_parameters(simulation_metrics):
    # Convert column names to lowercase for consistent access
    simulation_metrics.columns = map(str.lower, simulation_metrics.columns)

    # Log the entire simulation_metrics DataFrame to the log file
    logging.info("Logging simulation metrics:")
    logging.info(simulation_metrics.to_string(index=False))

    # Filter the DataFrame for rows where 'type' equals 'start' (case-insensitively)
    start_event = simulation_metrics[simulation_metrics['type'].str.lower() == 'start']
    
    if not start_event.empty:
        # Check for the 'max arrival count' and 'arrival interval' columns and fetch their values
        max_arrival_count = (
            int(start_event['max arrival count'].iloc[0]) 
            if 'max arrival count' in simulation_metrics.columns 
            else 0
        )
        arrival_interval_minutes = (
            int(start_event['arrival interval'].iloc[0]) 
            if 'arrival interval' in simulation_metrics.columns 
            else 0
        )
        return max_arrival_count, arrival_interval_minutes

    logging.warning("Start event parameters not found. Using default values.")
    return 0, 0  # Default values if not found

# Helper function to advance simulation time in 1-second intervals
def advance_time_in_seconds(current_time, event_queue, active_resources, resource_wait_queue, active_tokens, resource_busy_periods):
    while current_time.second != 0:
        current_time += timedelta(seconds=1)
        # Check and assign resources based on FIFO
        for resource_name, queue in resource_wait_queue.items():
            if queue and active_resources[resource_name] < len(resource_busy_periods[resource_name]):
                token_id, task_name = queue.pop(0)  # FIFO queue
                
                # Inline logic for assigning resource
                if active_resources[resource_name] < len(resource_busy_periods[resource_name]):
                    active_resources[resource_name] += 1
                
                start_event_time = current_time
                heapq.heappush(event_queue, Event(start_event_time, token_id, task_name, 'start'))
                active_tokens[token_id]['wait_start_time'] = None

# Fetch conditional probabilities
# Fetch conditional probabilities
def get_condition_probability(simulation_metrics, from_activity, condition_type):
    # Normalize column names to lowercase for consistent access
    simulation_metrics.columns = map(str.lower, simulation_metrics.columns)
    
    # Convert the input strings to lowercase for comparison
    condition_key = condition_type.split("-")[1].strip().lower()
    row = simulation_metrics.loc[simulation_metrics['name'].str.lower() == from_activity.lower()]
    
    # Check if the row is not empty and the condition key exists in the columns
    if not row.empty and condition_key in simulation_metrics.columns:
        return row.iloc[0][condition_key]
    
    logging.warning("Probability for condition '%s' not found. Defaulting to 0.5.", condition_type)
    return 0.5  # Default probability

# Check if the current time is within work hours and days
def is_work_time(current_time):
    work_start = datetime.combine(current_time.date(), datetime.min.time()) + timedelta(hours=6)
    work_end = datetime.combine(current_time.date(), datetime.min.time()) + timedelta(hours=12)
    is_work_day = current_time.weekday() < 5  # Monday to Friday
    return is_work_day and work_start <= current_time < work_end


# Advance to the next work time if outside of work hours
def advance_to_work_time(current_time):
    if current_time.weekday() >= 5 or current_time.hour >= 12:  # Weekend or after work hours
        days_to_advance = (7 - current_time.weekday()) % 7 if current_time.weekday() >= 5 else 1
        current_time = datetime.combine(current_time.date(), datetime.min.time()) + timedelta(days=days_to_advance, hours=6)
    elif current_time.hour < 6:  # Before work hours
        current_time = datetime.combine(current_time.date(), datetime.min.time()) + timedelta(hours=6)
    return current_time

# Get task duration
def get_task_duration(task_name, simulation_metrics):
    """
    Calculate task duration using Bizagi's triangular distribution. 
    Return None if Min Time, Avg Time, and Max Time are all None.
    """
    # Normalize column names to lowercase for consistent access
    simulation_metrics.columns = map(str.lower, simulation_metrics.columns)

    row = simulation_metrics.loc[simulation_metrics['name'].str.lower() == task_name.lower()]
    if not row.empty:
        min_time = row['min time'].iloc[0]
        avg_time = row['avg time'].iloc[0]
        max_time = row['max time'].iloc[0]

        if pd.isna(min_time) and pd.isna(avg_time) and pd.isna(max_time):
            logging.info(f"No processing time specified for task '{task_name}'. Skipping processing time.")
            return None

        # Ensure values for calculation
        min_time = int(min_time) if pd.notna(min_time) else 1
        avg_time = int(avg_time) if pd.notna(avg_time) else min_time
        max_time = int(max_time) if pd.notna(max_time) else avg_time

        duration = int(random.triangular(min_time, avg_time, max_time))
        logging.info(f"Calculated duration for '{task_name}': Min = {min_time}, Avg = {avg_time}, Max = {max_time}, Chosen = {duration}")
        return duration

    logging.warning(f"Task duration not found for '{task_name}'. Defaulting to None.")
    return None

def choose_node(source_node, links):
    """
    Determine the next node(s) based on the gateway type of the source node.
    If the gateway is empty or does not exist, find the next target node(s) using the links.
    """
    gateway = source_node.get("gateway", None)
    nodetype = source_node.get("type", None)
    source_id = source_node["id"]

    if gateway == "[Parallel Gateway]":
        # Return all target nodes linked to the source
        return [link["target"] for link in links if link["source"] == source_id]

    elif gateway == "[Exclusive Gateway]" and "CONDITION-" not in nodetype:
        # Collect probabilities from source node attributes
        probabilities = {k.lower(): v for k, v in source_node.items() if isinstance(v, (int, float)) and v < 1}
        if not probabilities or sum(probabilities.values()) != 1:
            raise ValueError(f"Invalid probabilities for Exclusive Gateway at node '{source_id}'.")

        # Randomly select based on probabilities
        choice = random.choices(
            population=list(probabilities.keys()),
            weights=list(probabilities.values()),
            k=1
        )[0]

        # Find the matching target node
        condition_type = f"CONDITION-{choice.capitalize()}"
        return [
            link["target"] for link in links
            if link["source"] == source_id and link["type"].lower() == condition_type.lower()
        ]

    elif gateway == "[Inclusive Gateway]":
        # Select one target node at random for now
        candidates = [link["target"] for link in links if link["source"] == source_id]
        if candidates:
            return [random.choice(candidates)]

    # Default: no gateway logic, find direct target(s) from links
    return [link["target"] for link in links if link["source"] == source_id]
