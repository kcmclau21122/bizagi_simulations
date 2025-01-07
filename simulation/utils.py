# Utilities used by the simulation.py file
from datetime import datetime, timedelta
import logging
import random
import pandas as pd
import heapq
from simulation.event import Event

# Read simulation parameters
def get_simulation_parameters(simulation_metrics):
    start_event = simulation_metrics[simulation_metrics['Type'].str.lower() == 'start event']
    if not start_event.empty:
        max_arrival_count = (
            int(start_event['Max arrival count'].iloc[0]) 
            if 'max arrival count' in map(str.lower, start_event.columns) 
            else 10
        )
        arrival_interval_minutes = (
            int(start_event['Arrival Interval'].iloc[0]) 
            if 'arrival interval' in map(str.lower, start_event.columns) 
            else 10
        )
        return max_arrival_count, arrival_interval_minutes
    logging.warning("Start event parameters not found. Using default values.")
    return 10, 10  # Default values if not found

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
def get_condition_probability(simulation_metrics, from_activity, condition_type):
    condition_key = condition_type.split("-")[1].strip()
    row = simulation_metrics.loc[simulation_metrics['Name'] == from_activity]
    if not row.empty and condition_key in row.columns:
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

def get_task_duration(task_name, simulation_metrics):
    """
    Calculate task duration using Bizagi's triangular distribution. 
    Return None if Min Time, Avg Time, and Max Time are all None.
    """
    row = simulation_metrics.loc[simulation_metrics['Name'] == task_name]
    if not row.empty:
        min_time = row['Min Time'].iloc[0]
        avg_time = row['Avg Time'].iloc[0]
        max_time = row['Max Time'].iloc[0]

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

