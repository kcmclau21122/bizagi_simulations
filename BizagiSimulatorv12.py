# Bug: Processing times and utilizations not being calculated correctly.  Appears a token is starting and/or finishing an activity more than once
import pandas as pd
import heapq
import random
import re
import logging
import os
from datetime import datetime, timedelta

# Importing the parse_xpdl_to_sequences function
from xpdl_parser import parse_xpdl_to_sequences

# Configure logging
logging.basicConfig(filename='simulation_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class Event:
    def __init__(self, time, token_id, task_name, event_type):
        self.time = time
        self.token_id = token_id
        self.task_name = task_name
        self.event_type = event_type

    def __lt__(self, other):
        return self.time < other.time

# Read simulation parameters
def get_simulation_parameters(simulation_metrics):
    start_event = simulation_metrics[simulation_metrics['Type'].str.lower() == 'start event']
    if not start_event.empty:
        max_arrival_count = (
            int(start_event['Max arrival'].iloc[0]) 
            if 'max arrival' in map(str.lower, start_event.columns) 
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

# Fetch conditional probabilities
def get_condition_probability(simulation_metrics, from_activity, condition_type):
    condition_key = condition_type.split("-")[1].strip()
    row = simulation_metrics.loc[simulation_metrics['Name'] == from_activity]
    if not row.empty and condition_key in row.columns:
        return row.iloc[0][condition_key]
    logging.warning("Probability for condition '%s' not found. Defaulting to 0.5.", condition_type)
    return 0.5  # Default probability

# Read process sequences from file
def read_output_sequences(file_path):
    transitions = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('->')
            from_task = parts[0].strip()
            to_task, type_info = parts[1].rsplit('[Type: ', 1)
            to_task = to_task.strip()
            type_info = type_info.rstrip(']').strip()
            transitions.append({'From': from_task, 'To': to_task, 'Type': type_info})
    return pd.DataFrame(transitions)

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

# Build paths for the simulation
def build_paths(df):
    paths = []

    def traverse(current_task, current_path):
        next_steps = df[df['From'] == current_task]
        if next_steps.empty:
            paths.append(current_path[:])
            logging.info("Path completed: %s", " -> ".join(current_path))
            return
        for _, row in next_steps.iterrows():
            current_path.append(f"{row['From']} -> {row['To']} [Type: {row['Type']}]")
            traverse(row['To'], current_path)
            current_path.pop()

    start_tasks = set(df['From']) - set(df['To'])
    for start_task in start_tasks:
        traverse(start_task, [])

    logging.info("All paths generated:")
    for idx, path in enumerate(paths, start=1):
        logging.info("Path %d: %s", idx, " -> ".join(path))

    return paths

# Discrete event simulation
def discrete_event_simulation(max_arrival_count, arrival_interval_minutes, simulation_days, paths, simulation_metrics, start_time):
    previous_date = None

    simulation_end_date = start_time + timedelta(days=simulation_days)
    print('Simuation End Date:', simulation_end_date)

    resource_busy_periods = {resource: [] for resource in simulation_metrics['Resource'].dropna().unique()}
    activity_processing_times = {}
    event_queue = []
    active_tokens = {}
    activity_processing_times = {}
    active_resources = {resource: 0 for resource in resource_busy_periods.keys()}
    available_resources = simulation_metrics.set_index('Resource')['Available Resources'].to_dict()

    # Helper to get task duration
    def get_task_duration(task_name):
        row = simulation_metrics.loc[simulation_metrics['Name'] == task_name]
        if not row.empty and 'Duration' in row.columns:
            return int(row['Duration'].iloc[0])
        return 1  # Default duration

    # Initialize simulation start time
    current_time = start_time

    for token_id in range(max_arrival_count):
        arrival_time = current_time + timedelta(minutes=token_id * arrival_interval_minutes)
        heapq.heappush(event_queue, Event(arrival_time, token_id, paths[0][0].split(" -> ")[0], 'start'))
        active_tokens[token_id] = {'current_task': None, 'start_time': arrival_time, 'end_time': None}

    completed_tasks = {token_id: set() for token_id in range(max_arrival_count)}

    while event_queue and current_time < simulation_end_date:
        event = heapq.heappop(event_queue)
        current_time = event.time

        # Ensure current time is adjusted to next work period if outside work hours
        if not is_work_time(current_time):
            current_time = advance_to_work_time(current_time)
            heapq.heappush(event_queue, Event(current_time, event.token_id, event.task_name, event.event_type))
            continue

        # Update date for simulation logs if changed
        current_date = current_time.date()
        if current_date != previous_date:
            logging.info("Simulation date: %s", current_date)
            print(f"Simulation date: {current_date}")
            previous_date = current_date

        # Handle 'start' and 'end' events
        if event.event_type == 'start':
            task_name = event.task_name.strip()

            # Skip if this task was already completed by the token
            if task_name in completed_tasks[event.token_id]:
                continue

            logging.info("Token %d starting task '%s' at time %s.", event.token_id, task_name, current_time)
            active_tokens[event.token_id]['current_task'] = task_name

            if task_name not in activity_processing_times:
                activity_processing_times[task_name] = []
            activity_processing_times[task_name].append([current_time, None])

            resource_name = simulation_metrics.loc[simulation_metrics['Name'] == task_name, 'Resource'].values
            if resource_name.size > 0 and pd.notna(resource_name[0]):
                resource_name = resource_name[0]
                if active_resources[resource_name] >= available_resources.get(resource_name, 1):
                    logging.info("Token %d waiting for resource '%s' at time %s.", event.token_id, resource_name, current_time)
                    heapq.heappush(event_queue, Event(current_time + timedelta(minutes=1), event.token_id, task_name, 'start'))
                    continue

                active_resources[resource_name] += 1
                resource_busy_periods[resource_name].append([current_time, None])
                logging.info("Resource '%s' is being used by Token %d.", resource_name, event.token_id)

            task_duration = timedelta(minutes=get_task_duration(task_name))
            heapq.heappush(event_queue, Event(current_time + task_duration, event.token_id, task_name, 'end'))

        elif event.event_type == 'end':
            task_name = event.task_name.strip()
            resource_name = simulation_metrics.loc[simulation_metrics['Name'] == task_name, 'Resource'].values
            if resource_name.size > 0 and pd.notna(resource_name[0]):
                resource_name = resource_name[0]
                active_resources[resource_name] -= 1
                for period in resource_busy_periods[resource_name]:
                    if period[1] is None:
                        period[1] = current_time
                        break

                logging.info("Resource '%s' released by Token %d at time %s.", resource_name, event.token_id, current_time)

            logging.info("Token %d finished task '%s' at time %s.", event.token_id, task_name, current_time)

            # Mark task as completed for this token
            completed_tasks[event.token_id].add(task_name)

            for period in activity_processing_times[task_name]:
                if period[1] is None:
                    period[1] = current_time
                    break

            active_tokens[event.token_id]['current_task'] = None

            # End of event processing
            active_tokens[event.token_id]['end_time'] = current_time

            # Ensure the task's duration is tracked correctly
            if task_name in activity_processing_times:
                for i, (start, end) in enumerate(activity_processing_times[task_name]):
                    if end is None:
                        activity_processing_times[task_name][i] = (start, current_time)  # Record end time
                        break
            else:
                logging.warning("Task '%s' was not properly started by Token %d.", task_name, event.token_id)

            # Move to the next task in the path if conditions are met
            for path in paths:
                for segment in path:
                    if segment.startswith(f"{task_name} ->"):
                        to_task, type_info = re.match(r".+ -> (.+) \[Type: (.+)\]", segment).groups()
                        if type_info.startswith("CONDITION"):
                            if random.random() <= get_condition_probability(simulation_metrics, task_name, type_info):
                                if to_task.lower() != 'stop':  # Avoid re-starting the process after 'stop'
                                    heapq.heappush(event_queue, Event(current_time, event.token_id, to_task, 'start'))
                                    active_tokens[event.token_id]['current_task'] = to_task
                        elif to_task.lower() != 'stop':  # Avoid re-starting the process after 'stop'
                            heapq.heappush(event_queue, Event(current_time, event.token_id, to_task, 'start'))
                            active_tokens[event.token_id]['current_task'] = to_task
                        break


    total_simulation_time = max((simulation_end_date - start_time).total_seconds() / 3600, 0)  # in hours

    print("\nActivity Processing Times:")
    for activity, times in activity_processing_times.items():
        # Calculate durations from (start, end) pairs
        durations = [(end - start).total_seconds() / 60 for start, end in times if start and end]
        if durations:
            min_time = min(durations)
            avg_time = sum(durations) / len(durations)
            max_time = max(durations)
            print(f"Activity '{activity}': Min = {min_time:.2f} min, Avg = {avg_time:.2f} min, Max = {max_time:.2f} min")
        else:
            print(f"Activity '{activity}': No valid durations recorded.")


    print("\nResource Utilization:")
    total_work_hours = (simulation_days * 6)  # 6 work hours/day
    for resource, periods in resource_busy_periods.items():
        total_busy_time = sum((end - start).total_seconds() for start, end in periods if end) / 3600  # Convert seconds to hours
        utilization = (total_busy_time / total_work_hours) * 100  # Based on total work hours
        if total_work_hours > 0:
            print(f"Resource '{resource}': Utilization = {utilization:.2f}%")
        else:
            print(f"Resource '{resource}': No work hours available.")

# Main function
def main():
    simulation_metrics_path = 'C:/Test Data/Bizagi/simulation_metrics.xlsx'
    xpdl_file_path = 'C:/Test Data/Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews.xpdl'
    output_sequences_path = 'output_sequences.txt'

    # Define work calendar
    simulation_days = 30  # Number of days for the simulation
    start_time = datetime(2023, 1, 2, 6)  # Monday 6:00 AM

    process_sequences = parse_xpdl_to_sequences(xpdl_file_path, output_sequences_path)
    print("Process Sequences:", process_sequences)

    simulation_metrics = pd.read_excel(simulation_metrics_path, sheet_name=0)
    df = read_output_sequences(output_sequences_path)
    paths = build_paths(df)
    max_arrival_count, arrival_interval_minutes = get_simulation_parameters(simulation_metrics)
    discrete_event_simulation(max_arrival_count, arrival_interval_minutes, simulation_days, paths, simulation_metrics, start_time)

if __name__ == "__main__":
    main()
