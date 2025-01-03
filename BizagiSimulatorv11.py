import pandas as pd
import heapq
import random
import re
import logging
import os

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
def discrete_event_simulation(max_arrival_count, arrival_interval_minutes, max_time, paths, simulation_metrics):
    resource_busy_periods = {resource: [] for resource in simulation_metrics['Resource'].dropna().unique()}
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

    for token_id in range(max_arrival_count):
        arrival_time = token_id * arrival_interval_minutes
        first_task = paths[0][0].split(" -> ")[0]
        heapq.heappush(event_queue, Event(arrival_time, token_id, first_task, 'start'))
        active_tokens[token_id] = {'current_task': None, 'start_time': None, 'end_time': None}

    current_time = 0
    while event_queue and current_time <= max_time:
        event = heapq.heappop(event_queue)
        current_time = event.time
        print('event_type:', event.event_type)

        if event.event_type == 'start':
            task_name = event.task_name.strip()
            active_tokens[event.token_id]['current_task'] = task_name
            if active_tokens[event.token_id]['start_time'] is None:
                active_tokens[event.token_id]['start_time'] = current_time

            logging.info("Token %d starting task '%s' at time %d.", event.token_id, task_name, current_time)

            if task_name.lower() == 'stop':
                active_tokens[event.token_id]['end_time'] = current_time
                logging.info("Token %d reached the end of the process at time %d.", event.token_id, current_time)
                continue

            activity_processing_times.setdefault(task_name, []).append(current_time)

            resource = simulation_metrics.loc[simulation_metrics['Name'] == task_name, 'Resource'].values
            if resource.size > 0 and pd.notna(resource[0]):
                resource_name = resource[0]

                if active_resources[resource_name] >= available_resources.get(resource_name, 1):
                    logging.info("Token %d waiting for resource '%s' at time %d.", event.token_id, resource_name, current_time)
                    heapq.heappush(event_queue, Event(current_time + 1, event.token_id, task_name, 'start'))
                    continue

                active_resources[resource_name] += 1
                resource_busy_periods[resource_name].append((current_time, None))
                active_tokens[event.token_id]['resource_name'] = resource_name
                logging.info("Resource '%s' is being used by Token %d.", resource_name, event.token_id)

            task_duration = get_task_duration(task_name)
            heapq.heappush(event_queue, Event(current_time + task_duration, event.token_id, task_name, 'end'))

        elif event.event_type == 'end':
            task_name = event.task_name.strip()
            resource_name = active_tokens[event.token_id].get('resource_name')
            if resource_name:
                duration = get_task_duration(task_name)
                active_resources[resource_name] -= 1
                for i, (start, end) in enumerate(resource_busy_periods[resource_name]):
                    if end is None:
                        resource_busy_periods[resource_name][i] = (start, current_time)
                        break
                logging.info("Resource '%s' released by Token %d at time %d.", resource_name, event.token_id, current_time)

            logging.info("Token %d finished task '%s' at time %d.", event.token_id, task_name, current_time)

            for path in paths:
                for segment in path:
                    if segment.startswith(f"{task_name} ->"):
                        to_task, type_info = re.match(r".+ -> (.+) \[Type: (.+)\]", segment).groups()
                        if type_info.startswith("CONDITION"):
                            probability = get_condition_probability(simulation_metrics, task_name, type_info)
                            if random.random() <= probability:
                                heapq.heappush(event_queue, Event(current_time, event.token_id, to_task, 'start'))
                        elif to_task.lower() == 'stop':
                            active_tokens[event.token_id]['end_time'] = current_time
                        else:
                            heapq.heappush(event_queue, Event(current_time, event.token_id, to_task, 'start'))
                        break

    total_simulation_time = max(current_time, max_time)
    for token_id, token in active_tokens.items():
        print(f"Token {token_id}: Start Time = {token['start_time']}, End Time = {token['end_time']}")
        if token['end_time'] is None:
            logging.warning("Token %d did not complete processing.", token_id)

    print("\nActivity Processing Times:")
    for activity, times in activity_processing_times.items():
        durations = [t2 - t1 for t1, t2 in zip(times, times[1:]) if t2 > t1]
        min_time = min(durations) if durations else 0
        avg_time = sum(durations) / len(durations) if durations else 0
        max_time = max(durations) if durations else 0
        print(f"Activity '{activity}': Min = {min_time}, Avg = {avg_time}, Max = {max_time}")

    print("\nResource Utilization:")
    for resource, periods in resource_busy_periods.items():
        total_busy_time = sum(end - start for start, end in periods if end is not None)
        utilization = (total_busy_time / total_simulation_time) * 100
        print(f"Resource '{resource}': Utilization = {utilization:.2f}%")

# Main function
def main():
    simulation_metrics_path = 'C:/Test Data/Bizagi/simulation_metrics.xlsx'
    xpdl_file_path = 'C:/Test Data/Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews.xpdl'
    output_sequences_path = 'output_sequences.txt'

    process_sequences = parse_xpdl_to_sequences(xpdl_file_path, output_sequences_path)
    print("Process Sequences:", process_sequences)

    simulation_metrics = pd.read_excel(simulation_metrics_path, sheet_name=0)
    df = read_output_sequences(output_sequences_path)
    paths = build_paths(df)
    max_arrival_count, arrival_interval_minutes = get_simulation_parameters(simulation_metrics)
    max_time = 240  # Example simulation time limit
    discrete_event_simulation(max_arrival_count, arrival_interval_minutes, max_time, paths, simulation_metrics)

if __name__ == "__main__":
    main()

