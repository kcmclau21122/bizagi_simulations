import pandas as pd
from xml.etree import ElementTree as ET
import heapq
import random
import re

# Importing the parse_xpdl_to_sequences function
from xpdl_parser import parse_xpdl_to_sequences

# File paths
simulation_metrics_path = 'C:/Test Data/Bizagi/simulation_metrics.xlsx'  # Excel file
xpdl_file_path = 'C:/Test Data/Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews.xpdl'

# Load simulation metrics
simulation_metrics = pd.read_excel(simulation_metrics_path, sheet_name=0)

# Parse the XPDL file
process_sequences = parse_xpdl_to_sequences(xpdl_file_path, "output_sequences.txt")
print("Process Sequences:", process_sequences)

# Map task durations and resources
task_durations = {}
task_resources = {}
resources = {}

# Identify Start Events, End Events, and Gateways for exclusion
excluded_task_types = {"Start event", "End event", "Gateway"}

for _, row in simulation_metrics.iterrows():
    task_name = re.sub(r'\s+', ' ', row["Name"].strip())  # Clean up task names
    task_type = row["Type"]

    if task_type not in excluded_task_types:
        task_durations[task_name] = {
            "min": row.get("Min Time", 0),
            "mode": row.get("Avg Time", 0),
            "max": row.get("Max Time", 0)
        }
        task_resources[task_name] = row.get("Resource", "Unspecified")

        # Handle available resources
        available_resources = row.get("Available Resources", 1)
        resources[row["Resource"]] = int(available_resources) if pd.notna(available_resources) else 1

print("Loaded Resources:", resources)
print("Task Resources:", task_resources)
print("Task Durations:", task_durations)

# Extract token arrival metrics
start_event_metrics = simulation_metrics[simulation_metrics["Type"] == "Start event"]
max_arrival_count = int(start_event_metrics["Max arrival count"].dropna().iloc[0])
arrival_interval_minutes = int(start_event_metrics["Arrival interval"].dropna().iloc[0])

# Set simulation maximum time
max_time = 30 * 24 * 60  # 30 days in minutes

print(f"Number of tokens (max_arrival_count): {max_arrival_count}")
print(f"Arrival interval (minutes): {arrival_interval_minutes}")
print(f"Maximum simulation time (minutes): {max_time}")

# Discrete-Event Simulation
class Event:
    def __init__(self, time, token_id, task_name, event_type):
        self.time = time
        self.token_id = token_id
        self.task_name = task_name
        self.event_type = event_type

    def __lt__(self, other):
        return self.time < other.time

def discrete_event_simulation(max_arrival_count, arrival_interval_minutes, max_time):
    event_queue = []
    resource_queues = {res: [] for res in resources.keys()}
    active_tokens = {token_id: {'current_task': None, 'start_time': None} for token_id in range(max_arrival_count)}
    resource_busy_time = {res: 0 for res in resources.keys()}  # Track busy time

    # Schedule token arrivals
    for token_id in range(max_arrival_count):
        arrival_time = token_id * arrival_interval_minutes
        first_task = list(process_sequences.keys())[0]
        heapq.heappush(event_queue, Event(arrival_time, token_id, first_task, 'start'))

    # Main simulation loop
    current_time = 0
    while event_queue and current_time <= max_time:
        event = heapq.heappop(event_queue)
        current_time = event.time

        if event.event_type == 'start':
            task_name = event.task_name
            if task_name in process_sequences:
                next_tasks = process_sequences[task_name]
                for next_task in next_tasks:
                    if next_task in task_durations:  # Only process tasks with durations
                        task_duration = task_durations[next_task]
                        resource_type = task_resources.get(next_task, "Unspecified")

                        if resource_type and resource_type in resource_queues:
                            if len(resource_queues[resource_type]) < resources.get(resource_type, 1):
                                resource_queues[resource_type].append(event.token_id)
                                active_tokens[event.token_id]['current_task'] = next_task
                                active_tokens[event.token_id]['start_time'] = current_time

                                duration = random.triangular(
                                    task_duration["min"], task_duration["max"], task_duration["mode"]
                                )
                                heapq.heappush(event_queue, Event(current_time + duration, event.token_id, next_task, 'end'))
                            else:
                                print(f"Resource '{resource_type}' unavailable for task '{next_task}' at time {current_time}.")
                    else:
                        heapq.heappush(event_queue, Event(current_time, event.token_id, next_task, 'start'))  # No delay
            else:
                print(f"End of sequence for token {event.token_id} at task {task_name}.")
        elif event.event_type == 'end':
            task_name = event.task_name
            token_id = event.token_id
            resource_type = task_resources.get(task_name)

            if resource_type and resource_type in resource_queues:
                resource_queues[resource_type].remove(token_id)
                resource_busy_time[resource_type] += current_time - active_tokens[token_id]['start_time']
                active_tokens[token_id]['current_task'] = None
                print(f"Token {token_id} completed task '{task_name}' at time {current_time}.")

    utilization_metrics = {
        res: (busy_time / (resources[res] * max_time)) * 100 for res, busy_time in resource_busy_time.items()
    }
    return utilization_metrics, resource_busy_time

# Run the simulation
utilization, busy_time = discrete_event_simulation(max_arrival_count, arrival_interval_minutes, max_time)

# Output results
print("Utilization Metrics:", utilization)
print("Resource Busy Time:", busy_time)
