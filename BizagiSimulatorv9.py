import pandas as pd
from xml.etree import ElementTree as ET
import heapq
import random
import re
import logging

# Configure logging
logging.basicConfig(filename='simulation_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Importing the parse_xpdl_to_sequences function
from xpdl_parser import parse_xpdl_to_sequences

# File paths
simulation_metrics_path = 'C:/Test Data/Bizagi/simulation_metrics.xlsx'  # Excel file
xpdl_file_path = 'C:/Test Data/Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews.xpdl'

# Load simulation metrics
simulation_metrics = pd.read_excel(simulation_metrics_path, sheet_name=0)

# Parse the XPDL file
process_sequences = parse_xpdl_to_sequences(xpdl_file_path, "output_sequences.txt")
logging.info("Process Sequences: %s", process_sequences)

# Extract activity types
activity_types = dict(zip(process_sequences['From'], process_sequences['Type']))

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

logging.info("Loaded Resources: %s", resources)
logging.info("Task Resources: %s", task_resources)
logging.info("Task Durations: %s", task_durations)

# Extract token arrival metrics
start_event_metrics = simulation_metrics[simulation_metrics["Type"] == "Start event"]
max_arrival_count = int(start_event_metrics["Max arrival count"].dropna().iloc[0])
arrival_interval_minutes = int(start_event_metrics["Arrival interval"].dropna().iloc[0])

# Set simulation maximum time
max_time = 30 * 24 * 60  # 30 days in minutes

logging.info("Number of tokens (max_arrival_count): %d", max_arrival_count)
logging.info("Arrival interval (minutes): %d", arrival_interval_minutes)
logging.info("Maximum simulation time (minutes): %d", max_time)

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
        first_task = process_sequences.iloc[0]['From']
        heapq.heappush(event_queue, Event(arrival_time, token_id, first_task, 'start'))

    # Main simulation loop
    current_time = 0
    while event_queue and current_time <= max_time:
        event = heapq.heappop(event_queue)
        current_time = event.time

        if event.event_type == 'start':
            task_name = event.task_name
            task_type = activity_types.get(task_name, "Unknown")

            # Extract path_condition if task_type contains "CONDITION"
            path_condition = None
            if "CONDITION" in task_type:
                path_condition = task_type.split("-")[1].strip().lower()

            logging.info("Starting task '%s' of type '%s' at time %d.", task_name, task_type, current_time)

            if task_name in process_sequences['From'].values:
                next_tasks = process_sequences[process_sequences['From'] == task_name]['To'].tolist()

                for next_task in next_tasks:
                    if path_condition and task_type.startswith("CONDITION"):
                        # Find the row for the Gateway
                        gateway_row = simulation_metrics[(simulation_metrics["Name"] == task_name) &
                                                         (simulation_metrics["Type"] == "Gateway")]

                        if not gateway_row.empty:
                            # Find the column with the matching path_condition
                            column_name = next((col for col in gateway_row.columns if col.lower() == path_condition), None)
                            if column_name:
                                probability = gateway_row.iloc[0][column_name]
                                logging.info("Probability for path '%s' in task '%s' is %s.", path_condition, task_name, probability)

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
                                logging.warning("Resource '%s' unavailable for task '%s' at time %d.", resource_type, next_task, current_time)
                    else:
                        heapq.heappush(event_queue, Event(current_time, event.token_id, next_task, 'start'))  # No delay
            else:
                logging.info("End of sequence for token %d at task '%s'.", event.token_id, task_name)
        elif event.event_type == 'end':
            task_name = event.task_name
            token_id = event.token_id
            resource_type = task_resources.get(task_name)

            if resource_type and resource_type in resource_queues:
                resource_queues[resource_type].remove(token_id)
                resource_busy_time[resource_type] += current_time - active_tokens[token_id]['start_time']
                active_tokens[token_id]['current_task'] = None
                logging.info("Token %d completed task '%s' at time %d.", token_id, task_name, current_time)

    utilization_metrics = {
        res: (busy_time / (resources[res] * max_time)) * 100 for res, busy_time in resource_busy_time.items()
    }
    return utilization_metrics, resource_busy_time

# Run the simulation
utilization, busy_time = discrete_event_simulation(max_arrival_count, arrival_interval_minutes, max_time)

# Output results to log
logging.info("Utilization Metrics: %s", utilization)
logging.info("Resource Busy Time: %s", busy_time)
