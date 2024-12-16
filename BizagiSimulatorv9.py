# BUG: Seems to only be simuating the first activity
import pandas as pd
import heapq
import random
import re
import logging

# Configure logging
logging.basicConfig(filename='simulation_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# File paths
simulation_metrics_path = 'C:/Test Data/Bizagi/simulation_metrics.xlsx'
output_sequences_path = 'output_sequences.txt'

# Load simulation metrics
simulation_metrics = pd.read_excel(simulation_metrics_path, sheet_name=0)

# Load process sequences from output_sequences.txt
process_sequences = pd.read_csv(output_sequences_path, sep='->', names=['From', 'ToType'], engine='python')
process_sequences[['To', 'Type']] = process_sequences['ToType'].str.split('[', expand=True)
process_sequences['Type'] = process_sequences['Type'].str.replace(r'\[Type: ', '').str.replace(r'\]', '', regex=True)
process_sequences.drop(columns=['ToType'], inplace=True)

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
max_time = 2 * 24 * 60  # 2 days in minutes

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
            active_tokens[event.token_id]['current_task'] = task_name
            active_tokens[event.token_id]['start_time'] = current_time

            next_tasks = process_sequences[process_sequences['From'] == task_name]

            logging.info("Token %d starting task '%s' at time %d.", event.token_id, task_name, current_time)

            for _, transition in next_tasks.iterrows():
                to_task = transition['To'].strip()
                condition_type = transition['Type'].strip()

                if to_task == "Stop":
                    logging.info("Token %d reached the end of the path at task '%s'.", event.token_id, task_name)
                    continue  # End the token's path

                if condition_type.startswith("CONDITION"):
                    if condition_type.endswith("No") and random.random() > 0.5:
                        logging.info("Token %d did not meet condition for path '%s'.", event.token_id, to_task)
                        continue  # Skip No path with probability

                if to_task in task_durations:
                    duration = random.triangular(
                        task_durations[to_task]["min"],
                        task_durations[to_task]["max"],
                        task_durations[to_task]["mode"]
                    )
                    resource = task_resources.get(to_task, None)
                    if resource in resources:
                        resource_queues[resource].append(event.token_id)
                        heapq.heappush(event_queue, Event(current_time + duration, event.token_id, to_task, 'end'))

        elif event.event_type == 'end':
            task_name = event.task_name
            token_id = event.token_id
            resource_type = task_resources.get(task_name)

            if resource_type and resource_type in resource_queues:
                resource_queues[resource_type].remove(token_id)

                # Safeguard against missing start_time
                start_time = active_tokens[token_id].get('start_time', current_time)
                resource_busy_time[resource_type] += current_time - start_time
                logging.info("Token %d completed task '%s' at time %d.", token_id, task_name, current_time)

    return resource_busy_time

# Run the simulation
busy_time = discrete_event_simulation(max_arrival_count, arrival_interval_minutes, max_time)

# Log final resource usage
logging.info("Resource Busy Time: %s", busy_time)
