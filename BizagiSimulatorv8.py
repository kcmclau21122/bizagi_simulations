# BUG - NEED TO FIX CERIFIER NOT BEING USED. NOT SURE IF TASKS ASSIGNED TO CERTIFIER ARE NOT INCLUDED IN THE PROCSS FLOW

import pandas as pd
from xml.etree import ElementTree as ET
import heapq
import random

# File paths
simulation_metrics_path = 'C:/Test Data/Bizagi/simulation_metrics.csv'
simulation_results_path = 'C:/Test Data/Bizagi/simulation_results.xlsx'
xpdl_file_path = 'C:/Test Data/Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews.xpdl'

# Load input data
simulation_metrics = pd.read_csv(simulation_metrics_path)
simulation_results = pd.ExcelFile(simulation_results_path)
resources_data = simulation_results.parse('Resources')

# Parse the XPDL file for process flow
tree = ET.parse(xpdl_file_path)
root = tree.getroot()
namespaces = {'xpdl': root.tag.split('}')[0].strip('{')}  # Extract namespace

process_flow = []
for process in root.findall('.//xpdl:WorkflowProcess', namespaces):
    for transition in process.findall('.//xpdl:Transition', namespaces):
        from_activity = transition.get('From')
        to_activity = transition.get('To')
        process_flow.append({'From': from_activity, 'To': to_activity})

# Map activity IDs to task names
activity_mapping = {}
for process in root.findall('.//xpdl:WorkflowProcess', namespaces):
    for activity in process.findall('.//xpdl:Activity', namespaces):
        activity_id = activity.get('Id')
        activity_name = activity.get('Name')
        activity_mapping[activity_id] = activity_name

# Replace IDs with task names in the process flow
for flow in process_flow:
    flow['From'] = activity_mapping.get(flow['From'], flow['From'])
    flow['To'] = activity_mapping.get(flow['To'], flow['To'])

# Ensure Start Event connects to the first task
start_event_tasks = [flow['To'] for flow in process_flow if flow['From'] == 'Start Event']
if not start_event_tasks:
    print("Error: Start Event does not connect to any task. Adding default connection.")
    process_flow.append({'From': 'Start Event', 'To': 'Review AM using Asset Change Tracker (5.5.13.1)'})

print("Updated Process Flow:", process_flow)

# Map tasks to resources and durations from simulation_metrics
task_durations = {}
task_resources = {}
routing_probs = {}

for _, row in simulation_metrics.iterrows():
    task_name = row['Name']
    task_type = row['Type']
    if task_type == 'Task':
        task_durations[task_name] = row['Avg Time']
        task_resources[task_name] = row['Resource']
    elif task_type == 'Gateway':
        routing_probs[task_name] = row['Probability']

# Extract available resources from simulation_metrics where Type == "Task"
resources = {}
task_metrics = simulation_metrics[simulation_metrics['Type'] == 'Task']  # Filter tasks only

for _, row in task_metrics.iterrows():
    resource_name = row['Resource']
    if pd.notna(row['Available Resources']):  # Check if 'Available Resources' is not NaN
        resources[resource_name] = int(row['Available Resources'])

print("Loaded resources:", resources)

# Extract token arrival metrics from "Start event"
start_event_metrics = simulation_metrics[simulation_metrics['Type'] == 'Start event']
max_arrival_count = int(start_event_metrics['Max arrival count'].iloc[0])  # Number of tokens to process
arrival_interval_minutes = int(start_event_metrics['Arrival interval'].iloc[0])  # Interval between token arrivals (minutes)

print(f"Number of tokens: {max_arrival_count}, Arrival interval: {arrival_interval_minutes} minutes")

# Discrete-Event Simulation
class Event:
    def __init__(self, time, token_id, task_name, event_type):
        self.time = time
        self.token_id = token_id
        self.task_name = task_name
        self.event_type = event_type

    def __lt__(self, other):
        return self.time < other.time  # Priority based on time


# Discrete-Event Simulation
def discrete_event_simulation(max_arrival_count, arrival_interval_minutes, max_time):
    """
    Discrete-event simulation for token processing.
    Args:
        max_arrival_count: Number of tokens to simulate.
        arrival_interval_minutes: Time interval between token arrivals.
        max_time: Maximum simulation time in minutes.

    Returns:
        Utilization metrics, resource busy times, and final resource counts.
    """
    # Initialize event queue and resource queues
    event_queue = []
    resource_queues = {res: [] for res in resources.keys()}
    active_tokens = {token_id: {'current_task': None, 'start_time': None} for token_id in range(max_arrival_count)}
    resource_busy_time = {res: 0 for res in resources.keys()}  # Track busy time

    # Schedule token arrivals dynamically based on interval
    for token_id in range(max_arrival_count):
        arrival_time = token_id * arrival_interval_minutes  # Tokens arrive sequentially
        heapq.heappush(event_queue, Event(arrival_time, token_id, 'Start Event', 'start'))

    # Main simulation loop
    current_time = 0
    while event_queue and current_time <= max_time:
        # Get the next event
        event = heapq.heappop(event_queue)
        current_time = event.time

        if event.event_type == 'start':
            print(f"Token {event.token_id} attempting to start task {event.task_name} at time {current_time}.")
            task_name = event.task_name
            if task_name == 'Start Event':
                # Route to the first task directly
                next_tasks = [flow['To'] for flow in process_flow if flow['From'] == 'Start Event']
                if not next_tasks:
                    print(f"Error: No tasks connected to {task_name}. Skipping token {event.token_id}.")
                else:
                    for next_task in next_tasks:
                        heapq.heappush(event_queue, Event(current_time, event.token_id, next_task, 'start'))
            else:
                task_duration = task_durations.get(task_name, 0)
                resource_type = task_resources.get(task_name, None)

                if resource_type and len(resource_queues[resource_type]) < resources[resource_type]:
                    # Allocate resource
                    resource_queues[resource_type].append(event.token_id)
                    active_tokens[event.token_id]['current_task'] = task_name
                    active_tokens[event.token_id]['start_time'] = current_time
                    print(f"Token {event.token_id} started task {task_name} using resource {resource_type} at time {current_time}.")

                    # Schedule end event for the task
                    end_time = current_time + task_duration
                    heapq.heappush(event_queue, Event(end_time, event.token_id, task_name, 'end'))

        elif event.event_type == 'end':
            task_name = event.task_name
            token_id = event.token_id
            resource_type = task_resources.get(task_name, None)

            if resource_type:
                # Free resource
                resource_queues[resource_type].remove(token_id)
                task_start_time = active_tokens[token_id]['start_time']
                resource_busy_time[resource_type] += current_time - task_start_time
                print(f"Token {token_id} completed task {task_name} using resource {resource_type} at time {current_time}.")

            # Route token to the next task
            next_tasks = [flow['To'] for flow in process_flow if flow['From'] == task_name]
            for next_task in next_tasks:
                resource = task_resources.get(next_task, None)
                print(f"Routing token {token_id} to task '{next_task}' using resource '{resource}'")
                heapq.heappush(event_queue, Event(current_time, token_id, next_task, 'start'))

    # Calculate utilization metrics
    utilization_metrics = {res: (time / (resources[res] * max_time)) * 100 for res, time in resource_busy_time.items()}

    return utilization_metrics, resource_busy_time


# Run the simulation
utilization, busy_time = discrete_event_simulation(max_arrival_count, arrival_interval_minutes, 1440)

# Output results
print("Utilization Metrics:", utilization)
print("Resource Busy Time:", busy_time)



# Example parameters
max_time = 1440  # 1 day in minutes

# Run the simulation
utilization, busy_time = discrete_event_simulation(max_arrival_count, arrival_interval_minutes, max_time)

# Output results
print("Utilization Metrics:", utilization)
print("Resource Busy Time:", busy_time)


# Example parameters
max_time = 1440  # 1 day in minutes

# Run the simulation
utilization, busy_time = discrete_event_simulation(max_arrival_count, arrival_interval_minutes, max_time)

# Output results
print("Utilization Metrics:", utilization)
print("Resource Busy Time:", busy_time)
