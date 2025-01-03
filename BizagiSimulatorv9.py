# Bug: Not calculating utilizations correctly, current at zero at the end; need to calculate the avg time to process a token
import pandas as pd
import heapq
import random
import re
import logging

# Configure logging
logging.basicConfig(filename='simulation_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# File paths
output_sequences_path = 'output_sequences.txt'
simulation_metrics_path = 'C:/Test Data/Bizagi/simulation_metrics.xlsx'

# Extract max_arrival_count and arrival_interval_minutes dynamically from simulation_metrics
def get_simulation_parameters():
   # Ensure 'Type' is compared in a case-insensitive manner
    start_event = simulation_metrics[simulation_metrics['Type'].str.lower() == 'start event']
    
    if not start_event.empty:
        # Access columns in a case-insensitive manner
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

# Step 1: Read the output_sequences.txt file
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

# Step 2: Traverse paths to build unique sequences
def build_paths(df):
    paths = []

    def traverse(current_task, current_path):
        # Find all transitions from the current task
        next_steps = df[df['From'] == current_task]
        if next_steps.empty:
            # No further transitions, end the current path
            paths.append(current_path[:])
            return
        for _, row in next_steps.iterrows():
            current_path.append(f"{row['From']} -> {row['To']} [Type: {row['Type']}]")
            traverse(row['To'], current_path)
            current_path.pop()  # Backtrack

    # Start traversal from tasks with no "From"
    start_tasks = set(df['From']) - set(df['To'])
    for start_task in start_tasks:
        traverse(start_task, [])
    return paths

# Load simulation metrics
simulation_metrics = pd.read_excel(simulation_metrics_path, sheet_name=0)
print('simulation_metrics=', simulation_metrics)

# Extract probabilities for conditions
def get_condition_probability(from_activity, condition_type):
    """
    Fetches the probability for a given CONDITION-Yes/No in the 'To' activity.
    """
    condition_key = condition_type.split("-")[1].strip()  # Extract Yes or No from CONDITION-Yes/No
    row = simulation_metrics.loc[simulation_metrics['Name'] == from_activity]
    if not row.empty and condition_key in row.columns:
        return row.iloc[0][condition_key]  # Fetch the probability value
    logging.warning("Probability for condition '%s' not found. Defaulting to 0.5.", condition_type)
    return 0.5  # Default probability if not found

# Initialize resource utilization tracking
resources = simulation_metrics['Resource'].dropna().unique()  # Unique resources
resource_busy_time = {resource: 0 for resource in resources}  # Tracks busy time for each resource

# Step 3: Integrate with simulation
class Event:
    def __init__(self, time, token_id, task_name, event_type):
        self.time = time
        self.token_id = token_id
        self.task_name = task_name
        self.event_type = event_type

    def __lt__(self, other):
        return self.time < other.time

def discrete_event_simulation(max_arrival_count, arrival_interval_minutes, max_time, paths):
    event_queue = []
    active_tokens = {token_id: {'current_task': None, 'start_time': None} for token_id in range(max_arrival_count)}

    # Resource tracking
    active_resources = {resource: 0 for resource in resource_busy_time.keys()}
    available_resources = simulation_metrics.set_index('Resource')['Available Resources'].to_dict()  # Get available resources

    # Schedule token arrivals
    for token_id in range(max_arrival_count):
        arrival_time = token_id * arrival_interval_minutes
        first_task = paths[0][0].split(" -> ")[0]  # Start from the first task in the sequence
        heapq.heappush(event_queue, Event(arrival_time, token_id, first_task, 'start'))

    # Main simulation loop
    current_time = 0
    while event_queue and current_time <= max_time:
        event = heapq.heappop(event_queue)
        current_time = event.time

        if event.event_type == 'start':
            task_name = event.task_name.strip()
            active_tokens[event.token_id]['current_task'] = task_name
            active_tokens[event.token_id]['start_time'] = current_time
            logging.info("Token %d starting task '%s' at time %d.", event.token_id, task_name, current_time)

            # Assign resource if needed
            resource = simulation_metrics.loc[simulation_metrics['Name'] == task_name, 'Resource'].values
            if resource.size > 0 and pd.notna(resource[0]):
                resource_name = resource[0]

                # Check resource availability
                if active_resources[resource_name] >= available_resources.get(resource_name, 1):
                    # Wait if no resource is available
                    logging.info("Token %d waiting for resource '%s' at time %d.", event.token_id, resource_name, current_time)
                    heapq.heappush(event_queue, Event(current_time + 1, event.token_id, task_name, 'start'))
                    continue

                # Use the resource
                active_resources[resource_name] += 1
                active_tokens[event.token_id]['resource_name'] = resource_name  # Track resource assigned
                logging.info("Resource '%s' is being used by Token %d.", resource_name, event.token_id)

            # Find the next task(s) in the paths
            for path in paths:
                for segment in path:
                    if segment.startswith(f"{task_name} ->"):
                        to_task, type_info = re.match(r".+ -> (.+) \[Type: (.+)\]", segment).groups()
                        if type_info.startswith("CONDITION"):
                            # Handle conditional transitions
                            probability = get_condition_probability(task_name, type_info)
                            if random.random() <= probability:
                                heapq.heappush(event_queue, Event(current_time + 1, event.token_id, to_task, 'start'))
                                logging.info(
                                    "Token %d took conditional path '%s' with probability %.2f.",
                                    event.token_id, to_task, probability
                                )
                        else:
                            # Handle non-conditional transitions
                            heapq.heappush(event_queue, Event(current_time + 1, event.token_id, to_task, 'start'))
                            logging.info(
                                "Token %d scheduled non-conditional next task '%s'.",
                                event.token_id, to_task
                            )
                        break

        elif event.event_type == 'end':
            # Release resource after task completion
            task_name = event.task_name.strip()
            resource_name = active_tokens[event.token_id].get('resource_name')
            if resource_name:
                duration = current_time - active_tokens[event.token_id]['start_time']
                resource_busy_time[resource_name] += duration  # Increment busy time by task duration
                active_resources[resource_name] -= 1  # Free the resource
                logging.info("Resource '%s' released by Token %d after %.2f minutes.", resource_name, event.token_id, duration)

    # Calculate and log resource utilization
    total_simulation_time = max_time
    for resource, busy_time in resource_busy_time.items():
        utilization = (busy_time / total_simulation_time) * 100
        print(f"Resource {resource} utilization: {utilization:.2f}%")
        logging.info("Resource %s utilization: %.2f%%", resource, utilization)

    return active_tokens

def main():
    df = read_output_sequences(output_sequences_path)
    paths = build_paths(df)

    # Fetch parameters from simulation metrics
    max_arrival_count, arrival_interval_minutes = get_simulation_parameters()

    # Use total simulation time (example: 120 minutes or other suitable value)
    max_time = 120  # Example value

    # Run the simulation
    active_tokens = discrete_event_simulation(max_arrival_count, arrival_interval_minutes, max_time, paths)

    print("Simulation complete. Check simulation_log.txt for detailed logs.")

if __name__ == "__main__":
    main()
