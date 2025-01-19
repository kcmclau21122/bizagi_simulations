# simulation/data_handler.py
import pandas as pd
from collections import defaultdict
import logging

def read_output_sequences(file_path):
    transitions = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('->')
            from_task = parts[0].strip()
            to_task, type_info = parts[1].rsplit('[Type: ', 1)
            to_task = to_task.strip()
            type_info = type_info.rstrip(']').strip()
            transitions.append({'from': from_task, 'to': to_task, 'type': type_info})  # Use lowercase keys

    transitions_df = pd.DataFrame(transitions)

    # Identify "Start" and "Stop" activities
    all_from_tasks = set(transitions_df['from'])
    all_to_tasks = set(transitions_df['to'])

    # Start activities: appear in 'from' but not in 'to'
    start_activities = all_from_tasks - all_to_tasks

    # Stop activities: appear in 'to' but not in 'from'
    stop_activities = all_to_tasks - all_from_tasks
    transitions_df.loc[transitions_df['to'].isin(stop_activities), 'type'] = 'Stop'

    return transitions_df


def remove_duplicate_activities(path):
    """
    Removes duplicate first activities in the next step if they match the last activity in the previous step.
    Once the unique path is created, ensures the first activity in the path has [Type: Start].
    """
    if not path:
        return path

    unique_path = []
    previous_to_activity = None

    # Create unique path by removing self-links
    for step in path:
        parts = step.split("->")
        if len(parts) != 2:
            unique_path.append(step)  # Add malformed steps as is
            continue
        
        from_activity = parts[0].split(" [Type:")[0].strip()  # Extract "from" activity name
        to_activity = parts[1].split(" [Type:")[0].strip()    # Extract "to" activity name

        # Skip the step if "from_activity" matches "previous_to_activity"
        if from_activity == previous_to_activity:
            continue

        unique_path.append(step)
        previous_to_activity = to_activity  # Update for the next iteration

    # Set the first activity's type to [Type: Start]
    if unique_path:
        first_step = unique_path[0]
        parts = first_step.split("->")
        if len(parts) == 2:
            from_activity = parts[0].split(" [Type:")[0].strip()  # Extract "from" activity name
            parts[0] = f"{from_activity} [Type: Start]"  # Set or replace the type of the first activity
            unique_path[0] = " ->".join(parts)

    return unique_path


def build_paths_old(df):
    paths = []

    def traverse(current_task, current_path):
        next_steps = df[df['from'] == current_task]  # Use lowercase key
        if next_steps.empty:
            paths.append(current_path[:])
            logging.info("Path completed: %s", " -> ".join(current_path))
            return
        for _, row in next_steps.iterrows():
            next_step = f"{row['from']} [Type: {row['type']}] -> {row['to']} [Type: {row['type']}]"
            # Skip adding the step if it repeats the last activity
            if current_path and current_path[-1] == next_step:
                continue
            current_path.append(next_step)
            traverse(row['to'], current_path)  # Use lowercase key
            current_path.pop()

    start_tasks = set(df['from']) - set(df['to'])  # Use lowercase keys
    for start_task in start_tasks:
        traverse(start_task, [])

    logging.info("All paths generated:")
    for idx, path in enumerate(paths, start=1):
        logging.info("Path %d: %s", idx, " -> ".join(path))

    # Remove duplicate activities before returning the path
    cleaned_paths = [remove_duplicate_activities(path) for path in paths]

    return cleaned_paths

# Define the function to build paths
from collections import defaultdict

def build_paths(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    # Parse the "from" and "to" activities from the lines
    connections = []
    for line in lines:
        from_activity = line.split(' -> ')[0]
        to_part = line.split(' -> ')[1]
        to_activity = to_part[:to_part.index(' [')]
        connections.append((from_activity, to_activity))

    # Build a mapping from "from" activities to "to" activities
    adjacency_list = defaultdict(list)
    for from_activity, to_activity in connections:
        adjacency_list[from_activity].append(to_activity)

    # Find the start activity
    start_activity = next((activity for activity in adjacency_list.keys() if "[Type: Start]" in activity), None)
    if not start_activity:
        raise ValueError("No start activity found in the input file.")

    # Recursive function to build paths
    def dfs(current_path, all_paths):
        last_activity = current_path[-1]

        # If the current activity is a stop activity, save the path
        if "[Type: Stop]" in last_activity:
            all_paths.append(current_path)
            return

        # Explore next activities
        if last_activity in adjacency_list:  # Check for valid transitions
            for next_activity in adjacency_list[last_activity]:
                if next_activity not in current_path:  # Prevent loops
                    dfs(current_path + [next_activity], all_paths)
        else:
            # Dead-end case: save the path
            all_paths.append(current_path)

    # Initialize the DFS
    all_paths = []
    dfs([start_activity], all_paths)

    # Debugging output for verification
    print(f"Start Activity: {start_activity}")
    print(f"Adjacency List: {dict(adjacency_list)}")
    print(f"All Paths: {all_paths}")

    # Convert the results to a DataFrame
    if all_paths:  # Ensure there are paths to process
        path_data = {"Path Index": [], "Activity": [], "Type": []}
        for index, path in enumerate(all_paths):
            for activity in path:
                path_data["Path Index"].append(index)
                path_data["Activity"].append(activity)
                # Extract the type from the activity string
                activity_type = activity[activity.index("["):activity.index("]") + 1] if "[" in activity and "]" in activity else "[Unknown]"
                path_data["Type"].append(activity_type)
        df = pd.DataFrame(path_data)
    else:
        df = pd.DataFrame(columns=["Path Index", "Activity", "Type"])  # Empty DataFrame in case of no paths

    return df

# Extract start tasks from paths DataFrame
def extract_start_tasks(paths):
    start_tasks = set()
    grouped = paths.groupby("Path Index")  # Group by path index
    for _, group in grouped:
        first_activity = group.iloc[0]["Activity"]  # Get the first activity in each path
        if "[type: start]" in first_activity.lower():
            activity_name = first_activity.split("[type:")[0].strip()  # Extract activity name
            start_tasks.add(activity_name)
    return start_tasks
