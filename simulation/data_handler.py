# simulation/data_handler.py
import pandas as pd
from collections import defaultdict
import logging

# Define the function to build paths
def build_sequence_df(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    # Initialize data for the DataFrame
    data = {
        "FromActivity": [],
        "FromActivityType": [],
        "FromGatewayType": [],
        "ToActivity": [],
        "ToActivityType": [],
        "ToGatewayType": []
    }

    for line in lines:
        # Extract "FromActivity" (substring before '[' or '->')
        from_activity = line.split(' -> ')[0].split(' [')[0].strip()
        data["FromActivity"].append(from_activity)

        # Extract "FromActivityType" (substring '[Type:...]')
        from_activity_type = None
        if "[Type:" in line:
            start = line.find("[Type:", 0, line.find(" -> "))
            if start != -1:
                end = line.find("]", start) + 1
                from_activity_type = line[start:end]
        data["FromActivityType"].append(from_activity_type)

        # Extract "FromGatewayType"
        from_gateway_type = None
        if "[Inclusive Gateway]" in line or "[Exclusive Gateway]" in line or "[Parallel Gateway]" in line:
            from_gateway_type = (
                "[Inclusive Gateway]" if "[Inclusive Gateway]" in line else
                "[Exclusive Gateway]" if "[Exclusive Gateway]" in line else
                "[Parallel Gateway]"
            )
        data["FromGatewayType"].append(from_gateway_type)

        # Extract "ToActivity" (substring after '->' and before '[')
        to_activity = line.split(' -> ')[1].split(' [')[0].strip()
        data["ToActivity"].append(to_activity)

        # Extract "ToActivityType" (substring '[Type:...]' after '->')
        to_activity_type = None
        if "[Type:" in line:
            start = line.find("[Type:", line.find(" -> "))
            if start != -1:
                end = line.find("]", start) + 1
                to_activity_type = line[start:end]
        data["ToActivityType"].append(to_activity_type)

        # Extract "ToGatewayType"
        to_gateway_type = None
        if ("[Inclusive Gateway]" in line.split(' -> ')[1] or 
            "[Exclusive Gateway]" in line.split(' -> ')[1] or 
            "[Parallel Gateway]" in line.split(' -> ')[1] or 
            "[Type: CONDITION" in line.split(' -> ')[1]):
            to_gateway_type = (
                "[Inclusive Gateway]" if "[Inclusive Gateway]" in line.split(' -> ')[1] else
                "[Exclusive Gateway]" if ("[Exclusive Gateway]" in line.split(' -> ')[1] or "[Type: CONDITION" in line.split(' -> ')[1]) else
                "[Parallel Gateway]"
            )
        data["ToGatewayType"].append(to_gateway_type)

    # Create and return the DataFrame
    df = pd.DataFrame(data)
    return df

# Not correct, dropping attribute values and the branches do not look correct.
def build_paths(sequence_df):
    # Parse the "FromActivity" and "ToActivity" into connections
    connections = list(zip(sequence_df["FromActivity"], sequence_df["ToActivity"]))

    # Build a mapping from "FromActivity" to "ToActivity"
    adjacency_list = defaultdict(list)
    for from_activity, to_activity in connections:
        adjacency_list[from_activity].append(to_activity)

    # Find the start activity
    start_activity = next((activity for activity in adjacency_list.keys() if "[Type: Start]" in sequence_df.loc[sequence_df["FromActivity"] == activity, "FromActivityType"].values), None)
    if not start_activity:
        raise ValueError("No start activity found in the input DataFrame.")

    # Recursive function to build paths
    def dfs(current_path, sub_path_index, parent_path):
        last_activity = current_path[-1]

        # Record the current path step
        parent_path.append((sub_path_index, last_activity))

        # If the current activity is a stop activity, return
        if "[Type: Stop]" in sequence_df.loc[sequence_df["FromActivity"] == last_activity, "FromActivityType"].values:
            return

        # Explore next activities
        if last_activity in adjacency_list:  # Check for valid transitions
            for next_activity in adjacency_list[last_activity]:
                row = sequence_df[(sequence_df["FromActivity"] == last_activity) & (sequence_df["ToActivity"] == next_activity)].iloc[0]
                gateway_type = row["FromGatewayType"]

                if gateway_type in ["[Parallel Gateway]", "[Inclusive Gateway]", "[Exclusive Gateway]"]:
                    # Create a new sub-path for each branch
                    new_sub_path_index = f"{sub_path_index}.{len(parent_path)}"
                    dfs([next_activity], new_sub_path_index, parent_path)
                else:
                    dfs(current_path + [next_activity], sub_path_index, parent_path)

    # Initialize the path traversal
    parent_path = []
    dfs([start_activity], "0", parent_path)

    # Convert the results to a DataFrame
    path_data = {
        "Path Index": [],
        "SubPath Index": [],
        "Step Number": [],
        "Activity": [],
        "ActivityType": [],
        "GatewayType": [],
        "Performers": [],
        "Performers Rule": [],
        "Number of Performers": [],
        "Probability": []
    }

    for step_number, (sub_path_index, activity) in enumerate(parent_path, start=1):
        # Check if the activity exists in sequence_df
        if activity not in sequence_df["FromActivity"].values:
            # Handle unknown activities
            path_data["Path Index"].append(0)  # Single path
            path_data["SubPath Index"].append(sub_path_index)
            path_data["Step Number"].append(step_number)
            path_data["Activity"].append(activity)
            path_data["ActivityType"].append("[Unknown]")
            path_data["GatewayType"].append(None)
            path_data["Performers"].append(None)
            path_data["Performers Rule"].append(None)
            path_data["Number of Performers"].append(None)
            path_data["Probability"].append(None)
            continue

        row = sequence_df[sequence_df["FromActivity"] == activity].iloc[0]

        path_data["Path Index"].append(0)  # Single path
        path_data["SubPath Index"].append(sub_path_index)
        path_data["Step Number"].append(step_number)
        path_data["Activity"].append(activity)
        path_data["ActivityType"].append(row["FromActivityType"])
        path_data["GatewayType"].append(row["FromGatewayType"])
        path_data["Performers"].append(None)
        path_data["Performers Rule"].append(None)
        path_data["Number of Performers"].append(None)
        path_data["Probability"].append(None)

    # Create and order DataFrame
    df = pd.DataFrame(path_data)
    df = df.sort_values(by=["Path Index", "SubPath Index", "Step Number"]).reset_index(drop=True)

    return df


# Extract start tasks from paths DataFrame
def extract_start_tasks(paths):
    start_tasks = set()
    grouped = paths.groupby("Path Index")  # Group by path index
    for _, group in grouped:
        first_activity = group.iloc[0]["FromActivity"]  # Get the first FromActivity in each path
        first_activity_type = group.iloc[0]["FromActivityType"]  # Get the type of the first activity
        if "[Type: Start]" in first_activity_type:
            activity_name = first_activity.split("[Type:")[0].strip()  # Extract activity name
            start_tasks.add(activity_name)
    return start_tasks

