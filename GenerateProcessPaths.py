import pandas as pd

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

# Step 3: Create DataFrames for each path
def create_path_dataframes(paths):
    dfs = []
    for path_index, path in enumerate(paths):
        dfs.append(pd.DataFrame({'Path': path}))
    return dfs

# Step 4: Main function to process and display results
def main():
    file_path = 'output_sequences.txt'  # Update with the correct file path
    df = read_output_sequences(file_path)
    paths = build_paths(df)
    path_dataframes = create_path_dataframes(paths)
    
    # Print each DataFrame
    for i, path_df in enumerate(path_dataframes):
        print(f"Path {i + 1}:\n", path_df.to_string(index=False), "\n")

# Run the script
main()
