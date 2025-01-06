# simulation/data_handler.py
import pandas as pd
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
            transitions.append({'From': from_task, 'To': to_task, 'Type': type_info})

    transitions_df = pd.DataFrame(transitions)

    # Identify "Start" and "Stop" activities
    all_from_tasks = set(transitions_df['From'])
    all_to_tasks = set(transitions_df['To'])

    # Start activities: appear in 'From' but not in 'To'
    start_activities = all_from_tasks - all_to_tasks
    transitions_df.loc[transitions_df['From'].isin(start_activities), 'Type'] = 'Start'

    # Stop activities: appear in 'To' but not in 'From'
    stop_activities = all_to_tasks - all_from_tasks
    transitions_df.loc[transitions_df['To'].isin(stop_activities), 'Type'] = 'Stop'

    return transitions_df

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

