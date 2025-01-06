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


def save_simulation_report(activity_processing_times, resource_utilization, tokens_completed, xpdl_file_path, transitions_df):
    import os
    base_filename = os.path.splitext(os.path.basename(xpdl_file_path))[0]
    output_path = f"{base_filename}_results.xlsx"

    # Initialize variables for the process summary
    process_summary = {
        "Tokens Started": 0,
        "Tokens Completed": 0,
        "Min Time (min)": None,
        "Max Time (min)": None,
        "Avg Time (min)": None
    }

    # Prepare activity processing times data
    activity_data = []
    for activity, data in activity_processing_times.items():
        durations = data.get("durations", [])
        wait_times = data.get("wait_times", [])
        logging.info(f"Activity '{activity}' durations: {durations}, wait times: {wait_times}")
        tokens_started = data.get("tokens_started", 0)
        tokens_completed = data.get("tokens_completed", 0)

        # Extract activity type
        activity_type = transitions_df.loc[transitions_df['From'] == activity, 'Type'].values
        activity_type = activity_type[0] if activity_type.size > 0 else "Unknown"

        # Update for "CONDITION" to "Gateway"
        if "CONDITION" in activity_type.upper():
            activity_type = "Gateway"

        if activity_type == "Gateway":
            # Skip processing time and wait time calculations for gateways
            activity_data.append({
                "Activity": activity,
                "Activity Type": activity_type,
                "Tokens Started": round(tokens_started, 2),
                "Tokens Completed": round(tokens_completed, 2),
                "Min Time (min)": "",
                "Max Time (min)": "",
                "Avg Time (min)": "",
                "Total Time Waiting for Resources (min)": "",
                "Min Time Waiting for Resources (min)": "",
                "Max Time Waiting for Resources (min)": "",
                "Avg Time Waiting for Resources (min)": "",
            })
            continue

        if activity_type == "Start":
            # Only report tokens started for "Start" activity type
            activity_data.append({
                "Activity": activity,
                "Activity Type": activity_type,
                "Tokens Started": round(tokens_started, 2),
                "Tokens Completed": "",
                "Min Time (min)": "",
                "Max Time (min)": "",
                "Avg Time (min)": "",
                "Total Time Waiting for Resources (min)": "",
                "Min Time Waiting for Resources (min)": "",
                "Max Time Waiting for Resources (min)": "",
                "Avg Time Waiting for Resources (min)": "",
            })
            process_summary["Tokens Started"] += tokens_started
        else:
            # For normal activities
            if durations:
                min_time = round(min(durations), 2)
                max_time = round(max(durations), 2)
                avg_time = round(sum(durations) / len(durations), 2)
                process_summary["Min Time (min)"] = min(process_summary["Min Time (min)"], min_time) if process_summary["Min Time (min)"] else min_time
                process_summary["Max Time (min)"] = max(process_summary["Max Time (min)"], max_time) if process_summary["Max Time (min)"] else max_time
                process_summary["Avg Time (min)"] = round((process_summary["Avg Time (min)"] + avg_time) / 2, 2) if process_summary["Avg Time (min)"] else avg_time

            if activity_type == "Stop":
                process_summary["Tokens Completed"] += tokens_completed

            total_wait_time = round(sum(wait_times), 2) if wait_times else ""
            min_wait_time = round(min(wait_times), 2) if wait_times else ""
            max_wait_time = round(max(wait_times), 2) if wait_times else ""
            avg_wait_time = round(sum(wait_times) / len(wait_times), 2) if wait_times else ""

            activity_data.append({
                "Activity": activity,
                "Activity Type": activity_type,
                "Tokens Started": round(tokens_started, 2),
                "Tokens Completed": round(tokens_completed, 2),
                "Min Time (min)": min_time if durations else "",
                "Max Time (min)": max_time if durations else "",
                "Avg Time (min)": avg_time if durations else "",
                "Total Time Waiting for Resources (min)": total_wait_time,
                "Min Time Waiting for Resources (min)": min_wait_time,
                "Max Time Waiting for Resources (min)": max_wait_time,
                "Avg Time Waiting for Resources (min)": avg_wait_time,
            })

    # Add the process summary as the first row
    activity_data.insert(0, {
        "Activity": base_filename,
        "Activity Type": "Process",
        "Tokens Started": round(process_summary["Tokens Started"], 2),
        "Tokens Completed": round(process_summary["Tokens Completed"], 2),
        "Min Time (min)": round(process_summary["Min Time (min)"], 2) if process_summary["Min Time (min)"] else "",
        "Max Time (min)": round(process_summary["Max Time (min)"], 2) if process_summary["Max Time (min)"] else "",
        "Avg Time (min)": round(process_summary["Avg Time (min)"], 2) if process_summary["Avg Time (min)"] else "",
        "Total Time Waiting for Resources (min)": "",
        "Min Time Waiting for Resources (min)": "",
        "Max Time Waiting for Resources (min)": "",
        "Avg Time Waiting for Resources (min)": "",
    })

    # Convert to DataFrame and save to Excel
    activity_df = pd.DataFrame(activity_data)
    resource_data = [{"Resource": res, "Utilization (%)": round(util, 2)} for res, util in resource_utilization.items()]
    resource_df = pd.DataFrame(resource_data)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        resource_df.to_excel(writer, index=False, sheet_name="Resource Utilization")
        activity_df.to_excel(writer, index=False, sheet_name="Activity Times")
    print(f"Simulation report saved to {output_path}")

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

