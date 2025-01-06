import logging
import datetime
import pandas as pd

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

def print_processing_times_and_utilization(activity_processing_times, resource_busy_periods, simulation_end_date, start_time, available_resources, transitions_df):
    total_simulation_time = max((simulation_end_date - start_time).total_seconds() / 3600, 0)
    resource_utilization = {}

    # Log structure of activity_processing_times for debugging
    logging.info(f"Activity processing times structure: {activity_processing_times}")

    # Print activity processing times
    print("\nActivity Processing Times:")
    for activity, data in activity_processing_times.items():
        times = data.get("durations", [])
        valid_durations = []
        for duration in times:
            if isinstance(duration, tuple) and len(duration) == 2:
                start, end = duration
                if isinstance(start, datetime) and isinstance(end, datetime):
                    valid_durations.append((end - start).total_seconds() / 60)
            elif isinstance(duration, (int, float)):
                valid_durations.append(duration)  # Handle direct durations if present

        if valid_durations:
            min_time = min(valid_durations)
            avg_time = sum(valid_durations) / len(valid_durations)
            max_time = max(valid_durations)
            print(f"Activity '{activity}': Min = {min_time:.2f} min, Avg = {avg_time:.2f} min, Max = {max_time:.2f} min")
        else:
            print(f"Activity '{activity}': No valid durations.")

    # Calculate and print resource utilization
    print("\nResource Utilization:")
    for resource, periods in resource_busy_periods.items():
        # Get the activity type for this resource
        is_gateway = False
        for activity, data in activity_processing_times.items():
            activity_type = transitions_df.loc[transitions_df['From'] == activity, 'Type'].values
            activity_type = activity_type[0] if activity_type.size > 0 else "Unknown"
            if activity_type == "Gateway":
                is_gateway = True
                break

        if is_gateway:
            continue

        total_busy_time = sum(
            (end - start).total_seconds() for start, end in periods if start and end
        )
        num_resources = available_resources.get(resource, 1)
        utilization = (
            (total_busy_time / (total_simulation_time * 3600 * num_resources)) * 100
            if total_simulation_time > 0 else 0
        )
        resource_utilization[resource] = min(utilization, 100)

        print(f"Resource '{resource}': Utilization = {resource_utilization[resource]:.2f}%")
        logging.info(f"Resource '{resource}' utilization: {resource_utilization[resource]:.2f}%")

    return resource_utilization
