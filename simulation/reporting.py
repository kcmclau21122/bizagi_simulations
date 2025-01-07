import logging
import datetime
import os
import pandas as pd

def save_simulation_report(activity_processing_times, resource_utilization, active_tokens, xpdl_file_path, transitions_df, completed_tokens):
    base_filename = os.path.splitext(os.path.basename(xpdl_file_path))[0]
    output_path = f"{base_filename}_results.xlsx"

    # Calculate total started tokens and process-level metrics
    if completed_tokens:
        process_durations = [
            (token['end_time'] - token['start_time']).total_seconds() / 60 for token in completed_tokens
        ]
        process_wait_times = [token['total_wait_time'] for token in completed_tokens]

        min_time = round(min(process_durations), 2)
        max_time = round(max(process_durations), 2)
        avg_time = round(sum(process_durations) / len(process_durations), 2)
        total_wait_time = round(sum(process_wait_times), 2)
        min_wait_time = round(min(process_wait_times), 2)
        max_wait_time = round(max(process_wait_times), 2)
        avg_wait_time = round(sum(process_wait_times) / len(process_wait_times), 2)
    else:
        min_time = max_time = avg_time = total_wait_time = min_wait_time = max_wait_time = avg_wait_time = 0

    # Create the process-level summary row
    process_row = {
        "Activity": base_filename,
        "Activity Type": "Process",
        "Tokens Started": active_tokens,
        "Tokens Completed": len(completed_tokens),
        "Min Time (min)": min_time,
        "Max Time (min)": max_time,
        "Avg Time (min)": avg_time,
        "Total Time Waiting for Resources (min)": total_wait_time,
        "Min Time Waiting for Resources (min)": min_wait_time,
        "Max Time Waiting for Resources (min)": max_wait_time,
        "Avg Time Waiting for Resources (min)": avg_wait_time,
    }

    # Insert the process row as the first row in the activity data
    activity_data = [process_row]

    # Continue processing individual activity data...
    for activity, data in activity_processing_times.items():
        durations = data.get("durations", [])
        wait_times = data.get("wait_times", [])
        tokens_started = data.get("tokens_started", 0)
        tokens_completed = data.get("tokens_completed", 0)

        activity_type = transitions_df.loc[transitions_df['From'] == activity, 'Type'].values
        activity_type = activity_type[0] if activity_type.size > 0 else "Unknown"

        if "CONDITION" in activity_type.upper():
            activity_type = "Gateway"

        min_time = round(min(durations), 2) if durations else 0
        max_time = round(max(durations), 2) if durations else 0
        avg_time = round(sum(durations) / len(durations), 2) if durations else 0
        total_wait_time = round(sum(wait_times), 2) if wait_times else 0
        min_wait_time = round(min(wait_times), 2) if wait_times else 0
        max_wait_time = round(max(wait_times), 2) if wait_times else 0
        avg_wait_time = round(sum(wait_times) / len(wait_times), 2) if wait_times else 0

        activity_data.append({
            "Activity": activity,
            "Activity Type": activity_type,
            "Tokens Started": tokens_started,
            "Tokens Completed": tokens_completed,
            "Min Time (min)": min_time,
            "Max Time (min)": max_time,
            "Avg Time (min)": avg_time,
            "Total Time Waiting for Resources (min)": total_wait_time,
            "Min Time Waiting for Resources (min)": min_wait_time,
            "Max Time Waiting for Resources (min)": max_wait_time,
            "Avg Time Waiting for Resources (min)": avg_wait_time,
        })

    # Save data to Excel
    activity_df = pd.DataFrame(activity_data)
    resource_df = pd.DataFrame([
        {"Resource": res, "Utilization (%)": round(util, 2)} for res, util in resource_utilization.items()
    ])

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
