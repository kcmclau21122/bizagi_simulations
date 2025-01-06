# Bug: Appears not to be calculating processing time correctly, should be able to process more than what is being reported.
# Bizagi: Starts 5,040 Finished 813 = 16%
# This code: Starts 3,960 Finished 127 = 3.2%
import pandas as pd
import heapq
import random
import re
import logging
import os
from datetime import datetime, timedelta
import pandas as pd
import openpyxl

# Importing the parse_xpdl_to_sequences function
from simulation.xpdl_parser import parse_xpdl_to_sequences

# Configure logging
logging.basicConfig(
    filename='simulation_log.txt',
    filemode='w',  # Overwrite the log file
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Event:
    def __init__(self, time, token_id, task_name, event_type):
        self.time = time
        self.token_id = token_id
        self.task_name = task_name
        self.event_type = event_type

    def __lt__(self, other):
        return self.time < other.time

# Read simulation parameters
def get_simulation_parameters(simulation_metrics):
    start_event = simulation_metrics[simulation_metrics['Type'].str.lower() == 'start event']
    if not start_event.empty:
        max_arrival_count = (
            int(start_event['Max arrival count'].iloc[0]) 
            if 'max arrival count' in map(str.lower, start_event.columns) 
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

# Fetch conditional probabilities
def get_condition_probability(simulation_metrics, from_activity, condition_type):
    condition_key = condition_type.split("-")[1].strip()
    row = simulation_metrics.loc[simulation_metrics['Name'] == from_activity]
    if not row.empty and condition_key in row.columns:
        return row.iloc[0][condition_key]
    logging.warning("Probability for condition '%s' not found. Defaulting to 0.5.", condition_type)
    return 0.5  # Default probability

# Read process sequences from file
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

# Check if the current time is within work hours and days
def is_work_time(current_time):
    work_start = datetime.combine(current_time.date(), datetime.min.time()) + timedelta(hours=6)
    work_end = datetime.combine(current_time.date(), datetime.min.time()) + timedelta(hours=12)
    is_work_day = current_time.weekday() < 5  # Monday to Friday
    return is_work_day and work_start <= current_time < work_end

# Advance to the next work time if outside of work hours
def advance_to_work_time(current_time):
    if current_time.weekday() >= 5 or current_time.hour >= 12:  # Weekend or after work hours
        days_to_advance = (7 - current_time.weekday()) % 7 if current_time.weekday() >= 5 else 1
        current_time = datetime.combine(current_time.date(), datetime.min.time()) + timedelta(days=days_to_advance, hours=6)
    elif current_time.hour < 6:  # Before work hours
        current_time = datetime.combine(current_time.date(), datetime.min.time()) + timedelta(hours=6)
    return current_time

# Build paths for the simulation
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

# Discrete event simulation
def discrete_event_simulation(max_arrival_count, arrival_interval_minutes, simulation_days, paths, 
                              simulation_metrics, start_time, xpdl_file_path, transitions_df, start_tasks):

    previous_date = None
    simulation_end_date = start_time + timedelta(days=simulation_days)
    print('Simulation End Date:', simulation_end_date)

    first_path = paths[0][0]
    start_task = first_path.split("->")[0].strip()

    resource_busy_periods = {resource: [] for resource in simulation_metrics['Resource'].dropna().unique()}
    activity_processing_times = {}
    event_queue = []
    active_tokens = {}
    active_resources = {resource: 0 for resource in resource_busy_periods.keys()}
    available_resources = simulation_metrics.set_index('Resource')['Available Resources'].to_dict()

    tokens_started = 0
    tokens_completed = 0

    def get_task_duration(task_name):
        row = simulation_metrics.loc[simulation_metrics['Name'] == task_name]
        if not row.empty and {'Min Time', 'Avg Time', 'Max Time'}.issubset(row.columns):
            min_time = int(row['Min Time'].fillna(1).iloc[0])
            avg_time = int(row['Avg Time'].fillna(min_time).iloc[0])
            max_time = int(row['Max Time'].fillna(avg_time).iloc[0])
            duration = int(random.triangular(min_time, avg_time, max_time))
            logging.info(f"Calculated duration for '{task_name}': Min = {min_time}, Avg = {avg_time}, Max = {max_time}, Chosen = {duration}")
            return duration
        logging.warning(f"Task duration not found for {task_name}. Defaulting to 1 minute.")
        return 1

    current_time = start_time
    token_count = 0

    # Schedule all tokens for independent start tasks
    while token_count < max_arrival_count:
        arrival_time = current_time + timedelta(minutes=arrival_interval_minutes * token_count)
        if arrival_time >= simulation_end_date:
            break
        for start_task in start_tasks:  # Schedule tokens for all start tasks
            heapq.heappush(event_queue, Event(arrival_time, token_count, start_task, 'start'))
            active_tokens[token_count] = {
                'current_task': None,
                'completed_tasks': set(),
                'start_time': arrival_time,
                'paths': paths[:],
                'wait_start_time': None  # Track waiting start time
            }
            logging.info(f"Token {token_count} scheduled to start at {arrival_time} for task '{start_task}'")
        token_count += 1

    # Advance the timer by 1 second if number of days not reached.   
    while current_time < simulation_end_date or event_queue:
        if not event_queue:
            current_time += timedelta(seconds=1)
            continue

        event = heapq.heappop(event_queue)
        current_time = event.time

        if not is_work_time(current_time):
            next_work_time = advance_to_work_time(current_time)
            if next_work_time < simulation_end_date:
                heapq.heappush(event_queue, Event(next_work_time, event.token_id, event.task_name, event.event_type))
            continue

        current_date = current_time.date()
        if current_date != previous_date:
            print(f"Simulation date: {current_date}")
            previous_date = current_date

        token_data = active_tokens[event.token_id]
        if event.event_type == 'start':
            task_name = event.task_name.strip()

            # Prevent duplicate processing for the same task
            if token_data['current_task'] == task_name or task_name in token_data['completed_tasks']:
                continue

            # Increment token start count for start tasks
            tokens_started += 1 if task_name == start_task else 0
            token_data['current_task'] = task_name
            logging.info(f"Token {event.token_id} started task '{task_name}' at {current_time}")

            # Check prerequisites for the task
            prerequisites = transitions_df[transitions_df['To'] == task_name]['From'].values
            if any(prerequisite not in token_data['completed_tasks'] for prerequisite in prerequisites):
                # Reschedule the start event if prerequisites are not met
                heapq.heappush(event_queue, Event(current_time + timedelta(seconds=1), event.token_id, task_name, 'start'))
                continue

            # Get activity type for the current task
            activity_type = transitions_df.loc[transitions_df['From'] == task_name, 'Type'].values
            activity_type = activity_type[0] if activity_type.size > 0 else "Unknown"

            if activity_type == "Gateway":
                # Gateway activities: Only determine token paths
                logging.info(f"Token {event.token_id} passed through gateway '{task_name}' at {current_time}")
                if task_name not in activity_processing_times:
                    activity_processing_times[task_name] = {
                        "durations": [],
                        "wait_times": [],
                        "tokens_started": 0,
                        "tokens_completed": 0,
                    }
                activity_processing_times[task_name]["tokens_started"] += 1
                continue

            # Normal activities: Process tokens, check resources, and track usage
            if task_name not in activity_processing_times:
                activity_processing_times[task_name] = {
                    "durations": [],
                    "wait_times": [],
                    "tokens_started": 0,
                    "tokens_completed": 0,
                }
            activity_processing_times[task_name]["tokens_started"] += 1

            # Check resource availability
            resource_name = simulation_metrics.loc[simulation_metrics['Name'] == task_name, 'Resource'].values
            if resource_name.size > 0 and pd.notna(resource_name[0]):
                resource_name = resource_name[0]
                if active_resources[resource_name] >= available_resources.get(resource_name, 1):
                    # If resource is unavailable, reschedule the task
                    if token_data['wait_start_time'] is None:
                        token_data['wait_start_time'] = current_time
                    heapq.heappush(event_queue, Event(current_time + timedelta(seconds=30), event.token_id, task_name, 'start'))
                    continue

                # If resources are available, allocate them
                if token_data['wait_start_time'] is not None:
                    wait_time = (current_time - token_data['wait_start_time']).total_seconds() / 60
                    activity_processing_times[task_name]["wait_times"].append(wait_time)
                    token_data['wait_start_time'] = None

                active_resources[resource_name] += 1
                resource_busy_periods[resource_name].append([current_time, None])

            # Schedule the task end event
            task_duration = timedelta(minutes=get_task_duration(task_name))
            heapq.heappush(event_queue, Event(current_time + task_duration, event.token_id, task_name, 'end'))


        elif event.event_type == 'end':
            task_name = event.task_name.strip()
            logging.info(f"Token {event.token_id} finished task '{task_name}' at {current_time}")

            if task_name in activity_processing_times:
                activity_processing_times[task_name]["tokens_completed"] += 1
                last_start_time = token_data['start_time']
                activity_processing_times[task_name]["durations"].append((current_time - last_start_time).total_seconds() / 60)

            resource_name = simulation_metrics.loc[simulation_metrics['Name'] == task_name, 'Resource'].values
            if resource_name.size > 0 and pd.notna(resource_name[0]):
                resource_name = resource_name[0]
                active_resources[resource_name] -= 1
                for period in resource_busy_periods[resource_name]:
                    if period[1] is None:
                        period[1] = current_time
                        break

            token_data['completed_tasks'].add(task_name)
            token_data['current_task'] = None

            for path in token_data['paths']:
                for segment in path:
                    if segment.startswith(f"{task_name} ->"):
                        to_task, type_info = re.match(r".+ -> (.+) \[Type: (.+)\]", segment).groups()
                        if type_info.startswith("CONDITION"):
                            probability = get_condition_probability(simulation_metrics, task_name, type_info)
                            if random.random() <= probability:
                                heapq.heappush(event_queue, Event(current_time, event.token_id, to_task, 'start'))
                        elif to_task.lower() != "stop":
                            heapq.heappush(event_queue, Event(current_time, event.token_id, to_task, 'start'))
                        break

    resource_utilization = print_processing_times_and_utilization(
        activity_processing_times,
        resource_busy_periods,
        simulation_end_date,
        start_time,
        available_resources,
        transitions_df
    )

    save_simulation_report(activity_processing_times, resource_utilization, tokens_completed, xpdl_file_path, transitions_df)
    print(f"Tokens started: {tokens_started}, Tokens completed: {tokens_completed}")


# Main function
def main():
    simulation_metrics_path = 'C:/Test Data/Bizagi/simulation_metrics.xlsx'
    xpdl_file_path = 'C:/Test Data/Bizagi/5.5_1/5.5.13 Real Property-Monthly_Reviews/5.5.13 Real Property-Monthly Reviews-org.xpdl'
    output_sequences_path = 'output_sequences.txt'

    # Define work calendar
    simulation_days = 2  # Number of days for the simulation
    start_time = datetime(2025, 1, 5, 0, 0)  # Monday 6:00 AM

    process_sequences = parse_xpdl_to_sequences(xpdl_file_path, output_sequences_path)
    print("Process Sequences:", process_sequences)

    simulation_metrics = pd.read_excel(simulation_metrics_path, sheet_name=0)
    transitions_df = read_output_sequences(output_sequences_path)
    
    # Extract paths and start tasks
    paths = build_paths(transitions_df)
    start_tasks = set(transitions_df[transitions_df['Type'] == 'Start']['From'])
    
    max_arrival_count, arrival_interval_minutes = get_simulation_parameters(simulation_metrics)
    discrete_event_simulation(
        max_arrival_count, arrival_interval_minutes, simulation_days, paths,
        simulation_metrics, start_time, xpdl_file_path, transitions_df, start_tasks
    )

if __name__ == "__main__":
    main()
