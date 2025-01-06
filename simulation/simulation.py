# Logic the runs the simulation
from datetime import timedelta
import heapq
from simulation.utils import is_work_time, advance_to_work_time, get_task_duration, get_condition_probability
from simulation.reporting import print_processing_times_and_utilization
from simulation.data_handler import save_simulation_report
from simulation.event import Event
import logging
import random
import pandas as pd

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
