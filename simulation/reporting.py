import logging
import datetime

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
