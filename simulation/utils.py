from datetime import datetime, timedelta
import logging
import random
import pandas as pd
import heapq

# Time formatting functions
def format_duration(minutes):
    """
    Format a duration in minutes to a more readable format.
    - If >= 60 minutes, show as hours and minutes
    - If >= 24 hours, show as days, hours, minutes and seconds
    
    Args:
        minutes (float): Duration in minutes
        
    Returns:
        str: Formatted duration string
    """
    if minutes is None or minutes == 0:
        return "0m 0s"
        
    total_seconds = int(minutes * 60)
    seconds = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes_part = total_minutes % 60
    total_hours = total_minutes // 60
    
    if total_hours >= 24:
        # Format as days, hours, minutes, seconds
        days_part = total_hours // 24
        hours_part = total_hours % 24
        return f"{days_part}d {hours_part}h {minutes_part}m {seconds}s"
    elif total_hours > 0:
        # Format as hours, minutes, seconds
        return f"{total_hours}h {minutes_part}m {seconds}s"
    else:
        # Format as minutes, seconds
        return f"{minutes_part}m {seconds}s"

def format_duration_for_display(minutes, include_raw=False):
    """
    Format duration for display purposes, optionally including the raw value.
    
    Args:
        minutes (float): Duration in minutes
        include_raw (bool): Whether to include raw minutes in parentheses
        
    Returns:
        str: Formatted string for display
    """
    if minutes is None:
        return "N/A"
        
    formatted = format_duration(minutes)
    if include_raw and minutes >= 60:
        return f"{formatted} ({minutes:.2f} min)"
    return formatted

# Read simulation parameters
def get_simulation_parameters(simulation_metrics):
    # Convert column names to lowercase for consistent access
    simulation_metrics.columns = map(str.lower, simulation_metrics.columns)

    # Log the entire simulation_metrics DataFrame to the log file
    logging.info("Logging simulation metrics:")
    logging.info(simulation_metrics.to_string(index=False))

    # Filter the DataFrame for rows where 'type' equals 'start' (case-insensitively)
    start_event = simulation_metrics[simulation_metrics['type'].str.lower() == 'start']
    
    if not start_event.empty:
        # Check for the 'max arrival count' and 'arrival interval' columns and fetch their values
        max_arrival_count = (
            int(start_event['max arrival count'].iloc[0]) 
            if 'max arrival count' in simulation_metrics.columns 
            else 0
        )
        arrival_interval_minutes = (
            int(start_event['arrival interval'].iloc[0]) 
            if 'arrival interval' in simulation_metrics.columns 
            else 0
        )
        return max_arrival_count, arrival_interval_minutes

    logging.warning("Start event parameters not found. Using default values.")
    return 0, 0  # Default values if not found

# Helper function to advance simulation time in 1-second intervals
def advance_time_in_seconds(current_time, event_queue, active_resources, resource_wait_queue, active_tokens, resource_busy_periods):
    while current_time.second != 0:
        current_time += timedelta(seconds=1)
        # Check and assign resources based on FIFO
        for resource_name, queue in resource_wait_queue.items():
            if queue and active_resources[resource_name] < len(resource_busy_periods[resource_name]):
                token_id, task_name = queue.pop(0)  # FIFO queue
                
                # Inline logic for assigning resource
                if active_resources[resource_name] < len(resource_busy_periods[resource_name]):
                    active_resources[resource_name] += 1
                
                start_event_time = current_time
                heapq.heappush(event_queue, (start_event_time, token_id, task_name, 'start'))
                active_tokens[token_id]['wait_start_time'] = None

# Fetch conditional probabilities using triangular distribution
def get_condition_probability(simulation_metrics, from_activity, condition_type):
    """
    Get probability for a condition path using triangular distribution if min/avg/max are available.
    
    Args:
        simulation_metrics: DataFrame with simulation metrics
        from_activity: Source activity name
        condition_type: Condition type string
        
    Returns:
        float: Probability value
    """
    # Normalize column names to lowercase for consistent access
    simulation_metrics.columns = map(str.lower, simulation_metrics.columns)
    
    # Convert the input strings to lowercase for comparison
    condition_key = condition_type.split("-")[1].strip().lower()
    row = simulation_metrics.loc[simulation_metrics['name'].str.lower() == from_activity.lower()]
    
    # Check if we have min/most likely/max values for triangular distribution
    min_col = f"min {condition_key}"
    avg_col = f"avg {condition_key}"
    max_col = f"max {condition_key}"
    
    if not row.empty and min_col in simulation_metrics.columns and avg_col in simulation_metrics.columns and max_col in simulation_metrics.columns:
        min_val = row.iloc[0][min_col]
        avg_val = row.iloc[0][avg_col]
        max_val = row.iloc[0][max_col]
        
        # Use triangular distribution if all values are available
        if pd.notna(min_val) and pd.notna(avg_val) and pd.notna(max_val):
            return random.triangular(min_val, avg_val, max_val)
    
    # Fall back to direct probability if triangular distribution parameters are not available
    if not row.empty and condition_key in simulation_metrics.columns:
        direct_prob = row.iloc[0][condition_key]
        if pd.notna(direct_prob):
            return direct_prob
    
    logging.warning(f"Probability for condition '{condition_type}' not found. Defaulting to 0.5.")
    return 0.5  # Default probability

# Check if the current time is within work hours and days
def is_work_time(current_time, start_time, number_workdays, number_work_hours_per_day):
    """
    Check if the given time falls within designated work hours and workdays.
    """
    work_start_hour = start_time.hour
    work_end_hour = work_start_hour + number_work_hours_per_day
    is_work_day = current_time.weekday() < number_workdays  # Work is allowed for the specified number of days
    return is_work_day and work_start_hour <= current_time.hour < work_end_hour

def advance_to_work_time(current_time, start_time, number_workdays, number_work_hours_per_day):
    """
    Advance the given time to the next available work period if outside work hours.
    """
    work_start_hour = start_time.hour
    work_end_hour = work_start_hour + number_work_hours_per_day
    
    # If outside work hours, move to the next work period
    if current_time.weekday() >= number_workdays or current_time.hour >= work_end_hour:
        days_to_advance = (7 - current_time.weekday()) % 7 if current_time.weekday() >= number_workdays else 1
        current_time = datetime.combine(current_time.date() + timedelta(days=days_to_advance),
                                        datetime.min.time()) + timedelta(hours=work_start_hour)
    elif current_time.hour < work_start_hour:  # Before work hours
        current_time = datetime.combine(current_time.date(), datetime.min.time()) + timedelta(hours=work_start_hour)
    
    return current_time

def choose_node(source_node, links):
    """
    Determine the next node(s) based on the gateway type of the source node,
    using triangular distribution for probabilistic selection when appropriate.
    
    Args:
        source_node: Source node data
        links: List of links in the process model
        
    Returns:
        list: List of target node IDs to proceed to
    """
    gateway = source_node.get("gateway", None)
    nodetype = source_node.get("type", None)
    source_id = source_node["id"]

    if gateway == "[Parallel Gateway]":
        # Return all target nodes linked to the source
        return [link["target"] for link in links if link["source"] == source_id]

    elif gateway == "[Exclusive Gateway]" and "CONDITION-" not in nodetype:
        # Enhanced implementation with triangular distribution
        outgoing_links = [link for link in links if link["source"] == source_id]
        
        # Extract condition types from outgoing links
        condition_types = {}
        for link in outgoing_links:
            if "type" in link and "CONDITION-" in link["type"]:
                condition = link["type"].split("CONDITION-")[1].strip()
                condition_types[condition.lower()] = link["target"]
        
        # If we have condition types, determine which one to follow based on probabilities
        if condition_types:
            # Get all probability-related attributes from the source node
            probabilities = {}
            min_probs = {}
            avg_probs = {}
            max_probs = {}
            
            for key, value in source_node.items():
                if isinstance(value, (int, float)) and value <= 1 and value >= 0:
                    key_lower = key.lower()
                    
                    if key_lower in condition_types:
                        probabilities[key_lower] = value
                    elif key_lower.startswith("min ") and key_lower[4:] in condition_types:
                        min_probs[key_lower[4:]] = value
                    elif key_lower.startswith("avg ") and key_lower[4:] in condition_types:
                        avg_probs[key_lower[4:]] = value
                    elif key_lower.startswith("max ") and key_lower[4:] in condition_types:
                        max_probs[key_lower[4:]] = value
            
            # Use triangular distribution if we have min/avg/max values
            triangular_probs = {}
            
            for condition in condition_types.keys():
                if condition in min_probs and condition in avg_probs and condition in max_probs:
                    triangular_probs[condition] = random.triangular(
                        min_probs[condition], 
                        avg_probs[condition], 
                        max_probs[condition]
                    )
            
            # Normalize triangular probabilities if we have any
            if triangular_probs:
                total = sum(triangular_probs.values())
                if total > 0:
                    for condition in triangular_probs:
                        triangular_probs[condition] /= total
                    
                    # Choose based on normalized triangular probabilities
                    choice = random.choices(
                        population=list(triangular_probs.keys()),
                        weights=list(triangular_probs.values()),
                        k=1
                    )[0]
                    
                    return [condition_types[choice]]
            
            # Fall back to direct probabilities if triangular not available
            if probabilities:
                total = sum(probabilities.values())
                if abs(total - 1.0) > 0.001:  # If total is not close to 1, normalize
                    for condition in probabilities:
                        probabilities[condition] /= total
                
                # Choose based on probabilities
                choice = random.choices(
                    population=list(probabilities.keys()),
                    weights=list(probabilities.values()),
                    k=1
                )[0]
                
                return [condition_types[choice]]
        
        # If no probabilities found, pick randomly
        outgoing_targets = [link["target"] for link in outgoing_links]
        if outgoing_targets:
            return [random.choice(outgoing_targets)]

    elif gateway == "[Inclusive Gateway]":
        # Enhanced implementation for inclusive gateway
        outgoing_links = [link for link in links if link["source"] == source_id]
        targets = []
        
        # For each outgoing link, determine whether to include it based on probability
        for link in outgoing_links:
            condition_type = link.get("type", "")
            if "CONDITION-" in condition_type:
                condition = condition_type.split("CONDITION-")[1].strip()
                
                # Try to get probability from source node attributes
                probability = source_node.get(condition.lower(), 0.5)
                
                # Check if we should include this path
                if random.random() <= probability:
                    targets.append(link["target"])
        
        # If no targets selected, pick one randomly as fallback
        if not targets and outgoing_links:
            targets = [random.choice([link["target"] for link in outgoing_links])]
        
        return targets

    # Default: no gateway logic, find direct target(s) from links
    targets = [link["target"] for link in links if link["source"] == source_id]
    return targets

# Format time values in a dictionary recursively
def format_time_values_in_dict(data_dict, time_keys=None):
    """
    Format time values in a dictionary recursively.
    
    Args:
        data_dict (dict): Dictionary containing time values to format
        time_keys (list): List of keys that should be treated as time values
        
    Returns:
        dict: Dictionary with formatted time values
    """
    if time_keys is None:
        # Default keys to look for time values
        time_keys = ["time", "duration", "wait", "processing"]
    
    result = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):
            result[key] = format_time_values_in_dict(value, time_keys)
        elif isinstance(value, list):
            result[key] = [
                format_time_values_in_dict(item, time_keys) if isinstance(item, dict) else item 
                for item in value
            ]
        elif isinstance(value, (int, float)) and any(time_word in key.lower() for time_word in time_keys):
            # This is a time/duration value that needs formatting
            if value >= 60:  # Only format if >= 60 minutes
                result[key] = value  # Keep original numeric value
                result[f"{key}_formatted"] = format_duration(value)  # Add formatted version
            else:
                result[key] = value
        else:
            result[key] = value
    
    return result

# Convert a duration between datetimes to a formatted string
def format_datetime_duration(start_time, end_time):
    """
    Format the duration between two datetime objects
    
    Args:
        start_time (datetime): Start time
        end_time (datetime): End time
        
    Returns:
        str: Formatted duration string
    """
    if not isinstance(start_time, datetime) or not isinstance(end_time, datetime):
        return "Invalid datetime"
    
    duration_seconds = (end_time - start_time).total_seconds()
    minutes = duration_seconds / 60
    return format_duration(minutes)
