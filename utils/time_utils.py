from datetime import datetime, timedelta
from typing import Optional, Union

def day_of_week_to_index(day_name: str) -> int:
    """
    Convert a day of the week name to its index (0-6 where 0 is Monday).
    
    Args:
        day_name: Name of the day of the week (case insensitive)
        
    Returns:
        Index of the day (0-6)
    """
    days = {
        'monday': 0,
        'tuesday': 1,
        'wednesday': 2,
        'thursday': 3,
        'friday': 4,
        'saturday': 5,
        'sunday': 6
    }
    return days.get(day_name.lower(), -1)  # Return -1 for invalid day names

def format_duration(minutes: float) -> str:
    """
    Format a duration in minutes to a more readable format.
    - If >= 60 minutes, show as hours and minutes
    - If >= 24 hours, show as days, hours, minutes and seconds
    
    Args:
        minutes: Duration in minutes
        
    Returns:
        Formatted duration string
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

def format_duration_for_display(minutes: float, include_raw: bool = False) -> str:
    """
    Format duration for display purposes, optionally including the raw value.
    
    Args:
        minutes: Duration in minutes
        include_raw: Whether to include raw minutes in parentheses
        
    Returns:
        Formatted string for display
    """
    if minutes is None:
        return "N/A"
        
    formatted = format_duration(minutes)
    if include_raw and minutes >= 60:
        return f"{formatted} ({minutes:.2f} min)"
    return formatted

def format_datetime_duration(start_time: datetime, end_time: datetime) -> str:
    """
    Format the duration between two datetime objects.
    
    Args:
        start_time: Start time
        end_time: End time
        
    Returns:
        Formatted duration string
    """
    if not isinstance(start_time, datetime) or not isinstance(end_time, datetime):
        return "Invalid datetime"
    
    duration_seconds = (end_time - start_time).total_seconds()
    minutes = duration_seconds / 60
    return format_duration(minutes)

def is_work_time(current_time: datetime, 
                start_time: datetime, 
                work_days: int, 
                work_hours_per_day: int) -> bool:
    """
    Check if the given time falls within designated work hours and workdays.
    
    Args:
        current_time: Time to check
        start_time: Baseline start time (for work hour reference)
        work_days: Number of workdays per week (1-7)
        work_hours_per_day: Number of work hours per day
        
    Returns:
        True if the time is within work hours, False otherwise
    """
    work_start_hour = start_time.hour
    work_end_hour = work_start_hour + work_hours_per_day
    
    # Check if it's a workday (0 = Monday, 6 = Sunday)
    is_work_day = current_time.weekday() < work_days
    
    # Check if it's within work hours
    is_work_hour = work_start_hour <= current_time.hour < work_end_hour
    
    return is_work_day and is_work_hour

def advance_to_work_time(current_time: datetime, 
                       start_time: datetime, 
                       work_days: int, 
                       work_hours_per_day: int) -> datetime:
    """
    Advance the given time to the next available work period if outside work hours.
    
    Args:
        current_time: Time to advance
        start_time: Baseline start time (for work hour reference)
        work_days: Number of workdays per week (1-7)
        work_hours_per_day: Number of work hours per day
        
    Returns:
        The next work time
    """
    work_start_hour = start_time.hour
    work_end_hour = work_start_hour + work_hours_per_day
    
    # If outside work days, move to the next work day
    if current_time.weekday() >= work_days:
        # Calculate days to next work day (Monday)
        days_to_add = (7 - current_time.weekday()) % 7
        if days_to_add == 0:
            days_to_add = 7  # If today is Monday but not a work day, go to next Monday
            
        next_work_day = current_time.date() + timedelta(days=days_to_add)
        return datetime.combine(next_work_day, datetime.min.time()) + timedelta(hours=work_start_hour)
    
    # If after work hours on a work day, move to next work day
    if current_time.hour >= work_end_hour:
        next_day = current_time.date() + timedelta(days=1)
        
        # Check if next day is a work day
        while next_day.weekday() >= work_days:
            next_day += timedelta(days=1)
            
        return datetime.combine(next_day, datetime.min.time()) + timedelta(hours=work_start_hour)
    
    # If before work hours on a work day, move to start of work hours
    if current_time.hour < work_start_hour:
        return datetime.combine(current_time.date(), datetime.min.time()) + timedelta(hours=work_start_hour)
    
    # Already within work hours
    return current_time

def get_next_work_datetime(current_time: datetime, 
                          minutes_to_add: float,
                          start_time: datetime, 
                          work_days: int, 
                          work_hours_per_day: int) -> datetime:
    """
    Calculate the next work datetime by adding working minutes.
    This accounts for non-working hours and days.
    
    Args:
        current_time: Starting time
        minutes_to_add: Minutes of work time to add
        start_time: Baseline start time (for work hour reference)
        work_days: Number of workdays per week (1-7)
        work_hours_per_day: Number of work hours per day
        
    Returns:
        The next work datetime
    """
    # Ensure we start from a valid work time
    if not is_work_time(current_time, start_time, work_days, work_hours_per_day):
        current_time = advance_to_work_time(current_time, start_time, work_days, work_hours_per_day)
    
    work_start_hour = start_time.hour
    work_end_hour = work_start_hour + work_hours_per_day
    
    # Convert minutes to timedelta
    remaining_minutes = minutes_to_add
    next_time = current_time
    
    while remaining_minutes > 0:
        # Calculate minutes until end of current work day
        minutes_until_end_of_day = (work_end_hour - next_time.hour) * 60 - next_time.minute
        
        if remaining_minutes <= minutes_until_end_of_day:
            # If we can fit within the current work day
            next_time += timedelta(minutes=remaining_minutes)
            remaining_minutes = 0
        else:
            # Move to end of current day
            next_time = datetime.combine(next_time.date(), datetime.min.time()) + timedelta(hours=work_end_hour)
            remaining_minutes -= minutes_until_end_of_day
            
            # Advance to next work day
            next_day = next_time.date() + timedelta(days=1)
            while next_day.weekday() >= work_days:
                next_day += timedelta(days=1)
                
            next_time = datetime.combine(next_day, datetime.min.time()) + timedelta(hours=work_start_hour)
    
    return next_time

def get_business_days_between(start_date: datetime, 
                            end_date: datetime, 
                            work_days: int) -> int:
    """
    Calculate the number of business days between two dates.
    
    Args:
        start_date: Starting date
        end_date: Ending date
        work_days: Number of workdays per week (1-7)
        
    Returns:
        Number of business days between the dates
    """
    if start_date > end_date:
        return 0
        
    # Convert to date objects if they're datetimes
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()
        
    business_days = 0
    current_date = start_date
    
    while current_date <= end_date:
        if current_date.weekday() < work_days:
            business_days += 1
        current_date += timedelta(days=1)
        
    return business_days

def get_working_minutes_between(start_time: datetime, 
                              end_time: datetime,
                              base_start_time: datetime, 
                              work_days: int, 
                              work_hours_per_day: int) -> float:
    """
    Calculate the number of working minutes between two datetimes.
    
    Args:
        start_time: Starting time
        end_time: Ending time
        base_start_time: Baseline start time (for work hour reference)
        work_days: Number of workdays per week (1-7)
        work_hours_per_day: Number of work hours per day
        
    Returns:
        Number of working minutes between the times
    """
    if start_time > end_time:
        return 0
        
    work_start_hour = base_start_time.hour
    work_end_hour = work_start_hour + work_hours_per_day
    
    # Ensure both times are within work hours
    if not is_work_time(start_time, base_start_time, work_days, work_hours_per_day):
        start_time = advance_to_work_time(start_time, base_start_time, work_days, work_hours_per_day)
    
    if not is_work_time(end_time, base_start_time, work_days, work_hours_per_day):
        # Find the last work time before end_time
        temp_time = end_time
        while not is_work_time(temp_time, base_start_time, work_days, work_hours_per_day):
            temp_time -= timedelta(minutes=1)
            if temp_time < start_time:
                return 0
        end_time = temp_time
    
    # If they're on the same day
    if start_time.date() == end_time.date():
        return (end_time - start_time).total_seconds() / 60
    
    # Calculate working minutes
    working_minutes = 0
    
    # Add minutes from start_time to end of its work day
    minutes_until_end_of_day = (work_end_hour - start_time.hour) * 60 - start_time.minute
    working_minutes += minutes_until_end_of_day
    
    # Add full work days in between
    current_date = start_time.date() + timedelta(days=1)
    while current_date < end_time.date():
        if current_date.weekday() < work_days:
            working_minutes += work_hours_per_day * 60
        current_date += timedelta(days=1)
    
    # Add minutes from start of end_time's work day to end_time
    if end_time.date() > start_time.date():
        working_minutes += (end_time.hour - work_start_hour) * 60 + end_time.minute
    
    return working_minutes
