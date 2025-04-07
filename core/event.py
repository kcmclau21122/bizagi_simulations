from datetime import datetime
from typing import Optional

class Event:
    """
    Represents a simulation event with timing and token information.
    """
    
    def __init__(self, time: datetime, token_id: str, task_name: str, event_type: str):
        """
        Initialize a new event.
        
        Args:
            time: Event time
            token_id: ID of the token this event relates to
            task_name: Name of the task/node
            event_type: Type of event (start, end, etc.)
        """
        self.time = time
        self.token_id = token_id
        self.task_name = task_name
        self.node_id = task_name  # Add this line to create an alias
        self.event_type = event_type

    def __lt__(self, other: 'Event') -> bool:
        """
        Compare events for priority queue ordering.
        Events are ordered primarily by time, then by event_type.
        
        Args:
            other: Other event to compare with
            
        Returns:
            True if this event should be processed before the other
        """
        if self.time != other.time:
            return self.time < other.time
        
        # If times are equal, process 'end' events before 'start' events
        # This ensures resources are released before being allocated again
        if self.event_type != other.event_type:
            return self.event_type == "end"
            
        # If both are the same type, just use token ID for stable ordering
        return self.token_id < other.token_id
    
    def __str__(self) -> str:
        """
        Convert event to string for debugging.
        
        Returns:
            String representation of the event
        """
        return f"Event({self.event_type}, {self.token_id}, {self.node_id}, {self.time})"
