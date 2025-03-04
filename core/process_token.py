
from datetime import datetime
from typing import List, Optional, Dict, Any

class Token:
    """
    Represents a token (process instance) moving through the process model during simulation.
    Tracks the path, waiting times, and overall state of a process instance.
    """
    
    def __init__(self, token_id: str, start_time: datetime, start_node: str):
        """Initialize a new token with start information."""
        self.token_id = token_id
        self.start_time = start_time
        self.current_node = start_node
        self.wait_start_time: Optional[datetime] = None
        self.total_wait_time = 0.0  # in minutes
        self.completed_tasks: List[str] = []
        self.path: List[str] = [start_node]
        self.end_time: Optional[datetime] = None
        
    def add_to_path(self, node_id: str) -> None:
        """Add a node to the path traversed by this token."""
        self.path.append(node_id)
        self.current_node = node_id
        
    def start_waiting(self, time: datetime) -> None:
        """Mark the token as waiting for a resource."""
        self.wait_start_time = time
        
    def stop_waiting(self, time: datetime) -> float:
        """
        Stop waiting and calculate the wait time.
        Returns the wait duration in minutes.
        """
        if self.wait_start_time:
            wait_duration = (time - self.wait_start_time).total_seconds() / 60
            self.total_wait_time += wait_duration
            self.wait_start_time = None
            return wait_duration
        return 0.0
        
    def complete_task(self, task_name: str) -> None:
        """Mark a task as completed by this token."""
        self.completed_tasks.append(task_name)
        
    def complete(self, time: datetime) -> None:
        """Mark the token as completed."""
        self.end_time = time
        
    def is_completed(self) -> bool:
        """Check if the token has completed its execution."""
        return self.end_time is not None
        
    def get_process_duration(self) -> Optional[float]:
        """Get the total process duration in minutes."""
        if self.is_completed():
            return (self.end_time - self.start_time).total_seconds() / 60
        return None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the token to a dictionary for reporting."""
        result = {
            "token_id": self.token_id,
            "start_time": self.start_time,
            "current_task": self.current_node,
            "total_wait_time": self.total_wait_time,
            "path": self.path.copy()
        }
        
        if self.is_completed():
            result["end_time"] = self.end_time
            
        return result