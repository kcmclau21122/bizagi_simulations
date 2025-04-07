from datetime import datetime
from typing import List, Optional, Dict, Any, Set

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
        
        # Gateway synchronization tracking
        self.parent_token_id: Optional[str] = None  # ID of the parent token if this is a split token
        self.split_tokens: Dict[str, bool] = {}  # Maps split token IDs to completion status
        self.active_splits: int = 0  # Count of active split paths
        self.is_split_token: bool = False  # Whether this token is a split from a parent token
        self.active_gateway: Optional[str] = None  # Current active gateway for this token
        self.merge_node: Optional[str] = None  # Node where split tokens should synchronize
        self.completed_branches: Set[str] = set()  # Set of completed branch paths
        
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
        
    def mark_split_complete(self, split_token_id: str, branch_id: str) -> bool:
        """
        Mark a split token as completed when it reaches the merge node.
        Enhanced for proper inclusive gateway support.
        
        Args:
            split_token_id: ID of the completed split token
            branch_id: ID of the completed branch/node
            
        Returns:
            True if all active splits are now complete, False otherwise
        """
        if split_token_id in self.split_tokens:
            # Mark this token as complete
            self.split_tokens[split_token_id] = True
            
            # Add the branch to completed branches
            self.completed_branches.add(branch_id)
            
            # Decrement active splits count
            self.active_splits -= 1
            
            # For inclusive gateways, we need to check if all active paths have completed
            # rather than requiring all possible paths to complete
            all_token_splits_complete = all(self.split_tokens.values())
            
            # For standard parallel gateway merges, require all tokens to complete
            if all_token_splits_complete and self.active_splits <= 0:
                return True
            
            # For inclusive gateways, it's more complex - we need to determine 
            # if we've received all tokens that were actually sent down active paths
            # This is based on the tokens we created during the split
            return all_token_splits_complete and self.active_splits <= 0
        
        # If token not found, something is wrong - return False
        return False

    def create_split_token(self, split_id: str, time: datetime, node_id: str) -> 'Token':
        """
        Create a new token split from this token.
        Enhanced to better track inclusive gateway paths.
        
        Args:
            split_id: ID for the new split token
            time: Current simulation time
            node_id: Node ID where the split is occurring
            
        Returns:
            A new Token instance representing the split
        """
        # Create a new token with the provided ID and starting point
        split_token = Token(split_id, time, node_id)
        
        # Mark this as a split token and set its parent
        split_token.parent_token_id = self.token_id
        split_token.is_split_token = True
        
        # Carry forward the wait time and path history
        split_token.total_wait_time = self.total_wait_time
        
        # Copy the path history up to this point (instead of just the current node)
        split_token.path = self.path.copy()
        
        # Record merge node information from parent if available
        split_token.merge_node = self.merge_node
        split_token.active_gateway = self.active_gateway
        
        # Track this split in the parent (mark as not completed yet)
        self.split_tokens[split_id] = False
        self.active_splits += 1
        
        return split_token

    def merge_split_token_data(self, split_token: 'Token') -> None:
        """
        Merge data from a split token back into the parent token.
        Used when synchronizing at merge points.
        
        Args:
            split_token: The split token to merge data from
        """
        # Merge wait time (take the maximum)
        self.total_wait_time = max(self.total_wait_time, split_token.total_wait_time)
        
        # Merge completed tasks (add any tasks completed by the split token)
        for task in split_token.completed_tasks:
            if task not in self.completed_tasks:
                self.completed_tasks.append(task)
        
        # Note: We don't merge paths as that would create divergent history
        # The parent token's path will be updated separately to continue from merge node

    def get_all_split_token_ids(self) -> List[str]:
        """
        Get all split token IDs associated with this token.
        
        Returns:
            List of split token IDs
        """
        return list(self.split_tokens.keys())

    def get_incomplete_split_token_ids(self) -> List[str]:
        """
        Get IDs of all incomplete split tokens.
        
        Returns:
            List of incomplete split token IDs
        """
        return [token_id for token_id, completed in self.split_tokens.items() if not completed]

    def has_active_splits(self) -> bool:
        """
        Check if this token has any active splits.
        
        Returns:
            True if token has active splits, False otherwise
        """
        return self.active_splits > 0 or not all(self.split_tokens.values())
            
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
