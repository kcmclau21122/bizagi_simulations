from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
import logging

class ResourceManager:
    """
    Manages resources during simulation, handling allocation,
    waiting queues, and calculating utilization.
    """
    
    def __init__(self):
        """Initialize the resource manager."""
        self.active_resources = defaultdict(int)  # Currently in use
        self.resource_wait_queue = defaultdict(list)  # Tokens waiting for resources (FIFO)
        self.resource_busy_periods = defaultdict(list)  # When resources were busy
        self.available_resources = {}  # Total available count of each resource
        self.token_resources = {}  # Maps token IDs to resources they're using
        
    def set_available_resources(self, resource_id: str, count: int) -> None:
        """
        Set the number of available resources of a given type.
        
        Args:
            resource_id: Resource identifier
            count: Number of resource instances
        """
        self.available_resources[resource_id] = max(1, count)  # Ensure at least 1
        
    def get_available_count(self, resource_id: str) -> int:
        """
        Get the number of available instances of a resource.
        
        Args:
            resource_id: Resource identifier
            
        Returns:
            Number of available resource instances
        """
        return self.available_resources.get(resource_id, 1)
        
    def is_resource_available(self, resource_id: str) -> bool:
        """
        Check if a resource is available.
        
        Args:
            resource_id: Resource identifier
            
        Returns:
            True if at least one instance is available, False otherwise
        """
        available = self.get_available_count(resource_id)
        active = self.active_resources[resource_id]
        return active < available
        
    def allocate_resource(self, resource_id: str, token_id: str, 
                        node_id: str, time: datetime) -> bool:
        """
        Try to allocate a resource for a token.
        Returns True if resource was allocated, False if token was added to wait queue.
        
        Args:
            resource_id: Resource identifier
            token_id: Token identifier
            node_id: Node identifier
            time: Current simulation time
            
        Returns:
            True if resource was allocated, False if added to wait queue
        """
        if self.is_resource_available(resource_id):
            self.active_resources[resource_id] += 1
            self.resource_busy_periods[resource_id].append([time, None])
            
            # Track which resource this token is using
            self.token_resources[token_id] = resource_id
            
            # Log resource allocation for debugging
            logging.debug(f"Resource {resource_id} allocated to token {token_id} at {time}")
            return True
        else:
            # Add to wait queue - store token_id, node_id, and queue time in FIFO order
            wait_entry = (token_id, node_id, time)
            self.resource_wait_queue[resource_id].append(wait_entry)
            
            # Log wait queue entry for debugging
            logging.debug(f"Token {token_id} added to wait queue for resource {resource_id} at position {len(self.resource_wait_queue[resource_id])-1}")
            return False
                
    def release_resource(self, resource_id: str, time: datetime) -> Optional[Tuple[str, str, datetime]]:
        """
        Release a resource and get the next token in queue if any.
        Uses FIFO queue for waiting tokens.

        Args:
            resource_id: Resource identifier
            time: Current simulation time
            
        Returns:
            (token_id, node_id, queued_time) of next token in queue, or None
        """
        if self.active_resources[resource_id] > 0:
            self.active_resources[resource_id] -= 1
            
            # Update busy period end time
            for period in reversed(self.resource_busy_periods[resource_id]):
                if period[1] is None:
                    period[1] = time
                    break
            
            # Check wait queue - get the next waiting token
            if self.resource_wait_queue[resource_id]:
                next_token = self.resource_wait_queue[resource_id].pop(0)  # FIFO - first in first out
                
                # Allocate resource to this token
                token_id = next_token[0]
                self.active_resources[resource_id] += 1
                self.resource_busy_periods[resource_id].append([time, None])
                self.token_resources[token_id] = resource_id
                
                # Log resource reallocation for debugging
                logging.debug(f"Resource {resource_id} reallocated to waiting token {token_id} from wait queue position 0")
                
                return next_token
            else:
                # Log that resource was released with no waiting tokens
                logging.debug(f"Resource {resource_id} released but no tokens waiting in queue")
        else:
            logging.warning(f"Attempted to release resource {resource_id} that has no active allocations")

        return None    
    
    def release_all_token_resources(self, token_id: str, time: datetime) -> Set[str]:
        """
        Release all resources held by a token.
        Useful for cleaning up when a token is completed or removed.
        
        Args:
            token_id: Token identifier
            time: Current simulation time
            
        Returns:
            Set of resource IDs that were released
        """
        released_resources = set()
        
        # Check if token is using any resources
        if token_id in self.token_resources:
            resource_id = self.token_resources[token_id]
            
            # Release the resource
            self.release_resource(resource_id, time)
            released_resources.add(resource_id)
            
            # Remove token from tracking
            del self.token_resources[token_id]
            
            logging.debug(f"Released resource {resource_id} held by token {token_id}")
        
        # Check if token is in any wait queues and remove it
        for resource_id, wait_queue in list(self.resource_wait_queue.items()):
            # Check each entry in the wait queue
            updated_queue = []
            removed = False
            
            for queue_entry in wait_queue:
                waiting_token_id = queue_entry[0]
                
                if waiting_token_id != token_id:
                    # Keep this entry
                    updated_queue.append(queue_entry)
                else:
                    # Skip this entry (token being removed)
                    removed = True
                    logging.debug(f"Removed token {token_id} from wait queue for resource {resource_id}")
            
            # Update the wait queue if changes were made
            if removed:
                self.resource_wait_queue[resource_id] = updated_queue
                released_resources.add(resource_id)
        
        return released_resources

    def get_next_waiting_token(self, resource_id: str) -> Optional[Tuple[str, str, datetime]]:
        """
        Get the next token waiting for a resource without removing it from the queue.
        
        Args:
            resource_id: Resource identifier
            
        Returns:
            (token_id, node_id, queued_time) of next token in queue, or None
        """
        if resource_id in self.resource_wait_queue and self.resource_wait_queue[resource_id]:
            return self.resource_wait_queue[resource_id][0]
        return None

    def transfer_resource(self, from_token_id: str, to_token_id: str) -> bool:
        """
        Transfer a resource from one token to another.
        Useful during token merging at gateways.
        
        Args:
            from_token_id: ID of token transferring the resource
            to_token_id: ID of token receiving the resource
            
        Returns:
            True if resource was transferred, False otherwise
        """
        if from_token_id not in self.token_resources:
            return False
        
        # Get the resource being used by the 'from' token
        resource_id = self.token_resources[from_token_id]
        
        # Update token resource mapping
        self.token_resources[to_token_id] = resource_id
        del self.token_resources[from_token_id]
        
        logging.debug(f"Transferred resource {resource_id} from token {from_token_id} to token {to_token_id}")
        return True

    def get_token_resource(self, token_id: str) -> Optional[str]:
        """
        Get the resource currently being used by a token.
        
        Args:
            token_id: Token identifier
            
        Returns:
            Resource ID if token is using a resource, None otherwise
        """
        return self.token_resources.get(token_id)

    def get_resource_utilization_snapshot(self) -> Dict[str, float]:
        """
        Get current resource utilization as a snapshot.
        
        Returns:
            Dictionary mapping resource IDs to current utilization percentages
        """
        utilization = {}
        
        for resource_id, available_count in self.available_resources.items():
            active_count = self.active_resources.get(resource_id, 0)
            if available_count > 0:
                utilization[resource_id] = (active_count / available_count) * 100
            else:
                utilization[resource_id] = 0
        
        return utilization

    def get_wait_queue_length(self, resource_id: str) -> int:
        """
        Get the number of tokens waiting for a resource.
        
        Args:
            resource_id: Resource identifier
            
        Returns:
            Number of tokens in the wait queue
        """
        return len(self.resource_wait_queue[resource_id])
    
    def get_token_queue_position(self, resource_id: str, token_id: str) -> int:
        """
        Get a token's position in the resource wait queue.
        
        Args:
            resource_id: Resource identifier
            token_id: Token identifier
            
        Returns:
            Position in queue (0-based), or -1 if not in queue
        """
        for i, (waiting_token_id, _, _) in enumerate(self.resource_wait_queue[resource_id]):
            if waiting_token_id == token_id:
                return i
        return -1
        
    def calculate_utilization(self, simulation_start: datetime, 
                            simulation_end: datetime,
                            is_work_time_func=None) -> Dict[str, float]:
        """
        Calculate resource utilization for the simulation period.
        
        Args:
            simulation_start: Start time of simulation
            simulation_end: End time of simulation
            is_work_time_func: Optional function to determine if a time is within work hours
            
        Returns:
            Dictionary mapping resource IDs to utilization percentages
        """
        resource_utilization = {}
        
        # Count total working minutes during the simulation period
        if is_work_time_func:
            # Calculate only working hours
            total_work_minutes = 0
            current_time = simulation_start
            
            while current_time <= simulation_end:
                if is_work_time_func(current_time):
                    total_work_minutes += 1
                current_time += timedelta(minutes=1)
                
            total_simulation_time = total_work_minutes * 60  # Convert to seconds
        else:
            # Use all time
            total_simulation_time = (simulation_end - simulation_start).total_seconds()
        
        for resource, periods in self.resource_busy_periods.items():
            # Calculate total busy time, filtering for work hours if is_work_time_func provided
            total_busy_time = 0
            
            for start, end in periods:
                if start and end:
                    # If we have a work time function, count only busy time during work hours
                    if is_work_time_func:
                        busy_time = 0
                        current = start
                        while current < end:
                            if is_work_time_func(current):
                                busy_time += 1  # Add 1 minute
                            current += timedelta(minutes=1)
                        total_busy_time += busy_time * 60  # Convert to seconds
                    else:
                        # Otherwise, count all busy time
                        total_busy_time += (end - start).total_seconds()
            
            # Get number of resource instances
            num_resources = self.get_available_count(resource)
            
            # Calculate utilization percentage
            if total_simulation_time > 0 and num_resources > 0:
                utilization = (total_busy_time / (total_simulation_time * num_resources)) * 100
                resource_utilization[resource] = min(utilization, 100)
            else:
                resource_utilization[resource] = 0
                
        return resource_utilization
        
    def get_resource_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed statistics for all resources.
        
        Returns:
            Dictionary mapping resource IDs to statistics dictionaries
        """
        stats = {}
        
        for resource, periods in self.resource_busy_periods.items():
            if not periods:
                continue
                
            busy_times = [
                (end - start).total_seconds() / 60  # in minutes
                for start, end in periods 
                if start and end
            ]
            
            if not busy_times:
                continue
                
            stats[resource] = {
                "average_busy_time": sum(busy_times) / len(busy_times),
                "total_busy_time": sum(busy_times),
                "min_busy_time": min(busy_times),
                "max_busy_time": max(busy_times),
                "busy_periods": len(busy_times),
                "available_count": self.get_available_count(resource),
                "wait_queue_length": self.get_wait_queue_length(resource)
            }
            
        return stats
