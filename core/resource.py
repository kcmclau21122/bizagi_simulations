from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

class ResourceManager:
    """
    Manages resources during simulation, handling allocation,
    waiting queues, and calculating utilization.
    """
    
    def __init__(self, initial_resources: Optional[Dict[str, Any]] = None):
        """Initialize the resource manager."""
        self.active_resources = defaultdict(int)  # Currently in use
        self.resource_wait_queue = defaultdict(list)  # Tokens waiting for resources
        self.resource_busy_periods = defaultdict(list)  # When resources were busy
        self.available_resources = initial_resources if initial_resources is not None else {}
        
    # ... (rest of the methods remain unchanged)

        
    def set_available_resources(self, resource_id: str, count: int) -> None:
        """Set the number of available resources of a given type."""
        self.available_resources[resource_id] = count
        
    def get_available_count(self, resource_id: str) -> int:
        """Get the number of available instances of a resource."""
        return self.available_resources.get(resource_id, 1)
        
    def is_resource_available(self, resource_id: str) -> bool:
        """Check if a resource is available."""
        available = self.get_available_count(resource_id)
        active = self.active_resources[resource_id]
        return active < available
        
    def allocate_resource(self, resource_id: str, token_id: str, 
                        node_id: str, time: datetime) -> bool:
        """
        Try to allocate a resource for a token.
        Returns True if resource was allocated, False if token was added to wait queue.
        """
        if self.is_resource_available(resource_id):
            self.active_resources[resource_id] += 1
            self.resource_busy_periods[resource_id].append([time, None])
            return True
        else:
            # Add to wait queue - store token_id, node_id, and queue time
            self.resource_wait_queue[resource_id].append((token_id, node_id, time))
            return False
            
# In resource.py
    def release_resource(self, resource_id: str, time: datetime) -> Optional[Tuple[str, str, datetime]]:
        """
        Release a resource and get the next token in queue if any.
        Returns (token_id, node_id, queued_time) of next token if one is waiting, None otherwise.
        """
        if self.active_resources[resource_id] > 0:
            self.active_resources[resource_id] -= 1
            
            # Update busy period end time
            for period in reversed(self.resource_busy_periods[resource_id]):
                if period[1] is None:
                    period[1] = time
                    break
                    
            # Check wait queue
            if self.resource_wait_queue[resource_id]:
                next_token = self.resource_wait_queue[resource_id].pop(0)
                
                # Immediately mark the resource as allocated again for the next token
                self.active_resources[resource_id] += 1
                self.resource_busy_periods[resource_id].append([time, None])
                
                return next_token
        
        return None
        
    def get_wait_queue_length(self, resource_id: str) -> int:
        """Get the number of tokens waiting for a resource."""
        return len(self.resource_wait_queue[resource_id])
        
    def calculate_utilization(self, simulation_start: datetime, 
                             simulation_end: datetime) -> Dict[str, float]:
        """
        Calculate resource utilization for the simulation period.
        Returns a dictionary mapping resource IDs to utilization percentages.
        """
        resource_utilization = {}
        total_simulation_time = (simulation_end - simulation_start).total_seconds()
        
        for resource, periods in self.resource_busy_periods.items():
            # Calculate total busy time
            total_busy_time = sum(
                (end - start).total_seconds() 
                for start, end in periods 
                if start and end
            )
            
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
        """Get detailed statistics for all resources."""
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
                "available_count": self.get_available_count(resource)
            }
            
        return stats
