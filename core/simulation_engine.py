import sys
import os
# Add project root to sys.path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import heapq
import logging
import random
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Callable

from core.event import Event
from core.process_token import Token
from core.resource import ResourceManager
from core.process_model import ProcessModel
from utils.time_utils import is_work_time, advance_to_work_time

class SimulationEngine:
    """
    Core simulation engine that processes events and manages the simulation state.
    Handles the event loop, token creation, resource allocation, and results collection.
    """
    
    def __init__(self, process_model: ProcessModel, start_time: datetime, 
                work_days: int, work_hours_per_day: int):
        """
        Initialize the simulation engine.
        
        Args:
            process_model: The process model to simulate
            start_time: The start time of the simulation
            work_days: Number of work days per week (1-7)
            work_hours_per_day: Number of work hours per day
        """
        self.process_model = process_model
        self.start_time = start_time
        self.work_days = work_days
        self.work_hours_per_day = work_hours_per_day
        self.event_queue = []  # Priority queue of events
        self.tokens = {}  # Active tokens by ID
        self.completed_tokens = []  # Completed tokens
        self.resource_manager = ResourceManager()
        self.simulation_days = 0  # Will be set during run_simulation
        
        # Set up resources based on the process model
        self._initialize_resources()
        
        # Statistics collection
        self.activity_stats = defaultdict(lambda: {
            "wait_times": [], 
            "durations": [], 
            "tokens_started": 0, 
            "tokens_completed": 0
        })
        
        self.total_tokens_started = 0
        
    def _initialize_resources(self) -> None:
        """Initialize resources from the process model."""
        resources = self.process_model.get_all_resources()
        for resource_id, count in resources.items():
            self.resource_manager.set_available_resources(resource_id, count)
            
    def schedule_tokens(self, count: int, interval_minutes: int, 
                       end_time: datetime) -> int:
        """
        Schedule tokens to start the process, distributed across the simulation period.
        
        Args:
            count: Target number of tokens to schedule
            interval_minutes: Minutes between token arrivals
            end_time: End time limit for scheduling
            
        Returns:
            Number of tokens actually scheduled
        """
        start_nodes = self.process_model.get_start_nodes()
        if not start_nodes:
            raise ValueError("No start nodes found in the process model.")
            
        current_time = self.start_time
        token_count = 0
        
        # Calculate total available time slots based on work hours
        total_minutes = 0
        temp_time = self.start_time
        
        # Count actual available working minutes in the simulation period
        while temp_time <= end_time:
            if is_work_time(temp_time, self.start_time, self.work_days, self.work_hours_per_day):
                total_minutes += 1
            temp_time += timedelta(minutes=1)
            # For efficiency, we can skip to the next work period if we're outside work hours
            if not is_work_time(temp_time, self.start_time, self.work_days, self.work_hours_per_day):
                temp_time = advance_to_work_time(temp_time, self.start_time, self.work_days, self.work_hours_per_day)
        
        # Calculate how many tokens we can realistically schedule with the given interval
        max_possible_tokens = total_minutes // interval_minutes
        target_tokens = min(count, max_possible_tokens)
        
        logging.info(f"Scheduling up to {target_tokens} tokens over {total_minutes} available minutes")
        
        while token_count < target_tokens and current_time <= end_time:
            # Ensure start time is within work hours
            if not is_work_time(current_time, self.start_time, 
                              self.work_days, self.work_hours_per_day):
                current_time = advance_to_work_time(
                    current_time, self.start_time, 
                    self.work_days, self.work_hours_per_day
                )
                if current_time > end_time:
                    break
                continue
                
            # Create and schedule token
            token_id = f"Token-{token_count + 1}"
            start_node = random.choice(start_nodes)
            
            token = Token(token_id, current_time, start_node)
            self.tokens[token_id] = token
            
            # Schedule start event
            heapq.heappush(
                self.event_queue, 
                Event(current_time, token_id, start_node, "start")
            )
            
            logging.info(f"Scheduled {token_id} to start at {current_time}.")
            token_count += 1
            self.total_tokens_started += 1
            current_time += timedelta(minutes=interval_minutes)
            
        return token_count
        
    def run_simulation(self, simulation_days: int, 
                      progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Run the simulation for the specified number of days.
        
        Args:
            simulation_days: Number of days to simulate
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with simulation results
        """
        self.simulation_days = simulation_days  # Store for later use
        simulation_end_date = self.start_time + timedelta(days=simulation_days)
        last_progress_update = datetime.now()
        progress_interval = timedelta(seconds=1)  # Update progress every second
        
        # Initialize progress
        if progress_callback:
            progress_callback("Starting simulation...")
            
        event_count = 0
        # Process all events
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            event_count += 1
            
            # Skip if event time is after simulation end
            if event.time > simulation_end_date:
                break
                
            # Process the event
            self.process_event(event)
            
            # Call progress callback periodically
            if progress_callback and datetime.now() - last_progress_update > progress_interval:
                completed = len(self.completed_tokens)
                total = self.total_tokens_started
                progress = (completed / total) * 100 if total > 0 else 0
                progress_callback(
                    f"Processing: {completed}/{total} tokens completed ({progress:.1f}%) - "
                    f"Events: {event_count} - Current sim time: {event.time}"
                )
                last_progress_update = datetime.now()
                
        # Final progress update
        if progress_callback:
            progress_callback("Simulation complete. Collecting results...")
            
        # Collect final statistics
        return self.get_results()
        
    def process_event(self, event: Event) -> None:
        """
        Process a single event in the simulation.
        
        Args:
            event: The event to process
        """
        if event.event_type == "start":
            self._handle_start_event(event)
        elif event.event_type == "end":
            self._handle_end_event(event)
            
    def _handle_start_event(self, event: Event) -> None:
        """
        Handle a start event - beginning of a task.
        
        Args:
            event: The start event to handle
        """
        token_id = event.token_id
        node_id = event.node_id
        event_time = event.time
        
        # Skip if token doesn't exist (already completed)
        if token_id not in self.tokens:
            return
            
        token = self.tokens[token_id]
        node = self.process_model.get_node(node_id)
        
        # Check if we're in work hours
        if not is_work_time(event_time, self.start_time, self.work_days, self.work_hours_per_day):
            next_work_time = advance_to_work_time(
                event_time, self.start_time, self.work_days, self.work_hours_per_day
            )
            heapq.heappush(
                self.event_queue, 
                Event(next_work_time, token_id, node_id, "start")
            )
            return
            
        # Check resource availability
        resource = node.get("resource")
        if resource:
            if not self.resource_manager.allocate_resource(resource, token_id, node_id, event_time):
                # Resource not available, add to wait queue (already done in allocate_resource)
                token.start_waiting(event_time)
                logging.info(
                    f"Token {token_id} waiting for resource '{resource}' at {event_time}"
                )
                return
        
        # Initialize processing_times array if it doesn't exist
        if "processing_times" not in self.activity_stats[node_id]:
            self.activity_stats[node_id]["processing_times"] = []
        
        # If the token was waiting, record wait time
        wait_duration = token.stop_waiting(event_time)
        if wait_duration > 0:
            self.activity_stats[node_id]["wait_times"].append(wait_duration)
            logging.info(
                f"Token {token_id} waited {wait_duration:.2f} minutes for resource"
            )
        
        # Update token state
        token.add_to_path(node_id)
        
        # Update activity statistics
        self.activity_stats[node_id]["tokens_started"] += 1
        
        # Calculate pure processing time
        task_duration = self._calculate_task_duration(node)
        
        # Store pure processing time separately
        self.activity_stats[node_id]["processing_times"].append(task_duration)
        
        # Store in durations (for backward compatibility)
        # For Bizagi compatibility: durations should track total time (wait + processing)
        # But we're only calculating it at the end of the activity
        self.activity_stats[node_id]["durations"].append(task_duration)
        
        # Calculate end time based on processing time
        end_time = event_time + timedelta(minutes=task_duration)
        
        # Ensure end time is within work hours
        if not is_work_time(end_time, self.start_time, self.work_days, self.work_hours_per_day):
            end_time = advance_to_work_time(
                end_time, self.start_time, self.work_days, self.work_hours_per_day
            )
        
        # Schedule end event
        heapq.heappush(
            self.event_queue, 
            Event(end_time, token_id, node_id, "end")
        )
        
        logging.info(
            f"Token {token_id} started task '{node_id}' at {event_time}, "
            f"scheduled to end at {end_time} (processing time: {task_duration:.2f} min)"
        )
        
        
    def _handle_end_event(self, event: Event) -> None:
        """
        Handle an end event - completion of a task.
        
        Args:
            event: The end event to handle
        """
        token_id = event.token_id
        node_id = event.node_id
        event_time = event.time
        
        # Skip if token doesn't exist (already completed)
        if token_id not in self.tokens:
            return
            
        token = self.tokens[token_id]
        node = self.process_model.get_node(node_id)
        
        logging.info(f"Token {token_id} completed task '{node_id}' at {event_time}.")
        
        # Update activity statistics
        self.activity_stats[node_id]["tokens_completed"] += 1
        
        # Release resources
        resource = node.get("resource")
        if resource:
            # Get next token from wait queue if any
            next_token_info = self.resource_manager.release_resource(resource, event_time)
            if next_token_info:
                next_token_id, next_node_id, _ = next_token_info
                heapq.heappush(
                    self.event_queue, 
                    Event(event_time, next_token_id, next_node_id, "start")
                )
                logging.info(
                    f"Token {next_token_id} released from wait queue for '{resource}'"
                )
                
        # Determine next nodes
        next_nodes = self.process_model.get_next_nodes(node_id)
        
        if not next_nodes or node.get("type") == "Stop":
            # Token has completed the process
            token.complete(event_time)
            logging.info(
                f"Token {token_id} has completed the entire process at {event_time}."
            )
            self.completed_tokens.append(token)
            del self.tokens[token_id]
        else:
            # Schedule events for next nodes
            for next_node_id in next_nodes:
                heapq.heappush(
                    self.event_queue, 
                    Event(event_time, token_id, next_node_id, "start")
                )
                
    def _calculate_task_duration(self, node: Dict[str, Any]) -> float:
        """
        Calculate task duration using triangular distribution.
        
        Args:
            node: Node data containing duration parameters
            
        Returns:
            Duration in minutes
        """
        min_time = float(node.get("min time", 0))
        avg_time = float(node.get("avg time", 0))  # Mode of the triangular distribution
        max_time = float(node.get("max time", 0))
        
        if avg_time > 0 and min_time > 0 and max_time > 0:
            return random.triangular(min_time, avg_time, max_time)
        else:
            return 0  # No processing time
            
    def get_results(self) -> Dict[str, Any]:
        """
        Get the simulation results.
        
        Returns:
            Dictionary with simulation results
        """
        # Calculate resource utilization using proper simulation end date
        simulation_end = self.start_time + timedelta(days=self.simulation_days)
        resource_utilization = self.resource_manager.calculate_utilization(
            self.start_time, simulation_end
        )
        
        # Convert tokens to dictionaries for JSON serialization
        completed_token_dicts = [token.to_dict() for token in self.completed_tokens]
        
        # Also include active tokens to see where they got stuck
        active_token_dicts = [token.to_dict() for token in self.tokens.values()]
        
        return {
            "activity_processing_times": self.activity_stats,
            "resource_utilization": resource_utilization,
            "total_tokens_started": self.total_tokens_started,
            "completed_tokens": completed_token_dicts,
            "active_tokens": active_token_dicts,
            "simulation_days": self.simulation_days
        }
