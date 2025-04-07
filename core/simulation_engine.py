import os
import heapq
import logging
import random
import re
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
import pandas as pd
import math

from core.event import Event
from core.process_token import Token
from core.resource import ResourceManager
from core.process_model import ProcessModel
from utils.time_utils import is_work_time, advance_to_work_time
from data.data_loader import DataLoader
from data.xpdl_parser import parse_xpdl_to_json, parse_xpdl_with_metrics
from data.process_builder import ProcessModelBuilder
from data.data_processing import ResultsExporter

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
        
        # Gateway tracking
        self.gateway_merge_nodes = {}  # Maps gateway IDs to their merge nodes
        
        # Set up resources based on the process model
        self._initialize_resources()
        
        # Statistics collection
        self.activity_stats = defaultdict(lambda: {
            "wait_times": [], 
            "processing_times": [],
            "durations": [], 
            "tokens_started": 0, 
            "tokens_completed": 0,
            "type": "Task"  # Default type
        })
        
        self.total_tokens_started = 0
        
    def _initialize_resources(self) -> None:
        """Initialize resources from the process model."""
        resources = self.process_model.get_all_resources()
        for resource_id, count in resources.items():
            self.resource_manager.set_available_resources(resource_id, count)
            logging.info(f"Initialized resource '{resource_id}' with {count} instances")
            
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
        
    def run_simulation(self, simulation_days: int, progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Run the simulation process for a specified number of days.
        
        Args:
            simulation_days: Number of days to simulate
            progress_callback: Optional callback for reporting progress
            
        Returns:
            Dictionary with simulation results
        """
        self.simulation_days = simulation_days
        end_time = self.start_time + timedelta(days=simulation_days)
        
        # Calculate the total number of tokens scheduled
        total_tokens = len(self.tokens)
        processed_tokens = 0
        last_progress_update = 0
        
        # Process events until the queue is empty or all tokens are complete
        while self.event_queue and len(self.tokens) > 0:
            # Get the next event
            event = heapq.heappop(self.event_queue)
            
            # Skip if event is after simulation end
            if event.time > end_time:
                logging.info(f"Skipping event after simulation end: {event}")
                continue
            
            # Process the event
            self.process_event(event)
            
            # Calculate progress as percentage of tokens processed
            if total_tokens > 0:
                processed_tokens = self.total_tokens_started - len(self.tokens)
                progress_percent = (processed_tokens / total_tokens) * 100
                
                # Report progress at 10% intervals
                if progress_callback and int(progress_percent / 10) > last_progress_update:
                    last_progress_update = int(progress_percent / 10)
                    progress_callback(f"Processing tokens: {processed_tokens}/{total_tokens} ({progress_percent:.1f}%)")
                    logging.info(f"Simulation progress: {progress_percent:.1f}%")
        
        # Log completion
        if progress_callback:
            progress_callback(f"Simulation complete. Processed {len(self.completed_tokens)} of {self.total_tokens_started} tokens.")
        
        logging.info(f"Simulation completed with {len(self.completed_tokens)} tokens completed out of {self.total_tokens_started} started.")
        
        # Return the results
        return self.get_results()
        
    def process_event(self, event: Event) -> None:
        """
        Process a single event in the simulation with improved time tracking.
        
        Args:
            event: The event to process
        """
        logging.debug(f"Processing event: {event.event_type} for token {event.token_id} at node {event.node_id} at time {event.time}")
        
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
        
        # Skip if token doesn't exist (already completed or removed)
        if token_id not in self.tokens:
            logging.debug(f"Token {token_id} no longer exists, skipping start event")
            return
            
        token = self.tokens[token_id]
        
        # Error handling: Ensure the node exists in the process model
        node = self.process_model.get_node(node_id)
        if not node:
            logging.error(f"Node {node_id} not found in process model. Removing token {token_id}")
            # Remove the token to avoid simulation getting stuck
            del self.tokens[token_id]
            return
        
        # Log for debugging
        logging.debug(f"Handling start event for token {token_id} at node {node_id}")
        
        # Check if we're in work hours
        if not is_work_time(event_time, self.start_time, self.work_days, self.work_hours_per_day):
            next_work_time = advance_to_work_time(
                event_time, self.start_time, self.work_days, self.work_hours_per_day
            )
            logging.debug(f"Outside work hours, rescheduling token {token_id} to {next_work_time}")
            heapq.heappush(
                self.event_queue, 
                Event(next_work_time, token_id, node_id, "start")
            )
            return
        
        # Ensure we have the node type
        node_type = node.get("type", "Task")
        self.activity_stats[node_id]["type"] = node_type
        
        # Update token state - add this node to the path
        token.add_to_path(node_id)
        
        # Check if this is a gateway node
        gateway_type = node.get("gateway")
        if gateway_type:
            logging.debug(f"Node {node_id} is a gateway of type {gateway_type}")
            # Gateways don't require resources, so we can immediately process them
            # Schedule the end event for the gateway node
            heapq.heappush(
                self.event_queue, 
                Event(event_time, token_id, node_id, "end")
            )
            return
            
        # Check resource availability
        resource = node.get("resource")
        if resource:
            logging.debug(f"Node {node_id} requires resource {resource}")
            
            # Check if token already has this resource assigned
            # This handles tokens coming from the wait queue that already have the resource
            current_resource = self.resource_manager.get_token_resource(token_id)
            if current_resource != resource:
                # Only try to allocate if the token doesn't already have this resource
                resource_allocated = self.resource_manager.allocate_resource(resource, token_id, node_id, event_time)
                if not resource_allocated:
                    # Resource not available, token was added to wait queue
                    token.start_waiting(event_time)
                    logging.info(f"Token {token_id} waiting for resource '{resource}' at {event_time}")
                    return
                logging.debug(f"Resource {resource} allocated to token {token_id}")

        # Update activity statistics
        self.activity_stats[node_id]["tokens_started"] += 1
        
        # If the token was waiting, record wait time
        wait_duration = 0
        if token.wait_start_time is not None:
            wait_duration = token.stop_waiting(event_time)
            if wait_duration > 0:
                self.activity_stats[node_id]["wait_times"].append(wait_duration)
                logging.info(f"Token {token_id} waited {wait_duration:.2f} minutes for resource")
        
        # Calculate pure processing time
        task_duration = self._calculate_task_duration(node)
        
        # Store processing time separately
        if "processing_times" not in self.activity_stats[node_id]:
            self.activity_stats[node_id]["processing_times"] = []
        self.activity_stats[node_id]["processing_times"].append(task_duration)
        
        # Store in durations (for backward compatibility)
        if "durations" not in self.activity_stats[node_id]:
            self.activity_stats[node_id]["durations"] = []
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
        
        # Skip if token doesn't exist (already completed or removed)
        if token_id not in self.tokens:
            logging.debug(f"Token {token_id} no longer exists, skipping end event")
            return
            
        token = self.tokens[token_id]
        node = self.process_model.get_node(node_id)
        
        # Update activity statistics
        self.activity_stats[node_id]["tokens_completed"] += 1
        
        # Release resources if used
        resource = node.get("resource")
        if resource:
            # Get next token from wait queue if any
            next_token_info = self.resource_manager.release_resource(resource, event_time)
            logging.debug(f"Released resource {resource} from token {token_id}")
            
            if next_token_info:
                next_token_id, next_node_id, queued_time = next_token_info
                logging.debug(f"Allocated resource {resource} to waiting token {next_token_id}")
                
                # Calculate wait time for statistics
                waiting_token = self.tokens.get(next_token_id)
                if waiting_token and waiting_token.wait_start_time is not None:
                    wait_duration = waiting_token.stop_waiting(event_time)
                    if wait_duration > 0:
                        self.activity_stats[next_node_id]["wait_times"].append(wait_duration)
                        logging.info(f"Token {next_token_id} waited {wait_duration:.2f} minutes for resource")
                
                # Schedule immediate start event for waiting token
                heapq.heappush(
                    self.event_queue, 
                    Event(event_time, next_token_id, next_node_id, "start")
                )
                logging.info(
                    f"Token {next_token_id} released from wait queue for '{resource}'"
                )
        
        # Check if this is a gateway
        gateway_type = node.get("gateway")
        if gateway_type:
            if "[Exclusive Gateway]" in str(gateway_type):
                # For exclusive gateway, proceed with one path based on conditions
                self._handle_exclusive_gateway(token, node_id, event_time)
                return
            elif "[Inclusive Gateway]" in str(gateway_type) or "[Parallel Gateway]" in str(gateway_type):
                # For inclusive or parallel gateway, split the token
                self._handle_splitting_gateway(token, node_id, gateway_type, event_time)
                return
        
        # Check if this is a merge node
        if self._is_merge_node(node_id):
            # Handle merging of split tokens
            self._handle_merge_node(token, node_id, event_time)
            return
        
        # For regular nodes, determine next nodes
        # Use the improved get_next_nodes method from ProcessModel
        next_nodes = self.process_model.get_next_nodes(node_id)
        
        if not next_nodes or node.get("type") == "Stop":
            # Token has completed the process
            token.complete(event_time)
            
            # Calculate and log the processing time
            process_duration = (event_time - token.start_time).total_seconds() / 60
            
            logging.info(
                f"Token {token_id} has completed the entire process at {event_time}. "
                f"Total processing time: {process_duration:.2f} minutes "
                f"(Wait time: {token.total_wait_time:.2f} minutes)"
            )
            
            self.completed_tokens.append(token)
            del self.tokens[token_id]
        else:
            # Schedule events for next nodes
            logging.debug(f"Token {token_id} next nodes: {next_nodes}")
            for next_node_id in next_nodes:
                heapq.heappush(
                    self.event_queue, 
                    Event(event_time, token_id, next_node_id, "start")
                )
                
    def _handle_exclusive_gateway(self, token: Token, gateway_id: str, event_time: datetime) -> None:
        """
        Handle an exclusive gateway - choose one outgoing path based on conditions.
        
        Args:
            token: The token at the gateway
            gateway_id: ID of the gateway node
            event_time: Current event time
        """
        # Get all outgoing paths with normalized node names
        next_nodes = self.process_model.get_next_nodes(gateway_id)
        
        if not next_nodes:
            logging.warning(f"Gateway {gateway_id} has no outgoing paths")
            # Handle as termination
            token.complete(event_time)
            self.completed_tokens.append(token)
            del self.tokens[token.token_id]
            logging.info(f"Token {token.token_id} completed process at gateway {gateway_id} with no paths")
            return
        
        # Evaluate conditions to find the first path that should be taken
        selected_node = self._evaluate_exclusive_gateway_conditions(gateway_id, next_nodes, token)
        
        logging.info(f"Token {token.token_id} taking path to {selected_node} at exclusive gateway {gateway_id}")
        
        # Schedule start event for the selected node
        heapq.heappush(
            self.event_queue, 
            Event(event_time, token.token_id, selected_node, "start")
        )

    def _evaluate_inclusive_gateway_conditions(self, gateway_id: str, next_nodes: List[str], token: Token) -> List[str]:
        """
        Evaluate conditions for inclusive gateway paths using triangular distribution.
        This matches Bizagi Modeler's methodology more closely by considering
        condition complexity and maintaining consistent decision-making.
        
        Args:
            gateway_id: ID of the gateway
            next_nodes: List of next node IDs
            token: The token being processed
            
        Returns:
            List of node IDs for paths that should be taken
        """
        paths_to_take = []
        paths_with_conditions = []
        default_paths = []
        
        # Get gateway properties
        gateway_node = self.process_model.get_node(gateway_id)
        gateway_name = gateway_node.get('name', gateway_id)
        
        logging.info(f"Evaluating conditions for inclusive gateway {gateway_id} with {len(next_nodes)} possible paths")
        
        # Use token ID as random seed for consistent decisions
        # This ensures the same token will make the same decisions at the same gateway
        token_seed = int(''.join([str(ord(c)) for c in token.token_id])[:8])
        local_random = random.Random(token_seed)
        
        for node_id in next_nodes:
            # Get the edge data (including condition)
            edge_data = self.process_model.graph.get_edge_data(gateway_id, node_id) or {}
            edge_type = edge_data.get("type", "")
            condition = edge_data.get("condition")
            
            # Check if this edge has a condition
            if edge_type.startswith("CONDITION-") or condition:
                # Extract the condition expression
                condition_expr = condition if condition else edge_type.replace("CONDITION-", "").strip()
                
                # Store information for later probability calculation
                paths_with_conditions.append((node_id, condition_expr))
                logging.debug(f"Found conditional path to {node_id} with condition: {condition_expr}")
            else:
                # If no condition, this is a default path
                default_paths.append(node_id)
                logging.debug(f"Found default path to {node_id}")
        
        # If there are conditional paths, evaluate them using triangular distribution
        if paths_with_conditions:
            # Calculate probabilities based on triangular distribution
            # Each path gets a different position in the distribution
            num_paths = len(paths_with_conditions)
            for i, (node_id, condition_expr) in enumerate(paths_with_conditions):
                # Calculate path position in triangular distribution (0 to 1)
                path_position = (i + 1) / (num_paths + 1)
                
                # Calculate condition complexity
                condition_complexity = condition_expr.count("==") + condition_expr.count(">") + \
                                    condition_expr.count("<") + condition_expr.count("!=") + \
                                    condition_expr.count("AND") + condition_expr.count("OR") + 1
                
                # Adjust the mode (peak) of triangular distribution based on complexity
                # More complex conditions get lower probability (like Bizagi)
                min_val = max(0.1, path_position - 0.3)
                max_val = min(1.0, path_position + 0.3)
                mode = path_position - (0.05 * condition_complexity)  # Lower mode for complex conditions
                
                # Sample from triangular distribution
                probability = local_random.triangular(min_val, max_val, mode)
                threshold = 0.5 - (0.05 * condition_complexity)  # Lower threshold for complex conditions
                
                # Decide if this path is taken
                if probability > threshold:
                    paths_to_take.append(node_id)
                    logging.info(f"Taking conditional path to {node_id} (prob={probability:.2f}, threshold={threshold:.2f})")
                else:
                    logging.info(f"Skipping conditional path to {node_id} (prob={probability:.2f}, threshold={threshold:.2f})")
        
        # Add default paths (if any)
        paths_to_take.extend(default_paths)
        
        # Ensure at least one path is taken for inclusive gateway
        if not paths_to_take and next_nodes:
            # If no paths were selected, choose one random path
            # Use same random seed for consistency
            random_path = next_nodes[local_random.randint(0, len(next_nodes)-1)] if next_nodes else None
            if random_path:
                paths_to_take.append(random_path)
                logging.info(f"No paths selected, taking random path to {random_path} as fallback")
        
        logging.info(f"Inclusive gateway {gateway_id} taking {len(paths_to_take)} of {len(next_nodes)} possible paths")
        return paths_to_take

    def _handle_splitting_gateway(self, token: Token, gateway_id: str, gateway_type: str, event_time: datetime) -> None:
        """
        Handle an inclusive or parallel gateway - split the token.
        
        Args:
            token: The token at the gateway
            gateway_id: ID of the gateway node
            gateway_type: Type of the gateway
            event_time: Current event time
        """
        # Log gateway entry for debugging
        logging.info(f"Token {token.token_id} entering {gateway_type} gateway {gateway_id} at {event_time}")
        
        # Get all outgoing paths with normalized node names
        next_nodes = self.process_model.get_next_nodes(gateway_id)
        
        if not next_nodes:
            logging.warning(f"Gateway {gateway_id} has no outgoing paths")
            # Handle as termination
            token.complete(event_time)
            self.completed_tokens.append(token)
            del self.tokens[token.token_id]
            logging.info(f"Token {token.token_id} completed process at gateway {gateway_id} with no paths")
            return
        
        # Find the merge node for this gateway
        merge_node = self.process_model.find_merge_node(gateway_id)
        
        if not merge_node:
            logging.warning(f"No merge node found for gateway {gateway_id}, continuing with split paths")
        else:
            logging.info(f"Found merge node {merge_node} for gateway {gateway_id}")
        
        # Store merge node information in token
        token.merge_node = merge_node
        token.active_gateway = gateway_id
        
        # For inclusive gateway, evaluate conditions to determine paths
        # For parallel gateway, take all paths
        paths_to_take = []
        
        if "[Parallel Gateway]" in str(gateway_type):
            # For parallel gateway, take all paths
            paths_to_take = next_nodes
            logging.info(f"Parallel gateway {gateway_id} taking all {len(next_nodes)} paths")
        
        elif "[Inclusive Gateway]" in str(gateway_type):
            # For inclusive gateway, evaluate conditions for each path
            # Use the improved inclusive gateway condition evaluation
            paths_to_take = self._evaluate_inclusive_gateway_conditions(gateway_id, next_nodes, token)
        else:  # Default to exclusive gateway behavior
            # Take one random path
            paths_to_take = [random.choice(next_nodes)]
            logging.info(f"Unknown gateway type '{gateway_type}', defaulting to exclusive behavior")
        
        # Log the paths being taken
        logging.info(f"Token {token.token_id} taking {len(paths_to_take)} paths at gateway {gateway_id}: {paths_to_take}")
        
        # If there is only one path to take, don't create split tokens
        if len(paths_to_take) == 1:
            # Schedule the single path without creating split tokens
            next_node_id = paths_to_take[0]
            heapq.heappush(
                self.event_queue, 
                Event(event_time, token.token_id, next_node_id, "start")
            )
            logging.info(f"Token {token.token_id} taking single path to {next_node_id}")
            return
        
        # Create split tokens for each path
        for i, next_node_id in enumerate(paths_to_take):
            if i == 0:
                # Use the original token for the first path
                split_token_id = token.token_id
                logging.debug(f"Using original token {split_token_id} for path to {next_node_id}")
            else:
                # Create new tokens for additional paths
                split_token_id = f"{token.token_id}-{i}"
                split_token = token.create_split_token(split_token_id, event_time, gateway_id)
                self.tokens[split_token_id] = split_token
                logging.debug(f"Created split token {split_token_id} for path to {next_node_id}")
            
            # Schedule start event for the next node
            heapq.heappush(
                self.event_queue, 
                Event(event_time, split_token_id, next_node_id, "start")
            )
            
            logging.info(f"Scheduled split token {split_token_id} to start at node {next_node_id}")

    def _evaluate_exclusive_gateway_conditions(self, gateway_id: str, next_nodes: List[str], token: Token) -> str:
        """
        Evaluate conditions at exclusive gateway using triangular distribution.
        This matches Bizagi Modeler's methodology by considering condition 
        complexity and probability when determining which path to take.
        
        Args:
            gateway_id: ID of the gateway
            next_nodes: List of next node IDs
            token: The token being processed
            
        Returns:
            Node ID of the selected path
        """
        # Get gateway properties
        gateway_node = self.process_model.get_node(gateway_id)
        gateway_name = gateway_node.get('name', gateway_id)
        
        # Use token ID as random seed for consistent decisions
        token_seed = int(''.join([str(ord(c)) for c in token.token_id])[:8])
        local_random = random.Random(token_seed)
        
        paths_with_conditions = []
        default_path = None
        
        # First pass - categorize paths
        for node_id in next_nodes:
            # Get the edge data (including condition)
            edge_data = self.process_model.graph.get_edge_data(gateway_id, node_id) or {}
            edge_type = edge_data.get("type", "")
            condition = edge_data.get("condition")
            
            # Check if this edge has a condition
            if edge_type.startswith("CONDITION-") or condition:
                # Extract the condition expression
                condition_expr = condition if condition else edge_type.replace("CONDITION-", "").strip()
                
                # Store information for later probability calculation
                paths_with_conditions.append((node_id, condition_expr))
                logging.debug(f"Found conditional path to {node_id} with condition: {condition_expr}")
            else:
                # If no condition, this is a default path
                default_path = node_id
                logging.debug(f"Found default path to {node_id}")
        
        # Calculate path probabilities using triangular distribution
        path_probabilities = []
        
        if paths_with_conditions:
            # Each path gets a different section in a normalized distribution
            for i, (node_id, condition_expr) in enumerate(paths_with_conditions):
                # Calculate condition complexity
                condition_complexity = condition_expr.count("==") + condition_expr.count(">") + \
                                    condition_expr.count("<") + condition_expr.count("!=") + \
                                    condition_expr.count("AND") + condition_expr.count("OR") + 1
                
                # Base probability - simpler conditions get higher probability
                base_prob = 1.0 / (condition_complexity * 0.5)
                
                # Calculate triangular distribution parameters
                min_val = max(0.1, base_prob - 0.2)
                max_val = min(0.9, base_prob + 0.2)
                mode = base_prob
                
                # Sample from triangular distribution
                probability = local_random.triangular(min_val, max_val, mode)
                
                path_probabilities.append((node_id, probability))
                logging.debug(f"Path to {node_id} has probability {probability:.2f}")
        
        # Select path using weighted probability
        if path_probabilities:
            # Normalize probabilities to sum to 1.0
            total_prob = sum(prob for _, prob in path_probabilities)
            if total_prob > 0:
                normalized_probs = [(node_id, prob/total_prob) for node_id, prob in path_probabilities]
                
                # Use a random value to select based on cumulative probability
                r = local_random.random()
                cumulative_prob = 0
                selected_node = None
                
                for node_id, prob in normalized_probs:
                    cumulative_prob += prob
                    if r <= cumulative_prob:
                        selected_node = node_id
                        break
                
                if selected_node:
                    logging.info(f"Selected path to {selected_node} at exclusive gateway {gateway_id}")
                    return selected_node
        
        # If no path was selected by probability or if there are no conditional paths, use default or random
        if default_path:
            logging.info(f"Using default path to {default_path} at exclusive gateway {gateway_id}")
            return default_path
        
        # If no default path, select a random path
        selected_node = local_random.choice(next_nodes) if next_nodes else None
        logging.info(f"Selected random path to {selected_node} at exclusive gateway {gateway_id}")
        return selected_node
    
    def _handle_merge_node(self, token: Token, node_id: str, event_time: datetime) -> None:
        """
        Handle a merge node - synchronize split tokens.
        Enhanced to properly support inclusive gateways.
        
        Args:
            token: The token reaching the merge node
            node_id: ID of the merge node
            event_time: Current event time
        """
        # Skip if this is not a split token or not involved in a split
        if not token.is_split_token and not token.split_tokens:
            logging.debug(f"Token {token.token_id} at merge node {node_id} is not a split token")
            self._handle_non_gateway_completion(token, node_id, event_time)
            return
        
        # Get the merge node attributes
        merge_node_data = self.process_model.get_node(node_id)
        merge_gateway_type = None
        
        # Check if the merge node is a gateway with a specific type
        if merge_node_data:
            if '[Inclusive Gateway]' in str(merge_node_data.get('name', '')):
                merge_gateway_type = 'Inclusive'
            elif '[Parallel Gateway]' in str(merge_node_data.get('name', '')):
                merge_gateway_type = 'Parallel'
        
        if token.is_split_token:
            # This is a split token arriving at a merge node
            parent_token_id = token.parent_token_id
            
            # Skip if parent token doesn't exist
            if parent_token_id not in self.tokens:
                logging.warning(f"Parent token {parent_token_id} no longer exists for split token {token.token_id}")
                # Clean up this orphaned token
                if token.token_id in self.tokens:
                    del self.tokens[token.token_id]
                return
            
            parent_token = self.tokens[parent_token_id]
            
            # Special handling for inclusive gateway
            if merge_gateway_type == 'Inclusive':
                # For inclusive merge, we need to track which branches were actually taken
                # Mark this branch as completed in the parent token
                branch_id = token.current_node  # Using the current node as branch ID
                all_completed = parent_token.mark_split_complete(token.token_id, branch_id)
                
                logging.info(f"Split token {token.token_id} reached inclusive merge node {node_id}, " +
                            f"all taken branches completed: {all_completed}")
                
                # If all active branches are complete, continue with the parent token
                if all_completed:
                    logging.info(f"All active branches completed for token {parent_token_id}, continuing after merge")
                    
                    # Update parent token path to include merge node
                    if node_id not in parent_token.path:
                        parent_token.add_to_path(node_id)
                    
                    # Clear split tracking data
                    parent_token.active_gateway = None
                    parent_token.merge_node = None
                    
                    # Determine next nodes after the merge
                    next_nodes = self.process_model.get_next_nodes(node_id)
                    
                    if not next_nodes:
                        # Process is complete
                        parent_token.complete(event_time)
                        self.completed_tokens.append(parent_token)
                        del self.tokens[parent_token_id]
                        logging.info(f"Token {parent_token_id} completed process at merge node {node_id}")
                    else:
                        # Schedule events for next nodes after merge
                        for next_node_id in next_nodes:
                            heapq.heappush(
                                self.event_queue, 
                                Event(event_time, parent_token_id, next_node_id, "start")
                            )
                        logging.info(f"Token {parent_token_id} continuing to nodes {next_nodes} after merge")
                    
                    # Clean up all split tokens
                    for split_token_id in list(parent_token.split_tokens.keys()):
                        if split_token_id != parent_token_id and split_token_id in self.tokens:
                            del self.tokens[split_token_id]
                    
                    parent_token.split_tokens.clear()
                
                # Always remove this split token after it reaches the merge node
                if token.token_id != parent_token_id and token.token_id in self.tokens:
                    del self.tokens[token.token_id]
            
            else:
                # Standard parallel gateway or unknown gateway type
                # Mark this branch as completed in the parent token
                all_completed = parent_token.mark_split_complete(token.token_id, token.current_node)
                
                logging.info(f"Split token {token.token_id} reached merge node {node_id}, " +
                            f"all branches completed: {all_completed}")
                
                # If all branches are complete, continue with the parent token
                if all_completed:
                    logging.info(f"All branches completed for token {parent_token_id}, continuing after merge")
                    
                    # Update parent token path to include merge node
                    if node_id not in parent_token.path:
                        parent_token.add_to_path(node_id)
                    
                    # Clear split tracking data
                    parent_token.active_gateway = None
                    parent_token.merge_node = None
                    
                    # Determine next nodes after the merge
                    next_nodes = self.process_model.get_next_nodes(node_id)
                    
                    if not next_nodes:
                        # Process is complete
                        parent_token.complete(event_time)
                        self.completed_tokens.append(parent_token)
                        del self.tokens[parent_token_id]
                        logging.info(f"Token {parent_token_id} completed process at merge node {node_id}")
                    else:
                        # Schedule events for next nodes after merge
                        for next_node_id in next_nodes:
                            heapq.heappush(
                                self.event_queue, 
                                Event(event_time, parent_token_id, next_node_id, "start")
                            )
                        logging.info(f"Token {parent_token_id} continuing to nodes {next_nodes} after merge")
                    
                    # Clean up all split tokens except parent token
                    for split_token_id in list(parent_token.split_tokens.keys()):
                        if split_token_id != parent_token_id and split_token_id in self.tokens:
                            del self.tokens[split_token_id]
                    
                    parent_token.split_tokens.clear()
                
                # Remove this split token if it's not the parent token
                if token.token_id != parent_token_id and token.token_id in self.tokens:
                    del self.tokens[token.token_id]
        
        else:
            # This is a parent token that has split tokens
            # It should wait for all split tokens to complete
            logging.debug(f"Parent token {token.token_id} at merge node {node_id}, waiting for splits to complete")
            # Do nothing - the token will proceed when all splits complete
    
    def _handle_non_gateway_completion(self, token: Token, node_id: str, event_time: datetime) -> None:
        """
        Handle completion of a non-gateway node.
        
        Args:
            token: The token completing the node
            node_id: ID of the node
            event_time: Current event time
        """
        # Determine next nodes
        next_nodes = self.process_model.get_next_nodes(node_id)
        
        if not next_nodes or self.process_model.get_node(node_id).get("type") == "Stop":
            # Token has completed the process
            token.complete(event_time)
            
            # Calculate and log the processing time
            process_duration = (event_time - token.start_time).total_seconds() / 60
            
            logging.info(
                f"Token {token.token_id} completed process at node {node_id}. "
                f"Total processing time: {process_duration:.2f} minutes "
                f"(Wait time: {token.total_wait_time:.2f} minutes)"
            )
            
            self.completed_tokens.append(token)
            del self.tokens[token.token_id]
        else:
            # Schedule events for next nodes
            logging.debug(f"Token {token.token_id} proceeding to nodes {next_nodes}")
            for next_node_id in next_nodes:
                heapq.heappush(
                    self.event_queue, 
                    Event(event_time, token.token_id, next_node_id, "start")
                )
    
    def _find_merge_node(self, gateway_id: str) -> Optional[str]:
        """
        Find the merge node for a splitting gateway.
        
        Args:
            gateway_id: ID of the gateway node
            
        Returns:
            ID of the merge node if found, None otherwise
        """
        # Use the process model's find_merge_node method
        return self.process_model.find_merge_node(gateway_id)
    
    def _is_merge_node(self, node_id: str) -> bool:
        """
        Check if a node is a merge node for any gateway.
        
        Args:
            node_id: Node ID to check
            
        Returns:
            True if this is a merge node, False otherwise
        """
        return node_id in self.process_model.gateway_merge_nodes.values()
    
    def _calculate_task_duration(self, node: Dict[str, Any]) -> float:
        """
        Calculate task duration using a true triangular distribution to match Bizagi's behavior.
        
        Args:
            node: Node data containing duration parameters
            
        Returns:
            Duration in minutes
        """
        # Extract duration parameters
        min_time = 0
        avg_time = 0
        max_time = 0
        
        # Try different column naming patterns
        for min_key in ["min time", "min", "minimum time", "minimum"]:
            if min_key in node and node[min_key] is not None:
                try:
                    min_time = float(node[min_key])
                    break
                except (ValueError, TypeError):
                    continue
                    
        for avg_key in ["avg time", "avg", "average time", "average", "mean time", "mean"]:
            if avg_key in node and node[avg_key] is not None:
                try:
                    avg_time = float(node[avg_key])
                    break
                except (ValueError, TypeError):
                    continue
                    
        for max_key in ["max time", "max", "maximum time", "maximum"]:
            if max_key in node and node[max_key] is not None:
                try:
                    max_time = float(node[max_key])
                    break
                except (ValueError, TypeError):
                    continue
        
        # Use defaults if not found
        if min_time <= 0:
            min_time = 1  # Default minimum time of 1 minute
        if avg_time <= 0:
            avg_time = 5  # Default average time of 5 minutes
        if max_time <= 0:
            max_time = 10  # Default maximum time of 10 minutes
        
        # Ensure we have valid values for triangular distribution
        # Ensure min <= avg <= max
        min_time = min(min_time, avg_time, max_time)
        max_time = max(min_time, avg_time, max_time)
        avg_time = max(min_time, min(avg_time, max_time))
        
        # Implement a proper triangular distribution
        # This matches the mathematical definition used by Bizagi
        u = random.random()  # Uniform random number between 0 and 1
        
        # Standard triangular distribution formula
        if u <= (avg_time - min_time) / (max_time - min_time):
            # Sample from the left side of the triangle
            duration = min_time + math.sqrt(u * (max_time - min_time) * (avg_time - min_time))
        else:
            # Sample from the right side of the triangle
            duration = max_time - math.sqrt((1 - u) * (max_time - min_time) * (max_time - avg_time))
        
        # Ensure duration stays within bounds
        duration = max(min_time, min(duration, max_time))
        
        # Log parameters used for duration calculation
        node_name = node.get('name', 'Unknown')
        logging.debug(f"Task duration for {node_name}: {duration:.2f} minutes (min={min_time}, avg={avg_time}, max={max_time})")
        
        return duration
            
    def get_results(self) -> Dict[str, Any]:
        """
        Get the simulation results with improved process time calculation.
        
        Returns:
            Dictionary with simulation results
        """
        # Calculate resource utilization using proper simulation end date
        simulation_end = self.start_time + timedelta(days=self.simulation_days)
        resource_utilization = self.resource_manager.calculate_utilization(
            self.start_time, simulation_end
        )
        
        # Convert tokens to dictionaries for JSON serialization
        completed_token_dicts = []
        
        for token in self.completed_tokens:
            token_dict = token.to_dict()
            
            # Calculate "ideal" processing time (without wait times)
            process_duration = (token.end_time - token.start_time).total_seconds() / 60
            ideal_duration = process_duration - token.total_wait_time
            
            # Add this to the token data
            token_dict["ideal_duration"] = ideal_duration
            token_dict["wait_percentage"] = (token.total_wait_time / process_duration * 100) if process_duration > 0 else 0
            
            completed_token_dicts.append(token_dict)
        
        # Also include active tokens to see where they got stuck
        active_token_dicts = [token.to_dict() for token in self.tokens.values()]
        
        # Calculate process-level metrics similar to Bizagi
        process_metrics = self._calculate_process_metrics(completed_token_dicts)
        
        return {
            "activity_processing_times": self.activity_stats,
            "resource_utilization": resource_utilization,
            "total_tokens_started": self.total_tokens_started,
            "completed_tokens": completed_token_dicts,
            "active_tokens": active_token_dicts,
            "simulation_days": self.simulation_days,
            "process_metrics": process_metrics
        }

    def _calculate_process_metrics(self, completed_tokens: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate process-level metrics similar to Bizagi.
        
        Args:
            completed_tokens: List of completed token data
            
        Returns:
            Process metrics dictionary
        """
        if not completed_tokens:
            return {
                "avg_time": 0,
                "min_time": 0,
                "max_time": 0,
                "avg_wait_time": 0,
                "avg_ideal_time": 0
            }
        
        # Extract durations and wait times
        durations = [token.get("total_duration", 0) for token in completed_tokens]
        wait_times = [token.get("total_wait_time", 0) for token in completed_tokens]
        ideal_times = [token.get("ideal_duration", 0) for token in completed_tokens]
        
        # Calculate metrics
        avg_time = sum(durations) / len(durations)
        min_time = min(durations)
        max_time = max(durations)
        avg_wait_time = sum(wait_times) / len(wait_times)
        avg_ideal_time = sum(ideal_times) / len(ideal_times)
        
        return {
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "avg_wait_time": avg_wait_time,
            "avg_ideal_time": avg_ideal_time
        }
    