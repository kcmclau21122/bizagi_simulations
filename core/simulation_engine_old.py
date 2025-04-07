import sys
import os
# Add project root to sys.path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import heapq
import logging
import random
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Callable, Set

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
        self._verify_resources()
        
        # Statistics collection
        self.activity_stats = defaultdict(lambda: {
            "wait_times": [], 
            "durations": [], 
            "tokens_started": 0, 
            "tokens_completed": 0
        })
        
        self.total_tokens_started = 0
        
    def _standardize_gateway_type(self, gateway_type: str) -> str:
        """
        Standardize gateway type names to handle various naming conventions.
        
        Args:
            gateway_type: The original gateway type string
            
        Returns:
            Standardized gateway type: "EXCLUSIVE", "INCLUSIVE", "PARALLEL", or "UNKNOWN"
        """
        if not gateway_type:
            return "UNKNOWN"
            
        gateway_type = str(gateway_type).upper()
        
        if "EXCLUSIVE" in gateway_type or "XOR" in gateway_type:
            return "EXCLUSIVE"
        elif "INCLUSIVE" in gateway_type or "OR" in gateway_type:
            return "INCLUSIVE"
        elif "PARALLEL" in gateway_type or "AND" in gateway_type:
            return "PARALLEL"
        else:
            logging.warning(f"Unrecognized gateway type: {gateway_type}, treating as UNKNOWN")
            return "UNKNOWN"
        
    def _initialize_resources(self) -> None:
        """Initialize resources from the process model."""
        resources = self.process_model.get_all_resources()
        
        # Ensure we have at least default resources if none defined
        if not resources:
            logging.warning("No resources found in process model! Adding defaults...")
            resources = {"Tech": 1, "Certifier": 1}
            
        for resource_id, count in resources.items():
            # Ensure count is valid
            try:
                count = int(count)
                if count <= 0:
                    logging.warning(f"Invalid resource count for {resource_id}: {count}, setting to 1")
                    count = 1
            except (ValueError, TypeError):
                logging.warning(f"Non-numeric resource count for {resource_id}: {count}, setting to 1")
                count = 1
                
            logging.info(f"Initializing resource {resource_id} with count {count}")
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
        
        # Run diagnostics before simulation
        self.run_diagnostics()
    
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
            # Try to allocate the resource
            if not self.resource_manager.allocate_resource(resource, token_id, node_id, event_time):
                # Resource not available, add to wait queue (already done in allocate_resource)
                token.start_waiting(event_time)
                logging.info(
                    f"Token {token_id} waiting for resource '{resource}' at {event_time}"
                )
                return
            else:
                # Successfully allocated resource
                logging.info(f"Token {token_id} acquired resource '{resource}' at {event_time}")
        
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

    def _determine_gateway_path(self, gateway_node: Dict[str, Any], gateway_id: str) -> Optional[int]:
        """
        Determine which path to take at a decision gateway.
        For XOR gateways, randomly select a path based on configured probabilities.
        
        Args:
            gateway_node: The gateway node data
            gateway_id: The ID of the gateway node
            
        Returns:
            The chosen path index or None if no valid decision could be made
        """
        # Get the configured probabilities (e.g., "yes" and "no" for a decision)
        yes_prob = float(gateway_node.get("yes", 50))
        no_prob = float(gateway_node.get("no", 50))
        
        # Normalize probabilities
        total = yes_prob + no_prob
        if total <= 0:
            return None
            
        yes_prob = yes_prob / total
        
        # Make a random decision
        rand_value = random.random()
        decision = 0 if rand_value < yes_prob else 1
        
        logging.info(f"Gateway {gateway_id} decision: probabilities yes={yes_prob:.2f}/no={(1-yes_prob):.2f}, " 
                     f"random value={rand_value:.4f}, chose path {decision} " 
                     f"({'yes' if decision == 0 else 'no'})")
        
        return decision

    def _determine_inclusive_gateway_paths(self, gateway_node: Dict[str, Any], 
                                          gateway_id: str, 
                                          outgoing_paths: List[str]) -> List[str]:
        """
        Determine which paths to take at an inclusive gateway.
        For inclusive gateways, multiple paths can be chosen based on conditions.
        
        Args:
            gateway_node: The gateway node data
            gateway_id: The ID of the gateway node
            outgoing_paths: List of outgoing path node IDs
            
        Returns:
            List of chosen path node IDs
        """
        chosen_paths = []
        
        # Check if there are explicit probabilities for each path
        path_probs = {}
        
        # Try to get probability for each path
        for i, path in enumerate(outgoing_paths):
            prob_key = f"path{i+1}"
            if prob_key in gateway_node:
                try:
                    path_probs[path] = float(gateway_node[prob_key])
                except (ValueError, TypeError):
                    # Default to 50% if invalid value
                    path_probs[path] = 50.0
            else:
                # No explicit probability, use default
                path_probs[path] = 50.0
        
        # If no path probabilities found, try yes/no probabilities
        if not path_probs and len(outgoing_paths) >= 2:
            yes_prob = float(gateway_node.get("yes", 50))
            no_prob = float(gateway_node.get("no", 50))
            
            if len(outgoing_paths) == 2:
                path_probs[outgoing_paths[0]] = yes_prob
                path_probs[outgoing_paths[1]] = no_prob
            else:
                # Distribute remaining probability among other paths
                remaining_prob = no_prob / (len(outgoing_paths) - 1) if len(outgoing_paths) > 1 else 0
                path_probs[outgoing_paths[0]] = yes_prob
                for i in range(1, len(outgoing_paths)):
                    path_probs[outgoing_paths[i]] = remaining_prob
        
        # Make decision for each path independently
        for path, prob in path_probs.items():
            # Normalize probability to 0-100 range
            prob = min(max(prob, 0), 100)
            
            # Decide whether to take this path
            rand_value = random.random() * 100
            take_path = rand_value < prob
            logging.info(f"Inclusive gateway {gateway_id} path {path} decision: "
                         f"probability={prob:.2f}%, random value={rand_value:.2f}, "
                         f"{'taking' if take_path else 'skipping'} path")
                         
            if take_path:
                chosen_paths.append(path)
        
        # If no paths were chosen, choose at least one path to ensure progress
        if not chosen_paths and outgoing_paths:
            logging.warning(f"No paths chosen at inclusive gateway {gateway_id}, selecting first path by default")
            chosen_paths.append(outgoing_paths[0])
        
        return chosen_paths
        
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
            logging.warning(f"Token {token_id} not found for end event at {node_id}")
            return
            
        token = self.tokens[token_id]
        node = self.process_model.get_node(node_id)
        
        # Log detailed information
        logging.info(f"Processing end event: Token {token_id} at node {node_id} ({node.get('name', 'Unknown')})")
        
        token.complete_task(node_id)  # Mark the task as completed
        
        logging.info(f"Token {token_id} completed task '{node_id}' at {event_time}.")
        
        # Update activity statistics
        self.activity_stats[node_id]["tokens_completed"] += 1
        
        # Release resources
        resource = node.get("resource")
        if resource:
            logging.info(f"Releasing resource '{resource}' for token {token_id}")
            # Get next token from wait queue if any
            next_token_info = self.resource_manager.release_resource(resource, event_time)
            if next_token_info:
                next_token_id, next_node_id, queued_time = next_token_info
                # Schedule start event for the token that was waiting
                heapq.heappush(
                    self.event_queue, 
                    Event(event_time, next_token_id, next_node_id, "start")
                )
                logging.info(
                    f"Token {next_token_id} released from wait queue for '{resource}'"
                )
        
        # Determine next nodes
        next_nodes = self.process_model.get_next_nodes(node_id)
        logging.info(f"Next nodes for {node_id}: {next_nodes}")
        
        if not next_nodes or node.get("type") == "Stop":
            # Token has completed the process
            token.complete(event_time)
            logging.info(
                f"Token {token_id} has completed the entire process at {event_time}."
            )
            self.completed_tokens.append(token)
            del self.tokens[token_id]
        else:
            # Check if the current node is a gateway or leads to a gateway
            self._process_outgoing_nodes(token_id, node_id, next_nodes, event_time)

    def _process_outgoing_nodes(self, token_id: str, node_id: str, 
                               next_nodes: List[str], event_time: datetime) -> None:
        """
        Process outgoing nodes from a task or gateway.
        
        Args:
            token_id: The ID of the token
            node_id: The ID of the current node
            next_nodes: List of next node IDs
            event_time: Current event time
        """
        # Flag to track if we've found a gateway
        is_gateway = False
        
        for next_node_id in next_nodes:
            next_node = self.process_model.get_node(next_node_id)
            gateway_type = next_node.get("gateway")
            
            # Handle gateway nodes
            if gateway_type:
                is_gateway = True
                std_gateway_type = self._standardize_gateway_type(gateway_type)
                logging.info(f"Processing gateway {next_node_id} of type {gateway_type} (standardized: {std_gateway_type})")
                
                if std_gateway_type == "EXCLUSIVE":
                    # Handle exclusive gateway (XOR) - only one path can be taken
                    decision = self._determine_gateway_path(next_node, next_node_id)
                    
                    if decision is not None:
                        gateway_next_nodes = self.process_model.get_next_nodes(next_node_id)
                        
                        if gateway_next_nodes and decision < len(gateway_next_nodes):
                            chosen_path = gateway_next_nodes[decision]
                            logging.info(f"Exclusive gateway {next_node_id} chose path {decision}: {chosen_path}")
                            
                            heapq.heappush(
                                self.event_queue,
                                Event(event_time, token_id, chosen_path, "start")
                            )
                        else:
                            logging.warning(f"Invalid gateway path selection for {next_node_id}: {decision}")
                
                elif std_gateway_type == "INCLUSIVE":
                    # Handle inclusive gateway (OR) - one or more paths can be taken
                    outgoing_paths = self.process_model.get_next_nodes(next_node_id)
                    chosen_paths = self._determine_inclusive_gateway_paths(next_node, next_node_id, outgoing_paths)
                    
                    logging.info(f"Inclusive gateway {next_node_id} chose paths: {chosen_paths}")
                    for path_node_id in chosen_paths:
                        heapq.heappush(
                            self.event_queue,
                            Event(event_time, token_id, path_node_id, "start")
                        )
                
                elif std_gateway_type == "PARALLEL":
                    # Handle parallel gateway (AND) - all paths must be taken
                    for path_node_id in self.process_model.get_next_nodes(next_node_id):
                        logging.info(f"Scheduling parallel path {path_node_id} from gateway {next_node_id}")
                        heapq.heappush(
                            self.event_queue,
                            Event(event_time, token_id, path_node_id, "start")
                        )
                
                else:
                    # Handle unknown gateway types (default to parallel behavior)
                    logging.warning(f"Unknown gateway type {gateway_type} at {next_node_id}, defaulting to parallel behavior")
                    for path_node_id in self.process_model.get_next_nodes(next_node_id):
                        heapq.heappush(
                            self.event_queue,
                            Event(event_time, token_id, path_node_id, "start")
                        )
            
        # If no gateways were encountered, just schedule all next nodes directly
        if not is_gateway:
            for next_node_id in next_nodes:
                logging.info(f"Scheduling direct next node {next_node_id}")
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
        # Extract time values with better error handling and defaults
        try:
            min_time = float(node.get("min time", 0) or 0)
            avg_time = float(node.get("avg time", 0) or 0)  # Mode of the triangular distribution
            max_time = float(node.get("max time", 0) or 0)
            
            # Log task duration parameters for debugging
            logging.debug(f"Task {node.get('name', 'Unknown')}: min={min_time}, avg={avg_time}, max={max_time}")
            
            if avg_time > 0:
                # Ensure we have valid min/max values
                if min_time <= 0:
                    min_time = max(0.1, avg_time * 0.8)
                if max_time <= 0:
                    max_time = avg_time * 1.2
                    
                # Ensure min <= avg <= max
                min_time = min(min_time, avg_time)
                max_time = max(max_time, avg_time)
                
                return random.triangular(min_time, max_time, avg_time)
            else:
                return 0  # No processing time
        except (ValueError, TypeError) as e:
            logging.warning(f"Error calculating task duration: {e}. Using default 0.")
            return 0
            
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
        
    def _verify_resources(self) -> None:
        """
        Verifies that all necessary resources are properly initialized.
        Logs warnings for any potential issues.
        """
        resources = self.process_model.get_all_resources()
        logging.info(f"Available resources in model: {resources}")
        
        # Check for specific resources that must be present
        required_resources = ["Tech", "Certifier"]
        for resource in required_resources:
            if resource not in resources:
                logging.warning(f"Required resource '{resource}' not found in the process model!")
            else:
                count = resources[resource]
                if count <= 0:
                    logging.warning(f"Resource '{resource}' has invalid count: {count}")
                    
        # Verify resources are properly set in the resource manager
        for resource_id, count in resources.items():
            actual_count = self.resource_manager.get_available_count(resource_id)
            if actual_count != count:
                logging.warning(
                    f"Resource manager has incorrect count for '{resource_id}': "
                    f"expected {count}, got {actual_count}"
                )

    def run_diagnostics(self) -> None:
        """
        Run pre-simulation diagnostics to detect potential issues.
        """
        logging.info("Running pre-simulation diagnostics...")
        
        # Check process model for basic structural issues
        start_nodes = self.process_model.get_start_nodes()
        end_nodes = self.process_model.get_end_nodes()
        
        logging.info(f"Start nodes: {start_nodes}")
        logging.info(f"End nodes: {end_nodes}")
        
        if not start_nodes:
            logging.error("No start nodes found in process model!")
        
        if not end_nodes:
            logging.error("No end nodes found in process model!")
        
        # Check task assignments
        resource_tasks = self.process_model.debug_resource_assignment()
        
        # Check and log gateway nodes and connections
        self._analyze_gateways()
        
        # Check resource allocation
        for resource_id, count in self.resource_manager.available_resources.items():
            logging.info(f"Resource {resource_id} has {count} instances available")
            
            if resource_id not in resource_tasks:
                logging.warning(f"Resource {resource_id} is not assigned to any tasks!")
        
        # Check if there are any disconnected components
        try:
            paths = self.process_model.get_all_paths()
            if not paths:
                logging.error("No valid paths found through the process model!")
            else:
                logging.info(f"Found {len(paths)} possible paths through the process")
        except Exception as e:
            logging.error(f"Error analyzing process paths: {e}")
            
    def _analyze_gateways(self) -> None:
        """
        Analyze all gateways in the process model to detect potential issues.
        Logs details about gateway types and connections.
        """
        # Get all nodes with gateway attribute
        gateway_connections = {}
        
        for node_id, node_data in self.process_model.nodes.items():
            gateway_type = node_data.get("gateway")
            if gateway_type:
                std_type = self._standardize_gateway_type(gateway_type)
                next_nodes = self.process_model.get_next_nodes(node_id)
                
                gateway_connections[gateway_type] = gateway_connections.get(gateway_type, {})
                gateway_connections[gateway_type][node_id] = next_nodes
                
                logging.info(f"Gateway {node_id} of type {gateway_type} (standardized: {std_type}) "
                            f"has outgoing paths: {next_nodes}")
                
                # Check for potential issues with gateway connections
                if std_type == "EXCLUSIVE" and len(next_nodes) <= 1:
                    logging.warning(f"Exclusive gateway {node_id} has {len(next_nodes)} outgoing paths, expected at least 2")
                elif std_type == "PARALLEL" and len(next_nodes) <= 1:
                    logging.warning(f"Parallel gateway {node_id} has {len(next_nodes)} outgoing paths, expected at least 2")
                    
        # Log summary of gateway connections
        logging.info(f"Gateway connections: {gateway_connections}")
