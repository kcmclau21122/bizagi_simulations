import sys
import os
import threading
import logging
import pandas as pd
import random
import heapq
import re
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_loader import DataLoader
from core.simulation_engine import SimulationEngine
from core.process_model import ProcessModel
from core.event import Event
from core.process_token import Token
from data.data_processing import ResultsExporter
from utils.config import ConfigManager
from utils.time_utils import day_of_week_to_index, is_work_time, advance_to_work_time
from data.xpdl_parser import parse_xpdl_with_metrics
from data.process_builder import ProcessModelBuilder

class SimulationRunner:
    """
    Manages the simulation process from configuration to results.
    Handles data loading, simulation setup, execution, and results processing.
    """
    
    def __init__(
        self, 
        config: ConfigManager, 
        progress_callback: Optional[Callable[[str], None]] = None,
        completion_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize the simulation runner.
        
        Args:
            config: Configuration manager with simulation settings
            progress_callback: Optional callback for reporting progress
            completion_callback: Optional callback for reporting completion
        """
        self.config = config
        self.progress_callback = progress_callback
        self.completion_callback = completion_callback
        self.running = False
        self.cancel_requested = False
        self.thread = None
        
    def run(self) -> None:
        """Run the simulation in a separate thread."""
        if self.running:
            logging.warning("Simulation already running")
            return
            
        self.running = True
        self.cancel_requested = False
        
        # Run in a separate thread
        self.thread = threading.Thread(target=self._run_simulation)
        self.thread.daemon = True
        self.thread.start()
        
    def _run_simulation(self) -> None:
        """Run the simulation process."""
        results = {}
        
        try:
            # Report progress
            self._update_progress("Loading data...")
            
            # Load XPDL file
            xpdl_path = self.config.get('xpdl_file_path', '')
            metrics_path = self.config.get('metrics_file_path', '')
            
            if not xpdl_path or not metrics_path:
                raise ValueError("Missing required file paths")
                
            # Load XPDL with proper error handling
            try:
                xpdl_root = DataLoader.load_xpdl(xpdl_path)
                self._update_progress(f"Successfully loaded XPDL from {xpdl_path}")
            except Exception as e:
                self._update_progress(f"Error loading XPDL: {str(e)}")
                raise ValueError(f"Failed to load XPDL file: {str(e)}")
            
            # Load metrics with proper error handling
            try:
                metrics_df = DataLoader.load_simulation_metrics(metrics_path)
                self._update_progress(f"Successfully loaded metrics from {metrics_path}")
                
                # Verify 'name' column exists in metrics
                if 'name' not in metrics_df.columns:
                    logging.warning("No 'name' column found in metrics file. Looking for alternatives.")
                    # Look for alternative column names
                    name_alternatives = ['activity', 'activity name', 'node', 'node name', 'id', 'activity id']
                    for alt in name_alternatives:
                        if alt in metrics_df.columns:
                            # Rename to 'name' for consistency
                            metrics_df.rename(columns={alt: 'name'}, inplace=True)
                            logging.info(f"Renamed column '{alt}' to 'name'")
                            self._update_progress(f"Using '{alt}' column as activity name identifier")
                            break
                            
                    # If still no 'name' column, create one with default values
                    if 'name' not in metrics_df.columns:
                        self._update_progress("No name column found in metrics. Creating default activity names.")
                        metrics_df['name'] = [f"Activity_{i}" for i in range(len(metrics_df))]
                        
            except Exception as e:
                self._update_progress(f"Error loading metrics: {str(e)}")
                raise ValueError(f"Failed to load metrics file: {str(e)}")
            
            # Validate metrics
            is_valid, errors = DataLoader.validate_simulation_metrics(metrics_df)
            if not is_valid:
                self._update_progress(f"Metrics validation errors: {'; '.join(errors)}")
                # Continue with warnings rather than failing
            
            # Preprocess metrics
            processed_metrics = DataLoader.preprocess_metrics(metrics_df)
            
            # Extract simulation parameters from UI settings
            simulation_params = {
                'max_arrival_count': self.config.get('token_count', 20),
                'min_interval': self.config.get('min_interval', 3.0),
                'avg_interval': self.config.get('avg_interval', 5.0),
                'max_interval': self.config.get('max_interval', 8.0)
            }
            
            # Report progress
            self._update_progress("Building process model from XPDL...")
            
            # Parse XPDL directly to JSON with error handling
            try:
                json_file_path = parse_xpdl_with_metrics(xpdl_path, metrics_path)
                self._update_progress(f"Generated process model JSON at {json_file_path}")
            except Exception as e:
                logging.error(f"Error parsing XPDL: {str(e)}", exc_info=True)
                self._update_progress(f"Error parsing XPDL: {str(e)}")
                raise ValueError(f"Failed to parse XPDL: {str(e)}")
            
            # Build the process model from JSON with error handling
            try:
                builder = ProcessModelBuilder()
                process_graph = builder.build_from_json(json_file_path, processed_metrics)
                self._update_progress(f"Built graph with {len(process_graph.nodes)} nodes and {len(process_graph.edges)} edges")
            except Exception as e:
                self._update_progress(f"Error building process graph: {str(e)}")
                raise ValueError(f"Failed to build process model: {str(e)}")
            
            # Check for empty graph early
            if len(process_graph.nodes) == 0:
                self._update_progress("ERROR: Generated process graph has no nodes. Check XPDL file format.")
                raise ValueError("Process model graph is empty. Check XPDL file format and content.")
            
            # Convert the NetworkX DiGraph to a ProcessModel object
            self._update_progress("Converting graph to ProcessModel...")
            process_model = ProcessModel()
            process_model.graph = process_graph
            
            # Add nodes and links from the graph
            for node_id, node_data in process_graph.nodes(data=True):
                process_model.add_node(node_id, node_data)
                
            # Add links (edges)
            for source, target, edge_data in process_graph.edges(data=True):
                link_data = {k: v for k, v in edge_data.items()}
                process_model.add_link(source, target, link_data)
                
            # Store node types for better later reference
            # This helps distinguish gateways from regular activities
            for node_id, node_data in process_model.nodes.items():
                # Check for gateway nodes
                if any(gateway in str(node_data.get('name', '')) for gateway in ['[Exclusive Gateway]', '[Inclusive Gateway]', '[Parallel Gateway]']):
                    gateway_type = next((g for g in ['[Exclusive Gateway]', '[Inclusive Gateway]', '[Parallel Gateway]'] 
                                        if g in str(node_data.get('name', ''))), None)
                    if gateway_type:
                        node_data['gateway'] = gateway_type
                
                # Check for node types
                node_type = node_data.get('type', '')
                if not node_type:
                    # Infer type from name
                    if 'Start' in str(node_data.get('name', '')):
                        node_data['type'] = 'Start'
                    elif 'Stop' in str(node_data.get('name', '')):
                        node_data['type'] = 'Stop'
                    else:
                        node_data['type'] = 'Task'
            
            # Debug information
            self._update_progress(f"Built process model with {len(process_model.nodes)} nodes and {len(process_model.links)} links")
            start_nodes = process_model.get_start_nodes()
            self._update_progress(f"Found {len(start_nodes)} start nodes: {start_nodes}")
            
            # Debug graph structure
            self._update_progress("Debugging graph structure...")
            for node in process_model.graph.nodes():
                successors = list(process_model.graph.successors(node))
                logging.debug(f"Node: {node} -> Successors: {successors}")
            
            # Validate process model
            self._validate_process_model(process_model)
            self._update_progress("Process model validation complete")
            
            # Pre-compute gateway merge nodes
            self._update_progress("Computing gateway relationships...")
            gateways = process_model.get_gateways()
            for gateway_id, gateway_data in gateways.items():
                merge_node = process_model.find_merge_node(gateway_id)
                if merge_node:
                    self._update_progress(f"Gateway {gateway_id} has merge node {merge_node}")
                    process_model.gateway_merge_nodes[gateway_id] = merge_node
            
            # Report progress
            self._update_progress("Setting up simulation...")
            
            # Setup simulation
            # Use current date for simulation start
            start_date = datetime.now().replace(hour=self.config.get('work_hours_start', 8), 
                                            minute=0, second=0, microsecond=0)
            
            # Calculate work days per week
            workdays = self.config.get('workdays', [True, True, True, True, True, False, False])
            work_days_per_week = sum(1 for day in workdays if day)
            
            # Get work hours
            work_hours_start = self.config.get('work_hours_start', 8)
            work_hours_end = self.config.get('work_hours_end', 17)
            work_hours_per_day = work_hours_end - work_hours_start
            
            # Get simulation days
            simulation_days = self.config.get('simulation_days', 5)
            
            # Create the simulation engine with the populated process model
            engine = SimulationEngine(
                process_model=process_model,
                start_time=start_date,
                work_days=work_days_per_week,
                work_hours_per_day=work_hours_per_day
            )
            
            # Store engine for optimization purposes
            self.engine = engine
            
            # Get target average processing time
            target_avg_time = self.config.get("target_avg_time", 0)
            
            # Schedule tokens using triangular distribution for intervals
            self._update_progress("Scheduling tokens...")
            
            max_tokens = simulation_params['max_arrival_count']
            min_interval = simulation_params['min_interval']
            avg_interval = simulation_params['avg_interval']
            max_interval = simulation_params['max_interval']
            
            # Calculate end date for simulation
            end_date = start_date + timedelta(days=simulation_days)
            
            # Use triangular distribution to generate token arrival times
            self._update_progress(f"Scheduling {max_tokens} tokens with triangular distribution...")
            
            tokens_scheduled = self._schedule_tokens_with_triangular_distribution(
                engine, max_tokens, min_interval, avg_interval, max_interval, end_date
            )
            
            self._update_progress(f"Scheduled {tokens_scheduled} tokens.")
            
            # Check if cancellation requested
            if self.cancel_requested:
                results = {"error": "Simulation cancelled by user"}
                self._on_complete(results)
                return
            
            # Check if target average time is specified and positive
            if target_avg_time > 0:
                # Use resource optimization
                self._update_progress(f"Target average processing time specified: {target_avg_time} minutes. Optimizing resources...")
                results = self._optimize_resources_for_target_time(
                    engine, 
                    simulation_days, 
                    target_avg_time,
                    process_model
                )
            else:
                # Run normal simulation
                self._update_progress("Running simulation...")
                results = engine.run_simulation(
                    simulation_days=simulation_days,
                    progress_callback=self._update_progress
                )
            
            self._update_progress("Simulation completed successfully")
            
            # Include transitions data for reporting
            try:
                transitions_df = pd.DataFrame(process_model.links)
                results["transitions_df"] = transitions_df
            except Exception as e:
                logging.warning(f"Could not create transitions dataframe: {str(e)}")
                # Create an empty dataframe as fallback
                results["transitions_df"] = pd.DataFrame(columns=['source', 'target', 'type'])
            
            # Process and export results
            self._update_progress("Processing results...")
            
            # Export results
            exporter = ResultsExporter()
            results_path = f"{os.path.splitext(os.path.basename(xpdl_path))[0]}_results.xlsx"
            exporter.export_to_excel(results, results_path, simulation_params)
            
            self._update_progress(f"Results exported to {results_path}")
            
        except Exception as e:
            logging.error(f"Simulation error: {str(e)}", exc_info=True)
            results = {"error": str(e)}
            
        finally:
            self.running = False
            self._on_complete(results)
            
    def _validate_process_model(self, process_model: ProcessModel) -> None:
        """
        Validate the process model before simulation, ensuring all nodes are properly connected.
        
        Args:
            process_model: The process model to validate
        """
        # Check if the graph is empty
        if len(process_model.graph.nodes) == 0:
            self._update_progress("ERROR: Process model graph is empty. Check XPDL file and parsing.")
            raise ValueError("Process model graph is empty. Please check your XPDL file.")
            
        # Check if start nodes have successors
        start_nodes = process_model.get_start_nodes()
        if not start_nodes:
            self._update_progress("WARNING: No explicit start nodes found in the process model.")
            # Try to infer start nodes by finding nodes with no incoming edges
            for node_id, node_data in process_model.nodes.items():
                if process_model.graph.in_degree(node_id) == 0:
                    # Mark this as a start node
                    node_data['type'] = 'Start'
                    self._update_progress(f"Inferred start node: {node_id}")
                    
            # Check again after inference
            start_nodes = process_model.get_start_nodes()
            if not start_nodes:
                self._update_progress("ERROR: Unable to identify any start nodes")
                raise ValueError("No start nodes found in the process model.")
        
        # Check for end nodes as well
        end_nodes = process_model.get_end_nodes()
        if not end_nodes:
            self._update_progress("WARNING: No explicit end nodes found in the process model.")
            # Try to infer end nodes by finding nodes with no outgoing edges
            for node_id, node_data in process_model.nodes.items():
                if process_model.graph.out_degree(node_id) == 0:
                    # Mark this as an end node
                    node_data['type'] = 'Stop'
                    self._update_progress(f"Inferred end node: {node_id}")
        
        # Validate connectivity: check that there's a path from at least one start node to at least one end node
        if start_nodes and end_nodes:
            has_complete_path = False
            for start_node in start_nodes:
                for end_node in end_nodes:
                    try:
                        path = nx.shortest_path(process_model.graph, start_node, end_node)
                        has_complete_path = True
                        self._update_progress(f"Found valid path from {start_node} to {end_node}: {path}")
                        break
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                if has_complete_path:
                    break
                    
            if not has_complete_path:
                self._update_progress("WARNING: No valid path from any start node to any end node!")
                # Attempt to fix: connect all start nodes to all end nodes directly
                for start_node in start_nodes:
                    for end_node in end_nodes:
                        process_model.add_link(start_node, end_node, {"type": "NORMAL"})
                        self._update_progress(f"Added fallback connection from {start_node} to {end_node}")
        
        # Check successors for each start node
        for start_node in start_nodes:
            next_nodes = process_model.get_next_nodes(start_node)
            if not next_nodes:
                # Try to find the issue
                all_edges = list(process_model.graph.edges(data=True))
                self._update_progress(f"WARNING: Start node {start_node} has no successors. Available edges: {all_edges}")
                
                # Fix: Try to find non-start, non-end activities to connect to
                potential_targets = []
                for node_id, node_data in process_model.nodes.items():
                    if node_id != start_node and node_data.get('type') not in ['Start', 'Stop']:
                        potential_targets.append(node_id)
                
                if potential_targets:
                    # Sort potential targets to get a deterministic order
                    potential_targets.sort()
                    
                    # Connect to the first activity
                    first_activity = potential_targets[0]
                    process_model.add_link(start_node, first_activity, {"type": "NORMAL"})
                    self._update_progress(f"Added fallback connection from {start_node} to {first_activity}")
                elif end_nodes:
                    # If no intermediate nodes, connect directly to end node
                    end_node = end_nodes[0]
                    process_model.add_link(start_node, end_node, {"type": "NORMAL"})
                    self._update_progress(f"Added fallback connection from {start_node} to {end_node}")
            
            # Check successors again after fixes
            next_nodes = process_model.get_next_nodes(start_node)
            self._update_progress(f"After validation, start node {start_node} has successors: {next_nodes}")
                
        # Verify all nodes have proper connections
        orphaned_nodes = []
        for node_id in process_model.graph.nodes():
            in_degree = process_model.graph.in_degree(node_id)
            out_degree = process_model.graph.out_degree(node_id)
            
            # Skip start and end nodes
            node_data = process_model.get_node(node_id)
            if node_data.get('type') == 'Start' or node_id in start_nodes:
                continue
            if node_data.get('type') == 'Stop' or node_id in end_nodes:
                continue
                
            # Check if node is disconnected (no incoming or outgoing edges)
            if in_degree == 0 and out_degree == 0:
                orphaned_nodes.append(node_id)
                self._update_progress(f"WARNING: Node {node_id} is disconnected (no edges)")
                
        # Try to fix orphaned nodes by connecting them to the process
        if orphaned_nodes:
            self._update_progress(f"Found {len(orphaned_nodes)} orphaned nodes: {orphaned_nodes}")
            
            # Connect orphaned nodes in a chain
            if len(orphaned_nodes) > 1:
                for i in range(len(orphaned_nodes) - 1):
                    process_model.add_link(orphaned_nodes[i], orphaned_nodes[i+1], {"type": "NORMAL"})
                    self._update_progress(f"Connected orphaned nodes: {orphaned_nodes[i]} -> {orphaned_nodes[i+1]}")
                    
            # Connect first orphaned node to a start node
            if start_nodes and orphaned_nodes:
                process_model.add_link(start_nodes[0], orphaned_nodes[0], {"type": "NORMAL"})
                self._update_progress(f"Connected start node to orphaned chain: {start_nodes[0]} -> {orphaned_nodes[0]}")
                
            # Connect last orphaned node to an end node
            if end_nodes and orphaned_nodes:
                process_model.add_link(orphaned_nodes[-1], end_nodes[0], {"type": "NORMAL"})
                self._update_progress(f"Connected orphaned chain to end node: {orphaned_nodes[-1]} -> {end_nodes[0]}")
                
        # Check for disconnected components (only if graph is not empty)
        if len(process_model.graph.nodes) > 0:
            try:
                if not nx.is_weakly_connected(process_model.graph):
                    self._update_progress("WARNING: Process model has disconnected components!")
                    # Find disconnected components
                    components = list(nx.weakly_connected_components(process_model.graph))
                    self._update_progress(f"Found {len(components)} disconnected components")
                    
                    # Try to connect components together
                    if len(components) > 1:
                        for i in range(len(components) - 1):
                            comp1 = list(components[i])
                            comp2 = list(components[i+1])
                            if comp1 and comp2:
                                # Connect a node from comp1 to a node from comp2
                                process_model.add_link(comp1[0], comp2[0], {"type": "NORMAL"})
                                self._update_progress(f"Connected components: {comp1[0]} -> {comp2[0]}")
            except Exception as e:
                self._update_progress(f"WARNING: Could not check graph connectivity: {str(e)}")
            
    def _schedule_tokens_with_triangular_distribution(
        self, 
        engine: SimulationEngine, 
        max_tokens: int,
        min_interval: float,
        avg_interval: float,
        max_interval: float,
        end_date: datetime
    ) -> int:
        """
        Schedule tokens using triangular distribution for arrival intervals.
        
        Args:
            engine: Simulation engine
            max_tokens: Maximum number of tokens to schedule
            min_interval: Minimum interval between tokens (minutes)
            avg_interval: Average interval between tokens (minutes) - the mode of triangular distribution
            max_interval: Maximum interval between tokens (minutes)
            end_date: End date for simulation
            
        Returns:
            Number of tokens actually scheduled
        """
        # Get start nodes from the process model
        start_nodes = engine.process_model.get_start_nodes()
        if not start_nodes:
            # Debug information before raising error
            logging.error(f"No start nodes found. Process model has {len(engine.process_model.nodes)} nodes.")
            node_types = {}
            for node_id, node_data in engine.process_model.nodes.items():
                node_type = node_data.get('type', 'Unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            logging.error(f"Node types: {node_types}")
            
            # Try to identify any potential start nodes
            potential_starts = []
            for node_id, node_data in engine.process_model.nodes.items():
                # Check for nodes with no incoming edges
                if engine.process_model.graph.in_degree(node_id) == 0:
                    potential_starts.append(node_id)
                # Or check for nodes with 'Start' in their name
                elif 'Start' in str(node_data.get('name', '')):
                    potential_starts.append(node_id)
            
            if potential_starts:
                logging.warning(f"No explicit start nodes found, but found {len(potential_starts)} potential start nodes: {potential_starts}")
                # Use these as start nodes
                start_nodes = potential_starts
            else:
                raise ValueError("No start nodes found in the process model.")
                
        # Make sure simulation starts at the first work hour (like Monday at 07:00)
        current_time = engine.start_time
        if not is_work_time(current_time, engine.start_time, engine.work_days, engine.work_hours_per_day):
            current_time = advance_to_work_time(current_time, engine.start_time, engine.work_days, engine.work_hours_per_day)
            
        token_count = 0
        
        # Calculate total available time slots based on work hours
        total_minutes = 0
        temp_time = current_time
        
        # Count actual available working minutes in the simulation period
        while temp_time <= end_date:
            if is_work_time(temp_time, engine.start_time, engine.work_days, engine.work_hours_per_day):
                total_minutes += 1
            temp_time += timedelta(minutes=1)
            # For efficiency, skip to next work period if outside work hours
            if not is_work_time(temp_time, engine.start_time, engine.work_days, engine.work_hours_per_day):
                temp_time = advance_to_work_time(temp_time, engine.start_time, engine.work_days, engine.work_hours_per_day)
                if temp_time > end_date:
                    break
                    
        # Ensure we don't try to schedule more tokens than time allows
        # Use triangular distribution mean for interval estimation
        # Mean of triangular = (min + max + mode) / 3
        estimated_avg_interval = (min_interval + avg_interval + max_interval) / 3
        max_possible_tokens = max(1, int(total_minutes / estimated_avg_interval))
        target_tokens = min(max_tokens, max_possible_tokens)
        
        logging.info(f"Scheduling up to {target_tokens} tokens over {total_minutes} available minutes")
        self._update_progress(f"Scheduling up to {target_tokens} tokens over {total_minutes} available minutes")
        
        # Use random seed for reproducibility
        random_seed = self.config.get('random_seed', 42)
        random.seed(random_seed)
        
        # Generate all token arrival times in advance using triangular distribution
        arrival_times = []
        temp_time = current_time
        
        for _ in range(target_tokens):
            # Ensure we're within work hours
            if not is_work_time(temp_time, engine.start_time, engine.work_days, engine.work_hours_per_day):
                temp_time = advance_to_work_time(temp_time, engine.start_time, engine.work_days, engine.work_hours_per_day)
                if temp_time > end_date:
                    break
                    
            # Add this arrival time
            arrival_times.append(temp_time)
            
            # Calculate next interval using triangular distribution
            interval = random.triangular(min_interval, max_interval, avg_interval)
            temp_time += timedelta(minutes=interval)
            
            # If we've gone past the end date, stop scheduling
            if temp_time > end_date:
                break
        
        # Sort arrival times chronologically (should already be in order, but just to be safe)
        arrival_times.sort()
        
        # Now schedule tokens at these pre-calculated times
        for i, arrival_time in enumerate(arrival_times):
            token_id = f"Token-{i + 1}"
            start_node = random.choice(start_nodes)
            
            token = Token(token_id, arrival_time, start_node)
            engine.tokens[token_id] = token
            
            # Schedule start event
            heapq.heappush(
                engine.event_queue, 
                Event(arrival_time, token_id, start_node, "start")
            )
            
            if i % 10 == 0 or i == len(arrival_times) - 1:
                logging.info(f"Scheduled {i+1} of {len(arrival_times)} tokens")
                self._update_progress(f"Scheduled {i+1} of {len(arrival_times)} tokens")
            
            token_count += 1
            engine.total_tokens_started += 1
        
        logging.info(f"Successfully scheduled {token_count} tokens with triangular distribution")
        self._update_progress(f"Successfully scheduled {token_count} tokens with triangular distribution")
        return token_count
            
    def _update_progress(self, message: str) -> None:
        """Update progress with a status message."""
        logging.info(message)
        if self.progress_callback:
            self.progress_callback(message)
            
    def _on_complete(self, results: Dict[str, Any]) -> None:
        """Handle simulation completion."""
        if self.completion_callback:
            self.completion_callback(results)
            
    def cancel(self) -> None:
        """Cancel the running simulation."""
        if self.running:
            self.cancel_requested = True
            self._update_progress("Cancelling simulation...")
            
    def is_running(self) -> bool:
        """Check if the simulation is running."""
        return self.running

    def _optimize_resources_for_target_time(
        self, 
        engine: SimulationEngine, 
        simulation_days: int, 
        target_avg_time: float,
        process_model: ProcessModel
    ) -> Dict[str, Any]:
        """
        Optimize resources to achieve the target average processing time.
        
        Args:
            engine: Simulation engine
            simulation_days: Number of days to simulate
            target_avg_time: Target average process time in minutes
            process_model: Process model
            
        Returns:
            Simulation results after optimization
        """
        self._update_progress(f"Optimizing resources to meet target average processing time: {target_avg_time} minutes")
        
        # Store original resource counts
        original_resources = {}
        for resource_id, count in engine.resource_manager.available_resources.items():
            original_resources[resource_id] = count
        
        # Run initial simulation
        initial_results = engine.run_simulation(
            simulation_days=simulation_days,
            progress_callback=self._update_progress
        )
        
        # Check if completed tokens are available
        if "completed_tokens" not in initial_results or not initial_results["completed_tokens"]:
            self._update_progress("No completed tokens in simulation. Cannot optimize resources.")
            return initial_results
        
        # Calculate current average processing time
        completed_tokens = initial_results["completed_tokens"]
        process_durations = [
            (token['end_time'] - token['start_time']).total_seconds() / 60 
            for token in completed_tokens
        ]
        current_avg_time = sum(process_durations) / len(process_durations)
        
        self._update_progress(f"Initial average processing time: {current_avg_time:.2f} minutes")
        
        # If current time is already below target, we're done
        if current_avg_time <= target_avg_time:
            self._update_progress("Target already met with current resources.")
            return initial_results
        
        # Track resource utilization to identify bottlenecks
        resource_utilization = initial_results.get("resource_utilization", {})
        
        # Set maximum optimization iterations to avoid infinite loops
        max_iterations = 10
        iterations = 0
        
        # Store best results so far
        best_results = initial_results
        best_avg_time = current_avg_time
        
        # Track resources we've already increased
        increased_resources = set()
        
        while current_avg_time > target_avg_time and iterations < max_iterations:
            iterations += 1
            
            # Find the most utilized resource that hasn't been increased yet
            bottleneck_resource = None
            max_utilization = 0
            
            for resource, utilization in resource_utilization.items():
                if resource not in increased_resources and utilization > max_utilization:
                    max_utilization = utilization
                    bottleneck_resource = resource
            
            if not bottleneck_resource:
                self._update_progress("No more resources to optimize.")
                break
                
            # Get original count
            original_count = engine.resource_manager.get_available_count(bottleneck_resource)
            
            # Increase resource count
            new_count = original_count + 1
            engine.resource_manager.set_available_resources(bottleneck_resource, new_count)
            
            self._update_progress(f"Iteration {iterations}: Increasing {bottleneck_resource} from {original_count} to {new_count}")
            
            # Mark this resource as increased
            increased_resources.add(bottleneck_resource)
            
            # Reset the engine state for a new simulation run
            engine.event_queue = []
            engine.tokens = {}
            engine.completed_tokens = []
            engine.total_tokens_started = 0
            
            # Reschedule tokens
            max_tokens = self.config.get('token_count', 20)
            min_interval = self.config.get('min_interval', 3.0)
            avg_interval = self.config.get('avg_interval', 5.0)
            max_interval = self.config.get('max_interval', 8.0)
            end_date = engine.start_time + timedelta(days=simulation_days)
            
            tokens_scheduled = self._schedule_tokens_with_triangular_distribution(
                engine, max_tokens, min_interval, avg_interval, max_interval, end_date
            )
            
            self._update_progress(f"Rescheduled {tokens_scheduled} tokens for next simulation run")
            
            # Run simulation with updated resources
            results = engine.run_simulation(
                simulation_days=simulation_days,
                progress_callback=self._update_progress
            )
            
            # Check if completed tokens are available
            if "completed_tokens" not in results or not results["completed_tokens"]:
                self._update_progress("No completed tokens in simulation. Reverting last change.")
                engine.resource_manager.set_available_resources(bottleneck_resource, original_count)
                continue
            
            # Calculate new average processing time
            completed_tokens = results["completed_tokens"]
            process_durations = [
                (token['end_time'] - token['start_time']).total_seconds() / 60 
                for token in completed_tokens
            ]
            current_avg_time = sum(process_durations) / len(process_durations)
            
            self._update_progress(f"New average processing time: {current_avg_time:.2f} minutes")
            
            # Update best results if this is an improvement
            if current_avg_time < best_avg_time:
                best_results = results
                best_avg_time = current_avg_time
            
            # Update resource utilization for next iteration
            resource_utilization = results.get("resource_utilization", {})
        
        # Check if target was met
        if current_avg_time <= target_avg_time:
            self._update_progress(f"Target met! Final average processing time: {current_avg_time:.2f} minutes")
        else:
            self._update_progress(f"Could not meet target time. Best achieved: {best_avg_time:.2f} minutes")
            # Use best results if target wasn't met
            results = best_results
        
        # Add optimization summary to results
        results["optimization_summary"] = {
            "target_time": target_avg_time,
            "achieved_time": best_avg_time,
            "optimization_iterations": iterations,
            "optimized_resources": list(increased_resources)
        }
        
        return results
