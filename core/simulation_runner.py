import sys
import os
import threading
import logging
import pandas as pd
import random
import heapq
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable

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
from data.xpdl_parser import parse_xpdl_to_sequences
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
                
            # Load XPDL
            xpdl_root = DataLoader.load_xpdl(xpdl_path)
            
            # Load metrics
            metrics_df = DataLoader.load_simulation_metrics(metrics_path)
            
            # Validate metrics
            is_valid, errors = DataLoader.validate_simulation_metrics(metrics_df)
            if not is_valid:
                raise ValueError(f"Invalid metrics file: {'; '.join(errors)}")
                
            # Preprocess metrics
            processed_metrics = DataLoader.preprocess_metrics(metrics_df)
            
            # Extract simulation parameters
            # Instead of getting parameters from metrics, use the ones set in the UI
            # simulation_params = DataLoader.get_simulation_parameters(processed_metrics)
            simulation_params = {
                'max_arrival_count': self.config.get('token_count', 20),
                'min_interval': self.config.get('min_interval', 3.0),
                'avg_interval': self.config.get('avg_interval', 5.0),
                'max_interval': self.config.get('max_interval', 8.0)
            }
            
            # Report progress
            self._update_progress("Building process model from XPDL...")
            
            # Parse XPDL to sequences file
            base_filename = os.path.splitext(os.path.basename(xpdl_path))[0]
            sequence_file_path = f"{base_filename}_sequences.txt"
            parse_xpdl_to_sequences(xpdl_path, sequence_file_path)
            
            # Build the process model from sequences
            builder = ProcessModelBuilder()
            process_graph = builder.build_from_sequences(sequence_file_path, processed_metrics)
            
            # Convert the NetworkX DiGraph to a ProcessModel object
            self._update_progress("Converting graph to ProcessModel...")
            process_model = ProcessModel()
            process_model.graph = process_graph
            
            # Add nodes and links from the graph
            for node_id, node_data in process_graph.nodes(data=True):
                process_model.nodes[node_id] = node_data
                
            # Add links (edges)
            for source, target, edge_data in process_graph.edges(data=True):
                link_data = {
                    'source': source,
                    'target': target,
                    **edge_data  # Include all edge attributes
                }
                process_model.links.append(link_data)
                
            # Debug information
            self._update_progress(f"Built process model with {len(process_model.nodes)} nodes and {len(process_model.links)} links")
            start_nodes = process_model.get_start_nodes()
            self._update_progress(f"Found {len(start_nodes)} start nodes: {start_nodes}")
            
            # Report progress
            self._update_progress("Setting up simulation...")
            
            # Setup simulation
            # Use current date for simulation start
            start_date = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
            
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
                process_model=process_model,  # Use the proper ProcessModel object
                start_time=start_date,
                work_days=work_days_per_week,
                work_hours_per_day=work_hours_per_day
            )
            
            # Schedule tokens using triangular distribution for intervals
            # This is where we implement the new token arrival logic
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
            
            self._update_progress(f"Scheduled {tokens_scheduled} tokens. Running simulation...")
            
            # Check if cancellation requested
            if self.cancel_requested:
                results = {"error": "Simulation cancelled by user"}
                self._on_complete(results)
                return
                
            # Run the simulation
            results = engine.run_simulation(
                simulation_days=simulation_days,
                progress_callback=self._update_progress
            )
            
            # Process and export results
            self._update_progress("Processing results...")
            
            # Export results
            exporter = ResultsExporter()
            results_path = f"{base_filename}_results.xlsx"
            exporter.export_to_excel(results, results_path, simulation_params)
            
            self._update_progress(f"Results exported to {results_path}")
            
        except Exception as e:
            logging.error(f"Simulation error: {str(e)}", exc_info=True)
            results = {"error": str(e)}
            
        finally:
            self.running = False
            self._on_complete(results)
            
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
            if engine.process_model.nodes:
                node_types = {}
                for node_id, node_data in engine.process_model.nodes.items():
                    node_type = node_data.get('type', 'Unknown')
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                logging.error(f"Node types: {node_types}")
            raise ValueError("No start nodes found in the process model.")
            
        current_time = engine.start_time
        token_count = 0
        
        # Calculate total available time slots based on work hours
        total_minutes = 0
        temp_time = engine.start_time
        
        # Count actual available working minutes in the simulation period
        while temp_time <= end_date:
            if is_work_time(temp_time, engine.start_time, engine.work_days, engine.work_hours_per_day):
                total_minutes += 1
            temp_time += timedelta(minutes=1)
            # For efficiency, skip to next work period if outside work hours
            if not is_work_time(temp_time, engine.start_time, engine.work_days, engine.work_hours_per_day):
                temp_time = advance_to_work_time(temp_time, engine.start_time, engine.work_days, engine.work_hours_per_day)
                
        # Ensure we don't try to schedule more tokens than time allows
        # This is a rough estimate - we'll calculate more precisely as we go
        estimated_avg_interval = (min_interval + avg_interval + max_interval) / 3
        max_possible_tokens = max(1, int(total_minutes / estimated_avg_interval))
        target_tokens = min(max_tokens, max_possible_tokens)
        
        logging.info(f"Scheduling up to {target_tokens} tokens over {total_minutes} available minutes")
        
        while token_count < target_tokens and current_time <= end_date:
            # Ensure start time is within work hours
            if not is_work_time(current_time, engine.start_time, engine.work_days, engine.work_hours_per_day):
                current_time = advance_to_work_time(current_time, engine.start_time, engine.work_days, engine.work_hours_per_day)
                if current_time > end_date:
                    break
                continue
                
            # Create and schedule token
            token_id = f"Token-{token_count + 1}"
            start_node = random.choice(start_nodes)
            
            token = Token(token_id, current_time, start_node)
            engine.tokens[token_id] = token
            
            # Schedule start event
            heapq.heappush(
                engine.event_queue, 
                Event(current_time, token_id, start_node, "start")
            )
            
            logging.info(f"Scheduled {token_id} to start at {current_time}.")
            token_count += 1
            engine.total_tokens_started += 1
            
            # Calculate next interval using triangular distribution
            interval = random.triangular(min_interval, avg_interval, max_interval)
            current_time += timedelta(minutes=interval)
            
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
