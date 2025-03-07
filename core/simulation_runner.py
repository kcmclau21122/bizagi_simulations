import os
import sys

# Add project root to sys.path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import threading
import logging
import random
import datetime
import pandas as pd
from typing import Callable, Dict, Any, Optional

# Use absolute imports for local modules
from core.process_model import ProcessModel
from core.simulation_engine import SimulationEngine

# Use absolute imports for other modules
from utils.config import ConfigManager
from data.xpdl_parser import parse_xpdl_to_sequences
from data.process_builder import ProcessModelBuilder
from data.visualizations import diagram_process
from reporting.report_generator import generate_report

class SimulationRunner:
    """
    Manages the execution of a simulation, handling file parsing,
    model building, simulation execution, and reporting.
    """
    
    def __init__(self, config: ConfigManager, 
                progress_callback: Optional[Callable[[str], None]] = None, 
                completion_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize the simulation runner.
        
        Args:
            config: Configuration manager with simulation settings
            progress_callback: Callback for progress updates
            completion_callback: Callback for simulation completion
        """
        self.config = config
        self.progress_callback = progress_callback
        self.completion_callback = completion_callback
        self.error = None
        self.simulation_thread = None
        
    def run(self) -> None:
        """Run the simulation in a background thread."""
        # Start a background thread for the simulation
        self.simulation_thread = threading.Thread(target=self._run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
    def _run_simulation(self) -> None:
        """Execute the simulation process."""
        try:
            self._update_progress("Initializing simulation...")
            
            # Set up logging
            self._setup_logging()
            
            # Set random seed
            random_seed = self.config.get("random_seed", 10)
            random.seed(random_seed)
            
            # Parse input files
            self._update_progress("Parsing XPDL file...")
            xpdl_file_path = self.config.get("xpdl_file_path")
            metrics_file_path = self.config.get("metrics_file_path")
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_sequences_path = f'output_sequences_{timestamp}.txt'
            parse_xpdl_to_sequences(xpdl_file_path, output_sequences_path)
            
            # Load simulation metrics
            self._update_progress("Loading simulation metrics...")
            simulation_metrics = pd.read_excel(metrics_file_path, sheet_name=0)
            simulation_metrics.columns = map(str.lower, simulation_metrics.columns)
            
            # Build process model
            self._update_progress("Building process model...")
            builder = ProcessModelBuilder()
            graph = builder.build_from_sequences(output_sequences_path, simulation_metrics)
            
            # Generate a timestamped JSON filename to ensure fresh model
            json_file_path = f"process_model_{timestamp}.json"
            json_file_path = builder.save_to_json(json_file_path)
            logging.info(f"Process model saved to: {json_file_path}")
            
            # Generate process diagram
            self._update_progress("Generating process diagram...")
            diagram_process(json_file_path)
            
            # Create process model
            process_model = ProcessModel.from_json(json_file_path)
            
            # Set up simulation parameters
            simulation_days = self.config.get("simulation_days", 2)
            number_workdays = self.config.get_number_of_workdays()
            work_hours_per_day = self.config.get_work_hours_per_day()
            
            work_hours_start = self.config.get("work_hours_start", 9)
            
            # Set up start time - default to Monday at start of work hours
            start_time = datetime.datetime(2025, 1, 6, work_hours_start, 0)  # Monday
            
            # Create simulation engine
            engine = SimulationEngine(
                process_model, start_time, number_workdays, work_hours_per_day
            )
            
            # Extract start node parameters
            try:
                start_node = process_model.get_start_nodes()[0]
                node_data = process_model.get_node(start_node)
                base_arrival_count = int(node_data.get("max arrival count", 20))
                arrival_interval = float(node_data.get("arrival interval", 5))
                
                # Scale the number of tokens based on simulation days
                # We use a more balanced approach to avoid overwhelming the system
                # For longer simulations, we scale the number of tokens
                tokens_per_day = base_arrival_count / 2  # Default assumption: base is for 2 days
                max_arrival_count = int(tokens_per_day * simulation_days)
                
                # Cap to avoid excessive processing in UI
                max_cap = 5000  # Reasonable upper limit
                if max_arrival_count > max_cap:
                    max_arrival_count = max_cap
                    logging.info(f"Capped token count to {max_cap} for performance reasons")
                
                logging.info(f"Adjusted arrival count: {max_arrival_count} for {simulation_days} days " +
                           f"(base: {base_arrival_count}, tokens per day: {tokens_per_day})")
            except (IndexError, ValueError) as e:
                logging.warning(f"Could not extract start node parameters: {e}")
                max_arrival_count = min(20 * simulation_days, 1000)  # Scale with days but cap at 1000
                arrival_interval = 5
                
            # Schedule tokens
            self._update_progress(f"Scheduling up to {max_arrival_count} tokens...")
            simulation_end_date = start_time + datetime.timedelta(days=simulation_days)
            tokens_scheduled = engine.schedule_tokens(
                max_arrival_count, arrival_interval, simulation_end_date
            )
            logging.info(f"Scheduled {tokens_scheduled} tokens")
            
            # Run simulation
            self._update_progress("Running simulation...")
            target_avg_time = self.config.get("target_avg_time", 0)
            if target_avg_time > 0:
                self._update_progress("Optimizing for target average time...")
                # Note: Target optimization would be implemented here
                
            simulation_results = engine.run_simulation(simulation_days, self._update_progress)
            
            # Generate report - MODIFIED to capture visualization paths
            self._update_progress("Generating simulation report...")
            report_path, visualization_paths = generate_report(
                simulation_results["activity_processing_times"],
                simulation_results["resource_utilization"],
                simulation_results["total_tokens_started"],
                xpdl_file_path,
                simulation_metrics,
                simulation_results["completed_tokens"]
            )
            
            # Add visualization paths to results - NEW
            simulation_results["visualization_paths"] = visualization_paths
            
            # Add model path to results
            simulation_results["process_model_path"] = json_file_path
            
            logging.info(f"Report generated at {report_path}")
            logging.info(f"Visualization paths: {visualization_paths}")
            self._update_progress(f"Simulation complete. Report saved to {report_path}")
            
            # Signal completion
            if self.completion_callback:
                self.completion_callback(simulation_results)
            
        except Exception as e:
            logging.error(f"Error in simulation: {str(e)}", exc_info=True)
            self.error = str(e)
            # Signal completion with error
            self._on_error()
            
    def _update_progress(self, message: str) -> None:
        """
        Update progress callback safely.
        
        Args:
            message: Progress message to report
        """
        if self.progress_callback:
            try:
                self.progress_callback(message)
            except Exception as e:
                logging.error(f"Error in progress callback: {str(e)}")
                
    def _setup_logging(self) -> None:
        """Set up logging for the simulation."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"simulation_log_{timestamp}.txt"
        
        logging.basicConfig(
            filename=log_filename,
            filemode='w',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Log configuration
        logging.info(f"Simulation started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Configuration:")
        for key, value in self.config.items():
            logging.info(f"  {key}: {value}")
            
    def _on_error(self) -> None:
        """Handle simulation error."""
        error_message = f"An error occurred during simulation: {self.error}"
        logging.error(error_message)
        self._update_progress(f"ERROR: {self.error}")
        
        if self.completion_callback:
            self.completion_callback({"error": self.error})
            
    def is_running(self) -> bool:
        """
        Check if simulation is still running.
        
        Returns:
            True if simulation thread is active, False otherwise
        """
        return self.simulation_thread is not None and self.simulation_thread.is_alive()
        
    def cancel(self) -> None:
        """
        Cancel a running simulation.
        Note: This is a best-effort attempt, as Python threads cannot be forcibly terminated.
        """
        # We can't really cancel a running thread in Python
        # But we can set a flag that the simulation can check
        self._update_progress("Cancellation requested...")
        logging.info("Simulation cancellation requested")
        
def main():
    """Main function to run a simulation."""
    # Create a simple main function if you need one
    config = ConfigManager()  # Initialize with default values or load from a file
    runner = SimulationRunner(config)
    runner.run()
    
    # Wait for simulation to complete
    import time
    while runner.is_running():
        time.sleep(1)
    
    print("Simulation completed")

if __name__ == "__main__":
    main()