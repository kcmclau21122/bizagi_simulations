#!/usr/bin/env python3
# Enhanced main.py with target processing time optimization

from simulation import run_simulation
from utils import get_simulation_parameters
from data_handler import build_paths, diagram_process, extract_start_tasks_from_json
from reporting import save_simulation_report
from xpdl_parser import parse_xpdl_to_sequences
import pandas as pd
from datetime import datetime
import random
import json
import logging
import argparse
from tabulate import tabulate

def parse_arguments():
    """Parse command line arguments for the simulation."""
    parser = argparse.ArgumentParser(description="Business Process Simulation")
    
    parser.add_argument(
        "--xpdl", 
        type=str, 
        default="./Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews-2.xpdl",
        help="Path to the XPDL file"
    )
    
    parser.add_argument(
        "--metrics", 
        type=str, 
        default="./Bizagi/simulation_metrics-2.xlsx",
        help="Path to the simulation metrics Excel file"
    )
    
    parser.add_argument(
        "--days", 
        type=int, 
        default=2,
        help="Number of days to simulate"
    )
    
    parser.add_argument(
        "--workdays", 
        type=int, 
        default=5,
        help="Number of workdays per week"
    )
    
    parser.add_argument(
        "--hours", 
        type=int, 
        default=6,
        help="Number of work hours per day"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=10,
        help="Random seed for simulation"
    )
    
    parser.add_argument(
        "--target-time", 
        type=float, 
        default=None,
        help="Target average token processing time in minutes"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        filename='simulation_log.txt',
        filemode='w',  # Overwrite the log file
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Set up simulation parameters
    simulation_metrics_path = args.metrics
    xpdl_file_path = args.xpdl
    output_sequences_path = 'output_sequences.txt'

    simulation_days = args.days
    start_time = datetime(2025, 1, 5, 0, 0)
    number_workdays = args.workdays
    number_work_hours_per_day = args.hours
    target_avg_time = args.target_time
    
    # Set random seed
    RANDOM_SEED = args.seed
    random.seed(RANDOM_SEED)

    logging.info(f"Starting simulation with parameters:")
    logging.info(f"  XPDL file: {xpdl_file_path}")
    logging.info(f"  Metrics file: {simulation_metrics_path}")
    logging.info(f"  Simulation days: {simulation_days}")
    logging.info(f"  Workdays: {number_workdays}")
    logging.info(f"  Work hours per day: {number_work_hours_per_day}")
    logging.info(f"  Random seed: {RANDOM_SEED}")
    
    if target_avg_time:
        logging.info(f"  Target average processing time: {target_avg_time} minutes")

    # Parse process sequences and load data
    process_sequences = parse_xpdl_to_sequences(xpdl_file_path, output_sequences_path)
    simulation_metrics = pd.read_excel(simulation_metrics_path, sheet_name=0)

    # Normalize column names to lowercase
    simulation_metrics.columns = map(str.lower, simulation_metrics.columns)

    # Log the entire simulation_metrics DataFrame as a formatted table
    logging.info("Logging simulation metrics as a formatted table:")
    table = tabulate(simulation_metrics, headers='keys', tablefmt='grid', showindex=False)
    logging.info("\n" + table)

    # Build the process paths and sub-paths
    json_file_path = build_paths(output_sequences_path, simulation_metrics)

    # Diagram the process to a png file
    diagram_process(json_file_path)

    # Run the simulation, optionally targeting a specific average processing time
    simulation_results = run_simulation(
        json_file_path, 
        simulation_days, 
        start_time, 
        number_workdays, 
        number_work_hours_per_day,
        target_avg_time
    )
    
    # Extract results for reporting
    activity_processing_times = simulation_results["activity_processing_times"]
    resource_utilization = simulation_results["resource_utilization"]
    total_tokens_started = simulation_results["total_tokens_started"]
    completed_tokens = simulation_results["completed_tokens"]
    
    # Calculate and display overall process statistics
    if completed_tokens:
        process_durations = [
            (token["end_time"] - token["start_time"]).total_seconds() / 60 
            for token in completed_tokens
        ]
        avg_process_time = sum(process_durations) / len(process_durations)
        
        print("\n===== PROCESS SUMMARY =====")
        print(f"Total tokens started: {total_tokens_started}")
        print(f"Total tokens completed: {len(completed_tokens)}")
        print(f"Average processing time: {avg_process_time:.2f} minutes")
        
        if target_avg_time:
            difference = avg_process_time - target_avg_time
            print(f"Target time: {target_avg_time:.2f} minutes")
            print(f"Difference from target: {difference:.2f} minutes ({(difference/target_avg_time)*100:.1f}%)")
    
    # Generate simulation report
    save_simulation_report(
        activity_processing_times, 
        resource_utilization, 
        total_tokens_started, 
        xpdl_file_path, 
        simulation_metrics, 
        completed_tokens
    )
    
    logging.info("Simulation completed successfully")

if __name__ == "__main__":
    main()
