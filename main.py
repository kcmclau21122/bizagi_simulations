# Main code file for the Bizagi Simulator emulator

from simulation.simulation import run_simulation
from simulation.utils import get_simulation_parameters
from simulation.data_handler import build_paths, diagram_process, extract_start_tasks_from_json
from simulation.reporting import save_simulation_report
from simulation.xpdl_parser import parse_xpdl_to_sequences
import pandas as pd
from datetime import datetime
import random
import json
import logging
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    filename='simulation_log.txt',
    filemode='w',  # Overwrite the log file
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    simulation_metrics_path = './Bizagi/simulation_metrics-2.xlsx'
    #xpdl_file_path = './Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews-1.xpdl'
    xpdl_file_path = './Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews-2.xpdl'
    #xpdl_file_path = './Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews-Link.xpdl'
    #xpdl_file_path = './Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews-Parallel.xpdl'
    #xpdl_file_path = './Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews-Inclusive.xpdl'
    output_sequences_path = 'output_sequences.txt'

    simulation_days = 2
    start_time = datetime(2025, 1, 5, 0, 0)
    RANDOM_SEED = 10
    random.seed(RANDOM_SEED)

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

    # Run the simulation
    run_simulation(json_file_path, simulation_metrics, simulation_days, start_time)

if __name__ == "__main__":
    main()
