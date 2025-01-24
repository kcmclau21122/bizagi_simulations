# Bug: Wait times are not calculated. Not properly tracking wait time. Not accurately reporting on number of tokens completed.
# Main code file for the Bizagi Simulator emulator

from simulation.simulation import discrete_event_simulation
from simulation.utils import get_simulation_parameters
from simulation.data_handler import build_paths, diagram_process, extract_start_tasks_from_json
from simulation.reporting import save_simulation_report
from simulation.xpdl_parser import parse_xpdl_to_sequences
import pandas as pd
from datetime import datetime
import random
import json

def main():
    simulation_metrics_path = './Bizagi/simulation_metrics.xlsx'
    #xpdl_file_path = './Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews-1.xpdl'
    #xpdl_file_path = './Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews-2.xpdl'
    #xpdl_file_path = './Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews-Link.xpdl'
    xpdl_file_path = './Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews-Parallel.xpdl'
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

    #transitions_df = read_output_sequences(output_sequences_path)
    #transitions_df.columns = map(str.lower, transitions_df.columns)
    transitions_df = []

    # Transform the output_sequneces.txt file to a dataframe 
    #df_path_sequences = build_sequence_df(output_sequences_path)

    # Build the process paths and sub-paths
    paths = build_paths(output_sequences_path)

    # Diagram the process to a png file
    diagram_process(paths)

    # Extract start tasks from paths
    #json_file_path = "process_model.json"  # Path to the attached JSON file
    start_tasks = extract_start_tasks_from_json(paths)
    
    # Get simulation parameters
    max_arrival_count, arrival_interval_minutes = get_simulation_parameters(simulation_metrics)

    # Run the simulation
    discrete_event_simulation(
        max_arrival_count, arrival_interval_minutes, simulation_days, paths,
        simulation_metrics, start_time, xpdl_file_path, transitions_df, start_tasks
    )

if __name__ == "__main__":
    main()
