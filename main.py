# Bug: Wait times are not calculated.  Not properly tracking wait time.  Not accuratly reporting on number of tokens completed.
# Main code file for the Bizagi Simulator eumlator

from simulation.simulation import discrete_event_simulation
from simulation.utils import get_simulation_parameters
from simulation.data_handler import read_output_sequences, build_paths
from simulation.reporting import save_simulation_report
from simulation.xpdl_parser import parse_xpdl_to_sequences
import pandas as pd
from datetime import datetime
import random

def main():
    simulation_metrics_path = 'C:/Test Data/Bizagi/simulation_metrics.xlsx'
    xpdl_file_path = 'C:/Test Data/Bizagi/5.5_1/5.5.13 Real Property-Monthly_Reviews/5.5.13 Real Property-Monthly Reviews-org.xpdl'
    output_sequences_path = 'output_sequences.txt'

    simulation_days = 2
    start_time = datetime(2025, 1, 5, 0, 0)
    RANDOM_SEED = 10
    random.seed(RANDOM_SEED)

    process_sequences = parse_xpdl_to_sequences(xpdl_file_path, output_sequences_path)
    simulation_metrics = pd.read_excel(simulation_metrics_path, sheet_name=0)
    transitions_df = read_output_sequences(output_sequences_path)
    paths = build_paths(transitions_df)

    # Extract start tasks from paths
    start_tasks = set()
    for path in paths:
        if path:  # Ensure the path is not empty
            first_activity = path[0].split("->")[0].strip()  # Get the first part before "->"
            if "[Type: Start]" in first_activity:
                activity_name = first_activity.split("[Type:")[0].strip()  # Extract activity name
                start_tasks.add(activity_name)

    max_arrival_count, arrival_interval_minutes = get_simulation_parameters(simulation_metrics)
    discrete_event_simulation(
        max_arrival_count, arrival_interval_minutes, simulation_days, paths,
        simulation_metrics, start_time, xpdl_file_path, transitions_df, start_tasks
    )

if __name__ == "__main__":
    main()
