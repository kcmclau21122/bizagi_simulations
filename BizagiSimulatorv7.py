import pandas as pd
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt

# Constants for simulation constraints
WORKING_HOURS_PER_DAY = 6
WORKING_DAYS_PER_WEEK = 5
SIMULATION_PERIOD_DAYS = 30
WORKING_HOURS_TOTAL = WORKING_HOURS_PER_DAY * WORKING_DAYS_PER_WEEK * (SIMULATION_PERIOD_DAYS / 7)

MAX_UTILIZATION = 95  # Maximum utilization percentage

# File paths
simulation_metrics_path = 'C:/Test Data/Bizagi/simulation_metrics.csv'
simulation_results_path = 'C:/Test Data/Bizagi/simulation_results.xlsx'
xpdl_file_path = 'C:/Test Data/Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews.xpdl'

# Load simulation metrics
simulation_metrics = pd.read_csv(simulation_metrics_path)

# Load simulation results
simulation_results = pd.ExcelFile(simulation_results_path)

# Load the resources sheet
resources_data = simulation_results.parse('Resources')

# Calculate max_cost as twice the current total cost from the "Resources" tab
current_total_cost = resources_data['Total cost'].sum()
MAX_COST = 2 * current_total_cost

# Parse the XPDL file for task dependencies
tree = ET.parse(xpdl_file_path)
root = tree.getroot()
namespaces = {'xpdl': root.tag.split('}')[0].strip('{')}  # Extract namespace

# Extract task dependencies from the XPDL file
process_flow = []
for process in root.findall('.//xpdl:WorkflowProcess', namespaces):
    for transition in process.findall('.//xpdl:Transition', namespaces):
        from_activity = transition.get('From')
        to_activity = transition.get('To')
        process_flow.append({'From': from_activity, 'To': to_activity})
process_flow_df = pd.DataFrame(process_flow)

# Extract task names and their IDs
tasks_data = []
for process in root.findall('.//xpdl:WorkflowProcess', namespaces):
    for activity in process.findall('.//xpdl:Activity', namespaces):
        activity_id = activity.get('Id')
        activity_name = activity.get('Name')
        tasks_data.append({'ID': activity_id, 'Name': activity_name})
tasks_mapping_df = pd.DataFrame(tasks_data)

# Map task names to process flow
mapped_process_flow_df = process_flow_df.merge(
    tasks_mapping_df.rename(columns={'ID': 'From', 'Name': 'From Task Name'}),
    how='left',
    on='From'
).merge(
    tasks_mapping_df.rename(columns={'ID': 'To', 'Name': 'To Task Name'}),
    how='left',
    on='To'
)

# Extract token arrival metrics from "Start event"
start_event_metrics = simulation_metrics[simulation_metrics['Type'] == 'Start event']
# Dynamically set the TOKEN_ARRIVAL_INTERVAL_MINUTES and TOTAL_TOKENS
arrival_interval_minutes = start_event_metrics['Arrival interval'].iloc[0]  # Token arrival interval in minutes
max_arrival_count = start_event_metrics['Max arrival count'].iloc[0]  # Total tokens to arrive

# Calculate total workload hours for each resource type
workload_hours_tech = 0
workload_hours_certifier = 0

for _, transition in mapped_process_flow_df.iterrows():
    from_task_name = transition['From Task Name']
    task_metric = simulation_metrics[
        (simulation_metrics['Name'] == from_task_name) & 
        (simulation_metrics['Type'] == 'Task')
    ]

    if not task_metric.empty:
        task = task_metric.iloc[0]
        resource = task['Resource']
        avg_time_minutes = task['Avg Time']
        avg_time_hours = avg_time_minutes / 60
        workload_hours = (max_arrival_count * avg_time_hours) / (60 / arrival_interval_minutes)

        if resource == 'Tech':
            workload_hours_tech += workload_hours
        elif resource == 'Certifier':
            workload_hours_certifier += workload_hours


# Simulation logic with iterative adjustment of resources
def calculate_scaled_resources(current_resources, current_utilization, max_utilization):
    """
    Scales the current resources to meet the max utilization constraint.
    Args:
        current_resources: Current number of resources.
        current_utilization: Current utilization percentage (e.g., 100.0 for 100%).
        max_utilization: Maximum allowed utilization percentage.
    Returns:
        Scaled resource count.
    """
    if current_utilization <= max_utilization:
        return current_resources
    scale_factor = current_utilization / max_utilization
    return int(current_resources * scale_factor) + 1


def find_optimized_resources_with_scaling():
    """
    Finds optimized resources using simulation metrics and results to set starting values.
    Returns:
    - DataFrame of optimal scenarios or least-cost fallback.
    """
    # Extract initial resource counts and utilization from simulation files
    tech_initial_resources = int(simulation_metrics[simulation_metrics['Resource'] == 'Tech']['Available Resources'].iloc[0])
    certifier_initial_resources = int(simulation_metrics[simulation_metrics['Resource'] == 'Certifier']['Available Resources'].iloc[0])

    tech_utilization = float(resources_data[resources_data['Resource'] == 'Tech']['Utilization'].iloc[0].strip('%'))
    certifier_utilization = float(resources_data[resources_data['Resource'] == 'Certifier']['Utilization'].iloc[0].strip('%'))

    # Scale resources to meet utilization constraints
    tech_start_resources = calculate_scaled_resources(tech_initial_resources, tech_utilization, MAX_UTILIZATION)
    certifier_start_resources = calculate_scaled_resources(certifier_initial_resources, certifier_utilization, MAX_UTILIZATION)

    optimal_scenarios = []
    fallback_scenario = None

    for tech_resources in range(tech_start_resources, tech_start_resources + 10):  # Start from scaled values
        for certifier_resources in range(certifier_start_resources, certifier_start_resources + 10):
            tech_utilization = workload_hours_tech / (tech_resources * WORKING_HOURS_TOTAL)
            certifier_utilization = workload_hours_certifier / (certifier_resources * WORKING_HOURS_TOTAL)

            # Calculate total cost
            tech_cost = resources_data[resources_data['Resource'] == 'Tech']['Total cost'].iloc[0] * tech_resources
            certifier_cost = resources_data[resources_data['Resource'] == 'Certifier']['Total cost'].iloc[0] * certifier_resources
            total_cost = tech_cost + certifier_cost

            # Check constraints
            if tech_utilization <= MAX_UTILIZATION and certifier_utilization <= MAX_UTILIZATION:
                scenario = {
                    'Tech Resources': tech_resources,
                    'Certifier Resources': certifier_resources,
                    'Tech Utilization': tech_utilization * 100,
                    'Certifier Utilization': certifier_utilization * 100,
                    'Total Cost': total_cost
                }

                if total_cost <= MAX_COST:
                    optimal_scenarios.append(scenario)
                    return pd.DataFrame(optimal_scenarios)  # Return the first valid scenario
                else:
                    # Track fallback scenario
                    if fallback_scenario is None or total_cost < fallback_scenario['Total Cost']:
                        fallback_scenario = scenario

    # If no valid scenario within cost constraints, return the fallback
    if fallback_scenario:
        print("Warning: Could not meet the cost constraint, showing best utilization-based allocation.")
        return pd.DataFrame([fallback_scenario])

    # If no fallback exists, return an empty DataFrame
    print("No feasible configurations found.")
    return pd.DataFrame()

# Run the optimized simulation
optimized_scenarios_df = find_optimized_resources_with_scaling()

# Print the results
if not optimized_scenarios_df.empty:
    print("Optimal or Fallback Resource Configurations:")
    print(optimized_scenarios_df)
else:
    print("No feasible configurations found.")

# Optionally save the results to a file
optimized_scenarios_df.to_csv('optimized_scenarios.csv', index=False)
