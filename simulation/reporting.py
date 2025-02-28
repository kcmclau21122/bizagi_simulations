import logging
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tabulate import tabulate

def save_simulation_report(activity_processing_times, resource_utilization, total_tokens_started, xpdl_file_path, transitions_df, completed_tokens):
    """
    Enhanced simulation report generation with detailed process statistics.
    
    Args:
        activity_processing_times: Dictionary of activity processing times
        resource_utilization: Dictionary of resource utilization percentages
        total_tokens_started: Total number of tokens that started the process
        xpdl_file_path: Path to the source XPDL file
        transitions_df: DataFrame with process transitions
        completed_tokens: List of completed token data
    """
    import logging
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    base_filename = os.path.splitext(os.path.basename(xpdl_file_path))[0]
    output_path = f"{base_filename}_results.xlsx"

    # Normalize column names in transitions_df to lowercase
    transitions_df.columns = map(str.lower, transitions_df.columns)
    
    # Log the columns available in transitions_df to help with debugging
    logging.info(f"Columns in transitions_df: {list(transitions_df.columns)}")

    # Calculate process-level metrics with enhanced statistics
    if completed_tokens:
        process_durations = [
            (token['end_time'] - token['start_time']).total_seconds() / 60 for token in completed_tokens
        ]
        process_wait_times = [token['total_wait_time'] for token in completed_tokens]

        min_time = round(min(process_durations), 2)
        max_time = round(max(process_durations), 2)
        avg_time = round(sum(process_durations) / len(process_durations), 2)
        median_time = round(sorted(process_durations)[len(process_durations) // 2], 2)
        std_dev_time = round(np.std(process_durations), 2)
        
        total_wait_time = round(sum(process_wait_times), 2)
        min_wait_time = round(min(process_wait_times), 2)
        max_wait_time = round(max(process_wait_times), 2)
        avg_wait_time = round(sum(process_wait_times) / len(process_wait_times), 2)
        
        # Calculate 90th percentile processing time
        percentile_90_time = round(np.percentile(process_durations, 90), 2)
        
        # Calculate completion rate
        completion_rate = round((len(completed_tokens) / total_tokens_started) * 100, 2) if total_tokens_started > 0 else 0
    else:
        min_time = max_time = avg_time = median_time = std_dev_time = total_wait_time = min_wait_time = max_wait_time = avg_wait_time = percentile_90_time = 0
        completion_rate = 0

    # Create the process-level summary row with enhanced metrics
    process_row = {
        "Activity": base_filename,
        "Activity Type": "Process",
        "Tokens Started": total_tokens_started,
        "Tokens Completed": len(completed_tokens),
        "Completion Rate (%)": completion_rate,
        "Min Time (min)": min_time,
        "Max Time (min)": max_time,
        "Avg Time (min)": avg_time,
        "Median Time (min)": median_time,
        "Std Dev Time (min)": std_dev_time,
        "90th Percentile Time (min)": percentile_90_time,
        "Total Time Waiting for Resources (min)": total_wait_time,
        "Min Time Waiting for Resources (min)": min_wait_time,
        "Max Time Waiting for Resources (min)": max_wait_time,
        "Avg Time Waiting for Resources (min)": avg_wait_time,
    }

    # Insert the process row as the first row in the activity data
    activity_data = [process_row]

    # Process individual activity data
    for activity, data in activity_processing_times.items():
        durations = data.get("durations", [])
        wait_times = data.get("wait_times", [])
        tokens_started = data.get("tokens_started", 0)
        tokens_completed = data.get("tokens_completed", 0)

        # Handle the case when 'from' column might not exist in transitions_df
        activity_type = "Unknown"
        
        # Check if 'name' and 'type' columns exist instead of 'from'
        if 'name' in transitions_df.columns and 'type' in transitions_df.columns:
            activity_type_row = transitions_df.loc[transitions_df['name'].str.lower() == activity.lower(), 'type']
            if not activity_type_row.empty:
                activity_type = activity_type_row.values[0]
        # Fallback - check for a "from" column if it exists
        elif 'from' in transitions_df.columns and 'type' in transitions_df.columns:
            activity_type_row = transitions_df.loc[transitions_df['from'].str.lower() == activity.lower(), 'type']
            if not activity_type_row.empty:
                activity_type = activity_type_row.values[0]

        if isinstance(activity_type, str) and "condition" in activity_type.lower():
            activity_type = "Gateway"

        min_time = round(min(durations), 2) if durations else 0
        max_time = round(max(durations), 2) if durations else 0
        avg_time = round(sum(durations) / len(durations), 2) if durations else 0
        median_time = round(sorted(durations)[len(durations) // 2], 2) if durations else 0
        std_dev_time = round(np.std(durations), 2) if durations else 0
        percentile_90_time = round(np.percentile(durations, 90), 2) if durations else 0
        
        total_wait_time = round(sum(wait_times), 2) if wait_times else 0
        min_wait_time = round(min(wait_times), 2) if wait_times else 0
        max_wait_time = round(max(wait_times), 2) if wait_times else 0
        avg_wait_time = round(sum(wait_times) / len(wait_times), 2) if wait_times else 0
        
        completion_rate = round((tokens_completed / tokens_started) * 100, 2) if tokens_started > 0 else 0

        activity_data.append({
            "Activity": activity,
            "Activity Type": activity_type,
            "Tokens Started": tokens_started,
            "Tokens Completed": tokens_completed,
            "Completion Rate (%)": completion_rate,
            "Min Time (min)": min_time,
            "Max Time (min)": max_time,
            "Avg Time (min)": avg_time,
            "Median Time (min)": median_time,
            "Std Dev Time (min)": std_dev_time,
            "90th Percentile Time (min)": percentile_90_time,
            "Total Time Waiting for Resources (min)": total_wait_time,
            "Min Time Waiting for Resources (min)": min_wait_time,
            "Max Time Waiting for Resources (min)": max_wait_time,
            "Avg Time Waiting for Resources (min)": avg_wait_time,
        })

    # Process individual token data for detailed token sheet
    token_data = []
    for token in completed_tokens:
        process_duration = (token['end_time'] - token['start_time']).total_seconds() / 60
        token_data.append({
            "Token ID": token.get('current_task', 'Unknown'),
            "Start Time": token['start_time'],
            "End Time": token['end_time'],
            "Total Duration (min)": round(process_duration, 2),
            "Wait Time (min)": round(token['total_wait_time'], 2),
            "Path": " -> ".join(token.get('path', [])),
        })

    # Create dataframes
    activity_df = pd.DataFrame(activity_data)
    resource_df = pd.DataFrame([
        {"Resource": res, "Utilization (%)": round(util, 2)} for res, util in resource_utilization.items()
    ])
    token_df = pd.DataFrame(token_data)

    # Generate visualizations
    try:
        generate_process_visualizations(base_filename, activity_df, resource_df, token_df)
    except Exception as e:
        logging.error(f"Error generating visualizations: {str(e)}")
        # Continue with the report generation even if visualizations fail

    # Save data to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        resource_df.to_excel(writer, index=False, sheet_name="Resource Utilization")
        activity_df.to_excel(writer, index=False, sheet_name="Activity Times")
        token_df.to_excel(writer, index=False, sheet_name="Token Details")
        
        # Add a summary sheet with key metrics
        summary_df = pd.DataFrame([{
            "Metric": "Total Tokens Started",
            "Value": total_tokens_started
        }, {
            "Metric": "Total Tokens Completed",
            "Value": len(completed_tokens)
        }, {
            "Metric": "Completion Rate (%)",
            "Value": completion_rate
        }, {
            "Metric": "Average Process Time (min)",
            "Value": avg_time
        }, {
            "Metric": "90th Percentile Process Time (min)",
            "Value": percentile_90_time
        }, {
            "Metric": "Average Wait Time (min)",
            "Value": avg_wait_time
        }])
        summary_df.to_excel(writer, index=False, sheet_name="Summary")

    print(f"Simulation report saved to {output_path}")
    print("\nProcess Summary:")
    print(f"  Tokens Started: {total_tokens_started}")
    print(f"  Tokens Completed: {len(completed_tokens)}")
    print(f"  Completion Rate: {completion_rate}%")
    print(f"  Average Process Time: {avg_time} minutes")
    print(f"  90th Percentile Process Time: {percentile_90_time} minutes")

def generate_process_visualizations(base_filename, activity_df, resource_df, token_df):
    """
    Generate visualizations for the simulation results.
    
    Args:
        base_filename: Base filename for the output files
        activity_df: DataFrame with activity statistics
        resource_df: DataFrame with resource utilization
        token_df: DataFrame with individual token data
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set styling
    sns.set(style="whitegrid")
    
    # 1. Resource utilization bar chart
    plt.figure(figsize=(10, 6))
    chart = sns.barplot(x="Resource", y="Utilization (%)", data=resource_df)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title("Resource Utilization")
    plt.tight_layout()
    plt.savefig(f"{base_filename}_resource_utilization.png")
    plt.close()
    
    # 2. Activity processing times (non-gateway activities)
    activity_times = activity_df[
        (activity_df["Activity Type"] != "Gateway") & 
        (activity_df["Activity Type"] != "Process")
    ].sort_values("Avg Time (min)", ascending=False).head(10)
    
    if not activity_times.empty:
        plt.figure(figsize=(12, 6))
        chart = sns.barplot(x="Activity", y="Avg Time (min)", data=activity_times)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.title("Top 10 Activities by Average Processing Time")
        plt.tight_layout()
        plt.savefig(f"{base_filename}_activity_times.png")
        plt.close()
    
    # 3. Token processing time histogram
    if not token_df.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(token_df["Total Duration (min)"], kde=True)
        plt.title("Distribution of Token Processing Times")
        plt.xlabel("Processing Time (minutes)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{base_filename}_token_time_distribution.png")
        plt.close()
        
        # 4. Scatter plot of duration vs wait time
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="Total Duration (min)", y="Wait Time (min)", data=token_df)
        plt.title("Process Duration vs Wait Time")
        plt.xlabel("Total Duration (minutes)")
        plt.ylabel("Wait Time (minutes)")
        plt.tight_layout()
        plt.savefig(f"{base_filename}_duration_vs_wait.png")
        plt.close()

def print_processing_times_and_utilization(activity_processing_times, resource_busy_periods, simulation_end_date, start_time, available_resources, transitions_df):
    """
    Print processing times and resource utilization statistics.
    
    Args:
        activity_processing_times: Dictionary of activity processing times
        resource_busy_periods: Dictionary of resource busy periods
        simulation_end_date: End date of the simulation
        start_time: Start time of the simulation
        available_resources: Dictionary of available resources
        transitions_df: DataFrame with process transitions
        
    Returns:
        Dictionary of resource utilization percentages
    """
    total_simulation_time = max((simulation_end_date - start_time).total_seconds() / 3600, 0)
    resource_utilization = {}

    # Normalize column names in transitions_df to lowercase
    transitions_df.columns = map(str.lower, transitions_df.columns)

    # Log structure of activity_processing_times for debugging
    logging.info(f"Activity processing times structure: {activity_processing_times}")

    # Print activity processing times in tabular format
    print("\nActivity Processing Times:")
    activity_data = []
    
    for activity, data in activity_processing_times.items():
        durations = data.get("durations", [])
        valid_durations = []
        for duration in durations:
            if isinstance(duration, tuple) and len(duration) == 2:
                start, end = duration
                if isinstance(start, datetime) and isinstance(end, datetime):
                    valid_durations.append((end - start).total_seconds() / 60)
            elif isinstance(duration, (int, float)):
                valid_durations.append(duration)  # Handle direct durations if present

        if valid_durations:
            min_time = min(valid_durations)
            avg_time = sum(valid_durations) / len(valid_durations)
            max_time = max(valid_durations)
            
            activity_data.append({
                "Activity": activity,
                "Min (min)": f"{min_time:.2f}",
                "Avg (min)": f"{avg_time:.2f}",
                "Max (min)": f"{max_time:.2f}",
                "Count": len(valid_durations)
            })
        else:
            activity_data.append({
                "Activity": activity,
                "Min (min)": "N/A",
                "Avg (min)": "N/A",
                "Max (min)": "N/A",
                "Count": 0
            })
    
    # Print as table
    if activity_data:
        print(tabulate(activity_data, headers="keys", tablefmt="grid"))
    else:
        print("No activity processing data available.")

    # Calculate and print resource utilization
    print("\nResource Utilization:")
    resource_data = []
    
    for resource, periods in resource_busy_periods.items():
        total_busy_time = sum(
            (end - start).total_seconds() for start, end in periods if start and end
        )
        num_resources = available_resources.get(resource, 1)
        utilization = (
            (total_busy_time / (total_simulation_time * 3600 * num_resources)) * 100
            if total_simulation_time > 0 else 0
        )
        utilization = min(utilization, 100)
        resource_utilization[resource] = utilization
        
        resource_data.append({
            "Resource": resource,
            "Utilization (%)": f"{utilization:.2f}",
            "Total Busy Time (hrs)": f"{total_busy_time/3600:.2f}",
            "Resource Count": num_resources
        })
    
    # Print as table
    if resource_data:
        print(tabulate(resource_data, headers="keys", tablefmt="grid"))
    else:
        print("No resource utilization data available.")
    
    for resource, utilization in resource_utilization.items():
        logging.info(f"Resource '{resource}' utilization: {utilization:.2f}%")

    return resource_utilization
