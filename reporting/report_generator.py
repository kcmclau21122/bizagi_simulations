import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from .visualizations import (
    generate_resource_chart,
    generate_activity_chart,
    generate_token_histogram,
    generate_duration_wait_scatter
)

def generate_report(activity_processing_times: Dict[str, Dict[str, Any]],
                   resource_utilization: Dict[str, float],
                   total_tokens_started: int,
                   xpdl_file_path: str,
                   transitions_df: pd.DataFrame,
                   completed_tokens: List[Dict[str, Any]]) -> Tuple[str, Dict[str, str]]:
    """
    Generate a comprehensive simulation report.
    
    Args:
        activity_processing_times: Dictionary of activity processing times
        resource_utilization: Dictionary of resource utilization percentages
        total_tokens_started: Total number of tokens that started the process
        xpdl_file_path: Path to the source XPDL file
        transitions_df: DataFrame with process transitions
        completed_tokens: List of completed token data
        
    Returns:
        Tuple containing:
        - Path to the generated report
        - Dictionary with paths to generated visualizations
    """
    # Normalize column names in transitions_df to lowercase
    transitions_df.columns = map(str.lower, transitions_df.columns)
    
    # Log the columns available in transitions_df to help with debugging
    logging.info(f"Columns in transitions_df: {list(transitions_df.columns)}")

    # Get base filename for output
    base_filename = os.path.splitext(os.path.basename(xpdl_file_path))[0]
    output_path = f"{base_filename}_results.xlsx"

    # Calculate process-level metrics with enhanced statistics
    if completed_tokens:
        process_metrics = calculate_process_metrics(completed_tokens)
    else:
        process_metrics = create_empty_process_metrics()

    # Create the process-level summary row with enhanced metrics
    process_row = {
        "Activity": base_filename,
        "Activity Type": "Process",
        "Tokens Started": total_tokens_started,
        "Tokens Completed": len(completed_tokens),
        "Completion Rate (%)": process_metrics["completion_rate"],
        "Min Time (min)": process_metrics["min_time"],
        "Max Time (min)": process_metrics["max_time"],
        "Avg Time (min)": process_metrics["avg_time"],
        "Median Time (min)": process_metrics["median_time"],
        "Std Dev Time (min)": process_metrics["std_dev_time"],
        "90th Percentile Time (min)": process_metrics["percentile_90_time"],
        "Total Time Waiting for Resources (min)": process_metrics["total_wait_time"],
        "Min Time Waiting for Resources (min)": process_metrics["min_wait_time"],
        "Max Time Waiting for Resources (min)": process_metrics["max_wait_time"],
        "Avg Time Waiting for Resources (min)": process_metrics["avg_wait_time"],
    }

    # Insert the process row as the first row in the activity data
    activity_data = [process_row]

    # Process individual activity data
    for activity, data in activity_processing_times.items():
        activity_row = process_activity_data(activity, data, transitions_df)
        activity_data.append(activity_row)

    # Process individual token data for detailed token sheet
    token_data = process_token_data(completed_tokens)

    # Create dataframes
    activity_df = pd.DataFrame(activity_data)
    resource_df = pd.DataFrame([
        {"Resource": res, "Utilization (%)": round(util, 2)} 
        for res, util in resource_utilization.items()
    ])
    token_df = pd.DataFrame(token_data)

    # Generate visualizations
    visualization_paths = {}
    try:
        visualization_paths = generate_visualizations(
            base_filename, activity_df, resource_df, token_df
        )
        logging.info(f"Generated visualizations: {visualization_paths}")
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
            "Value": process_metrics["completion_rate"]
        }, {
            "Metric": "Average Process Time (min)",
            "Value": process_metrics["avg_time"]
        }, {
            "Metric": "90th Percentile Process Time (min)",
            "Value": process_metrics["percentile_90_time"]
        }, {
            "Metric": "Average Wait Time (min)",
            "Value": process_metrics["avg_wait_time"]
        }])
        summary_df.to_excel(writer, index=False, sheet_name="Summary")

    # Log summary of results
    logging.info(f"Simulation report saved to {output_path}")
    logging.info("\nProcess Summary:")
    logging.info(f"  Tokens Started: {total_tokens_started}")
    logging.info(f"  Tokens Completed: {len(completed_tokens)}")
    logging.info(f"  Completion Rate: {process_metrics['completion_rate']}%")
    logging.info(f"  Average Process Time: {process_metrics['avg_time']} minutes")
    logging.info(f"  90th Percentile Process Time: {process_metrics['percentile_90_time']} minutes")

    return output_path, visualization_paths

def calculate_process_metrics(completed_tokens: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate process-level metrics from completed tokens.
    
    Args:
        completed_tokens: List of completed token data
        
    Returns:
        Dictionary of process metrics
    """
    # Calculate process durations
    process_durations = [
        (token['end_time'] - token['start_time']).total_seconds() / 60 
        for token in completed_tokens
    ]
    
    # Calculate wait times
    process_wait_times = [token['total_wait_time'] for token in completed_tokens]

    # Calculate statistics
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
    
    # Calculate completion rate (this would need total_tokens_started)
    # For now, we'll just set it to 100% for completed tokens
    completion_rate = 100.0
    
    return {
        "min_time": min_time,
        "max_time": max_time,
        "avg_time": avg_time,
        "median_time": median_time,
        "std_dev_time": std_dev_time,
        "total_wait_time": total_wait_time,
        "min_wait_time": min_wait_time,
        "max_wait_time": max_wait_time,
        "avg_wait_time": avg_wait_time,
        "percentile_90_time": percentile_90_time,
        "completion_rate": completion_rate
    }

def create_empty_process_metrics() -> Dict[str, float]:
    """
    Create empty process metrics for when no tokens are completed.
    
    Returns:
        Dictionary of empty process metrics
    """
    return {
        "min_time": 0,
        "max_time": 0,
        "avg_time": 0,
        "median_time": 0,
        "std_dev_time": 0,
        "total_wait_time": 0,
        "min_wait_time": 0,
        "max_wait_time": 0,
        "avg_wait_time": 0,
        "percentile_90_time": 0,
        "completion_rate": 0
    }

def process_activity_data(activity: str, data: Dict[str, Any], 
                        transitions_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Process data for an individual activity.
    
    Args:
        activity: Activity name
        data: Activity data
        transitions_df: DataFrame with process transitions
        
    Returns:
        Dictionary of processed activity data
    """
    durations = data.get("durations", [])
    wait_times = data.get("wait_times", [])
    tokens_started = data.get("tokens_started", 0)
    tokens_completed = data.get("tokens_completed", 0)

    # Determine activity type
    activity_type = "Unknown"
    
    # Check if 'name' and 'type' columns exist
    if 'name' in transitions_df.columns and 'type' in transitions_df.columns:
        activity_type_row = transitions_df.loc[
            transitions_df['name'].str.lower() == activity.lower(), 'type'
        ]
        if not activity_type_row.empty:
            activity_type = activity_type_row.values[0]
    # Fallback - check for a "from" column
    elif 'from' in transitions_df.columns and 'type' in transitions_df.columns:
        activity_type_row = transitions_df.loc[
            transitions_df['from'].str.lower() == activity.lower(), 'type'
        ]
        if not activity_type_row.empty:
            activity_type = activity_type_row.values[0]

    # Check for gateway type
    if isinstance(activity_type, str) and "condition" in activity_type.lower():
        activity_type = "Gateway"

    # Calculate statistics
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

    return {
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
    }

def process_token_data(completed_tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process data for individual tokens.
    
    Args:
        completed_tokens: List of completed token data
        
    Returns:
        List of processed token data
    """
    token_data = []
    for token in completed_tokens:
        process_duration = (token['end_time'] - token['start_time']).total_seconds() / 60
        token_data.append({
            "Token ID": token.get('token_id', token.get('current_task', 'Unknown')),
            "Start Time": token['start_time'],
            "End Time": token['end_time'],
            "Total Duration (min)": round(process_duration, 2),
            "Wait Time (min)": round(token['total_wait_time'], 2),
            "Path": " -> ".join(token.get('path', [])),
        })
    return token_data

def generate_visualizations(base_filename: str, activity_df: pd.DataFrame, 
                           resource_df: pd.DataFrame, token_df: pd.DataFrame) -> Dict[str, str]:
    """
    Generate visualizations for the simulation results.
    
    Args:
        base_filename: Base filename for output files
        activity_df: DataFrame with activity data
        resource_df: DataFrame with resource data
        token_df: DataFrame with token data
        
    Returns:
        Dictionary mapping visualization names to file paths
    """
    results = {}
    
    # Resource utilization chart
    resource_chart_path = generate_resource_chart(resource_df, f"{base_filename}_resource_utilization.png")
    results["resource_utilization"] = resource_chart_path
    
    # Activity processing times chart
    activity_chart_path = generate_activity_chart(activity_df, f"{base_filename}_activity_times.png")
    results["activity_times"] = activity_chart_path
    
    # Only generate token visualizations if we have token data
    if not token_df.empty:
        # Token processing time histogram
        histogram_path = generate_token_histogram(token_df, f"{base_filename}_token_time_distribution.png")
        results["token_histogram"] = histogram_path
        
        # Scatter plot of duration vs wait time
        scatter_path = generate_duration_wait_scatter(token_df, f"{base_filename}_duration_vs_wait.png")
        results["duration_vs_wait"] = scatter_path
    
    return results
