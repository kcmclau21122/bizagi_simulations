import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional

def generate_resource_chart(resource_df: pd.DataFrame, output_path: str) -> str:
    """
    Generate a resource utilization bar chart.
    
    Args:
        resource_df: DataFrame with resource utilization data
        output_path: Path to save the chart
        
    Returns:
        Path to the saved chart
    """
    # Set styling
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    
    # Sort by utilization for better visualization
    sorted_df = resource_df.sort_values("Utilization (%)", ascending=False)
    
    # Create the bar chart
    chart = sns.barplot(
        x="Resource", 
        y="Utilization (%)", 
        data=sorted_df,
        palette="viridis"
    )
    
    # Rotate x labels for better readability
    chart.set_xticklabels(
        chart.get_xticklabels(), 
        rotation=45, 
        horizontalalignment='right'
    )
    
    # Add labels and title
    plt.title("Resource Utilization", fontsize=14)
    plt.xlabel("Resource", fontsize=12)
    plt.ylabel("Utilization (%)", fontsize=12)
    
    # Add value labels on top of bars
    for p in chart.patches:
        chart.annotate(
            f'{p.get_height():.1f}%', 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha = 'center', va = 'bottom',
            fontsize=9
        )
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def generate_activity_chart(activity_df: pd.DataFrame, output_path: str) -> str:
    """
    Generate a chart of top activities by processing time.
    
    Args:
        activity_df: DataFrame with activity data
        output_path: Path to save the chart
        
    Returns:
        Path to the saved chart
    """
    # Filter non-gateway activities and process row
    activity_times = activity_df[
        (activity_df["Activity Type"] != "Gateway") & 
        (activity_df["Activity Type"] != "Process")
    ].sort_values("Avg Time (min)", ascending=False).head(10)
    
    if activity_times.empty:
        # Create a placeholder chart if no data
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No activity data available", 
                 horizontalalignment='center', fontsize=14)
        plt.savefig(output_path)
        plt.close()
        return output_path
    
    plt.figure(figsize=(12, 6))
    
    # Create bar chart with dual colors for processing and waiting time
    ax = plt.subplot(111)
    
    # Processing time bars
    bars1 = ax.bar(
        activity_times["Activity"], 
        activity_times["Avg Time (min)"],
        label='Processing Time',
        color='#5975a4'
    )
    
    # Wait time bars (stacked)
    bars2 = ax.bar(
        activity_times["Activity"], 
        activity_times["Avg Time Waiting for Resources (min)"],
        label='Wait Time',
        color='#a4596d'
    )
    
    # Add labels and title
    plt.title("Top 10 Activities by Average Processing Time", fontsize=14)
    plt.xlabel("Activity", fontsize=12)
    plt.ylabel("Time (minutes)", fontsize=12)
    plt.legend()
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom',
                fontsize=9
            )
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def generate_token_histogram(token_df: pd.DataFrame, output_path: str) -> str:
    """
    Generate a histogram of token processing times.
    
    Args:
        token_df: DataFrame with token data
        output_path: Path to save the chart
        
    Returns:
        Path to the saved chart
    """
    plt.figure(figsize=(10, 6))
    
    # Create the histogram with KDE
    sns.histplot(
        token_df["Total Duration (min)"], 
        kde=True,
        bins=20,
        color='#5975a4'
    )
    
    # Add a vertical line for the mean
    mean_duration = token_df["Total Duration (min)"].mean()
    plt.axvline(
        mean_duration, 
        color='red', 
        linestyle='dashed', 
        linewidth=1,
        label=f'Mean: {mean_duration:.2f} min'
    )
    
    # Add a vertical line for the median
    median_duration = token_df["Total Duration (min)"].median()
    plt.axvline(
        median_duration, 
        color='green', 
        linestyle='dashed', 
        linewidth=1,
        label=f'Median: {median_duration:.2f} min'
    )
    
    # Add a vertical line for the 90th percentile
    percentile_90 = token_df["Total Duration (min)"].quantile(0.9)
    plt.axvline(
        percentile_90, 
        color='orange', 
        linestyle='dashed', 
        linewidth=1,
        label=f'90th Percentile: {percentile_90:.2f} min'
    )
    
    # Add labels and title
    plt.title("Distribution of Token Processing Times", fontsize=14)
    plt.xlabel("Processing Time (minutes)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def generate_duration_wait_scatter(token_df: pd.DataFrame, output_path: str) -> str:
    """
    Generate a scatter plot of process duration vs wait time.
    
    Args:
        token_df: DataFrame with token data
        output_path: Path to save the chart
        
    Returns:
        Path to the saved chart
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate wait time percentage
    token_df = token_df.copy()
    token_df['Wait Time Percentage'] = (
        token_df['Wait Time (min)'] / token_df['Total Duration (min)'] * 100
    ).fillna(0).clip(0, 100)
    
    # Create a scatter plot with hue based on wait time percentage
    scatter = plt.scatter(
        token_df["Total Duration (min)"], 
        token_df["Wait Time (min)"],
        c=token_df['Wait Time Percentage'],
        cmap='YlOrRd',
        alpha=0.7,
        s=70
    )
    
    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Wait Time %', rotation=270, labelpad=20)
    
    # Add a diagonal line representing y=x (100% wait time)
    max_val = max(
        token_df["Total Duration (min)"].max(),
        token_df["Wait Time (min)"].max()
    )
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    
    # Calculate the correlation
    corr = token_df["Total Duration (min)"].corr(token_df["Wait Time (min)"])
    
    # Add labels and title
    plt.title(f"Process Duration vs Wait Time (correlation: {corr:.2f})", fontsize=14)
    plt.xlabel("Total Duration (minutes)", fontsize=12)
    plt.ylabel("Wait Time (minutes)", fontsize=12)
    
    # Add text for data points percentage
    wait_pct = (token_df["Wait Time (min)"].sum() / token_df["Total Duration (min)"].sum()) * 100
    plt.text(
        0.05, 0.95, 
        f"Overall wait time: {wait_pct:.1f}% of total process time",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def generate_resource_efficiency_chart(activity_df: pd.DataFrame, resource_df: pd.DataFrame, 
                                    output_path: str) -> str:
    """
    Generate a chart showing resource efficiency vs utilization.
    
    Args:
        activity_df: DataFrame with activity data
        resource_df: DataFrame with resource utilization data
        output_path: Path to save the chart
        
    Returns:
        Path to the saved chart
    """
    # This is a more advanced visualization that could correlate
    # activity wait times with resource utilization
    
    # For now, we'll create a simpler placeholder chart
    plt.figure(figsize=(10, 6))
    
    # Create a bubble chart where:
    # - x-axis is resource utilization
    # - y-axis is average wait time
    # - bubble size is token count
    
    # For this placeholder, we'll just use resource_df data
    x = resource_df["Utilization (%)"]
    y = np.random.rand(len(x)) * 20  # Random wait times for placeholder
    size = np.random.rand(len(x)) * 100 + 50  # Random sizes for placeholder
    
    plt.scatter(x, y, s=size, alpha=0.6, c=x, cmap='viridis')
    
    plt.title("Resource Efficiency Analysis", fontsize=14)
    plt.xlabel("Resource Utilization (%)", fontsize=12)
    plt.ylabel("Average Wait Time (minutes)", fontsize=12)
    
    # Add text noting this is a placeholder
    plt.text(
        0.5, 0.5, 
        "Enhanced resource efficiency analysis\nwill be available in future versions",
        transform=plt.gca().transAxes,
        fontsize=12,
        horizontalalignment='center',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path
