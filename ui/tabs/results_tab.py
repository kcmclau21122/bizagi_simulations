import sys
import os

# Add project root to sys.path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import pandas as pd
from typing import Dict, Any, List, Optional

from utils.config import ConfigManager
from utils.time_utils import format_duration_for_display

class ResultsTab:
    """
    Tab for displaying simulation results.
    Shows process statistics, charts, and detailed information.
    """
    
    def __init__(self, parent: ttk.Notebook, config: ConfigManager):
        """
        Initialize the results tab.
        
        Args:
            parent: Parent notebook widget
            config: Configuration manager
        """
        self.parent = parent
        self.config = config
        
        # Create tab frame
        self.frame = ttk.Frame(parent)
        
        # Setup UI elements
        self.setup_ui()
        
        # Store latest results
        self.latest_results = None
        
    def setup_ui(self):
        """Set up the UI components of the tab."""
        # Create a frame for the results text area with scrollbar
        text_frame = ttk.Frame(self.frame)
        text_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add vertical scrollbar
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")
        
        # Results text area with scrollbar
        self.results_text = tk.Text(text_frame, wrap="word", height=15, width=80, yscrollcommand=scrollbar.set)
        self.results_text.pack(side="left", fill="both", expand=True)
        self.results_text.insert("1.0", "Simulation results will appear here after running a simulation.")
        self.results_text.config(state="disabled")
        
        # Configure scrollbar to scroll the text
        scrollbar.config(command=self.results_text.yview)
        
        # Create notebook for results visualization
        self.vis_notebook = ttk.Notebook(self.frame)
        self.vis_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create tabs for different result visualizations
        self.overview_frame = ttk.Frame(self.vis_notebook)
        self.resources_frame = ttk.Frame(self.vis_notebook)
        self.activities_frame = ttk.Frame(self.vis_notebook)
        self.paths_frame = ttk.Frame(self.vis_notebook)
        
        self.vis_notebook.add(self.overview_frame, text="Overview")
        self.vis_notebook.add(self.resources_frame, text="Resources")
        self.vis_notebook.add(self.activities_frame, text="Activities")
        self.vis_notebook.add(self.paths_frame, text="Process Paths")
        
        # Add placeholder text for each tab
        for frame in [self.overview_frame, self.resources_frame, 
                    self.activities_frame, self.paths_frame]:
            ttk.Label(
                frame, 
                text="Charts will appear here after running a simulation."
            ).pack(padx=20, pady=40)
            
        # Add export buttons
        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(
            btn_frame,
            text="Export Results",
            command=self.export_results
        ).pack(side="left", padx=5)
        
        ttk.Button(
            btn_frame,
            text="Export Charts",
            command=self.export_charts
        ).pack(side="left", padx=5)
        
        ttk.Button(
            btn_frame,
            text="View Full Report",
            command=self.view_full_report
        ).pack(side="right", padx=5)
        

    def setup_ui(self):
        """Set up the UI components of the tab."""
        # Results text area
        self.results_text = tk.Text(self.frame, wrap="word", height=15, width=80)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.results_text.insert("1.0", "Simulation results will appear here after running a simulation.")
        self.results_text.config(state="disabled")
        
        # Create notebook for results visualization
        self.vis_notebook = ttk.Notebook(self.frame)
        self.vis_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create tabs for different result visualizations
        self.overview_frame = ttk.Frame(self.vis_notebook)
        self.resources_frame = ttk.Frame(self.vis_notebook)
        self.activities_frame = ttk.Frame(self.vis_notebook)
        self.paths_frame = ttk.Frame(self.vis_notebook)
        
        self.vis_notebook.add(self.overview_frame, text="Overview")
        self.vis_notebook.add(self.resources_frame, text="Resources")
        self.vis_notebook.add(self.activities_frame, text="Activities")
        self.vis_notebook.add(self.paths_frame, text="Process Paths")
        
        # Add placeholder text for each tab
        for frame in [self.overview_frame, self.resources_frame, 
                    self.activities_frame, self.paths_frame]:
            ttk.Label(
                frame, 
                text="Charts will appear here after running a simulation."
            ).pack(padx=20, pady=40)
            
        # Add export buttons
        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(
            btn_frame,
            text="Export Results",
            command=self.export_results
        ).pack(side="left", padx=5)
        
        ttk.Button(
            btn_frame,
            text="Export Charts",
            command=self.export_charts
        ).pack(side="left", padx=5)
        
        ttk.Button(
            btn_frame,
            text="View Full Report",
            command=self.view_full_report
        ).pack(side="right", padx=5)
    
    def update_results(self, results: Dict[str, Any]) -> None:
        """
        Update the tab with simulation results.
        
        Args:
            results: Dictionary with simulation results
        """
        self.latest_results = results
        
        # Extract result components
        activity_processing_times = results.get("activity_processing_times", {})
        resource_utilization = results.get("resource_utilization", {})
        total_tokens_started = results.get("total_tokens_started", 0)
        completed_tokens = results.get("completed_tokens", [])
        
        # Update results text
        self._update_results_text(
            activity_processing_times, 
            resource_utilization, 
            total_tokens_started, 
            completed_tokens
        )
        
        # Clear existing charts
        for frame in [self.overview_frame, self.resources_frame, 
                    self.activities_frame, self.paths_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
        
        # Create and display charts
        self._create_overview_charts(
            total_tokens_started, 
            completed_tokens
        )
        
        self._create_resource_charts(resource_utilization)
        
        self._create_activity_charts(activity_processing_times)
        
        self._create_path_analysis(completed_tokens)
    
    def _update_results_text(self, 
                           activity_processing_times: Dict[str, Dict[str, Any]],
                           resource_utilization: Dict[str, float],
                           total_tokens_started: int,
                           completed_tokens: List[Dict[str, Any]]) -> None:
        """
        Update the results text area with simulation statistics.
        
        Args:
            activity_processing_times: Dictionary of activity processing times
            resource_utilization: Dictionary of resource utilization percentages
            total_tokens_started: Total number of tokens that started the process
            completed_tokens: List of completed token data
        """
        # Calculate overall statistics
        results_text = "SIMULATION RESULTS\n"
        results_text += "=================\n\n"
        
        results_text += f"Total tokens started: {total_tokens_started}\n"
        results_text += f"Total tokens completed: {len(completed_tokens)}\n"
        
        if completed_tokens:
            process_durations = [
                (token['end_time'] - token['start_time']).total_seconds() / 60 
                for token in completed_tokens
            ]
            avg_time = sum(process_durations) / len(process_durations)
            min_time = min(process_durations)
            max_time = max(process_durations)
            
            # Format times for display with raw values
            avg_time_formatted = format_duration_for_display(avg_time, include_raw=True)
            min_time_formatted = format_duration_for_display(min_time, include_raw=True)
            max_time_formatted = format_duration_for_display(max_time, include_raw=True)
            
            results_text += f"Average processing time: {avg_time_formatted}\n"
            results_text += f"Minimum processing time: {min_time_formatted}\n"
            results_text += f"Maximum processing time: {max_time_formatted}\n\n"
            
            # Target time comparison if applicable
            if self.config.get("target_avg_time", 0) > 0:
                target = self.config.get("target_avg_time")
                diff = avg_time - target
                target_formatted = format_duration_for_display(target, include_raw=True)
                diff_formatted = format_duration_for_display(abs(diff), include_raw=True)
                
                results_text += f"Target time: {target_formatted}\n"
                diff_direction = "over" if diff > 0 else "under"
                results_text += f"Difference from target: {diff_formatted} {diff_direction} target ({(diff/target)*100:.1f}%)\n\n"
        
        # Resource utilization
        results_text += "RESOURCE UTILIZATION\n"
        results_text += "====================\n"
        for resource, utilization in sorted(
            resource_utilization.items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            results_text += f"{resource}: {utilization:.2f}%\n"
        
        # Update results text
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", results_text)
        self.results_text.config(state="disabled")
    
    def _create_overview_charts(self, 
                              total_tokens_started: int,
                              completed_tokens: List[Dict[str, Any]]) -> None:
        """
        Create overview charts for the simulation results.
        
        Args:
            total_tokens_started: Total number of tokens that started the process
            completed_tokens: List of completed token data
        """
        if not completed_tokens:
            ttk.Label(
                self.overview_frame, 
                text="No completed tokens to display."
            ).pack(padx=20, pady=40)
            return
        
        # Create a grid layout for overview charts
        for i in range(2):
            self.overview_frame.columnconfigure(i, weight=1)
        for i in range(2):
            self.overview_frame.rowconfigure(i, weight=1)
        
        # Create process duration histogram (top left)
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        process_durations = [
            (token['end_time'] - token['start_time']).total_seconds() / 60 
            for token in completed_tokens
        ]
        ax1.hist(process_durations, bins=20, alpha=0.7, color='blue')
        ax1.set_title('Process Duration Distribution')
        ax1.set_xlabel('Duration (minutes)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add formatted time annotations for key values
        mean_duration = sum(process_durations) / len(process_durations)
        median_duration = sorted(process_durations)[len(process_durations) // 2]
        
        ax1.axvline(mean_duration, color='red', linestyle='--')
        ax1.axvline(median_duration, color='green', linestyle='--')
        
        ax1.text(
            mean_duration, ax1.get_ylim()[1] * 0.9, 
            f'Mean: {mean_duration:.1f}m', 
            color='red', ha='right', va='top'
        )
        
        ax1.text(
            median_duration, ax1.get_ylim()[1] * 0.8, 
            f'Median: {median_duration:.1f}m', 
            color='green', ha='right', va='top'
        )
        
        # Create a frame for the histogram
        chart_frame1 = ttk.Frame(self.overview_frame)
        chart_frame1.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Add the figure to the frame
        canvas1 = FigureCanvasTkAgg(fig1, master=chart_frame1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create completion rate donut chart (top right)
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        completion_rate = len(completed_tokens) / total_tokens_started * 100
        incomplete_rate = 100 - completion_rate
        
        ax2.pie(
            [completion_rate, incomplete_rate],
            labels=['Completed', 'Incomplete'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#5cb85c', '#d9534f'],
            wedgeprops=dict(width=0.3, edgecolor='w')
        )
        ax2.set_title('Process Completion Rate')
        ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Create a frame for the donut chart
        chart_frame2 = ttk.Frame(self.overview_frame)
        chart_frame2.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Add the figure to the frame
        canvas2 = FigureCanvasTkAgg(fig2, master=chart_frame2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create wait time vs duration scatter plot (bottom left)
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        
        wait_times = [token['total_wait_time'] for token in completed_tokens]
        
        ax3.scatter(process_durations, wait_times, alpha=0.7)
        ax3.set_title('Process Duration vs Wait Time')
        ax3.set_xlabel('Duration (minutes)')
        ax3.set_ylabel('Wait Time (minutes)')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Add a trend line
        if len(process_durations) > 1:
            coeffs = np.polyfit(process_durations, wait_times, 1)
            trend_line = np.poly1d(coeffs)
            ax3.plot(
                process_durations, 
                trend_line(process_durations), 
                'r--', 
                alpha=0.7
            )
        
        # Create a frame for the scatter plot
        chart_frame3 = ttk.Frame(self.overview_frame)
        chart_frame3.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        # Add the figure to the frame
        canvas3 = FigureCanvasTkAgg(fig3, master=chart_frame3)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create summary statistics (bottom right)
        stats_frame = ttk.LabelFrame(self.overview_frame, text="Process Statistics")
        stats_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        avg_duration = sum(process_durations) / len(process_durations)
        avg_wait = sum(wait_times) / len(wait_times)
        wait_percentage = (avg_wait / avg_duration) * 100 if avg_duration > 0 else 0
        
        stats_text = f"""
        Process Metrics:
        
        • Total Tokens: {total_tokens_started}
        • Completed: {len(completed_tokens)}
        • Completion Rate: {completion_rate:.1f}%
        
        Time Metrics:
        
        • Avg Duration: {avg_duration:.2f} minutes
        • Avg Wait Time: {avg_wait:.2f} minutes
        • Wait Time %: {wait_percentage:.1f}% of total
        """
        
        ttk.Label(
            stats_frame, 
            text=stats_text, 
            justify="left"
        ).pack(padx=10, pady=10, anchor="nw")
    
    def _create_resource_charts(self, resource_utilization: Dict[str, float]) -> None:
        """
        Create resource-related charts.
        
        Args:
            resource_utilization: Dictionary of resource utilization percentages
        """
        if not resource_utilization:
            ttk.Label(
                self.resources_frame, 
                text="No resource data to display."
            ).pack(padx=20, pady=40)
            return
        
        # Create a frame for the utilization chart
        chart_frame = ttk.Frame(self.resources_frame)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Sort resources by utilization for better visualization
        resources = []
        utils = []
        for res, util in sorted(resource_utilization.items(), key=lambda x: x[1], reverse=True):
            resources.append(res)
            utils.append(util)
        
        # Create the chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(resources, utils, color=['green' if u < 80 else 'orange' if u < 95 else 'red' for u in utils])
        
        ax.set_title('Resource Utilization')
        ax.set_xlabel('Resource')
        ax.set_ylabel('Utilization (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom'
            )
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        fig.tight_layout()
        
        # Add the figure to the frame
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add resource utilization interpretation
        interp_frame = ttk.LabelFrame(
            self.resources_frame, 
            text="Resource Utilization Interpretation"
        )
        interp_frame.pack(fill="x", padx=10, pady=10)
        
        # Color legend
        legend_frame = ttk.Frame(interp_frame)
        legend_frame.pack(fill="x", padx=5, pady=5)
        
        # Green square (good utilization)
        green_canvas = tk.Canvas(legend_frame, width=15, height=15, bg="green")
        green_canvas.grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(
            legend_frame, 
            text="Good utilization (<80%)"
        ).grid(row=0, column=1, sticky="w")
        
        # Orange square (high utilization)
        orange_canvas = tk.Canvas(legend_frame, width=15, height=15, bg="orange")
        orange_canvas.grid(row=1, column=0, padx=5, pady=2)
        ttk.Label(
            legend_frame, 
            text="High utilization (80-95%)"
        ).grid(row=1, column=1, sticky="w")
        
        # Red square (critical utilization)
        red_canvas = tk.Canvas(legend_frame, width=15, height=15, bg="red")
        red_canvas.grid(row=2, column=0, padx=5, pady=2)
        ttk.Label(
            legend_frame, 
            text="Critical utilization (>95%)"
        ).grid(row=2, column=1, sticky="w")
        
        # Find overutilized and underutilized resources
        overutilized = [res for res, util in resource_utilization.items() if util > 90]
        underutilized = [res for res, util in resource_utilization.items() if util < 50]
        
        # Create interpretation text
        if overutilized:
            overutilized_text = f"Potential bottlenecks: {', '.join(overutilized)}"
        else:
            overutilized_text = "No potential bottlenecks detected."
            
        if underutilized:
            underutilized_text = f"Underutilized resources: {', '.join(underutilized)}"
        else:
            underutilized_text = "No significantly underutilized resources."
        
        ttk.Label(
            interp_frame, 
            text=overutilized_text,
            foreground="red" if overutilized else "black"
        ).pack(anchor="w", padx=10, pady=2)
        
        ttk.Label(
            interp_frame, 
            text=underutilized_text,
            foreground="blue" if underutilized else "black"
        ).pack(anchor="w", padx=10, pady=2)
    
    def _create_activity_charts(self, activity_processing_times: Dict[str, Dict[str, Any]]) -> None:
        """
        Create activity-related charts.
        
        Args:
            activity_processing_times: Dictionary of activity processing times
        """
        if not activity_processing_times:
            ttk.Label(
                self.activities_frame, 
                text="No activity data to display."
            ).pack(padx=20, pady=40)
            return
        
        # Create a frame for activity completion rates
        frame1 = ttk.LabelFrame(self.activities_frame, text="Activity Completion Rates")
        frame1.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Prepare data for completion rate chart
        activities = []
        started = []
        completed = []
        
        for activity, data in activity_processing_times.items():
            activities.append(activity)
            started.append(data.get("tokens_started", 0))
            completed.append(data.get("tokens_completed", 0))
        
        # Create completion rate chart
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        
        x = range(len(activities))
        bar_width = 0.35
        
        bars1 = ax1.bar(
            [i - bar_width/2 for i in x], 
            started, 
            bar_width, 
            label='Started'
        )
        
        bars2 = ax1.bar(
            [i + bar_width/2 for i in x], 
            completed, 
            bar_width, 
            label='Completed'
        )
        
        ax1.set_title('Activity Token Counts')
        ax1.set_xticks(x)
        ax1.set_xticklabels(activities, rotation=45, ha='right')
        ax1.legend()
        
        fig1.tight_layout()
        
        # Add the figure to the frame
        canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for activity times
        frame2 = ttk.LabelFrame(self.activities_frame, text="Activity Processing Times")
        frame2.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Prepare data for time chart
        activities = []
        avg_times = []
        wait_times = []
        
        for activity, data in activity_processing_times.items():
            if data.get("durations") and len(data["durations"]) > 0:
                durations = data["durations"]
                activities.append(activity)
                avg_times.append(sum(durations) / len(durations))
                
                if "wait_times" in data and len(data["wait_times"]) > 0:
                    wait_times.append(sum(data["wait_times"]) / len(data["wait_times"]))
                else:
                    wait_times.append(0)
        
        # Sort by avg_times for better visualization
        if activities:
            # Sort all three lists by avg_times
            sorted_data = sorted(
                zip(activities, avg_times, wait_times), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Unpack the sorted data
            activities, avg_times, wait_times = zip(*sorted_data)
            
            # Limit to top 10 for readability
            activities = activities[:10]
            avg_times = avg_times[:10]
            wait_times = wait_times[:10]
            
            # Create time chart
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            
            x = range(len(activities))
            bar_width = 0.35
            
            bars1 = ax2.bar(
                [i - bar_width/2 for i in x], 
                avg_times, 
                bar_width, 
                label='Processing Time'
            )
            
            bars2 = ax2.bar(
                [i + bar_width/2 for i in x], 
                wait_times, 
                bar_width, 
                label='Wait Time'
            )
            
            ax2.set_title('Top 10 Activities by Processing Time')
            ax2.set_xticks(x)
            ax2.set_xticklabels(activities, rotation=45, ha='right')
            ax2.set_ylabel('Time (minutes)')
            ax2.legend()
            
            fig2.tight_layout()
            
            # Add the figure to the frame
            canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(
                frame2, 
                text="No activity duration data available."
            ).pack(padx=20, pady=20)
    
    def _create_path_analysis(self, completed_tokens: List[Dict[str, Any]]) -> None:
        """
        Create path analysis charts.
        
        Args:
            completed_tokens: List of completed token data
        """
        if not completed_tokens:
            ttk.Label(
                self.paths_frame, 
                text="No token path data to display."
            ).pack(padx=20, pady=40)
            return
        
        # Extract paths from completed tokens
        paths = [tuple(token.get('path', [])) for token in completed_tokens if 'path' in token]
        
        if not paths:
            ttk.Label(
                self.paths_frame, 
                text="No path data available in tokens."
            ).pack(padx=20, pady=40)
            return
        
        # Count path frequencies
        path_counts = {}
        for path in paths:
            path_str = ' -> '.join(path)
            path_counts[path_str] = path_counts.get(path_str, 0) + 1
        
        # Create a frame for path frequencies
        frame = ttk.Frame(self.paths_frame)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Sort paths by frequency
        sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to top 10 for readability
        top_paths = sorted_paths[:10]
        
        # Create path frequency chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        path_labels = [p[0] if len(p[0]) < 50 else p[0][:47] + '...' for p in top_paths]
        frequencies = [p[1] for p in top_paths]
        
        bars = ax.barh(path_labels, frequencies, color='skyblue')
        
        ax.set_title('Most Common Process Paths')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Path')
        
        # Add frequency labels to bars
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.3, 
                bar.get_y() + bar.get_height()/2,
                f'{int(width)}',
                ha='left', 
                va='center'
            )
        
        fig.tight_layout()
        
        # Add the figure to the frame
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add path analysis text
        analysis_frame = ttk.LabelFrame(self.paths_frame, text="Path Analysis")
        analysis_frame.pack(fill="x", padx=10, pady=10)
        
        # Calculate some statistics
        total_tokens = len(paths)
        most_common_path, most_common_count = sorted_paths[0]
        most_common_percentage = (most_common_count / total_tokens) * 100
        
        unique_paths = len(path_counts)
        path_diversity = (unique_paths / total_tokens) * 100 if total_tokens > 0 else 0
        
        analysis_text = f"""
        Path Statistics:
        
        • Total completed tokens: {total_tokens}
        • Unique paths observed: {unique_paths}
        • Path diversity: {path_diversity:.1f}%
        • Most common path: {most_common_percentage:.1f}% of tokens
        
        {most_common_path}
        """
        
        ttk.Label(
            analysis_frame, 
            text=analysis_text,
            justify="left"
        ).pack(padx=10, pady=10, anchor="nw")
    
    def export_results(self) -> None:
        """Export simulation results to a CSV file."""
        if not self.latest_results:
            tk.messagebox.showinfo("Export Results", "No simulation results to export.")
            return
        
        # Ask user for file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Results to CSV"
        )
        
        if not file_path:
            return  # User cancelled
            
        try:
            # Extract results
            completed_tokens = self.latest_results.get("completed_tokens", [])
            
            # Create a DataFrame
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
            
            df = pd.DataFrame(token_data)
            df.to_csv(file_path, index=False)
            
            tk.messagebox.showinfo(
                "Export Results", 
                f"Results successfully exported to {file_path}"
            )
        
        except Exception as e:
            tk.messagebox.showerror(
                "Export Error", 
                f"An error occurred while exporting results: {str(e)}"
            )
    
    def export_charts(self) -> None:
        """Export charts to image files."""
        if not self.latest_results:
            tk.messagebox.showinfo("Export Charts", "No simulation results to export charts from.")
            return
        
        # Ask user for directory location
        dir_path = filedialog.askdirectory(
            title="Select Directory for Exported Charts"
        )
        
        if not dir_path:
            return  # User cancelled
            
        try:
            # Save a screenshot of each chart
            for i, frame in enumerate([
                self.overview_frame, 
                self.resources_frame, 
                self.activities_frame, 
                self.paths_frame
            ]):
                for widget in frame.winfo_children():
                    if isinstance(widget, FigureCanvasTkAgg):
                        fig = widget.figure
                        file_name = f"chart_{i}_{id(widget)}.png"
                        file_path = os.path.join(dir_path, file_name)
                        fig.savefig(file_path, dpi=300, bbox_inches='tight')
            
            tk.messagebox.showinfo(
                "Export Charts", 
                f"Charts successfully exported to {dir_path}"
            )
        
        except Exception as e:
            tk.messagebox.showerror(
                "Export Error", 
                f"An error occurred while exporting charts: {str(e)}"
            )
    
    def view_full_report(self) -> None:
        """View the full simulation report."""
        # This would typically open the Excel report
        base_filename = os.path.splitext(
            os.path.basename(self.config.get("xpdl_file_path", ""))
        )[0]
        report_path = f"{base_filename}_results.xlsx"
        
        if not os.path.exists(report_path):
            tk.messagebox.showinfo(
                "View Report", 
                "Report file not found. Run a simulation first."
            )
            return
            
        # Try to open the report with the default application
        try:
            import subprocess
            import platform
            
            if platform.system() == 'Windows':
                os.startfile(report_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', report_path))
            else:  # Linux
                subprocess.call(('xdg-open', report_path))
                
        except Exception as e:
            tk.messagebox.showerror(
                "Open Report Error", 
                f"Could not open report: {str(e)}"
            )
