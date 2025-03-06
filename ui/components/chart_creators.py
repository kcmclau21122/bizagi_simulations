# chart_creators.py - Chart Creation Classes
# ------------------------------------------------------------
import sys
import os
# Add project root to sys.path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Dict, Any, List, Tuple

from .ui_components import ScrollableFrame

class BaseChartCreator:
    """Base class for all chart creators."""
    
    def _create_scrollable_frame(self, parent):
        """
        Create a scrollable frame.
        
        Args:
            parent: Parent widget
            
        Returns:
            Tuple containing the container frame and the inner frame for content
        """
        # Create a scrollable frame container
        scrollable_container = ScrollableFrame(parent)
        scrollable_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Return the container and its scrollable inner frame
        return scrollable_container, scrollable_container.scrollable_frame


class OverviewChartCreator(BaseChartCreator):
    """Creates overview charts for the simulation results."""
    
    def create_charts(self, parent, total_tokens_started, completed_tokens):
        """Create overview charts."""
        if not completed_tokens:
            ttk.Label(
                parent, 
                text="No completed tokens to display."
            ).pack(padx=20, pady=40)
            return
        
        # Create a scrollable frame for the overview content
        overview_container, scrollable_frame = self._create_scrollable_frame(parent)
        
        # Configure grid layout
        self._setup_grid(scrollable_frame)
        
        # Create process duration histogram
        self._create_duration_histogram(scrollable_frame, completed_tokens)
        
        # Create completion rate donut chart
        self._create_completion_chart(scrollable_frame, total_tokens_started, completed_tokens)
        
        # Create wait time vs duration scatter plot
        self._create_wait_time_chart(scrollable_frame, completed_tokens)
        
        # Create summary statistics
        self._create_statistics_panel(scrollable_frame, total_tokens_started, completed_tokens)
    
    def _setup_grid(self, frame):
        """Configure the grid layout for charts."""
        for i in range(2):
            frame.columnconfigure(i, weight=1)
        for i in range(2):
            frame.rowconfigure(i, weight=1)
    
    def _create_duration_histogram(self, frame, completed_tokens):
        """Create process duration histogram chart."""
        fig, ax = plt.subplots(figsize=(5, 3))
        process_durations = [
            (token['end_time'] - token['start_time']).total_seconds() / 60 
            for token in completed_tokens
        ]
        ax.hist(process_durations, bins=20, alpha=0.7, color='blue')
        ax.set_title('Process Duration Distribution')
        ax.set_xlabel('Duration (minutes)')
        ax.set_ylabel('Frequency')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add annotations
        mean_duration = sum(process_durations) / len(process_durations)
        median_duration = sorted(process_durations)[len(process_durations) // 2]
        
        ax.axvline(mean_duration, color='red', linestyle='--')
        ax.axvline(median_duration, color='green', linestyle='--')
        
        ax.text(
            mean_duration, ax.get_ylim()[1] * 0.9, 
            f'Mean: {mean_duration:.1f}m', 
            color='red', ha='right', va='top'
        )
        
        ax.text(
            median_duration, ax.get_ylim()[1] * 0.8, 
            f'Median: {median_duration:.1f}m', 
            color='green', ha='right', va='top'
        )
        
        # Add to frame
        chart_frame = ttk.Frame(frame)
        chart_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_completion_chart(self, frame, total_tokens_started, completed_tokens):
        """Create completion rate donut chart."""
        fig, ax = plt.subplots(figsize=(5, 3))
        completion_rate = len(completed_tokens) / total_tokens_started * 100
        incomplete_rate = 100 - completion_rate
        
        ax.pie(
            [completion_rate, incomplete_rate],
            labels=['Completed', 'Incomplete'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#5cb85c', '#d9534f'],
            wedgeprops=dict(width=0.3, edgecolor='w')
        )
        ax.set_title('Process Completion Rate')
        ax.axis('equal')
        
        # Add to frame
        chart_frame = ttk.Frame(frame)
        chart_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_wait_time_chart(self, frame, completed_tokens):
        """Create wait time vs duration scatter plot."""
        fig, ax = plt.subplots(figsize=(5, 3))
        
        process_durations = [
            (token['end_time'] - token['start_time']).total_seconds() / 60 
            for token in completed_tokens
        ]
        wait_times = [token['total_wait_time'] for token in completed_tokens]
        
        ax.scatter(process_durations, wait_times, alpha=0.7)
        ax.set_title('Process Duration vs Wait Time')
        ax.set_xlabel('Duration (minutes)')
        ax.set_ylabel('Wait Time (minutes)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add trend line
        if len(process_durations) > 1:
            coeffs = np.polyfit(process_durations, wait_times, 1)
            trend_line = np.poly1d(coeffs)
            ax.plot(
                process_durations, 
                trend_line(process_durations), 
                'r--', 
                alpha=0.7
            )
        
        # Add to frame
        chart_frame = ttk.Frame(frame)
        chart_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_statistics_panel(self, frame, total_tokens_started, completed_tokens):
        """Create statistics summary panel."""
        process_durations = [
            (token['end_time'] - token['start_time']).total_seconds() / 60 
            for token in completed_tokens
        ]
        wait_times = [token['total_wait_time'] for token in completed_tokens]
        
        avg_duration = sum(process_durations) / len(process_durations)
        avg_wait = sum(wait_times) / len(wait_times)
        wait_percentage = (avg_wait / avg_duration) * 100 if avg_duration > 0 else 0
        completion_rate = len(completed_tokens) / total_tokens_started * 100
        
        stats_frame = ttk.LabelFrame(frame, text="Process Statistics")
        stats_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
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


class ResourceChartCreator(BaseChartCreator):
    """Creates resource utilization charts."""
    
    def create_charts(self, parent, resource_utilization):
        """Create resource-related charts."""
        if not resource_utilization:
            ttk.Label(
                parent, 
                text="No resource data to display."
            ).pack(padx=20, pady=40)
            return
        
        # Create a scrollable frame
        resource_container, scrollable_frame = self._create_scrollable_frame(parent)
        
        # Create utilization chart
        self._create_utilization_chart(scrollable_frame, resource_utilization)
        
        # Create interpretation panel
        self._create_interpretation_panel(scrollable_frame, resource_utilization)
    
    def _create_utilization_chart(self, frame, resource_utilization):
        """Create resource utilization chart."""
        chart_frame = ttk.Frame(frame)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Sort resources by utilization
        resources = []
        utils = []
        for res, util in sorted(resource_utilization.items(), key=lambda x: x[1], reverse=True):
            resources.append(res)
            utils.append(util)
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(resources, utils, color=['green' if u < 80 else 'orange' if u < 95 else 'red' for u in utils])
        
        ax.set_title('Resource Utilization')
        ax.set_xlabel('Resource')
        ax.set_ylabel('Utilization (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom'
            )
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        fig.tight_layout()
        
        # Add to frame
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_interpretation_panel(self, frame, resource_utilization):
        """Create resource utilization interpretation panel."""
        interp_frame = ttk.LabelFrame(frame, text="Resource Utilization Interpretation")
        interp_frame.pack(fill="x", padx=10, pady=10)
        
        # Create color legend
        legend_frame = ttk.Frame(interp_frame)
        legend_frame.pack(fill="x", padx=5, pady=5)
        
        # Green legend item
        green_canvas = tk.Canvas(legend_frame, width=15, height=15, bg="green")
        green_canvas.grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(legend_frame, text="Good utilization (<80%)").grid(row=0, column=1, sticky="w")
        
        # Orange legend item
        orange_canvas = tk.Canvas(legend_frame, width=15, height=15, bg="orange")
        orange_canvas.grid(row=1, column=0, padx=5, pady=2)
        ttk.Label(legend_frame, text="High utilization (80-95%)").grid(row=1, column=1, sticky="w")
        
        # Red legend item
        red_canvas = tk.Canvas(legend_frame, width=15, height=15, bg="red")
        red_canvas.grid(row=2, column=0, padx=5, pady=2)
        ttk.Label(legend_frame, text="Critical utilization (>95%)").grid(row=2, column=1, sticky="w")
        
        # Identify bottlenecks
        overutilized = [res for res, util in resource_utilization.items() if util > 90]
        underutilized = [res for res, util in resource_utilization.items() if util < 50]
        
        # Create interpretation text
        overutilized_text = f"Potential bottlenecks: {', '.join(overutilized)}" if overutilized else "No potential bottlenecks detected."
        underutilized_text = f"Underutilized resources: {', '.join(underutilized)}" if underutilized else "No significantly underutilized resources."
        
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


class ActivityChartCreator(BaseChartCreator):
    """Creates activity charts."""
    
    def create_charts(self, parent, activity_processing_times):
        """Create activity-related charts."""
        if not activity_processing_times:
            ttk.Label(
                parent, 
                text="No activity data to display."
            ).pack(padx=20, pady=40)
            return
        
        # Create a scrollable frame
        activities_container, scrollable_frame = self._create_scrollable_frame(parent)
        
        # Create activity completion chart
        self._create_completion_chart(scrollable_frame, activity_processing_times)
        
        # Create activity times chart
        self._create_times_chart(scrollable_frame, activity_processing_times)
    
    def _create_completion_chart(self, frame, activity_processing_times):
        """Create activity completion rates chart."""
        frame1 = ttk.LabelFrame(frame, text="Activity Completion Rates")
        frame1.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Prepare data
        activities = []
        started = []
        completed = []
        
        for activity, data in activity_processing_times.items():
            activities.append(activity)
            started.append(data.get("tokens_started", 0))
            completed.append(data.get("tokens_completed", 0))
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 5))
        
        x = range(len(activities))
        bar_width = 0.35
        
        bars1 = ax.bar(
            [i - bar_width/2 for i in x], 
            started, 
            bar_width, 
            label='Started'
        )
        
        bars2 = ax.bar(
            [i + bar_width/2 for i in x], 
            completed, 
            bar_width, 
            label='Completed'
        )
        
        ax.set_title('Activity Token Counts')
        ax.set_xticks(x)
        ax.set_xticklabels(activities, rotation=45, ha='right')
        ax.legend()
        
        fig.tight_layout()
        
        # Add to frame
        canvas = FigureCanvasTkAgg(fig, master=frame1)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_times_chart(self, frame, activity_processing_times):
        """Create activity processing times chart."""
        frame2 = ttk.LabelFrame(frame, text="Activity Processing Times")
        frame2.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Prepare data
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
        
        # Sort and limit to top 10
        if activities:
            # Sort all three lists
            sorted_data = sorted(
                zip(activities, avg_times, wait_times), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Unpack the sorted data
            activities, avg_times, wait_times = zip(*sorted_data)
            
            # Limit to top 10
            activities = activities[:10]
            avg_times = avg_times[:10]
            wait_times = wait_times[:10]
            
            # Create chart
            fig, ax = plt.subplots(figsize=(10, 5))
            
            x = range(len(activities))
            bar_width = 0.35
            
            bars1 = ax.bar(
                [i - bar_width/2 for i in x], 
                avg_times, 
                bar_width, 
                label='Processing Time'
            )
            
            bars2 = ax.bar(
                [i + bar_width/2 for i in x], 
                wait_times, 
                bar_width, 
                label='Wait Time'
            )
            
            ax.set_title('Top 10 Activities by Processing Time')
            ax.set_xticks(x)
            ax.set_xticklabels(activities, rotation=45, ha='right')
            ax.set_ylabel('Time (minutes)')
            ax.legend()
            
            fig.tight_layout()
            
            # Add to frame
            canvas = FigureCanvasTkAgg(fig, master=frame2)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(
                frame2, 
                text="No activity duration data available."
            ).pack(padx=20, pady=20)


class PathChartCreator(BaseChartCreator):
    """Creates process path charts."""
    
    def create_charts(self, parent, completed_tokens):
        """Create path analysis charts."""
        if not completed_tokens:
            ttk.Label(
                parent, 
                text="No token path data to display."
            ).pack(padx=20, pady=40)
            return
        
        # Create a scrollable frame
        paths_container, scrollable_frame = self._create_scrollable_frame(parent)
        
        # Extract path data
        paths = [tuple(token.get('path', [])) for token in completed_tokens if 'path' in token]
        
        if not paths:
            ttk.Label(
                scrollable_frame, 
                text="No path data available in tokens."
            ).pack(padx=20, pady=40)
            return
        
        # Create frequency chart
        self._create_frequency_chart(scrollable_frame, paths)
        
        # Create path analysis panel
        self._create_analysis_panel(scrollable_frame, paths)
    
    def _create_frequency_chart(self, frame, paths):
        """Create path frequency chart."""
        chart_frame = ttk.Frame(frame)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Count paths
        path_counts = {}
        for path in paths:
            path_str = ' -> '.join(path)
            path_counts[path_str] = path_counts.get(path_str, 0) + 1
        
        # Sort and limit
        sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
        top_paths = sorted_paths[:10]
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        path_labels = [p[0] if len(p[0]) < 50 else p[0][:47] + '...' for p in top_paths]
        frequencies = [p[1] for p in top_paths]
        
        bars = ax.barh(path_labels, frequencies, color='skyblue')
        
        ax.set_title('Most Common Process Paths')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Path')
        
        # Add labels
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
        
        # Add to frame
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_analysis_panel(self, frame, paths):
        """Create path analysis panel."""
        analysis_frame = ttk.LabelFrame(frame, text="Path Analysis")
        analysis_frame.pack(fill="x", padx=10, pady=10)
        
        # Count paths
        path_counts = {}
        for path in paths:
            path_str = ' -> '.join(path)
            path_counts[path_str] = path_counts.get(path_str, 0) + 1
        
        # Calculate statistics
        total_tokens = len(paths)
        sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
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


