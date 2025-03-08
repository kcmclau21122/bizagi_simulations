import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import timedelta
import subprocess
import platform
from typing import Dict, Any, List, Optional

# Add project root to sys.path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import ConfigManager

# Define ScrollableFrame class directly to avoid import issues
class ScrollableFrame(ttk.Frame):
    """
    A base frame that provides scrolling capabilities.
    """
    
    def __init__(self, parent, **kwargs):
        """
        Initialize the scrollable frame.
        
        Args:
            parent: Parent widget
            **kwargs: Additional keyword arguments for Frame
        """
        super().__init__(parent, **kwargs)
        
        # Create a canvas for scrolling
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        
        # Create the scrollable frame
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window inside canvas
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configure canvas to expand with the frame
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack widgets
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Configure canvas to expand with window
        self.bind("<Configure>", self._on_frame_configure)
        
        # Mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
    def _on_frame_configure(self, event=None):
        """Handle frame resize event."""
        # Update the canvas width to match the frame
        self.canvas.configure(width=self.winfo_width())
        
        # Ensure the inner frame expands to fill the canvas width
        self.canvas.itemconfig(self.canvas_window, width=self.canvas.winfo_width())
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        # The event.delta value is negative when scrolling down, positive when scrolling up
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def unbind_mousewheel(self):
        """Unbind the mousewheel event when the frame loses focus."""
        self.canvas.unbind_all("<MouseWheel>")
        
    def rebind_mousewheel(self):
        """Rebind the mousewheel event when the frame gains focus."""
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

class EnhancedResultsTab:
    """
    Enhanced tab for displaying simulation results in a tabular format.
    Shows process statistics in a format similar to Excel, with time values
    displayed as days, hours, minutes, and seconds.
    
    Instead of tabs, uses buttons to open visualization files directly.
    """
    
    def __init__(self, parent: ttk.Notebook, config: ConfigManager):
        """
        Initialize the enhanced results tab.
        
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
        self.latest_df = None
        
        # Store visualization file paths
        self.visualization_paths = {}
    
    def setup_ui(self):
        """Set up the UI components of the tab."""
        # Create a PanedWindow for resizable sections
        self.paned_window = ttk.PanedWindow(self.frame, orient=tk.VERTICAL)
        self.paned_window.pack(fill="both", expand=True)
        
        # Create top frame for summary statistics
        summary_container = ttk.Frame(self.paned_window)
        self.paned_window.add(summary_container, weight=1)
        
        summary_frame = ttk.LabelFrame(summary_container, text="Simulation Summary")
        summary_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add vertical scrollbar for summary
        summary_scroll = ttk.Scrollbar(summary_frame)
        summary_scroll.pack(side="right", fill="y")
        
        # Results text area with scrollbar
        self.summary_text = tk.Text(summary_frame, wrap="word", height=5, width=80, yscrollcommand=summary_scroll.set)
        self.summary_text.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.summary_text.insert("1.0", "Simulation results will appear here after running a simulation.")
        self.summary_text.config(state="disabled")
        
        # Configure scrollbar to scroll the text
        summary_scroll.config(command=self.summary_text.yview)
        
        # Create a frame for the table view
        table_container = ttk.Frame(self.paned_window)
        self.paned_window.add(table_container, weight=3)
        
        # Add a LabelFrame for the table
        table_frame = ttk.LabelFrame(table_container, text="Activity Times")
        table_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create a frame for the table with scrollbars
        tree_frame = ttk.Frame(table_frame)
        tree_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create container frame for treeview and scrollbars
        self.tree_container = ttk.Frame(tree_frame)
        self.tree_container.pack(fill="both", expand=True)
        
        # Add scrollbars
        self.tree_vsb = ttk.Scrollbar(self.tree_container, orient="vertical")
        self.tree_hsb = ttk.Scrollbar(self.tree_container, orient="horizontal")
        
        # Create the treeview widget for tabular data with fixed horizontal scrollbar
        self.tree = ttk.Treeview(
            self.tree_container, 
            columns=(), 
            show="headings",
            yscrollcommand=self.tree_vsb.set,
            xscrollcommand=self.tree_hsb.set
        )
        
        # Configure scrollbars
        self.tree_vsb.config(command=self.tree.yview)
        self.tree_hsb.config(command=self.tree.xview)
        
        # Pack widgets - order is important for scrollbars
        self.tree_hsb.pack(side="bottom", fill="x")
        self.tree_vsb.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)
        
        # Bind focus and mouse events to ensure scrollbars work properly
        self.tree.bind("<FocusIn>", self._on_tree_focus)
        self.tree.bind("<Button-1>", self._on_tree_click)
        self.tree.bind("<ButtonRelease-1>", self._on_tree_release)
        
        # Add buttons frame at the bottom
        buttons_frame = ttk.Frame(self.frame)
        buttons_frame.pack(fill="x", padx=10, pady=5)
        
        # Add export button
        self.export_excel_btn = ttk.Button(
            buttons_frame,
            text="Export to Excel",
            command=self.export_to_excel
        )
        self.export_excel_btn.pack(side="right", padx=5)
        
        # Add view visualization buttons (initially disabled)
        self.resource_btn = ttk.Button(
            buttons_frame,
            text="View Resource Utilization",
            command=lambda: self.open_visualization("resource_utilization"),
            state="disabled"
        )
        self.resource_btn.pack(side="left", padx=5)
        
        self.activity_btn = ttk.Button(
            buttons_frame,
            text="View Activity Times",
            command=lambda: self.open_visualization("activity_times"),
            state="disabled"
        )
        self.activity_btn.pack(side="left", padx=5)
        
        self.token_btn = ttk.Button(
            buttons_frame,
            text="View Token Distribution",
            command=lambda: self.open_visualization("token_histogram"),
            state="disabled"
        )
        self.token_btn.pack(side="left", padx=5)
        
        self.duration_btn = ttk.Button(
            buttons_frame,
            text="View Duration vs Wait",
            command=lambda: self.open_visualization("duration_vs_wait"),
            state="disabled"
        )
        self.duration_btn.pack(side="left", padx=5)
        
    def _set_button_states(self, state):
        """Set the state of all results buttons."""
        for btn in [
            self.resource_btn, 
            self.activity_btn, 
            self.token_btn, 
            self.duration_btn,
            self.export_excel_btn
        ]:
            btn.config(state=state)
            
    def _on_tree_focus(self, event):
        """Handle treeview focus events to ensure scrollbars work."""
        # When the treeview gets focus, ensure scrollbars are visible
        self._update_scrollbar_visibility()
        
    def _on_tree_click(self, event):
        """Handle treeview click events."""
        # When clicking the treeview, ensure scrollbars are visible
        self._update_scrollbar_visibility()
        
    def _on_tree_release(self, event):
        """Handle mouse button release events."""
        # After releasing mouse button, ensure scrollbars are visible
        self._update_scrollbar_visibility()
        
    def _update_scrollbar_visibility(self):
        """Update scrollbar visibility to ensure horizontal scrollbar is always shown."""
        # Force a redraw of the scrollbars
        self.tree.update_idletasks()
        
        # Check if the treeview contains data and needs a horizontal scrollbar
        if len(self.tree["columns"]) > 0:
            # Set horizontal scrollbar values to ensure it's visible
            total_width = sum(int(self.tree.column(col, "width")) for col in self.tree["columns"])
            visible_width = self.tree_container.winfo_width()
            
            # If content is wider than visible area, show scrollbar
            if total_width > visible_width:
                # Force the horizontal scrollbar to be visible by setting its values
                # First parameter is the position, second is how much is visible
                # Setting it to values less than 1.0 ensures the scrollbar is shown
                self.tree_hsb.set(0.0, 0.9)
    
    def format_time(self, minutes: float, include_seconds: bool = True) -> str:
        """
        Format time in minutes to days, hours, minutes, and seconds.
        
        Args:
            minutes: Time in minutes
            include_seconds: Whether to include seconds in the formatted string
            
        Returns:
            Formatted time string
        """
        if minutes is None or pd.isna(minutes) or minutes == 0:
            return "0s"
        
        total_seconds = int(minutes * 60)
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        mins, secs = divmod(remainder, 60)
        
        time_parts = []
        if days > 0:
            time_parts.append(f"{days}d")
        if hours > 0 or days > 0:
            time_parts.append(f"{hours}h")
        if mins > 0 or hours > 0 or days > 0:
            time_parts.append(f"{mins}m")
        if include_seconds and (secs > 0 or not time_parts):
            time_parts.append(f"{secs}s")
            
        return " ".join(time_parts) if time_parts else "0s"
    
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
        transitions_df = results.get("transitions_df", pd.DataFrame())
        
        # Get visualization paths from results if available
        self.visualization_paths = results.get("visualization_paths", {})
        
        # Debug print to verify the paths
        print("Visualization paths received:", self.visualization_paths)
        
        # Update summary text
        self._update_summary_text(
            activity_processing_times, 
            resource_utilization, 
            total_tokens_started, 
            completed_tokens
        )
        
        # Update the table with activity times
        self._update_activity_table(
            activity_processing_times,
            transitions_df,
            total_tokens_started,
            completed_tokens
        )
        
        # Make sure scrollbars update after adding data
        self._update_scrollbar_visibility()
        
        # Enable all buttons when results are available
        self._set_button_states("normal")
    
    def open_visualization(self, viz_type: str) -> None:
        """
        Open a visualization file using the system's default application.
        If the file doesn't exist, create it on demand.
        
        Args:
            viz_type: Type of visualization to open
        """
        # If we have a file path and it exists, open it
        if viz_type in self.visualization_paths and os.path.exists(self.visualization_paths[viz_type]):
            self._open_file(self.visualization_paths[viz_type])
            return
            
        # Otherwise generate the visualization on demand
        if not self.latest_results:
            messagebox.showinfo("No Data", "No simulation results available for visualization.")
            return
            
        try:
            # Generate the visualization based on type
            file_path = self._generate_visualization(viz_type)
            
            if file_path:
                # Store the path and open the file
                self.visualization_paths[viz_type] = file_path
                self._open_file(file_path)
            else:
                messagebox.showinfo(
                    "Visualization Failed", 
                    f"Could not generate the {viz_type} visualization."
                )
        except Exception as e:
            messagebox.showerror(
                "Visualization Error", 
                f"An error occurred while creating visualization: {str(e)}"
            )
    
    def _open_file(self, file_path: str) -> None:
        """
        Open a file with the system's default application.
        
        Args:
            file_path: Path to the file to open
        """
        try:
            # Open the file with the default application based on the operating system
            if platform.system() == 'Windows':
                os.startfile(file_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(['open', file_path])
            else:  # Linux and other Unix-like systems
                subprocess.call(['xdg-open', file_path])
                
            # Log that the file was opened
            print(f"Opened visualization: {file_path}")
            
        except Exception as e:
            messagebox.showerror(
                "Error Opening File", 
                f"An error occurred while trying to open the file: {str(e)}"
            )
            
    def _generate_visualization(self, viz_type: str) -> Optional[str]:
        """
        Generate a visualization file based on the type and return the path.
        
        Args:
            viz_type: Type of visualization to generate
            
        Returns:
            Path to the generated file or None if generation failed
        """
        # Get base filename from XPDL file
        base_filename = os.path.splitext(
            os.path.basename(self.config.get("xpdl_file_path", "simulation"))
        )[0]
        
        # Create output directory if it doesn't exist
        output_dir = "visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate the appropriate visualization based on type
        if viz_type == "resource_utilization":
            return self._create_resource_utilization_chart(os.path.join(output_dir, f"{base_filename}_resource_utilization.png"))
        elif viz_type == "activity_times":
            return self._create_activity_times_chart(os.path.join(output_dir, f"{base_filename}_activity_times.png"))
        elif viz_type == "token_histogram":
            return self._create_token_histogram(os.path.join(output_dir, f"{base_filename}_token_histogram.png"))
        elif viz_type == "duration_vs_wait":
            return self._create_duration_wait_scatter(os.path.join(output_dir, f"{base_filename}_duration_vs_wait.png"))
        
        return None
        
    def _create_resource_utilization_chart(self, output_path: str) -> str:
        """
        Create resource utilization chart and save to file.
        
        Args:
            output_path: Path to save the chart to
            
        Returns:
            Path to the saved chart file
        """
        resource_utilization = self.latest_results.get("resource_utilization", {})
        
        if not resource_utilization:
            raise ValueError("No resource utilization data available")
            
        # Create figure and axes
        plt.figure(figsize=(10, 6))
        
        # Sort resources by utilization
        resources = []
        utils = []
        for res, util in sorted(resource_utilization.items(), key=lambda x: x[1], reverse=True):
            resources.append(res)
            utils.append(util)
            
        # Create the bar chart
        bars = plt.bar(resources, utils, color=['green' if u < 80 else 'orange' if u < 95 else 'red' for u in utils])
        
        # Add labels
        plt.title('Resource Utilization')
        plt.xlabel('Resource')
        plt.ylabel('Utilization (%)')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom'
            )
            
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
        
    def _create_activity_times_chart(self, output_path: str) -> str:
        """
        Create activity times chart and save to file.
        
        Args:
            output_path: Path to save the chart to
            
        Returns:
            Path to the saved chart file
        """
        activity_times = self.latest_results.get("activity_processing_times", {})
        
        if not activity_times:
            raise ValueError("No activity processing times data available")
            
        # Prepare data
        activities = []
        processing_times = []
        wait_times = []
        
        for activity, data in activity_times.items():
            if data.get("durations") and data.get("type") != "Gateway":
                durations = data.get("durations", [])
                avg_processing_time = sum(data.get("processing_times", durations)) / len(data.get("processing_times", durations))
                avg_wait_time = sum(data.get("wait_times", [0])) / len(data.get("wait_times", [1])) if data.get("wait_times") else 0
                
                activities.append(activity)
                processing_times.append(avg_processing_time)
                wait_times.append(avg_wait_time)
                
        # Sort by total time and get top 10
        if activities:
            sorted_data = sorted(
                zip(activities, processing_times, wait_times),
                key=lambda x: x[1] + x[2],
                reverse=True
            )
            
            # Take top 10
            sorted_data = sorted_data[:10]
            activities, processing_times, wait_times = zip(*sorted_data)
            
        # Create figure and axes
        plt.figure(figsize=(12, 6))
        
        # Create stacked bars
        x = range(len(activities))
        plt.bar(x, processing_times, label='Processing Time', color='blue')
        plt.bar(x, wait_times, bottom=processing_times, label='Wait Time', color='orange')
        
        # Add labels
        plt.title('Top Activities by Time')
        plt.xlabel('Activity')
        plt.ylabel('Time (minutes)')
        plt.xticks(x, activities, rotation=45, ha='right')
        plt.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
        
    def _create_token_histogram(self, output_path: str) -> str:
        """
        Create token duration histogram chart and save to file.
        
        Args:
            output_path: Path to save the chart to
            
        Returns:
            Path to the saved chart file
        """
        completed_tokens = self.latest_results.get("completed_tokens", [])
        
        if not completed_tokens:
            raise ValueError("No completed tokens data available")
            
        # Calculate durations
        process_durations = [
            (token['end_time'] - token['start_time']).total_seconds() / 60 
            for token in completed_tokens
        ]
        
        # Create figure and axes
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        plt.hist(process_durations, bins=20, alpha=0.7, color='blue')
        
        # Add mean and median lines
        mean_duration = sum(process_durations) / len(process_durations)
        median_duration = sorted(process_durations)[len(process_durations) // 2]
        
        plt.axvline(mean_duration, color='red', linestyle='--')
        plt.axvline(median_duration, color='green', linestyle='--')
        
        plt.text(
            mean_duration, plt.ylim()[1] * 0.9, 
            f'Mean: {mean_duration:.1f}m', 
            color='red', ha='right', va='top'
        )
        
        plt.text(
            median_duration, plt.ylim()[1] * 0.8, 
            f'Median: {median_duration:.1f}m', 
            color='green', ha='right', va='top'
        )
        
        # Add labels
        plt.title('Process Duration Distribution')
        plt.xlabel('Duration (minutes)')
        plt.ylabel('Frequency')
        plt.grid(linestyle='--', alpha=0.7)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
        
    def _create_duration_wait_scatter(self, output_path: str) -> str:
        """
        Create duration vs wait time scatter plot and save to file.
        
        Args:
            output_path: Path to save the chart to
            
        Returns:
            Path to the saved chart file
        """
        completed_tokens = self.latest_results.get("completed_tokens", [])
        
        if not completed_tokens:
            raise ValueError("No completed tokens data available")
            
        # Calculate durations and wait times
        process_durations = [
            (token['end_time'] - token['start_time']).total_seconds() / 60 
            for token in completed_tokens
        ]
        wait_times = [token['total_wait_time'] for token in completed_tokens]
        
        # Create figure and axes
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        colors = [w/d*100 if d > 0 else 0 for w, d in zip(wait_times, process_durations)]
        scatter = plt.scatter(
            process_durations, 
            wait_times, 
            alpha=0.7, 
            c=colors,
            cmap='YlOrRd'
        )
        
        # Add a colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Wait Time %')
        
        # Add a trend line
        if len(process_durations) > 1:
            coeffs = np.polyfit(process_durations, wait_times, 1)
            trend_line = np.poly1d(coeffs)
            
            # Calculate correlation
            correlation = np.corrcoef(process_durations, wait_times)[0, 1]
            
            # Add line to plot
            x_range = np.linspace(min(process_durations), max(process_durations), 100)
            plt.plot(x_range, trend_line(x_range), 'r--', alpha=0.7)
            
            # Add correlation text
            plt.text(
                0.05, 0.95, 
                f'Correlation: {correlation:.2f}',
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )
            
        # Add labels
        plt.title('Process Duration vs Wait Time')
        plt.xlabel('Duration (minutes)')
        plt.ylabel('Wait Time (minutes)')
        plt.grid(linestyle='--', alpha=0.7)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def _update_summary_text(self, 
                           activity_processing_times: Dict[str, Dict[str, Any]],
                           resource_utilization: Dict[str, float],
                           total_tokens_started: int,
                           completed_tokens: List[Dict[str, Any]]) -> None:
        """
        Update the summary text area with simulation statistics.
        
        Args:
            activity_processing_times: Dictionary of activity processing times
            resource_utilization: Dictionary of resource utilization percentages
            total_tokens_started: Total number of tokens that started the process
            completed_tokens: List of completed token data
        """
        # Calculate overall statistics
        summary_text = "SIMULATION RESULTS SUMMARY\n"
        summary_text += "=======================\n\n"
        
        summary_text += f"Total tokens started: {total_tokens_started}\n"
        summary_text += f"Total tokens completed: {len(completed_tokens)}\n"
        
        if completed_tokens:
            process_durations = [
                (token['end_time'] - token['start_time']).total_seconds() / 60 
                for token in completed_tokens
            ]
            avg_time = sum(process_durations) / len(process_durations)
            min_time = min(process_durations)
            max_time = max(process_durations)
            
            # Format times for display with the new format
            avg_time_formatted = self.format_time(avg_time)
            min_time_formatted = self.format_time(min_time)
            max_time_formatted = self.format_time(max_time)
            
            summary_text += f"Average processing time: {avg_time_formatted} ({avg_time:.2f} min)\n"
            summary_text += f"Minimum processing time: {min_time_formatted} ({min_time:.2f} min)\n"
            summary_text += f"Maximum processing time: {max_time_formatted} ({max_time:.2f} min)\n\n"
            
            # Resource utilization summary
            summary_text += "RESOURCE UTILIZATION SUMMARY:\n"
            for resource, utilization in sorted(
                resource_utilization.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]:  # Show top 5 resources
                summary_text += f"  {resource}: {utilization:.2f}%\n"
            
            if len(resource_utilization) > 5:
                summary_text += f"  ... and {len(resource_utilization) - 5} more resources\n"
            
            # Add information about visualizations
            summary_text += "\nUSE BUTTONS BELOW TO VIEW DETAILED VISUALIZATIONS.\n"
        
        # Update summary text
        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", summary_text)
        self.summary_text.config(state="disabled")
    
    def _update_activity_table(self, 
                             activity_processing_times: Dict[str, Dict[str, Any]],
                             transitions_df: pd.DataFrame,
                             total_tokens_started: int,
                             completed_tokens: List[Dict[str, Any]]) -> None:
        """
        Update the treeview table with activity times.
        
        Args:
            activity_processing_times: Dictionary of activity processing times
            transitions_df: DataFrame with process transitions info
            total_tokens_started: Total number of tokens that started
            completed_tokens: List of completed token data
        """
        # First, clear the existing table
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Define the columns we want to show (similar to simulation_results_modified.xlsx)
        columns = [
            "Activity", 
            "Activity Type", 
            "Tokens Started", 
            "Tokens Completed", 
            "Completion Rate (%)",
            "Min Time (min)", 
            "Max Time (min)", 
            "Avg Time (min)",
            "Median Time (min)",
            "Std Dev Time (min)",
            "90th Percentile Time (min)",
            "Total Time Waiting for Resources (min)",
            "Min Time Waiting for Resources (min)",
            "Max Time Waiting for Resources (min)",
            "Avg Time Waiting for Resources (min)"
        ]
        
        # Configure treeview columns
        self.tree["columns"] = columns
        
        # Set column headings and widths
        for col in columns:
            if "Time" in col:
                self.tree.heading(col, text=col, command=lambda c=col: self._sort_by_column(c, False))
                self.tree.column(col, width=150, minwidth=100, anchor='e')  # Increased width for better display
            elif "Tokens" in col or "Rate" in col:
                self.tree.heading(col, text=col, command=lambda c=col: self._sort_by_column(c, False))
                self.tree.column(col, width=100, minwidth=80, anchor='e')  # Increased width for better display
            else:
                self.tree.heading(col, text=col, command=lambda c=col: self._sort_by_column(c, False))
                self.tree.column(col, width=150, minwidth=100, anchor='w')  # Increased width for better display
        
        # Process data for the table
        table_data = []
        
        # Get process-level data
        base_filename = os.path.splitext(
            os.path.basename(self.config.get("xpdl_file_path", ""))
        )[0]
        
        # Process-level metrics
        if completed_tokens:
            process_durations = [
                (token['end_time'] - token['start_time']).total_seconds() / 60 
                for token in completed_tokens
            ]
            
            process_wait_times = [token.get('total_wait_time', 0) for token in completed_tokens]
            
            # Process row data
            process_row = {
                "Activity": base_filename,
                "Activity Type": "Process",
                "Tokens Started": total_tokens_started,
                "Tokens Completed": len(completed_tokens),
                "Completion Rate (%)": round((len(completed_tokens) / total_tokens_started) * 100, 2) if total_tokens_started > 0 else 0,
                "Min Time (min)": min(process_durations) if process_durations else 0,
                "Max Time (min)": max(process_durations) if process_durations else 0,
                "Avg Time (min)": round(sum(process_durations) / len(process_durations), 2) if process_durations else 0,
                "Median Time (min)": round(sorted(process_durations)[len(process_durations) // 2], 2) if process_durations else 0,
                "Std Dev Time (min)": round(np.std(process_durations), 2) if process_durations else 0,
                "90th Percentile Time (min)": round(np.percentile(process_durations, 90), 2) if process_durations else 0,
                "Total Time Waiting for Resources (min)": round(sum(process_wait_times), 2),
                "Min Time Waiting for Resources (min)": round(min(process_wait_times), 2) if process_wait_times else 0,
                "Max Time Waiting for Resources (min)": round(max(process_wait_times), 2) if process_wait_times else 0,
                "Avg Time Waiting for Resources (min)": round(sum(process_wait_times) / len(process_wait_times), 2) if process_wait_times else 0
            }
            table_data.append(process_row)
        
        # Activity-level data
        for activity, data in activity_processing_times.items():
            durations = data.get("durations", [])
            wait_times = data.get("wait_times", [])
            tokens_started = data.get("tokens_started", 0)
            tokens_completed = data.get("tokens_completed", 0)
            
            # Determine activity type from transitions_df
            activity_type = "Task"  # Default
            
            if not transitions_df.empty:
                if 'name' in transitions_df.columns and 'type' in transitions_df.columns:
                    activity_type_row = transitions_df.loc[
                        transitions_df['name'].str.lower() == activity.lower(), 'type'
                    ]
                    if not activity_type_row.empty:
                        activity_type = activity_type_row.values[0]
                elif 'from' in transitions_df.columns and 'type' in transitions_df.columns:
                    activity_type_row = transitions_df.loc[
                        transitions_df['from'].str.lower() == activity.lower(), 'type'
                    ]
                    if not activity_type_row.empty:
                        activity_type = activity_type_row.values[0]
            
            # Check for gateway type
            if isinstance(activity_type, str) and "condition" in activity_type.lower():
                activity_type = "Gateway"
            
            # Calculate statistics for the activity
            min_time = round(min(durations), 2) if durations else 0
            max_time = round(max(durations), 2) if durations else 0
            avg_time = round(sum(durations) / len(durations), 2) if durations else 0
            median_time = round(sorted(durations)[len(durations) // 2], 2) if durations and len(durations) > 0 else 0
            std_dev_time = round(np.std(durations), 2) if durations and len(durations) > 0 else 0
            percentile_90_time = round(np.percentile(durations, 90), 2) if durations and len(durations) > 0 else 0
            
            total_wait_time = round(sum(wait_times), 2) if wait_times else 0
            min_wait_time = round(min(wait_times), 2) if wait_times and len(wait_times) > 0 else 0
            max_wait_time = round(max(wait_times), 2) if wait_times and len(wait_times) > 0 else 0
            avg_wait_time = round(sum(wait_times) / len(wait_times), 2) if wait_times and len(wait_times) > 0 else 0
            
            completion_rate = round((tokens_completed / tokens_started) * 100, 2) if tokens_started > 0 else 0
            
            # Create activity row
            activity_row = {
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
                "Avg Time Waiting for Resources (min)": avg_wait_time
            }
            table_data.append(activity_row)
        
        # Convert to DataFrame and store for export
        self.latest_df = pd.DataFrame(table_data)
        
        # Insert data into the treeview
        for i, row in enumerate(table_data):
            values = []
            for col in columns:
                val = row.get(col, "")
                
                # Format time values for display (only for the UI, not for export)
                if "Time" in col and "Activity Type" != col and isinstance(val, (int, float)):
                    values.append(self.format_time(val))
                else:
                    values.append(val)
            
            # Insert with tags for styling
            if i == 0:  # Process row
                self.tree.insert("", "end", values=values, tags=("process",))
            else:
                self.tree.insert("", "end", values=values)
        
        # Apply tag styling
        self.tree.tag_configure("process", background="#e6f2ff")
        
        # Trigger scrollbar visibility update
        self._update_scrollbar_visibility()
        
        # Schedule another update after tree is fully rendered
        self.tree.after(100, self._update_scrollbar_visibility)
    
    def _sort_by_column(self, col, reverse):
        """
        Sort the treeview by a column.
        
        Args:
            col: Column to sort by
            reverse: Whether to reverse the sort order
        """
        # Get all items in the treeview
        items = [(self.tree.set(k, col), k) for k in self.tree.get_children('')]
        
        # Determine sort function based on column
        if "Time" in col and "Activity Type" != col:
            # Need to convert time strings back to numeric values
            def extract_minutes(time_str):
                # Default value if parsing fails
                if not time_str or time_str == "0s":
                    return 0
                
                # Parse format like "1d 2h 3m 4s" and convert to minutes
                total_minutes = 0
                parts = time_str.split()
                
                for part in parts:
                    if part.endswith('d'):
                        total_minutes += int(part[:-1]) * 24 * 60
                    elif part.endswith('h'):
                        total_minutes += int(part[:-1]) * 60
                    elif part.endswith('m'):
                        total_minutes += int(part[:-1])
                    elif part.endswith('s'):
                        total_minutes += int(part[:-1]) / 60
                
                return total_minutes
            
            # Sort by extracted minutes
            items.sort(key=lambda x: extract_minutes(x[0]), reverse=reverse)
        elif "Tokens" in col or "Rate" in col:
            # Sort numeric columns
            items.sort(key=lambda x: float(x[0]) if x[0] and isinstance(x[0], (int, float, str)) and x[0].replace('.', '', 1).isdigit() else 0, reverse=reverse)
        else:
            # Sort text columns
            items.sort(reverse=reverse)
        
        # Rearrange items in sorted positions
        for index, (val, k) in enumerate(items):
            self.tree.move(k, '', index)
        
        # Reverse the sort order for next time
        self.tree.heading(col, command=lambda: self._sort_by_column(col, not reverse))
        
        # Ensure horizontal scrollbar remains visible after sorting
        self._update_scrollbar_visibility()
    
    def export_to_excel(self) -> None:
        """Export simulation results to an Excel file."""
        if self.latest_df is None:
            messagebox.showinfo("Export Results", "No simulation results to export.")
            return
        
        # Ask user for file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            title="Export Results to Excel"
        )
        
        if not file_path:
            return  # User cancelled
            
        try:
            # Create a copy of the dataframe for export
            df_export = self.latest_df.copy()
            
            # For time columns, ensure they remain as numeric values (minutes)
            # No conversion needed since they're already stored as minutes in the DataFrame
            
            # Export to Excel
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df_export.to_excel(writer, sheet_name='Activity Times', index=False)
            
            messagebox.showinfo(
                "Export Results", 
                f"Results successfully exported to {file_path}"
            )
        
        except Exception as e:
            messagebox.showerror(
                "Export Error", 
                f"An error occurred while exporting results: {str(e)}"
            )
