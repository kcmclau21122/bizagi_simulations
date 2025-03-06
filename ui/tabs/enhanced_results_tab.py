import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
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
        ttk.Button(
            buttons_frame,
            text="Export to Excel",
            command=self.export_to_excel
        ).pack(side="right", padx=5)
        
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
        
        # Update the visualization button states based on available paths
        self.update_button_states()
        
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
    
    def update_button_states(self):
        """Update the state of visualization buttons based on available files."""
        # Enable/disable buttons based on available visualization files
        if "resource_utilization" in self.visualization_paths:
            self.resource_btn.config(state="normal")
            print("Resource utilization button enabled")
        else:
            self.resource_btn.config(state="disabled")
            
        if "activity_times" in self.visualization_paths:
            self.activity_btn.config(state="normal")
            print("Activity times button enabled")
        else:
            self.activity_btn.config(state="disabled")
            
        if "token_histogram" in self.visualization_paths:
            self.token_btn.config(state="normal")
            print("Token histogram button enabled")
        else:
            self.token_btn.config(state="disabled")
            
        if "duration_vs_wait" in self.visualization_paths:
            self.duration_btn.config(state="normal")
            print("Duration vs wait button enabled")
        else:
            self.duration_btn.config(state="disabled")
    
    def open_visualization(self, viz_type: str) -> None:
        """
        Open a visualization file using the system's default application.
        
        Args:
            viz_type: Type of visualization to open
        """
        if viz_type not in self.visualization_paths:
            messagebox.showinfo(
                "Visualization Not Available", 
                f"The {viz_type} visualization is not available."
            )
            return
            
        file_path = self.visualization_paths[viz_type]
        print(f"Opening visualization: {file_path}")
        
        if not os.path.exists(file_path):
            messagebox.showinfo(
                "File Not Found", 
                f"The visualization file was not found at {file_path}."
            )
            return
            
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
            
            # Add information about visualization files
            if self.visualization_paths:
                summary_text += "\nVISUALIZATION FILES:\n"
                summary_text += "Use the buttons below to view detailed visualizations.\n"
        
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
