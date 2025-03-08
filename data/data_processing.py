# data_processing.py - Data Processing Classes
# ------------------------------------------------------------

import sys
import os
# Add project root to sys.path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import subprocess
import platform
import tkinter as tk
from tkinter import ttk

from utils.config import ConfigManager
from utils.time_utils import format_duration_for_display
from ui.components.ui_components import ScrollableFrame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ResultsFormatter:
    """Formats simulation results into readable text."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def format_results(self, 
                     activity_processing_times: Dict[str, Dict[str, Any]],
                     resource_utilization: Dict[str, float],
                     total_tokens_started: int,
                     completed_tokens: List[Dict[str, Any]]) -> str:
        """Format results into readable text."""
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
        
        return results_text


class ResultsExporter:
    """Handles exporting results to files."""
    
    def export_to_csv(self, results: Dict[str, Any], file_path: str) -> None:
        """Export results to CSV file."""
        # Extract results
        completed_tokens = results.get("completed_tokens", [])
        
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
        
    def export_to_excel(self, results: Dict[str, Any], file_path: str, simulation_params: Dict[str, Any]) -> None:
        """Export results to Excel file with multiple sheets."""
        # Create a Pandas Excel writer
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            # Export completed tokens
            completed_tokens = results.get("completed_tokens", [])
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
            
            if token_data:
                df_tokens = pd.DataFrame(token_data)
                df_tokens.to_excel(writer, sheet_name='Completed Tokens', index=False)
            
            # Export activity statistics
            activity_stats = results.get("activity_processing_times", {})
            activity_data = []
            for activity_id, stats in activity_stats.items():
                avg_processing_time = 0
                if len(stats.get("processing_times", [])) > 0:
                    avg_processing_time = sum(stats.get("processing_times", [])) / len(stats.get("processing_times", []))
                
                avg_wait_time = 0
                if len(stats.get("wait_times", [])) > 0:
                    avg_wait_time = sum(stats.get("wait_times", [])) / len(stats.get("wait_times", []))
                
                activity_data.append({
                    "Activity": activity_id,
                    "Tokens Started": stats.get("tokens_started", 0),
                    "Tokens Completed": stats.get("tokens_completed", 0),
                    "Avg Processing Time (min)": round(avg_processing_time, 2),
                    "Avg Wait Time (min)": round(avg_wait_time, 2),
                    "Total Avg Time (min)": round(avg_processing_time + avg_wait_time, 2)
                })
            
            if activity_data:
                df_activities = pd.DataFrame(activity_data)
                df_activities.to_excel(writer, sheet_name='Activity Statistics', index=False)
            
            # Export resource utilization
            resource_utilization = results.get("resource_utilization", {})
            resource_data = [
                {"Resource": resource, "Utilization (%)": utilization}
                for resource, utilization in resource_utilization.items()
            ]
            
            if resource_data:
                df_resources = pd.DataFrame(resource_data)
                df_resources.to_excel(writer, sheet_name='Resource Utilization', index=False)
            
            # Export simulation parameters
            simulation_data = [
                {"Parameter": "Total Tokens Started", "Value": results.get("total_tokens_started", 0)},
                {"Parameter": "Completed Tokens", "Value": len(completed_tokens)},
                {"Parameter": "Simulation Days", "Value": results.get("simulation_days", 0)},
                {"Parameter": "Token Count Setting", "Value": simulation_params.get("max_arrival_count", 0)},
                {"Parameter": "Min Interval (min)", "Value": simulation_params.get("min_interval", 0)},
                {"Parameter": "Avg Interval (min)", "Value": simulation_params.get("avg_interval", 0)},
                {"Parameter": "Max Interval (min)", "Value": simulation_params.get("max_interval", 0)}
            ]
            
            # Calculate average process time if tokens completed
            if completed_tokens:
                process_durations = [
                    (token['end_time'] - token['start_time']).total_seconds() / 60 
                    for token in completed_tokens
                ]
                avg_time = sum(process_durations) / len(process_durations)
                simulation_data.append({"Parameter": "Average Process Time (min)", "Value": round(avg_time, 2)})
            
            df_params = pd.DataFrame(simulation_data)
            df_params.to_excel(writer, sheet_name='Simulation Parameters', index=False)


class ChartExporter:
    """Handles exporting charts to images."""
    
    def export_charts(self, frames: List[ttk.Frame], dir_path: str) -> None:
        """Export charts from frames to image files."""
        # Save a screenshot of each chart
        for i, frame in enumerate(frames):
            for widget in frame.winfo_children():
                if isinstance(widget, ScrollableFrame):
                    for child in widget.scrollable_frame.winfo_children():
                        for grandchild in child.winfo_children():
                            if isinstance(grandchild, FigureCanvasTkAgg):
                                fig = grandchild.figure
                                file_name = f"chart_{i}_{id(grandchild)}.png"
                                file_path = os.path.join(dir_path, file_name)
                                fig.savefig(file_path, dpi=300, bbox_inches='tight')


class ReportViewer:
    """Handles viewing full reports."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def open_report(self) -> Tuple[bool, str]:
        """
        Open the report file.
        
        Returns:
            Tuple of (success, message)
        """
        base_filename = os.path.splitext(
            os.path.basename(self.config.get("xpdl_file_path", ""))
        )[0]
        report_path = f"{base_filename}_results.xlsx"
        
        if not os.path.exists(report_path):
            return False, "Report file not found. Run a simulation first."
            
        try:
            if platform.system() == 'Windows':
                os.startfile(report_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', report_path))
            else:  # Linux
                subprocess.call(('xdg-open', report_path))
            return True, "Report opened successfully."
                
        except Exception as e:
            return False, f"Could not open report: {str(e)}"