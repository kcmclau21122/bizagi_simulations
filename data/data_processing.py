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

from utils.config import ConfigManager
from utils.time_utils import format_duration_for_display

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