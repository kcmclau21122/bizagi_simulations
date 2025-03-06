import sys
import os

# Add project root to sys.path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import Dict, Any, Optional, List
from utils.config import ConfigManager
from core.simulation_runner import SimulationRunner
from ui.tabs.files_tab import FilesTab
from ui.tabs.calendar_tab import CalendarTab
from ui.tabs.simulation_tab import SimulationTab
from ui.tabs.enhanced_results_tab import EnhancedResultsTab


class MainWindow:
    """
    Main application window for the Bizagi Process Simulator.
    Manages tabs, configuration, and simulation control.
    """
    
    def __init__(self, root: tk.Tk, config: ConfigManager):
        """
        Initialize the main window.
        
        Args:
            root: The tkinter root window
            config: Configuration manager
        """
        self.root = root
        self.config = config
        self.runner = None
        
        # Setup window
        self.root.title("Bizagi Process Simulator")
        self.root.geometry("900x700")
        
        # Create main frame that will contain everything
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)
        
        # Create a vertical paned window for main content
        self.main_paned = ttk.PanedWindow(self.main_frame, orient=tk.VERTICAL)
        self.main_paned.pack(fill="both", expand=True)
        
        # Create a frame for tabs that will go in the top pane
        self.tabs_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.tabs_frame, weight=1)
        
        # Set up tabs
        self.tab_control = ttk.Notebook(self.tabs_frame)
        self.tab_control.pack(expand=1, fill="both")
        
        # Create tabs
        self.files_tab = FilesTab(self.tab_control, self.config)
        self.calendar_tab = CalendarTab(self.tab_control, self.config)
        self.simulation_tab = SimulationTab(self.tab_control, self.config)
        self.results_tab = EnhancedResultsTab(self.tab_control, self.config)
        
        # Add tabs to notebook
        self.tab_control.add(self.files_tab.frame, text='Files')
        self.tab_control.add(self.calendar_tab.frame, text='Calendar')
        self.tab_control.add(self.simulation_tab.frame, text='Simulation')
        self.tab_control.add(self.results_tab.frame, text='Results')
        
        # Add control buttons frame - place at bottom of window
        self.btn_frame = ttk.Frame(self.main_frame)
        self.btn_frame.pack(fill="x", side="bottom", padx=10, pady=10)
        
        # Save settings button
        ttk.Button(
            self.btn_frame, 
            text="Save Settings", 
            command=self.save_settings
        ).pack(side="left", padx=10, pady=5)
        
        # Run simulation button
        self.run_button = ttk.Button(
            self.btn_frame, 
            text="Run Simulation", 
            command=self.run_simulation
        )
        self.run_button.pack(side="right", padx=10, pady=5)
        
        # Set protocol for closing window
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def create_control_buttons(self) -> None:
        """Create global control buttons at the bottom of the window."""
        # Button frame is now created in __init__
        pass
            
    def save_settings(self) -> None:
        """Save current settings to the configuration file."""
        # Update config from all tabs
        self.files_tab.update_config()
        self.calendar_tab.update_config()
        self.simulation_tab.update_config()
        
        # Save to file
        if self.config.save():
            messagebox.showinfo("Settings Saved", "Your settings have been saved successfully.")
        else:
            messagebox.showerror("Error", "Failed to save settings.")
            
    def run_simulation(self) -> None:
        """Run the simulation with the current configuration."""
        # Validate inputs first
        if not self.validate_inputs():
            return
            
        # Update configuration from all tabs
        self.files_tab.update_config()
        self.calendar_tab.update_config()
        self.simulation_tab.update_config()
        
        # Show progress dialog
        self.show_progress_dialog()
        
        # Create and start the simulation runner
        self.runner = SimulationRunner(
            self.config, 
            progress_callback=self.update_progress,
            completion_callback=self.on_simulation_complete
        )
        self.runner.run()
        
    def validate_inputs(self) -> bool:
        """
        Validate user inputs before running simulation.
        
        Returns:
            True if inputs are valid, False otherwise
        """
        xpdl_path = self.config.get("xpdl_file_path")
        metrics_path = self.config.get("metrics_file_path")
        
        # Check if files exist
        if not os.path.exists(xpdl_path):
            messagebox.showerror("Invalid Input", "XPDL file does not exist or is not selected.")
            return False
        
        if not os.path.exists(metrics_path):
            messagebox.showerror("Invalid Input", "Metrics file does not exist or is not selected.")
            return False
        
        # Check if at least one day is selected as working day
        workdays = self.config.get("workdays", [])
        if not any(workdays):
            messagebox.showerror("Invalid Input", "At least one working day must be selected.")
            return False
        
        # Check if work hours are valid
        work_hours_start = self.config.get("work_hours_start", 0)
        work_hours_end = self.config.get("work_hours_end", 0)
        if work_hours_start >= work_hours_end:
            messagebox.showerror("Invalid Input", "End time must be later than start time.")
            return False
        
        # Check simulation days
        simulation_days = self.config.get("simulation_days", 0)
        if simulation_days <= 0:
            messagebox.showerror("Invalid Input", "Simulation days must be positive.")
            return False
        
        # Check target time (if specified)
        target_avg_time = self.config.get("target_avg_time", 0)
        if target_avg_time < 0:
            messagebox.showerror("Invalid Input", "Target processing time cannot be negative.")
            return False
        
        return True
        
    def show_progress_dialog(self) -> None:
        """Show progress dialog for the simulation."""
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Simulation Progress")
        self.progress_window.geometry("400x150")
        self.progress_window.transient(self.root)
        self.progress_window.grab_set()
        
        ttk.Label(self.progress_window, text="Running simulation...").pack(pady=10)
        
        self.progress_bar = ttk.Progressbar(
            self.progress_window, 
            mode="indeterminate", 
            length=300
        )
        self.progress_bar.pack(pady=10, padx=20)
        self.progress_bar.start()
        
        self.status_var = tk.StringVar(value="Initializing...")
        ttk.Label(self.progress_window, textvariable=self.status_var).pack(pady=10)
        
        # Add cancel button
        ttk.Button(
            self.progress_window, 
            text="Cancel", 
            command=self.cancel_simulation
        ).pack(pady=5)
        
    def update_progress(self, message: str) -> None:
        """
        Update progress dialog with a status message.
        
        Args:
            message: Progress message to display
        """
        if hasattr(self, 'status_var'):
            self.status_var.set(message)
            self.root.update_idletasks()  # Force UI update
            
    def cancel_simulation(self) -> None:
        """Cancel the running simulation."""
        if self.runner and self.runner.is_running():
            self.runner.cancel()
            self.update_progress("Cancelling simulation...")
        
    def on_simulation_complete(self, results: Dict[str, Any]) -> None:
        """
        Handle simulation completion.
        
        Args:
            results: Simulation results dictionary
        """
        # Close progress window
        if hasattr(self, 'progress_window') and self.progress_window:
            self.progress_window.destroy()
            
        # Check for errors
        if "error" in results:
            messagebox.showerror(
                "Simulation Error", 
                f"An error occurred during simulation:\n{results['error']}"
            )
            return
            
        # Update results tab
        self.results_tab.update_results(results)
        
        # Switch to results tab
        self.tab_control.select(3)  # Index of results tab
        
        # Ensure buttons remain visible
        self.root.update_idletasks()
        
        # Show success message
        messagebox.showinfo(
            "Simulation Complete", 
            "Simulation completed successfully.\n"
            f"Results saved to {os.path.splitext(os.path.basename(self.config.get('xpdl_file_path')))[0]}_results.xlsx"
        )
        
    def on_close(self) -> None:
        """Handle window close event."""
        if messagebox.askyesno("Save Settings", "Do you want to save your settings before exiting?"):
            self.save_settings()
        self.root.destroy()