#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import logging
import datetime
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from simulation import run_simulation
from utils import get_simulation_parameters, format_duration, format_duration_for_display
from data_handler import build_paths, diagram_process, extract_start_tasks_from_json
from reporting import save_simulation_report
from xpdl_parser import parse_xpdl_to_sequences
import threading
import json

# Settings file path
SETTINGS_FILE = "bizagi_simulator_settings.json"

class BizagiSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bizagi Process Simulator")
        self.root.geometry("900x700")
        
        # Default values
        self.xpdl_file_path = tk.StringVar(value="")
        self.metrics_file_path = tk.StringVar(value="")
        self.simulation_days = tk.IntVar(value=2)
        self.target_avg_time = tk.DoubleVar(value=0.0)  # 0 means no target
        self.random_seed = tk.IntVar(value=10)
        
        # Calendar settings
        self.workdays = [tk.BooleanVar(value=True) for _ in range(7)]  # Mon-Sun
        self.work_hours_start = tk.IntVar(value=9)  # 9 AM
        self.work_hours_end = tk.IntVar(value=17)  # 5 PM
        
        # Load saved settings if available
        self.load_settings()
        
        self.create_gui()
    
    def create_gui(self):
        """Create the main GUI layout"""
        # Create tabbed interface
        self.tab_control = ttk.Notebook(self.root)
        
        # Create tabs
        self.tab_files = ttk.Frame(self.tab_control)
        self.tab_calendar = ttk.Frame(self.tab_control)
        self.tab_simulation = ttk.Frame(self.tab_control)
        self.tab_results = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.tab_files, text='Files')
        self.tab_control.add(self.tab_calendar, text='Calendar')
        self.tab_control.add(self.tab_simulation, text='Simulation')
        self.tab_control.add(self.tab_results, text='Results')
        
        self.tab_control.pack(expand=1, fill="both")
        
        # Populate tabs
        self.create_files_tab()
        self.create_calendar_tab()
        self.create_simulation_tab()
        self.create_results_tab()
        
        # Add run button at the bottom - MAKING SURE IT'S VISIBLE
        self.btn_frame = ttk.Frame(self.root)
        self.btn_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(self.btn_frame, text="Save Settings", command=self.save_settings).pack(side="left", padx=10, pady=5)
        
        self.run_button = ttk.Button(
            self.btn_frame, 
            text="Run Simulation", 
            command=self.run_simulation
        )
        self.run_button.pack(side="right", padx=10, pady=5)
        
        # Set protocol for closing window
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def create_files_tab(self):
        """Create the files selection tab"""
        frame = ttk.LabelFrame(self.tab_files, text="Input Files")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # XPDL File selection
        ttk.Label(frame, text="XPDL Process File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frame, textvariable=self.xpdl_file_path, width=50).grid(row=0, column=1, padx=5, pady=5, sticky="we")
        ttk.Button(frame, text="Browse...", command=self.browse_xpdl).grid(row=0, column=2, padx=5, pady=5)
        
        # Metrics File selection
        ttk.Label(frame, text="Simulation Metrics File:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frame, textvariable=self.metrics_file_path, width=50).grid(row=1, column=1, padx=5, pady=5, sticky="we")
        ttk.Button(frame, text="Browse...", command=self.browse_metrics).grid(row=1, column=2, padx=5, pady=5)
        
        # File info section
        info_frame = ttk.LabelFrame(self.tab_files, text="File Information")
        info_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.file_info_text = tk.Text(info_frame, wrap="word", height=15, width=80)
        self.file_info_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.file_info_text.insert("1.0", "Select files to view information about them.")
        self.file_info_text.config(state="disabled")
        
        # Add buttons to analyze files
        btn_frame = ttk.Frame(self.tab_files)
        btn_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(btn_frame, text="Analyze Files", command=self.analyze_files).pack(side="right", padx=5)
    
    def create_calendar_tab(self):
        """Create the work calendar tab"""
        frame = ttk.LabelFrame(self.tab_calendar, text="Work Calendar")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Days of the week selector
        days_frame = ttk.LabelFrame(frame, text="Working Days")
        days_frame.pack(fill="x", padx=10, pady=10)
        
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_colors = ["#f0f0f0", "#f0f0f0", "#f0f0f0", "#f0f0f0", "#f0f0f0", "#e6e6e6", "#e6e6e6"]
        
        self.day_buttons = []
        for i, day in enumerate(day_names):
            day_frame = ttk.Frame(days_frame, padding=5)
            day_frame.grid(row=0, column=i, padx=2)
            
            # Create colored button-like label
            day_label = ttk.Label(day_frame, text=day, background=day_colors[i], width=10, anchor="center")
            day_label.pack(pady=2)
            
            # Create checkbox
            day_check = ttk.Checkbutton(day_frame, variable=self.workdays[i])
            day_check.pack()
            
            self.day_buttons.append((day_label, day_check))
        
        # Work hours selector
        hours_frame = ttk.LabelFrame(frame, text="Working Hours")
        hours_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(hours_frame, text="Start Time:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        start_combo = ttk.Combobox(hours_frame, textvariable=self.work_hours_start, width=5)
        start_combo['values'] = list(range(0, 24))
        start_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(hours_frame, text="Hours").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        ttk.Label(hours_frame, text="End Time:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        end_combo = ttk.Combobox(hours_frame, textvariable=self.work_hours_end, width=5)
        end_combo['values'] = list(range(0, 24))
        end_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(hours_frame, text="Hours").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        
        # Calendar visualization
        cal_frame = ttk.LabelFrame(frame, text="Calendar Visualization")
        cal_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.calendar_canvas = tk.Canvas(cal_frame, bg="white", height=200)
        self.calendar_canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Draw initial calendar
        self.update_calendar_visualization()
        
        # Add button to update calendar
        ttk.Button(frame, text="Update Calendar", command=self.update_calendar_visualization).pack(pady=10)
    
    def create_simulation_tab(self):
        """Create the simulation parameters tab"""
        frame = ttk.LabelFrame(self.tab_simulation, text="Simulation Parameters")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Simulation days
        ttk.Label(frame, text="Simulation Days:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(frame, from_=1, to=365, textvariable=self.simulation_days, width=10).grid(
            row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Target average time
        ttk.Label(frame, text="Target Average Processing Time (minutes):").grid(
            row=1, column=0, padx=5, pady=5, sticky="w")
        target_entry = ttk.Spinbox(frame, from_=0, to=1000, increment=0.1, textvariable=self.target_avg_time, width=10)
        target_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(frame, text="(0 = no target optimization)").grid(
            row=1, column=2, padx=5, pady=5, sticky="w")
        
        # Random seed
        ttk.Label(frame, text="Random Seed:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(frame, from_=0, to=1000, textvariable=self.random_seed, width=10).grid(
            row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Advanced parameters frame
        adv_frame = ttk.LabelFrame(frame, text="Advanced Parameters")
        adv_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=10, sticky="we")
        
        # Add some advanced parameters here if needed
        ttk.Label(adv_frame, text="These parameters will be added in a future version.").pack(padx=5, pady=5)
        
        # Add another Run Simulation button in this tab too
        run_btn_frame = ttk.Frame(frame)
        run_btn_frame.grid(row=4, column=0, columnspan=3, padx=5, pady=20, sticky="we")
        ttk.Button(run_btn_frame, text="Run Simulation", command=self.run_simulation).pack(pady=10)
    
    def create_results_tab(self):
        """Create the results tab"""
        frame = ttk.Frame(self.tab_results)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Results text area
        self.results_text = tk.Text(frame, wrap="word", height=15, width=80)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.results_text.insert("1.0", "Simulation results will appear here after running a simulation.")
        self.results_text.config(state="disabled")
        
        # Placeholder for charts
        self.chart_frame = ttk.LabelFrame(frame, text="Visualization")
        self.chart_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ttk.Label(self.chart_frame, text="Charts will appear here after running a simulation.").pack(padx=20, pady=40)
    
    def browse_xpdl(self):
        """Open file dialog to select XPDL file"""
        file_path = filedialog.askopenfilename(
            title="Select XPDL File",
            filetypes=[("XPDL Files", "*.xpdl"), ("All Files", "*.*")]
        )
        if file_path:
            self.xpdl_file_path.set(file_path)
    
    def browse_metrics(self):
        """Open file dialog to select metrics file"""
        file_path = filedialog.askopenfilename(
            title="Select Metrics File",
            filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")]
        )
        if file_path:
            self.metrics_file_path.set(file_path)
    
    def analyze_files(self):
        """Analyze the selected files and show information"""
        xpdl_path = self.xpdl_file_path.get()
        metrics_path = self.metrics_file_path.get()
        
        info_text = ""
        
        if not os.path.exists(xpdl_path):
            info_text += "XPDL file not found or not selected.\n\n"
        else:
            info_text += f"XPDL File: {os.path.basename(xpdl_path)}\n"
            info_text += f"Full Path: {xpdl_path}\n"
            info_text += f"Size: {os.path.getsize(xpdl_path) / 1024:.1f} KB\n\n"
            
            try:
                # Try to extract basic info
                info_text += "Analyzing XPDL file...\n"
                # This would need to be implemented based on XPDL structure
                info_text += "XPDL analysis will be shown in a future version.\n\n"
            except Exception as e:
                info_text += f"Error analyzing XPDL: {str(e)}\n\n"
        
        if not os.path.exists(metrics_path):
            info_text += "Metrics file not found or not selected.\n"
        else:
            info_text += f"Metrics File: {os.path.basename(metrics_path)}\n"
            info_text += f"Full Path: {metrics_path}\n"
            info_text += f"Size: {os.path.getsize(metrics_path) / 1024:.1f} KB\n\n"
            
            try:
                # Try to read metrics file
                info_text += "Reading metrics file...\n"
                df = pd.read_excel(metrics_path)
                info_text += f"Number of rows: {len(df)}\n"
                info_text += f"Columns: {', '.join(df.columns)}\n\n"
                
                # Try to find activities (fixed to avoid the Series ambiguity)
                if 'Type' in df.columns:
                    type_counts = df['Type'].value_counts().to_dict()
                    info_text += "Activity Types:\n"
                    for activity_type, count in type_counts.items():
                        info_text += f"  - {activity_type}: {count}\n"
                else:
                    info_text += "No 'Type' column found in metrics file.\n"
            except Exception as e:
                info_text += f"Error reading metrics file: {str(e)}\n"
        
        # Update info text
        self.file_info_text.config(state="normal")
        self.file_info_text.delete("1.0", tk.END)
        self.file_info_text.insert("1.0", info_text)
        self.file_info_text.config(state="disabled")
    
    def update_calendar_visualization(self):
        """Update the calendar visualization based on current settings"""
        # Get canvas dimensions - make sure it's initialized first
        self.root.update()
        canvas = self.calendar_canvas
        canvas.delete("all")
        
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        # Ensure minimum size
        if width < 100:
            width = 700
        if height < 100:
            height = 200
        
        # Draw week grid
        day_width = width / 7
        working_hours = self.work_hours_end.get() - self.work_hours_start.get()
        hour_height = height / 24
        
        # Draw hours (vertical lines)
        for hour in range(25):  # 0-24 hours
            x = 50  # Left margin
            y = hour * hour_height
            canvas.create_line(x, y, width, y, fill="#e0e0e0")
            if hour % 2 == 0:  # Label every 2 hours
                canvas.create_text(25, y, text=f"{hour}:00", anchor="e")
        
        # Draw days (horizontal sections)
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for i, day in enumerate(day_names):
            # Draw day column
            x = 50 + (i * day_width)
            canvas.create_line(x, 0, x, height, fill="#d0d0d0")
            canvas.create_text(x + (day_width/2), 10, text=day)
            
            # Highlight working hours if it's a working day
            if self.workdays[i].get():
                start_y = self.work_hours_start.get() * hour_height
                end_y = self.work_hours_end.get() * hour_height
                canvas.create_rectangle(
                    x, start_y, 
                    x + day_width, end_y,
                    fill="#c5e0b4", outline="#70ad47"
                )
    
    def run_simulation(self):
        """Run the simulation with the configured parameters"""
        # Validate inputs
        if not self.validate_inputs():
            return
        
        # Set up simulation parameters
        xpdl_file_path = self.xpdl_file_path.get()
        metrics_file_path = self.metrics_file_path.get()
        simulation_days = self.simulation_days.get()
        target_avg_time = self.target_avg_time.get() if self.target_avg_time.get() > 0 else None
        
        # Calculate work hours per day
        work_hours_per_day = self.work_hours_end.get() - self.work_hours_start.get()
        
        # Count number of workdays
        number_workdays = sum(1 for day in self.workdays if day.get())
        
        # Set random seed
        random_seed = self.random_seed.get()
        random.seed(random_seed)
        
        # Set up start time (using the configured work start hour)
        start_time = datetime.datetime(2025, 1, 5, self.work_hours_start.get(), 0)
        
        # Set up logging with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"simulation_log_{timestamp}.txt"
        
        logging.basicConfig(
            filename=log_filename,
            filemode='w',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Log all configuration settings
        logging.info(f"Simulation started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Configuration:")
        logging.info(f"  XPDL file: {xpdl_file_path}")
        logging.info(f"  Metrics file: {metrics_file_path}")
        logging.info(f"  Simulation days: {simulation_days}")
        logging.info(f"  Number of workdays: {number_workdays}")
        logging.info(f"  Work hours per day: {work_hours_per_day}")
        logging.info(f"  Start time: {start_time}")
        logging.info(f"  Working days: {[i for i, day in enumerate(self.workdays) if day.get()]}")
        logging.info(f"  Work hours: {self.work_hours_start.get()}-{self.work_hours_end.get()}")
        if target_avg_time:
            logging.info(f"  Target average processing time: {target_avg_time} minutes")
        logging.info(f"  Random seed: {random_seed}")
        
        # Show progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Simulation Progress")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        ttk.Label(progress_window, text="Running simulation...").pack(pady=10)
        progress = ttk.Progressbar(progress_window, mode="indeterminate", length=300)
        progress.pack(pady=10, padx=20)
        progress.start()
        
        status_var = tk.StringVar(value="Initializing...")
        ttk.Label(progress_window, textvariable=status_var).pack(pady=10)
        
        # Save error message for later use
        error_message = [""]  # Use a list so it can be modified in the inner function
        
        # Function to run simulation in background
        def run_simulation_task():
            try:
                self.root.after(100, lambda: status_var.set("Parsing XPDL file..."))
                
                # Parse process sequences and load data
                output_sequences_path = 'output_sequences.txt'
                process_sequences = parse_xpdl_to_sequences(xpdl_file_path, output_sequences_path)
                
                self.root.after(100, lambda: status_var.set("Loading simulation metrics..."))
                simulation_metrics = pd.read_excel(metrics_file_path, sheet_name=0)
                simulation_metrics.columns = map(str.lower, simulation_metrics.columns)
                
                self.root.after(100, lambda: status_var.set("Building process paths..."))
                json_file_path = build_paths(output_sequences_path, simulation_metrics)
                
                self.root.after(100, lambda: status_var.set("Generating process diagram..."))
                # Run diagram_process in the main thread to avoid matplotlib warnings
                self.root.after(0, lambda: self._run_diagram_process(json_file_path))
                
                self.root.after(100, lambda: status_var.set("Running simulation..."))
                # Run the simulation with the configured parameters
                simulation_results = run_simulation(
                    json_file_path, 
                    simulation_days, 
                    start_time, 
                    number_workdays, 
                    work_hours_per_day,
                    target_avg_time
                )
                
                # Extract results
                activity_processing_times = simulation_results["activity_processing_times"]
                resource_utilization = simulation_results["resource_utilization"]
                total_tokens_started = simulation_results["total_tokens_started"]
                completed_tokens = simulation_results["completed_tokens"]
                
                self.root.after(100, lambda: status_var.set("Generating simulation report..."))
                # Generate report
                save_simulation_report(
                    activity_processing_times, 
                    resource_utilization, 
                    total_tokens_started, 
                    xpdl_file_path, 
                    simulation_metrics, 
                    completed_tokens
                )
                
                # Store results for UI update
                self.simulation_results = simulation_results
                self.xpdl_file_for_results = xpdl_file_path
                
                # Schedule UI updates in the main thread
                self.root.after(0, lambda: self._finish_simulation(True))
                
            except Exception as e:
                logging.error(f"Error in simulation: {str(e)}", exc_info=True)
                # Store the error message to use in the main thread
                error_message[0] = str(e)
                self.root.after(0, lambda: self._finish_simulation(False))
        
        # Run the simulation in a separate thread to keep UI responsive
        simulation_thread = threading.Thread(target=run_simulation_task)
        simulation_thread.daemon = True
        simulation_thread.start()
        
        # Helper method to run diagram_process in main thread
        self._progress_window = progress_window  # Store reference for _finish_simulation
        self._error_message = error_message  # Store reference for _finish_simulation
    
    def _run_diagram_process(self, json_file_path):
        """Run diagram_process in the main thread to avoid matplotlib warnings"""
        try:
            diagram_process(json_file_path)
        except Exception as e:
            logging.error(f"Error generating diagram: {str(e)}", exc_info=True)
            self._error_message[0] = f"Error generating diagram: {str(e)}"
            self._finish_simulation(False)
    
    def _finish_simulation(self, success):
        """Handle completion of simulation"""
        # Close progress window
        if hasattr(self, '_progress_window') and self._progress_window:
            self._progress_window.destroy()
            self._progress_window = None
            
        if success:
            # Update results in the UI
            self.update_results(self.simulation_results, self.xpdl_file_for_results)
            
            # Switch to results tab
            self.tab_control.select(self.tab_results)
            
            # Show success message
            messagebox.showinfo(
                "Simulation Complete", 
                f"Simulation completed successfully.\nResults saved to {os.path.splitext(os.path.basename(self.xpdl_file_for_results))[0]}_results.xlsx"
            )
        else:
            # Show error message
            messagebox.showerror(
                "Simulation Error", 
                f"An error occurred during simulation:\n{self._error_message[0]}"
            )

    def format_duration(self, minutes):
        """
        Format a duration in minutes to a more readable format.
        - If >= 60 minutes, show as hours and minutes
        - If >= 24 hours, show as days, hours, minutes and seconds
        
        Args:
            minutes (float): Duration in minutes
            
        Returns:
            str: Formatted duration string
        """
        if minutes is None or minutes == 0:
            return "0m 0s"
            
        total_seconds = int(minutes * 60)
        seconds = total_seconds % 60
        total_minutes = total_seconds // 60
        minutes_part = total_minutes % 60
        total_hours = total_minutes // 60
        
        if total_hours >= 24:
            # Format as days, hours, minutes, seconds
            days_part = total_hours // 24
            hours_part = total_hours % 24
            return f"{days_part}d {hours_part}h {minutes_part}m {seconds}s"
        elif total_hours > 0:
            # Format as hours, minutes, seconds
            return f"{total_hours}h {minutes_part}m {seconds}s"
        else:
            # Format as minutes, seconds
            return f"{minutes_part}m {seconds}s"
            
    def format_duration_for_display(self, minutes, include_raw=False):
        """
        Format duration for display purposes, optionally including the raw value.
        
        Args:
            minutes (float): Duration in minutes
            include_raw (bool): Whether to include raw minutes in parentheses
            
        Returns:
            str: Formatted string for display
        """
        if minutes is None:
            return "N/A"
            
        formatted = self.format_duration(minutes)
        if include_raw and minutes >= 60:
            return f"{formatted} ({minutes:.2f} min)"
        return formatted

    def update_results(self, simulation_results, xpdl_file_path):
        """Update the results tab with simulation results using formatted time displays"""
        activity_processing_times = simulation_results["activity_processing_times"]
        resource_utilization = simulation_results["resource_utilization"]
        total_tokens_started = simulation_results["total_tokens_started"]
        completed_tokens = simulation_results["completed_tokens"]
        
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
            avg_time_formatted = self.format_duration_for_display(avg_time, include_raw=True)
            min_time_formatted = self.format_duration_for_display(min_time, include_raw=True)
            max_time_formatted = self.format_duration_for_display(max_time, include_raw=True)
            
            results_text += f"Average processing time: {avg_time_formatted}\n"
            results_text += f"Minimum processing time: {min_time_formatted}\n"
            results_text += f"Maximum processing time: {max_time_formatted}\n\n"
            
            # Target time comparison if applicable
            if self.target_avg_time.get() > 0:
                target = self.target_avg_time.get()
                diff = avg_time - target
                target_formatted = self.format_duration_for_display(target, include_raw=True)
                diff_formatted = self.format_duration_for_display(abs(diff), include_raw=True)
                
                results_text += f"Target time: {target_formatted}\n"
                diff_direction = "over" if diff > 0 else "under"
                results_text += f"Difference from target: {diff_formatted} {diff_direction} target ({(diff/target)*100:.1f}%)\n\n"
        
        # Resource utilization
        results_text += "RESOURCE UTILIZATION\n"
        results_text += "====================\n"
        for resource, utilization in resource_utilization.items():
            results_text += f"{resource}: {utilization:.2f}%\n"
        
        # Update results text
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", results_text)
        self.results_text.config(state="disabled")
        
        # Clear existing charts
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # Create charts if we have completed tokens
        if completed_tokens:
            # Create notebook for charts
            charts_notebook = ttk.Notebook(self.chart_frame)
            charts_notebook.pack(fill="both", expand=True)
            
            # Process duration histogram
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            process_durations = [
                (token['end_time'] - token['start_time']).total_seconds() / 60 
                for token in completed_tokens
            ]
            ax1.hist(process_durations, bins=20, alpha=0.7, color='blue')
            ax1.set_title('Process Duration Distribution')
            ax1.set_xlabel('Duration (minutes)')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Add formatted time annotations to the top 3 longest durations
            top_durations = sorted(process_durations, reverse=True)[:3]
            for duration in top_durations:
                formatted_time = self.format_duration_for_display(duration, include_raw=True)
                ax1.annotate(formatted_time, 
                             xy=(duration, 0),
                             xytext=(duration, 1),
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                             rotation=45)
            
            # Resource utilization chart
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            resources = list(resource_utilization.keys())
            utils = list(resource_utilization.values())
            
            # Sort by utilization for better visualization
            if resources and utils:
                utils, resources = zip(*sorted(zip(utils, resources), reverse=True))
            
            ax2.bar(resources, utils, color='green', alpha=0.7)
            ax2.set_title('Resource Utilization')
            ax2.set_xlabel('Resource')
            ax2.set_ylabel('Utilization (%)')
            ax2.set_ylim(0, 100)
            ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            fig2.tight_layout()
            
            # Embed charts in the UI
            chart_frame1 = ttk.Frame(charts_notebook)
            chart_frame2 = ttk.Frame(charts_notebook)
            
            charts_notebook.add(chart_frame1, text='Process Duration')
            charts_notebook.add(chart_frame2, text='Resource Utilization')
            
            canvas1 = FigureCanvasTkAgg(fig1, master=chart_frame1)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill="both", expand=True)
            
            canvas2 = FigureCanvasTkAgg(fig2, master=chart_frame2)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill="both", expand=True)
    
    def validate_inputs(self):
        """Validate user inputs before running simulation"""
        xpdl_path = self.xpdl_file_path.get()
        metrics_path = self.metrics_file_path.get()
        
        # Check if files exist
        if not os.path.exists(xpdl_path):
            messagebox.showerror("Invalid Input", "XPDL file does not exist or is not selected.")
            return False
        
        if not os.path.exists(metrics_path):
            messagebox.showerror("Invalid Input", "Metrics file does not exist or is not selected.")
            return False
        
        # Check if at least one day is selected as working day
        if not any(day.get() for day in self.workdays):
            messagebox.showerror("Invalid Input", "At least one working day must be selected.")
            return False
        
        # Check if work hours are valid
        if self.work_hours_start.get() >= self.work_hours_end.get():
            messagebox.showerror("Invalid Input", "End time must be later than start time.")
            return False
        
        # Check simulation days
        if self.simulation_days.get() <= 0:
            messagebox.showerror("Invalid Input", "Simulation days must be positive.")
            return False
        
        # Check target time (if specified)
        if self.target_avg_time.get() < 0:
            messagebox.showerror("Invalid Input", "Target processing time cannot be negative.")
            return False
        
        return True
    
    def save_settings(self):
        """Save current GUI settings to a file"""
        settings = {
            "xpdl_file_path": self.xpdl_file_path.get(),
            "metrics_file_path": self.metrics_file_path.get(),
            "simulation_days": self.simulation_days.get(),
            "target_avg_time": self.target_avg_time.get(),
            "random_seed": self.random_seed.get(),
            "workdays": [day.get() for day in self.workdays],
            "work_hours_start": self.work_hours_start.get(),
            "work_hours_end": self.work_hours_end.get()
        }
        
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=4)
            messagebox.showinfo("Settings Saved", "Your settings have been saved successfully.")
        except Exception as e:
            messagebox.showerror("Error Saving Settings", f"An error occurred while saving settings: {str(e)}")
    
    def load_settings(self):
        """Load GUI settings from file if it exists"""
        if not os.path.exists(SETTINGS_FILE):
            return  # Use defaults if no settings file exists
            
        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                
            # Apply loaded settings
            if "xpdl_file_path" in settings and os.path.exists(settings["xpdl_file_path"]):
                self.xpdl_file_path.set(settings["xpdl_file_path"])
                
            if "metrics_file_path" in settings and os.path.exists(settings["metrics_file_path"]):
                self.metrics_file_path.set(settings["metrics_file_path"])
                
            if "simulation_days" in settings:
                self.simulation_days.set(settings["simulation_days"])
                
            if "target_avg_time" in settings:
                self.target_avg_time.set(settings["target_avg_time"])
                
            if "random_seed" in settings:
                self.random_seed.set(settings["random_seed"])
                
            if "workdays" in settings and len(settings["workdays"]) == 7:
                for i, value in enumerate(settings["workdays"]):
                    self.workdays[i].set(value)
                    
            if "work_hours_start" in settings:
                self.work_hours_start.set(settings["work_hours_start"])
                
            if "work_hours_end" in settings:
                self.work_hours_end.set(settings["work_hours_end"])
                
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
            # Continue with defaults if settings can't be loaded
    
    def on_close(self):
        """Handle window close event"""
        if messagebox.askyesno("Save Settings", "Do you want to save your settings before exiting?"):
            self.save_settings()
        self.root.destroy()


def fix_networkx_node_link_data():
    """Patch networkx node_link_data function to fix warnings and compatibility issues"""
    from networkx.readwrite import json_graph
    
    # Store original function
    original_node_link_data = json_graph.node_link_data
    
    # Define patched function that handles both the old and new API
    def patched_node_link_data(G, attrs=None, *, edges=None):
        # Handle the keyword-only argument properly
        if edges is None:
            # Default to 'links' as in your code
            edges = 'links'
        
        # Call the original function with the correct arguments
        # Check if it expects 1 or 2 positional arguments
        import inspect
        sig = inspect.signature(original_node_link_data)
        param_names = list(sig.parameters.keys())
        
        if len(param_names) >= 2 and 'attrs' in param_names:
            # Original function accepts attrs parameter
            return original_node_link_data(G, attrs, edges=edges)
        else:
            # Original function only accepts G parameter
            return original_node_link_data(G, edges=edges)
    
    # Replace original with patched version
    json_graph.node_link_data = patched_node_link_data


def main():
    # Fix NetworkX warning
    fix_networkx_node_link_data()
    
    # Start the GUI
    root = tk.Tk()
    app = BizagiSimulatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
