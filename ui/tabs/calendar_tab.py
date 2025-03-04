import sys
import os

# Add project root to sys.path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Any

from utils.config import ConfigManager

class CalendarTab:
    """
    Tab for managing work calendar settings.
    Allows configuration of working days and hours.
    """
    
    def __init__(self, parent: ttk.Notebook, config: ConfigManager):
        """
        Initialize the calendar tab.
        
        Args:
            parent: Parent notebook widget
            config: Configuration manager
        """
        self.parent = parent
        self.config = config
        
        # Create tab frame
        self.frame = ttk.Frame(parent)
        
        # Initialize work day variables
        self.workdays = [tk.BooleanVar(value=True) for _ in range(7)]  # Mon-Sun
        self.work_hours_start = tk.IntVar(value=9)  # 9 AM
        self.work_hours_end = tk.IntVar(value=17)  # 5 PM
        
        # Load config values
        self._load_from_config()
        
        # Set up UI elements
        self.setup_ui()
        
    def _load_from_config(self):
        """Load values from configuration."""
        workdays = self.config.get("workdays", [True, True, True, True, True, False, False])
        for i, value in enumerate(workdays):
            if i < len(self.workdays):
                self.workdays[i].set(value)
                
        self.work_hours_start.set(self.config.get("work_hours_start", 9))
        self.work_hours_end.set(self.config.get("work_hours_end", 17))
        
    def setup_ui(self):
        """Set up the UI components of the tab."""
        # Main calendar frame
        calendar_frame = ttk.LabelFrame(self.frame, text="Work Calendar")
        calendar_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Days of the week selector
        days_frame = ttk.LabelFrame(calendar_frame, text="Working Days")
        days_frame.pack(fill="x", padx=10, pady=10)
        
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_colors = ["#f0f0f0", "#f0f0f0", "#f0f0f0", "#f0f0f0", "#f0f0f0", "#e6e6e6", "#e6e6e6"]
        
        self.day_buttons = []
        for i, day in enumerate(day_names):
            day_frame = ttk.Frame(days_frame, padding=5)
            day_frame.grid(row=0, column=i, padx=2)
            
            # Create colored button-like label
            day_label = ttk.Label(
                day_frame, 
                text=day, 
                background=day_colors[i], 
                width=10, 
                anchor="center"
            )
            day_label.pack(pady=2)
            
            # Create checkbox
            day_check = ttk.Checkbutton(day_frame, variable=self.workdays[i])
            day_check.pack()
            
            self.day_buttons.append((day_label, day_check))
        
        # Work hours selector
        hours_frame = ttk.LabelFrame(calendar_frame, text="Working Hours")
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
        cal_frame = ttk.LabelFrame(calendar_frame, text="Calendar Visualization")
        cal_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.calendar_canvas = tk.Canvas(cal_frame, bg="white", height=200)
        self.calendar_canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Draw initial calendar
        self.update_calendar_visualization()
        
        # Add button to update calendar
        ttk.Button(
            calendar_frame, 
            text="Update Calendar", 
            command=self.update_calendar_visualization
        ).pack(pady=10)
        
    def update_calendar_visualization(self):
        """Update the calendar visualization based on current settings."""
        # Get canvas dimensions - make sure it's initialized first
        self.parent.update()
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
        
    def update_config(self):
        """Update configuration from UI elements."""
        self.config.set("workdays", [day.get() for day in self.workdays])
        self.config.set("work_hours_start", self.work_hours_start.get())
        self.config.set("work_hours_end", self.work_hours_end.get())
