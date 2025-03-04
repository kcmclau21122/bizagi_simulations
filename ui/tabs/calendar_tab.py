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
        # Create a PanedWindow for resizable sections
        self.paned_window = ttk.PanedWindow(self.frame, orient=tk.VERTICAL)
        self.paned_window.pack(fill="both", expand=True)
        
        # Main calendar frame in the top pane
        cal_settings_container = ttk.Frame(self.paned_window)
        self.paned_window.add(cal_settings_container, weight=1)
        
        calendar_frame = ttk.LabelFrame(cal_settings_container, text="Work Calendar")
        calendar_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create a scrollable container for days selection to handle many options
        days_container = ttk.Frame(calendar_frame)
        days_container.pack(fill="x", padx=10, pady=10)
        
        # Add horizontal scrollbar for days selection if needed
        h_scrollbar_days = ttk.Scrollbar(days_container, orient=tk.HORIZONTAL)
        h_scrollbar_days.pack(side="bottom", fill="x")
        
        # Create canvas for days selection
        days_canvas = tk.Canvas(
            days_container, 
            height=100,
            xscrollcommand=h_scrollbar_days.set
        )
        days_canvas.pack(fill="x", expand=True)
        
        # Connect scrollbar to canvas
        h_scrollbar_days.config(command=days_canvas.xview)
        
        # Create a frame inside canvas for days
        days_frame = ttk.LabelFrame(days_canvas, text="Working Days")
        days_canvas.create_window((0, 0), window=days_frame, anchor="nw")
        
        # Update the scroll region when the inner frame changes size
        days_frame.bind(
            "<Configure>",
            lambda e: days_canvas.configure(scrollregion=days_canvas.bbox("all"), width=e.width)
        )
        
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
        
        # Calendar visualization in the bottom pane
        viz_container = ttk.Frame(self.paned_window)
        self.paned_window.add(viz_container, weight=2)
        
        cal_frame = ttk.LabelFrame(viz_container, text="Calendar Visualization")
        cal_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create a frame with scrollbar for the calendar canvas
        canvas_frame = ttk.Frame(cal_frame)
        canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add vertical scrollbar
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side="right", fill="y")
        
        # Add horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side="bottom", fill="x")
        
        # Create the calendar canvas with scrollbars
        self.calendar_canvas = tk.Canvas(
            canvas_frame, 
            bg="white", 
            height=200,
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set
        )
        self.calendar_canvas.pack(fill="both", expand=True)
        
        # Connect scrollbars to canvas
        v_scrollbar.config(command=self.calendar_canvas.yview)
        h_scrollbar.config(command=self.calendar_canvas.xview)
        
        # Add mousewheel scrolling for vertical scrolling
        self.calendar_canvas.bind("<MouseWheel>", self._on_mousewheel)
        
        # Add Shift+MouseWheel for horizontal scrolling
        self.calendar_canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
        
        # Draw initial calendar
        self.update_calendar_visualization()
        
        # Add button to update calendar (in a separate frame at the bottom)
        button_frame = ttk.Frame(calendar_frame)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(
            button_frame, 
            text="Update Calendar", 
            command=self.update_calendar_visualization
        ).pack(side="right", padx=5, pady=5)
        
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling for vertical scrolling"""
        # Scroll up/down (-1 = up, 1 = down)
        self.calendar_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
    def _on_shift_mousewheel(self, event):
        """Handle Shift+mousewheel for horizontal scrolling"""
        # Scroll left/right
        self.calendar_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        
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
        
        # Calculate the total size needed
        total_width = max(width, 800)
        total_height = max(height, 300)
        
        # Configure scrolling region to be larger than visible area
        canvas.config(scrollregion=(0, 0, total_width, total_height))
        
        # Draw week grid
        day_width = total_width / 7
        hour_height = total_height / 24
        
        # Draw hours (vertical lines)
        for hour in range(25):  # 0-24 hours
            x = 50  # Left margin
            y = hour * hour_height
            canvas.create_line(x, y, total_width, y, fill="#e0e0e0")
            if hour % 2 == 0:  # Label every 2 hours
                canvas.create_text(25, y, text=f"{hour}:00", anchor="e")
        
        # Draw days (horizontal sections)
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for i, day in enumerate(day_names):
            # Draw day column
            x = 50 + (i * day_width)
            canvas.create_line(x, 0, x, total_height, fill="#d0d0d0")
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