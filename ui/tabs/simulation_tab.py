import sys
import os

# Add project root to sys.path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional

from utils.config import ConfigManager

class SimulationTab:
    """
    Tab for configuring simulation parameters.
    Allows setting simulation days, target times, token counts, arrival intervals, and other parameters.
    """
    
    def __init__(self, parent: ttk.Notebook, config: ConfigManager):
        """
        Initialize the simulation tab.
        
        Args:
            parent: Parent notebook widget
            config: Configuration manager
        """
        self.parent = parent
        self.config = config
        
        # Create tab frame
        self.frame = ttk.Frame(parent)
        
        # Create variables for simulation parameters
        self.simulation_days = tk.IntVar(value=2)
        self.target_avg_time = tk.DoubleVar(value=0.0)  # 0 means no target
        self.random_seed = tk.IntVar(value=10)
        
        # New variables for token generation
        self.token_count = tk.IntVar(value=20)  # Default to 20 tokens
        self.min_interval = tk.DoubleVar(value=3.0)  # Min minutes between tokens
        self.avg_interval = tk.DoubleVar(value=5.0)  # Avg minutes between tokens
        self.max_interval = tk.DoubleVar(value=8.0)  # Max minutes between tokens
        
        # Load from config
        self._load_from_config()
        
        # Set up UI elements
        self.setup_ui()
        
    def _load_from_config(self):
        """Load values from configuration."""
        self.simulation_days.set(self.config.get("simulation_days", 2))
        self.target_avg_time.set(self.config.get("target_avg_time", 0.0))
        self.random_seed.set(self.config.get("random_seed", 10))
        
        # Load token generation parameters
        self.token_count.set(self.config.get("token_count", 20))
        self.min_interval.set(self.config.get("min_interval", 3.0))
        self.avg_interval.set(self.config.get("avg_interval", 5.0))
        self.max_interval.set(self.config.get("max_interval", 8.0))
        
    def setup_ui(self):
        """Set up the UI components of the tab."""
        # Create a main container frame
        container = ttk.Frame(self.frame)
        container.pack(fill="both", expand=True)
        
        # Create a PanedWindow for the main content
        self.paned_window = ttk.PanedWindow(container, orient=tk.VERTICAL)
        self.paned_window.pack(fill="both", expand=True)
        
        # ----- Simulation Parameters Section -----
        params_container = ttk.Frame(self.paned_window)
        self.paned_window.add(params_container, weight=1)
        
        # Create scrollable frame for parameters
        params_scroll_container = ttk.Frame(params_container)
        params_scroll_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Add a vertical scrollbar
        params_scroll = ttk.Scrollbar(params_scroll_container)
        params_scroll.pack(side="right", fill="y")
        
        # Create canvas for scrolling
        params_canvas = tk.Canvas(
            params_scroll_container, 
            yscrollcommand=params_scroll.set
        )
        params_canvas.pack(side="left", fill="both", expand=True)
        
        # Configure the scrollbar
        params_scroll.config(command=params_canvas.yview)
        
        # Create a frame inside the canvas
        params_frame = ttk.Frame(params_canvas)
        params_window = params_canvas.create_window(
            (0, 0), 
            window=params_frame, 
            anchor="nw", 
            tags="params_frame"
        )
        
        # Configure the canvas to resize the inner frame when it's resized
        def resize_params_frame(event):
            params_canvas.itemconfig(
                params_window,
                width=event.width
            )
        params_canvas.bind("<Configure>", resize_params_frame)
        
        # Make sure the scroll region is updated when the frame changes size
        def on_params_frame_configure(event):
            params_canvas.configure(scrollregion=params_canvas.bbox("all"))
        params_frame.bind("<Configure>", on_params_frame_configure)
        
        # Add mousewheel scrolling
        def on_params_mousewheel(event):
            params_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        params_canvas.bind_all("<MouseWheel>", on_params_mousewheel)
        
        # Create LabelFrame for simulation parameters
        sim_params_frame = ttk.LabelFrame(params_frame, text="Simulation Parameters")
        sim_params_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configure the grid for better alignment
        sim_params_frame.columnconfigure(0, weight=1)  # Label column
        sim_params_frame.columnconfigure(1, weight=0)  # Input column
        sim_params_frame.columnconfigure(2, weight=2)  # Help text column
        
        # Simulation days
        ttk.Label(sim_params_frame, text="Simulation Days:").grid(
            row=0, column=0, padx=5, pady=10, sticky="w"
        )
        days_spin = ttk.Spinbox(
            sim_params_frame, 
            from_=1, 
            to=365, 
            textvariable=self.simulation_days, 
            width=10
        )
        days_spin.grid(row=0, column=1, padx=5, pady=10, sticky="w")
        
        # Help info for simulation days
        ttk.Label(
            sim_params_frame,
            text="Number of calendar days to run the simulation",
            font=("", 8, "italic"),
            foreground="gray"
        ).grid(row=0, column=2, padx=5, pady=10, sticky="w")
        
        # Token count
        ttk.Label(sim_params_frame, text="Number of Tokens:").grid(
            row=1, column=0, padx=5, pady=10, sticky="w"
        )
        token_spin = ttk.Spinbox(
            sim_params_frame, 
            from_=1, 
            to=1000, 
            textvariable=self.token_count, 
            width=10
        )
        token_spin.grid(row=1, column=1, padx=5, pady=10, sticky="w")
        
        # Help info for token count
        ttk.Label(
            sim_params_frame,
            text="Total number of process instances to simulate",
            font=("", 8, "italic"),
            foreground="gray"
        ).grid(row=1, column=2, padx=5, pady=10, sticky="w")
        
        # Create LabelFrame for token arrival intervals
        arrival_frame = ttk.LabelFrame(sim_params_frame, text="Token Arrival Intervals (minutes)")
        arrival_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=10, sticky="we")
        
        # Configure the grid for arrival parameters
        arrival_frame.columnconfigure(0, weight=1)  # Label column
        arrival_frame.columnconfigure(1, weight=1)  # Input column
        arrival_frame.columnconfigure(2, weight=2)  # Help text column
        
        # Minimum interval
        ttk.Label(arrival_frame, text="Minimum:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        min_spin = ttk.Spinbox(
            arrival_frame, 
            from_=0.1, 
            to=60, 
            increment=0.1,
            textvariable=self.min_interval, 
            width=10
        )
        min_spin.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Average interval
        ttk.Label(arrival_frame, text="Average:").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        avg_spin = ttk.Spinbox(
            arrival_frame, 
            from_=0.2, 
            to=60, 
            increment=0.1,
            textvariable=self.avg_interval, 
            width=10
        )
        avg_spin.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Maximum interval
        ttk.Label(arrival_frame, text="Maximum:").grid(
            row=2, column=0, padx=5, pady=5, sticky="w"
        )
        max_spin = ttk.Spinbox(
            arrival_frame, 
            from_=0.3, 
            to=60, 
            increment=0.1,
            textvariable=self.max_interval, 
            width=10
        )
        max_spin.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Help text for arrival intervals
        ttk.Label(
            arrival_frame,
            text="Uses triangular distribution to calculate token arrival times\nbased on these intervals (similar to Bizagi)",
            font=("", 8, "italic"),
            foreground="gray"
        ).grid(row=0, column=2, rowspan=3, padx=5, pady=5, sticky="w")
        
        # Target average time
        ttk.Label(sim_params_frame, text="Target Average Processing Time (minutes):").grid(
            row=3, column=0, padx=5, pady=10, sticky="w"
        )
        target_entry = ttk.Spinbox(
            sim_params_frame, 
            from_=0, 
            to=1000, 
            increment=0.1, 
            textvariable=self.target_avg_time, 
            width=10
        )
        target_entry.grid(row=3, column=1, padx=5, pady=10, sticky="w")
        ttk.Label(
            sim_params_frame, 
            text="(0 = no target optimization)",
            font=("", 8, "italic"),
            foreground="gray"
        ).grid(row=3, column=2, padx=5, pady=10, sticky="w")
        
        # Random seed
        ttk.Label(sim_params_frame, text="Random Seed:").grid(
            row=4, column=0, padx=5, pady=10, sticky="w"
        )
        seed_spin = ttk.Spinbox(
            sim_params_frame, 
            from_=0, 
            to=1000, 
            textvariable=self.random_seed, 
            width=10
        )
        seed_spin.grid(row=4, column=1, padx=5, pady=10, sticky="w")
        
        # Help info for random seed
        ttk.Label(
            sim_params_frame,
            text="Controls randomization for reproducible results",
            font=("", 8, "italic"),
            foreground="gray"
        ).grid(row=4, column=2, padx=5, pady=10, sticky="w")
        
        # Add validation for arrival intervals
        def validate_intervals(*args):
            try:
                min_val = self.min_interval.get()
                avg_val = self.avg_interval.get()
                max_val = self.max_interval.get()
                
                if min_val > avg_val:
                    self.avg_interval.set(min_val)
                
                if avg_val > max_val:
                    self.max_interval.set(avg_val)
                    
            except Exception:
                pass
                
        # Add trace to the interval variables
        self.min_interval.trace_add("write", validate_intervals)
        self.avg_interval.trace_add("write", validate_intervals)
        self.max_interval.trace_add("write", validate_intervals)
        
        # Advanced parameters section
        adv_frame = ttk.LabelFrame(sim_params_frame, text="Advanced Parameters")
        adv_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=10, sticky="we")
        
        # Configure the grid for advanced parameters
        adv_frame.columnconfigure(0, weight=0)  # Label column
        adv_frame.columnconfigure(1, weight=1)  # Input column
        adv_frame.columnconfigure(2, weight=2)  # Help text column
        
        # Arrival pattern
        ttk.Label(adv_frame, text="Arrival Pattern:").grid(
            row=0, column=0, padx=5, pady=8, sticky="w"
        )
        arrival_pattern = ttk.Combobox(adv_frame, width=20)
        arrival_pattern['values'] = ["Fixed Interval", "Exponential", "Normal Distribution"]
        arrival_pattern.current(0)
        arrival_pattern.grid(row=0, column=1, padx=5, pady=8, sticky="w")
        
        # Help info for arrival pattern
        ttk.Label(
            adv_frame,
            text="Distribution of token arrival times",
            font=("", 8, "italic"),
            foreground="gray"
        ).grid(row=0, column=2, padx=5, pady=8, sticky="w")
        
        # Resource optimization strategy
        ttk.Label(adv_frame, text="Resource Optimization:").grid(
            row=1, column=0, padx=5, pady=8, sticky="w"
        )
        resource_opt = ttk.Combobox(adv_frame, width=20)
        resource_opt['values'] = ["Minimize Cost", "Maximize Throughput", "Balance Load"]
        resource_opt.current(0)
        resource_opt.grid(row=1, column=1, padx=5, pady=8, sticky="w")
        
        # Help info for resource optimization
        ttk.Label(
            adv_frame,
            text="Strategy for resource allocation",
            font=("", 8, "italic"),
            foreground="gray"
        ).grid(row=1, column=2, padx=5, pady=8, sticky="w")
        
        # Load balance strategy
        ttk.Label(adv_frame, text="Load Balancing:").grid(
            row=2, column=0, padx=5, pady=8, sticky="w"
        )
        load_balance = ttk.Combobox(adv_frame, width=20)
        load_balance['values'] = ["Round Robin", "Least Utilized", "Shortest Queue"]
        load_balance.current(0)
        load_balance.grid(row=2, column=1, padx=5, pady=8, sticky="w")
        
        # Help info for load balancing
        ttk.Label(
            adv_frame,
            text="Method for distributing work among resources",
            font=("", 8, "italic"),
            foreground="gray"
        ).grid(row=2, column=2, padx=5, pady=8, sticky="w")
        
        # Add note about advanced parameters
        ttk.Label(
            adv_frame, 
            text="Note: Advanced parameters will be fully implemented in future versions.",
            font=("", 8, "italic"),
            foreground="red"
        ).grid(row=3, column=0, columnspan=3, padx=5, pady=8, sticky="w")
        
        # ----- Simulation Description Section -----
        desc_container = ttk.Frame(self.paned_window)
        self.paned_window.add(desc_container, weight=1)
        
        desc_frame = ttk.LabelFrame(desc_container, text="Simulation Information")
        desc_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create a text widget with scrollbar
        text_frame = ttk.Frame(desc_frame)
        text_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        text_scrollbar = ttk.Scrollbar(text_frame)
        text_scrollbar.pack(side="right", fill="y")
        
        self.info_text = tk.Text(
            text_frame, 
            wrap="word", 
            height=8, 
            yscrollcommand=text_scrollbar.set,
            background="#f8f8f8"  # Light gray background for text area
        )
        self.info_text.pack(side="left", fill="both", expand=True)
        
        text_scrollbar.config(command=self.info_text.yview)
        
        # Add mousewheel scrolling for text widget
        def on_text_mousewheel(event):
            self.info_text.yview_scroll(int(-1*(event.delta/120)), "units")
        self.info_text.bind("<MouseWheel>", on_text_mousewheel)
        
        # Information text with formatted content
        info_content = """The simulation will process tokens through the model based on the specified parameters:

• Simulation Days: The number of calendar days to simulate.
• Number of Tokens: Total number of process instances to generate.
• Token Arrival Intervals: Controls how frequently new tokens are created using triangular distribution.
• Target Average Time: If specified, the simulator will attempt to optimize resource allocation to meet this target.
• Random Seed: Controls randomization for reproducible results.

The simulator respects the work calendar settings and will only process tasks during defined work hours.

Advanced parameters allow fine-tuning of the simulation behavior:
• Arrival Pattern: Controls how new tokens are created over time
• Resource Optimization: Determines how resources are allocated to activities
• Load Balancing: Controls how work is distributed among available resources"""

        self.info_text.insert("1.0", info_content)
        
        # Apply some basic styling to the text
        self.info_text.tag_configure("heading", font=("TkDefaultFont", 10, "bold"))
        self.info_text.tag_configure("bullet", foreground="blue")
        
        # Find and tag bullet points
        start_index = "1.0"
        while True:
            bullet_pos = self.info_text.search("•", start_index, tk.END)
            if not bullet_pos:
                break
            
            line_end = self.info_text.search("\n", bullet_pos, tk.END)
            if not line_end:
                line_end = tk.END
                
            self.info_text.tag_add("bullet", bullet_pos, f"{bullet_pos}+1c")
            start_index = line_end
        
        self.info_text.config(state="disabled")
        
    def update_config(self):
        """Update configuration from UI elements."""
        self.config.set("simulation_days", self.simulation_days.get())
        self.config.set("target_avg_time", self.target_avg_time.get())
        self.config.set("random_seed", self.random_seed.get())
        
        # Add token generation parameters to config
        self.config.set("token_count", self.token_count.get())
        self.config.set("min_interval", self.min_interval.get())
        self.config.set("avg_interval", self.avg_interval.get())
        self.config.set("max_interval", self.max_interval.get())
