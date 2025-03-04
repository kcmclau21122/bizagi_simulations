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
    Allows setting simulation days, target times, and other parameters.
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
        
        # Load from config
        self._load_from_config()
        
        # Set up UI elements
        self.setup_ui()
        
    def _load_from_config(self):
        """Load values from configuration."""
        self.simulation_days.set(self.config.get("simulation_days", 2))
        self.target_avg_time.set(self.config.get("target_avg_time", 0.0))
        self.random_seed.set(self.config.get("random_seed", 10))
        
    def setup_ui(self):
        """Set up the UI components of the tab."""
        frame = ttk.LabelFrame(self.frame, text="Simulation Parameters")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Simulation days
        ttk.Label(frame, text="Simulation Days:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        ttk.Spinbox(
            frame, 
            from_=1, 
            to=365, 
            textvariable=self.simulation_days, 
            width=10
        ).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Target average time
        ttk.Label(frame, text="Target Average Processing Time (minutes):").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        target_entry = ttk.Spinbox(
            frame, 
            from_=0, 
            to=1000, 
            increment=0.1, 
            textvariable=self.target_avg_time, 
            width=10
        )
        target_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(
            frame, 
            text="(0 = no target optimization)"
        ).grid(row=1, column=2, padx=5, pady=5, sticky="w")
        
        # Random seed
        ttk.Label(frame, text="Random Seed:").grid(
            row=2, column=0, padx=5, pady=5, sticky="w"
        )
        ttk.Spinbox(
            frame, 
            from_=0, 
            to=1000, 
            textvariable=self.random_seed, 
            width=10
        ).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Advanced parameters frame
        adv_frame = ttk.LabelFrame(frame, text="Advanced Parameters")
        adv_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=10, sticky="we")
        
        # Arrival pattern
        ttk.Label(adv_frame, text="Arrival Pattern:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        arrival_pattern = ttk.Combobox(adv_frame, width=15)
        arrival_pattern['values'] = ["Fixed Interval", "Exponential", "Normal Distribution"]
        arrival_pattern.current(0)
        arrival_pattern.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Resource optimization strategy
        ttk.Label(adv_frame, text="Resource Optimization:").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        resource_opt = ttk.Combobox(adv_frame, width=15)
        resource_opt['values'] = ["Minimize Cost", "Maximize Throughput", "Balance Load"]
        resource_opt.current(0)
        resource_opt.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Load balance strategy
        ttk.Label(adv_frame, text="Load Balancing:").grid(
            row=2, column=0, padx=5, pady=5, sticky="w"
        )
        load_balance = ttk.Combobox(adv_frame, width=15)
        load_balance['values'] = ["Round Robin", "Least Utilized", "Shortest Queue"]
        load_balance.current(0)
        load_balance.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Add note about advanced parameters
        ttk.Label(
            adv_frame, 
            text="Note: Advanced parameters will be fully implemented in future versions.",
            font=("", 8, "italic")
        ).grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        
        # Simulation Description Frame
        desc_frame = ttk.LabelFrame(self.frame, text="Simulation Information")
        desc_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        info_text = tk.Text(desc_frame, wrap="word", height=8)
        info_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        info_text.insert("1.0", """The simulation will process tokens through the model based on the specified parameters:

• Simulation Days: The number of calendar days to simulate.
• Target Average Time: If specified, the simulator will attempt to optimize resource allocation to meet this target.
• Random Seed: Controls randomization for reproducible results.

The simulator respects the work calendar settings and will only process tasks during defined work hours.""")
        
        info_text.config(state="disabled")
    
    def update_config(self):
        """Update configuration from UI elements."""
        self.config.set("simulation_days", self.simulation_days.get())
        self.config.set("target_avg_time", self.target_avg_time.get())
        self.config.set("random_seed", self.random_seed.get())
