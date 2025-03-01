import tkinter as tk
from tkinter import ttk, filedialog
import os
import pandas as pd
from typing import Dict, Any, Optional

from ...utils.config import ConfigManager

class FilesTab:
    """
    Tab for managing input files (XPDL and metrics).
    Provides file selection and basic file information display.
    """
    
    def __init__(self, parent: ttk.Notebook, config: ConfigManager):
        """
        Initialize the files tab.
        
        Args:
            parent: Parent notebook widget
            config: Configuration manager
        """
        self.parent = parent
        self.config = config
        
        # Create tab frame
        self.frame = ttk.Frame(parent)
        self.setup_ui()
        
        # File path variables
        self.xpdl_file_path = tk.StringVar(value=config.get("xpdl_file_path", ""))
        self.metrics_file_path = tk.StringVar(value=config.get("metrics_file_path", ""))
        
        # Update UI with current values
        self.update_ui_from_config()
        
    def setup_ui(self) -> None:
        """Set up the UI components of the tab."""
        # Input files frame
        file_frame = ttk.LabelFrame(self.frame, text="Input Files")
        file_frame.pack(fill="both", expand=False, padx=10, pady=10)
        
        # XPDL File selection
        ttk.Label(file_frame, text="XPDL Process File:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        self.xpdl_entry = ttk.Entry(file_frame, width=50)
        self.xpdl_entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        ttk.Button(
            file_frame, 
            text="Browse...", 
            command=self.browse_xpdl
        ).grid(row=0, column=2, padx=5, pady=5)
        
        # Metrics File selection
        ttk.Label(file_frame, text="Simulation Metrics File:").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        self.metrics_entry = ttk.Entry(file_frame, width=50)
        self.metrics_entry.grid(row=1, column=1, padx=5, pady=5, sticky="we")
        ttk.Button(
            file_frame, 
            text="Browse...", 
            command=self.browse_metrics
        ).grid(row=1, column=2, padx=5, pady=5)
        
        # File info section
        info_frame = ttk.LabelFrame(self.frame, text="File Information")
        info_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.file_info_text = tk.Text(info_frame, wrap="word", height=15, width=80)
        self.file_info_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.file_info_text.insert("1.0", "Select files to view information about them.")
        self.file_info_text.config(state="disabled")
        
        # Add buttons to analyze files
        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(
            btn_frame, 
            text="Analyze Files", 
            command=self.analyze_files
        ).pack(side="right", padx=5)
        
    def update_ui_from_config(self) -> None:
        """Update UI elements from the current configuration."""
        self.xpdl_entry.delete(0, tk.END)
        self.xpdl_entry.insert(0, self.config.get("xpdl_file_path", ""))
        
        self.metrics_entry.delete(0, tk.END)
        self.metrics_entry.insert(0, self.config.get("metrics_file_path", ""))
        
    def update_config(self) -> None:
        """Update configuration from UI elements."""
        self.config.set("xpdl_file_path", self.xpdl_entry.get())
        self.config.set("metrics_file_path", self.metrics_entry.get())
        
    def browse_xpdl(self) -> None:
        """Open file dialog to select XPDL file."""
        file_path = filedialog.askopenfilename(
            title="Select XPDL File",
            filetypes=[("XPDL Files", "*.xpdl"), ("All Files", "*.*")]
        )
        if file_path:
            self.xpdl_entry.delete(0, tk.END)
            self.xpdl_entry.insert(0, file_path)
            
    def browse_metrics(self) -> None:
        """Open file dialog to select metrics file."""
        file_path = filedialog.askopenfilename(
            title="Select Metrics File",
            filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")]
        )
        if file_path:
            self.metrics_entry.delete(0, tk.END)
            self.metrics_entry.insert(0, file_path)
            
    def analyze_files(self) -> None:
        """Analyze the selected files and show information."""
        xpdl_path = self.xpdl_entry.get()
        metrics_path = self.metrics_entry.get()
        
        info_text = ""
        
        if not os.path.exists(xpdl_path):
            info_text += "XPDL file not found or not selected.\n\n"
        else:
            info_text += f"XPDL File: {os.path.basename(xpdl_path)}\n"
            info_text += f"Full Path: {xpdl_path}\n"
            info_text += f"Size: {os.path.getsize(xpdl_path) / 1024:.1f} KB\n\n"
            
            try:
                # Add basic XPDL file analysis here
                info_text += "Analyzing XPDL file...\n"
                # This could be expanded with more detailed XPDL analysis
                info_text += "XPDL analysis will be enhanced in a future version.\n\n"
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
                
                # Get activity types
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
