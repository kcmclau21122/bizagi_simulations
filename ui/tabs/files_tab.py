import sys
import os

# Add project root to sys.path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from tkinter import ttk, filedialog
import os
import pandas as pd
from typing import Dict, Any, Optional

from utils.config import ConfigManager

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
        # Create a PanedWindow for resizable sections
        self.paned_window = ttk.PanedWindow(self.frame, orient=tk.VERTICAL)
        self.paned_window.pack(fill="both", expand=True)
        
        # Input files frame in the top pane
        file_container = ttk.Frame(self.paned_window)
        self.paned_window.add(file_container, weight=1)
        
        file_frame = ttk.LabelFrame(file_container, text="Input Files")
        file_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # XPDL File selection - using a grid layout for better alignment
        file_frame.columnconfigure(1, weight=1)  # Make entry column expandable
        
        ttk.Label(file_frame, text="XPDL Process File:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        self.xpdl_entry = ttk.Entry(file_frame, width=50)
        self.xpdl_entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        
        # Add a tooltip-like system for long paths
        self.xpdl_entry.bind("<Enter>", self._show_xpdl_path_tooltip)
        self.xpdl_entry.bind("<Leave>", self._hide_path_tooltip)
        
        browse_xpdl_btn = ttk.Button(
            file_frame, 
            text="Browse...", 
            command=self.browse_xpdl
        )
        browse_xpdl_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Metrics File selection
        ttk.Label(file_frame, text="Simulation Metrics File:").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        self.metrics_entry = ttk.Entry(file_frame, width=50)
        self.metrics_entry.grid(row=1, column=1, padx=5, pady=5, sticky="we")
        
        # Add a tooltip-like system for long paths
        self.metrics_entry.bind("<Enter>", self._show_metrics_path_tooltip)
        self.metrics_entry.bind("<Leave>", self._hide_path_tooltip)
        
        browse_metrics_btn = ttk.Button(
            file_frame, 
            text="Browse...", 
            command=self.browse_metrics
        )
        browse_metrics_btn.grid(row=1, column=2, padx=5, pady=5)
        
        # Path tooltip label (initially hidden)
        self.path_tooltip = ttk.Label(
            file_frame, 
            background="#FFFFCC", 
            relief="solid", 
            borderwidth=1,
            font=("TkDefaultFont", 8),
            wraplength=500
        )
        
        # Add analyze button to the file frame for better visibility
        analyze_btn = ttk.Button(
            file_frame,
            text="Analyze Files",
            command=self.analyze_files,
            style="Accent.TButton"  # Custom style for emphasis
        )
        analyze_btn.grid(row=2, column=0, columnspan=3, padx=5, pady=10, sticky="e")
        
        # Create a style for the accent button if not already defined
        style = ttk.Style()
        
        # The fixed line: Check if the style exists by trying to get its current config
        # instead of using style.map() incorrectly
        try:
            existing_style = style.lookup("Accent.TButton", "font")
            if not existing_style:  # If style doesn't exist, create it
                style.configure("Accent.TButton", font=("TkDefaultFont", 9, "bold"))
                if style.theme_use() == "alt":
                    style.map("Accent.TButton",
                        background=[('active', '#4CAF50'), ('!active', '#45a049')],
                        foreground=[('active', 'white'), ('!active', 'white')]
                    )
        except tk.TclError:  # Style doesn't exist
            style.configure("Accent.TButton", font=("TkDefaultFont", 9, "bold"))
            if style.theme_use() == "alt":
                style.map("Accent.TButton",
                    background=[('active', '#4CAF50'), ('!active', '#45a049')],
                    foreground=[('active', 'white'), ('!active', 'white')]
                )
        
        # File info section in the bottom pane
        info_container = ttk.Frame(self.paned_window)
        self.paned_window.add(info_container, weight=2)  # Give more space to the info section
        
        info_frame = ttk.LabelFrame(info_container, text="File Information")
        info_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create a frame for the text with scrollbar
        text_frame = ttk.Frame(info_frame)
        text_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add vertical scrollbar
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")
        
        # Text area with scrollbar
        self.file_info_text = tk.Text(
            text_frame, 
            wrap="word", 
            height=15, 
            width=80,
            yscrollcommand=scrollbar.set
        )
        self.file_info_text.pack(side="left", fill="both", expand=True)
        self.file_info_text.insert("1.0", "Select files to view information about them.")
        self.file_info_text.config(state="disabled")
        
        # Configure scrollbar to scroll the text
        scrollbar.config(command=self.file_info_text.yview)
        
        # Add mousewheel support for scrolling
        self.file_info_text.bind("<MouseWheel>", self._on_mousewheel)
        
        # Add horizontal scrollbar for when wrap is disabled
        h_scrollbar = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=self.file_info_text.xview)
        self.file_info_text.config(xscrollcommand=h_scrollbar.set)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add toggle for word wrapping
        wrap_var = tk.BooleanVar(value=True)
        
        def toggle_wrap():
            if wrap_var.get():
                self.file_info_text.config(wrap="word")
                h_scrollbar.pack_forget()
            else:
                self.file_info_text.config(wrap="none")
                h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
                
        wrap_cb = ttk.Checkbutton(
            info_frame, 
            text="Word Wrap", 
            variable=wrap_var, 
            command=toggle_wrap
        )
        wrap_cb.pack(side="bottom", anchor="w", padx=5, pady=2)
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling for text widget"""
        self.file_info_text.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _show_xpdl_path_tooltip(self, event):
        """Show tooltip with full XPDL path on hover"""
        path = self.xpdl_entry.get()
        if path:
            self.path_tooltip.config(text=path)
            x, y, _, height = self.xpdl_entry.bbox("insert")
            self.path_tooltip.place(x=x, y=y+height+2, relwidth=0.8)
    
    def _show_metrics_path_tooltip(self, event):
        """Show tooltip with full metrics path on hover"""
        path = self.metrics_entry.get()
        if path:
            self.path_tooltip.config(text=path)
            x, y, _, height = self.metrics_entry.bbox("insert")
            self.path_tooltip.place(x=x, y=y+height+2, relwidth=0.8)
    
    def _hide_path_tooltip(self, event):
        """Hide the path tooltip"""
        self.path_tooltip.place_forget()
        
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
        # Set cursor to wait state
        self.frame.config(cursor="wait")
        self.parent.update()  # Update the UI to show the wait cursor
        
        xpdl_path = self.xpdl_entry.get()
        metrics_path = self.metrics_entry.get()
        
        info_text = ""
        
        # Process XPDL file first
        if not os.path.exists(xpdl_path):
            info_text += "XPDL file not found or not selected.\n\n"
        else:
            info_text += f"XPDL File: {os.path.basename(xpdl_path)}\n"
            info_text += f"Full Path: {xpdl_path}\n"
            info_text += f"Size: {os.path.getsize(xpdl_path) / 1024:.1f} KB\n\n"
            
            try:
                # Add XPDL file analysis
                info_text += "Analyzing XPDL file...\n"
                
                # Parse XPDL and generate sequences
                from data.xpdl_parser import parse_xpdl_to_sequences
                base_name = os.path.splitext(os.path.basename(xpdl_path))[0]
                sequence_file_path = f"{base_name}_sequences.txt"
                
                info_text += "Parsing XPDL to sequences...\n"
                parse_xpdl_to_sequences(xpdl_path, sequence_file_path)
                
                # Count the number of sequences
                sequences_count = 0
                with open(sequence_file_path, 'r') as f:
                    for line in f:
                        if '->' in line:
                            sequences_count += 1
                
                info_text += f"Sequences saved to: {sequence_file_path}\n"
                info_text += f"Number of sequences: {sequences_count}\n\n"
                
                # Generate process paths
                from data.process_paths import analyze_process_paths
                
                info_text += "Analyzing process paths...\n"
                try:
                    paths_file, summary = analyze_process_paths(xpdl_path)
                    info_text += f"Process paths saved to: {paths_file}\n\n"
                    info_text += "Path Analysis Summary:\n"
                    info_text += summary
                except Exception as e:
                    import traceback
                    info_text += f"Error analyzing process paths: {str(e)}\n"
                    info_text += f"Error details: {traceback.format_exc()}\n\n"
                
            except Exception as e:
                import traceback
                info_text += f"Error analyzing XPDL: {str(e)}\n"
                info_text += f"Error details: {traceback.format_exc()}\n\n"
        
        # Process metrics file if it exists
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
        
        # Reset cursor to normal
        self.frame.config(cursor="")
        