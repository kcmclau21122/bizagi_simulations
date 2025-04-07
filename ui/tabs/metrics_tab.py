import sys
import os

# Add project root to sys.path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from utils.config import ConfigManager

class MetricsTab:
    """
    Tab for viewing and editing simulation metrics from the spreadsheet.
    Allows users to modify parameters and save changes to a new file.
    """
    
    def __init__(self, parent: ttk.Notebook, config: ConfigManager):
        """
        Initialize the metrics tab.
        
        Args:
            parent: Parent notebook widget
            config: Configuration manager
        """
        self.parent = parent
        self.config = config
        
        # Create tab frame
        self.frame = ttk.Frame(parent)
        
        # DataFrame to store the metrics data
        self.metrics_df = None
        
        # List to store Entry widgets for the grid
        self.entry_widgets = []
        
        # Current column headers
        self.column_headers = []
        
        # Set up UI elements
        self.setup_ui()
        
        # Load data if metrics file path is already set
        metrics_path = self.config.get("metrics_file_path", "")
        if metrics_path and os.path.exists(metrics_path):
            self.load_metrics_data(metrics_path)
        
    def setup_ui(self):
        """Set up the UI components of the tab."""
        # Create a PanedWindow for resizable sections
        self.paned_window = ttk.PanedWindow(self.frame, orient=tk.VERTICAL)
        self.paned_window.pack(fill="both", expand=True)
        
        # Top section for controls
        control_container = ttk.Frame(self.paned_window)
        self.paned_window.add(control_container, weight=1)
        
        control_frame = ttk.LabelFrame(control_container, text="Metrics Controls")
        control_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Metrics file path display
        ttk.Label(control_frame, text="Current Metrics File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.metrics_path_var = tk.StringVar()
        metrics_path_entry = ttk.Entry(control_frame, textvariable=self.metrics_path_var, width=50, state="readonly")
        metrics_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        
        # Buttons frame
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="we")
        
        # Load metrics button
        load_btn = ttk.Button(btn_frame, text="Load Metrics", command=self.browse_and_load_metrics)
        load_btn.pack(side="left", padx=5, pady=5)
        
        # Refresh button
        refresh_btn = ttk.Button(btn_frame, text="Refresh from File", command=self.refresh_metrics)
        refresh_btn.pack(side="left", padx=5, pady=5)
        
        # Add row button
        add_row_btn = ttk.Button(btn_frame, text="Add Row", command=self.add_row)
        add_row_btn.pack(side="left", padx=5, pady=5)
        
        # Save button with bold styling 
        save_btn = ttk.Button(
            btn_frame, 
            text="Save Changes", 
            command=self.save_changes,
            style="Accent.TButton"
        )
        save_btn.pack(side="right", padx=5, pady=5)
        
        # Create a style for the accent button if not already defined
        style = ttk.Style()
        try:
            existing_style = style.lookup("Accent.TButton", "font")
            if not existing_style:  
                style.configure("Accent.TButton", font=("TkDefaultFont", 9, "bold"))
        except tk.TclError:  
            style.configure("Accent.TButton", font=("TkDefaultFont", 9, "bold"))
        
        # Bottom section for data grid
        grid_container = ttk.Frame(self.paned_window)
        self.paned_window.add(grid_container, weight=3)
        
        grid_frame = ttk.LabelFrame(grid_container, text="Simulation Parameters")
        grid_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create a frame for the grid with scrollbars
        self.scroll_frame = ttk.Frame(grid_frame)
        self.scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Vertical scrollbar
        v_scrollbar = ttk.Scrollbar(self.scroll_frame, orient="vertical")
        v_scrollbar.pack(side="right", fill="y")
        
        # Horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(self.scroll_frame, orient="horizontal")
        h_scrollbar.pack(side="bottom", fill="x")
        
        # Canvas for scrollable content
        self.canvas = tk.Canvas(
            self.scroll_frame,
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Connect scrollbars to canvas
        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)
        
        # Create frame inside canvas to hold the grid
        self.grid_inner_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window(
            (0, 0), 
            window=self.grid_inner_frame, 
            anchor="nw"
        )
        
        # Configure the canvas to resize with the frame
        def on_frame_configure(event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.grid_inner_frame.bind("<Configure>", on_frame_configure)
        
        # Update canvas window size when the canvas changes
        def on_canvas_configure(event):
            self.canvas.itemconfig(self.canvas_window, width=event.width)
        self.canvas.bind("<Configure>", on_canvas_configure)
        
        # Add mousewheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Add status bar at bottom
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.frame, 
            textvariable=self.status_var, 
            relief="sunken", 
            anchor="w"
        )
        status_bar.pack(side="bottom", fill="x")
        
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
    def browse_and_load_metrics(self):
        """Browse for metrics file and load it"""
        file_path = filedialog.askopenfilename(
            title="Select Metrics File",
            filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")]
        )
        if file_path:
            # Update config with the selected path
            self.config.set("metrics_file_path", file_path)
            # Load the data
            self.load_metrics_data(file_path)
        
    def load_metrics_data(self, file_path):
        """Load metrics data from Excel file"""
        try:
            # Set cursor to wait
            self.frame.config(cursor="wait")
            self.frame.update()
            
            # Update the path display
            self.metrics_path_var.set(file_path)
            
            # Load the Excel file with explicit index_col=None to prevent using any columns as index
            self.metrics_df = pd.read_excel(file_path, index_col=None)
            
            # Create the grid once data is loaded
            self.create_data_grid()
            
            # Update status
            row_count = len(self.metrics_df) if self.metrics_df is not None else 0
            self.status_var.set(f"Loaded {row_count} rows from {os.path.basename(file_path)}")
            
        except Exception as e:
            # Show more detailed error information
            import traceback
            error_details = traceback.format_exc()
            messagebox.showerror(
                "Error Loading File", 
                f"An error occurred: {str(e)}\n\nDetails:\n{error_details}"
            )
            self.status_var.set("Error loading file")
        finally:
            # Reset cursor
            self.frame.config(cursor="")
        
    
    def refresh_metrics(self):
        """Refresh metrics from the current file path"""
        metrics_path = self.config.get("metrics_file_path", "")
        if metrics_path and os.path.exists(metrics_path):
            self.load_metrics_data(metrics_path)
        else:
            messagebox.showinfo("No File", "No metrics file is currently loaded.")
    
    def create_data_grid(self):
        """Create or update the data grid with current metrics data"""
        # Clear previous widgets
        for widget in self.grid_inner_frame.winfo_children():
            widget.destroy()
        self.entry_widgets = []
        
        if self.metrics_df is None or self.metrics_df.empty:
            ttk.Label(self.grid_inner_frame, text="No data available").grid(row=0, column=0, padx=10, pady=10)
            return
        
        # Get column headers
        self.column_headers = list(self.metrics_df.columns)
        
        # Create column headers (row 0) - starting at column 1 to leave room for row headers
        for col_idx, col_name in enumerate(self.column_headers):
            header_frame = ttk.Frame(self.grid_inner_frame)
            header_frame.grid(row=0, column=col_idx + 1, sticky="nsew", padx=1, pady=1)  # +1 to shift columns right
            
            header_label = ttk.Label(
                header_frame, 
                text=col_name, 
                background="#e6e6e6",
                padding=5,
                font=("TkDefaultFont", 9, "bold"),
                anchor="center"
            )
            header_label.pack(fill="both", expand=True)
        
        # Create row headers and data cells
        for row_idx, (_, row_data) in enumerate(self.metrics_df.iterrows()):
            # Row header (row number) - now in column 0 instead of -1
            row_header = ttk.Label(
                self.grid_inner_frame,
                text=str(row_idx + 1),
                background="#e6e6e6",
                padding=5,
                anchor="center"
            )
            row_header.grid(row=row_idx + 1, column=0, sticky="nsew", padx=1, pady=1)  # Now column 0 instead of -1
            
            # Row data cells - shifted by 1 to make room for row headers
            row_entries = []
            for col_idx, col_name in enumerate(self.column_headers):
                cell_value = row_data[col_name]
                
                # Create a variable to hold cell value
                cell_var = tk.StringVar(value=str(cell_value) if pd.notna(cell_value) else "")
                
                # Create an entry widget for the cell - column is shifted by 1
                entry = ttk.Entry(self.grid_inner_frame, textvariable=cell_var)
                entry.grid(row=row_idx + 1, column=col_idx + 1, sticky="nsew", padx=1, pady=1)  # +1 for column
                
                # Store the entry widget with its row and column information
                row_entries.append({
                    "widget": entry,
                    "var": cell_var,
                    "row": row_idx,
                    "col": col_idx,
                    "col_name": col_name
                })
            
            self.entry_widgets.append(row_entries)
        
        # Configure grid weights
        for i in range(len(self.metrics_df) + 1):
            self.grid_inner_frame.rowconfigure(i, weight=1)
        
        # Include column 0 (row headers) in column configurations
        self.grid_inner_frame.columnconfigure(0, weight=0)  # Row header column - fixed width
        for i in range(len(self.column_headers)):
            self.grid_inner_frame.columnconfigure(i + 1, weight=1)  # Data columns - expandable
        
        # Update the canvas scroll region
        self.canvas.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def add_row(self):
        """Add a new row to the grid and dataframe"""
        if self.metrics_df is None:
            messagebox.showinfo("No Data", "Please load a metrics file first.")
            return
        
        # Create a new empty row for the dataframe
        new_row = pd.Series([None] * len(self.column_headers), index=self.column_headers)
        
        # Append to the dataframe
        self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Refresh the grid
        self.create_data_grid()
        
        # Update status
        self.status_var.set(f"Added new row. Total rows: {len(self.metrics_df)}")
        
        # Scroll to the bottom to show the new row
        self.canvas.yview_moveto(1.0)
    
    def update_dataframe_from_grid(self):
        """Update the dataframe with values from the grid"""
        if self.metrics_df is None or not self.entry_widgets:
            return
        
        # Loop through all entry widgets and update the dataframe
        for row_entries in self.entry_widgets:
            for cell in row_entries:
                row_idx = cell["row"]
                col_name = cell["col_name"]
                value = cell["var"].get()
                
                # Try to convert to appropriate type
                try:
                    # Check if original value was numeric
                    original_type = self.metrics_df[col_name].dtype
                    if pd.api.types.is_numeric_dtype(original_type):
                        if value.strip() == "":
                            self.metrics_df.at[row_idx, col_name] = np.nan
                        else:
                            try:
                                # Try as int first, then float
                                if '.' in value:
                                    self.metrics_df.at[row_idx, col_name] = float(value)
                                else:
                                    self.metrics_df.at[row_idx, col_name] = int(value)
                            except ValueError:
                                # If conversion fails, keep as string
                                self.metrics_df.at[row_idx, col_name] = value
                    else:
                        # Non-numeric column
                        self.metrics_df.at[row_idx, col_name] = value
                except Exception:
                    # If any error occurs, just set the raw value
                    self.metrics_df.at[row_idx, col_name] = value
    
    def save_changes(self):
        """Save changes to a new file with _modified suffix"""
        if self.metrics_df is None:
            messagebox.showinfo("No Data", "No data to save.")
            return
        
        # Update dataframe from grid
        self.update_dataframe_from_grid()
        
        # Generate the new file path
        original_path = self.config.get("metrics_file_path", "")
        if not original_path:
            # If no path set, ask user where to save
            self.save_as_new_file()
            return
        
        # Create modified file path
        file_name, file_ext = os.path.splitext(original_path)
        modified_path = f"{file_name}_modified{file_ext}"
        
        try:
            # Save the DataFrame to Excel
            self.metrics_df.to_excel(modified_path, index=False)
            
            # Update status
            self.status_var.set(f"Changes saved to {os.path.basename(modified_path)}")
            
            # Show success message
            messagebox.showinfo(
                "Save Successful", 
                f"Changes have been saved to:\n{modified_path}"
            )
        except Exception as e:
            messagebox.showerror("Save Error", f"An error occurred while saving: {str(e)}")
    
    def save_as_new_file(self):
        """Save the metrics data to a new file selected by the user"""
        if self.metrics_df is None:
            messagebox.showinfo("No Data", "No data to save.")
            return
        
        # Update dataframe from grid
        self.update_dataframe_from_grid()
        
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            title="Save Metrics Data As"
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            # Save the DataFrame to Excel
            self.metrics_df.to_excel(file_path, index=False)
            
            # Update status
            self.status_var.set(f"Data saved to {os.path.basename(file_path)}")
            
            # Show success message
            messagebox.showinfo(
                "Save Successful", 
                f"Data has been saved to:\n{file_path}"
            )
        except Exception as e:
            messagebox.showerror("Save Error", f"An error occurred while saving: {str(e)}")
