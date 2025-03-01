#!/usr/bin/env python3
"""
Bizagi Process Simulator - Main Application Entry Point

This module initializes the application, sets up logging,
and starts the main UI.
"""

import tkinter as tk
import logging
import sys
import os
from datetime import datetime

# Ensure the parent package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .main_window import MainWindow
from ..utils.config import ConfigManager

def setup_logging():
    """Set up logging for the application."""
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        
    # Set up timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"bizagi_simulator_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Logging initialized")
    return log_file

def fix_networkx_compatibility():
    """
    Patch networkx node_link_data function to fix warnings and compatibility issues.
    This is a workaround for NetworkX API changes between versions.
    """
    from networkx.readwrite import json_graph
    
    # Store original function
    original_node_link_data = json_graph.node_link_data
    
    # Define patched function that handles both the old and new API
    def patched_node_link_data(G, attrs=None, *, edges=None):
        # Handle the keyword-only argument properly
        if edges is None:
            # Default to 'links' as in the original code
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
    logging.info("NetworkX compatibility patch applied")

def main():
    """Main application entry point."""
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting Bizagi Process Simulator")
    
    # Fix NetworkX warning
    fix_networkx_compatibility()
    
    # Initialize configuration
    config = ConfigManager()
    logging.info("Configuration loaded")
    
    # Create and start the GUI
    root = tk.Tk()
    app = MainWindow(root, config)
    
    # Log application start
    logging.info("Application UI initialized")
    
    # Set icon if available
    try:
        icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'icon.ico')
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except Exception as e:
        logging.warning(f"Could not set application icon: {e}")
    
    # Start the main loop
    try:
        root.mainloop()
        logging.info("Application exited normally")
    except Exception as e:
        logging.error(f"Application crashed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
