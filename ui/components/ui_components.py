# ui_components.py - UI Component Classes
# ------------------------------------------------------------

import tkinter as tk
from tkinter import ttk

class ScrollableFrame(ttk.Frame):
    """
    A base frame that provides scrolling capabilities.
    """
    
    def __init__(self, parent, **kwargs):
        """
        Initialize the scrollable frame.
        
        Args:
            parent: Parent widget
            **kwargs: Additional keyword arguments for Frame
        """
        super().__init__(parent, **kwargs)
        
        # Create a canvas for scrolling
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        
        # Create the scrollable frame
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window inside canvas
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configure canvas to expand with the frame
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack widgets
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Configure canvas to expand with window
        self.bind("<Configure>", self._on_frame_configure)
        
        # Mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
    def _on_frame_configure(self, event=None):
        """Handle frame resize event."""
        # Update the canvas width to match the frame
        self.canvas.configure(width=self.winfo_width())
        
        # Ensure the inner frame expands to fill the canvas width
        self.canvas.itemconfig(self.canvas_window, width=self.canvas.winfo_width())
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        # The event.delta value is negative when scrolling down, positive when scrolling up
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def unbind_mousewheel(self):
        """Unbind the mousewheel event when the frame loses focus."""
        self.canvas.unbind_all("<MouseWheel>")
        
    def rebind_mousewheel(self):
        """Rebind the mousewheel event when the frame gains focus."""
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)