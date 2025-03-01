import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable, List, Dict, Any
import threading
import time

class ProgressDialog(tk.Toplevel):
    """
    A dialog for displaying progress during long operations.
    Supports both determinate and indeterminate progress modes.
    """
    
    def __init__(self, parent: tk.Tk, title: str = "Progress",
                message: str = "Please wait...", 
                cancelable: bool = True,
                on_cancel: Optional[Callable[[], None]] = None,
                **kwargs):
        """
        Initialize the progress dialog.
        
        Args:
            parent: Parent window
            title: Dialog title
            message: Initial message to display
            cancelable: Whether the operation can be canceled
            on_cancel: Callback function when cancel is clicked
            **kwargs: Additional keyword arguments for Toplevel
        """
        super().__init__(parent, **kwargs)
        
        # Configure the dialog
        self.title(title)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        # Center the dialog on the parent window
        self._center_window()
        
        # Store parameters
        self.parent = parent
        self.cancelable = cancelable
        self.on_cancel = on_cancel
        
        # Create dialog content
        self._create_widgets(message)
        
        # Set up state tracking
        self.is_canceled = False
        self.auto_update_thread = None
        self.auto_update_active = False
        
        # Set protocol for closing dialog
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
    def _center_window(self) -> None:
        """Center the dialog on its parent window."""
        parent = self.parent
        
        # Get parent geometry
        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        
        # Set dialog size
        width = 400
        height = 150
        
        # Calculate position
        x = parent_x + (parent_width - width) // 2
        y = parent_y + (parent_height - height) // 2
        
        # Set geometry
        self.geometry(f"{width}x{height}+{x}+{y}")
        
    def _create_widgets(self, message: str) -> None:
        """
        Create the dialog widgets.
        
        Args:
            message: Initial message to display
        """
        # Main frame with padding
        main_frame = ttk.Frame(self, padding="20 20 20 20")
        main_frame.pack(fill="both", expand=True)
        
        # Message label
        self.message_var = tk.StringVar(value=message)
        self.message_label = ttk.Label(
            main_frame, 
            textvariable=self.message_var,
            wraplength=350
        )
        self.message_label.pack(fill="x", pady=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            main_frame,
            variable=self.progress_var,
            mode="indeterminate",
            length=350
        )
        self.progress_bar.pack(fill="x", pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Starting...")
        status_label = ttk.Label(
            main_frame, 
            textvariable=self.status_var,
            font=("", 8),
            foreground="gray"
        )
        status_label.pack(fill="x", pady=(0, 10))
        
        # Cancel button
        if self.cancelable:
            btn_frame = ttk.Frame(main_frame)
            btn_frame.pack(fill="x")
            
            ttk.Button(
                btn_frame, 
                text="Cancel", 
                command=self._on_cancel
            ).pack(side="right")
            
        # Start the progress bar animation
        self.progress_bar.start(10)
        
    def _on_close(self) -> None:
        """Handle dialog close event."""
        # Treat closing the window as canceling
        if self.cancelable:
            self._on_cancel()
        
    def _on_cancel(self) -> None:
        """Handle cancel button click."""
        self.is_canceled = True
        self.update_message("Canceling operation...")
        
        # Call the cancel callback if provided
        if self.on_cancel:
            self.on_cancel()
            
    def update_message(self, message: str) -> None:
        """
        Update the displayed message.
        
        Args:
            message: New message to display
        """
        self.message_var.set(message)
        self.update_idletasks()
        
    def update_status(self, status: str) -> None:
        """
        Update the status text.
        
        Args:
            status: New status text
        """
        self.status_var.set(status)
        self.update_idletasks()
        
    def update_progress(self, value: float, 
                       max_value: float = 100.0) -> None:
        """
        Update the progress bar value.
        
        Args:
            value: Current progress value
            max_value: Maximum progress value
        """
        # Switch to determinate mode if needed
        if self.progress_bar["mode"] == "indeterminate":
            self.progress_bar.stop()
            self.progress_bar["mode"] = "determinate"
            self.progress_bar["maximum"] = max_value
            
        # Update the progress value
        self.progress_var.set(value)
        self.update_idletasks()
        
        # Update percentage in status if not cancelled
        if not self.is_canceled:
            percent = int((value / max_value) * 100)
            self.update_status(f"{percent}% complete")
        
    def start_auto_update(self, update_interval: float = 0.5) -> None:
        """
        Start automatic updates for indeterminate progress.
        
        Args:
            update_interval: Update interval in seconds
        """
        if self.auto_update_thread is not None:
            return  # Already running
            
        self.auto_update_active = True
        
        # Create and start the update thread
        self.auto_update_thread = threading.Thread(
            target=self._auto_update_loop,
            args=(update_interval,),
            daemon=True
        )
        self.auto_update_thread.start()
        
    def stop_auto_update(self) -> None:
        """Stop automatic updates."""
        self.auto_update_active = False
        if self.auto_update_thread is not None:
            self.auto_update_thread.join(1.0)  # Wait up to 1 second
            self.auto_update_thread = None
        
    def _auto_update_loop(self, update_interval: float) -> None:
        """
        Background thread for automatic updates.
        
        Args:
            update_interval: Update interval in seconds
        """
        dots = 0
        statuses = [
            "Initializing", 
            "Processing", 
            "Computing", 
            "Analyzing", 
            "Working"
        ]
        status_index = 0
        
        while self.auto_update_active:
            # Update the status with animated dots
            dots = (dots + 1) % 4
            status = statuses[status_index] + "." * dots
            
            # Update in the main thread
            self.parent.after(
                0, 
                lambda s=status: self.update_status(s)
            )
            
            # Sleep for the update interval
            time.sleep(update_interval)
            
            # Occasionally change the status message
            if dots == 0:
                status_index = (status_index + 1) % len(statuses)
        
    def close(self) -> None:
        """Close the dialog."""
        self.stop_auto_update()
        self.grab_release()
        self.destroy()

class TaskProgressDialog(ProgressDialog):
    """
    Extended progress dialog for tracking multiple tasks.
    Shows overall progress and individual task progress.
    """
    
    def __init__(self, parent: tk.Tk, tasks: List[str], **kwargs):
        """
        Initialize the task progress dialog.
        
        Args:
            parent: Parent window
            tasks: List of task names
            **kwargs: Additional keyword arguments for ProgressDialog
        """
        super().__init__(parent, **kwargs)
        
        # Override height to accommodate task list
        height = 150 + (len(tasks) * 30)
        width = 400
        self.geometry(f"{width}x{height}")
        
        # Store tasks
        self.tasks = tasks
        self.current_task_index = -1
        self.task_progress = {}
        
        # Create task progress indicators
        self._create_task_indicators()
        
    def _create_task_indicators(self) -> None:
        """Create UI elements for task progress indicators."""
        # Create a frame for tasks
        self.tasks_frame = ttk.LabelFrame(self, text="Tasks")
        self.tasks_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Create progress indicators for each task
        self.task_labels = {}
        self.task_progresses = {}
        
        for i, task in enumerate(self.tasks):
            # Task frame
            task_frame = ttk.Frame(self.tasks_frame)
            task_frame.pack(fill="x", pady=2)
            
            # Task label
            label = ttk.Label(task_frame, text=task, width=20, anchor="w")
            label.pack(side="left", padx=(0, 10))
            
            # Task progress
            progress = ttk.Progressbar(task_frame, length=200, mode="determinate")
            progress.pack(side="right", fill="x", expand=True)
            
            # Store references
            self.task_labels[task] = label
            self.task_progresses[task] = progress
            self.task_progress[task] = 0.0
            
    def start_task(self, task_name: str) -> None:
        """
        Mark a task as started.
        
        Args:
            task_name: Name of the task to start
        """
        if task_name not in self.tasks:
            return
            
        # Update current task
        self.current_task_index = self.tasks.index(task_name)
        
        # Update message
        self.update_message(f"Working on: {task_name}")
        
        # Update label style
        label = self.task_labels.get(task_name)
        if label:
            label.configure(font=("", 9, "bold"))
            
        # Update overall progress
        self._update_overall_progress()
        
    def update_task_progress(self, task_name: str, 
                           value: float, max_value: float = 100.0) -> None:
        """
        Update progress for a specific task.
        
        Args:
            task_name: Name of the task
            value: Current progress value
            max_value: Maximum progress value
        """
        if task_name not in self.tasks:
            return
            
        # Update task progress
        progress = self.task_progresses.get(task_name)
        if progress:
            percent = value / max_value
            progress["value"] = percent * 100
            self.task_progress[task_name] = percent
            
        # Update overall progress
        self._update_overall_progress()
        
    def complete_task(self, task_name: str) -> None:
        """
        Mark a task as completed.
        
        Args:
            task_name: Name of the task to complete
        """
        if task_name not in self.tasks:
            return
            
        # Update task progress to 100%
        progress = self.task_progresses.get(task_name)
        if progress:
            progress["value"] = 100
            self.task_progress[task_name] = 1.0
            
        # Update label style
        label = self.task_labels.get(task_name)
        if label:
            label.configure(font=("", 9, ""), foreground="green")
            
        # Move to next task if this was the current one
        if self.current_task_index == self.tasks.index(task_name):
            if self.current_task_index < len(self.tasks) - 1:
                next_task = self.tasks[self.current_task_index + 1]
                self.start_task(next_task)
                
        # Update overall progress
        self._update_overall_progress()
        
    def _update_overall_progress(self) -> None:
        """Update the overall progress based on task progress."""
        if not self.tasks:
            return
            
        # Calculate average progress
        total_progress = sum(self.task_progress.values())
        overall_progress = (total_progress / len(self.tasks)) * 100
        
        # Update the main progress bar
        self.update_progress(overall_progress, 100.0)
