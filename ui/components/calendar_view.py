import tkinter as tk
from tkinter import ttk
import calendar
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable

class CalendarView(ttk.Frame):
    """
    A reusable calendar visualization component.
    Displays work days and hours in a visual grid.
    """
    
    def __init__(self, parent, **kwargs):
        """
        Initialize the calendar view component.
        
        Args:
            parent: Parent widget
            **kwargs: Additional keyword arguments for the Frame
        """
        super().__init__(parent, **kwargs)
        
        # Default configuration
        self.workdays = [True, True, True, True, True, False, False]  # Mon-Sun
        self.work_hours_start = 9  # 9 AM
        self.work_hours_end = 17  # 5 PM
        
        # Callbacks
        self.on_day_click: Optional[Callable[[int], None]] = None
        self.on_hour_click: Optional[Callable[[int], None]] = None
        
        # Create the calendar canvas
        self.canvas = tk.Canvas(self, bg="white", height=200)
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Bind events
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        
        # Draw initial calendar
        self.draw_calendar()
        
    def set_workdays(self, workdays: List[bool]) -> None:
        """
        Set which days are working days.
        
        Args:
            workdays: List of 7 boolean values (Mon-Sun)
        """
        if len(workdays) != 7:
            raise ValueError("Workdays must be a list of 7 boolean values")
            
        self.workdays = workdays
        self.draw_calendar()
        
    def set_work_hours(self, start_hour: int, end_hour: int) -> None:
        """
        Set work hour range.
        
        Args:
            start_hour: Start hour (0-23)
            end_hour: End hour (0-23)
        """
        if not (0 <= start_hour < 24 and 0 <= end_hour <= 24):
            raise ValueError("Hours must be between 0 and 24")
            
        if start_hour >= end_hour:
            raise ValueError("Start hour must be less than end hour")
            
        self.work_hours_start = start_hour
        self.work_hours_end = end_hour
        self.draw_calendar()
        
    def draw_calendar(self) -> None:
        """Draw the calendar visualization."""
        canvas = self.canvas
        canvas.delete("all")
        
        # Get canvas dimensions
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        # Ensure minimum size
        if width < 100:
            width = 700
        if height < 100:
            height = 200
            
        self.width = width
        self.height = height
        
        # Draw week grid
        day_width = (width - 50) / 7  # Account for left margin
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
        day_colors = ["#f0f0f0", "#f0f0f0", "#f0f0f0", "#f0f0f0", "#f0f0f0", "#e6e6e6", "#e6e6e6"]
        
        for i, day in enumerate(day_names):
            # Draw day column
            x = 50 + (i * day_width)
            canvas.create_line(x, 0, x, height, fill="#d0d0d0")
            canvas.create_text(x + (day_width/2), 10, text=day)
            
            # Highlight working hours if it's a working day
            if self.workdays[i]:
                start_y = self.work_hours_start * hour_height
                end_y = self.work_hours_end * hour_height
                canvas.create_rectangle(
                    x, start_y, 
                    x + day_width, end_y,
                    fill="#c5e0b4", outline="#70ad47",
                    tags=f"workday_{i}"
                )
            
        # Create a now indicator line
        now = datetime.now()
        if 0 <= now.hour < 24:
            y = now.hour * hour_height + (now.minute / 60) * hour_height
            canvas.create_line(50, y, width, y, fill="red", width=2, dash=(4, 2))
            canvas.create_text(45, y, text="Now", anchor="e", fill="red")
            
    def _on_canvas_click(self, event) -> None:
        """
        Handle canvas click events.
        
        Args:
            event: Mouse event
        """
        # Calculate which day was clicked
        width = self.width
        day_width = (width - 50) / 7
        
        # Check if click is in the days area
        if event.x < 50:
            return
            
        # Calculate day index (0-6)
        day_index = int((event.x - 50) / day_width)
        if day_index < 0 or day_index > 6:
            return
            
        # Calculate hour
        hour = int(event.y / (self.height / 24))
        if hour < 0 or hour >= 24:
            return
            
        # Toggle workday if clicking on a day
        if event.y < 20:  # Day header
            if self.on_day_click:
                self.on_day_click(day_index)
        else:
            # Handle hour click
            if self.on_hour_click:
                self.on_hour_click(hour)
                
    def simulate_week(self) -> None:
        """
        Simulate a week of activity in the calendar (animation).
        This is a demo feature to show work hours.
        """
        canvas = self.canvas
        
        # Clear any existing animation
        canvas.delete("animation")
        
        # Width and height calculations
        day_width = (self.width - 50) / 7
        hour_height = self.height / 24
        
        # Start simulation on Monday at work start
        current_day = 0
        current_hour = self.work_hours_start
        
        def animate_step(day, hour):
            # Check if we've reached the end of the week
            if day > 6:
                return
                
            # Skip non-work days and hours
            if not self.workdays[day] or hour < self.work_hours_start or hour >= self.work_hours_end:
                # Move to next hour
                next_hour = hour + 1
                next_day = day
                
                # If we're at the end of a day, move to next day
                if next_hour >= 24:
                    next_hour = 0
                    next_day += 1
                    
                self.after(100, lambda: animate_step(next_day, next_hour))
                return
                
            # Calculate position
            x = 50 + (day * day_width) + (day_width / 2)
            y = hour * hour_height + (hour_height / 2)
            
            # Draw an activity indicator
            indicator = canvas.create_oval(
                x - 5, y - 5, x + 5, y + 5, 
                fill="red", 
                tags="animation"
            )
            
            # Fade out after a delay
            def fade_out():
                canvas.delete(indicator)
            
            self.after(500, fade_out)
            
            # Schedule next step
            next_hour = hour + 1
            next_day = day
            
            # If we're at the end of a day, move to next day
            if next_hour >= 24:
                next_hour = 0
                next_day += 1
                
            self.after(500, lambda: animate_step(next_day, next_hour))
            
        # Start animation
        animate_step(current_day, current_hour)
        
    def get_workdays_count(self) -> int:
        """
        Get the number of workdays configured.
        
        Returns:
            Number of workdays (1-7)
        """
        return sum(1 for day in self.workdays if day)
        
    def get_work_hours_per_day(self) -> int:
        """
        Get the number of work hours per day.
        
        Returns:
            Number of work hours per day
        """
        return self.work_hours_end - self.work_hours_start

class MonthCalendarView(ttk.Frame):
    """
    A calendar view showing a full month with working days highlighted.
    Useful for visualizing simulation periods.
    """
    
    def __init__(self, parent, **kwargs):
        """
        Initialize the month calendar view.
        
        Args:
            parent: Parent widget
            **kwargs: Additional keyword arguments for the Frame
        """
        super().__init__(parent, **kwargs)
        
        # Current displayed month and year
        self.current_date = datetime.now()
        self.current_month = self.current_date.month
        self.current_year = self.current_date.year
        
        # Working days (0=Monday, 6=Sunday)
        self.workdays = [True, True, True, True, True, False, False]
        
        # Create UI
        self._create_ui()
        
        # Update calendar
        self._update_calendar()
        
    def _create_ui(self) -> None:
        """Create the calendar UI components."""
        # Month navigation frame
        nav_frame = ttk.Frame(self)
        nav_frame.pack(fill="x", padx=5, pady=5)
        
        # Previous month button
        ttk.Button(
            nav_frame, 
            text="<", 
            width=3, 
            command=self._prev_month
        ).pack(side="left", padx=5)
        
        # Month/year label
        self.month_label = ttk.Label(nav_frame, text="", font=("", 10, "bold"))
        self.month_label.pack(side="left", expand=True)
        
        # Next month button
        ttk.Button(
            nav_frame, 
            text=">", 
            width=3, 
            command=self._next_month
        ).pack(side="right", padx=5)
        
        # Calendar frame
        self.cal_frame = ttk.Frame(self)
        self.cal_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Day labels (Mon-Sun)
        day_labels_frame = ttk.Frame(self.cal_frame)
        day_labels_frame.pack(fill="x", pady=2)
        
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for i, day in enumerate(days):
            label = ttk.Label(
                day_labels_frame, 
                text=day, 
                width=4, 
                anchor="center"
            )
            label.grid(row=0, column=i, padx=1, pady=1)
            
        # Calendar grid - will be populated in _update_calendar
        
    def _update_calendar(self) -> None:
        """Update the calendar display for the current month/year."""
        # Clear existing calendar
        for widget in self.cal_frame.winfo_children()[1:]:
            widget.destroy()
            
        # Update month/year label
        month_name = calendar.month_name[self.current_month]
        self.month_label.config(text=f"{month_name} {self.current_year}")
        
        # Get the calendar for current month
        cal = calendar.monthcalendar(self.current_year, self.current_month)
        
        # Create a new grid frame
        grid_frame = ttk.Frame(self.cal_frame)
        grid_frame.pack(fill="both", expand=True, pady=2)
        
        # Create day buttons for the calendar
        for week_num, week in enumerate(cal):
            for day_num, day in enumerate(week):
                if day == 0:
                    # Empty cell for days outside the month
                    frame = ttk.Frame(grid_frame, width=30, height=30)
                    frame.grid(row=week_num, column=day_num, padx=1, pady=1)
                    frame.grid_propagate(False)  # Keep the size fixed
                else:
                    # Create a frame for the day
                    frame = ttk.Frame(grid_frame, width=30, height=30)
                    frame.grid(row=week_num, column=day_num, padx=1, pady=1)
                    frame.grid_propagate(False)  # Keep the size fixed
                    
                    # Calculate the weekday (0=Monday, 6=Sunday)
                    date = datetime(self.current_year, self.current_month, day)
                    weekday = date.weekday()
                    
                    # Check if it's a working day
                    is_working_day = self.workdays[weekday]
                    
                    # Check if it's today
                    is_today = (date.date() == datetime.now().date())
                    
                    # Create the day button with appropriate style
                    bg_color = "#c5e0b4" if is_working_day else "#f0f0f0"
                    if is_today:
                        bg_color = "#ffd700"  # Gold for today
                    
                    day_label = tk.Label(
                        frame, 
                        text=str(day),
                        background=bg_color,
                        width=4,
                        height=2
                    )
                    day_label.pack(fill="both", expand=True)
                    
                    # Store the date in the label for reference
                    day_label.date = date
                    
                    # Bind click event
                    day_label.bind("<Button-1>", self._on_day_click)
        
    def _prev_month(self) -> None:
        """Move to the previous month."""
        self.current_month -= 1
        if self.current_month < 1:
            self.current_month = 12
            self.current_year -= 1
        self._update_calendar()
        
    def _next_month(self) -> None:
        """Move to the next month."""
        self.current_month += 1
        if self.current_month > 12:
            self.current_month = 1
            self.current_year += 1
        self._update_calendar()
        
    def _on_day_click(self, event) -> None:
        """
        Handle day click events.
        
        Args:
            event: Mouse event
        """
        # Get the date from the clicked label
        date = event.widget.date
        
        # For demo purposes, just print the date
        print(f"Clicked on {date.strftime('%Y-%m-%d')}")
        
    def set_workdays(self, workdays: List[bool]) -> None:
        """
        Set which days are working days.
        
        Args:
            workdays: List of 7 boolean values (Mon-Sun)
        """
        if len(workdays) != 7:
            raise ValueError("Workdays must be a list of 7 boolean values")
            
        self.workdays = workdays
        self._update_calendar()
        
    def set_date(self, year: int, month: int) -> None:
        """
        Set the displayed month and year.
        
        Args:
            year: Year to display
            month: Month to display (1-12)
        """
        if not (1 <= month <= 12):
            raise ValueError("Month must be between 1 and 12")
            
        self.current_year = year
        self.current_month = month
        self._update_calendar()
