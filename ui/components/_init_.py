"""
UI components for the Bizagi Process Simulator.
Provides reusable UI elements like charts, dialogs, and visualizations.
"""

# Calendar components
from .calendar_view import CalendarView, MonthCalendarView

# Progress dialog components
from .progress_dialog import ProgressDialog, TaskProgressDialog

# Chart components
from .charts import (
    ChartFrame,
    BarChartFrame,
    LineChartFrame,
    PieChartFrame,
    ScatterChartFrame,
    HistogramChartFrame,
    BoxPlotFrame
)

__all__ = [
    # Calendar components
    'CalendarView',
    'MonthCalendarView',
    
    # Progress dialog components
    'ProgressDialog',
    'TaskProgressDialog',
    
    # Chart components
    'ChartFrame',
    'BarChartFrame',
    'LineChartFrame',
    'PieChartFrame',
    'ScatterChartFrame',
    'HistogramChartFrame',
    'BoxPlotFrame'
]
