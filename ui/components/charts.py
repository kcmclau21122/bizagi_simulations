import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from scipy import stats

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

class ChartFrame(ScrollableFrame):
    """
    Base class for chart frames that embed matplotlib figures in tkinter.
    """
    
    def __init__(self, parent, figsize: Tuple[int, int] = (6, 4), 
                 with_toolbar: bool = False, **kwargs):
        """
        Initialize the chart frame.
        
        Args:
            parent: Parent widget
            figsize: Figure size (width, height) in inches
            with_toolbar: Whether to include the matplotlib navigation toolbar
            **kwargs: Additional keyword arguments for Frame
        """
        super().__init__(parent, **kwargs)
        
        # Create figure and canvas
        self.figure, self.ax = plt.subplots(figsize=figsize)
        self.matplotlib_canvas = FigureCanvasTkAgg(self.figure, master=self.scrollable_frame)
        self.matplotlib_canvas.draw()
        
        # Set up layout
        self.matplotlib_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar if requested
        if with_toolbar:
            self.toolbar = NavigationToolbar2Tk(self.matplotlib_canvas, self.scrollable_frame)
            self.toolbar.update()
            self.toolbar.pack(fill=tk.X)
            
        # Configure canvas and frame to handle resizing properly
        self.matplotlib_canvas.get_tk_widget().bind("<Configure>", self._on_canvas_resize)
            
    def _on_canvas_resize(self, event):
        """Handle matplotlib canvas resize events."""
        # Update the figure layout when the canvas is resized
        self.figure.tight_layout()
        self.matplotlib_canvas.draw()
            
    def clear(self) -> None:
        """Clear the chart."""
        self.ax.clear()
        self.matplotlib_canvas.draw()
        
    def update(self) -> None:
        """Update the canvas drawing."""
        self.matplotlib_canvas.draw()
        
    def save_figure(self, path: str, dpi: int = 300) -> None:
        """
        Save the figure to a file.
        
        Args:
            path: File path to save to
            dpi: Resolution in dots per inch
        """
        self.figure.savefig(path, dpi=dpi, bbox_inches='tight')

class BarChartFrame(ChartFrame):
    """
    Frame for displaying bar charts.
    """
    
    def __init__(self, parent, horizontal: bool = False, **kwargs):
        """
        Initialize the bar chart frame.
        
        Args:
            parent: Parent widget
            horizontal: Whether to display bars horizontally
            **kwargs: Additional keyword arguments for ChartFrame
        """
        super().__init__(parent, **kwargs)
        self.horizontal = horizontal
        
    def plot(self, categories: List[str], values: List[float], 
             title: str = "", xlabel: str = "", ylabel: str = "",
             color: str = 'blue', **kwargs) -> None:
        """
        Plot a bar chart.
        
        Args:
            categories: Category labels
            values: Values for each category
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            color: Bar color
            **kwargs: Additional keyword arguments for bar plot
        """
        self.clear()
        
        if self.horizontal:
            bars = self.ax.barh(categories, values, color=color, **kwargs)
            self.ax.set_xlabel(ylabel or "Value")
            self.ax.set_ylabel(xlabel or "Category")
        else:
            bars = self.ax.bar(categories, values, color=color, **kwargs)
            self.ax.set_xlabel(xlabel or "Category")
            self.ax.set_ylabel(ylabel or "Value")
            
            # Rotate x-axis labels for better readability
            plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right')
        
        self.ax.set_title(title)
        
        # Add value labels on bars
        self._add_value_labels(bars)
        
        # Adjust layout
        self.figure.tight_layout()
        self.update()
        
    def _add_value_labels(self, bars) -> None:
        """
        Add value labels on top of bars.
        
        Args:
            bars: Bar container from matplotlib
        """
        for bar in bars:
            if self.horizontal:
                width = bar.get_width()
                self.ax.text(
                    width + (width * 0.01),  # Small offset
                    bar.get_y() + bar.get_height() / 2,
                    f'{width:.1f}',
                    ha='left', 
                    va='center'
                )
            else:
                height = bar.get_height()
                self.ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + (height * 0.01),  # Small offset
                    f'{height:.1f}',
                    ha='center', 
                    va='bottom'
                )

class LineChartFrame(ChartFrame):
    """
    Frame for displaying line charts.
    """
    
    def plot(self, x_values: List[Any], y_values: List[float], 
             title: str = "", xlabel: str = "", ylabel: str = "",
             color: str = 'blue', marker: str = 'o', **kwargs) -> None:
        """
        Plot a line chart.
        
        Args:
            x_values: X-axis values
            y_values: Y-axis values
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            color: Line color
            marker: Point marker style
            **kwargs: Additional keyword arguments for line plot
        """
        self.clear()
        
        self.ax.plot(x_values, y_values, color=color, marker=marker, **kwargs)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        
        # Add grid for better readability
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        self.figure.tight_layout()
        self.update()
        
    def plot_multiple(self, x_values: List[Any], y_data: Dict[str, List[float]], 
                    title: str = "", xlabel: str = "", ylabel: str = "",
                    **kwargs) -> None:
        """
        Plot multiple lines on the same chart.
        
        Args:
            x_values: X-axis values
            y_data: Dictionary mapping series names to y-values
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            **kwargs: Additional keyword arguments for line plot
        """
        self.clear()
        
        for name, y_values in y_data.items():
            self.ax.plot(x_values, y_values, marker='o', label=name, **kwargs)
            
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        
        # Add legend
        self.ax.legend()
        
        # Add grid for better readability
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        self.figure.tight_layout()
        self.update()

class PieChartFrame(ChartFrame):
    """
    Frame for displaying pie charts.
    """
    
    def plot(self, labels: List[str], values: List[float], 
             title: str = "", autopct: str = '%1.1f%%',
             colors: Optional[List[str]] = None, 
             explode: Optional[List[float]] = None,
             **kwargs) -> None:
        """
        Plot a pie chart.
        
        Args:
            labels: Category labels
            values: Values for each category
            title: Chart title
            autopct: Format string for percentage display
            colors: Slice colors
            explode: Explode values for slices
            **kwargs: Additional keyword arguments for pie plot
        """
        self.clear()
        
        if not colors:
            colors = plt.cm.Set3.colors
            
        self.ax.pie(
            values,
            labels=labels,
            autopct=autopct,
            colors=colors,
            explode=explode,
            shadow=True,
            startangle=90,
            **kwargs
        )
        
        self.ax.set_title(title)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        self.ax.axis('equal')
        
        # Adjust layout
        self.figure.tight_layout()
        self.update()
        
    def plot_donut(self, labels: List[str], values: List[float], 
                  title: str = "", autopct: str = '%1.1f%%',
                  colors: Optional[List[str]] = None,
                  inner_radius: float = 0.3,
                  **kwargs) -> None:
        """
        Plot a donut chart (pie chart with a hole).
        
        Args:
            labels: Category labels
            values: Values for each category
            title: Chart title
            autopct: Format string for percentage display
            colors: Slice colors
            inner_radius: Size of the inner circle (hole)
            **kwargs: Additional keyword arguments for pie plot
        """
        self.clear()
        
        if not colors:
            colors = plt.cm.Set3.colors
            
        # Create a pie chart
        wedges, texts, autotexts = self.ax.pie(
            values,
            labels=labels,
            autopct=autopct,
            colors=colors,
            startangle=90,
            **kwargs
        )
        
        # Create the center circle to make it a donut
        centre_circle = plt.Circle((0, 0), inner_radius, fc='white')
        self.ax.add_patch(centre_circle)
        
        self.ax.set_title(title)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        self.ax.axis('equal')
        
        # Adjust layout
        self.figure.tight_layout()
        self.update()

class ScatterChartFrame(ChartFrame):
    """
    Frame for displaying scatter plots.
    """
    
    def plot(self, x_values: List[float], y_values: List[float], 
             title: str = "", xlabel: str = "", ylabel: str = "",
             color: str = 'blue', marker: str = 'o',
             with_regression: bool = False, **kwargs) -> None:
        """
        Plot a scatter chart.
        
        Args:
            x_values: X-axis values
            y_values: Y-axis values
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            color: Point color
            marker: Point marker style
            with_regression: Whether to add a regression line
            **kwargs: Additional keyword arguments for scatter plot
        """
        self.clear()
        
        self.ax.scatter(x_values, y_values, color=color, marker=marker, **kwargs)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        
        # Add regression line if requested
        if with_regression and len(x_values) > 1 and len(y_values) > 1:
            try:
                # Calculate trend line
                z = np.polyfit(x_values, y_values, 1)
                p = np.poly1d(z)
                
                # Add line to the plot
                x_range = np.linspace(min(x_values), max(x_values), 100)
                self.ax.plot(x_range, p(x_range), 'r--', alpha=0.7)
                
                # Add correlation coefficient
                correlation = np.corrcoef(x_values, y_values)[0, 1]
                self.ax.text(
                    0.05, 0.95, 
                    f'Correlation: {correlation:.2f}',
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
            except Exception as e:
                print(f"Error calculating regression: {e}")
        
        # Add grid for better readability
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        self.figure.tight_layout()
        self.update()
        
    def plot_bubble(self, x_values: List[float], y_values: List[float], 
                   sizes: List[float], labels: Optional[List[str]] = None,
                   title: str = "", xlabel: str = "", ylabel: str = "",
                   cmap: str = 'viridis', alpha: float = 0.7,
                   **kwargs) -> None:
        """
        Plot a bubble chart (scatter plot with varying point sizes).
        
        Args:
            x_values: X-axis values
            y_values: Y-axis values
            sizes: Point sizes
            labels: Point labels
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            cmap: Colormap name
            alpha: Transparency level
            **kwargs: Additional keyword arguments for scatter plot
        """
        self.clear()
        
        # Scale sizes for better visibility
        min_size = 20  # Minimum bubble size
        max_size = 500  # Maximum bubble size
        
        if len(sizes) > 0:
            size_min = min(sizes)
            size_max = max(sizes)
            
            # Avoid division by zero
            if size_min == size_max:
                scaled_sizes = [min_size + (max_size - min_size) / 2] * len(sizes)
            else:
                # Scale sizes between min_size and max_size
                scaled_sizes = [
                    min_size + (s - size_min) * (max_size - min_size) / (size_max - size_min)
                    for s in sizes
                ]
        else:
            scaled_sizes = [min_size] * len(x_values)
        
        # Create scatter plot with scaled sizes
        scatter = self.ax.scatter(
            x_values, y_values, 
            s=scaled_sizes, 
            c=range(len(x_values)),  # Color by index for variety
            cmap=cmap, 
            alpha=alpha, 
            **kwargs
        )
        
        # Add labels if provided
        if labels:
            for i, label in enumerate(labels):
                self.ax.annotate(
                    label,
                    (x_values[i], y_values[i]),
                    xytext=(5, 5),
                    textcoords='offset points'
                )
        
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        
        # Add grid for better readability
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        self.figure.tight_layout()
        self.update()

class HistogramChartFrame(ChartFrame):
    """
    Frame for displaying histograms.
    """
    
    def plot(self, values: List[float], bins: Union[int, List[float]] = 10,
             title: str = "", xlabel: str = "", ylabel: str = "Frequency",
             color: str = 'blue', with_kde: bool = False, **kwargs) -> None:
        """
        Plot a histogram.
        
        Args:
            values: Data values
            bins: Number of bins or bin edges
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            color: Bar color
            with_kde: Whether to add a kernel density estimate curve
            **kwargs: Additional keyword arguments for histogram plot
        """
        self.clear()
        
        # Plot histogram
        n, bins, patches = self.ax.hist(
            values, 
            bins=bins, 
            color=color, 
            alpha=0.7, 
            **kwargs
        )
        
        # Add KDE curve if requested
        if with_kde and len(values) > 1:
            try:
                from scipy import stats
                
                # Calculate KDE
                kde_x = np.linspace(min(values), max(values), 1000)
                kde = stats.gaussian_kde(values)
                kde_y = kde(kde_x)
                
                # Scale KDE to match histogram height
                scaling_factor = max(n) / max(kde_y) if max(kde_y) > 0 else 1
                kde_y_scaled = kde_y * scaling_factor
                
                # Plot KDE curve
                self.ax.plot(kde_x, kde_y_scaled, 'r-', linewidth=2)
                
                # Add a second y-axis for the KDE
                ax2 = self.ax.twinx()
                ax2.plot(kde_x, kde_y, 'r-', linewidth=0)  # Invisible line for scaling
                ax2.set_ylabel('Density')
                ax2.tick_params(axis='y', colors='red')
                
            except ImportError:
                print("scipy not available, skipping KDE")
            except Exception as e:
                print(f"Error calculating KDE: {e}")
        
        # Add mean line
        if len(values) > 0:
            mean = np.mean(values)
            self.ax.axvline(
                mean, 
                color='red', 
                linestyle='dashed', 
                linewidth=1
            )
            self.ax.text(
                mean, 
                max(n) * 0.9, 
                f'Mean: {mean:.2f}', 
                color='red',
                horizontalalignment='right' 
                if mean > np.median(values) else 'left'
            )
            
            # Add median line
            median = np.median(values)
            self.ax.axvline(
                median, 
                color='green', 
                linestyle='dashed', 
                linewidth=1
            )
            self.ax.text(
                median, 
                max(n) * 0.8, 
                f'Median: {median:.2f}', 
                color='green',
                horizontalalignment='left' 
                if mean > median else 'right'
            )
        
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        
        # Add grid for better readability
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        self.figure.tight_layout()
        self.update()

class BoxPlotFrame(ChartFrame):
    """
    Frame for displaying box plots.
    """
    
    def plot(self, data: Union[List[List[float]], Dict[str, List[float]]],
             title: str = "", xlabel: str = "", ylabel: str = "",
             vert: bool = True, **kwargs) -> None:
        """
        Plot a box plot.
        
        Args:
            data: Dictionary mapping group names to data values, or list of data lists
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            vert: Whether to display boxes vertically
            **kwargs: Additional keyword arguments for box plot
        """
        self.clear()
        
        if isinstance(data, dict):
            # If data is a dictionary, convert it to the format expected by boxplot
            labels = list(data.keys())
            data_values = [data[label] for label in labels]
        else:
            # If data is a list of lists, use it directly
            data_values = data
            labels = [f'Group {i+1}' for i in range(len(data))]
        
        # Create box plot
        boxplot = self.ax.boxplot(
            data_values,
            vert=vert,
            patch_artist=True,  # Fill boxes with color
            labels=labels,
            **kwargs
        )
        
        # Add some styling to the boxes
        for box in boxplot['boxes']:
            box.set(
                facecolor='lightblue',  # Box fill color
                edgecolor='blue',       # Box edge color
                alpha=0.7
            )
            
        # Style the whiskers
        for whisker in boxplot['whiskers']:
            whisker.set(
                color='gray',
                linewidth=1.5,
                linestyle='--'
            )
            
        # Style the caps
        for cap in boxplot['caps']:
            cap.set(color='gray', linewidth=2)
            
        # Style the median lines
        for median in boxplot['medians']:
            median.set(color='red', linewidth=2)
            
        # Style the fliers (outliers)
        for flier in boxplot['fliers']:
            flier.set(
                marker='o',
                markerfacecolor='red',
                markersize=6,
                alpha=0.5
            )
        
        if vert:
            self.ax.set_xlabel(xlabel or "Groups")
            self.ax.set_ylabel(ylabel or "Values")
        else:
            self.ax.set_xlabel(ylabel or "Values")
            self.ax.set_ylabel(xlabel or "Groups")
            
        self.ax.set_title(title)
        
        # Add grid for better readability
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        self.figure.tight_layout()
        self.update()

        # Calculate and display statistics if there's enough data
        if all(len(d) > 0 for d in data_values):
            stats_text = "Statistics:\n"
            
            for i, group_data in enumerate(data_values):
                if not group_data:
                    continue
                    
                group_name = labels[i]
                mean = np.mean(group_data)
                median = np.median(group_data)
                std_dev = np.std(group_data)
                
                stats_text += f"\n{group_name}:\n"
                stats_text += f"Mean: {mean:.2f}\n"
                stats_text += f"Median: {median:.2f}\n"
                stats_text += f"Std Dev: {std_dev:.2f}\n"
                
            # Add statistics in a text box
            self.ax.text(
                1.05, 0.5, 
                stats_text, 
                transform=self.ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='center'
            )
            