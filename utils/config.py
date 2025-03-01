import os
import json
import logging
from typing import Dict, Any, Optional, List

class ConfigManager:
    """
    Manages application configuration and settings.
    Handles loading, saving, and accessing configuration values.
    """
    
    def __init__(self, settings_file: str = "bizagi_simulator_settings.json"):
        """
        Initialize the configuration manager.
        
        Args:
            settings_file: Path to the settings file
        """
        self.settings_file = settings_file
        self.defaults = {
            "xpdl_file_path": "",
            "metrics_file_path": "",
            "simulation_days": 2,
            "target_avg_time": 0.0,
            "random_seed": 10,
            "workdays": [True, True, True, True, True, False, False],
            "work_hours_start": 9,
            "work_hours_end": 17
        }
        self.config = self.defaults.copy()
        self.load()
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key to get
            default: Default value if key not found
            
        Returns:
            The configuration value or default
        """
        return self.config.get(key, default)
        
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: The configuration key to set
            value: The value to set
        """
        self.config[key] = value
        
    def load(self) -> bool:
        """
        Load settings from file.
        
        Returns:
            True if settings were loaded successfully, False otherwise
        """
        if not os.path.exists(self.settings_file):
            logging.info(f"Settings file {self.settings_file} not found. Using defaults.")
            return False
            
        try:
            with open(self.settings_file, 'r') as f:
                loaded_config = json.load(f)
                
            # Apply loaded settings, keeping defaults for missing values
            for key, value in loaded_config.items():
                self.config[key] = value
                
            logging.info(f"Settings loaded from {self.settings_file}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading settings: {str(e)}")
            return False
            
    def save(self) -> bool:
        """
        Save settings to file.
        
        Returns:
            True if settings were saved successfully, False otherwise
        """
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.config, f, indent=4)
                
            logging.info(f"Settings saved to {self.settings_file}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving settings: {str(e)}")
            return False
            
    def reset_to_defaults(self) -> None:
        """Reset all settings to default values."""
        self.config = self.defaults.copy()
        
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Dictionary of all configuration values
        """
        return self.config.copy()
        
    def get_number_of_workdays(self) -> int:
        """
        Get the number of configured work days per week.
        
        Returns:
            Number of work days (1-7)
        """
        workdays = self.get("workdays", [True] * 5 + [False] * 2)
        return sum(1 for day in workdays if day)
        
    def get_work_hours_per_day(self) -> int:
        """
        Get the number of work hours per day.
        
        Returns:
            Number of work hours per day
        """
        start = self.get("work_hours_start", 9)
        end = self.get("work_hours_end", 17)
        return max(0, end - start)
        
    def __getitem__(self, key: str) -> Any:
        """Dictionary-like access to configuration."""
        return self.get(key)
        
    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-like setting of configuration."""
        self.set(key, value)
        
    def items(self) -> List[tuple]:
        """Get items like a dictionary."""
        return self.config.items()
