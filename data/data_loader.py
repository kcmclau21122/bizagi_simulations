import pandas as pd
import xml.etree.ElementTree as ET
import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union

class DataLoader:
    """
    Handles loading and preprocessing data from various file formats.
    Responsible for loading XPDL files, simulation metrics, and other data sources.
    """
    
    @staticmethod
    def load_xpdl(file_path: str) -> ET.Element:
        """
        Load and parse an XPDL file.
        
        Args:
            file_path: Path to the XPDL file
            
        Returns:
            The parsed XML element tree root
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"XPDL file not found: {file_path}")
            
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            logging.info(f"Successfully loaded XPDL file: {file_path}")
            return root
            
        except ET.ParseError as e:
            logging.error(f"Error parsing XPDL file {file_path}: {str(e)}")
            raise ValueError(f"Invalid XPDL file: {str(e)}")
    
    @staticmethod
    def load_simulation_metrics(file_path: str, sheet_name: Union[str, int] = 0) -> pd.DataFrame:
        """
        Load simulation metrics from an Excel file.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Name or index of the sheet to load
            
        Returns:
            DataFrame containing simulation metrics
        """
        if not os.path.exists(file_path):
            logging.error(f"Metrics file not found: {file_path}")
            raise FileNotFoundError(f"Metrics file not found: {file_path}")
            
        try:
            # Load Excel file
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Handle empty dataframe
            if df.empty:
                logging.warning(f"Metrics file {file_path} is empty. Using default metrics.")
                # Create a minimal default dataframe
                df = pd.DataFrame({
                    'name': ['Start', 'Process', 'End'],
                    'type': ['Start', 'Task', 'Stop'],
                    'min time': [0, 5, 0],
                    'avg time': [0, 10, 0],
                    'max time': [0, 15, 0]
                })
            
            # Normalize column names to lowercase for consistency
            df.columns = [str(col).lower() for col in df.columns]
            
            # Handle missing required columns
            required_columns = ['name']
            for col in required_columns:
                if col not in df.columns:
                    logging.warning(f"Required column '{col}' missing from metrics. Adding default column.")
                    df[col] = [f"Activity_{i}" for i in range(len(df))]
            
            # Add type column if missing
            if 'type' not in df.columns:
                logging.warning("Type column missing from metrics. Adding default 'Task' type.")
                df['type'] = 'Task'
                # Set first row to Start if there's at least one row
                if len(df) > 0:
                    df.loc[0, 'type'] = 'Start'
                # Set last row to Stop if there are at least two rows
                if len(df) > 1:
                    df.loc[len(df)-1, 'type'] = 'Stop'
            
            # Add time columns if missing
            time_columns = ['min time', 'avg time', 'max time']
            for col in time_columns:
                if col not in df.columns:
                    logging.warning(f"Time column '{col}' missing from metrics. Adding default values.")
                    default_value = 5 if col == 'avg time' else (1 if col == 'min time' else 10)
                    df[col] = default_value
                else:
                    # Ensure numeric values
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(
                        5 if col == 'avg time' else (1 if col == 'min time' else 10)
                    )
            
            # Log basic information about the loaded data
            logging.info(f"Successfully loaded metrics from {file_path}")
            logging.info(f"Loaded {len(df)} rows with columns: {', '.join(df.columns)}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading metrics file {file_path}: {str(e)}")
            raise ValueError(f"Invalid metrics file: {str(e)}")

    
    @staticmethod
    def validate_simulation_metrics(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate simulation metrics data for required columns and data integrity.
        
        Args:
            df: DataFrame containing simulation metrics
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        required_columns = ['name', 'type']
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(
                f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        # Check for duplicate names
        if 'name' in df.columns:
            duplicate_names = df[df.duplicated('name', keep=False)]
            if not duplicate_names.empty:
                errors.append(
                    f"Found {len(duplicate_names)} duplicate activity names: "
                    f"{', '.join(duplicate_names['name'].unique())}"
                )
        
        # Check for missing values in critical columns
        for col in [c for c in required_columns if c in df.columns]:
            missing_values = df[col].isnull().sum()
            if missing_values > 0:
                errors.append(
                    f"Column '{col}' has {missing_values} missing values"
                )
        
        # Check numeric columns for invalid values
        numeric_columns = [
            col for col in df.columns 
            if col.startswith(('min', 'avg', 'max', 'arrival')) and col in df.columns
        ]
        
        for col in numeric_columns:
            if df[col].dtype != 'object':  # Skip non-numeric columns
                negative_values = (df[col] < 0).sum()
                if negative_values > 0:
                    errors.append(
                        f"Column '{col}' has {negative_values} negative values"
                    )
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def extract_start_tasks(root: ET.Element) -> List[str]:
        """
        Extract start task names from XPDL XML.
        
        Args:
            root: Root element of the XPDL XML
            
        Returns:
            List of start task names
        """
        # Note: XPDL namespace varies between versions, so we use a more flexible approach
        start_tasks = []
        
        # Look for activities with trigger="None" or type="StartEvent"
        activities = root.findall(".//Activity") + root.findall(".//*[@ActivityType='StartEvent']")
        
        for activity in activities:
            # Check for start event indicators in different XPDL versions
            event = activity.find(".//Event")
            trigger = activity.get("Trigger") or (event.get("Trigger") if event is not None else None)
            activity_type = activity.get("ActivityType")
            
            if trigger == "None" or activity_type == "StartEvent":
                # Get the activity name
                name = activity.get("Name")
                if name:
                    start_tasks.append(name)
                else:
                    # Try to find name in nested elements
                    name_elem = activity.find(".//Name")
                    if name_elem is not None and name_elem.text:
                        start_tasks.append(name_elem.text)
        
        logging.info(f"Extracted {len(start_tasks)} start tasks: {', '.join(start_tasks)}")
        return start_tasks
    
    @staticmethod
    def extract_end_tasks(root: ET.Element) -> List[str]:
        """
        Extract end task names from XPDL XML.
        
        Args:
            root: Root element of the XPDL XML
            
        Returns:
            List of end task names
        """
        end_tasks = []
        
        # Look for activities with trigger="None" or type="EndEvent"
        activities = root.findall(".//Activity") + root.findall(".//*[@ActivityType='EndEvent']")
        
        for activity in activities:
            # Check for end event indicators in different XPDL versions
            event = activity.find(".//Event")
            result = activity.get("Result") or (event.get("Result") if event is not None else None)
            activity_type = activity.get("ActivityType")
            
            if result == "None" or activity_type == "EndEvent":
                # Get the activity name
                name = activity.get("Name")
                if name:
                    end_tasks.append(name)
                else:
                    # Try to find name in nested elements
                    name_elem = activity.find(".//Name")
                    if name_elem is not None and name_elem.text:
                        end_tasks.append(name_elem.text)
        
        logging.info(f"Extracted {len(end_tasks)} end tasks: {', '.join(end_tasks)}")
        return end_tasks
    
    @staticmethod
    def load_process_model_json(json_file_path: str) -> Dict[str, Any]:
        """
        Load a process model from a JSON file.
        
        Args:
            json_file_path: Path to the JSON file
            
        Returns:
            Dictionary containing the process model data
        """
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
            
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                
            logging.info(f"Successfully loaded process model from {json_file_path}")
            
            # Verify basic structure
            if not all(key in data for key in ['nodes', 'links']):
                logging.warning(f"Process model in {json_file_path} may be missing required elements")
                
            return data
            
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON file {json_file_path}: {str(e)}")
            raise ValueError(f"Invalid JSON file: {str(e)}")
    
    @staticmethod
    def load_simulation_results(xlsx_file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load simulation results from an Excel file.
        
        Args:
            xlsx_file_path: Path to the Excel file
            
        Returns:
            Dictionary mapping sheet names to DataFrames
        """
        if not os.path.exists(xlsx_file_path):
            raise FileNotFoundError(f"Results file not found: {xlsx_file_path}")
            
        try:
            # Load all sheets
            sheet_dict = pd.read_excel(xlsx_file_path, sheet_name=None)
            
            logging.info(f"Successfully loaded simulation results from {xlsx_file_path}")
            logging.info(f"Loaded {len(sheet_dict)} sheets: {', '.join(sheet_dict.keys())}")
            
            return sheet_dict
            
        except Exception as e:
            logging.error(f"Error loading results file {xlsx_file_path}: {str(e)}")
            raise ValueError(f"Invalid results file: {str(e)}")
    
    @staticmethod
    def preprocess_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess simulation metrics for use in simulation.
        Extracts gateway probabilities and organizes them for easy access.
        
        Args:
            df: DataFrame containing simulation metrics
            
        Returns:
            Preprocessed DataFrame
        """
        # Create a copy to avoid modifying the original
        preprocessed = df.copy()
        
        # Ensure column names are lowercase
        preprocessed.columns = [str(col).lower() for col in preprocessed.columns]
        
        # Fill missing values for numeric columns with appropriate defaults
        numeric_patterns = ['min ', 'avg ', 'max ', 'arrival']
        for col in preprocessed.columns:
            if any(col.startswith(pattern) for pattern in numeric_patterns):
                # For numeric columns related to time, use 0 as default
                preprocessed[col] = preprocessed[col].fillna(0)
        
        # Ensure resource counts are at least 1
        if 'available resources' in preprocessed.columns:
            preprocessed['available resources'] = preprocessed['available resources'].fillna(1)
            preprocessed['available resources'] = preprocessed['available resources'].apply(
                lambda x: max(1, int(x))
            )
        
        # Handle arrival parameters for start events
        if 'type' in preprocessed.columns and 'arrival interval' in preprocessed.columns:
            # For start events, ensure arrival interval is set
            mask = preprocessed['type'].str.lower() == 'start'
            if mask.any() and preprocessed.loc[mask, 'arrival interval'].isnull().any():
                preprocessed.loc[mask, 'arrival interval'] = 5  # Default to 5 minutes
        
        # Handle max arrival count
        if 'type' in preprocessed.columns and 'max arrival count' in preprocessed.columns:
            # For start events, ensure max arrival count is set
            mask = preprocessed['type'].str.lower() == 'start'
            if mask.any() and preprocessed.loc[mask, 'max arrival count'].isnull().any():
                preprocessed.loc[mask, 'max arrival count'] = 20  # Default to 20 tokens
        
        # Create columns for condition probabilities
        # Instead of storing dictionaries (which can cause issues), we'll create explicit columns
        
        # First, find all condition probability columns
        condition_cols = []
        for col in preprocessed.columns:
            if 'probability' in col.lower() and 'condition' in col.lower():
                condition_cols.append(col)
        
        # Process all gateway nodes
        if 'type' in preprocessed.columns:
            gateway_rows = preprocessed['type'].str.lower().str.contains('gateway|exclusive|inclusive', na=False)
            
            # For each gateway node, make sure its condition columns are properly formatted
            if any(gateway_rows):
                for idx, row in preprocessed[gateway_rows].iterrows():
                    for condition_col in condition_cols:
                        if pd.notna(row.get(condition_col)):
                            # Extract condition name from column name
                            parts = condition_col.lower().split('probability')
                            if len(parts) > 0:
                                condition_name = parts[0].strip()
                                
                                # Create a specific column for this gateway's condition
                                new_col_name = f"prob_{row['name']}_{condition_name}".replace(' ', '_')
                                preprocessed.loc[idx, new_col_name] = float(row[condition_col])
        
        logging.info("Preprocessed simulation metrics data")
        return preprocessed
    
    @staticmethod
    def get_simulation_parameters(metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract simulation parameters from metrics data.
        
        Args:
            metrics_df: DataFrame containing simulation metrics
            
        Returns:
            Dictionary of simulation parameters
        """
        params = {}
        
        # Find start events
        if 'type' in metrics_df.columns:
            start_events = metrics_df[metrics_df['type'].str.lower() == 'start']
            
            if not start_events.empty:
                # Extract arrival parameters
                first_start = start_events.iloc[0]
                
                if 'max arrival count' in metrics_df.columns:
                    params['max_arrival_count'] = int(
                        first_start.get('max arrival count', 10)
                    )
                
                if 'arrival interval' in metrics_df.columns:
                    params['arrival_interval_minutes'] = float(
                        first_start.get('arrival interval', 5)
                    )
        
        # If no start events found, use defaults
        if 'max_arrival_count' not in params:
            params['max_arrival_count'] = 10
        
        if 'arrival_interval_minutes' not in params:
            params['arrival_interval_minutes'] = 5
        
        logging.info(f"Extracted simulation parameters: {params}")
        return params
