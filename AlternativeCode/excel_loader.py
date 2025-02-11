# ExcelLoader class with methods to load each sheet. For ActivityTimes, it loads min, avg, max into a dictionary. 
# ActivityResources are stored as a DataFrame for easy lookup.
# Created Date: 2 Feb 2025

import pandas as pd
from typing import Dict, Any
from pathlib import Path

class ExcelLoader:
    @staticmethod
    def load_all_sheets(file_path: Path) -> Dict[str, Any]:
        """
        Loads all sheets and their columns with values from the provided spreadsheet.
        Returns a dictionary where keys are sheet names and values are DataFrames or dictionaries.
        """
        xls = pd.ExcelFile(file_path)
        sheets_data = {}
        
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name)
            
            # If the first column is not labeled, reset the index
            if df.columns[0] is None:
                df.reset_index(drop=True, inplace=True)
            
            sheets_data[sheet_name] = df.to_dict(orient='list')  # Convert DataFrame to dictionary
        
        return sheets_data
