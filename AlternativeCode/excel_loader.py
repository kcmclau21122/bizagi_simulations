# ExcelLoader class with methods to load each sheet. For ActivityTimes, it loads min, avg, max into a dictionary. 
# ActivityResources are stored as a DataFrame for easy lookup.
# Created Date: 2 Feb 2025

import pandas as pd
from typing import Dict, Any
from pathlib import Path
from .models import Calendar, CalendarResource  # Added import

class ExcelLoader:
    @staticmethod
    def load_simulation_parameters(file_path: Path) -> Dict[str, Any]:
        xls = pd.ExcelFile(file_path)
        
        return {
            'resources': ExcelLoader._load_resources(xls),
            'activity_resources': ExcelLoader._load_activity_resources(xls),
            'activity_times': ExcelLoader._load_activity_times(xls),
            'gateway_probs': ExcelLoader._load_gateway_probabilities(xls),
            'arrival_rate': ExcelLoader._load_arrival_rate(xls)
        }

    @staticmethod
    def _load_resources(xls: pd.ExcelFile) -> pd.DataFrame:
        return pd.read_excel(xls, 'Resources')

    @staticmethod
    def _load_activity_resources(xls: pd.ExcelFile) -> pd.DataFrame:
        return pd.read_excel(xls, 'ActivityResources')

    @staticmethod
    def _load_activity_times(xls: pd.ExcelFile) -> Dict[str, Dict]:
        df = pd.read_excel(xls, 'ActivityTimes')
        return {
            row['Activity']: {
                'min': row['Minimum Time'],
                'avg': row['Average Time'],
                'max': row['Maximum Time']
            }
            for _, row in df.iterrows()
        }

    @staticmethod
    def _load_gateway_probabilities(xls: pd.ExcelFile) -> Dict[str, float]:
        df = pd.read_excel(xls, 'GatewayProbs')
        return df.set_index('Gateway')['Probability'].to_dict()

    @staticmethod
    def _load_arrival_rate(xls: pd.ExcelFile) -> Dict[str, int]:
        return pd.read_excel(xls, 'ArrivalRate').iloc[0].to_dict()
    
    def load_calendars(xls: pd.ExcelFile) -> list[Calendar]:
        df = pd.read_excel(xls, 'Calendar')
        calendars = []
        for _, row in df.iterrows():
            calendars.append(Calendar(
                name=row['CalendarName'],
                start_date=row['StartDate'].date(),
                end_date=row['EndDate'].date(),
                start_time=row['StartTime'].time(),
                end_time=row['EndTime'].time()
            ))
        return calendars

    @staticmethod
    def load_calendar_resources(xls: pd.ExcelFile) -> list[CalendarResource]:
        df = pd.read_excel(xls, 'CalendarResources')
        return [
            CalendarResource(
                calendar_name=row['CalendarName'],
                resource_type=row['ResourceType'],
                count=row['Count']
            )
            for _, row in df.iterrows()
        ]    
    