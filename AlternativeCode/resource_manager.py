# ResourceManager class tracks available, total, allocated resources. 
# Methods to allocate and release resources, check availability.
# Created: 2 Feb 2025

from typing import Dict, Any
from AlternativeCode.models import SimulationEvent
import pandas as pd

class ResourceManager:
    def __init__(self, resources_df: pd.DataFrame):
        self.resources = self._initialize_resources(resources_df)

    def _initialize_resources(self, resources_df: pd.DataFrame) -> Dict[str, Dict]:
        return {
            row['ResourceType']: {
                'total': row['Count'],
                'available': row['Count'],
                'allocated': 0
            }
            for _, row in resources_df.iterrows()
        }

    def allocate_resources(self, required: Dict[str, int]) -> bool:
        if self.check_availability(required):
            for rt, qty in required.items():
                self.resources[rt]['available'] -= qty
                self.resources[rt]['allocated'] += qty
            return True
        return False

    def release_resources(self, required: Dict[str, int]):
        for rt, qty in required.items():
            self.resources[rt]['available'] += qty
            self.resources[rt]['allocated'] -= qty

    def check_availability(self, required: Dict[str, int]) -> bool:
        return all(
            self.resources[rt]['available'] >= qty
            for rt, qty in required.items()
        )
    