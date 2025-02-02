# TimeCalculator with static method for triangular distribution. 
# This keeps the calculation separate, making it easy to change later.
# Created: 2 Feb 2025

import random

class TimeCalculator:
    @staticmethod
    def triangular_duration(min_time: float, avg_time: float, max_time: float) -> float:
        return random.triangular(min_time, max_time, avg_time)
    
    