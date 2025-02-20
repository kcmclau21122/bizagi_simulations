# TimeCalculator with static method for triangular distribution. 
# This keeps the calculation separate, making it easy to change later.
# Created: 2 Feb 2025

# time_calculator.py (assumed)
import random

class TimeCalculator:
    @staticmethod
    def triangular_duration(min_value: float, mode_value: float, max_value: float) -> float:
        return random.triangular(min_value, mode_value, max_value)

    @staticmethod
    def triangular_probability(min_prob: float, mode_prob: float, max_prob: float) -> float:
        return random.triangular(min_prob, mode_prob, max_prob)