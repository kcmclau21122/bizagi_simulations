# Define data structures. ProcessNode has id, name, type, gateway_type. 
# # ProcessTransition has id, from, to. SimulationEvent has timestamp, token, node, event, details.
# Original Date: 2 Feb 2025

from dataclasses import dataclass
from datetime import datetime, time, date
from typing import Optional, Dict, List

@dataclass
class ProcessNode:
    id: str
    name: str
    node_type: str  # 'activity' or 'gateway'
    gateway_type: Optional[str] = None  # 'exclusive', 'parallel', etc.

@dataclass
class ProcessTransition:
    id: str
    from_node: str
    to_node: str

@dataclass
class SimulationEvent:
    timestamp: datetime
    token: int
    node: Optional[str]
    event_type: str
    details: Optional[Dict] = None

@dataclass
class Calendar:
    name: str
    start_date: date
    end_date: date
    start_time: time
    end_time: time

@dataclass
class CalendarResource:
    calendar_name: str
    resource_type: str
    count: int