#
#
# Created: 2 Feb 2025

from datetime import datetime
from typing import List
from pathlib import Path
from .models import SimulationEvent

class EventLogger:
    def __init__(self):
        self.log: List[SimulationEvent] = []

    def add_event(self, event: SimulationEvent):
        self.log.append(event)

    def write_log_file(self, file_path: Path):
        with open(file_path, 'w') as f:
            for event in self.log:
                log_line = self._format_event(event)
                f.write(log_line + "\n")

    def _format_event(self, event: SimulationEvent) -> str:
        base = f"{event.timestamp.isoformat()} - Token {event.token} - {event.node or 'SYSTEM'} - {event.event_type}"
        
        if event.details:
            details = []
            if 'resources' in event.details:
                res_str = ", ".join(f"{k} ({v})" for k, v in event.details['resources'].items())
                details.append(f"Resources: {res_str}")
            if 'duration' in event.details:
                details.append(f"Duration: {event.details['duration']:.2f} mins")
            
            if details:
                base += " - " + " | ".join(details)
        
        return base
    