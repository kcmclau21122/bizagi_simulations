from pathlib import Path
from typing import List
from core.event import Event  # This imports your Event class from core/event.py

class EventLogger:
    def __init__(self):
        # Initialize an empty list to store events.
        self.events: List[Event] = []

    def add_event(self, event: Event) -> None:
        """
        Add an Event to the log.
        
        Args:
            event (Event): The event object to log.
        """
        self.events.append(event)

    def write_log_file(self, file_path: Path) -> None:
        """
        Write all logged events to a file.
        
        Args:
            file_path (Path): The path to the output log file.
        """
        # Sort events by time for chronological order.
        self.events.sort(key=lambda e: e.time)
        with file_path.open("w", encoding="utf-8") as f:
            for event in self.events:
                # Format each event as: ISO time - Token <token_id>: <event_type> - <task_name>
                f.write(f"{event.time.isoformat()} - Token {event.token_id}: {event.event_type} - {event.task_name}\n")
