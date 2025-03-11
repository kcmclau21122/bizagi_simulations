import sys
import os
# Add project root to sys.path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import heapq
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

from core.event import Event
from core.process_token import Token
from core.resource import ResourceManager
from core.process_model import ProcessModel
from utils.time_utils import is_work_time, advance_to_work_time, TimeCalculator
from data.xpdl_parser import parse_xpdl_to_sequences

import pandas as pd

# Helper: load all sheets from an Excel file
def load_all_sheets(file_path: str) -> Dict[str, Any]:
    return pd.read_excel(file_path, sheet_name=None)

# Assume you have created an EventLogger class in reporting/event_logger.py
from reporting.event_logger import EventLogger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SimulationEngine:
    def __init__(self, *, process_model: ProcessModel, start_time: datetime, work_days: int, work_hours_per_day: int, excel_path: Optional[Path] = None):
        """
        Initialize the simulation engine using an existing ProcessModel.
        
        Args:
            process_model: The ProcessModel object to simulate.
            start_time: The start time for the simulation.
            work_days: Number of work days per week.
            work_hours_per_day: Number of work hours per day.
            excel_path: Optional path to an Excel file containing simulation parameters.
        """
        self.process = process_model
        if excel_path:
            self.params = load_all_sheets(str(excel_path))
        else:
            self.params = {}  # Use an empty dict or load defaults as needed
        self.resource_manager = ResourceManager(self.params.get('resources', {}))
        self.time_calculator = TimeCalculator()
        self.logger = EventLogger()
        self.event_queue: list = []
        self.current_time: datetime = start_time
        self.token_counter: int = 0

        logger.info(f"SimulationEngine initialized at {self.current_time.isoformat()}.")
        if 'resources' in self.params:
            available_resources = {res: count for res, count in self.params['resources'].items()}
            logger.info(f"Available resources: {available_resources}")

    def run(self, output_path: Path):
        self._schedule_initial_events()
        logger.info("Starting simulation event processing.")
        self._process_events()
        logger.info("Simulation completed. Writing log and exporting process structure.")
        self.logger.write_log_file(output_path / 'simulation_log.txt')
        self._export_process_json(output_path / 'process_structure.json')
        logger.info(f"Simulation log saved to {output_path / 'simulation_log.txt'}")
        logger.info(f"Process structure saved to {output_path / 'process_structure.json'}")

    def _schedule_initial_events(self):
        arrival_cfg = self.params.get('arrival_rate', {})
        num_tokens = arrival_cfg.get('NumberOfTokens', 0)
        if 'MinInterval' in arrival_cfg:
            min_int = arrival_cfg['MinInterval']
            avg_int = arrival_cfg.get('AvgInterval', min_int)
            max_int = arrival_cfg.get('MaxInterval', avg_int)
            logger.info(f"Scheduling {num_tokens} tokens with triangular inter-arrival distribution (min={min_int}, mode={avg_int}, max={max_int} minutes).")
            for _ in range(num_tokens):
                interval = random.triangular(min_int, max_int, avg_int)
                arrival_time = self.current_time + timedelta(minutes=interval)
                heapq.heappush(self.event_queue, (arrival_time, 'TOKEN_ARRIVAL', None))
                logger.debug(f"Event queued: TOKEN_ARRIVAL at {arrival_time.isoformat()}")
        else:
            interval_mean = arrival_cfg.get('IntervalMinutes', 1)
            logger.info(f"Scheduling {num_tokens} tokens with exponential inter-arrival distribution (mean interval={interval_mean} minutes).")
            for _ in range(num_tokens):
                arrival_time = self.current_time + timedelta(minutes=random.expovariate(1/interval_mean))
                heapq.heappush(self.event_queue, (arrival_time, 'TOKEN_ARRIVAL', None))
                logger.debug(f"Event queued: TOKEN_ARRIVAL at {arrival_time.isoformat()}")

    def _process_events(self):
        while self.event_queue:
            event_time, event_type, event_data = heapq.heappop(self.event_queue)
            self.current_time = event_time
            token_info = event_data.get('token') if event_data and 'token' in event_data else None
            node_info = event_data.get('node') if event_data and 'node' in event_data else None
            logger.debug(f"Processing event: {event_type} for token {token_info} at node {node_info} @ {event_time.isoformat()}")
            if event_type == 'TOKEN_ARRIVAL':
                self._handle_token_arrival()
            elif event_type == 'PROCESS_NODE':
                self._handle_process_node(event_data)
            elif event_type == 'ACTIVITY_COMPLETE':
                self._handle_activity_completion(event_data)

    def _handle_token_arrival(self):
        self.token_counter += 1
        token_id = self.token_counter
        # Find the start node (node with 'start' in its name)
        start_node_id, start_node = next((node_id, n) for node_id, n in self.process.nodes.items()
                          if n.get('node_type', '').lower() != 'gateway' and 'start' in n['name'].lower())
        logger.info(f"Token {token_id} created at {self.current_time.isoformat()}, entering start node '{start_node['name']}'.")
        self.logger.add_event(Event(self.current_time, token_id, start_node['name'], "TOKEN_CREATED"))
        self._schedule_node_processing(token_id, start_node_id)

    def _handle_process_node(self, event_data: Dict[str, Any]):
        token_id = event_data['token']
        node_id = event_data['node']
        node = self.process.get_node(node_id)
        logger.info(f"Token {token_id} now processing node '{node.get('name')}' (Type: {node.get('node_type')}).")
        if node.get('node_type', '').lower() == 'activity':
            self._process_activity(token_id, node)
        elif node.get('node_type', '').lower() == 'gateway':
            self._process_gateway(token_id, node)
        else:
            logger.info(f"Token {token_id} reached end node '{node.get('name')}'. Process complete.")

    def _process_activity(self, token_id: int, node: Dict[str, Any]):
        required_resources = self._get_required_resources(node.get('name'))
        duration_params = self.params['activity_times'].get(node.get('name'), {'min': 1, 'avg': 1, 'max': 1})
        duration = self.time_calculator.triangular_duration(**duration_params)
        can_start = all(self.resource_manager.get_available_resources(res, self.current_time, duration) >= qty
                        for res, qty in required_resources.items())
        if can_start:
            self._allocate_and_schedule(token_id, node, required_resources, duration)
        else:
            self._handle_resource_wait(token_id, node, required_resources, duration)

    def _allocate_and_schedule(self, token_id: int, node: Dict[str, Any], resources: Dict[str, int], duration: float):
        res_str = ", ".join(f"{r}({q})" for r, q in resources.items()) if resources else "none"
        logger.info(f"Token {token_id} started activity '{node.get('name')}' at {self.current_time.isoformat()} (duration {duration:.2f} mins, resources allocated: {res_str}).")
        self.logger.add_event(Event(self.current_time, token_id, node.get('id'), "ACTIVITY_START"))
        completion_time = self.current_time + timedelta(minutes=duration)
        heapq.heappush(self.event_queue, (completion_time, 'ACTIVITY_COMPLETE', {'token': token_id, 'node': node.get('id')}))
        logger.debug(f"Event queued: ACTIVITY_COMPLETE for token {token_id} at node '{node.get('name')}' @ {completion_time.isoformat()}")

    def _handle_resource_wait(self, token_id: int, node: Dict[str, Any], resources: Dict[str, int], duration: float):
        next_times = []
        for res, qty in resources.items():
            next_time = self.resource_manager.get_next_available_time(res, self.current_time, duration)
            if next_time:
                next_times.append(next_time)
        retry_time = max(next_times) if next_times else (self.current_time + timedelta(minutes=15))
        res_str = ", ".join(f"{r}({q})" for r, q in resources.items()) if resources else "none"
        logger.info(f"Token {token_id} waiting for resources for activity '{node.get('name')}' (needed: {res_str}). Will retry at {retry_time.isoformat()}.")
        self.logger.add_event(Event(self.current_time, token_id, node.get('id'), "WAITING_FOR_RESOURCES"))
        heapq.heappush(self.event_queue, (retry_time, 'PROCESS_NODE', {'token': token_id, 'node': node.get('id')}))
        logger.debug(f"Event queued: PROCESS_NODE (retry) for token {token_id} at node '{node.get('name')}' @ {retry_time.isoformat()}")

    def _handle_activity_completion(self, event_data: Dict[str, Any]):
        token_id = event_data['token']
        node_id = event_data['node']
        node = self.process.get_node(node_id)
        logger.info(f"Token {token_id} completed activity '{node.get('name')}' at {self.current_time.isoformat()}.")
        self.logger.add_event(Event(self.current_time, token_id, node.get('id'), "ACTIVITY_END"))
        outgoing_nodes = self.process.get_next_nodes(node_id)
        if not outgoing_nodes:
            logger.info(f"Token {token_id} has no further transitions after '{node.get('name')}' and will exit the process.")
        for next_node in outgoing_nodes:
            self._schedule_node_processing(token_id, next_node)

    def _process_gateway(self, token_id: int, node: Dict[str, Any]):
        next_nodes = []
        if node.get('gateway_type', '').lower() == 'exclusive':
            outgoing = [t for t in self.process.transitions if t.from_node == node.get('id')]
            probs = [self.params['gateway_probs'].get(t.id, 1) for t in outgoing]
            chosen_trans = random.choices(outgoing, weights=probs, k=1)[0]
            next_nodes.append(chosen_trans.to_node)
            logger.info(f"Exclusive gateway '{node.get('name')}' selected path to '{self.process.get_node(chosen_trans.to_node).get('name')}' for token {token_id}.")
        elif node.get('gateway_type', '').lower() == 'parallel':
            next_nodes = [t.to_node for t in self.process.transitions if t.from_node == node.get('id')]
            branch_names = [self.process.get_node(n).get('name') for n in next_nodes]
            logger.info(f"Parallel gateway '{node.get('name')}' sending token {token_id} to paths: {branch_names}.")
        else:
            outgoing = [t for t in self.process.transitions if t.from_node == node.get('id')]
            next_nodes = [t.to_node for t in outgoing]
            path_names = [self.process.get_node(n).get('name') for n in next_nodes]
            logger.info(f"Gateway '{node.get('name')}' (type: {node.get('gateway_type')}) sending token {token_id} to paths: {path_names}.")
        for nid in next_nodes:
            self._schedule_node_processing(token_id, nid)

    def _schedule_node_processing(self, token_id: int, node_id: str):
        heapq.heappush(self.event_queue, (self.current_time, 'PROCESS_NODE', {'token': token_id, 'node': node_id}))
        node = self.process.get_node(node_id)
        node_name = node.get('name') if node else str(node_id)
        logger.debug(f"Event queued: PROCESS_NODE for token {token_id} to node '{node_name}' @ {self.current_time.isoformat()}")
        logger.info(f"Token {token_id} scheduled to enter node '{node_name}' at {self.current_time.isoformat()}.")

    def _get_required_resources(self, activity_name: str) -> Dict[str, int]:
        if 'activity_resources' not in self.params:
            return {}
        resources_df = self.params['activity_resources']
        req = resources_df[resources_df['Activity'] == activity_name]
        if req.empty:
            return {}
        return dict(req[['ResourceType', 'Quantity']].values)

    def _export_process_json(self, output_path: Path):
        import json
        with open(output_path, 'w') as f:
            json.dump({
                'nodes': {nid: n for nid, n in self.process.nodes.items()},
                'transitions': [vars(t) for t in self.process.transitions]
            }, f, indent=4)
