import json
import heapq
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import random
from models import ProcessNode, ProcessTransition, SimulationEvent, Token
from xpdl_parser import XPDLParser
from excel_loader import ExcelLoader
from resource_manager import ResourceManager
from time_calculator import TimeCalculator
from event_logger import EventLogger
import pandas as pd

class SimulationEngine:
    def __init__(self, xpdl_path: Path, excel_path: Path, json_parsed_path: Path = Path("xpdl_parsed.json")):
        if json_parsed_path.exists():
            # Load from JSON if available
            with open(json_parsed_path, 'r') as json_file:
                parser_result = json.load(json_file)
            
            # Convert JSON back to ProcessNode & ProcessTransition objects
            self.nodes = {k: ProcessNode(**v) for k, v in parser_result['nodes'].items()}
            self.transitions = [ProcessTransition(**t) for t in parser_result['transitions']]
        else:
            # Parse XPDL if JSON doesn't exist
            parser_result = XPDLParser.parse_xpdl(str(xpdl_path), str(json_parsed_path))
            self.nodes = {k: ProcessNode(**v) for k, v in parser_result['nodes'].items()}
            self.transitions = [ProcessTransition(**t) for t in parser_result['transitions']]

        self.params = ExcelLoader.load_all_sheets(excel_path)
        self.event_queue = []
        self.token_counter = 0
        self.current_time = 0
        self.logger = EventLogger()
        self.resource_manager = ResourceManager(self.params.get('Resources', pd.DataFrame()))
        self.time_calculator = TimeCalculator()


    def _schedule_initial_events(self):
        # Ensure the correct sheet name is used
        arrival_rate_df = self.params.get('ArrivalRate', pd.DataFrame())

        if arrival_rate_df.empty:
            raise ValueError("ArrivalRate sheet is missing or empty in the Excel file.")

        # Extracting the first row as a dictionary (assuming only one row exists)
        arrival_rate = arrival_rate_df.iloc[0].to_dict()

        # Extract required parameters
        num_tokens = int(arrival_rate.get('Number Of Tokens', 1))
        min_interval = float(arrival_rate.get('Minimum Arrival Time', 1))
        avg_interval = float(arrival_rate.get('Average Arrival Time', 2))
        max_interval = float(arrival_rate.get('Maximum Arrival Time', 5))

        # Find the starting node
        start_node = next(
            (n for n in self.nodes if n.node_type == 'activity' and 'start' in n.name.lower()),
            None
        )
        if start_node is None:
            raise ValueError("No starting activity node found with 'start' in name")

        for _ in range(num_tokens):
            # Using triangular distribution for arrival interval
            arrival_interval = random.triangular(min_interval, avg_interval, max_interval)
            arrival_time = self.current_time + timedelta(minutes=arrival_interval)
            
            token = Token(current_node_id=start_node.id)
            heapq.heappush(self.event_queue, (arrival_time, 'TOKEN_ARRIVAL', token))

    def _process_events(self):
        while self.event_queue:
            event_time, event_type, event_data = heapq.heappop(self.event_queue)
            self.current_time = event_time
            
            if event_type == 'TOKEN_ARRIVAL':
                self._handle_token_arrival(event_data)  # Pass the token
            elif event_type == 'PROCESS_NODE':
                self._handle_process_node(event_data)
            elif event_type == 'ACTIVITY_COMPLETE':
                self._handle_activity_completion(event_data)

    def _handle_token_arrival(self, token: Token):
        """
        Handles the arrival of a token at a node, determining the next steps based on the node's type.
        """
        self.logger.add_event(SimulationEvent(
            timestamp=self.current_time,
            token=token_id,
            node=token.current_node_id,
            event_type='DEBUG',
            details={'message': f"Token current_node_id: {token.current_node_id}"}
        ))

        self.logger.add_event(SimulationEvent(
            timestamp=self.current_time,
            token=token_id,
            node=token.current_node_id,
            event_type='DEBUG',
            details={'message': f"Process model nodes: {[node.id for node in self.nodes]}"}
        ))

        start_node = next(
            (node for node in self.nodes if node.id == token.current_node_id),
            None
        )
        
        if start_node is None:
            raise ValueError(f"No node found with ID {token.current_node_id} in process model nodes")
        
        self.token_counter += 1
        token_id = self.token_counter
        
        self.logger.add_event(SimulationEvent(
            timestamp=self.current_time,
            token=token_id,
            node=start_node.id,
            event_type='TOKEN_CREATED'
        ))
        
        # Log token and node information instead of printing
        self.logger.add_event(SimulationEvent(
            timestamp=self.current_time,
            token=token_id,
            node=start_node.id,
            event_type='DEBUG',
            details={'message': f"Token arrived at node {token.current_node_id}"}
        ))
        self.logger.add_event(SimulationEvent(
            timestamp=self.current_time,
            token=token_id,
            node=start_node.id,
            event_type='DEBUG',
            details={'message': f"Available nodes: {[node.id for node in self.nodes]}"}
        ))
        
        self._schedule_node_processing(token_id, start_node.id)

    def _handle_process_node(self, event_data: Dict):
        token_id = event_data['token']
        node_id = event_data['node']
        node = next((n for n in self.nodes if n.id == node_id), None)
        
        if node is None:
            raise ValueError(f"No node found with ID {node_id}")
        
        if node.node_type == 'activity':
            self._process_activity(token_id, node)
        elif node.node_type in ['exclusive', 'parallel']:
            self._process_gateway(token_id, node)
            
    def _process_activity(self, token_id: int, node: ProcessNode):
        activity_times = self.params.get('ActivityTimes', {})
        times = activity_times.get(node.name, {'min': 1, 'mode': 5, 'max': 10})  # Default values
        duration = self.time_calculator.triangular_duration(
            min_value=times['min'],
            mode_value=times['mode'],  # Use 'mode' instead of 'avg' for triangular
            max_value=times['max']
        )
        
        required_resources = self._get_required_resources(node.name)
        can_start = all(
            self.resource_manager.get_available_resources(rt, self.current_time, duration) >= qty
            for rt, qty in required_resources.items()
        )
        
        if can_start:
            self._allocate_and_schedule(token_id, node, required_resources, duration)
        else:
            self._handle_resource_wait(token_id, node, required_resources, duration)

    def _handle_resource_wait(self, token_id: int, node: ProcessNode, resources: Dict[str, int], duration: float):
        next_available_times = [
            self.resource_manager.get_next_available_time(rt, self.current_time, duration)
            for rt, qty in resources.items()
        ]
        retry_time = max(next_available_times) if next_available_times else self.current_time + timedelta(minutes=15)

        self.logger.add_event(SimulationEvent(
            timestamp=self.current_time,
            token=token_id,
            node=node.id,
            event_type='WAITING_FOR_RESOURCES',
            details={'resources': resources, 'retry_at': retry_time}
        ))
        
        heapq.heappush(
            self.event_queue,
            (retry_time, 'PROCESS_NODE', {'token': token_id, 'node': node.id})
        )
    
    def _process_gateway(self, token_id: int, node: ProcessNode):
        outgoing_transitions = [t for t in self.transitions if t.from_node == node.id]
        next_nodes = []
        
        if node.node_type == 'exclusive':
            if not outgoing_transitions:
                raise ValueError(f"No outgoing transitions for exclusive gateway {node.id}")
            
            # Use gateway probabilities if provided, otherwise use uniform/triangular distribution
            probs = self.params.get('gateway_probs', {}).get(node.id, None)
            if probs:
                # probs is a dict of transition IDs to probabilities (summing to 1)
                weights = [probs.get(t.id, 1.0 / len(outgoing_transitions)) for t in outgoing_transitions]
            else:
                # Default to triangular distribution for probabilities (simulating Bizagi's behavior)
                weights = [self.time_calculator.triangular_probability(0, 0.5, 1) for _ in outgoing_transitions]
                weights = [w / sum(weights) for w in weights]  # Normalize to sum to 1
            
            chosen_transition = random.choices(outgoing_transitions, weights=weights)[0]
            next_nodes.append(chosen_transition.to_node)
        
        elif node.node_type == 'parallel':
            next_nodes = [t.to_node for t in outgoing_transitions]
        
        for next_node_id in next_nodes:
            self._schedule_node_processing(token_id, next_node_id)
    
    def _schedule_node_processing(self, token_id: int, node_id: str):
        heapq.heappush(
            self.event_queue,
            (self.current_time, 'PROCESS_NODE', {'token': token_id, 'node': node_id})
        )
    
    def _export_process_json(self, output_path: Path):
        import json
        with open(output_path, 'w') as f:
            json.dump({
                'nodes': {node.id: vars(node) for node in self.nodes},
                'transitions': [vars(t) for t in self.transitions]
            }, f, indent=4)

    def run(self, output_path: Path):
        self._schedule_initial_events()
        self._process_events()
        self.logger.write_log_file(output_path / 'simulation_log.txt')
        self._export_process_json(output_path / 'process_structure.json')
