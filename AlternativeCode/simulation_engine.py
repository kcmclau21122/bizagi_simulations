#
#
# Created: 2 Feb 2025

import heapq
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
from AlternativeCode.models import ProcessNode, SimulationEvent
from AlternativeCode.xpdl_parser import XPDLParser
from AlternativeCode.excel_loader import ExcelLoader
from AlternativeCode.resource_manager import ResourceManager
from AlternativeCode.time_calculator import TimeCalculator
from AlternativeCode.event_logger import EventLogger
import random

class SimulationEngine:
    def __init__(self, xpdl_path: Path, excel_path: Path):
        self.process = XPDLParser.parse_xpdl(xpdl_path)
        self.params = ExcelLoader.load_simulation_parameters(excel_path)
        self.resource_manager = ResourceManager(self.params['resources'])
        self.time_calculator = TimeCalculator()
        self.logger = EventLogger()
        self.event_queue = []
        self.current_time = datetime.now()
        self.token_counter = 0

    def run(self, output_path: Path):
        self._schedule_initial_events()
        self._process_events()
        self.logger.write_log_file(output_path / 'simulation_log.txt')
        self._export_process_json(output_path / 'process_structure.json')

    def _schedule_initial_events(self):
        interval = self.params['arrival_rate']['IntervalMinutes']
        num_tokens = self.params['arrival_rate']['NumberOfTokens']
        
        for _ in range(num_tokens):
            arrival_time = self.current_time + timedelta(
                minutes=random.expovariate(1/interval)
            )
            heapq.heappush(self.event_queue, (arrival_time, 'TOKEN_ARRIVAL', None))

    def _process_events(self):
        while self.event_queue:
            event_time, event_type, event_data = heapq.heappop(self.event_queue)
            self.current_time = event_time

            if event_type == 'TOKEN_ARRIVAL':
                self._handle_token_arrival()
            elif event_type == 'PROCESS_NODE':
                self._handle_process_node(event_data)
            elif event_type == 'ACTIVITY_COMPLETE':
                self._handle_activity_completion(event_data)

    def _handle_token_arrival(self):
        self.token_counter += 1
        token_id = self.token_counter
        start_node = next(
            n for n in self.process['nodes'].values() 
            if n.node_type == 'activity' and 'start' in n.name.lower()
        )
        
        self.logger.add_event(SimulationEvent(
            timestamp=self.current_time,
            token=token_id,
            node=None,
            event_type='TOKEN_CREATED'
        ))
        
        self._schedule_node_processing(token_id, start_node.id)

    def _handle_process_node(self, event_data: Dict):
        token_id = event_data['token']
        node_id = event_data['node']
        node = self.process['nodes'][node_id]

        if node.node_type == 'activity':
            self._process_activity(token_id, node)
        else:
            self._process_gateway(token_id, node)

class SimulationEngine:
    def _process_activity(self, token_id: int, node: ProcessNode):
        required_resources = self._get_required_resources(node.name)
        duration = self.time_calculator.triangular_duration(
            **self.params['activity_times'][node.name]
        )

        # Check resource availability considering calendars
        can_start = all(
            self.resource_manager.get_available_resources(rt, self.current_time, duration) >= qty
            for rt, qty in required_resources.items()
        )

        if can_start:
            self._allocate_and_schedule(token_id, node, required_resources, duration)
        else:
            self._handle_resource_wait(token_id, node, required_resources, duration)

    def _allocate_and_schedule(self, token_id: int, node: ProcessNode, resources: Dict[str, int], duration: float):
        # Allocation logic
        self.logger.add_event(SimulationEvent(
            timestamp=self.current_time,
            token=token_id,
            node=node.id,
            event_type='ACTIVITY_START',
            details={'resources': resources, 'duration': duration}
        ))
        
        completion_time = self.current_time + timedelta(minutes=duration)
        heapq.heappush(
            self.event_queue,
            (completion_time, 'ACTIVITY_COMPLETE', {'token': token_id, 'node': node.id})
        )

    def _handle_resource_wait(self, token_id: int, node: ProcessNode, resources: Dict[str, int], duration: float):
        next_available_times = []
        for rt, qty in resources.items():
            next_time = self.resource_manager.get_next_available_time(rt, self.current_time, duration)
            if next_time:
                next_available_times.append(next_time)
        
        if next_available_times:
            retry_time = max(next_available_times)
        else:
            retry_time = self.current_time + timedelta(minutes=15)

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
        
    def _handle_activity_completion(self, event_data: Dict):
        token_id = event_data['token']
        node_id = event_data['node']
        node = self.process['nodes'][node_id]
        required_resources = self._get_required_resources(node.name)

        self.resource_manager.release_resources(required_resources)
        
        self.logger.add_event(SimulationEvent(
            timestamp=self.current_time,
            token=token_id,
            node=node.id,
            event_type='ACTIVITY_END',
            details={'resources': required_resources}
        ))
        
        outgoing = [t.to_node for t in self.process['transitions'] if t.from_node == node_id]
        for next_node in outgoing:
            self._schedule_node_processing(token_id, next_node)

    def _process_gateway(self, token_id: int, node: ProcessNode):
        next_nodes = []
        
        if node.gateway_type == 'exclusive':
            outgoing = [t for t in self.process['transitions'] if t.from_node == node.id]
            probs = [self.params['gateway_probs'][t.id] for t in outgoing]
            chosen = random.choices(outgoing, weights=probs)[0]
            next_nodes.append(chosen.to_node)
        elif node.gateway_type == 'parallel':
            next_nodes = [t.to_node for t in self.process['transitions'] if t.from_node == node.id]

        for n in next_nodes:
            self._schedule_node_processing(token_id, n)

    def _schedule_node_processing(self, token_id: int, node_id: str):
        heapq.heappush(
            self.event_queue,
            (self.current_time, 'PROCESS_NODE', {'token': token_id, 'node': node_id})
        )

    def _get_required_resources(self, activity_name: str) -> Dict[str, int]:
        df = self.params['activity_resources']
        activity_df = df[df['Activity'] == activity_name]
        return activity_df.set_index('ResourceType')['Quantity'].to_dict()

    def _export_process_json(self, output_path: Path):
        import json
        with open(output_path, 'w') as f:
            json.dump({
                'nodes': {nid: vars(node) for nid, node in self.process['nodes'].items()},
                'transitions': [vars(t) for t in self.process['transitions']]
            }, f, indent=4)