import heapq
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import random
from models import ProcessNode, SimulationEvent
from xpdl_parser import XPDLParser
from excel_loader import ExcelLoader
from resource_manager import ResourceManager
from time_calculator import TimeCalculator
from event_logger import EventLogger

class SimulationEngine:
    def __init__(self, xpdl_path: Path, excel_path: Path):
        self.process = XPDLParser.parse_xpdl(xpdl_path)
        self.params = ExcelLoader.load_all_sheets(excel_path)
        self.resource_manager = ResourceManager(self.params.get('Resources', {}))
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
        arrival_rate = self.params.get('ArrivalRate', {})
        interval = arrival_rate.get('IntervalMinutes', 10)
        num_tokens = arrival_rate.get('NumberOfTokens', 1)
        
        for _ in range(num_tokens):
            arrival_time = self.current_time + timedelta(
                minutes=random.expovariate(1 / interval)
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

    def _process_activity(self, token_id: int, node: ProcessNode):
        activity_times = self.params.get('Activity Times', {})
        required_resources = self._get_required_resources(node.name)
        duration = self.time_calculator.triangular_duration(
            **activity_times.get(node.name, {'min': 1, 'avg': 2, 'max': 3})
        )
        
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
        next_nodes = []
        if node.gateway_type == 'exclusive':
            outgoing = [t for t in self.process['transitions'] if t.from_node == node.id]
            probs = [self.params['gateway_probs'].get(t.id, 1) for t in outgoing]
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
    
    def _export_process_json(self, output_path: Path):
        import json
        with open(output_path, 'w') as f:
            json.dump({
                'nodes': {nid: vars(node) for nid, node in self.process['nodes'].items()},
                'transitions': [vars(t) for t in self.process['transitions']]
            }, f, indent=4)
