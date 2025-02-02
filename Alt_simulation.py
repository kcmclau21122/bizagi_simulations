# Alternative Generated Code
# 2 Feb 2025

import xml.etree.ElementTree as ET
import pandas as pd
import heapq
import random
import json
from datetime import datetime, timedelta

class SimulationEngine:
    def __init__(self, xpdl_file, excel_file):
        self.process_graph = self.parse_xpdl(xpdl_file)
        self.export_process_json()
        self.simulation_params = self.load_excel_params(excel_file)
        self.resources = self.initialize_resources()
        self.event_queue = []
        self.current_time = datetime.now()
        self.token_counter = 0
        self.activity_log = []

    def parse_xpdl(self, file_path):
        """Parse XPDL file to extract process structure with gateway types"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        process_graph = {
            'nodes': {},
            'transitions': []
        }

        # Extract nodes with type information
        namespace = {'xpdl': 'http://www.wfmc.org/2008/XPDL2.2'}
        for node in root.findall('.//xpdl:Activity', namespace):
            node_id = node.attrib['Id']
            node_info = {
                'id': node_id,
                'name': node.attrib['Name'],
                'type': 'activity'
            }

            # Check for gateway type
            route = node.find('.//xpdl:Route', namespace)
            if route is not None:
                node_info['type'] = 'gateway'
                gateway_type = route.find('.//xpdl:GatewayType', namespace)
                if gateway_type is not None:
                    node_info['gateway_type'] = gateway_type.text.lower()
                else:
                    node_info['gateway_type'] = 'exclusive'  # default

            process_graph['nodes'][node_id] = node_info

        # Extract transitions
        for transition in root.findall('.//xpdl:Transition', namespace):
            process_graph['transitions'].append({
                'id': transition.attrib['Id'],
                'from': transition.attrib['From'],
                'to': transition.attrib['To']
            })

        return process_graph

    def export_process_json(self):
        """Export process structure to JSON file"""
        with open('process_structure.json', 'w') as f:
            json.dump(self.process_graph, f, indent=4)


    def log_event(self, token_id, node_id, event_type):
        """Log token movement events with timestamps"""
        entry = {
            'timestamp': self.current_time.isoformat(),
            'token': token_id,
            'node': node_id,
            'event': event_type
        }
        self.activity_log.append(entry)

    def initialize_resources(self):
        """Create resource pools based on Excel data"""
        return {res['ResourceType']: {
            'total': res['Count'],
            'available': res['Count']
        } for res in self.simulation_params['resources']}

    def generate_token_arrivals(self):
        """Schedule initial token arrivals"""
        interval = self.simulation_params['arrival_rate']['IntervalMinutes']
        num_tokens = self.simulation_params['arrival_rate']['NumberOfTokens']
        
        for _ in range(num_tokens):
            arrival_time = self.current_time + timedelta(minutes=random.expovariate(1/interval))
            heapq.heappush(self.event_queue, (arrival_time, 'TOKEN_ARRIVAL', None))

    def calculate_activity_duration(self, activity_name):
        """Calculate duration using triangular distribution"""
        times = self.simulation_params['activity_times'][activity_name]
        return random.triangular(
            times['min'],
            times['max'],
            times['avg']
        )

    def handle_activity(self, token, node_id):
        """Process activities with triangular time distribution"""
        node_info = self.process_graph['nodes'][node_id]
        activity_name = node_info['name']
        required_resources = self.get_required_resources(activity_name)

        # Check resource availability
        can_allocate = all(
            self.resources[rt]['available'] >= qty
            for rt, qty in required_resources.items()
        )

        if can_allocate:
            # Allocate resources
            for rt, qty in required_resources.items():
                self.resources[rt]['available'] -= qty
                self.resources[rt]['allocated'] += qty

            # Calculate duration using triangular distribution
            duration = self.calculate_activity_duration(activity_name)
            
            # Log resource allocation and duration
            self.log_event(token, node_id, 'RESOURCE_ALLOCATED', {
                'resources': required_resources.copy(),
                'duration': duration
            })
            
            # Schedule completion
            completion_time = self.current_time + timedelta(minutes=duration)
            heapq.heappush(self.event_queue, 
                (completion_time, 'ACTIVITY_COMPLETE', {'token': token, 'node': node_id}))
            
            # Log activity start
            self.log_event(token, node_id, 'ACTIVITY_START', None)
        else:
            # Log waiting for resources
            self.log_event(token, node_id, 'WAITING_FOR_RESOURCES', required_resources.copy())
            
            # Retry after 1 minute
            retry_time = self.current_time + timedelta(minutes=1)
            heapq.heappush(self.event_queue, 
                (retry_time, 'PROCESS_ACTIVITY', {'token': token, 'node': node_id}))

    def log_event(self, token_id, node_id, event_type, details=None):
        """Enhanced logging with duration information"""
        entry = {
            'timestamp': self.current_time.isoformat(),
            'token': token_id,
            'node': node_id,
            'event': event_type,
            'details': details
        }
        self.activity_log.append(entry)

    def handle_gateway(self, token, node_id):
        """Process different gateway types with logging"""
        node_info = self.process_graph['nodes'][node_id]
        self.log_event(token, node_id, 'GATEWAY_START')
        
        gateway_type = node_info.get('gateway_type', 'exclusive')
        next_nodes = []
        
        if gateway_type == 'exclusive':
            outgoing = [t for t in self.process_graph['transitions'] if t['from'] == node_id]
            probs = [self.simulation_params['gateway_probs'][t['id']] for t in outgoing]
            chosen = random.choices(outgoing, weights=probs)[0]
            next_nodes.append(chosen['to'])
        elif gateway_type == 'parallel':
            outgoing = [t for t in self.process_graph['transitions'] if t['from'] == node_id]
            next_nodes = [t['to'] for t in outgoing]
        
        self.log_event(token, node_id, 'GATEWAY_END')
        return next_nodes

    def run_simulation(self):
        """Main simulation loop with enhanced logging"""
        self.generate_token_arrivals()
        
        while self.event_queue:
            event_time, event_type, event_data = heapq.heappop(self.event_queue)
            self.current_time = event_time
            
            if event_type == 'TOKEN_ARRIVAL':
                self.token_counter += 1
                token_id = self.token_counter
                start_node = next(n for n in self.process_graph['nodes'].values() 
                                if n['type'] == 'activity' and 'start' in n['name'].lower())
                self.log_event(token_id, None, 'TOKEN_CREATED')
                heapq.heappush(self.event_queue, 
                    (self.current_time, 'PROCESS_NODE', {'token': token_id, 'node': start_node['id']}))
                
            elif event_type == 'PROCESS_NODE':
                token_id = event_data['token']
                node_id = event_data['node']
                node_info = self.process_graph['nodes'][node_id]
                
                if node_info['type'] == 'activity':
                    self.handle_activity(token_id, node_id)
                else:
                    next_nodes = self.handle_gateway(token_id, node_id)
                    for n in next_nodes:
                        heapq.heappush(self.event_queue, 
                            (self.current_time, 'PROCESS_NODE', {'token': token_id, 'node': n}))
            
            elif event_type == 'ACTIVITY_COMPLETE':
                token_id = event_data['token']
                node_id = event_data['node']
                node_info = self.process_graph['nodes'][node_id]
                
                # Release resources
                activity_name = node_info['name']
                required_resources = next(r for r in self.simulation_params['resources'] 
                                        if r['ResourceType'] == activity_name)
                for rt, qty in required_resources.items():
                    self.resources[rt]['available'] += qty
                
                # Log activity end
                self.log_event(token_id, node_id, 'ACTIVITY_END')
                
                # Move to next node
                outgoing = [t['to'] for t in self.process_graph['transitions'] if t['from'] == node_id]
                for next_node in outgoing:
                    heapq.heappush(self.event_queue,
                        (self.current_time, 'PROCESS_NODE', {'token': token_id, 'node': next_node}))

        # Write simulation log with durations
        with open('simulation_log.txt', 'w') as f:
            for entry in self.activity_log:
                log_line = f"{entry['timestamp']} - Token {entry['token']} - {entry.get('node', 'SYSTEM')} - {entry['event']}"
                
                if entry['details']:
                    if 'resources' in entry['details']:
                        res_str = ", ".join([f"{k} ({v})" for k,v in entry['details']['resources'].items()])
                        log_line += f" - Resources: {res_str}"
                    if 'duration' in entry['details']:
                        log_line += f" - Duration: {entry['details']['duration']:.2f} mins"
                
                f.write(log_line + "\n")

# Example usage
if __name__ == "__main__":
    simulator = SimulationEngine('./Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews-2.xpdl', 'simulation_parameters_alt.xlsx')
    simulator.run_simulation()
    print("Simulation complete. Results written to process_structure.json and simulation_log.txt")
