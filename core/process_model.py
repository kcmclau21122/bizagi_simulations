import json
import networkx as nx
from networkx.readwrite import json_graph
from typing import Dict, List, Any, Optional, Set, Tuple
import random
import logging

class ProcessModel:
    """
    Represents a business process model with nodes and links.
    Provides methods to navigate and manipulate the process structure.
    """
    
    def __init__(self):
        """Initialize an empty process model."""
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.links: List[Dict[str, Any]] = []
        
    @classmethod
    def from_json(cls, json_file_path: str) -> 'ProcessModel':
        """Load a process model from a JSON file."""
        model = cls()
        
        with open(json_file_path, 'r') as file:
            process_model_data = json.load(file)
            
        # Load nodes
        for node_data in process_model_data.get('nodes', []):
            node_id = node_data.get('id')
            if node_id:
                model.nodes[node_id] = node_data
                model.graph.add_node(node_id, **node_data)
        
        # Load links
        for link_data in process_model_data.get('links', []):
            source = link_data.get('source')
            target = link_data.get('target')
            if source and target:
                model.links.append(link_data)
                model.graph.add_edge(source, target, **link_data)
                
        return model
    
    def add_node(self, node_id: str, **attributes) -> None:
        """Add a node to the process model."""
        self.nodes[node_id] = attributes
        self.graph.add_node(node_id, **attributes)
        
    def add_link(self, source_id: str, target_id: str, **attributes) -> None:
        """Add a link between nodes in the process model."""
        link_data = {
            'source': source_id,
            'target': target_id,
            **attributes
        }
        self.links.append(link_data)
        self.graph.add_edge(source_id, target_id, **attributes)
        
    def get_node(self, node_id: str) -> Dict[str, Any]:
        """Get node data by ID."""
        return self.nodes.get(node_id, {})
        
    def get_outgoing_links(self, node_id: str) -> List[Dict[str, Any]]:
        """Get outgoing links from a node."""
        return [link for link in self.links if link.get('source') == node_id]
        
    def get_incoming_links(self, node_id: str) -> List[Dict[str, Any]]:
        """Get incoming links to a node."""
        return [link for link in self.links if link.get('target') == node_id]
        
    def get_start_nodes(self) -> List[str]:
        """Get all start nodes in the process."""
        return [node_id for node_id, data in self.nodes.items() 
                if data.get('type') == 'Start']
                
    def get_end_nodes(self) -> List[str]:
        """Get all end nodes in the process."""
        return [node_id for node_id, data in self.nodes.items() 
                if data.get('type') == 'Stop']
    
    def get_next_nodes(self, node_id: str) -> List[str]:
        """
        Determine the next node(s) based on the gateway type and probabilities.
        Uses a strategy similar to the choose_node function but as a method.
        """
        node_data = self.get_node(node_id)
        gateway = node_data.get("gateway")
        node_type = node_data.get("type")
        
        # Get outgoing links
        outgoing_links = self.get_outgoing_links(node_id)
        
        if gateway == "[Parallel Gateway]":
            # Return all target nodes for parallel gateway
            return [link.get('target') for link in outgoing_links]
            
        elif gateway == "[Exclusive Gateway]" and "CONDITION-" not in str(node_type):
            # Handle exclusive gateway with probabilistic selection
            condition_targets = {}
            for link in outgoing_links:
                link_type = link.get('type', '')
                if "CONDITION-" in str(link_type):
                    condition = link_type.split("CONDITION-")[1].strip()
                    condition_targets[condition.lower()] = link.get('target')
            
            if condition_targets:
                # Get probabilities
                probabilities = {}
                for condition, target in condition_targets.items():
                    probability = node_data.get(condition, 0.5)
                    probabilities[condition] = probability
                
                # Normalize probabilities
                total = sum(probabilities.values())
                if total > 0:
                    for condition in probabilities:
                        probabilities[condition] /= total
                
                # Choose based on probabilities
                conditions = list(probabilities.keys())
                weights = list(probabilities.values())
                chosen_condition = random.choices(conditions, weights=weights, k=1)[0]
                
                return [condition_targets[chosen_condition]]
            
            # If no conditions, choose randomly
            if outgoing_links:
                return [random.choice(outgoing_links).get('target')]
                
        elif gateway == "[Inclusive Gateway]":
            # Handle inclusive gateway
            targets = []
            for link in outgoing_links:
                link_type = link.get('type', '')
                if "CONDITION-" in str(link_type):
                    condition = link_type.split("CONDITION-")[1].strip()
                    probability = node_data.get(condition.lower(), 0.5)
                    
                    if random.random() <= probability:
                        targets.append(link.get('target'))
            
            # If no targets selected, pick one randomly as fallback
            if not targets and outgoing_links:
                targets = [random.choice(outgoing_links).get('target')]
                
            return targets
        
        # Default: return all targets (normal flow)
        return [link.get('target') for link in outgoing_links]
        
    def to_json(self, output_path: str) -> str:
        """Save the process model to a JSON file."""
        process_model_data = json_graph.node_link_data(self.graph, edges="links")
        
        with open(output_path, "w") as json_file:
            json.dump(process_model_data, json_file, indent=4)
            
        return output_path
        
    def get_all_resources(self) -> Dict[str, int]:
        """
        Get all resources used in the process model with their available counts.
        Returns a dictionary mapping resource names to available counts.
        """
        resources = {}
        for node_data in self.nodes.values():
            resource = node_data.get("resource")
            if resource:
                resources[resource] = max(resources.get(resource, 0), 
                                         int(node_data.get("available resources", 1)))
        return resources
