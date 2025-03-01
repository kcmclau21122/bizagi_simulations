import re
import json
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph
from typing import Dict, List, Any, Optional, Set, Tuple

class ProcessModelBuilder:
    """
    Builds a process model graph from sequence files and simulation metrics.
    Responsible for parsing process paths and constructing a networkx graph.
    """
    
    def __init__(self):
        """Initialize the process model builder."""
        self.process_model = nx.DiGraph()
        self.gateway_types = ["[Exclusive Gateway]", "[Inclusive Gateway]", "[Parallel Gateway]"]
        
    def build_from_sequences(self, sequence_file_path: str, simulation_metrics: pd.DataFrame) -> nx.DiGraph:
        """
        Build a process model from a sequence file and simulation metrics.
        
        Args:
            sequence_file_path: Path to the sequence file
            simulation_metrics: DataFrame containing simulation metrics
            
        Returns:
            The constructed process model graph
        """
        # Normalize column names in simulation_metrics for consistency
        simulation_metrics.columns = map(str.lower, simulation_metrics.columns)
        
        # Parse the sequence file
        with open(sequence_file_path, "r") as file:
            lines = file.readlines()
            
        # Parse the lines to construct the graph
        for line in lines:
            line = line.strip()
            if not line or "->" not in line:
                continue
                
            # Extract source and target activity strings
            source, target = line.split("->")
            source = source.strip()
            target = target.strip()
            
            # Process source node
            self._add_node_from_string(source, simulation_metrics)
            
            # Process target node
            self._add_node_from_string(target, simulation_metrics)
            
            # Extract source and target IDs (names)
            source_name = re.split(r"\[", source)[0].strip()
            target_name = re.split(r"\[", target)[0].strip()
            
            # Determine edge type
            source_gateway = next((gw for gw in self.gateway_types if gw in source), None)
            if source_gateway in ["[Inclusive Gateway]", "[Parallel Gateway]"]:
                edge_type = source_gateway  # Use source gateway type for these cases
            else:
                # Check if target has a type
                target_type_match = re.search(r"\[Type: (.*?)\]", target)
                edge_type = target_type_match.group(1) if target_type_match else "Activity Step"
                
            # Add edge to the graph
            self.process_model.add_edge(source_name, target_name, type=edge_type)
            
        return self.process_model
        
    def _add_node_from_string(self, node_string: str, simulation_metrics: pd.DataFrame) -> None:
        """
        Parse a node string and add it to the graph with attributes.
        
        Args:
            node_string: String representation of a node
            simulation_metrics: DataFrame containing simulation metrics
        """
        # Extract node name
        node_name = re.split(r"\[", node_string)[0].strip()
        
        # Skip if already added
        if node_name in self.process_model:
            return
            
        # Extract node type
        node_type_match = re.search(r"\[Type: (.*?)\]", node_string)
        node_type = node_type_match.group(1) if node_type_match else "Activity Step"
        
        # Check for gateway
        node_gateway = next((gw for gw in self.gateway_types if gw in node_string), None)
        
        # If a gateway is inclusive or parallel, then set the type as the same as the gateway
        if node_gateway in ["[Inclusive Gateway]", "[Parallel Gateway]"]:
            node_type = node_gateway
            
        # Get attributes from simulation metrics
        attributes = {}
        attributes_row = simulation_metrics[simulation_metrics['name'].str.lower() == node_name.lower()]
        
        if not attributes_row.empty:
            attributes = attributes_row.iloc[0].dropna().to_dict()
            # Remove redundant keys
            attributes.pop('type', None)
            attributes.pop('name', None)
            
        # Add the node with its attributes
        self.process_model.add_node(
            node_name,
            type=node_type,
            gateway=node_gateway,
            **attributes
        )
        
    def save_to_json(self, output_path: str = "process_model.json") -> str:
        """
        Save the process model to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
            
        Returns:
            Path to the saved JSON file
        """
        process_model_data = json_graph.node_link_data(self.process_model, edges="links")
        
        with open(output_path, "w") as json_file:
            json.dump(process_model_data, json_file, indent=4)
            
        return output_path
        
    def get_start_nodes(self) -> List[str]:
        """
        Get start nodes from the process model.
        
        Returns:
            List of start node names
        """
        return [
            node for node, data in self.process_model.nodes(data=True)
            if data.get('type') == 'Start'
        ]
        
    def get_end_nodes(self) -> List[str]:
        """
        Get end nodes from the process model.
        
        Returns:
            List of end node names
        """
        return [
            node for node, data in self.process_model.nodes(data=True)
            if data.get('type') == 'Stop'
        ]
