import re
import json
import logging
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
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def get_all_resources(self) -> Dict[str, int]:
        """
        Get all resources defined in the process model with their counts.
        
        Returns:
            Dict mapping resource IDs to their available counts
        """
        resources = {}
        # Scan all nodes for resources
        for node_id, node_data in self.process_model.nodes(data=True):
            resource = node_data.get('resource')
            if resource:
                # Get count from node data if available, otherwise use default 1
                count = int(node_data.get('available resources', 1))
                # Update the resource count (use max if resource appears multiple times)
                if resource in resources:
                    resources[resource] = max(resources[resource], count)
                else:
                    resources[resource] = count
                
        logging.info(f"Found {len(resources)} resources in process model: {resources}")
        return resources

    def build_from_sequences(self, sequence_file_path: str, simulation_metrics: pd.DataFrame) -> nx.DiGraph:
        """
        Build a process model from a sequence file and simulation metrics.
        
        Args:
            sequence_file_path: Path to the sequence file
            simulation_metrics: DataFrame containing simulation metrics
            
        Returns:
            The constructed process model graph
        """
        logging.info(f"Building process model from sequence file: {sequence_file_path}")
        
        # Normalize column names in simulation_metrics for consistency
        simulation_metrics.columns = [str(col).lower() for col in simulation_metrics.columns]
        
        # Parse the sequence file to extract transition information and gateway types
        transitions = []
        gateway_connections = {}  # Track which activities are connected to which gateways
        
        with open(sequence_file_path, "r") as file:
            for line in file:
                line = line.strip()
                if not line or "->" not in line:
                    continue
                
                # Extract source and target activity strings
                source, target = line.split("->")
                source = source.strip()
                target = target.strip()
                
                # Extract base names (before any '[' character)
                source_name = re.split(r"\[", source)[0].strip()
                target_name = re.split(r"\[", target)[0].strip()
                
                # Determine gateway types
                source_gateway = next((gw for gw in self.gateway_types if gw in source), None)
                target_gateway = next((gw for gw in self.gateway_types if gw in target), None)
                
                # Store the gateway connections
                if source_gateway:
                    if source_gateway not in gateway_connections:
                        gateway_connections[source_gateway] = {}
                    gateway_connections[source_gateway][source_name] = []
                
                # If source is a gateway, track its targets
                if source_gateway:
                    if source_name not in gateway_connections[source_gateway]:
                        gateway_connections[source_gateway][source_name] = []
                    gateway_connections[source_gateway][source_name].append(target_name)
                
                # Determine edge type
                target_type_match = re.search(r"\[Type: (.*?)\]", target)
                edge_type = target_type_match.group(1) if target_type_match else "Activity Step"
                
                transitions.append({
                    "source": source_name,
                    "target": target_name,
                    "source_full": source,
                    "target_full": target,
                    "source_gateway": source_gateway,
                    "target_gateway": target_gateway,
                    "edge_type": edge_type
                })
        
        logging.info(f"Extracted {len(transitions)} transitions from sequence file")
        logging.info(f"Gateway connections: {gateway_connections}")
        
        # First pass: Add all nodes to the graph
        for transition in transitions:
            source_name = transition["source"]
            target_name = transition["target"]
            source_full = transition["source_full"]
            target_full = transition["target_full"]
            source_gateway = transition["source_gateway"]
            target_gateway = transition["target_gateway"]
            
            # Add source node if not already in graph
            if source_name not in self.process_model:
                self._add_node_from_string(source_full, simulation_metrics)
                
            # Add target node if not already in graph
            if target_name not in self.process_model:
                self._add_node_from_string(target_full, simulation_metrics)
        
        # Second pass: Update gateway associations for connected nodes
        for gateway_type, gateways in gateway_connections.items():
            logging.info(f"Processing connections for gateway type: {gateway_type}")
            
            for gateway_name, target_nodes in gateways.items():
                logging.info(f"  Gateway {gateway_name} connects to: {target_nodes}")
                
                # If this is an inclusive or parallel gateway
                if gateway_type in ["[Inclusive Gateway]", "[Parallel Gateway]"]:
                    # Update all connected target nodes to have the same gateway type
                    for target_name in target_nodes:
                        if target_name in self.process_model:
                            logging.info(f"    Setting gateway type for {target_name} to {gateway_type}")
                            self.process_model.nodes[target_name]["gateway"] = gateway_type
        
        # Third pass: Add all edges
        for transition in transitions:
            source_name = transition["source"]
            target_name = transition["target"]
            edge_type = transition["edge_type"]
            
            # Add edge to the graph
            self.process_model.add_edge(source_name, target_name, type=edge_type)
            
        logging.info(f"Process model built with {len(self.process_model.nodes)} nodes and {len(self.process_model.edges)} edges")
        return self.process_model
        
    def _add_node_from_string(self, node_string: str, simulation_metrics: pd.DataFrame) -> None:
        """
        Parse a node string and add it to the graph with attributes.
        
        Args:
            node_string: String representation of a node
            simulation_metrics: DataFrame containing simulation metrics
        """
        # Extract node name (before any '[' character)
        node_name = re.split(r"\[", node_string)[0].strip()
        
        # Skip if already added
        if node_name in self.process_model:
            return
            
        # Extract node type
        node_type_match = re.search(r"\[Type: (.*?)\]", node_string)
        node_type = node_type_match.group(1) if node_type_match else "Activity Step"
        
        # Check for gateway type
        gateway_match = next((gw for gw in self.gateway_types if gw in node_string), None)
        
        # If a gateway is inclusive or parallel, then set the type as the same as the gateway
        if gateway_match in ["[Inclusive Gateway]", "[Parallel Gateway]"]:
            node_type = gateway_match
            
        # Get attributes from simulation metrics
        attributes = {}
        # Look for the node in simulation metrics (case-insensitive matching)
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
            gateway=gateway_match,
            **attributes
        )
        
        logging.info(f"Added node: {node_name}, type: {node_type}, gateway: {gateway_match}")
        
    def save_to_json(self, output_path: str = None) -> str:
        """
        Save the process model to a JSON file.
        
        Args:
            output_path: Path to save the JSON file. If None, a timestamped filename is used.
            
        Returns:
            Path to the saved JSON file
        """
        import datetime
        
        # Create timestamped filename if output_path not provided
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"process_model_{timestamp}.json"
        
        process_model_data = json_graph.node_link_data(self.process_model)
        
        with open(output_path, "w") as json_file:
            json.dump(process_model_data, json_file, indent=4)
            
        logging.info(f"Process model saved to: {output_path}")
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
