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
        self.node_aliases = {}  # Maps normalized node names to original names
        
    def build_from_json(self, json_file_path: str, simulation_metrics: pd.DataFrame = None) -> nx.DiGraph:
        """
        Build a process model from a JSON file.
        
        Args:
            json_file_path: Path to the JSON file containing process model
            simulation_metrics: DataFrame containing simulation metrics (optional)
            
        Returns:
            The constructed process model graph
        """
        # Load the JSON file
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
        except Exception as e:
            logging.error(f"Error loading JSON file {json_file_path}: {str(e)}")
            # Return an empty graph but at least initialize it
            self.process_model = nx.DiGraph()
            return self.process_model
        
        # Validate expected structure
        if 'nodes' not in data or 'links' not in data:
            logging.error(f"Invalid JSON structure in {json_file_path}: missing 'nodes' or 'links' key")
            # Return an empty graph but at least initialize it
            self.process_model = nx.DiGraph()
            return self.process_model
        
        # Create a new directed graph
        self.process_model = nx.DiGraph()
        
        # Add nodes from the JSON data
        node_count = 0
        for node in data.get('nodes', []):
            node_id = node.get('name', node.get('id', ''))
            if not node_id:
                continue
                
            # Add the node with all its attributes
            self.process_model.add_node(node_id, **node)
            node_count += 1
            
            # If simulation metrics are provided, merge them in
            if simulation_metrics is not None:
                self._add_metrics_to_node(node_id, simulation_metrics)
        
        # Add links from the JSON data
        edge_count = 0
        for link in data.get('links', []):
            source = link.get('source', '')
            target = link.get('target', '')
            
            if not source or not target:
                continue
                
            # Strip any type annotations if they exist in source/target
            source_name = re.split(r"\[", source)[0].strip() if '[' in source else source
            target_name = re.split(r"\[", target)[0].strip() if '[' in target else target
            
            # Add the edge with all its attributes
            self.process_model.add_edge(source_name, target_name, **link)
            edge_count += 1
        
        # Handle empty or invalid graphs
        if node_count == 0:
            logging.warning(f"No valid nodes found in JSON file {json_file_path}")
        if edge_count == 0:
            logging.warning(f"No valid edges found in JSON file {json_file_path}")
        
        logging.info(f"Built process model from JSON with {node_count} nodes and {edge_count} edges")
        return self.process_model

        
    def _add_metrics_to_node(self, node_id: str, simulation_metrics: pd.DataFrame) -> None:
        """
        Add simulation metrics to a node.
        
        Args:
            node_id: Node identifier
            simulation_metrics: DataFrame containing simulation metrics
        """
        # Normalize column names
        simulation_metrics.columns = [str(col).lower() for col in simulation_metrics.columns]
        
        # Find the matching row in metrics
        # First strip any type annotations from node_id if they exist
        simple_name = self._normalize_node_id(node_id)
        
        # Try exact match first
        attributes_row = simulation_metrics[simulation_metrics['name'].str.lower() == simple_name.lower()]
        
        # If no exact match, try partial matching
        if attributes_row.empty:
            # Try finding rows where the name is contained within the node_id
            for idx, row in simulation_metrics.iterrows():
                row_name = str(row.get('name', '')).lower()
                if row_name and (row_name in simple_name.lower() or simple_name.lower() in row_name):
                    attributes_row = simulation_metrics.iloc[[idx]]
                    break
        
        if not attributes_row.empty:
            # Get the attributes
            attributes = attributes_row.iloc[0].dropna().to_dict()
            
            # Remove redundant keys
            attributes.pop('type', None)
            attributes.pop('name', None)
            
            # Add the attributes to the node
            for key, value in attributes.items():
                self.process_model.nodes[node_id][key] = value
            
    def get_all_resources(self) -> Dict[str, int]:
        """
        Get all resources defined in the process model with their counts.
        
        Returns:
            Dict mapping resource IDs to their available counts
        """
        resources = {}
        # Scan all nodes for resources
        for node_id, node_data in self.process_model.nodes.items():  # Fixed to use process_model.nodes
            resource = node_data.get('resource')
            if resource:
                # Get count from node data if available, otherwise use default 1
                count = int(node_data.get('resource count', 1))
                # Update the resource count (use max if resource appears multiple times)
                if resource in resources:
                    resources[resource] = max(resources[resource], count)
                else:
                    resources[resource] = count
                
        # Debug log the resources found
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
            self.process_model.add_edge(source, target, type=edge_type)
            
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
        
        # Store normalized version in node_aliases
        norm_name = self._normalize_node_id(node_name)
        if norm_name != node_name:
            if norm_name not in self.node_aliases:
                self.node_aliases[norm_name] = []
            self.node_aliases[norm_name].append(node_name)
        
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
        
    def _normalize_node_id(self, node_id: str) -> str:
        """
        Normalize a node ID by removing type annotations and whitespace.
        
        Args:
            node_id: Node ID to normalize
            
        Returns:
            Normalized node ID
        """
        # Remove type annotations like [Type: Start]
        normalized = re.sub(r'\s*\[.*?\]', '', node_id).strip()
        return normalized
