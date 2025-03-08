import networkx as nx
import logging
from typing import Dict, List, Any, Optional, Set, Tuple

class ProcessModel:
    """
    Represents a business process model with nodes (activities, gateways)
    and links between them.
    """
    
    def __init__(self):
        """Initialize an empty process model."""
        self.nodes = {}  # Dictionary of nodes by ID
        self.links = []  # List of links between nodes
        self.graph = nx.DiGraph()  # NetworkX directed graph for analysis
        
    def add_node(self, node_id: str, node_data: Dict[str, Any]) -> None:
        """
        Add a node to the process model.
        
        Args:
            node_id: Unique identifier for the node
            node_data: Dictionary of node attributes
        """
        self.nodes[node_id] = node_data
        self.graph.add_node(node_id, **node_data)
        
    def add_link(self, source: str, target: str, link_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a link between nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            link_data: Optional dictionary of link attributes
        """
        link = {
            "source": source,
            "target": target
        }
        
        if link_data:
            link.update(link_data)
            
        self.links.append(link)
        self.graph.add_edge(source, target, **link_data if link_data else {})
        
    def get_node(self, node_id: str) -> Dict[str, Any]:
        """
        Get a node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Dictionary of node attributes
        """
        return self.nodes.get(node_id, {})
        
    def get_start_nodes(self) -> List[str]:
        """
        Get all start nodes in the process.
        
        Returns:
            List of start node IDs
        """
        start_nodes = []
        
        for node_id, node_data in self.nodes.items():
            # Check for nodes explicitly marked as Start nodes
            if node_data.get("type") == "Start":
                start_nodes.append(node_id)
                
        # If no explicit start nodes, find nodes with no incoming edges
        if not start_nodes:
            for node_id in self.nodes:
                if self.graph.in_degree(node_id) == 0:
                    start_nodes.append(node_id)
                    
        logging.info(f"Found {len(start_nodes)} start nodes: {start_nodes}")
        return start_nodes
        
    def get_end_nodes(self) -> List[str]:
        """
        Get all end nodes in the process.
        
        Returns:
            List of end node IDs
        """
        end_nodes = []
        
        for node_id, node_data in self.nodes.items():
            # Check for nodes explicitly marked as End nodes
            if node_data.get("type") == "Stop":
                end_nodes.append(node_id)
                
        # If no explicit end nodes, find nodes with no outgoing edges
        if not end_nodes:
            for node_id in self.nodes:
                if self.graph.out_degree(node_id) == 0:
                    end_nodes.append(node_id)
                    
        return end_nodes
        
    def get_next_nodes(self, node_id: str) -> List[str]:
        """
        Get all nodes that follow this node.
        
        Args:
            node_id: Current node ID
            
        Returns:
            List of following node IDs
        """
        # Get all successors from the graph
        successors = list(self.graph.successors(node_id))
        
        # For debugging
        if not successors:
            logging.debug(f"Node {node_id} has no successors")
            
        return successors
        
    def get_all_paths(self) -> List[List[str]]:
        """
        Get all possible paths through the process.
        
        Returns:
            List of paths, where each path is a list of node IDs
        """
        start_nodes = self.get_start_nodes()
        end_nodes = self.get_end_nodes()
        
        all_paths = []
        
        for start in start_nodes:
            for end in end_nodes:
                try:
                    # Find all simple paths between start and end
                    paths = list(nx.all_simple_paths(self.graph, start, end))
                    all_paths.extend(paths)
                except nx.NetworkXNoPath:
                    # No path exists between this start and end
                    pass
                    
        return all_paths
        
    def get_all_resources(self) -> Dict[str, int]:
        """
        Get all resources defined in the process model with their counts.
        
        Returns:
            Dict mapping resource IDs to their available counts
        """
        resources = {}
        # Scan all nodes for resources
        for node_id, node_data in self.nodes.items():
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
        
    def find_critical_path(self) -> Tuple[List[str], float]:
        """
        Find the critical path through the process using average times.
        
        Returns:
            Tuple of (critical path as list of node IDs, total duration)
        """
        # Create weighted graph using avg time as edge weight
        weighted_graph = nx.DiGraph()
        
        # Add all nodes
        for node_id, node_data in self.nodes.items():
            weighted_graph.add_node(node_id, **node_data)
            
        # Add edges with negative weight (for longest path calculation)
        for source, target, edge_data in self.graph.edges(data=True):
            target_node = self.nodes.get(target, {})
            # Use average time as weight, default to 0
            weight = float(target_node.get("avg time", 0))
            weighted_graph.add_edge(source, target, weight=-weight)
            
        # Find critical path as the longest path
        start_nodes = self.get_start_nodes()
        end_nodes = self.get_end_nodes()
        
        critical_path = []
        max_duration = 0
        
        for start in start_nodes:
            for end in end_nodes:
                try:
                    # Find shortest path with negative weights (equivalent to longest path)
                    path = nx.shortest_path(weighted_graph, start, end, weight='weight')
                    
                    # Calculate path duration
                    duration = sum(
                        float(self.nodes.get(node, {}).get("avg time", 0))
                        for node in path
                    )
                    
                    if duration > max_duration:
                        max_duration = duration
                        critical_path = path
                        
                except nx.NetworkXNoPath:
                    # No path exists between this start and end
                    pass
                    
        return critical_path, max_duration
        
    def get_gateways(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all gateways in the process.
        
        Returns:
            Dictionary mapping gateway IDs to their data
        """
        gateways = {}
        
        for node_id, node_data in self.nodes.items():
            gateway_type = node_data.get("gateway")
            if gateway_type:
                gateways[node_id] = node_data
                
        return gateways
