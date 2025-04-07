import networkx as nx
import logging
import re
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
        self.gateway_merge_nodes = {}  # Maps gateway nodes to their merge nodes
        self.node_aliases = {}  # Maps normalized node names to original names
        
    def add_node(self, node_id: str, node_data: Dict[str, Any]) -> None:
        """
        Add a node to the process model.
        
        Args:
            node_id: Unique identifier for the node
            node_data: Dictionary of node attributes
        """
        self.nodes[node_id] = node_data
        self.graph.add_node(node_id, **node_data)
        
        # Store a normalized version of the node_id for easier lookup
        normalized_id = self._normalize_node_id(node_id)
        if normalized_id != node_id:
            if normalized_id not in self.node_aliases:
                self.node_aliases[normalized_id] = []
            if node_id not in self.node_aliases[normalized_id]:
                self.node_aliases[normalized_id].append(node_id)
        
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
        if node_id in self.nodes:
            return self.nodes.get(node_id, {})
        
        # Try to find by normalized name
        normalized_id = self._normalize_node_id(node_id)
        if normalized_id in self.node_aliases:
            for alias in self.node_aliases[normalized_id]:
                if alias in self.nodes:
                    return self.nodes.get(alias, {})
        
        return {}
        
    def get_start_nodes(self) -> List[str]:
        """
        Get all start nodes in the process.
        
        Returns:
            List of start node IDs
        """
        # Check if the graph is empty
        if len(self.graph.nodes()) == 0:
            logging.warning("Attempting to get start nodes from an empty graph")
            return []
            
        start_nodes = []
        
        for node_id, node_data in self.nodes.items():
            # Check for nodes explicitly marked as Start nodes
            if node_data.get("type") == "Start":
                start_nodes.append(node_id)
                
        # If no explicit start nodes, find nodes with no incoming edges
        if not start_nodes:
            for node_id in self.nodes:
                try:
                    if self.graph.in_degree(node_id) == 0:
                        start_nodes.append(node_id)
                except Exception as e:
                    logging.warning(f"Error checking in-degree for node {node_id}: {str(e)}")
                        
        logging.info(f"Found {len(start_nodes)} start nodes: {start_nodes}")
        return start_nodes
        
    def get_end_nodes(self) -> List[str]:
        """
        Get all end nodes in the process.
        
        Returns:
            List of end node IDs
        """
        # Check if the graph is empty
        if len(self.graph.nodes()) == 0:
            logging.warning("Attempting to get end nodes from an empty graph")
            return []
            
        end_nodes = []
        
        for node_id, node_data in self.nodes.items():
            # Check for nodes explicitly marked as End nodes
            if node_data.get("type") == "Stop":
                end_nodes.append(node_id)
                
        # If no explicit end nodes, find nodes with no outgoing edges
        if not end_nodes:
            for node_id in self.nodes:
                try:
                    if self.graph.out_degree(node_id) == 0:
                        end_nodes.append(node_id)
                except Exception as e:
                    logging.warning(f"Error checking out-degree for node {node_id}: {str(e)}")
                        
        return end_nodes
        
    def get_next_nodes(self, node_id: str) -> List[str]:
        """
        Get all nodes that follow this node.
        
        Args:
            node_id: Current node ID
            
        Returns:
            List of following node IDs
        """
        # Check if the graph is empty or if the node doesn't exist
        if len(self.graph.nodes()) == 0 or node_id not in self.graph:
            logging.warning(f"Attempting to get successors for non-existent node {node_id} or empty graph")
            return []
            
        # Get all successors from the graph
        try:
            successors = list(self.graph.successors(node_id))
            
            # Debug log the successors
            if not successors:
                logging.debug(f"Node {node_id} has no successors")
            else:
                logging.debug(f"Node {node_id} successors: {successors}")
                
            return successors
        except Exception as e:
            logging.error(f"Error getting successors for node {node_id}: {str(e)}")
            return []
        
        
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
                except Exception as e:
                    logging.error(f"Error finding paths from {start} to {end}: {str(e)}")
                    
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
                except Exception as e:
                    logging.error(f"Error finding critical path from {start} to {end}: {str(e)}")
                    
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
        
    def find_merge_node(self, gateway_id: str) -> Optional[str]:
        """
        Find the merge node for a splitting gateway.
        Uses network analysis to find the common point where all paths converge.
        
        Args:
            gateway_id: ID of the gateway node
            
        Returns:
            ID of the merge node if found, None otherwise
        """
        # Check if we already found the merge node for this gateway
        if gateway_id in self.gateway_merge_nodes:
            return self.gateway_merge_nodes[gateway_id]
        
        # Get the gateway node
        gateway_node = self.get_node(gateway_id)
        gateway_type = gateway_node.get("gateway")
        
        if not gateway_type:
            logging.debug(f"Node {gateway_id} is not a gateway")
            return None
        
        # Get all outgoing paths from the gateway
        next_nodes = self.get_next_nodes(gateway_id)
        
        if not next_nodes:
            logging.debug(f"Gateway {gateway_id} has no outgoing paths")
            return None
        
        # For each outgoing path, find all reachable nodes
        # and potential merge candidates
        path_reachable_nodes = []
        merge_candidates = set()
        first_path = True
        
        for next_node in next_nodes:
            reachable = set()
            visited = set()
            self._collect_reachable_nodes(next_node, reachable, visited)
            
            # Add to reachable nodes for this path
            path_reachable_nodes.append(reachable)
            
            # Find potential merge candidates - nodes that could be merge points
            # These are nodes with the same gateway type or any node with multiple incoming edges
            for node_id in reachable:
                node = self.get_node(node_id)
                
                # Skip the start node and other gateways of different types
                if node_id == gateway_id:
                    continue
                    
                # Check if this is a merge-like node (same gateway type or multiple incoming edges)
                node_gateway_type = node.get("gateway")
                in_degree = self.graph.in_degree(node_id)
                
                is_merge_candidate = False
                
                # Same gateway type could be a merge node
                if node_gateway_type and node_gateway_type == gateway_type:
                    is_merge_candidate = True
                
                # Nodes with multiple incoming edges could also be merge points
                elif in_degree > 1:
                    is_merge_candidate = True
                
                if is_merge_candidate:
                    if first_path:
                        merge_candidates.add(node_id)
                    else:
                        # After first path, only keep candidates reachable from all paths
                        if node_id in merge_candidates:
                            merge_candidates.add(node_id)
            
            first_path = False
        
        # Find nodes that are reachable from all paths (common to all paths)
        common_nodes = set.intersection(*path_reachable_nodes) if path_reachable_nodes else set()
        
        # Remove the original gateway node
        if gateway_id in common_nodes:
            common_nodes.remove(gateway_id)
        
        # Find the earliest common node from each path that has the same gateway type
        best_merge_node = None
        min_distance = float('inf')
        
        # First priority: Find matching gateway type that's a merge candidate
        for node_id in common_nodes:
            if node_id in merge_candidates:
                node = self.get_node(node_id)
                node_gateway_type = node.get("gateway")
                
                # Check if this is a merge node of matching type
                if node_gateway_type and node_gateway_type == gateway_type:
                    # Calculate average distance from gateway to this node
                    total_distance = 0
                    for next_node in next_nodes:
                        try:
                            # Find shortest path length
                            distance = len(nx.shortest_path(self.graph, next_node, node_id)) - 1
                            total_distance += distance
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            # No path exists or node not found
                            total_distance += float('inf')
                            
                    avg_distance = total_distance / len(next_nodes) if len(next_nodes) > 0 else float('inf')
                    
                    if avg_distance < min_distance:
                        min_distance = avg_distance
                        best_merge_node = node_id
        
        # Second priority: Any node with multiple incoming edges
        if best_merge_node is None:
            for node_id in common_nodes:
                if node_id in merge_candidates:
                    # Calculate average distance from gateway to this node
                    total_distance = 0
                    for next_node in next_nodes:
                        try:
                            # Find shortest path length
                            distance = len(nx.shortest_path(self.graph, next_node, node_id)) - 1
                            total_distance += distance
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            # No path exists or node not found
                            total_distance += float('inf')
                    
                    avg_distance = total_distance / len(next_nodes) if len(next_nodes) > 0 else float('inf')
                    
                    if avg_distance < min_distance:
                        min_distance = avg_distance
                        best_merge_node = node_id
        
        # Third priority: Just take the first common node (better than nothing)
        if best_merge_node is None and common_nodes:
            best_merge_node = list(common_nodes)[0]
        
        # Save the result for future reference
        if best_merge_node:
            self.gateway_merge_nodes[gateway_id] = best_merge_node
            logging.info(f"Found merge node {best_merge_node} for gateway {gateway_id}")
        else:
            logging.warning(f"No merge node found for gateway {gateway_id}")
        
        return best_merge_node
    
    def _collect_reachable_nodes(self, start_node: str, reachable: Set[str], visited: Optional[Set[str]] = None) -> None:
        """
        Collect all nodes reachable from a start node.
        
        Args:
            start_node: Starting node ID
            reachable: Set to populate with reachable nodes
            visited: Set of already visited nodes
        """
        if visited is None:
            visited = set()
        
        if start_node in visited:
            return
        
        visited.add(start_node)
        reachable.add(start_node)
        
        next_nodes = self.get_next_nodes(start_node)
        for next_node in next_nodes:
            self._collect_reachable_nodes(next_node, reachable, visited)
            
    def _normalize_node_id(self, node_id: str) -> str:
        """
        Normalize a node ID by removing type annotations and whitespace.
        
        Args:
            node_id: Node ID to normalize
            
        Returns:
            Normalized node ID
        """
        # Remove type annotations like [Type: Start] or [Exclusive Gateway]
        if node_id:
            # First try to match and remove annotations in square brackets
            normalized = re.sub(r'\s*\[.*?\]', '', node_id).strip()
            
            # Also remove any leading/trailing whitespace
            normalized = normalized.strip()
            
            # If the normalized result is empty, return the original input
            if not normalized:
                return node_id
                
            return normalized
        
        return node_id

    def validate_process_graph(self) -> Dict[str, Any]:
        """
        Comprehensive validation of process graph connectivity.
        
        Returns:
            Dictionary with graph connectivity details
        """
        validation_results = {
            'is_connected': nx.is_weakly_connected(self.graph),
            'start_nodes': self.get_start_nodes(),
            'end_nodes': self.get_end_nodes(),
            'total_nodes': len(self.graph.nodes()),
            'total_edges': len(self.graph.edges()),
            'disconnected_components': list(nx.weakly_connected_components(self.graph))
        }
        
        # Detailed edge analysis
        edge_details = []
        for source, target, data in self.graph.edges(data=True):
            edge_details.append({
                'source': source,
                'target': target,
                'data': data
            })
        
        validation_results['edge_details'] = edge_details
        
        return validation_results