# process_paths.py - Module for generating and analyzing process paths
# ------------------------------------------------------------

import os
import networkx as nx
import logging
from typing import List, Dict, Set, Tuple, Optional

from data.xpdl_parser import parse_xpdl_to_sequences

class ProcessPathAnalyzer:
    """
    Analyzes process paths from an XPDL file.
    Extracts and reports all possible paths through the process model.
    """
    
    def __init__(self, xpdl_file_path: str):
        """
        Initialize the process path analyzer.
        
        Args:
            xpdl_file_path: Path to the XPDL file to analyze
        """
        self.xpdl_file_path = xpdl_file_path
        self.process_model = None
        self.sequence_file_path = None
        
    def parse_xpdl(self) -> None:
        """Parse the XPDL file to extract sequences."""
        # Create a temporary file path for sequences
        base_name = os.path.splitext(os.path.basename(self.xpdl_file_path))[0]
        self.sequence_file_path = f"{base_name}_sequences.txt"
        
        # Parse XPDL to sequences
        parse_xpdl_to_sequences(self.xpdl_file_path, self.sequence_file_path)
        logging.info(f"XPDL parsed into sequences: {self.sequence_file_path}")
        
    def build_process_model(self) -> None:
        """Build the process model from the sequences."""
        if not self.sequence_file_path:
            self.parse_xpdl()
            
        # Create a simple graph from the sequence file
        self.process_model = nx.DiGraph()
        
        # Read the sequence file
        with open(self.sequence_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if '->' in line:
                    parts = line.split('->')
                    source = parts[0].strip()
                    target = parts[1].strip()
                    
                    # Extract node names (before any '[' character)
                    source_name = source.split('[')[0].strip()
                    target_name = target.split('[')[0].strip()
                    
                    # Add nodes with their full descriptions for later reference
                    if source_name not in self.process_model:
                        self.process_model.add_node(source_name, full_desc=source)
                    if target_name not in self.process_model:
                        self.process_model.add_node(target_name, full_desc=target)
                    
                    # Extract node types
                    source_type = "Unknown"
                    if "[Type:" in source:
                        source_type = source.split("[Type:")[1].split("]")[0].strip()
                    elif "Start" in source:
                        source_type = "Start"
                    
                    target_type = "Unknown"
                    if "[Type:" in target:
                        target_type = target.split("[Type:")[1].split("]")[0].strip()
                    elif "Stop" in target:
                        target_type = "Stop"
                    
                    # Set node types
                    self.process_model.nodes[source_name]['type'] = source_type
                    self.process_model.nodes[target_name]['type'] = target_type
                    
                    # Check for gateways
                    if "[Exclusive Gateway]" in source:
                        self.process_model.nodes[source_name]['gateway'] = "[Exclusive Gateway]"
                    elif "[Inclusive Gateway]" in source:
                        self.process_model.nodes[source_name]['gateway'] = "[Inclusive Gateway]"
                    elif "[Parallel Gateway]" in source:
                        self.process_model.nodes[source_name]['gateway'] = "[Parallel Gateway]"
                        
                    if "[Exclusive Gateway]" in target:
                        self.process_model.nodes[target_name]['gateway'] = "[Exclusive Gateway]"
                    elif "[Inclusive Gateway]" in target:
                        self.process_model.nodes[target_name]['gateway'] = "[Inclusive Gateway]"
                    elif "[Parallel Gateway]" in target:
                        self.process_model.nodes[target_name]['gateway'] = "[Parallel Gateway]"
                    
                    # Add edge
                    edge_type = "NORMAL"
                    if "CONDITION-" in source or "CONDITION-" in target:
                        # Extract condition if present
                        if "CONDITION-" in source:
                            condition = source.split("CONDITION-")[1].split("]")[0].strip()
                            edge_type = f"CONDITION-{condition}"
                        elif "CONDITION-" in target:
                            condition = target.split("CONDITION-")[1].split("]")[0].strip()
                            edge_type = f"CONDITION-{condition}"
                    
                    self.process_model.add_edge(source_name, target_name, type=edge_type)
        
        logging.info(f"Process model built with {len(self.process_model.nodes)} nodes "
                    f"and {len(self.process_model.edges)} edges")
        
    def identify_all_paths(self, max_paths: int = 1000) -> List[List[str]]:
        """
        Identify all possible paths through the process model.
        
        Args:
            max_paths: Maximum number of paths to find (to avoid combinatorial explosion)
            
        Returns:
            List of paths, where each path is a list of node names
        """
        if not self.process_model:
            self.build_process_model()
            
        # Get start and end nodes
        start_nodes = [
            node for node, data in self.process_model.nodes(data=True)
            if 'Start' in data.get('type', '')
        ]
        
        # If no explicit start nodes, consider nodes with no incoming edges as start nodes
        if not start_nodes:
            start_nodes = [n for n in self.process_model.nodes() 
                         if self.process_model.in_degree(n) == 0]
        
        end_nodes = [
            node for node, data in self.process_model.nodes(data=True)
            if 'Stop' in data.get('type', '')
        ]
        
        # If no explicit end nodes, consider nodes with no outgoing edges as end nodes
        if not end_nodes:
            end_nodes = [n for n in self.process_model.nodes() 
                        if self.process_model.out_degree(n) == 0]
        
        if not start_nodes or not end_nodes:
            logging.warning("No start or end nodes found in the process model")
            return []
            
        all_paths = []
        paths_count = 0
        
        # Use a depth-first search approach to find all paths
        for start_node in start_nodes:
            for end_node in end_nodes:
                try:
                    # Handle NetworkX version differences
                    try:
                        # Version 2.4+ supports the cutoff parameter
                        for path in nx.all_simple_paths(
                            self.process_model, start_node, end_node, cutoff=30
                        ):
                            all_paths.append(path)
                            paths_count += 1
                            
                            if paths_count >= max_paths:
                                logging.warning(f"Max paths limit reached ({max_paths}). "
                                              f"There may be more paths not shown.")
                                return all_paths
                    except TypeError:
                        # Older versions don't support cutoff
                        for path in nx.all_simple_paths(self.process_model, start_node, end_node):
                            all_paths.append(path)
                            paths_count += 1
                            
                            if paths_count >= max_paths:
                                logging.warning(f"Max paths limit reached ({max_paths}). "
                                              f"There may be more paths not shown.")
                                return all_paths
                except nx.NetworkXNoPath:
                    logging.info(f"No path found from {start_node} to {end_node}")
                    
        return all_paths
        
    def write_paths_to_file(self, output_file_path: str = None) -> str:
        """
        Write all possible paths to a text file.
        
        Args:
            output_file_path: Path to save the paths file (if None, uses default)
            
        Returns:
            Path to the saved file
        """
        paths = self.identify_all_paths()
        
        if not output_file_path:
            base_name = os.path.splitext(os.path.basename(self.xpdl_file_path))[0]
            output_file_path = f"{base_name}_process_paths.txt"
            
        with open(output_file_path, 'w') as file:
            file.write(f"PROCESS PATHS ANALYSIS\n")
            file.write(f"======================\n\n")
            file.write(f"XPDL File: {self.xpdl_file_path}\n")
            file.write(f"Total Paths: {len(paths)}\n\n")
            
            # Write information about gateways
            gateways = [
                node for node, data in self.process_model.nodes(data=True)
                if data.get('gateway')
            ]
            
            if gateways:
                file.write("Gateways in the process:\n")
                for gateway in gateways:
                    file.write(f"  - {gateway}\n")
                file.write("\n")
                
            # Write all paths
            file.write("ALL POSSIBLE PROCESS PATHS:\n")
            file.write("=========================\n\n")
            
            for i, path in enumerate(paths, 1):
                file.write(f"Path {i}:\n")
                for j, node in enumerate(path):
                    # Add an arrow except for the last node
                    arrow = " -> " if j < len(path) - 1 else ""
                    file.write(f"  {node}{arrow}")
                file.write("\n\n")
                
        logging.info(f"Process paths written to: {output_file_path}")
        return output_file_path
        
    def get_path_statistics(self) -> Dict:
        """
        Get statistics about the process paths.
        
        Returns:
            Dictionary with path statistics
        """
        paths = self.identify_all_paths()
        
        if not paths:
            return {
                "total_paths": 0,
                "min_length": 0,
                "max_length": 0,
                "avg_length": 0,
                "gateway_count": 0
            }
            
        # Calculate statistics
        path_lengths = [len(path) for path in paths]
        
        # Count gateways in the graph
        gateways = [
            node for node, data in self.process_model.nodes(data=True)
            if data.get('gateway')
        ]
        
        return {
            "total_paths": len(paths),
            "min_length": min(path_lengths),
            "max_length": max(path_lengths),
            "avg_length": sum(path_lengths) / len(path_lengths),
            "gateway_count": len(gateways)
        }
        
    def analyze_paths(self) -> str:
        """
        Analyze the process paths and generate a summary.
        
        Returns:
            Path analysis summary text
        """
        if not self.process_model:
            self.build_process_model()
            
        paths = self.identify_all_paths()
        stats = self.get_path_statistics()
        
        summary = "PROCESS PATH ANALYSIS SUMMARY\n"
        summary += "============================\n\n"
        summary += f"Total number of possible paths: {stats['total_paths']}\n"
        summary += f"Shortest path length: {stats['min_length']} steps\n"
        summary += f"Longest path length: {stats['max_length']} steps\n"
        summary += f"Average path length: {stats['avg_length']:.2f} steps\n\n"
        
        # Analyze gateway impact
        gateways = [
            node for node, data in self.process_model.nodes(data=True)
            if data.get('gateway')
        ]
        
        if gateways:
            summary += "Gateway Analysis:\n"
            summary += f"  Total gateways: {len(gateways)}\n"
            
            # Count paths passing through each gateway
            gateway_paths = {}
            for gateway in gateways:
                gateway_paths[gateway] = sum(1 for path in paths if gateway in path)
                
            # Sort gateways by path count (most impactful first)
            sorted_gateways = sorted(
                gateway_paths.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            summary += "  Most impactful gateways:\n"
            for gateway, path_count in sorted_gateways[:5]:  # Top 5 gateways
                impact_pct = (path_count / stats['total_paths']) * 100
                summary += f"    - {gateway}: affects {path_count} paths ({impact_pct:.1f}%)\n"
                
        return summary

# Function to run the path analysis and generate output file
def analyze_process_paths(xpdl_file_path: str, output_file_path: str = None) -> Tuple[str, str]:
    """
    Analyze the process paths from an XPDL file and generate a report.
    
    Args:
        xpdl_file_path: Path to the XPDL file
        output_file_path: Path to save the output file (if None, uses default)
        
    Returns:
        Tuple of (path to output file, analysis summary text)
    """
    analyzer = ProcessPathAnalyzer(xpdl_file_path)
    file_path = analyzer.write_paths_to_file(output_file_path)
    summary = analyzer.analyze_paths()
    
    return file_path, summary
