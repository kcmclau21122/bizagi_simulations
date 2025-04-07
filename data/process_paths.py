# process_paths.py - Module for generating and analyzing process paths
# ------------------------------------------------------------

import os
import networkx as nx
import logging
from typing import List, Dict, Set, Tuple, Optional
import xml.etree.ElementTree as ET

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
        
        # Track if we found any valid sequences
        found_sequences = False
        
        # Dictionary to track gateway types
        gateway_types = {}
        
        # Read the sequence file
        with open(self.sequence_file_path, 'r') as file:
            lines = file.readlines()
            
        # First pass: collect all nodes and their types
        for i, line in enumerate(lines):
            line = line.strip()
            if '->' in line:
                found_sequences = True
                parts = line.split('->')
                source = parts[0].strip()
                target = parts[1].strip()
                
                # Extract node names (before any '[' character if present)
                source_name = source.split('[')[0].strip() if '[' in source else source
                target_name = target.split('[')[0].strip() if '[' in target else target
                
                # Skip empty node names
                if not source_name or not target_name:
                    continue
                
                # Default node types
                source_type = "Unknown"
                target_type = "Unknown"
                
                # Look for gateway indicators
                if "[Inclusive Gateway]" in source:
                    source_type = "Inclusive Gateway"
                    gateway_types[source_name] = "Inclusive"
                elif "[Parallel Gateway]" in source:
                    source_type = "Parallel Gateway"
                    gateway_types[source_name] = "Parallel"
                elif "[Exclusive Gateway]" in source:
                    source_type = "Exclusive Gateway"
                    gateway_types[source_name] = "Exclusive"
                elif "[Gateway]" in source:
                    source_type = "Gateway"
                    gateway_types[source_name] = "Generic"
                elif "Type: Start" in source:
                    source_type = "Start"
                elif "Type: Stop" in source:
                    source_type = "Stop"
                    
                if "[Inclusive Gateway]" in target:
                    target_type = "Inclusive Gateway"
                    gateway_types[target_name] = "Inclusive"
                elif "[Parallel Gateway]" in target:
                    target_type = "Parallel Gateway"
                    gateway_types[target_name] = "Parallel"
                elif "[Exclusive Gateway]" in target:
                    target_type = "Exclusive Gateway"
                    gateway_types[target_name] = "Exclusive"
                elif "[Gateway]" in target:
                    target_type = "Gateway"
                    gateway_types[target_name] = "Generic"
                elif "Type: Start" in target:
                    target_type = "Start"
                elif "Type: Stop" in target:
                    target_type = "Stop"
                
                # Add nodes to the graph with type information
                if source_name not in self.process_model:
                    self.process_model.add_node(source_name, type=source_type, full_desc=source)
                if target_name not in self.process_model:
                    self.process_model.add_node(target_name, type=target_type, full_desc=target)
        
        # Second pass: add edges with conditions
        edge_types = {}
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if '->' in line:
                parts = line.split('->')
                source = parts[0].strip()
                target = parts[1].strip()
                
                # Extract node names
                source_name = source.split('[')[0].strip() if '[' in source else source
                target_name = target.split('[')[0].strip() if '[' in target else target
                
                # Skip empty node names
                if not source_name or not target_name:
                    i += 1
                    continue
                
                # Check for condition in the next line
                edge_type = "NORMAL"
                if i + 1 < len(lines) and lines[i+1].strip().startswith("Type:"):
                    condition_line = lines[i+1].strip()
                    edge_type = condition_line.replace("Type:", "").strip()
                    i += 1  # Skip the next line since we've processed it
                    
                # Store edge type for reference
                edge_key = (source_name, target_name)
                edge_types[edge_key] = edge_type
                
                # Add edge to the graph with condition information
                self.process_model.add_edge(
                    source_name, 
                    target_name, 
                    type=edge_type,
                    condition=edge_type if "CONDITION-" in edge_type else None
                )
                
                # Log the edge added
                logging.debug(f"Added edge: {source_name} -> {target_name} [{edge_type}]")
                
                i += 1
            else:
                i += 1
        
        # Check if no sequences were found (fall back to default model creation)
        if not found_sequences:
            logging.warning("No valid sequences found in the sequence file. Using fallback method.")
            self._add_fallback_nodes()
        else:
            # Ensure gateway edges have proper types for correct path analysis
            for node, node_type in gateway_types.items():
                # Update outgoing edges from gateways with the gateway type
                for _, target in self.process_model.out_edges(node):
                    edge_data = self.process_model.edges[node, target]
                    # Mark edges from inclusive or parallel gateways
                    if node_type in ["Inclusive", "Parallel"]:
                        # Only update if not already a condition
                        if not edge_data.get('condition'):
                            edge_data['type'] = f"GATEWAY-{node_type.upper()}"
                            
            # Log summary of the built model
            logging.info(f"Process model built with {len(self.process_model.nodes)} nodes "
                        f"and {len(self.process_model.edges)} edges")
            
            # Log the gateways found for debugging
            if gateway_types:
                logging.info(f"Found gateways: {gateway_types}")
        
    def _add_fallback_nodes(self) -> None:
        """Add fallback nodes if no sequences were found in the file."""
        # Extract the base XPDL filename to use as process name
        base_filename = os.path.splitext(os.path.basename(self.xpdl_file_path))[0]
        
        # Create a minimal process model with start and end nodes
        start_node = f"{base_filename} Start"
        end_node = f"{base_filename} End"
        
        # Add nodes
        self.process_model.add_node(start_node, type="Start", full_desc=f"{start_node} [Type: Start]")
        self.process_model.add_node(end_node, type="Stop", full_desc=f"{end_node} [Type: Stop]")
        
        # Add edge between them
        self.process_model.add_edge(start_node, end_node, type="NORMAL")
        
        logging.warning(f"Created fallback process model with {len(self.process_model.nodes)} nodes "
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
        
        # Check if the model has any nodes - if not, return an empty list
        if len(self.process_model.nodes) == 0:
            logging.warning("Cannot identify paths: Process model has no nodes.")
            return []
            
        # Get start and end nodes
        start_nodes = [
            node for node, data in self.process_model.nodes(data=True)
            if data.get('type', '') == 'Start'
        ]
        
        # If no explicit start nodes, consider nodes with no incoming edges as start nodes
        if not start_nodes:
            start_nodes = [n for n in self.process_model.nodes() 
                        if self.process_model.in_degree(n) == 0]
            logging.info(f"No explicit Start nodes found. Using nodes with no incoming edges: {start_nodes}")
        
        end_nodes = [
            node for node, data in self.process_model.nodes(data=True)
            if data.get('type', '') == 'Stop'
        ]
        
        # If no explicit end nodes, consider nodes with no outgoing edges as end nodes
        if not end_nodes:
            end_nodes = [n for n in self.process_model.nodes() 
                        if self.process_model.out_degree(n) == 0]
            logging.info(f"No explicit Stop nodes found. Using nodes with no outgoing edges: {end_nodes}")
        
        if not start_nodes or not end_nodes:
            logging.warning(f"Cannot identify paths: Missing start or end nodes. Start nodes: {start_nodes}, End nodes: {end_nodes}")
            
            # Fallback: If we have nodes but couldn't identify start/end, create simple paths
            if len(self.process_model.nodes) > 0:
                # Take any node as start and any other as end
                all_nodes = list(self.process_model.nodes())
                start_nodes = [all_nodes[0]]
                if len(all_nodes) > 1:
                    end_nodes = [all_nodes[-1]]
                else:
                    # Only one node - it's both start and end
                    end_nodes = start_nodes
                logging.warning(f"Using fallback start node: {start_nodes[0]} and end node: {end_nodes[0]}")
            else:
                return []
            
        all_paths = []
        paths_count = 0
        
        # Use a depth-first search approach to find all paths
        for start_node in start_nodes:
            for end_node in end_nodes:
                if start_node == end_node:
                    # Special case: start and end are the same node
                    all_paths.append([start_node])
                    paths_count += 1
                    continue
                    
                try:
                    # Try to find paths with networkx
                    try:
                        # Version 2.4+ supports the cutoff parameter (for very large models)
                        for path in nx.all_simple_paths(
                            self.process_model, start_node, end_node, cutoff=30
                        ):
                            all_paths.append(path)
                            paths_count += 1
                            
                            if paths_count >= max_paths:
                                logging.warning(f"Max paths limit reached ({max_paths}). "
                                            f"There may be more paths not shown.")
                                return all_paths
                    except (TypeError, AttributeError):
                        # Older versions don't support cutoff or different interface
                        for path in nx.all_simple_paths(self.process_model, start_node, end_node):
                            all_paths.append(path)
                            paths_count += 1
                            
                            if paths_count >= max_paths:
                                logging.warning(f"Max paths limit reached ({max_paths}). "
                                            f"There may be more paths not shown.")
                                return all_paths
                except nx.NetworkXNoPath:
                    logging.warning(f"No path found from {start_node} to {end_node}")
                except Exception as e:
                    logging.error(f"Error finding paths from {start_node} to {end_node}: {str(e)}")
        
        # If no paths were found but we have nodes, create a single default path
        if not all_paths and start_nodes and end_nodes:
            logging.warning("No paths found through networkx. Creating a single default path.")
            all_paths.append([start_nodes[0], end_nodes[0]])
        
        logging.info(f"Identified {len(all_paths)} paths through the process model")
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
    try:
        # First, parse the XPDL file directly to extract sequence information
        base_name = os.path.splitext(os.path.basename(xpdl_file_path))[0]
        sequence_file_path = f"{base_name}_sequences.txt"
        
        # Parse XPDL to sequences with detailed debugging
        logging.info(f"Parsing XPDL file: {xpdl_file_path}")
        parse_xpdl_to_sequences(xpdl_file_path, sequence_file_path)
        
        # Check if the sequence file was created and has content
        if not os.path.exists(sequence_file_path) or os.path.getsize(sequence_file_path) == 0:
            logging.warning(f"Sequence file {sequence_file_path} is empty or not created")
            # Create a minimal sequence file as fallback
            with open(sequence_file_path, 'w') as f:
                f.write(f"PROCESS SEQUENCES FROM {os.path.basename(xpdl_file_path)}\n")
                f.write("===============================================\n\n")
                f.write(f"Start [Type: Start] -> End [Type: Stop]\n")
            logging.info(f"Created fallback sequence file: {sequence_file_path}")
        
        # Initialize the ProcessPathAnalyzer
        analyzer = ProcessPathAnalyzer(xpdl_file_path)
        analyzer.sequence_file_path = sequence_file_path
        
        # Build the process model from sequences
        logging.info(f"Building process model from sequences")
        analyzer.build_process_model()
        
        # Check if the process model has nodes and edges
        if len(analyzer.process_model.nodes) == 0:
            logging.warning("Process model has no nodes after building from sequences")
            # Add some diagnostic information to a fallback output file
            if not output_file_path:
                output_file_path = f"{base_name}_process_paths.txt"
            
            with open(output_file_path, 'w') as f:
                f.write(f"PROCESS PATHS ANALYSIS (FALLBACK)\n")
                f.write(f"===============================\n\n")
                f.write(f"XPDL File: {xpdl_file_path}\n")
                f.write(f"Total Paths: 0\n\n")
                f.write("ERROR: Process model could not be built correctly.\n")
                f.write("The XPDL parser could not extract valid sequences from the file.\n\n")
                
                # Try to identify the issue
                f.write("DIAGNOSTIC INFORMATION:\n")
                f.write("======================\n\n")
                
                # Check if sequence file exists and has content
                if os.path.exists(sequence_file_path):
                    with open(sequence_file_path, 'r') as seq_file:
                        seq_content = seq_file.read()
                    f.write(f"Sequence file content:\n")
                    f.write(f"---------------------\n")
                    f.write(seq_content)
                    f.write("\n\n")
                else:
                    f.write(f"Sequence file {sequence_file_path} does not exist.\n\n")
                
                # Try to parse the XPDL directly and report
                f.write("Attempting direct XPDL parsing for diagnostics:\n")
                try:
                    tree = ET.parse(xpdl_file_path)
                    root = tree.getroot()
                    f.write(f"XPDL root tag: {root.tag}\n")
                    
                    # Look for activities
                    activities = root.findall(".//Activity") or root.findall(".//*[@ActivityType]")
                    f.write(f"Activities found: {len(activities)}\n")
                    for activity in activities[:5]:  # Show first 5 only
                        f.write(f"  - ID: {activity.get('Id')}, Name: {activity.get('Name')}\n")
                    
                    # Look for transitions
                    transitions = root.findall(".//Transition") or root.findall(".//*[@TransitionType]")
                    f.write(f"Transitions found: {len(transitions)}\n")
                    for transition in transitions[:5]:  # Show first 5 only
                        f.write(f"  - From: {transition.get('From')}, To: {transition.get('To')}\n")
                except Exception as e:
                    f.write(f"Error in direct XPDL parsing: {str(e)}\n")
            
            # Return a placeholder summary
            summary = "ERROR: Process model could not be built correctly. No paths could be identified."
            return output_file_path, summary
        
        # Generate the paths file and summary
        file_path = analyzer.write_paths_to_file(output_file_path)
        summary = analyzer.analyze_paths()
        
        return file_path, summary
    except Exception as e:
        import traceback
        logging.error(f"Error in process path analysis: {str(e)}")
        logging.error(traceback.format_exc())
        
        # Create a minimal output file with error information
        if not output_file_path:
            base_name = os.path.splitext(os.path.basename(xpdl_file_path))[0]
            output_file_path = f"{base_name}_process_paths.txt"
        
        with open(output_file_path, 'w') as f:
            f.write(f"PROCESS PATHS ANALYSIS (ERROR)\n")
            f.write(f"============================\n\n")
            f.write(f"XPDL File: {xpdl_file_path}\n")
            f.write(f"Error: {str(e)}\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())
        
        summary = f"ERROR: Process path analysis failed: {str(e)}"
        return output_file_path, summary