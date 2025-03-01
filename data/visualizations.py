import json
import logging
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
from typing import Dict, Any, Optional

def diagram_process(json_file_path: str, output_path: str = "process_model_diagram.png") -> str:
    """
    Create a diagram of the process model and save it to a file.
    
    Args:
        json_file_path: Path to the process model JSON file
        output_path: Path to save the diagram
        
    Returns:
        Path to the saved diagram
    """
    # Load the JSON file
    with open(json_file_path, "r") as json_file:
        json_data = json.load(json_file)

    # Recreate the graph from the JSON data with explicit edges="links"
    process_model = json_graph.node_link_graph(json_data, edges="links")

    # Initialize node shapes
    node_shapes = {}
    for node in process_model.nodes:
        node_data = process_model.nodes[node]
        # Default shape is rectangle
        node_shape = "s"  # Matplotlib marker for square/rectangle

        # Check for gateway type of the node itself
        node_gateway = node_data.get("gateway")
        if node_gateway in ["[Parallel Gateway]", "[Inclusive Gateway]"]:
            node_shape = "D"  # Diamond shape for gateways

        # Check if this node is the source of a link with type "CONDITION-"
        for _, target, edge_data in process_model.out_edges(node, data=True):
            edge_type = edge_data.get("type", "")
            if isinstance(edge_type, str) and "CONDITION-" in edge_type:
                node_shape = "D"  # Diamond shape for sources of "CONDITION-" edges
                break

        # Store the shape for the node
        node_shapes[node] = node_shape

    # Use Agg backend for non-interactive plots
    plt.switch_backend('Agg')
    
    # Draw the graph with custom shapes
    plt.figure(figsize=(24, 24))  # Larger figure size for better spacing
    pos = nx.spring_layout(process_model, k=8.0, scale=3.0, iterations=500)  # Adjust spacing

    # Group nodes by shape
    grouped_nodes = {
        shape: [n for n in process_model if node_shapes.get(n) == shape]
        for shape in set(node_shapes.values())
    }
    
    # Draw nodes by shape group
    for shape, nodes in grouped_nodes.items():
        if nodes:  # Ensure there are nodes for the shape
            nx.draw_networkx_nodes(
                process_model,
                pos,
                nodelist=nodes,
                node_size=1000,
                node_shape=shape,
                node_color="lightblue"
            )

    # Draw edges with arrowheads
    nx.draw_networkx_edges(
        process_model,
        pos,
        edge_color="gray",
        arrows=True,
        connectionstyle="arc3,rad=0.1",  # Adds curvature for better visualization
        arrowstyle="-|>",  # Defines arrowhead style
        min_target_margin=15  # Space between node and arrowhead
    )

    # Add edge labels with CONDITION text
    edge_labels = {}
    for u, v, data in process_model.edges(data=True):
        edge_type = data.get('type', '')
        if isinstance(edge_type, str) and "CONDITION-" in edge_type:
            # Extract the text after "CONDITION-"
            label = edge_type.split("CONDITION-")[1]
            edge_labels[(u, v)] = label

    # Draw node labels
    nx.draw_networkx_labels(process_model, pos, font_size=8, font_weight="bold")
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(
        process_model,
        pos,
        edge_labels=edge_labels,  # Use the custom edge labels
        font_size=6
    )

    # Save the diagram as a file
    plt.title("Process Model Diagram", fontsize=8)
    plt.savefig(output_path)
    plt.close()

    return output_path

def create_process_summary_chart(process_model_path: str, output_path: str = "process_summary.png") -> str:
    """
    Create a summary chart of the process model showing key statistics.
    
    Args:
        process_model_path: Path to the process model JSON file
        output_path: Path to save the chart
        
    Returns:
        Path to the saved chart
    """
    # Load the process model
    with open(process_model_path, "r") as json_file:
        json_data = json.load(json_file)
        
    # Extract node types and count
    nodes = json_data.get('nodes', [])
    node_types = {}
    
    for node in nodes:
        node_type = node.get('type', 'Unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
        
    # Create a chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(node_types.keys(), node_types.values(), color='skyblue')
    
    # Add labels and title
    plt.xlabel('Node Type')
    plt.ylabel('Count')
    plt.title('Process Model Node Types')
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.1,
            str(int(height)),
            ha='center', 
            va='bottom'
        )
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def visualize_process_paths(process_model_path: str, output_path: str = "process_paths.png") -> str:
    """
    Visualize possible paths through the process model.
    
    Args:
        process_model_path: Path to the process model JSON file
        output_path: Path to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    # Load the process model
    with open(process_model_path, "r") as json_file:
        json_data = json.load(json_file)
        
    # Create graph from JSON
    graph = json_graph.node_link_graph(json_data, edges="links")
    
    # Find start and end nodes
    start_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'Start']
    end_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'Stop']
    
    # If no explicit end nodes, assume nodes with no outgoing edges are end nodes
    if not end_nodes:
        end_nodes = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        
    # Use a different layout for path visualization
    plt.figure(figsize=(12, 8))
    pos = nx.kamada_kawai_layout(graph)
    
    # Draw regular nodes
    regular_nodes = [n for n in graph.nodes() if n not in start_nodes and n not in end_nodes]
    nx.draw_networkx_nodes(
        graph, pos, 
        nodelist=regular_nodes,
        node_color='lightblue',
        node_size=500
    )
    
    # Draw start nodes
    nx.draw_networkx_nodes(
        graph, pos, 
        nodelist=start_nodes,
        node_color='green',
        node_size=700
    )
    
    # Draw end nodes
    nx.draw_networkx_nodes(
        graph, pos, 
        nodelist=end_nodes,
        node_color='red',
        node_size=700
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        graph, pos,
        edge_color='gray',
        arrows=True,
        arrowstyle='-|>',
        width=1.5
    )
    
    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=8)
    
    # Add legend
    plt.plot([], [], 'o', color='green', label='Start')
    plt.plot([], [], 'o', color='red', label='End')
    plt.plot([], [], 'o', color='lightblue', label='Activity')
    plt.legend()
    
    plt.title('Process Paths Visualization')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path
