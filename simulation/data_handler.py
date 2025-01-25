# simulation/data_handler.py
import pandas as pd
#from collections import defaultdict
#import logging
import json
import re
from networkx.readwrite import json_graph
import networkx as nx
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from networkx.readwrite import json_graph
import networkx as nx


def build_paths(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Initialize directed graph
    process_model = nx.DiGraph()

    # Define gateway types
    gateway_types = ["[Exclusive Gateway]", "[Inclusive Gateway]", "[Parallel Gateway]"]

    # Parse the lines to construct the graph
    for line in lines:
        line = line.strip()
        if not line or "->" not in line:
            continue

        # Extract source and target activity strings
        source, target = line.split("->")
        source = source.strip()
        target = target.strip()

        # Extract source attributes
        source_name = re.split(r"\[", source)[0].strip()
        source_type_match = re.search(r"\[Type: (.*?)\]", source)
        source_type = source_type_match.group(1) if source_type_match else None
        source_gateway = next((gw for gw in gateway_types if gw in source), None)

        # Extract target attributes
        target_name = re.split(r"\[", target)[0].strip()
        target_type_match = re.search(r"\[Type: (.*?)\]", target)
        target_type = target_type_match.group(1) if target_type_match else "Activity Step"
        target_gateway = next((gw for gw in gateway_types if gw in target), None)

        # If a getway is inclusive or parallel then set the type as the same as the gateway
        if target_gateway in ["[Inclusive Gateway]","[Parallel Gateway]"]:
            target_type = target_gateway


        if source_gateway in ["[Inclusive Gateway]","[Parallel Gateway]"]:
            source_type = source_gateway

        # Add source node with attributes
        if source_name not in process_model:
            process_model.add_node(
                source_name,
                type=source_type or "Activity Step",
                gateway=source_gateway
            )

        # Add target node with attributes
        if target_name not in process_model:
            process_model.add_node(
                target_name,
                type=target_type,
                gateway=target_gateway
            )

        # Determine edge type
        if source_gateway in ["[Inclusive Gateway]", "[Parallel Gateway]"]:
            edge_type = source_gateway  # Use source gateway type for these cases
        else:
            edge_type = target_type  # Default to target type otherwise

        # Add edge to the graph
        process_model.add_edge(source_name, target_name, type=edge_type)

    # Convert the graph to JSON
    process_model_data = json_graph.node_link_data(process_model)

    # Save the updated JSON file
    json_output_path = "process_model.json"
    with open(json_output_path, "w") as json_file:
        json.dump(process_model_data, json_file, indent=4)

    return json_output_path


def diagram_process(json_output_path):
    # Load the JSON file
    with open(json_output_path, "r") as json_file:
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
            if "CONDITION-" in edge_type:
                node_shape = "D"  # Diamond shape for sources of "CONDITION-" edges
                break

        # Store the shape for the node
        node_shapes[node] = node_shape

    # Draw the graph with custom shapes
    plt.figure(figsize=(24, 24))  # Larger figure size for better spacing
    pos = nx.spring_layout(process_model, k=8.0, scale=3.0, iterations=500)  # Adjust spacing

    # Group nodes by shape
    grouped_nodes = {
        shape: [n for n in process_model if node_shapes.get(n) == shape]
        for shape in set(node_shapes.values())
    }
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
        if "CONDITION-" in edge_type:
            # Extract the text after "CONDITION-"
            label = edge_type.split("CONDITION-")[1]
            edge_labels[(u, v)] = label

    nx.draw_networkx_labels(process_model, pos, font_size=8, font_weight="bold")
    nx.draw_networkx_edge_labels(
        process_model,
        pos,
        edge_labels=edge_labels,  # Use the custom edge labels
        font_size=6
    )

    # Save the diagram as a file
    diagram_output_path = "process_model_diagram.png"
    plt.title("Process Model Diagram", fontsize=8)
    plt.savefig(diagram_output_path)
    plt.close()

    return

# Extract start tasks from paths DataFrame
def extract_start_tasks_from_json(json_file_path):
    """
    Extracts start tasks from the JSON file based on the node type.

    Parameters:
        json_file_path (str): Path to the JSON file.

    Returns:
        set: A set of start task names.
    """
    # Load the JSON file
    with open(json_file_path, "r") as file:
        process_model_data = json.load(file)

    # Extract nodes from JSON
    nodes = process_model_data.get("nodes", [])

    # Find nodes with type "Start"
    start_tasks = set()
    for node in nodes:
        if node.get("type") == "Start":
            start_tasks.add(node["id"])  # Add the id of the start task

    return start_tasks

