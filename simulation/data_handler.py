# simulation/data_handler.py
import pandas as pd
from collections import defaultdict
import logging
import json
import re
from networkx.readwrite import json_graph
import networkx as nx
import matplotlib.pyplot as plt

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

        # Add source node with attributes
        if source_name not in process_model:
            process_model.add_node(
                source_name,
                type=source_type or "Activity Step",  # Default type if not specified
                gateway=source_gateway
            )

        # Add target node with attributes
        if target_name not in process_model:
            process_model.add_node(
                target_name,
                type=target_type,  # Default type if not specified
                gateway=target_gateway
            )

        # Add edge between source and target
        process_model.add_edge(source_name, target_name)

    # Convert the graph to JSON
    process_model_data = json_graph.node_link_data(process_model)

    # Post-process to update "Unknown" Stop node and its links
    for node in process_model_data["nodes"]:
        if node["type"] == "Stop" and node["id"] == "Unknown":
            old_id = node["id"]
            node["id"] = "Stop"
            for link in process_model_data["links"]:
                if link["target"] == old_id:
                    link["target"] = "Stop"

    # Save the updated JSON file
    json_output_path = "process_model.json"
    with open(json_output_path, "w") as json_file:
        json.dump(process_model_data, json_file, indent=4)

    return json_output_path

def diagram_process(json_output_path):
    # Load the JSON file again
    with open(json_output_path, "r") as json_file:
        json_data = json.load(json_file)

    # Recreate the graph from the JSON data
    process_model = json_graph.node_link_graph(json_data)

    # Draw the graph
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(process_model)  # Use spring layout for better visualization
    nx.draw(
        process_model,
        pos,
        with_labels=True,
        node_size=1000,
        node_color="lightblue",
        font_size=8,
        font_weight="bold",
        edge_color="gray"
    )
    nx.draw_networkx_edge_labels(
        process_model,
        pos,
        edge_labels={(u, v): d.get('type', '') for u, v, d in process_model.edges(data=True)},
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

