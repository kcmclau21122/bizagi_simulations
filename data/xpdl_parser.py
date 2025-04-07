import xml.etree.ElementTree as ET
import pandas as pd
import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple

def parse_xpdl_to_sequences(xpdl_file_path: str, output_file_path: str) -> str:
    """
    Parses an XPDL file to extract process sequences and saves them to a text file.

    Parameters:
        xpdl_file_path (str): Path to the XPDL file.
        output_file_path (str): Path to the output text file.

    Returns:
        str: Path to the generated sequences file.
    """
    # Parse the XPDL file
    tree = ET.parse(xpdl_file_path)
    root = tree.getroot()

    # Extract namespace
    namespaces = {}
    ns_match = re.match(r'{(.+)}', root.tag)
    if ns_match:
        namespaces['xpdl'] = ns_match.group(1)
        # Create a proper namespace dict for ElementTree
        namespaces_dict = {'xpdl': namespaces['xpdl']}
    else:
        # Try to find namespace in attributes
        for key, value in root.attrib.items():
            if key.startswith('xmlns:'):
                namespaces['xpdl'] = value
                namespaces_dict = {'xpdl': value}
                break
        # If still not found, use empty namespace
        if 'xpdl' not in namespaces:
            namespaces['xpdl'] = ''
            namespaces_dict = {}

    # Log namespace for debugging
    logging.debug(f"Using namespace: {namespaces}")

    # Helper function to clean activity names
    def clean_name(name):
        return name.strip() if name else "Unknown"

    # Extract activities and transitions
    activities = {}
    transitions = {}
    gateways = {}

    # Extract process IDs from the XPDL
    process_ids = []
    for process in root.findall(".//xpdl:WorkflowProcess", namespaces_dict) or root.findall(".//WorkflowProcess"):
        process_id = process.get("Id")
        if process_id:
            process_ids.append(process_id)
            logging.debug(f"Found process ID: {process_id}")

    # Try multiple ways to find activities
    activity_elements = []
    
    # Try with namespace
    if namespaces_dict:
        for process_id in process_ids:
            # Try with direct process ID reference
            xpath = f".//xpdl:WorkflowProcess[@Id='{process_id}']/xpdl:Activities/xpdl:Activity"
            activities_found = root.findall(xpath, namespaces_dict)
            if activities_found:
                activity_elements.extend(activities_found)
                logging.debug(f"Found {len(activities_found)} activities with namespace and process ID")
        
        # If still no activities, try general path
        if not activity_elements:
            activity_elements = root.findall(".//xpdl:Activity", namespaces_dict)
            logging.debug(f"Found {len(activity_elements)} activities with namespace")
    
    # Try without namespace
    if not activity_elements:
        for process_id in process_ids:
            xpath = f".//WorkflowProcess[@Id='{process_id}']/Activities/Activity"
            activities_found = root.findall(xpath)
            if activities_found:
                activity_elements.extend(activities_found)
                logging.debug(f"Found {len(activities_found)} activities without namespace and with process ID")
        
        # If still no activities, try general path without namespace
        if not activity_elements:
            activity_elements = root.findall(".//Activity")
            logging.debug(f"Found {len(activity_elements)} activities without namespace")

    # Debug all activity elements
    logging.debug(f"Total activities found: {len(activity_elements)}")
    
    # Enhanced parsing logic to handle gateways and events with different attributes
    for activity in activity_elements:
        activity_id = activity.get("Id")
        if not activity_id:
            continue
            
        activity_name = clean_name(activity.get("Name"))
        node_type = "Activity Step"  # Default type

        # Try to find a Route element (gateway)
        route_element = None
        if namespaces_dict:
            route_element = activity.find(".//xpdl:Route", namespaces_dict)
        
        if route_element is None:
            route_element = activity.find(".//Route")
        
        # Check for gateway attributes
        if route_element is not None:
            gateway_type = route_element.get("GatewayType")
            gateway_direction = route_element.get("GatewayDirection")
            
            if gateway_type == "Inclusive":
                activity_name += " [Inclusive Gateway]"
                node_type = "Inclusive Gateway"
                gateways[activity_id] = "Inclusive"
            elif gateway_type == "Parallel":
                activity_name += " [Parallel Gateway]"
                node_type = "Parallel Gateway"
                gateways[activity_id] = "Parallel"
            elif gateway_direction == "Diverging":
                activity_name += " [Exclusive Gateway]"
                node_type = "Exclusive Gateway"
                gateways[activity_id] = "Exclusive"
            elif gateway_direction == "Converging":
                # Check if there was a specific gateway type
                if gateway_type:
                    activity_name += f" [{gateway_type} Gateway]"
                    node_type = f"{gateway_type} Gateway"
                    gateways[activity_id] = gateway_type
                else:
                    activity_name += " [Gateway]"
                    node_type = "Gateway"
                    gateways[activity_id] = "Generic"
            else:
                # Default gateway label if type not specified
                activity_name += " [Gateway]"
                node_type = "Gateway"
                gateways[activity_id] = "Generic"

        # Check for StartEvent with any trigger type (including Timer)
        start_event = None
        if namespaces_dict:
            start_event = activity.find(".//xpdl:Event/xpdl:StartEvent", namespaces_dict)
        
        if start_event is None:
            start_event = activity.find(".//Event/StartEvent")
            
        if start_event is None and namespaces_dict:
            start_event = activity.find(".//*[@EventType='Start']", namespaces_dict)
            
        if start_event is None:
            start_event = activity.find(".//*[@EventType='Start']")
            
        if start_event is not None:
            activity_name += " [Type: Start]"
            node_type = "Start"
            logging.debug(f"Found start event: {activity_name}")

        # Check for EndEvent with any result type (including None)
        end_event = None
        if namespaces_dict:
            end_event = activity.find(".//xpdl:Event/xpdl:EndEvent", namespaces_dict)
        
        if end_event is None:
            end_event = activity.find(".//Event/EndEvent")
            
        if end_event is None and namespaces_dict:
            end_event = activity.find(".//*[@EventType='End']", namespaces_dict)
            
        if end_event is None:
            end_event = activity.find(".//*[@EventType='End']")
            
        if end_event is not None:
            activity_name += " [Type: Stop]"
            node_type = "Stop"
            logging.debug(f"Found end event: {activity_name}")

        # Also check generic Event elements
        event_element = None
        if namespaces_dict:
            event_element = activity.find(".//xpdl:Event", namespaces_dict)
        
        if event_element is None:
            event_element = activity.find(".//Event")
            
        if event_element is not None:
            start_event_in_event = None
            end_event_in_event = None
            
            if namespaces_dict:
                start_event_in_event = event_element.find("./xpdl:StartEvent", namespaces_dict)
                end_event_in_event = event_element.find("./xpdl:EndEvent", namespaces_dict)
            
            if start_event_in_event is None:
                start_event_in_event = event_element.find("./StartEvent")
            
            if end_event_in_event is None:
                end_event_in_event = event_element.find("./EndEvent")
            
            if start_event_in_event is not None:
                activity_name += " [Type: Start]"
                node_type = "Start"
                logging.debug(f"Found start event via Event element: {activity_name}")
            elif end_event_in_event is not None:
                activity_name += " [Type: Stop]"
                node_type = "Stop"
                logging.debug(f"Found end event via Event element: {activity_name}")

        # Store the activity with its processed information
        activities[activity_id] = {"name": activity_name, "type": node_type}
        
        # Log found activity for debugging
        logging.debug(f"Found activity: {activity_id} -> {activity_name} ({node_type})")

    # Extract all transitions with conditions
    transition_elements = []
    
    # Try with namespace
    if namespaces_dict:
        for process_id in process_ids:
            # Try with direct process ID reference
            xpath = f".//xpdl:WorkflowProcess[@Id='{process_id}']/xpdl:Transitions/xpdl:Transition"
            transitions_found = root.findall(xpath, namespaces_dict)
            if transitions_found:
                transition_elements.extend(transitions_found)
                logging.debug(f"Found {len(transitions_found)} transitions with namespace and process ID")
        
        # If still no transitions, try general path
        if not transition_elements:
            transition_elements = root.findall(".//xpdl:Transition", namespaces_dict)
            logging.debug(f"Found {len(transition_elements)} transitions with namespace")
    
    # Try without namespace
    if not transition_elements:
        for process_id in process_ids:
            xpath = f".//WorkflowProcess[@Id='{process_id}']/Transitions/Transition"
            transitions_found = root.findall(xpath)
            if transitions_found:
                transition_elements.extend(transitions_found)
                logging.debug(f"Found {len(transitions_found)} transitions without namespace and with process ID")
        
        # If still no transitions, try general path without namespace
        if not transition_elements:
            transition_elements = root.findall(".//Transition")
            logging.debug(f"Found {len(transition_elements)} transitions without namespace")

    # Process all transitions
    for transition in transition_elements:
        from_id = transition.get("From")
        to_id = transition.get("To")
        
        if not from_id or not to_id:
            continue
            
        transition_name = transition.get("Name", "").strip()  # Extract the transition name
        
        # Find condition using multiple methods
        condition = None
        condition_text = ""
        
        # Try with namespace
        if namespaces_dict:
            condition = transition.find(".//xpdl:Condition", namespaces_dict)
        
        # If not found, try without namespace
        if condition is None:
            condition = transition.find(".//Condition")
        
        # Try attribute-based condition
        if condition is None and namespaces_dict:
            condition = transition.find(".//*[@ConditionType]", namespaces_dict)
        
        if condition is None:
            condition = transition.find(".//*[@ConditionType]")
            
        # Extract condition text
        if condition is not None:
            condition_text = condition.text or condition.get("Expression", "")
            condition_text = condition_text.strip()
            
        # If no condition found in Condition element, try Condition attribute
        if not condition_text:
            condition_text = transition.get("Condition", "").strip()

        # Assign condition_type based on transition_name or condition
        condition_type = "Activity Step"  # Default
        if transition_name:
            condition_type = f"CONDITION-{transition_name}"
        elif condition_text:
            condition_type = f"CONDITION-{condition_text}"
            
        # Get the activities for these IDs
        from_activity_info = activities.get(from_id, {"name": f"Activity-{from_id}", "type": "Unknown"})
        to_activity_info = activities.get(to_id, {"name": f"Activity-{to_id}", "type": "Unknown"})
        
        from_activity = from_activity_info["name"]
        to_activity = to_activity_info["name"]

        # Check if the from_id is a gateway and update condition_type accordingly
        if from_id in gateways:
            gateway_type = gateways[from_id]
            if gateway_type == "Inclusive" or gateway_type == "Parallel":
                if not "CONDITION-" in condition_type:
                    # For inclusive/parallel gateways, highlight the path in condition_type
                    if transition_name:
                        condition_type = f"CONDITION-{transition_name} (Path from {gateway_type} Gateway)"
                    else:
                        condition_type = f"CONDITION-Path from {gateway_type} Gateway"

        if from_id not in transitions:
            transitions[from_id] = []
            
        transitions[from_id].append((to_id, to_activity, condition_type))
        
        # Log the transition
        logging.debug(f"Found transition: {from_id}({from_activity}) -> {to_id}({to_activity}) [{condition_type}]")

    # Build sequence rows for output
    sequence_rows = []

    for from_id, to_transitions in transitions.items():
        # Get the full name for the From activity
        from_activity_info = activities.get(from_id, {"name": f"Unknown({from_id})", "type": "Unknown"})
        from_activity = from_activity_info["name"]
        
        for to_id, to_activity, condition in to_transitions:
            # Use full name for the to activity
            to_activity_info = activities.get(to_id, {"name": f"Unknown({to_id})", "type": "Unknown"})
            to_activity_actual = to_activity_info["name"]
            
            # Add condition if it exists and isn't already part of the activity name
            if "CONDITION" in condition and condition not in to_activity_actual:
                condition_display = condition
            else:
                condition_display = condition
            
            # Append the row to the sequence
            sequence_rows.append({
                "From": from_activity,
                "To": to_activity_actual,
                "Type": condition_display
            })
    
    # Log summary
    logging.info(f"Found {len(activities)} activities and {len(transitions)} transitions in XPDL file")
    logging.info(f"Generated {len(sequence_rows)} sequence rows")
    
    # Check if we found any sequences
    if not sequence_rows:
        logging.warning(f"No sequences found in XPDL file {xpdl_file_path}")
        
        # Try to provide diagnostic information
        if not activities:
            logging.warning("No activities found. Check XPDL namespace and structure.")
        if not transitions:
            logging.warning("No transitions found. Check XPDL transition elements.")
    
    # Write sequences to text file
    with open(output_file_path, 'w') as f:
        f.write(f"PROCESS SEQUENCES FROM {os.path.basename(xpdl_file_path)}\n")
        f.write("===============================================\n\n")
        
        for row in sequence_rows:
            f.write(f"{row['From']} -> {row['To']}\n")
            # Add type information in a new line if it's a condition
            if "CONDITION-" in row['Type']:
                f.write(f"    Type: {row['Type']}\n")
    
    logging.info(f"Saved process sequences to {output_file_path}")
    
    return output_file_path

def parse_xpdl_to_json(xpdl_file_path: str, metrics_df: Optional[pd.DataFrame] = None) -> str:
    """
    Parses an XPDL file to extract process sequences and saves the model to a JSON file.

    Parameters:
        xpdl_file_path (str): Path to the XPDL file.
        metrics_df (pd.DataFrame, optional): DataFrame containing simulation metrics.

    Returns:
        str: Path to the generated JSON file.
    """
    # Parse the XPDL file
    tree = ET.parse(xpdl_file_path)
    root = tree.getroot()

    # Extract namespace
    namespaces = {}
    ns_match = re.match(r'{(.+)}', root.tag)
    if ns_match:
        namespaces['xpdl'] = ns_match.group(1)
        # Create a proper namespace dict for ElementTree
        namespaces_dict = {'xpdl': namespaces['xpdl']}
    else:
        # Try to find namespace in attributes
        for key, value in root.attrib.items():
            if key.startswith('xmlns:'):
                namespaces['xpdl'] = value
                namespaces_dict = {'xpdl': value}
                break
        # If still not found, use empty namespace
        if 'xpdl' not in namespaces:
            namespaces['xpdl'] = ''
            namespaces_dict = {}

    # Log namespace for debugging
    logging.debug(f"Using namespace: {namespaces}")

    # Helper function to clean activity names
    def clean_name(name):
        return name.strip() if name else "Unknown"

    # Extract activities and transitions
    activities = {}
    transitions = {}
    nodes = []
    links = []

    # Extract process IDs from the XPDL
    process_ids = []
    for process in root.findall(".//xpdl:WorkflowProcess", namespaces_dict) or root.findall(".//WorkflowProcess"):
        process_id = process.get("Id")
        if process_id:
            process_ids.append(process_id)
            logging.debug(f"Found process ID: {process_id}")

    # Try multiple ways to find activities
    activity_elements = []
    
    # Try with namespace
    if namespaces_dict:
        for process_id in process_ids:
            # Try with direct process ID reference
            xpath = f".//xpdl:WorkflowProcess[@Id='{process_id}']/xpdl:Activities/xpdl:Activity"
            activities_found = root.findall(xpath, namespaces_dict)
            if activities_found:
                activity_elements.extend(activities_found)
                logging.debug(f"Found {len(activities_found)} activities with namespace and process ID")
        
        # If still no activities, try general path
        if not activity_elements:
            activity_elements = root.findall(".//xpdl:Activity", namespaces_dict)
            logging.debug(f"Found {len(activity_elements)} activities with namespace")
    
    # Try without namespace
    if not activity_elements:
        for process_id in process_ids:
            xpath = f".//WorkflowProcess[@Id='{process_id}']/Activities/Activity"
            activities_found = root.findall(xpath)
            if activities_found:
                activity_elements.extend(activities_found)
                logging.debug(f"Found {len(activities_found)} activities without namespace and with process ID")
        
        # If still no activities, try general path without namespace
        if not activity_elements:
            activity_elements = root.findall(".//Activity")
            logging.debug(f"Found {len(activity_elements)} activities without namespace")

    # Enhanced parsing logic to handle activities, gateways, and events
    for activity in activity_elements:
        activity_id = activity.get("Id")
        if not activity_id:
            continue
            
        activity_name = clean_name(activity.get("Name"))
        node_type = "Activity Step"  # Default type

        # Check if this is a gateway
        route = activity.find(".//Route", namespaces_dict) or activity.find(".//*[@GatewayType]", namespaces_dict)
        if route is not None:
            gateway_type = route.get("GatewayType")
            gateway_direction = route.get("GatewayDirection")
            
            if gateway_type == "Inclusive":
                activity_name += " [Inclusive Gateway]"
                node_type = "Inclusive Gateway"
            elif gateway_type == "Parallel":
                activity_name += " [Parallel Gateway]"
                node_type = "Parallel Gateway"
            elif gateway_direction == "Diverging":
                activity_name += " [Exclusive Gateway]"
                node_type = "Exclusive Gateway"

        # Check for StartEvent
        start_event = activity.find(".//Event/StartEvent", namespaces_dict) or activity.find(".//*[@EventType='Start']", namespaces_dict)
        if start_event is not None:
            activity_name += " [Type: Start]"
            node_type = "Start"

        # Check for EndEvent
        end_event = activity.find(".//Event/EndEvent", namespaces_dict) or activity.find(".//*[@EventType='End']", namespaces_dict)
        if end_event is not None:
            activity_name += " [Type: Stop]"
            node_type = "Stop"

        activities[activity_id] = activity_name
        
        # Create node data
        node_data = {
            "id": activity_id,
            "name": activity_name,
            "type": node_type
        }
        
        # Add metrics data if available
        if metrics_df is not None:
            try:
                # Make sure 'name' column exists in metrics_df
                if 'name' in metrics_df.columns:
                    # Find matching row in metrics by name
                    activity_name_clean = activity_name.lower().split('[')[0].strip()
                    metrics_row = metrics_df[metrics_df['name'].str.lower() == activity_name_clean]
                    if not metrics_row.empty:
                        for col, value in metrics_row.iloc[0].items():
                            if pd.notna(value) and col not in ['name', 'id']:
                                node_data[col] = value
                else:
                    logging.warning("Cannot add metrics: 'name' column not found in metrics dataframe")
            except Exception as e:
                logging.warning(f"Error adding metrics to node {activity_name}: {str(e)}")
        
        # Add to nodes list
        nodes.append(node_data)

    # Extract all transitions with conditions
    transition_elements = []
    
    # Try with namespace
    if namespaces_dict:
        for process_id in process_ids:
            # Try with direct process ID reference
            xpath = f".//xpdl:WorkflowProcess[@Id='{process_id}']/xpdl:Transitions/xpdl:Transition"
            transitions_found = root.findall(xpath, namespaces_dict)
            if transitions_found:
                transition_elements.extend(transitions_found)
                logging.debug(f"Found {len(transitions_found)} transitions with namespace and process ID")
        
        # If still no transitions, try general path
        if not transition_elements:
            transition_elements = root.findall(".//xpdl:Transition", namespaces_dict)
            logging.debug(f"Found {len(transition_elements)} transitions with namespace")
    
    # Try without namespace
    if not transition_elements:
        for process_id in process_ids:
            xpath = f".//WorkflowProcess[@Id='{process_id}']/Transitions/Transition"
            transitions_found = root.findall(xpath)
            if transitions_found:
                transition_elements.extend(transitions_found)
                logging.debug(f"Found {len(transitions_found)} transitions without namespace and with process ID")
        
        # If still no transitions, try general path without namespace
        if not transition_elements:
            transition_elements = root.findall(".//Transition")
            logging.debug(f"Found {len(transition_elements)} transitions without namespace")

    # Process all transitions
    for transition in transition_elements:
        from_id = transition.get("From")
        to_id = transition.get("To")
        
        if not from_id or not to_id:
            continue
            
        transition_name = transition.get("Name", "").strip()  # Extract the transition name
        
        # Find condition using multiple methods
        condition = None
        condition_text = ""
        
        # Try with namespace
        if namespaces_dict:
            condition = transition.find(".//xpdl:Condition", namespaces_dict)
        
        # If not found, try without namespace
        if condition is None:
            condition = transition.find(".//Condition")
        
        # Try attribute-based condition
        if condition is None and namespaces_dict:
            condition = transition.find(".//*[@ConditionType]", namespaces_dict)
        
        if condition is None:
            condition = transition.find(".//*[@ConditionType]")
            
        # Extract condition text
        if condition is not None:
            condition_text = condition.text or condition.get("Expression", "")
            condition_text = condition_text.strip()
            
        # If no condition found in Condition element, try Condition attribute
        if not condition_text:
            condition_text = transition.get("Condition", "").strip()

        # Assign condition_type based on transition_name or condition
        if transition_name:
            condition_type = f"CONDITION-{transition_name}"
        elif condition_text:
            condition_type = f"CONDITION-{condition_text}"
        else:
            condition_type = "NORMAL"  # Default to normal if no condition

        # Make sure we have the activities for these IDs
        from_activity = activities.get(from_id, f"Activity-{from_id}")
        to_activity = activities.get(to_id, f"Activity-{to_id}")
        
        # Create link data
        link_data = {
            "source": from_activity,
            "target": to_activity,
            "type": condition_type
        }
        
        # Add to links list
        links.append(link_data)
        
        # Also track in transitions dict for sequence file
        if from_id not in transitions:
            transitions[from_id] = []
        transitions[from_id].append((to_id, to_activity, condition_type))

    # Create jsons directory if it doesn't exist
    jsons_dir = "jsons"
    if not os.path.exists(jsons_dir):
        os.makedirs(jsons_dir)
        logging.info(f"Created directory: {jsons_dir}")
    
    # Generate JSON filename using just the base name without path
    base_filename = os.path.splitext(os.path.basename(xpdl_file_path))[0]
    json_output_path = os.path.join(jsons_dir, f"{base_filename}_sequences.json")
    
    # If no nodes were found, add a fallback start and end node
    if not nodes:
        logging.warning("No nodes found in XPDL. Adding fallback nodes.")
        start_node = {
            "id": "start_node",
            "name": f"{base_filename} Start [Type: Start]",
            "type": "Start"
        }
        end_node = {
            "id": "end_node",
            "name": f"{base_filename} End [Type: Stop]",
            "type": "Stop"
        }
        nodes = [start_node, end_node]
        links = [{
            "source": start_node["name"],
            "target": end_node["name"],
            "type": "NORMAL"
        }]
        logging.warning("Using fallback nodes as no activities were found in the XPDL")
    
    # Enhanced JSON structure
    json_data = {
        "metadata": {
            "source_file": os.path.basename(xpdl_file_path),
            "creation_date": pd.Timestamp.now().isoformat(),
            "total_sequences": len(links)
        },
        "nodes": nodes,
        "links": links
    }
    
    # Save to JSON file with nice formatting
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
    
    logging.info(f"Saved process model JSON to {json_output_path}")
    
    return json_output_path

def parse_xpdl_with_metrics(xpdl_file_path, metrics_file_path=None):
    """
    Parse XPDL file and merge with metrics data from Excel file.
    
    Parameters:
        xpdl_file_path (str): Path to the XPDL file
        metrics_file_path (str, optional): Path to the Excel file with metrics
    
    Returns:
        str: Path to the generated JSON file
    """
    metrics_df = None
    
    # Load metrics if provided
    if metrics_file_path and os.path.exists(metrics_file_path):
        try:
            metrics_df = pd.read_excel(metrics_file_path)
            # Normalize column names to lowercase for consistency
            metrics_df.columns = [str(col).lower() for col in metrics_df.columns]
            logging.info(f"Loaded metrics from {metrics_file_path} with columns: {list(metrics_df.columns)}")
            
            # Check if 'name' column exists, try alternatives if not
            if 'name' not in metrics_df.columns:
                # Look for alternative column names
                name_alternatives = ['activity', 'activity name', 'node', 'node name', 'id', 'activity id']
                for alt in name_alternatives:
                    if alt in metrics_df.columns:
                        # Rename to 'name' for consistency
                        metrics_df.rename(columns={alt: 'name'}, inplace=True)
                        logging.info(f"Renamed column '{alt}' to 'name'")
                        break
                
                # If still no 'name' column, create one with numeric IDs
                if 'name' not in metrics_df.columns:
                    logging.warning("No name column found in metrics. Creating default names.")
                    metrics_df['name'] = [f"Activity_{i}" for i in range(len(metrics_df))]
        except Exception as e:
            logging.error(f"Error loading metrics file: {e}")
            metrics_df = None
    
    # Parse XPDL and merge with metrics
    return parse_xpdl_to_json(xpdl_file_path, metrics_df)

# Usage example
if __name__ == "__main__":
    # This code will run if the file is executed directly
    xpdl_file_path = './example.xpdl'
    metrics_file_path = './simulation_metrics.xlsx'
    
    # Uncomment to test with actual files
    # parse_xpdl_with_metrics(xpdl_file_path, metrics_file_path)
