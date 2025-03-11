import xml.etree.ElementTree as ET
import pandas as pd
import logging

def parse_xpdl_to_sequences(xpdl_file_path, output_file_path):
    """
    Parses an XPDL file to extract process sequences with correct gateway associations.

    Parameters:
        xpdl_file_path (str): Path to the XPDL file.
        output_file_path (str): Path to save the sequences as a text file.

    Returns:
        pd.DataFrame: A DataFrame with the process sequences where each row represents a transition.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parse the XPDL file
    tree = ET.parse(xpdl_file_path)
    root = tree.getroot()

    # Extract namespace
    namespaces = {'xpdl': root.tag.split('}')[0].strip('{')}
    logging.info(f"Parsing XPDL file: {xpdl_file_path} with namespace: {namespaces}")

    # Helper function to clean activity names
    def clean_name(name):
        return name.strip() if name else "Unknown"

    # Extract activities and their gateway types
    activities = {}
    gateway_activities = {}  # Map to track gateway activities and their types
    
    # Step 1: First pass to identify all gateways and activities
    for activity in root.findall(".//xpdl:Activity", namespaces):
        activity_id = activity.get("Id")
        activity_name = clean_name(activity.get("Name"))

        # Check for StartEvent
        event = activity.find(".//xpdl:Event/xpdl:StartEvent", namespaces)
        if event is not None:
            activity_name += " [Type: Start]"
            activities[activity_id] = {"name": activity_name, "type": "Start", "gateway": None}
            continue

        # Check for EndEvent
        event = activity.find(".//xpdl:Event/xpdl:EndEvent", namespaces)
        if event is not None:
            activity_name += " [Type: Stop]"
            activities[activity_id] = {"name": activity_name, "type": "Stop", "gateway": None}
            continue

        # Check for gateway attributes
        route = activity.find(".//xpdl:Route", namespaces)
        if route is not None:
            gateway_type = route.get("GatewayType")
            gateway_direction = route.get("GatewayDirection")
            
            if gateway_type == "Inclusive":
                activity_name += " [Inclusive Gateway]"
                gateway_type = "[Inclusive Gateway]"
                gateway_activities[activity_id] = gateway_type
                activities[activity_id] = {"name": activity_name, "type": "[Inclusive Gateway]", "gateway": gateway_type}
            elif gateway_type == "Parallel":
                activity_name += " [Parallel Gateway]"
                gateway_type = "[Parallel Gateway]"
                gateway_activities[activity_id] = gateway_type
                activities[activity_id] = {"name": activity_name, "type": "[Parallel Gateway]", "gateway": gateway_type}
            elif gateway_direction == "Diverging":
                activity_name += " [Exclusive Gateway]"
                gateway_type = "[Exclusive Gateway]"
                gateway_activities[activity_id] = gateway_type
                activities[activity_id] = {"name": activity_name, "type": "Activity Step", "gateway": gateway_type}
            else:
                activities[activity_id] = {"name": activity_name, "type": "Activity Step", "gateway": None}
        else:
            # Regular activity
            activities[activity_id] = {"name": activity_name, "type": "Activity Step", "gateway": None}
    
    logging.info(f"Found {len(activities)} activities, including {len(gateway_activities)} gateways")
    logging.info(f"Gateway activities: {gateway_activities}")

    # Track outgoing transitions from each gateway
    gateway_outgoing = {gw_id: [] for gw_id in gateway_activities.keys()}
    
    # Step 2: Extract all transitions and build a source-target mapping
    transitions = []
    for transition in root.findall(".//xpdl:Transition", namespaces):
        from_id = transition.get("From")
        to_id = transition.get("To")
        transition_name = transition.get("Name", "").strip()
        
        # If the source is a gateway, add the target to its outgoing list
        if from_id in gateway_activities:
            gateway_outgoing[from_id].append(to_id)
        
        # Determine condition type based on transition name
        if transition_name:
            condition_type = f"CONDITION-{transition_name}"
        else:
            condition_type = "Activity Step"
        
        transitions.append({
            "from_id": from_id, 
            "to_id": to_id, 
            "condition": condition_type
        })
    
    logging.info(f"Found {len(transitions)} transitions in the process")
    
    # Step 3: Update activities based on their association with gateways
    for gw_id, outgoing_ids in gateway_outgoing.items():
        gateway_type = gateway_activities[gw_id]
        logging.info(f"Processing outgoing activities for gateway {gw_id} of type {gateway_type}")
        
        # For each activity connected to this gateway
        for activity_id in outgoing_ids:
            if activity_id in activities:
                # If this is an inclusive or parallel gateway, propagate the gateway type
                if gateway_type in ["[Inclusive Gateway]", "[Parallel Gateway]"]:
                    logging.info(f"  Setting gateway type for {activity_id} to {gateway_type}")
                    # Update the gateway type for this activity
                    activities[activity_id]["gateway"] = gateway_type
    
    # Step 4: Build sequence rows for output
    sequence_rows = []
    
    for transition in transitions:
        from_id = transition["from_id"]
        to_id = transition["to_id"]
        condition = transition["condition"]
        
        if from_id not in activities or to_id not in activities:
            logging.warning(f"Missing activity reference: from_id={from_id}, to_id={to_id}")
            continue
        
        from_activity = activities[from_id]
        to_activity = activities[to_id]
        
        # Format the From activity string
        from_str = from_activity["name"]
        if from_activity["gateway"]:
            # Make sure the gateway type is included in the activity name
            if from_activity["gateway"] not in from_str:
                from_str += f" {from_activity['gateway']}"
        
        # Format the To activity string
        to_str = to_activity["name"]
        if to_activity["gateway"]:
            # Make sure the gateway type is included in the activity name
            if to_activity["gateway"] not in to_str:
                to_str += f" {to_activity['gateway']}"
        
        # Add type annotation if not already in the name
        if "[Type:" not in to_str:
            to_str += f" [Type: {condition}]"
            
        # Append the row to the sequence
        sequence_rows.append({
            "From": from_str,
            "To": to_str,
            "Type": condition,
            "FromGateway": from_activity["gateway"],
            "ToGateway": to_activity["gateway"]
        })
    
    # Convert to DataFrame and save to text file
    sequences_df = pd.DataFrame(sequence_rows)
    logging.info(f"Generated {len(sequences_df)} sequence rows")
    
    with open(output_file_path, 'w') as file:
        for _, row in sequences_df.iterrows():
            file.write(f"{row['From']} -> {row['To']}\n")
    
    logging.info(f"Sequence file saved to: {output_file_path}")
    return sequences_df

# Main function to parse XPDL and build/save process model
def parse_and_build_process_model(xpdl_file_path, output_dir='.', metrics_file=None):
    """
    Parse XPDL file, build process model, and save to JSON file with timestamp.
    
    Args:
        xpdl_file_path: Path to the XPDL file
        output_dir: Directory to save output files (default: current directory)
        metrics_file: Path to simulation metrics CSV file (optional)
        
    Returns:
        Path to the saved JSON file
    """
    import os
    from datetime import datetime
    from process_builder import ProcessModelBuilder
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Parse XPDL to sequences
    xpdl_base_name = os.path.splitext(os.path.basename(xpdl_file_path))[0]
    sequence_file_path = os.path.join(output_dir, f"{xpdl_base_name}_sequences.txt")
    
    logging.info(f"Parsing XPDL file: {xpdl_file_path}")
    sequences_df = parse_xpdl_to_sequences(xpdl_file_path, sequence_file_path)
    
    # Load simulation metrics if provided, or create a minimal DataFrame
    if metrics_file and os.path.exists(metrics_file):
        logging.info(f"Loading simulation metrics from: {metrics_file}")
        import pandas as pd
        simulation_metrics = pd.read_csv(metrics_file)
    else:
        logging.warning("No metrics file provided or file not found. Using minimal metrics.")
        import pandas as pd
        # Create a minimal metrics DataFrame with a 'name' column
        activity_names = set()
        for _, row in sequences_df.iterrows():
            source_name = row['From'].split('[')[0].strip()
            target_name = row['To'].split('[')[0].strip()
            activity_names.add(source_name)
            activity_names.add(target_name)
        
        simulation_metrics = pd.DataFrame({'name': list(activity_names)})
    
    # Build the process model
    logging.info("Building process model")
    model_builder = ProcessModelBuilder()
    process_model = model_builder.build_from_sequences(sequence_file_path, simulation_metrics)
    
    # Save the process model to JSON with timestamp
    json_output_path = os.path.join(output_dir, f"process_model_{timestamp}.json")
    saved_path = model_builder.save_to_json(json_output_path)
    
    logging.info(f"Process completed successfully")
    logging.info(f"- Sequences file: {sequence_file_path}")
    logging.info(f"- Process model JSON: {saved_path}")
    
    return saved_path

# Example usage - uncomment to run
# xpdl_file_path = './Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews-Parallel.xpdl'
# json_file_path = parse_and_build_process_model(xpdl_file_path)
