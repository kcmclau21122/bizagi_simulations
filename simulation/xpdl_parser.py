import xml.etree.ElementTree as ET
import pandas as pd

def parse_xpdl_to_sequences(xpdl_file_path, output_file_path):
    """
    Parses an XPDL file to extract process sequences with correct conditions.

    Parameters:
        xpdl_file_path (str): Path to the XPDL file.
        output_file_path (str): Path to save the sequences as a text file.

    Returns:
        pd.DataFrame: A DataFrame with the process sequences where each row represents a transition.
    """
    # Parse the XPDL file
    tree = ET.parse(xpdl_file_path)
    root = tree.getroot()

    # Extract namespace
    namespaces = {'xpdl': root.tag.split('}')[0].strip('{')}

    # Helper function to clean activity names
    def clean_name(name):
        return name.strip() if name else "Unknown"

    # Extract activities and transitions
    activities = {}
    transitions = {}

    # Enhanced parsing logic to handle gateways, including Parallel Gateway
    for activity in root.findall(".//xpdl:Activity", namespaces):
        activity_id = activity.get("Id")
        activity_name = clean_name(activity.get("Name"))

        # Check for StartEvent within Event
        event = activity.find(".//xpdl:Event/xpdl:StartEvent", namespaces)
        if event is not None:
            activity_name += " [Type: Start]"

        # Check for gateway attributes
        route = activity.find(".//xpdl:Route", namespaces)
        if route is not None:
            gateway_type = route.get("GatewayType")
            gateway_direction = route.get("GatewayDirection")
            if gateway_type == "Inclusive":
                activity_name += " [Inclusive Gateway]"
            elif gateway_type == "Parallel":
                activity_name += " [Parallel Gateway]"
            elif gateway_direction == "Diverging":
                activity_name += " [Exclusive Gateway]"

        activities[activity_id] = activity_name

    # Extract all transitions with conditions
    for transition in root.findall(".//xpdl:Transition", namespaces):
        from_id = transition.get("From")
        to_id = transition.get("To")
        transition_name = transition.get("Name", "").strip()  # Extract the transition name

        # Assign condition_type based on transition_name
        if transition_name:
            condition_type = f"CONDITION-{transition_name}"
        else:
            condition_type = "Activity Step"  # Default to Activity Step if Name is empty

        # Handle "Unknown" activities for To
        to_activity = activities.get(to_id, "Unknown")
        if to_activity == "Unknown":
            to_activity = "Unknown"
            # Ensure the condition type is preserved
            if "CONDITION" not in condition_type:
                condition_type = "Stop"

        if from_id not in transitions:
            transitions[from_id] = []
        transitions[from_id].append((to_id, to_activity, condition_type))

    # Build sequence rows for output
    sequence_rows = []
    is_first_row = True  # Track the first row

    for from_id, to_transitions in transitions.items():
        # Get the full name for the From activity (including [Type: Start] for the first row only)
        from_activity_full = activities.get(from_id, f"Unknown({from_id})")
        if not is_first_row:
            from_activity_base = from_activity_full.split(" [")[0]  # Base name without type annotation

        for to_id, to_activity, condition in to_transitions:
            # Use full name for the first row; base name for subsequent rows
            from_activity_with_type = from_activity_full if is_first_row else from_activity_base
            is_first_row = False  # Set to False after processing the first row

            # Include annotations or conditions for the To activity
            if to_activity == "Unknown" and "CONDITION" in condition:
                to_activity_with_type = f"Unknown [Type: {condition}]"
            else:
                to_activity_with_type = activities.get(to_id, f"Unknown({to_id})")
                to_activity_with_type += f" [Type: {condition}]"

            # Append the row to the sequence
            sequence_rows.append({
                "From": from_activity_with_type,
                "To": to_activity_with_type,
                "Type": condition
            })

    # Convert to DataFrame and save to text file
    sequences_df = pd.DataFrame(sequence_rows)
    with open(output_file_path, 'w') as file:
        for _, row in sequences_df.iterrows():
            file.write(f"{row['From']} -> {row['To']}\n")
    
    return sequences_df

# Example usage
#xpdl_file_path = './Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews-1.xpdl'
#xpdl_file_path = './Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews-2.xpdl'
#xpdl_file_path = './Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews-Link.xpdl'
#xpdl_file_path = './Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews-Inclusive.xpdl'
xpdl_file_path = './Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews-Parallel.xpdl'
output_sequences_path = 'output_sequences.txt'
parse_xpdl_to_sequences(xpdl_file_path, output_sequences_path)
