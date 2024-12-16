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

    # Extract all activities
    for activity in root.findall(".//xpdl:Activity", namespaces):
        activity_id = activity.get("Id")
        activity_name = clean_name(activity.get("Name"))
        activities[activity_id] = activity_name

    # Extract all transitions with conditions
    for transition in root.findall(".//xpdl:Transition", namespaces):
        from_id = transition.get("From")
        to_id = transition.get("To")
        transition_name = transition.get("Name", "Unknown")  # Extract the transition name (e.g., "Yes", "No")

        # Resolve "Unknown" to "Stop" for all cases
        to_activity = activities.get(to_id, "Unknown")
        if to_activity == "Unknown":
            to_activity = "Stop"

        condition_type = f"CONDITION-{transition_name}" if transition_name in ["Yes", "No"] else "Activity Step"
        if from_id not in transitions:
            transitions[from_id] = []
        transitions[from_id].append((to_id, to_activity, condition_type))

    # Build sequence rows for output
    sequence_rows = []
    for from_id, to_transitions in transitions.items():
        from_activity = activities.get(from_id, f"Unknown({from_id})")
        for to_id, to_activity, condition in to_transitions:
            sequence_rows.append({
                "From": from_activity,
                "To": to_activity,
                "Type": condition
            })

    # Convert to DataFrame and save to text file
    sequences_df = pd.DataFrame(sequence_rows)
    with open(output_file_path, 'w') as file:
        for _, row in sequences_df.iterrows():
            file.write(f"{row['From']} -> {row['To']} [Type: {row['Type']}]\n")
    
    return sequences_df

# Example usage
# xpdl_file_path = "path_to_xpdl_file.xpdl"
# output_file_path = "output_sequences.txt"
# parse_xpdl_to_sequences(xpdl_file_path, output_file_path)
