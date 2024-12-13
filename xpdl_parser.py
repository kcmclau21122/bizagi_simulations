# BUG: Treats all CONDITION as a "No" path 

import xml.etree.ElementTree as ET
import pandas as pd

def parse_xpdl_to_sequences(xpdl_file_path, output_file_path):
    """
    Parses an XPDL file to extract process sequences, including handling intermediate link events.

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
    activity_types = {}
    start_event_ids = []
    end_event_ids = []
    transitions = {}
    throw_links = {}  # Name -> Activity ID
    catch_links = {}  # Name -> List of Activity IDs

    # Extract all activities
    for activity in root.findall(".//xpdl:Activity", namespaces):
        activity_id = activity.get("Id")
        activity_name = clean_name(activity.get("Name"))
        activities[activity_id] = activity_name

        # Check if it's a StartEvent
        if activity.find("./xpdl:Event/xpdl:StartEvent", namespaces) is not None:
            start_event_ids.append(activity_id)

        # Check if it's an EndEvent
        if activity.find("./xpdl:Event/xpdl:EndEvent", namespaces) is not None:
            end_event_ids.append(activity_id)

        # Check for Intermediate Events with Link Trigger
        intermediate_event = activity.find("./xpdl:Event/xpdl:IntermediateEvent[@Trigger='Link']", namespaces)
        if intermediate_event is not None:
            # Check if it's a Throw event
            throw_tag = intermediate_event.find("./xpdl:TriggerResultLink[@CatchThrow='THROW']", namespaces)
            if throw_tag is not None:
                throw_links[activity_name] = activity_id
            else:  # It's a Catch event
                if activity_name not in catch_links:
                    catch_links[activity_name] = []
                catch_links[activity_name].append(activity_id)

    # Extract all transitions
    for transition in root.findall(".//xpdl:Transition", namespaces):
        from_id = transition.get("From")
        to_id = transition.get("To")
        if from_id not in transitions:
            transitions[from_id] = []
        transitions[from_id].append(to_id)

        # Identify activity type from the Condition tag
        condition = transition.find("./xpdl:Condition[@Type='CONDITION']", namespaces)
        if condition is not None and from_id in activities:
            condition_type = condition.get("Type", "Unknown")
            gateway_name = transition.get("Name", "Unknown")
            if from_id not in activity_types:
                activity_types[from_id] = []
            activity_types[from_id].append(f"{condition_type}-{gateway_name}")

    # Ensure each condition is handled separately
    activity_conditions = {}
    for activity_id, conditions in activity_types.items():
        activity_conditions[activity_id] = conditions

    # Default all other activities to "Activity Step"
    for activity_id in activities:
        if activity_id not in activity_conditions:
            activity_conditions[activity_id] = ["Activity Step"]

    # Recursive function to traverse sequences
    def traverse_sequence(current_id, path, visited):
        if current_id in visited:  # Avoid infinite loops
            return
        visited.add(current_id)

        # Add the current activity to the path
        path.append(activities.get(current_id, f"Unknown({current_id})"))

        # Check if this is an end event
        if current_id in end_event_ids:
            sequences.append(path.copy())
        else:
            # Continue to all possible transitions
            for next_id in transitions.get(current_id, []):
                traverse_sequence(next_id, path, visited)

            # Check for link transitions if the current activity is a Throw
            if current_id in throw_links.values():
                throw_name = activities[current_id]
                if throw_name in catch_links:
                    for catch_id in catch_links[throw_name]:
                        traverse_sequence(catch_id, path, visited)

        # Backtrack
        path.pop()
        visited.remove(current_id)

    # Find sequences from all Start Events
    sequences = []
    for start_id in start_event_ids:
        traverse_sequence(start_id, [], set())

    # Convert sequences to a DataFrame
    sequence_rows = []
    for sequence in sequences:
        for i in range(len(sequence) - 1):
            from_activity = sequence[i]
            to_activity = sequence[i + 1]
            from_id = [key for key, value in activities.items() if value == from_activity][0]
            conditions = activity_conditions.get(from_id, ["Unknown"])
            for condition in conditions:
                sequence_rows.append({
                    "From": from_activity,
                    "To": to_activity,
                    "Type": condition
                })

    sequences_df = pd.DataFrame(sequence_rows)

    # Save sequences to a text file
    with open(output_file_path, 'w') as file:
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                from_activity = sequence[i]
                to_activity = sequence[i + 1]
                from_id = [key for key, value in activities.items() if value == from_activity][0]
                conditions = activity_conditions.get(from_id, ["Unknown"])
                for condition in conditions:
                    file.write(f"{from_activity} -> {to_activity} [Type: {condition}]\n")

    return sequences_df

# Example usage
# xpdl_file_path = "path_to_your_xpdl_file.xpdl"
# output_file_path = "output_sequences.txt"
# sequences_df = parse_xpdl_to_sequences(xpdl_file_path, output_file_path)
# print(sequences_df)
