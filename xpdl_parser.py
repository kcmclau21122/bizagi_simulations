# Used to parse a Bizagi XPDL file and create the activity sequences which are saved to a text file and 
# passed back to the caller.
#
# Needed Improvements:
# 1. Use the Transition child tag <Condition Type="CONDITION"> to note a Gateway and then get the "Name" value to use with the 
#    simulations_metrics to get the percentages for the different paths from the Gateway.
#
#################################################################################################################################
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
            sequence_rows.append({"From": sequence[i], "To": sequence[i + 1]})

    sequences_df = pd.DataFrame(sequence_rows)

    # Save sequences to a text file
    with open(output_file_path, 'w') as file:
        for sequence in sequences:
            file.write(" -> ".join(sequence) + "\n")

    return sequences_df

# Example usage
# xpdl_file_path = "path_to_your_xpdl_file.xpdl"
# output_file_path = "output_sequences.txt"
# sequences_df = parse_xpdl_to_sequences(xpdl_file_path, output_file_path)
# print(sequences_df)
