# Contains XPDLParser class with parse_xpdl method. 
# # Returns a dictionary of nodes and transitions, converting XML elements into the data classes.
# Created Date: 2 Feb 2025

import xml.etree.ElementTree as ET
from typing import Dict, List
from models import ProcessNode, ProcessTransition
from event_logger import EventLogger
from models import SimulationEvent
from datetime import datetime
from pathlib import Path


class XPDLParser:
    NAMESPACE = {'xpdl': 'http://www.wfmc.org/2008/XPDL2.2'}
    logger = EventLogger()  # Initialize logger

    @classmethod
    def parse_xpdl(cls, file_path: str) -> Dict:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        nodes = cls._parse_nodes(root)
        transitions = cls._parse_transitions(root)

        # Save log after parsing
        cls.logger.write_log_file(Path("xpdl_parsing_log.txt"))

        return {'nodes': nodes, 'transitions': transitions}

    @classmethod
    def _parse_nodes(cls, root: ET.Element) -> Dict[str, ProcessNode]:
        nodes = {}
        log_path = Path("xpdl_parsing_log.txt")  # Define log path

        for element in root.findall('.//xpdl:Activity', cls.NAMESPACE):
            node_id = element.attrib['Id']
            node_name = element.attrib.get('Name', 'Unnamed Node')

            # Log node parsing details
            cls.logger.add_event(SimulationEvent(
                timestamp=datetime.now(),
                token=-1,  # -1 to indicate system-level event
                node=node_id,
                event_type="NODE_PARSED",
                details={'Node Name': node_name}
            ))

            # Immediately write to log after each node is added
            cls.logger.write_log_file(log_path)

            nodes[node_id] = cls._create_node(element)

        return nodes

    @classmethod
    def _create_node(cls, element: ET.Element) -> ProcessNode:
        node_type = 'activity'
        gateway_type = None
        
        if route := element.find('.//xpdl:Route', cls.NAMESPACE):
            node_type = 'gateway'
            gateway_element = route.find('.//xpdl:GatewayType', cls.NAMESPACE)
            gateway_type = gateway_element.text.lower() if gateway_element else 'exclusive'

        return ProcessNode(
            id=element.attrib['Id'],
            name=element.attrib['Name'],
            node_type=node_type,
            gateway_type=gateway_type
        )

    @classmethod
    def _parse_transitions(cls, root: ET.Element) -> List[ProcessTransition]:
        return [
            ProcessTransition(
                id=element.attrib['Id'],
                from_node=element.attrib['From'],
                to_node=element.attrib['To']
            )
            for element in root.findall('.//xpdl:Transition', cls.NAMESPACE)
        ]