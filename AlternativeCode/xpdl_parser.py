# Contains XPDLParser class with parse_xpdl method. 
# # Returns a dictionary of nodes and transitions, converting XML elements into the data classes.
# Created Date: 2 Feb 2025

import xml.etree.ElementTree as ET
from typing import Dict, List
from AlternativeCode.models import ProcessNode, ProcessTransition

class XPDLParser:
    NAMESPACE = {'xpdl': 'http://www.wfmc.org/2008/XPDL2.2'}

    @classmethod
    def parse_xpdl(cls, file_path: str) -> Dict:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        return {
            'nodes': cls._parse_nodes(root),
            'transitions': cls._parse_transitions(root)
        }

    @classmethod
    def _parse_nodes(cls, root: ET.Element) -> Dict[str, ProcessNode]:
        nodes = {}
        for element in root.findall('.//xpdl:Activity', cls.NAMESPACE):
            node_id = element.attrib['Id']
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