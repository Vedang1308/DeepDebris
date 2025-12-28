import networkx as nx
import requests
import csv
import os
import io
from datetime import datetime

class DebrisKnowledgeGraph:
    def __init__(self, data_path="satcat.csv"):
        self.data_path = data_path
        self.graph = nx.DiGraph()
        self._load_data()
        
    def _load_data(self):
        """Fetch and load REAL data from CelesTrak."""
        # Check cache
        if not os.path.exists(self.data_path):
            print("[Knowledge Graph] Fetching official SATCAT from CelesTrak...")
            try:
                url = "https://celestrak.org/pub/satcat.csv"
                response = requests.get(url)
                response.raise_for_status()
                with open(self.data_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"FAILED to fetch CelesTrak data: {e}. Using empty graph.")
                return

        # Parse CSV
        print("[Knowledge Graph] Parsing SATCAT data...")
        launch_groups = {} # Map 'YYYY-NNN' -> Payload Node ID
        
        with open(self.data_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Pass 1: Identify Objects & Owners
            for row in rows:
                norad_id = row.get('NORAD_CAT_ID')
                name = row.get('OBJECT_NAME')
                owner = row.get('OWNER')
                intl_des = row.get('OBJECT_ID') # YYYY-NNNA
                obj_type = row.get('OBJECT_TYPE') # PAY, R/B, DEB
                
                if not norad_id: continue
                
                # Add Object Node
                self.graph.add_node(norad_id, 
                                    name=name, 
                                    type=obj_type, 
                                    intl_des=intl_des,
                                    label=f"{name} ({norad_id})")
                
                # Add Owner Node & Edge
                if owner:
                    self.graph.add_node(owner, type="CountryCode")
                    self.graph.add_edge(norad_id, owner, relation="OWNED_BY")
                
                # Grouping Logic:
                # Intl Des format: YYYY-NNNPPP (Year, Launch Num, Piece)
                # If Piece == 'A', it's likely the primary payload.
                if intl_des and len(intl_des) >= 8:
                    group_id = intl_des[:8] # YYYY-NNN
                    piece = intl_des[8:].strip()
                    
                    if piece == 'A':
                        launch_groups[group_id] = norad_id
            
            # Pass 2: Link Debris to Parents (Payloads)
            for row in rows:
                norad_id = row.get('NORAD_CAT_ID')
                intl_des = row.get('OBJECT_ID')
                
                if not norad_id or not intl_des: continue
                if len(intl_des) < 8: continue
                
                group_id = intl_des[:8]
                piece = intl_des[8:].strip()
                
                # If this is NOT the payload ('A'), link to the payload of the same launch
                if piece != 'A' and group_id in launch_groups:
                    parent_id = launch_groups[group_id]
                    # Don't link to self (rare case if map is wrong)
                    if parent_id != norad_id:
                        self.graph.add_edge(parent_id, norad_id, relation="PARENT_OF")

        print(f"[Knowledge Graph] Built real network: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

    def query_attribution(self, identifier):
        """
        Trace ownership and origin using REAL Graph.
        Input: NORAD ID (String)
        """
        target_node = str(identifier)
        
        if target_node not in self.graph:
            return None
            
        context = {
            "object": self.graph.nodes[target_node].get('name', 'Unknown'),
            "id": target_node,
            "type": self.graph.nodes[target_node].get('type', 'Unknown'),
            "owner": "Unknown",
            "parent": "None",
            "origin_event": f"Launch {self.graph.nodes[target_node].get('intl_des', 'Unknown')}"
        }
        
        # 1. Find Owner (Direct Edge)
        for succ in self.graph.successors(target_node):
            if self.graph.nodes[succ].get('type') == 'CountryCode':
                context['owner'] = succ
        
        # 2. Find Parent (Reverse of PARENT_OF) -> Predecessor
        # Relation: Parent -> PARENT_OF -> Child
        # So Child -> Predecessor -> Parent
        for pred in self.graph.predecessors(target_node):
            if self.graph[pred][target_node]['relation'] == "PARENT_OF":
                context['parent'] = f"{self.graph.nodes[pred].get('name')} ({pred})"
                # If owner unknown, inherit from parent
                if context['owner'] == "Unknown":
                    # Check parent's owner
                    for p_succ in self.graph.successors(pred):
                         if self.graph.nodes[p_succ].get('type') == 'CountryCode':
                             context['owner'] = p_succ
                             
        return context

if __name__ == "__main__":
    kg = DebrisKnowledgeGraph()
    # Test Fengyun 1C Debris (Real ID check needed, just pick one if known or search)
    # FY-1C ID: 25730. Debris example: 29844
    print(kg.query_attribution("25544")) # ISS
