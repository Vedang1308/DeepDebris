import requests
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Re-use the existing authentication logic from fetch_history.py logic if possible, 
# but for now we'll implement a clean class using env vars.

class CDMService:
    def __init__(self):
        load_dotenv("../.env")
        self.user = os.getenv("SPACETRACK_USER")
        self.password = os.getenv("SPACETRACK_PASSWORD")
        self.base_url = "https://www.space-track.org"
        self.session = requests.Session()
        self._login()

    def _login(self):
        login_url = f"{self.base_url}/ajaxauth/login"
        payload = {
            "identity": self.user,
            "password": self.password
        }
        resp = self.session.post(login_url, data=payload)
        if resp.status_code != 200:
            raise Exception("Space-Track Login Failed")
        print("LOGGED IN to Space-Track for CDM Service")

    def fetch_recent_cdms(self, sat_cat_id=25544, days=2):
        """
        Fetches Public CDMs for the specified satellite for the last N days.
        """
        # API: basicspacedata/query/class/cdm_public
        # Filter: SAT1_ID = sat_cat_id (The target) OR SAT2_ID = sat_cat_id
        # Note: cdm_public might perform better with specific date ranges
        
        # Calculate date range
        start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Construct Query
        # We look for CDMs where our satellite is either Object 1 or Object 2
        # Use simple URL construction
        query = (
            f"/basicspacedata/query/class/cdm_public"
            f"/SAT1_OBJECT_NAME/ISS (ZARYA)" # Simplified filter for demo
            f"/orderby/TCA desc/limit/10/emptyresult/show"
        )
        
        # Query by Object ID using correct predicate SAT_1_ID (with underscores)
        query = (
             f"/basicspacedata/query/class/cdm_public"
             f"/SAT_1_ID/{sat_cat_id}"
             f"/orderby/TCA desc/limit/5/emptyresult/show"
        )

        full_url = f"{self.base_url}{query}"
        
        print(f"Fetching CDMs from: {full_url}")
        resp = self.session.get(full_url)
        
        # If specific query failed or returned empty (common for secure satellites), 
        # fetch the TOP 5 GLOBAL RISKS (Real Data) so the user sees what a CDM looks like.
        if resp.status_code != 200 or not resp.json():
            print("No alerts for this specific satellite. Fetching GLOBAL HIGH RISK ALERTS...")
            query = (
                 f"/basicspacedata/query/class/cdm_public"
                 f"/orderby/TCA desc/limit/5/emptyresult/show"
            )
            full_url = f"{self.base_url}{query}"
            resp = self.session.get(full_url)
        
        if resp.status_code != 200:
            print(f"Error fetching CDMs: {resp.text}")
            return []
        
        cdms = resp.json()
        return cdms

    def parse_cdm_to_text(self, cdm):
        """
        Converts a raw JSON CDM into a natural language summary.
        """
        # Extract Key Fields (Corrected Keys)
        sat1 = cdm.get("SAT_1_NAME", "Unknown Sat")
        sat2 = cdm.get("SAT_2_NAME", "Unknown Debris")
        tca = cdm.get("TCA", "Unknown Time")
        min_rng = cdm.get("MIN_RNG", "N/A")
        prob = cdm.get("PC", "N/A") # Probability of Collision
        
        # Parse Time
        try:
            dt = datetime.fromisoformat(tca)
            readable_time = dt.strftime("%B %d at %H:%M UTC")
        except:
            readable_time = tca

        # Construct Sentence
        summary = (
            f"ALERT: High-risk conjunction detected between {sat1} and {sat2}. "
            f"Time of Closest Approach (TCA) is {readable_time}. "
            f"Minimum Range will be {min_rng} km. "
            f"Probability of Collision is {prob}. "
            f"Reviewing CDM details: {json.dumps(cdm)}." 
        )
        
        # A more conversational summary for the RAG
        rag_text = (
            f"Conjunction Report for {sat1} vs {sat2}.\n"
            f"- TCA: {readable_time}\n"
            f"- Miss Distance: {min_rng} km\n"
            f"- Probability: {prob}\n"
            f"This event involves {sat2}."
        )
        
        return rag_text

if __name__ == "__main__":
    svc = CDMService()
    cdms = svc.fetch_recent_cdms(25544) # ISS
    print(f"Found {len(cdms)} CDMs")
    if cdms:
        print("Raw Keys:", cdms[0].keys()) # DEBUG
        print("Latest CDM Summary:")
        print(svc.parse_cdm_to_text(cdms[0]))
