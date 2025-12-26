import requests
import json
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv("../.env")

SPACETRACK_USER = os.getenv("SPACETRACK_USER")
SPACETRACK_PASSWORD = os.getenv("SPACETRACK_PASSWORD")
NORAD_ID = "25544" # ISS

def fetch_history():
    if not SPACETRACK_USER or not SPACETRACK_PASSWORD:
        print("Error: Credentials not found in .env")
        return

    print(f"Logging in as {SPACETRACK_USER}...")
    
    login_url = "https://www.space-track.org/ajaxauth/login"
    base_query = "https://www.space-track.org/basicspacedata/query"
    
    # Range: Last 3 months
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=90)
    
    date_range = f"{start_date.strftime('%Y-%m-%d')}--{end_date.strftime('%Y-%m-%d')}"
    
    # Query: TLEs for ISS in date range
    query = f"{base_query}/class/tle/NORAD_CAT_ID/{NORAD_ID}/EPOCH/{date_range}/orderby/EPOCH asc/format/json"
    
    session = requests.Session()
    
    # login
    resp = session.post(login_url, data={
        "identity": SPACETRACK_USER,
        "password": SPACETRACK_PASSWORD
    })
    
    if resp.status_code != 200:
         print("Login failed")
         return

    print("Fetching TLE history (this may take a moment)...")
    # GET with cookies
    resp = session.get(query)
    
    if resp.status_code != 200:
        print(f"Failed to fetch: {resp.text}")
        return

    tles = resp.json()
    print(f"Downloaded {len(tles)} TLE records.")
    
    with open("iss_history.json", "w") as f:
        json.dump(tles, f, indent=2)
    print("Saved to iss_history.json")

if __name__ == "__main__":
    fetch_history()
