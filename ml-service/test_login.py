import requests
import os
import json
from dotenv import load_dotenv

# Try loading from typical locations
load_dotenv("../.env") # If running from ml-service/
load_dotenv(".env")    # If running from DeepDebris/

USER = os.getenv("SPACETRACK_USER")
PASS = os.getenv("SPACETRACK_PASSWORD")

print(f"Loaded Identity: {USER}")
# Mask password
masked_pass = "*" * len(PASS) if PASS else "None"
print(f"Loaded Password: {masked_pass}")

if not USER or not PASS:
    print("❌ Credentials Missing!")
    exit(1)

session = requests.Session()
login_url = "https://www.space-track.org/ajaxauth/login"

print("Attempting Login...")
try:
    resp = session.post(login_url, data={"identity": USER, "password": PASS})
    
    if resp.status_code == 200:
        print("✅ Login Success (HTTP 200)")
        # Verify Session by fetching a TLE
        print("Fetching Test TLE (ISS)...")
        # DEPRECATED: query = "https://www.space-track.org/basicspacedata/query/class/tle_latest/NORAD_CAT_ID/25544/orderby/ORDINAL desc/format/json"
        # NEW: use class/gp which returns latest SGP4 TLE by default
        query = "https://www.space-track.org/basicspacedata/query/class/gp/NORAD_CAT_ID/25544/orderby/EPOCH desc/format/json"
        resp = session.get(query)
        if resp.status_code == 200:
            data = resp.json()
            if len(data) > 0:
                print(f"✅ Data Fetch Success: {data[0]['OBJECT_NAME']}")
            else:
                print("⚠ Login OK but Data Empty")
                print(resp.text)
        else:
             print(f"❌ Data Fetch Failed: {resp.status_code}")
             print(resp.text)
    else:
        print(f"❌ Login Failed: {resp.status_code}")
        print(resp.text)

except Exception as e:
    print(f"❌ Exception: {e}")
