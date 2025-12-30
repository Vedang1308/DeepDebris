import requests
import os
import json
from dotenv import load_dotenv

load_dotenv(".env")

USER = os.getenv("SPACETRACK_USER")
PASS = os.getenv("SPACETRACK_PASSWORD")
BASE_URL = "https://www.space-track.org"

session = requests.Session()
login_url = f"{BASE_URL}/ajaxauth/login"
resp = session.post(login_url, data={"identity": USER, "password": PASS})
if resp.status_code != 200:
    print("Login Failed")
    exit(1)

print("Testing CDM Query (ISS)...")
# SAT_1_ID query
query = f"{BASE_URL}/basicspacedata/query/class/cdm_public/SAT_1_ID/25544/orderby/TCA desc/limit/5/emptyresult/show"
resp = session.get(query)
data = resp.json()
print(f"ISS Specific CDMs: {len(data)}")

if not data:
    print("Testing Fallback (Global High Risks)...")
    query = f"{BASE_URL}/basicspacedata/query/class/cdm_public/orderby/TCA desc/limit/5/emptyresult/show"
    resp = session.get(query)
    data = resp.json()
    print(f"Global CDMs: {len(data)}")
    if data:
        print("Sample CDM Keys:", data[0].keys())
