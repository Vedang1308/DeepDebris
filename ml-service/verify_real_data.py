
import os
import requests
import json
import sys
from dotenv import load_dotenv

# Load Environment from Root
# current dir is ml-service, root is ..
load_dotenv("../.env")

BASE_URL = "http://localhost:8000"

def check(name, success, message=""):
    if success:
        print(f"✅ {name}: PASS {message}")
    else:
        print(f"❌ {name}: FAIL {message}")
        # sys.exit(1) # Don't exit, just report

def verify_credentials():
    u = os.getenv("SPACETRACK_USER")
    p = os.getenv("SPACETRACK_PASSWORD")
    if u and p:
        check("Credentials", True, "(Found SPACETRACK_USER)")
    else:
        check("Credentials", False, "(Missing SPACETRACK_USER/PASSWORD)")

def verify_risks_endpoint():
    try:
        resp = requests.get(f"{BASE_URL}/risks")
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                 check("/risks", True, f"(Found {len(data)} items)")
                 # Verify structure
                 if "tca" in data[0] and "id" in data[0]:
                     check("/risks Structure", True)
                 else:
                     check("/risks Structure", False, f"Keys: {data[0].keys()}")
            else:
                 check("/risks", False, "(Returned empty list - Logic might be falling back to empty)")
        else:
            check("/risks", False, f"(Status {resp.status_code})")
    except Exception as e:
        check("/risks", False, f"(Exception: {e})")

def verify_chat_endpoint():
    try:
        # Ask about a specific satellite to trigger Knowledge Graph
        query = "Who owns object 25544?"
        resp = requests.post(f"{BASE_URL}/chat", json={"query": query})
        if resp.status_code == 200:
            ans = resp.json().get("response", "")
            if "ISS" in ans or "International Space Station" in ans:
                check("/chat (Knowledge Graph)", True, "(Identified ISS correctly)")
            else:
                check("/chat (Knowledge Graph)", False, f"(Answer: {ans})")
        else:
            check("/chat", False, f"(Status {resp.status_code})")
    except Exception as e:
        check("/chat", False, f"(Exception: {e})")

def verify_diplomat_endpoint():
    try:
        context = "Collision imminent. Fuel critical."
        resp = requests.post(f"{BASE_URL}/negotiate", json={"context": context})
        if resp.status_code == 200:
            transcript = resp.json().get("transcript", [])
            if len(transcript) >= 3:
                check("/negotiate (Diplomat)", True, f"(Generated {len(transcript)} messages)")
                # Check for dynamic content (Agent names)
                agents = [m['sender'] for m in transcript]
                if "Agent A" in agents and "Agent B" in agents:
                    check("/negotiate Structure", True)
                else:
                    check("/negotiate Structure", False, f"(Agents found: {agents})")
            else:
                 check("/negotiate", False, "(Transcript too short)")
        else:
            check("/negotiate", False, f"(Status {resp.status_code})")
    except Exception as e:
        check("/negotiate", False, f"(Exception: {e})")

if __name__ == "__main__":
    print("--- DeepDebris 4.0 Real Data Verification ---")
    verify_credentials()
    verify_risks_endpoint()
    verify_chat_endpoint()
    verify_diplomat_endpoint()
