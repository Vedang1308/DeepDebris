import sys
import os
import time

# Ensure we can import from ml-service
sys.path.append(os.path.join(os.getcwd(), 'ml-service'))

try:
    from fastapi.testclient import TestClient
    from main import app
except ImportError as e:
    print(f"CRITICAL: Failed to import application: {e}")
    sys.exit(1)

client = TestClient(app)

def test_api_ground_link():
    print("\n--- API TEST 1: Ground Link Endpoint (/contact_status) ---")
    # ISS TLE
    payload = {
        "line1": "1 25544U 98067A   23356.54321689  .00016717  00000+0  30283-3 0  9997",
        "line2": "2 25544  51.6416  21.9684 0005432  35.2163  86.1264 15.49507156430342"
    }
    
    try:
        response = client.post("/contact_status", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"PASS: HTTP 200 OK")
            print(f"      Response: {data}")
            if 'visible' in data:
                print("      Schema Validated.")
        else:
            print(f"FAIL: HTTP {response.status_code} - {response.text}")
    except Exception as e:
        print(f"FAIL: Request Error: {e}")

def test_api_fleet_safety():
    print("\n--- API TEST 2: Fleet Safety Endpoint (/check_fleet_safety) ---")
    # Use a dummy TLE
    payload = {
        "line1": "1 00001U 23001A   23356.50000000  .00000000  00000-0  00000-0 0  9999",
        "line2": "2 00001  53.0000   0.0000 0001000   0.0000   0.0000 15.10000000    1"
    }
    
    try:
        response = client.post("/check_fleet_safety", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"PASS: HTTP 200 OK")
            print(f"      Response: {data}")
            if 'safe' in data:
                print("      Schema Validated.")
        else:
            print(f"FAIL: HTTP {response.status_code} - {response.text}")
    except Exception as e:
        print(f"FAIL: Request Error: {e}")

if __name__ == "__main__":
    print("Starting Integration Tests against 'main.py'...")
    # Allow some time for startup modules (singleton inits)
    test_api_ground_link()
    test_api_fleet_safety()
    print("\nIntegration Suite Complete.")
