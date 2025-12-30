
import sys
import os
import json

# Add to path
sys.path.append(os.getcwd())

# Mock Env vars to ensure they are unset (simulating no credentials)
if "SPACETRACK_USER" in os.environ: del os.environ["SPACETRACK_USER"]
if "SPACETRACK_PASSWORD" in os.environ: del os.environ["SPACETRACK_PASSWORD"]

try:
    from main import get_latest_tle, TLE_CACHE
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

def test_tle_fallback():
    print("Testing TLE Fallback Logic...")
    
    # 1. Test ISS (Should exist in fallback)
    print("\n[Test 1] Fetching ISS (25544)...")
    try:
        tle = get_latest_tle(25544)
        print(f"Success! Source: {tle.get('source')}")
        if tle.get('source') == "FALLBACK-DEMO":
            print("PASS: Correctly used fallback data.")
        else:
            print(f"FAIL: Expected FALLBACK-DEMO, got {tle.get('source')}")
    except Exception as e:
        print(f"FAIL: Exception raised: {e}")

    # 2. Test Unknown Satellite (Should 404)
    print("\n[Test 2] Fetching Unknown ID (99999)...")
    try:
        get_latest_tle(99999)
        print("FAIL: Should have raised 404.")
    except Exception as e:
        if "404" in str(e):
            print("PASS: Correctly raised 404 for unknown ID.")
        else:
            print(f"FAIL: Unexpected exception: {e}")

if __name__ == "__main__":
    test_tle_fallback()
