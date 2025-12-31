import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'ml-service'))

from pass_scheduler import PassScheduler
from anomaly_detector import PhysicsValidator
from diplomat.fleet_manager import FleetManager
from datetime import datetime, timezone

def test_ground_link():
    print("\n--- TEST 1: Ground Link (PassScheduler) ---")
    sched = PassScheduler() # Maui
    # ISS TLE (High inclination, should pass eventually)
    l1 = "1 25544U 98067A   23356.54321689  .00016717  00000+0  30283-3 0  9997"
    l2 = "2 25544  51.6416  21.9684 0005432  35.2163  86.1264 15.49507156430342"
    
    nxt = sched.get_next_pass(l1, l2)
    if nxt:
        print(f"PASS: Found next contact window: {nxt['aos']} to {nxt['los']}")
        print(f"      Duration: {nxt['duration']:.1f}s, Max El: {nxt['max_el']:.1f} deg")
    else:
        print("FAIL: No pass found (unlikely for ISS/Maui)")

def test_cyber_security():
    print("\n--- TEST 2: Cyber-Physical Security (SpyHunter) ---")
    validator = PhysicsValidator()
    
    # Baseline TLE
    t1 = {
        'line2': "2 25544  51.6416  21.9684 0005432  35.2163  86.1264 15.49507156430342"
    }
    
    # Case A: Good Update (Small change)
    t2_good = {
        'line2': "2 25544  51.6420  22.0000 0005432  35.3000  86.2000 15.49507156430342"
    }
    res_a = validator.check_consistency(t1, t2_good)
    print(f"Case A (Valid): {res_a['valid']} - {res_a['reason']}")
    
    # Case B: SPOOF ATTACK (10 deg Plane Change instant)
    t2_bad = {
        'line2': "2 25544  61.6416  21.9684 0005432  35.2163  86.1264 15.49507156430342"
    }
    res_b = validator.check_consistency(t1, t2_bad)
    if not res_b['valid']:
        print(f"PASS: Blocked Spoofing Attempt: {res_b['reason']}")
    else:
        print(f"FAIL: Physics Firewall breached! {res_b}")

def test_fratricide():
    print("\n--- TEST 3: Fratricide Prevention (FleetManager) ---")
    fm = FleetManager(size=10)
    
    
    # Generate a target that collides with Sat #10000 (The first one)
    # We must match the EPOCH of the fleet, otherwise they are years apart.
    friend_0 = fm.fleet[0]
    l1_bad = friend_0['line1'].replace("10000U", "99999U") # Same orbit, diff ID
    l2_bad = friend_0['line2'].replace("10000 ", "99999 ")
    
    res = fm.check_safety(l1_bad, l2_bad, minutes_check=10)
    if not res['safe']:
        print(f"PASS: Collision Detected: {res['reason']}")
    else:
        print("FAIL: Failed to detect collision with Fleet.")

if __name__ == "__main__":
    test_ground_link()
    test_cyber_security()
    test_fratricide()
