
import sys
import os
import torch
import numpy as np
from datetime import datetime, timedelta
from pydantic import BaseModel

# Add current directory to path to import main
sys.path.append(os.getcwd())

from main import app, predict_batch, BatchRequest, model

# Mock Request
def test_prediction(flux, kp, desc):
    print(f"\n--- Testing {desc} (Flux={flux}, Kp={kp}) ---")
    
    # HST TLE (Example)
    line1 = "1 20580U 90037B   25363.51060729  .00009987  00000+0  34412-3 0  9992"
    line2 = "2 20580  28.4653 158.3162 0002577  64.1454 295.9407 15.28271296762561"
    
    start_time = datetime.utcnow().isoformat()
    
    req = BatchRequest(
        line1=line1,
        line2=line2,
        start_time=start_time,
        minutes_duration=120, # 2 hours
        step_minutes=5,
        solar_flux=float(flux),
        kp_index=float(kp)
    )
    
    try:
        results = predict_batch(req)
        
        # Analyze last point
        last_pt = results[-1]
        
        # Calculate deviation from physics
        phys_pos = np.array([last_pt['physics_x'], last_pt['physics_y'], last_pt['physics_z']])
        ai_pos = np.array([last_pt['x'], last_pt['y'], last_pt['z']])
        
        diff = np.linalg.norm(ai_pos - phys_pos)
        
        print(f"Time: {last_pt['ts']}")
        print(f"Physics Pos (km): {phys_pos}")
        print(f"AI Pos (km):      {ai_pos}")
        print(f"Deviation:        {diff:.4f} km")
        
        return diff
    except Exception as e:
        print(f"Error: {e}")
        return 0.0

if __name__ == "__main__":
    print("Verifying Space Weather AI Divergence...")
    
    # 1. Normal Weather
    diff_normal = test_prediction(100, 2, "NORMAL Weather")
    
    # 2. Storm Weather
    diff_storm = test_prediction(300, 9, "STORM Weather")
    
    print("\n--- Summary ---")
    print(f"Normal Deviation: {diff_normal:.4f} km")
    print(f"Storm Deviation:  {diff_storm:.4f} km")
    
    if diff_storm > diff_normal * 1.5: # Expecting significant increase
        print("SUCCESS: Storm caused significant path divergence compared to normal.")
    else:
        print("FAILURE: Storm divergence was not significantly different from normal.")
