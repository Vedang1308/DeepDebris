import json
import numpy as np
import torch
from propagator import Propagator
from datetime import datetime

DATE_FMT = "%Y-%m-%d %H:%M:%S"

def parse_tle_date(epoch_str):
    # Depending on Space-Track format, usually YYYY-MM-DD HH:MM:SS
    # or TLE epoch. Space-Track JSON 'EPOCH' field is ISO-like usually.
    # Example: "2023-12-01 12:00:00"
    return datetime.strptime(epoch_str, DATE_FMT)

def create_dataset():
    with open("iss_history.json", "r") as f:
        tles = json.load(f)
    
    print(f"Processing {len(tles)} TLEs...")
    propagator = Propagator()
    
    X = []
    Y = []
    
    # Sort by epoch
    tles.sort(key=lambda x: x['EPOCH'])
    
    # Find pairs separated by ~3 days (within small tolerance)
    target_dt = 3.0 # days
    tolerance = 0.5 # hours / 24 = days tolerance
    tolerance_days = 0.1
    
    for i, t1 in enumerate(tles):
        date1 = parse_tle_date(t1['EPOCH'])
        
        for j in range(i+1, len(tles)):
            t2 = tles[j]
            date2 = parse_tle_date(t2['EPOCH'])
            
            diff = (date2 - date1).total_seconds() / 86400.0
            
            if diff > target_dt + tolerance_days:
                break # Too far
            
            if abs(diff - target_dt) < tolerance_days:
                # Found a pair!
                # Input: TLE1, Flux (Mocked for now), Kp (Mocked)
                # Output: Position(TLE2) - SGP4(TLE1 -> TLE2 time)
                
                try:
                    # 1. Physics Prediction
                    pos_pred = propagator.get_position(t1['TLE_LINE1'], t1['TLE_LINE2'], date2)
                    
                    # 2. Truth (Position of TLE2 at its own epoch is ground truth for that moment)
                    # Note: TLE2's 'position' at its epoch is just SGP4(TLE2, date2).
                    # Since TLE2 is a FRESH fit from radar data, its SGP4(0) is very close to truth.
                    pos_truth = propagator.get_position(t2['TLE_LINE1'], t2['TLE_LINE2'], date2)
                    
                    residual = pos_truth - pos_pred
                    
                    # Features: [Flux, Kp, X, Y, Z]
                    # Mocking Flux/Kp as random for this demo since we don't have historical space weather loaded yet
                    flux = 150.0 + np.random.randn() * 20
                    kp = 3.0 + np.random.randn() * 1
                    
                    # Normalized features roughly
                    input_feat = [flux, kp, pos_pred[0], pos_pred[1], pos_pred[2]]
                    
                    X.append(input_feat)
                    Y.append(residual)
                    
                except Exception as e:
                    # SGP4 error
                    continue
    
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    
    print(f"Generated {len(X)} samples.")
    torch.save(torch.tensor(X), "X_train.pt")
    torch.save(torch.tensor(Y), "y_train.pt")
    print("Saved training tensors.")

if __name__ == "__main__":
    create_dataset()
