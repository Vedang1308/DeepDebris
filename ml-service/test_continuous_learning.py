import json
import torch
from model.residual_net import ResidualCorrectionNet
from continuous_learner import ContinuousLearner
import os

def test_learning():
    print("Testing Continuous Learner...")
    
    # 1. Load History
    if not os.path.exists("iss_history.json"):
        print("Error: iss_history.json not found. Run fetch_history.py first.")
        return

    with open("iss_history.json", "r") as f:
        history = json.load(f)
        
    if len(history) < 2:
        print("Not enough history to test.")
        return
        
    # Sort by EPOCH just in case
    history.sort(key=lambda x: x['EPOCH'])
    
    # Pick a transition
    prev = history[0]
    curr = history[1]
    
    print(f"Previous Epoch: {prev['EPOCH']}")
    print(f"Current  Epoch: {curr['EPOCH']}")
    
    # 2. Init Model & Learner
    model = ResidualCorrectionNet()
    learner = ContinuousLearner(model, model_path="test_model.pth")
    
    # 3. Run Step
    loss = learner.train_on_single_step(
        {
            "line1": prev["TLE_LINE1"],
            "line2": prev["TLE_LINE2"],
            "epoch": prev["EPOCH"]
        },
        {
            "line1": curr["TLE_LINE1"],
            "line2": curr["TLE_LINE2"],
            "epoch": curr["EPOCH"]
        }
    )
    
    if loss is not None:
        print(f"SUCCESS: Training step completed. Loss: {loss}")
    else:
        print("FAILURE: Training step returned None.")

if __name__ == "__main__":
    test_learning()
