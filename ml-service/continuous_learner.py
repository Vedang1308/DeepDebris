import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import numpy as np
from model.residual_net import ResidualCorrectionNet
from propagator import Propagator

class ContinuousLearner:
    def __init__(self, model: ResidualCorrectionNet, model_path="residual_model.pth", learning_rate=0.001):
        self.model = model
        self.model_path = model_path
        self.propagator = Propagator()
        # Use a very low learning rate for stability (online learning risks drift)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        print(f"[ContinuousLearner] Initialized with LR={learning_rate}")

    def train_on_single_step(self, prev_tle: dict, current_tle: dict, solar_flux=150.0, kp_index=3.0):
        """
        Performs a SINGLE training step using a pair of TLEs.
        
        Logic:
        1. We have a 'prev_tle' (Past).
        2. We have a 'current_tle' (The Future relative to Past).
        3. Propagate 'prev_tle' to the time of 'current_tle' -> Physics Prediction.
        4. Calculate 'current_tle' position at its own time -> Ground Truth.
        5. Difference is the ERROR that the Physics model made.
        6. Train our AI net to predict this Error.
        """
        try:
            self.model.train() # Switch to training mode
            
            # 1. Parse timestamps
            # Space-Track EPOCH is often isoformat or close to it. 
            # Assuming these are dicts from our TLE_CACHE or Space-Track JSON
            # Format: "2025-12-26T12:00:00" or datetime objects
            
            # Helper to ensure datetime
            def to_dt(val):
                if isinstance(val, str):
                    return datetime.fromisoformat(val.replace("Z", "+00:00"))
                return val

            # Current TLE's Epoch is the target time
            target_time = to_dt(current_tle['epoch'])
            
            # 2. Physics Prediction (The "Wrong" Guess)
            # Propagate the OLD TLE forward to the NEW time
            pos_physics = self.propagator.get_position(
                prev_tle['line1'], 
                prev_tle['line2'], 
                target_time
            )
            
            # 3. Ground Truth (The "Actual" Position)
            # The new TLE *at its own epoch* is the most accurate definition of "where it is now"
            pos_truth = self.propagator.get_position(
                current_tle['line1'], 
                current_tle['line2'], 
                target_time
            )
            
            # 4. Calculate Target Residual (The Error to learn)
            # Error = Truth - Physics
            # If Physics says 100, Truth is 105, Error is +5.
            # We want model to output +5.
            target_correction = np.array(pos_truth) - np.array(pos_physics)
            
            # 5. Prepare Tensors
            # Input: [Flux, Kp, Physics_X, Physics_Y, Physics_Z]
            input_tensor = torch.tensor([[
                solar_flux,
                kp_index,
                pos_physics[0],
                pos_physics[1],
                pos_physics[2]
            ]], dtype=torch.float32)
            
            target_tensor = torch.tensor([[
                target_correction[0],
                target_correction[1],
                target_correction[2]
            ]], dtype=torch.float32)
            
            # 6. Training Step
            self.optimizer.zero_grad()
            
            prediction = self.model(input_tensor)
            loss = self.criterion(prediction, target_tensor)
            
            loss.backward()
            self.optimizer.step()
            
            # 7. Save and Log
            torch.save(self.model.state_dict(), self.model_path)
            
            loss_val = loss.item()
            print(f"[ContinuousLearner] Training Step Complete. Loss: {loss_val:.5f}")
            print(f"  > Correction Target: {target_correction}")
            print(f"  > AI Prediction:     {prediction.detach().numpy()[0]}")
            
            return loss_val
            
        except Exception as e:
            print(f"[ContinuousLearner] Error during training step: {e}")
            return None
