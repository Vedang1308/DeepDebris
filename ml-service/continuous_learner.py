import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import numpy as np
from model.residual_net import ResidualCorrectionNet
from propagator import Propagator

from model.probabilistic_loss import GaussianNLLLoss

class ContinuousLearner:
    def __init__(self, model: ResidualCorrectionNet, model_path="residual_model_probabilistic.pth", learning_rate=0.0005):
        self.model = model
        self.model_path = model_path
        self.propagator = Propagator()
        # Use a very low learning rate for stability (online learning risks drift)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # FEATURE 2: Probabilistic Loss
        self.criterion = GaussianNLLLoss()
        print(f"[ContinuousLearner] Initialized with GNLL Loss & LR={learning_rate}")

    def train_on_single_step(self, prev_tle: dict, current_tle: dict, solar_flux=150.0, kp_index=3.0):
        """
        Performs a SINGLE training step using a pair of TLEs.
        """
        try:
            self.model.train() # Switch to training mode
            
            # ... (Time parsing same as before) ...
            # Helper to ensure datetime
            def to_dt(val):
                if isinstance(val, str):
                    return datetime.fromisoformat(val.replace("Z", "+00:00"))
                return val

            # Current TLE's Epoch is the target time
            target_time = to_dt(current_tle['epoch'])
            
            # 2. Physics Prediction
            pos_physics = self.propagator.get_position(
                prev_tle['line1'], 
                prev_tle['line2'], 
                target_time
            )
            
            # 3. Ground Truth
            pos_truth = self.propagator.get_position(
                current_tle['line1'], 
                current_tle['line2'], 
                target_time
            )
            
            # 4. Calculate Target Residual
            target_correction = np.array(pos_truth) - np.array(pos_physics)
            
            # 5. Prepare Tensors with NORMALIZATION (Critical)
            # Flux / 300, Kp / 10, Pos / 10000
            input_tensor = torch.tensor([[
                solar_flux / 300.0,
                kp_index / 10.0,
                pos_physics[0] / 10000.0,
                pos_physics[1] / 10000.0,
                pos_physics[2] / 10000.0
            ]], dtype=torch.float32)
            
            target_tensor = torch.tensor([[
                target_correction[0],
                target_correction[1],
                target_correction[2]
            ]], dtype=torch.float32)
            
            # 6. Training Step
            self.optimizer.zero_grad()
            
            prediction = self.model(input_tensor) # Outputs 6 dims [Mean, LogVar]
            loss = self.criterion(prediction, target_tensor)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 7. Save and Log
            torch.save(self.model.state_dict(), self.model_path)
            
            loss_val = loss.item()
            print(f"[ContinuousLearner] GNLL Step. Loss: {loss_val:.5f}")
            # Mean is first 3
            mu = prediction.detach().numpy()[0][:3]
            sigma = np.exp(0.5 * prediction.detach().numpy()[0][3:])
            print(f"  > Target: {target_correction}")
            print(f"  > AI Mean: {mu} | Uncert: {sigma}")
            
            return loss_val
            
        except Exception as e:
            print(f"[ContinuousLearner] Error during training step: {e}")
            return None
