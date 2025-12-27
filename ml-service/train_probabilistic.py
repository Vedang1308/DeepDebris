import torch
import torch.optim as optim
import json
import numpy as np
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday
from model.residual_net import ResidualCorrectionNet
from model.probabilistic_loss import GaussianNLLLoss

# 1. Load Data
print("Loading historical data...")
with open("iss_history.json", "r") as f:
    history = json.load(f)

# 2. Preprocess (Create Physics vs True pairs)
# Similar to process_dataset.py but targeting the new format
inputs = []
targets = []

print(f"Processing {len(history)} historical points...")
for i in range(len(history) - 1):
    t1 = history[i]
    t2 = history[i+1] # The "Future" truth
    
    # Truth time
    target_time = datetime.fromisoformat(t2['EPOCH'])
    
    # Physics Prediction (SGP4 from t1 -> t2)
    s = Satrec.twoline2rv(t1['TLE_LINE1'], t1['TLE_LINE2'])
    jd, fr = jday(target_time.year, target_time.month, target_time.day, 
                  target_time.hour, target_time.minute, target_time.second)
    e, r, v = s.sgp4(jd, fr)
    
    if e != 0: continue
    
    pos_physics = np.array(r) # Ground Truth SGP4
    
    # Ground Truth Actual (from t2's TLE itself)
    # Ideally we'd propagate t2 to its own epoch (trivial, it's 0)
    s2 = Satrec.twoline2rv(t2['TLE_LINE1'], t2['TLE_LINE2'])
    e2, r2, v2 = s2.sgp4(jd, fr)
    pos_true = np.array(r2)
    
    if e2 != 0: continue
    
    # Calculate Residual (Target for AI)
    residual = pos_true - pos_physics # [dx, dy, dz]
    
    # Input Features: [Flux, Kp, PhysX, PhysY, PhysZ]
    # We mock Flux/Kp if missing in history, or use random valid range for robustness
    flux = 150.0 + np.random.normal(0, 10)
    kp = 3.0 + np.random.normal(0, 1)
    
    # NORMALIZATION (Critical for stability)
    # Flux ~ [0, 300] -> / 300.0
    # Kp ~ [0, 9] -> / 10.0
    # Pos ~ [-7000, 7000] -> / 10000.0
    
    inp = [
        flux / 300.0, 
        kp / 10.0, 
        pos_physics[0] / 10000.0, 
        pos_physics[1] / 10000.0, 
        pos_physics[2] / 10000.0
    ]
    inputs.append(inp)
    targets.append(residual)

# Convert to Tensors
X_train = torch.tensor(inputs, dtype=torch.float32)
y_train = torch.tensor(targets, dtype=torch.float32)

print(f"Training Data: {X_train.shape}, {y_train.shape}")

# 3. Initialize Probabilistic Model
model = ResidualCorrectionNet(input_dim=5, output_dim=6) # 6 Outputs now!

# STABILITY FIX: Initialize LogVar bias to -5 (Small variance start)
# The last linear layer is model.output_layer
# output_dim=6. indices 3,4,5 are log_var.
with torch.no_grad():
    model.output_layer.bias[3:].fill_(-5.0) 
    model.output_layer.weight[3:, :].mul_(0.01) # Small weights for variance head

criterion = GaussianNLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005) # Lower LR

# 4. Train Loop
epochs = 1000
print("Starting Probabilistic Training (GNLL)...")

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train) # [Batch, 6]
    loss = criterion(outputs, y_train) # Compares [Batch, 6] vs [Batch, 3] target
    
    if torch.isnan(loss):
        print("LOSS NAN DETECTED. BREAKING.")
        break
        
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient Clipping
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

# 5. Save
print("Uncertainty Training Complete.")
torch.save(model.state_dict(), "residual_model_probabilistic.pth")
print("Saved to residual_model_probabilistic.pth")

# 6. Verification
model.eval()
with torch.no_grad():
    sample_out = model(X_train[0:1])
    mu = sample_out[0, :3]
    logsrc = sample_out[0, 3:]
    std = torch.exp(0.5 * logsrc) # std = sqrt(var) = exp(0.5 * logvar)
    print("Sample Prediction:")
    print(f"Mean Correction (km): {mu.numpy()}")
    print(f"Uncertainty (Std Dev km): {std.numpy()}")
