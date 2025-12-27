import torch
import torch.optim as optim
import numpy as np
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday
from model.residual_net import ResidualCorrectionNet
from model.probabilistic_loss import GaussianNLLLoss

print("Generating realistic training data with atmospheric drag simulation...")

# Simulate realistic orbital perturbations
def simulate_atmospheric_drag(pos, vel, flux, kp, dt_hours):
    """
    Simulate atmospheric drag effect on position.
    Higher flux/kp = more drag = larger corrections
    """
    altitude_km = np.linalg.norm(pos) - 6371  # Earth radius
    
    # Atmospheric density model (simplified)
    # Density increases exponentially as altitude decreases
    scale_height = 50  # km
    rho_base = 1e-12  # kg/m^3 at reference altitude
    rho = rho_base * np.exp(-(altitude_km - 400) / scale_height)
    
    # Solar activity multiplier
    solar_factor = 1.0 + (flux - 150) / 150.0  # Flux effect
    geomag_factor = 1.0 + (kp - 3) / 6.0  # Kp effect
    rho *= solar_factor * geomag_factor
    
    # Drag force direction (opposite to velocity)
    v_mag = np.linalg.norm(vel)
    if v_mag == 0:
        return np.zeros(3)
    
    drag_dir = -vel / v_mag
    
    # Drag acceleration (simplified)
    # F = 0.5 * Cd * A * rho * v^2
    # a = F / m
    Cd = 2.2  # Drag coefficient
    A_over_m = 0.01  # Area-to-mass ratio (m^2/kg)
    
    drag_accel = 0.5 * Cd * A_over_m * rho * v_mag**2 * drag_dir
    
    # Position change due to drag over time
    # Δpos ≈ 0.5 * a * t^2 (simplified)
    dt_seconds = dt_hours * 3600
    pos_correction = 0.5 * drag_accel * dt_seconds**2
    
    return pos_correction

# Generate synthetic training data
inputs = []
targets = []

# Use ISS TLE as base
tle1 = "1 25544U 98067A   25361.54173603  .00013978  00000+0  25382-3 0  9998"
tle2 = "2 25544  51.6320  70.2581 0003231 308.5588  51.5099 15.49844361545105"

print("Generating 2000 training samples...")
np.random.seed(42)

for i in range(2000):
    # Random time offset (0 to 48 hours in the future)
    dt_hours = np.random.uniform(0.1, 48.0)
    
    # Random space weather conditions
    flux = np.random.uniform(70, 300)  # Solar flux
    kp = np.random.uniform(0, 9)  # Geomagnetic index
    
    # Base time
    base_time = datetime(2025, 12, 27, 7, 0, 0)
    target_time = base_time + timedelta(hours=dt_hours)
    
    # Physics prediction (SGP4 only)
    sat = Satrec.twoline2rv(tle1, tle2)
    jd, fr = jday(target_time.year, target_time.month, target_time.day,
                  target_time.hour, target_time.minute, target_time.second)
    e, r, v = sat.sgp4(jd, fr)
    
    if e != 0:
        continue
    
    pos_physics = np.array(r)
    vel = np.array(v)
    
    # Simulate "true" position with atmospheric drag
    drag_correction = simulate_atmospheric_drag(pos_physics, vel, flux, kp, dt_hours)
    
    # Add some random noise to make it more realistic
    noise = np.random.normal(0, 2.0, 3)  # 2km standard deviation
    
    # True residual = drag effect + noise
    residual = drag_correction + noise
    
    # Input features (normalized)
    inp = [
        flux / 300.0,
        kp / 10.0,
        pos_physics[0] / 10000.0,
        pos_physics[1] / 10000.0,
        pos_physics[2] / 10000.0
    ]
    
    inputs.append(inp)
    targets.append(residual)

# Convert to tensors
X_train = torch.tensor(inputs, dtype=torch.float32)
y_train = torch.tensor(targets, dtype=torch.float32)

print(f"Training Data Shape: {X_train.shape}, {y_train.shape}")
print(f"Residual Statistics:")
print(f"  Mean: {y_train.mean(dim=0).numpy()} km")
print(f"  Std:  {y_train.std(dim=0).numpy()} km")
print(f"  Max:  {y_train.max(dim=0)[0].numpy()} km")

# Initialize model
model = ResidualCorrectionNet(input_dim=5, output_dim=6)

# Initialize variance head to small values
with torch.no_grad():
    model.output_layer.bias[3:].fill_(-5.0)
    model.output_layer.weight[3:, :].mul_(0.01)

criterion = GaussianNLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 2000
print(f"\nTraining for {epochs} epochs...")

best_loss = float('inf')
patience = 200
patience_counter = 0

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    if torch.isnan(loss):
        print("NaN detected! Stopping.")
        break
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Early stopping
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "residual_model_probabilistic.pth")
print("\nModel saved to residual_model_probabilistic.pth")

# Test predictions
model.eval()
with torch.no_grad():
    # Test with different flux/kp scenarios
    test_cases = [
        ("Quiet", 70, 1),
        ("Normal", 150, 3),
        ("Storm", 300, 7)
    ]
    
    print("\nSample Predictions:")
    for name, flux, kp in test_cases:
        test_input = torch.tensor([[
            flux / 300.0,
            kp / 10.0,
            0.0, 0.0, 0.6  # Sample position
        ]], dtype=torch.float32)
        
        output = model(test_input)
        mu = output[0, :3].numpy()
        logvar = output[0, 3:].numpy()
        std = np.sqrt(np.exp(logvar))
        
        print(f"\n{name} Conditions (Flux={flux}, Kp={kp}):")
        print(f"  Correction: [{mu[0]:.2f}, {mu[1]:.2f}, {mu[2]:.2f}] km")
        print(f"  Uncertainty: [{std[0]:.2f}, {std[1]:.2f}, {std[2]:.2f}] km")

print("\n✅ Training complete! Restart the server to load the new model.")
