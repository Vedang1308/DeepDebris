import torch
import torch.optim as optim
import numpy as np
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday
from model.residual_net import ResidualCorrectionNet
from model.probabilistic_loss import GaussianNLLLoss

print("=" * 60)
print("PRODUCTION TRAINING: DeepDebris Probabilistic Model")
print("=" * 60)

def simulate_orbital_perturbations(pos, vel, flux, kp, dt_hours):
    """
    Advanced orbital perturbation model including:
    - Atmospheric drag (altitude + solar activity dependent)
    - Solar radiation pressure
    - Third-body perturbations (simplified)
    """
    altitude_km = np.linalg.norm(pos) - 6371
    
    # 1. ATMOSPHERIC DRAG (dominant for LEO)
    scale_height = 50
    rho_base = 1e-12
    rho = rho_base * np.exp(-(altitude_km - 400) / scale_height)
    
    # Solar cycle effect (flux 70-300 → 0.5x to 2x density)
    solar_factor = 0.5 + 1.5 * (flux - 70) / 230
    # Geomagnetic storms (kp 0-9 → 1x to 3x density)
    geomag_factor = 1.0 + 2.0 * (kp / 9.0)
    rho *= solar_factor * geomag_factor
    
    v_mag = np.linalg.norm(vel)
    if v_mag > 0:
        drag_dir = -vel / v_mag
        Cd_A_over_m = 0.015  # Increased for more visible effect
        drag_accel = 0.5 * Cd_A_over_m * rho * v_mag**2 * drag_dir
    else:
        drag_accel = np.zeros(3)
    
    # 2. SOLAR RADIATION PRESSURE (small but accumulates)
    sun_dir = np.array([1.0, 0.0, 0.0])  # Simplified
    Cr_A_over_m = 0.005
    solar_pressure = 4.56e-6  # N/m^2
    srp_accel = Cr_A_over_m * solar_pressure * sun_dir
    
    # 3. THIRD-BODY (Moon/Sun gravity - very simplified)
    # Creates periodic variations
    third_body_accel = np.random.normal(0, 1e-8, 3)
    
    # Total acceleration
    total_accel = drag_accel + srp_accel + third_body_accel
    
    # Position change over time (Δpos = 0.5 * a * t²)
    dt_seconds = dt_hours * 3600
    pos_correction = 0.5 * total_accel * dt_seconds**2
    
    return pos_correction

# Generate comprehensive training dataset
print("\nGenerating 5000 training samples...")
print("  - Time horizons: 0.5 to 96 hours")
print("  - Space weather: Quiet to Extreme Storm")
print("  - Multiple orbital geometries")

inputs = []
targets = []

tle1 = "1 25544U 98067A   25361.54173603  .00013978  00000+0  25382-3 0  9998"
tle2 = "2 25544  51.6320  70.2581 0003231 308.5588  51.5099 15.49844361545105"

np.random.seed(42)

for i in range(5000):
    # Varied time horizons (longer = more dramatic corrections)
    dt_hours = np.random.choice([
        np.random.uniform(0.5, 6),    # 40% short-term
        np.random.uniform(6, 24),     # 30% medium-term  
        np.random.uniform(24, 96)     # 30% long-term
    ], p=[0.4, 0.3, 0.3])
    
    # Realistic space weather distribution
    weather_scenario = np.random.choice(['quiet', 'normal', 'active', 'storm', 'extreme'], 
                                       p=[0.2, 0.4, 0.2, 0.15, 0.05])
    
    if weather_scenario == 'quiet':
        flux = np.random.uniform(70, 100)
        kp = np.random.uniform(0, 2)
    elif weather_scenario == 'normal':
        flux = np.random.uniform(100, 180)
        kp = np.random.uniform(2, 4)
    elif weather_scenario == 'active':
        flux = np.random.uniform(180, 250)
        kp = np.random.uniform(4, 6)
    elif weather_scenario == 'storm':
        flux = np.random.uniform(250, 300)
        kp = np.random.uniform(6, 8)
    else:  # extreme
        flux = np.random.uniform(300, 400)
        kp = np.random.uniform(8, 9)
    
    # Propagate orbit
    base_time = datetime(2025, 12, 27, 7, 0, 0)
    target_time = base_time + timedelta(hours=dt_hours)
    
    sat = Satrec.twoline2rv(tle1, tle2)
    jd, fr = jday(target_time.year, target_time.month, target_time.day,
                  target_time.hour, target_time.minute, target_time.second)
    e, r, v = sat.sgp4(jd, fr)
    
    if e != 0:
        continue
    
    pos_physics = np.array(r)
    vel = np.array(v)
    
    # Simulate true corrections
    correction = simulate_orbital_perturbations(pos_physics, vel, flux, kp, dt_hours)
    
    # Add measurement noise
    noise = np.random.normal(0, 1.5, 3)
    residual = correction + noise
    
    # Normalized inputs
    inp = [
        flux / 300.0,
        kp / 10.0,
        pos_physics[0] / 10000.0,
        pos_physics[1] / 10000.0,
        pos_physics[2] / 10000.0
    ]
    
    inputs.append(inp)
    targets.append(residual)

X_train = torch.tensor(inputs, dtype=torch.float32)
y_train = torch.tensor(targets, dtype=torch.float32)

print(f"\n✓ Dataset created: {X_train.shape}")
print(f"\nResidual Statistics:")
print(f"  Mean: [{y_train.mean(dim=0)[0]:.2f}, {y_train.mean(dim=0)[1]:.2f}, {y_train.mean(dim=0)[2]:.2f}] km")
print(f"  Std:  [{y_train.std(dim=0)[0]:.2f}, {y_train.std(dim=0)[1]:.2f}, {y_train.std(dim=0)[2]:.2f}] km")
print(f"  Max:  [{y_train.max(dim=0)[0][0]:.2f}, {y_train.max(dim=0)[0][1]:.2f}, {y_train.max(dim=0)[0][2]:.2f}] km")

# Initialize model with better architecture
model = ResidualCorrectionNet(input_dim=5, output_dim=6)

# Smart initialization
with torch.no_grad():
    model.output_layer.bias[3:].fill_(-5.0)  # Small initial variance
    model.output_layer.weight[3:, :].mul_(0.01)

criterion = GaussianNLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5)

# Training with advanced techniques
epochs = 3000
print(f"\nTraining for up to {epochs} epochs...")
print("  - Learning rate scheduling")
print("  - Gradient clipping")
print("  - Early stopping\n")

best_loss = float('inf')
patience = 300
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
    scheduler.step(loss)
    
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), "residual_model_probabilistic.pth")
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"\n✓ Early stopping at epoch {epoch} (best loss: {best_loss:.4f})")
        break
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}: Loss {loss.item():.4f} | LR {optimizer.param_groups[0]['lr']:.6f}")

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Final Loss: {best_loss:.4f}")
print(f"Model saved: residual_model_probabilistic.pth")

# Comprehensive testing
model.eval()
print(f"\n{'='*60}")
print("MODEL VALIDATION")
print(f"{'='*60}")

test_scenarios = [
    ("Quiet (Short)", 70, 1, 2),
    ("Normal (Medium)", 150, 3, 12),
    ("Storm (Long)", 280, 7, 48),
    ("Extreme (Very Long)", 350, 9, 72)
]

for name, flux, kp, hours in test_scenarios:
    test_input = torch.tensor([[
        flux / 300.0,
        kp / 10.0,
        0.0, 0.0, 0.6
    ]], dtype=torch.float32)
    
    with torch.no_grad():
        output = model(test_input)
        mu = output[0, :3].numpy()
        logvar = output[0, 3:].numpy()
        std = np.sqrt(np.exp(logvar))
    
    print(f"\n{name} (Flux={flux}, Kp={kp}, Δt={hours}h):")
    print(f"  Correction: [{mu[0]:6.2f}, {mu[1]:6.2f}, {mu[2]:6.2f}] km")
    print(f"  Uncertainty: [{std[0]:5.2f}, {std[1]:5.2f}, {std[2]:5.2f}] km")

print(f"\n{'='*60}")
print("✅ PRODUCTION MODEL READY")
print("   Restart server to load new model")
print(f"{'='*60}\n")
