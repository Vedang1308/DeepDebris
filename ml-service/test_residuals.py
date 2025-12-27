import torch
import numpy as np
from main import model, propagator
import datetime

def test_residuals():
    print("Testing Model Residuals...")
    
    # Mock Input (Standard LEO conditions)
    # Normalized inputs as per training
    # Solar Flux / 300, Kp / 10, Pos / 10000, Vel / 8
    
    # Example Position (7000km orbit)
    pos = np.array([7000.0, 0.0, 0.0])
    vel = np.array([0.0, 7.5, 0.0])
    
    flux = 300.0 # Extreme Solar Storm
    kp = 9.0     # Extreme Geomagnetic Storm
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_tensor = torch.tensor([[
        flux / 300.0,
        kp / 10.0,
        pos[0] / 10000.0,
        pos[1] / 10000.0,
        pos[2] / 10000.0
    ]], dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        residuals_norm = model(input_tensor)
        
    residuals_km = residuals_norm.cpu().numpy()[0] * 100.0 # Output scale is 100km? 
    # Check main.py for scaling. Usually output is normalized.
    
    print(f"Input Flux: {flux}, Kp: {kp}")
    print(f"Raw Output Tensor: {residuals_norm.cpu().numpy()}")
    print(f"Magnitude: {np.linalg.norm(residuals_norm.cpu().numpy())}")

if __name__ == "__main__":
    test_residuals()
