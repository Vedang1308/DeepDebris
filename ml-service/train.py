import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from model.residual_net import ResidualCorrectionNet

def generate_synthetic_data(n_samples=1000):
    # Inputs: Flux (0-200), Kp (0-9), SGP4 Pos (km)
    X = np.random.rand(n_samples, 5).astype(np.float32)
    X[:, 0] *= 200 # Flux
    X[:, 1] *= 9   # Kp
    X[:, 2:] *= 7000 # Pos ~7000km

    # Truth = 0.01 * Kp * Pos (Correlation: Higher Kp -> Higher Drag -> Error adds up)
    # This is a very simplified mock function
    Y = (X[:, 1:2] * 0.001 * X[:, 2:]).astype(np.float32) 
    
    return torch.tensor(X), torch.tensor(Y)

def train():
    model = ResidualCorrectionNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    if os.path.exists("X_train.pt") and os.path.exists("y_train.pt"):
        print("Loading real historical data...")
        X_train = torch.load("X_train.pt")
        y_train = torch.load("y_train.pt")
    else:
        print("Warning: Real data not found. Falling back to synthetic.")
        X_train, y_train = generate_synthetic_data()
    
    print("Training model...")
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            
    torch.save(model.state_dict(), "residual_model.pth")
    print("Model saved to residual_model.pth")

if __name__ == "__main__":
    train()
