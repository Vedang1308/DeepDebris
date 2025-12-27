import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out + residual

class ResidualCorrectionNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=6): # 3 Mean + 3 LogVar
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.res_block1 = ResidualBlock(hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.input_layer(x))
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.output_layer(out)
        return out
