import torch
import torch.nn as nn
import torchvision.models as models

class SatellitePoseNet(nn.Module):
    """
    DeepDebris Vision Model
    Predicts 6D Pose (Position + Orientation) and Tumble Rate from a single image.
    Backbone: ResNet-18 (Lightweight)
    """
    def __init__(self, pretrained=False):
        super(SatellitePoseNet, self).__init__()
        
        # Load ResNet-18 backbone
        # We set weights=None by default to avoid internet dependency during startup
        # In production, we would load 'ResNet18_Weights.IMAGENET1K_V1'
        self.backbone = models.resnet18(weights=None)
        
        # Remove the classification head (fc layer)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Define 3 Heads for Multi-Task Regression
        
        # 1. Position Head (x, y, z) - Relative translation
        self.head_pos = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 3) 
        )
        
        # 2. Orientation Head (qw, qx, qy, qz) - Quaternion
        self.head_quat = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        
        # 3. Tumble Rate Head (wx, wy, wz) - Angular velocity
        self.head_tumble = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        
        # Predict heads
        pos_pred = self.head_pos(features)
        quat_pred = self.head_quat(features)
        
        # Normalize quaternion to ensure valid rotation
        quat_pred = torch.nn.functional.normalize(quat_pred, p=2, dim=1)
        
        tumble_pred = self.head_tumble(features)
        
        return pos_pred, quat_pred, tumble_pred

if __name__ == "__main__":
    # Test Architecture
    model = SatellitePoseNet()
    dummy_input = torch.randn(1, 3, 224, 224)
    pos, quat, tumble = model(dummy_input)
    print("Architecture Check Passed:")
    print(f"Pos Shape: {pos.shape} (Expected: [1, 3])")
    print(f"Quat Shape: {quat.shape} (Expected: [1, 4])")
    print(f"Tumble Shape: {tumble.shape} (Expected: [1, 3])")
