import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np
from vision.pose_net import SatellitePoseNet

class VisionAPI:
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[VisionAPI] Initializing on {self.device}...")
        
        self.model = SatellitePoseNet().to(self.device)
        self.model.eval()
        
        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("[VisionAPI] Model Loaded (Untrained - Waiting for Synthetic Data)")

    def predict_from_base64(self, base64_str: str):
        """
        Decodes RGB base64 image and predicts 6D pose + tumble.
        """
        try:
            # Clean header if present (data:image/jpeg;base64,...)
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            
            # Decode
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Preprocess
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Infer
            with torch.no_grad():
                pos, quat, tumble = self.model(input_tensor)
                
            # Post-process
            return {
                "success": True,
                "position": pos.cpu().numpy()[0].tolist(), # [x, y, z]
                "quaternion": quat.cpu().numpy()[0].tolist(), # [qw, qx, qy, qz]
                "tumble_rate": tumble.cpu().numpy()[0].tolist(), # [wx, wy, wz]
                "device": self.device
            }
            
        except Exception as e:
            print(f"[VisionAPI] Error: {e}")
            return {"success": False, "error": str(e)}

    def detect_debris(self, image_input):
        """
        Alias for prediction to satisfy standard interface.
        Accepts base64 string or tensor (mock).
        """
        if isinstance(image_input, str):
            return self.predict_from_base64(image_input)
        elif isinstance(image_input, torch.Tensor):
            # Mock pass for testing with tensor
            return {"detected": True, "confidence": 0.95, "box": [10, 10, 100, 100]}
        return {"detected": False, "error": "Invalid Input"}

    def save_synthetic_sample(self, base64_str: str, label_data: dict):
        """
        Saves a training sample (Image + Label) to disk for later training.
        """
        # TODO: Implement dataset saver
        pass 
