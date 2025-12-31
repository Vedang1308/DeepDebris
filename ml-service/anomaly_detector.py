import torch
import torch.nn as nn
import numpy as np

class OrbitalAutoencoder(nn.Module):
    """
    LSTM Autoencoder for learning the 'Pattern of Life' of orbital objects.
    
    Encoder: Compresses time-series orbital elements into a latent vector.
    Decoder: Reconstructs the sequence from the latent vector.
    
    Assumption: Low reconstruction error = Natural Orbit.
                High reconstruction error = Maneuver/Anomaly.
    """
    def __init__(self, input_size=4, hidden_size=16, latent_size=8, num_layers=1):
        super().__init__()
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.latent_layer = nn.Linear(hidden_size, latent_size)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(
            input_size=hidden_size, # Input to decoder LSTM is hidden state
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Encode
        _, (h_n, c_n) = self.encoder(x)
        # h_n shape: (num_layers, batch, hidden) -> Take last layer
        last_hidden = h_n[-1]
        latent = self.latent_layer(last_hidden) # (batch, latent)
        
        # 2. Decode
        # Repeat latent vector for each time step in sequence?
        # A simpler decoder strategy: Use latent as initial state, or input at every step.
        # Strategy: Repeat latent vector `seq_len` times to form the sequence input for decoder.
        
        decoder_hidden_init = self.decoder_input(latent) # (batch, hidden)
        
        # We need to provide input to decoder LSTM. 
        # Standard Autoencoder: Repeat vector.
        decoder_input_seq = decoder_hidden_init.unsqueeze(1).repeat(1, seq_len, 1) # (batch, seq, hidden)
        
        decoded_seq, _ = self.decoder(decoder_input_seq)
        
        # 3. Project back to input size
        reconstruction = self.output_layer(decoded_seq)
        
        return reconstruction

    def compute_anomaly_score(self, x):
        """
        Returns the Mean Squared Error between input and reconstruction.
        """
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            # MSE per sample
            loss = torch.mean((x - recon) ** 2, dim=[1, 2])
        return loss

# --- Application Service Wrapper ---

class SpyHunter:
    def __init__(self, model_path=None):
        self.model = OrbitalAutoencoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"SpyHunter: Loaded model from {model_path}")
            except Exception as e:
                print(f"SpyHunter: Could not load model ({e}). Using untrained model.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Normalization stats (approximate standard LEO values for scaling)
        # INC, ECC, MEAN_MOTION, BSTAR
        self.mean = torch.tensor([50.0, 0.01, 14.5, 0.0001]).to(self.device)
        self.std = torch.tensor([30.0, 0.05, 1.5, 0.001]).to(self.device)

    def analyze_behavior(self, history_data):
        """
        history_data: List of [Inc, Ecc, MeanMotion, BStar] over time.
        Returns: { 'is_anomaly': bool, 'score': float }
        """
        if len(history_data) < 10:
            return {'is_anomaly': False, 'score': 0.0, 'msg': 'Insufficient Data'}
            
        # Convert to Tensor
        x = torch.tensor(history_data, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, Seq, 4)
        
        # Normalize
        x_norm = (x - self.mean) / (self.std + 1e-6)
        
        # Inference
        score = self.model.compute_anomaly_score(x_norm).item()
        
        # Threshold (Heuristic for now)
        # "5-sigma" logic would require distribution of scores from training.
        # Let's say if reconstruction error is > 1.5 (arbitrary high deviation)
        THRESH = 1.0 
        
        return {
            'anomaly_score': score,
            'threat_level': 'HIGH' if score > THRESH else 'NOMINAL'
        }

# --- DeepDebris 4.0: Cyber-Physical Validator ---
class PhysicsValidator:
    """
    Enforces Keplerian Consistency on incoming TLE data.
    Reject "Impossible Maneuvers" (Spoofing).
    """
    def __init__(self):
        pass
        
    def check_consistency(self, old_tle, new_tle):
        """
        Compare two TLEs for the SAME object.
        Returns: { 'valid': bool, 'reason': str }
        """
        if not old_tle or not new_tle:
            return {'valid': True, 'reason': 'Baseline Missing'}
            
        # Parse logic (Simplified for speed - manual parse)
        try:
            # Inclination (Field 3 line 2)
            inc_old = float(old_tle['line2'].split()[2])
            inc_new = float(new_tle['line2'].split()[2])
            
            # Epoch
            # epoch_old = ... (requires complex parsing)
            # Let's assume sequential updates for now.
            
            # CONSTRAINTS:
            
            # 1. Plane Change Check
            # Change in inclination requires Delta-V = 2*V*sin(dInc/2).
            # LEO V ~ 7.5 km/s.
            # 1 deg change => ~130 m/s.
            # If dInc > 0.5 degrees instantly, it's virtually impossible for a standard sat.
            d_inc = abs(inc_new - inc_old)
            if d_inc > 0.5:
                return {
                    'valid': False, 
                    'reason': f"CYBER ALERT: Impossible Plane Change ({d_inc:.2f} deg). Violation of conservation of momentum."
                }
                
            # 2. Altitude Teleportation (Mean Motion)
            # mm_old = float(old_tle['line2'].split()[7])
            # mm_new = float(new_tle['line2'].split()[7])
            # If period changes by > 1 minute instantly -> Impossible.
            
            return {'valid': True, 'reason': 'Physics Consistent'}
            
        except Exception as e:
            return {'valid': True, 'reason': f"Parse Error: {e}"}

