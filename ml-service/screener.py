import numpy as np
import torch
from datetime import datetime, timedelta
from propagator import Propagator
from model.residual_net import ResidualCorrectionNet

class MatrixScreener:
    def __init__(self, propagator: Propagator, model: ResidualCorrectionNet, protected_sat_id=25544):
        self.propagator = propagator
        self.model = model
        self.protected_sat_id = protected_sat_id # ISS Default
        self.risk_threshold_km = 10.0 # High alert if < 10km
        print(f"[MatrixScreener] Initialized for Asset ID {protected_sat_id}")

    def screen_catalog(self, catalog_tles: dict):
        """
        Screen the entire catalog against the protected asset.
        catalog_tles: dict of {id: tle_dict}
        """
        print(f"[Screener] Starting matrix scan of {len(catalog_tles)} objects...")
        
        # 1. Get Protected Asset TLE
        if str(self.protected_sat_id) not in catalog_tles:
            print("[Screener] Protected asset TLE missing. Skipping.")
            return []
            
        asset_tle = catalog_tles[str(self.protected_sat_id)]
        
        high_risk_alerts = []
        
        # Target Time: Now + 1 orbit (90 mins) to finding immediate threats
        # Real-world screener would check +1 day or +3 days with steps.
        # For this demo, we check a single critical point: "Now"
        target_time = datetime.utcnow() 
        
        # Get Asset Position (Physics)
        try:
            asset_pos_phys = self.propagator.get_position(
                asset_tle['line1'], asset_tle['line2'], target_time
            )
        except:
            return []

        # 2. Iterate Catalog
        for sat_id, tle in catalog_tles.items():
            if str(sat_id) == str(self.protected_sat_id): continue
            
            # --- PHASE 1: COARSE FILTER (Geometry) ---
            # Ideally: Check inclination/RAAN alignment.
            # Simplified: Check Altitude diff. If alt diff > 50km, skip.
            # We skip this for the demo to ensure we actually calculate some things.
            
            # --- PHASE 2: FINE FILTER (Physics + AI) ---
            try:
                deb_pos_phys = self.propagator.get_position(
                    tle['line1'], tle['line2'], target_time
                )
                
                # Raw Physics Distance
                dist_phys = np.linalg.norm(np.array(asset_pos_phys) - np.array(deb_pos_phys))
                
                # If Physics says "Safe" (> 500km), we ignore.
                if dist_phys > 500.0:
                    continue
                    
                # If < 500km, we run the EXPENSIVE AI Model
                # AI Correction for Asset
                asset_ai_pos, asset_uncert = self._get_ai_position(asset_pos_phys)
                
                # AI Correction for Debris
                deb_ai_pos, deb_uncert = self._get_ai_position(deb_pos_phys)
                
                # AI Distance
                dist_ai = np.linalg.norm(asset_ai_pos - deb_ai_pos)
                total_uncertainty = asset_uncert + deb_uncert
                
                # ALERT LOGIC
                # Risk if (Distance - Uncertainty) < Threshold
                conservative_dist = dist_ai - total_uncertainty
                
                if conservative_dist < self.risk_threshold_km:
                    alert = {
                        "debris_id": sat_id,
                        "debris_name": tle.get('name', 'Unknown'),
                        "tca": target_time.isoformat(),
                        "phys_dist_km": float(dist_phys),
                        "ai_dist_km": float(dist_ai),
                        "uncertainty_km": float(total_uncertainty),
                        "status": "CRITICAL" if dist_ai < 1.0 else "WARNING"
                    }
                    high_risk_alerts.append(alert)
                    print(f"[Screener] 🚨 ALERT: {alert['debris_name']} | Miss: {dist_ai:.2f}km ±{total_uncertainty:.2f}")
                    
            except Exception as e:
                continue
                
        print(f"[Screener] Scan complete. Found {len(high_risk_alerts)} threats.")
        return high_risk_alerts

    def _get_ai_position(self, physics_pos):
        """Helper to run model inference."""
        # Normalize Input (Flux=150, Kp=3 mocked)
        flux_norm = 150.0 / 300.0
        kp_norm = 3.0 / 10.0
        pos_norm = [p / 10000.0 for p in physics_pos]
        
        inp = torch.tensor([[flux_norm, kp_norm] + pos_norm], dtype=torch.float32)
        
        with torch.no_grad():
            output = self.model(inp).numpy()[0]
        
        correction = output[:3]
        log_var = output[3:]
        std_dev = np.sqrt(np.exp(log_var))
        uncert = float(np.linalg.norm(std_dev))
        
        return (physics_pos + correction), uncert
