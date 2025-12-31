import numpy as np
import torch
from datetime import datetime, timedelta
from propagator import Propagator
from model.residual_net import ResidualCorrectionNet

class MatrixScreener:
    def __init__(self, propagator=None, model=None, protected_sat_id=25544):
        if propagator is None:
            from propagator import Propagator
            self.propagator = Propagator()
        else:
            self.propagator = propagator
            
        if model is None:
            # Load dummy or real model if needed for screening
            self.model = lambda x: torch.zeros((1, 6)) # Dummy identity model
        else:
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
        target_time = datetime.utcnow() 
        
        # Asset Position checked in helper, but we might want to cache it.
        # For simplicity, we just pass TLEs to helper.

        # 2. Iterate Catalog
        for sat_id, tle in catalog_tles.items():
            if str(sat_id) == str(self.protected_sat_id): continue
            
            try:
                res = self._check_conjunction(
                    asset_tle['line1'], asset_tle['line2'],
                    tle['line1'], tle['line2'],
                    target_time, sat_id, tle.get('name', 'Unknown')
                )
                
                if res and res['status'] in ["CRITICAL", "WARNING"]:
                    high_risk_alerts.append(res)
                    print(f"[Screener] 🚨 ALERT: {res['debris_name']} | Miss: {res['ai_dist_km']:.2f}km ±{res['uncertainty_km']:.2f}")

            except Exception as e:
                continue
                
        print(f"[Screener] Scan complete. Found {len(high_risk_alerts)} threats.")
        return high_risk_alerts

    def _check_conjunction(self, asset_l1, asset_l2, deb_l1, deb_l2, target_time, deb_id="Unknown", deb_name="Unknown"):
        """
        Helper to check a single pair for collision.
        """
        try:
            # 1. Physics Positions
            asset_pos = self.propagator.get_position(asset_l1, asset_l2, target_time)
            deb_pos = self.propagator.get_position(deb_l1, deb_l2, target_time)
            
            dist_phys = np.linalg.norm(np.array(asset_pos) - np.array(deb_pos))
            
            # Coarse Filter
            if dist_phys > 500.0:
                return None

            # 2. AI Refinement
            asset_ai_pos, asset_uncert = self._get_ai_position(asset_pos)
            deb_ai_pos, deb_uncert = self._get_ai_position(deb_pos)
            
            dist_ai = np.linalg.norm(asset_ai_pos - deb_ai_pos)
            total_uncert = asset_uncert + deb_uncert
            
            dist_conservative = dist_ai - total_uncert
            
            if dist_conservative < self.risk_threshold_km:
                 return {
                    "debris_id": deb_id,
                    "debris_name": deb_name,
                    "tca": target_time.isoformat() if isinstance(target_time, datetime) else str(target_time),
                    "phys_dist_km": float(dist_phys),
                    "ai_dist_km": float(dist_ai),
                    "uncertainty_km": float(total_uncert),
                    "status": "CRITICAL" if dist_ai < 1.0 else "WARNING"
                }
            return None
        except Exception as e:
            return None

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
