import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta

class FleetManager:
    """
    Manages "Own Force" Constellation Safety.
    Prevents Fratricide (Self-Collision) by screening maneuvers against the fleet.
    """
    def __init__(self, size=20):
        self.fleet = []
        self._init_constellation(size)
        print(f"FleetManager: Tracking {len(self.fleet)} friendly assets.")

    def _init_constellation(self, size):
        """
        Generates a physics-valid Walker Delta-like constellation.
        Base: 550km, 53 deg inclination (Starlink-ish).
        """
        # Base TLE template
        # 1 00001U 23001A   23356.50000000  .00000000  00000-0  00000-0 0  9999
        # 2 00001  53.0000   0.0000 0001000   0.0000   0.0000 15.10000000    1
        
        base_epoch = datetime.utcnow()
        year = base_epoch.year % 100
        day = base_epoch.strftime('%j.%f')
        
        for i in range(size):
            # Distribute RAAN (Right Ascension) and Anomaly
            raan = (i * (360.0 / size)) % 360.0
            mean_anomaly = (i * (360.0 / size) + 15.0) % 360.0
            
            # Construct TLE Line 2 manually (Simplified)
            # 2 AAABB  II.IIII RR.RRRR EEEEEEE pp.pppp mm.mmmmm NN.nnnnnnnn
            l2 = f"2 {10000+i:05d}  53.0000 {raan:08.4f} 0001000   0.0000 {mean_anomaly:08.4f} 15.10000000    1"
            l1 = f"1 {10000+i:05d}U 23001A   {year:02d}{day}  .00000000  00000-0  00000-0 0  9999"
            
            sat = Satrec.twoline2rv(l1, l2)
            self.fleet.append({'id': 10000+i, 'sat': sat, 'line1': l1, 'line2': l2})

    def check_safety(self, proposed_line1, proposed_line2, minutes_check=60):
        """
        Screens a proposed orbit against the fleet for X minutes.
        Returns: { 'safe': bool, 'conflict': str|None }
        """
        target_sat = Satrec.twoline2rv(proposed_line1, proposed_line2)
        
        start_time = datetime.utcnow()
        step_min = 1
        
        for m in range(0, minutes_check, step_min):
            t = start_time + timedelta(minutes=m)
            jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
            
            # Propagate Target
            e, r_tgt, v = target_sat.sgp4(jd, fr)
            if e != 0: continue
            r_tgt = np.array(r_tgt)
            
            # Propagate Fleet
            for friend in self.fleet:
                e_f, r_friend, v_f = friend['sat'].sgp4(jd, fr)
                if e_f != 0: continue
                r_friend = np.array(r_friend)
                
                # Check Distance
                dist = np.linalg.norm(r_tgt - r_friend)
                
                # Safety Bubble: 10 km
                if dist < 10.0:
                    return {
                        'safe': False,
                        'reason': f"Collision Risk with Friendly-{friend['id']} at T+{m}min (Dist: {dist:.1f}km)"
                    }
                    
        return {'safe': True, 'reason': 'Orbit Clear'}

if __name__ == "__main__":
    fm = FleetManager(size=10)
    # Test safe TLE (different altitude)
    l1 = "1 25544U 98067A   23356.54321689  .00016717  00000+0  30283-3 0  9997"
    l2 = "2 25544  51.6416  21.9684 0005432  35.2163  86.1264 15.49507156430342"
    print(fm.check_safety(l1, l2))
