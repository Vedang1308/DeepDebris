from skyfield.api import load, wgs84, EarthSatellite
from skyfield.framelib import itrs
from datetime import datetime, timedelta, timezone
import numpy as np

class PassScheduler:
    """
    Manages Ground-to-Space connectivity windows using Skyfield physics.
    Simulates a realistic Ground Station (GS) availability constraint.
    """
    def __init__(self, station_name="Maui Space Surveillance Complex", lat=20.7082, lon=-156.2567, el=3058):
        self.ts = load.timescale()
        # Define Ground Station (Topos)
        self.gs_location = wgs84.latlon(lat, lon, elevation_m=el)
        self.station_name = station_name
        print(f"PassScheduler: Initialized Ground Station at {station_name}")

    def get_next_pass(self, line1, line2, start_time_iso=None):
        """
        Calculates the next visibility window (AOS to LOS).
        Returns: { 'aos': iso_str, 'los': iso_str, 'duration_sec': float, 'max_el': float }
        """
        sat = EarthSatellite(line1, line2, 'Target', self.ts)
        
        if start_time_iso:
            t0_dt = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00"))
        else:
            t0_dt = datetime.now(timezone.utc)

        t0 = self.ts.from_datetime(t0_dt)
        t1 = self.ts.from_datetime(t0_dt + timedelta(days=1)) # Look ahead 24h

        # Find events: (time, events). events: 0=rise, 1=culminate, 2=set
        times, events = sat.find_events(self.gs_location, t0, t1, altitude_degrees=10.0) # 10 deg horizon mask

        passes = []
        current_pass = {}
        
        for ti, event in zip(times, events):
            name = ('rise', 'culminate', 'set')[event]
            if name == 'rise':
                current_pass['aos'] = ti.utc_datetime().isoformat()
            elif name == 'culminate':
                # Correct Topocentric vector math
                alt, az, distance = (sat - self.gs_location).at(ti).altaz()
                current_pass['max_el'] = float(alt.degrees)
            elif name == 'set':
                current_pass['los'] = ti.utc_datetime().isoformat()
                if 'aos' in current_pass:
                    # Completed pass
                    t_aos = datetime.fromisoformat(current_pass['aos'])
                    t_los = datetime.fromisoformat(current_pass['los'])
                    current_pass['duration'] = (t_los - t_aos).total_seconds()
                    passes.append(current_pass)
                current_pass = {}

        if not passes:
            return None
        
        return passes[0] # Return immediate next pass

    def is_in_view(self, line1, line2, query_time_iso=None):
        """
        Checks if the satellite is CURRENTLY visible to the ground station.
        Used to GATE API requests.
        """
        sat = EarthSatellite(line1, line2, 'Target', self.ts)
        
        if query_time_iso:
            t_dt = datetime.fromisoformat(query_time_iso.replace("Z", "+00:00"))
        else:
            t_dt = datetime.now(timezone.utc)
            
        t = self.ts.from_datetime(t_dt)
        
        # Calculate Altitude (Elevation)
        difference = sat - self.gs_location
        topocentric = difference.at(t)
        alt, az, distance = topocentric.altaz()
        
        is_visible = alt.degrees > 10.0 # Horizon mask
        
        return {
            'visible': bool(is_visible),
            'elevation': float(alt.degrees),
            'azimuth': float(az.degrees),
            'station': self.station_name,
            'timestamp': t.utc_datetime().isoformat()
        }

if __name__ == "__main__":
    # Test with ISS
    l1 = "1 25544U 98067A   23356.54321689  .00016717  00000+0  30283-3 0  9997"
    l2 = "2 25544  51.6416  21.9684 0005432  35.2163  86.1264 15.49507156430342"
    
    scheduler = PassScheduler()
    
    # Check current visibility
    status = scheduler.is_in_view(l1, l2)
    print(f"Current Status: {status}")
    
    # Check next pass
    nxt = scheduler.get_next_pass(l1, l2)
    print(f"Next Pass: {nxt}")
