from sgp4.api import Satrec, jday
import numpy as np
from datetime import datetime, timedelta

class Propagator:
    def __init__(self):
        pass

    def get_position(self, line1: str, line2: str, target_time: datetime):
        """
        Propagate satellite orbit to target_time using SGP4.
        Returns position (x, y, z) in km (TEME frame).
        """
        satellite = Satrec.twoline2rv(line1, line2)
        jd, fr = jday(target_time.year, target_time.month, target_time.day,
                      target_time.hour, target_time.minute, target_time.second)
        e, r, v = satellite.sgp4(jd, fr)
        
        if e != 0:
             raise ValueError(f"SGP4 propagation error code: {e}")
             
        return np.array(r)

if __name__ == "__main__":
    # Test with ISS TLE
    line1 = "1 25544U 98067A   23356.54321689  .00016717  00000+0  30283-3 0  9997"
    line2 = "2 25544  51.6416  21.9684 0005432  35.2163  86.1264 15.49507156430342"
    
    prop = Propagator()
    pos = prop.get_position(line1, line2, datetime.utcnow() + timedelta(days=3))
    print(f"Predicted Position (+3 days): {pos} km")
