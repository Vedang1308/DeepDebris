import requests
import json

class WeatherService:
    def __init__(self):
        # NOAA SWPC Endpoints
        self.flux_url = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
        self.kp_url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

    def get_live_weather(self):
        """
        Fetches the latest observed F10.7 Solar Flux and Kp Index.
        Returns:
            dict: {"flux": float, "kp": float}
        """
        try:
            # 1. Fetch Solar Flux (F10.7)
            # Returns a list of dicts. We want the last 'observed' value.
            flux_resp = requests.get(self.flux_url, timeout=5)
            flux_resp.raise_for_status()
            flux_data = flux_resp.json()
            
            # Simple logic: Take the last entry
            latest_flux = float(flux_data[-1]['flux'])

            # 2. Fetch Kp Index
            # Returns a list of lists: [time, kp, a_running, station_count]
            kp_resp = requests.get(self.kp_url, timeout=5)
            kp_resp.raise_for_status()
            kp_data = kp_resp.json()
            
            # The first row is header, last row is latest prediction/observation
            # We want the last valid numeric entry
            latest_kp = float(kp_data[-1][1])

            return {
                "flux": latest_flux,
                "kp": latest_kp,
                "source": "NOAA SWPC (Live)"
            }

        except Exception as e:
            print(f"Error fetching NOAA data: {e}")
            # Fallback to moderate defaults
            return {
                "flux": 150.0,
                "kp": 3.0,
                "source": "FALLBACK (Network Error)"
            }

if __name__ == "__main__":
    ws = WeatherService()
    print(ws.get_live_weather())
