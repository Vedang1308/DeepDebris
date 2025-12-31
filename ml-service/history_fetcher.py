import requests
import os
import json
from datetime import datetime, timedelta

class HistoryFetcher:
    def __init__(self):
        self.base_url = "https://www.space-track.org"
        self.login_url = f"{self.base_url}/ajaxauth/login"
        self.session = requests.Session()
        self.logged_in = False
        
    def _login(self):
        identity = os.getenv('SPACETRACK_USER')
        password = os.getenv('SPACETRACK_PASSWORD')
        if not identity or not password:
            print("⚠ HistoryFetcher: Missing SPACETRACK_USER or SPACETRACK_PASSWORD environment variables.")
            return False
            
        try:
            resp = self.session.post(self.login_url, data={'identity': identity, 'password': password})
            if resp.status_code == 200:
                self.logged_in = True
                print("✓ HistoryFetcher: Logged in successfully")
                return True
            else:
                print(f"⚠ HistoryFetcher: Login failed with status {resp.status_code}")
                return False
        except Exception as e:
            print(f"⚠ HistoryFetcher Login Error: {e}")
            return False

    def get_tle_history(self, norad_id, days=30):
        if not self.logged_in:
            if not self._login():
                return [] # Fail gracefully
                
        # API Query for TLEs
        start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # /basicspacedata/query/class/gp_history/NORAD_CAT_ID/{id}/EPOCH/>{date}/orderby/EPOCH asc
        query = f"/basicspacedata/query/class/gp_history/NORAD_CAT_ID/{norad_id}/EPOCH/>{start_date}/orderby/EPOCH asc"
        full_url = f"{self.base_url}{query}"
        
        try:
            resp = self.session.get(full_url)
            if resp.status_code != 200:
                print(f"Spy Hunter Error: API returned status {resp.status_code}")
                return []
            
            # Check if response is JSON before parsing
            try:
                data = resp.json()
            except json.JSONDecodeError as je:
                print(f"Spy Hunter Error: Response is not valid JSON. Content: {resp.text[:100]}")
                return []
            
            # Extract features: [Inc, Ecc, MeanMotion, BStar]
            features = []
            for tle in data:
                try:
                    inc = float(tle['INCLINATION'])
                    ecc = float(tle['ECCENTRICITY'])
                    mm = float(tle['MEAN_MOTION'])
                    bstar = float(tle['BSTAR'])
                    features.append([inc, ecc, mm, bstar])
                except:
                    continue
            return features
            
        except Exception as e:
            print(f"Spy Hunter Error: {e}")
            return []

