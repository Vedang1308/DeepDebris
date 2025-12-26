from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from propagator import Propagator
from datetime import datetime
import numpy as np
import requests
import os
from dotenv import load_dotenv

load_dotenv("../.env") # Load from root

app = FastAPI()
propagator = Propagator()

SPACETRACK_USER = os.getenv("SPACETRACK_USER")
SPACETRACK_PASSWORD = os.getenv("SPACETRACK_PASSWORD")

from model.residual_net import ResidualCorrectionNet
import torch

# Load model
model = ResidualCorrectionNet()
try:
    model.load_state_dict(torch.load("residual_model.pth"))
    model.eval()
except Exception as e:
    print(f"Warning: Model not found or error loading: {e}")

@app.get("/tle/{norad_id}")
def get_latest_tle(norad_id: int):
    """
    Fetch lateset TLE from Space-Track (Auth) or Mock (Fallback).
    """
    if SPACETRACK_USER and SPACETRACK_PASSWORD:
        try:
            print(f"Fetching real TLE for {norad_id}...")
            login_url = "https://www.space-track.org/ajaxauth/login"
            query = f"https://www.space-track.org/basicspacedata/query/class/tle_latest/NORAD_CAT_ID/{norad_id}/orderby/ORDINAL desc/format/json"
            
            payload = {
                "identity": SPACETRACK_USER,
                "password": SPACETRACK_PASSWORD,
                "query": query
            }
            
            # Post credentials and query
            resp = requests.post(login_url, data=payload)
            
            if resp.status_code == 200:
                data = resp.json()
                if len(data) > 0:
                    sat = data[0]
                    return {
                        "line1": sat["TLE_LINE1"],
                        "line2": sat["TLE_LINE2"],
                        "name": sat["OBJECT_NAME"],
                        "epoch": sat["EPOCH"],
                        "source": "SPACE-TRACK-LIVE"
                    }
        except Exception as e:
            print(f"Error fetching real TLE: {e}")
            pass
            
    # Fallback / Mock
    if str(norad_id) == "25544": # ISS
        return {
            "line1": "1 25544U 98067A   23356.54321689  .00016717  00000+0  30283-3 0  9997",
            "line2": "2 25544  51.6416  21.9684 0005432  35.2163  86.1264 15.49507156430342",
            "name": "ISS (MOCK)",
            "source": "MOCK"
        }
    elif str(norad_id) == "20580": # Hubble
        return {
            "line1": "1 20580U 90037B   23355.12345678  .00001234  00000+0  12345-4 0  9993",
            "line2": "2 20580  28.4699 123.4567 0001234  12.3456 123.4567 15.09345678 12345",
            "name": "HUBBLE (MOCK)",
            "source": "MOCK"
        }
    
    raise HTTPException(status_code=404, detail="Satellite not found in mock DB and live fetch failed")

class TleRequest(BaseModel):
    line1: str
    line2: str
    target_time: datetime
    solar_flux: float = 150.0
    kp_index: float = 3.0

@app.post("/predict")
def predict_orbit(request: TleRequest):
    try:
        # 1. Physics Prediction (SGP4)
        pos_physics = propagator.get_position(request.line1, request.line2, request.target_time)
        
        # 2. ML Correction
        # Input: [Flux, Kp, X, Y, Z]
        input_tensor = torch.tensor([[
            request.solar_flux, 
            request.kp_index, 
            pos_physics[0], 
            pos_physics[1], 
            pos_physics[2]
        ]], dtype=torch.float32)
        
        with torch.no_grad():
            correction = model(input_tensor).numpy()[0]

        # 3. Final Prediction
        final_pos = pos_physics + correction
        
        return {
            "x": float(final_pos[0]), "y": float(final_pos[1]), "z": float(final_pos[2]),
            "physics_x": float(pos_physics[0]), "physics_y": float(pos_physics[1]), "physics_z": float(pos_physics[2]),
            "correction_x": float(correction[0]), "correction_y": float(correction[1]), "correction_z": float(correction[2]),
            "source": "HYBRID_PHYSICS_ML"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
