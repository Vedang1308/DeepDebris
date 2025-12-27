from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from propagator import Propagator
from weather_service import WeatherService
from cdm_service import CDMService
from rag_engine import OrbitGPTEngine
from datetime import datetime, timedelta
import numpy as np
import requests
import os
import time
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler
import pytz

load_dotenv("../.env") # Load from root

app = FastAPI(title="DeepDebris ML Service")

# CORS (Allow local frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves the Three.js Frontend
# StaticFiles mount moved to end of file to prevent route masking

propagator = Propagator()
weather_service = WeatherService()
cdm_service = CDMService()
# Initialize OrbitGPT (Ingest on startup for demo)
orbit_gpt = OrbitGPTEngine(cdm_service)
print("Ingesting initial CDMs...")
orbit_gpt.ingest_cdms()
print("OrbitGPT Ready.")

# --- Adaptive Architecture: Background Scheduler (Production Mode) ---
# "What works better": A hybrid. We fetch on startup and then scheduled.
scheduler = BackgroundScheduler()

def refresh_data_cache():
    """Run via Scheduler: Keep the 'Live State' fresh for the user."""
    print("[Scheduler] Running background data refresh...")
    try:
        # 1. Fetch Latest TLEs and Train (Continuous Learning)
        # We focus on ISS (25544) for this demo
        nid = 25544
        nid_str = str(nid)
        
        # Check if we have a previous TLE in cache to compare against
        prev_tle = None
        if nid_str in TLE_CACHE:
            prev_tle = TLE_CACHE[nid_str]
        
        # Force a "Live Check" - logic similar to get_latest_tle but specifically looking for updates
        # We reuse get_latest_tle logic implicitly by calling it or copying relevant parts?
        # Better: Let's extract TLE fetching into a helper or just do it here for the scheduler.
        # Ideally, we want to see if Space-Track has a NEWER one than what we have.
        
        # We'll use get_latest_tle to update the cache.
        # But get_latest_tle returns cached if fresh. We want to FORCE check for training.
        # Actually, get_latest_tle respects CACHE_DURATION.
        # If we reduce cache duration or force fetch...
        
        # Let's trust get_latest_tle's logic for now, but maybe we need a dedicated "check_for_updates"
        # For simplicity in this demo:
        # We will assume this scheduler runs less frequently than TLE updates (6 hours).
        # So essentially every time it runs, it's likely there is a new TLE if we force fetch.
        
        # Let's peek at Space-Track directly here to get "Current" vs "Previous"
        if SPACETRACK_USER and SPACETRACK_PASSWORD:
             session = requests.Session()
             login_url = "https://www.space-track.org/ajaxauth/login"
             # Fetch LAST 2 TLEs
             query = f"https://www.space-track.org/basicspacedata/query/class/tle/NORAD_CAT_ID/{nid}/orderby/EPOCH desc/limit/2/format/json"
             
             resp = session.post(login_url, data={"identity": SPACETRACK_USER, "password": SPACETRACK_PASSWORD})
             if resp.status_code == 200:
                 resp = session.get(query)
                 if resp.status_code == 200:
                     data = resp.json()
                     if len(data) >= 2:
                         curr = data[0]
                         prev = data[1]
                         
                         # Check if 'curr' is newer than 'prev' (it should be)
                         # And if we haven't trained on this pair yet?
                         # For now, just train on the latest pair found.
                         # This is "Online Learning" on the latest available transition.
                         
                         current_tle_dict = {
                            "line1": curr["TLE_LINE1"],
                            "line2": curr["TLE_LINE2"],
                            "epoch": curr["EPOCH"]
                         }
                         prev_tle_dict = {
                            "line1": prev["TLE_LINE1"],
                            "line2": prev["TLE_LINE2"],
                            "epoch": prev["EPOCH"]
                         }
                         
                         print(f"[Continuous Learning] Found 2 recent TLEs. Training on transition...")
                         learner.train_on_single_step(prev_tle_dict, current_tle_dict)
                         
                         # Update Cache with latest
                         TLE_CACHE[nid_str] = {
                             "line1": curr["TLE_LINE1"],
                             "line2": curr["TLE_LINE2"],
                             "name": curr["OBJECT_NAME"],
                             "epoch": curr["EPOCH"],
                             "source": "SPACE-TRACK-LIVE",
                             "timestamp": time.time()
                         }

        # 2. Refetch CDMs for context
        cdm_service.fetch_recent_cdms(25544) 
        # Ingest into RAG so the Agent allows knows the latest risks
        orbit_gpt.ingest_cdms()
        # Fetch Weather
        weather_service.get_live_weather()
        print("[Scheduler] Data Refresh Complete.")
    except Exception as e:
        print(f"[Scheduler] Error refreshing data: {e}")

# Run every 6 hours (Standard TLE update cycle)
scheduler.add_job(refresh_data_cache, 'interval', hours=6, timezone=pytz.utc)
# scheduler.start() # Moved to after all jobs are added
print("Adaptive Scheduler Configured: Background refresh every 6 hours.")



class ChatRequest(BaseModel):
    query: str

SPACETRACK_USER = os.getenv("SPACETRACK_USER")
SPACETRACK_PASSWORD = os.getenv("SPACETRACK_PASSWORD")

from model.residual_net import ResidualCorrectionNet
from continuous_learner import ContinuousLearner
import torch

# Load Probabilistic Model (6 outputs)
model = ResidualCorrectionNet(input_dim=5, output_dim=6)
try:
    model.load_state_dict(torch.load("residual_model_probabilistic.pth"))
    model.eval()
    print("Loaded PROBABILISTIC Residual Model (Mean+Variance).")
except Exception as e:
    print(f"Warning: Model not found or error loading: {e}")

# Initialize Continuous Learner
learner = ContinuousLearner(model, model_path="residual_model_probabilistic.pth")

# --- DeepDebris 3.0: Load RL Maneuver Agent ---
maneuver_agent = None
try:
    from stable_baselines3 import PPO
    from rl.space_gym import SpaceGym, action_to_vector, calculate_burn_time, calculate_optimal_time
    
    if os.path.exists("rl/models/maneuver_agent.zip"):
        maneuver_agent = PPO.load("rl/models/maneuver_agent.zip")
        print("✓ Loaded RL Maneuver Agent (DeepDebris 3.0)")
    else:
        print("⚠ RL Maneuver Agent not found. Train with: python rl/train_maneuver_agent.py")
except Exception as e:
    print(f"⚠ RL Agent unavailable: {e}")

# ... Imports ...
from screener import MatrixScreener

# ... (Existing Init: Model, Propagator, Weather) ...
# PROBABILISTIC MODEL LOADED ABOVE

# Initialize Continuous Learner
learner = ContinuousLearner(model)

# FEATURE 3: Automated Matrix Screener
print("Initializing Automated Matrix Screener...")
screener_service = MatrixScreener(propagator, model, protected_sat_id=25544)
print("Screener Service Ready.")

# Global Alerts Cache (Updated by Screener)
ACTIVE_ALERTS = []

# --- API ENDPOINTS ---

@app.get("/alerts")
def get_active_alerts():
    """Return the latest alerts from the Matrix Screener."""
    return {"alerts": ACTIVE_ALERTS, "count": len(ACTIVE_ALERTS)}

@app.post("/chat")
def chat_with_orbitgpt(request: ChatRequest):
    """Ask OrbitGPT a question based on retrieved CDMs and Live Risk Analysis."""
    
    # Context Injection (Simplified RAG)
    context_injection = ""
    query_lower = request.query.lower()
    
    if any(k in query_lower for k in ["risk", "safe", "collision", "danger", "status"]):
        try:
            # 1. Get Top Risks
            risks = get_risk_objects()
            if risks:
                top_risk = risks[0]
                context_injection += f"\n[SYSTEM ALERT: High Risk Detected with {top_risk['name']} (ID: {top_risk['id']}) at {top_risk['tca']}]. "
                
                # 2. Run AI Analysis on Top Risk (Auto-Analyst)
                # We reuse the logic from analyze_risk but call it internally if needed
                # context_injection += " DeepDebris AI suggests running a detailed analysis."
        except Exception as e:
            print(f"Context Injection Error: {e}")

    # augment query
    full_query = request.query + context_injection
    
    try:
        response = orbit_gpt.ask(full_query)
        return {"response": response}
    except Exception as e:
        print(f"Chat Error: {e}")
        return {"response": "System Error: The Analyst is currently offline."}

# Schedule Screener Job (Run every 1 minute for demo)
def run_screener_job():
    """Background job to screen for collisions."""
    global ACTIVE_ALERTS
    print("[Scheduler] Running Auto-Screening...")
    if not TLE_CACHE:
        print("[Scheduler] Catalog empty. Skipping screen.")
        return
        
    alerts = screener_service.screen_catalog(TLE_CACHE)
    ACTIVE_ALERTS = alerts  # Update global cache
    if alerts:
        print(f"!!! CRITICAL ALERTS FOUND: {len(alerts)} !!!")
        for a in alerts:
            print(f" >> ALERT: {a['debris_name']} (Prob Dist: {a['ai_dist_km']:.2f}km ±{a['uncertainty_km']:.2f})")


# Add to Scheduler
scheduler.add_job(run_screener_job, 'interval', minutes=1)
# Keep existing refresh job
scheduler.add_job(refresh_data_cache, 'interval', hours=6) 
scheduler.start()

@app.get("/weather/live")
def get_live_weather():
    """Fetch real-time space weather from NOAA."""
    return weather_service.get_live_weather()

@app.get("/debris/catalog")
def get_debris_catalog(limit: int = 20):
    """
    Fetch real debris objects from Space-Track catalog.
    Returns TLE data for tracked debris in LEO.
    """
    try:
        # Real debris NORAD IDs from major collision events
        debris_ids = [
            22403,  # SL-16 DEB  
            25400,  # COSMOS 2251 DEB
            33591,  # IRIDIUM 33 DEB
            37756,  # BREEZE-M DEB
            40294,  # H-2A DEB
            41731,  # DELTA 2 DEB
            43947,  # CZ-4B DEB
            27436   # COSMOS 2389
        ]
        
        debris_list = []
        for debris_id in debris_ids[:limit]:
            try:
                tle_data = get_latest_tle(debris_id)
                if tle_data:
                    debris_list.append({
                        "id": str(debris_id),
                        "name": tle_data.get("name", f"DEBRIS-{debris_id}"),
                        "line1": tle_data["line1"],
                        "line2": tle_data["line2"]
                    })
            except Exception as outer_e:
                print(f"Skipping debris {debris_id}: {outer_e}")
                continue
        
        return {"debris": debris_list, "count": len(debris_list)}
    except Exception as e:
        print(f"Catalog Error: {e}")
        return {"debris": [], "count": 0, "error": str(e)}

# --- CDM Cache (In-Memory) ---
CDM_CACHE = {}

# ... (omitted) ...

# Validated Risky Objects (exposed as API)
@app.get("/risks")
def get_risk_objects():
    """Get high-risk debris objects from CDMs (Real or Cached)."""
    # 1. Return flattened cache if available
    all_risks = []
    for sat_id, entry in CDM_CACHE.items():
        if "risks" in entry:
            all_risks.extend(entry["risks"])
            
    if all_risks:
        return all_risks

    # 2. Fallback: Fetch Live for ISS (Context)
    print("Cache empty/stale. Fetching live risks for ISS...")
    try:
        raw_cdms = cdm_service.fetch_recent_cdms(25544)
        new_risks = []
        for cdm in raw_cdms:
            r = {
                "id": cdm.get("SAT_2_ID", "UNKNOWN"),
                "name": cdm.get("SAT_2_NAME", "UNKNOWN DEBRIS"),
                "tca": cdm.get("TCA", datetime.utcnow().isoformat()),
                "probability": cdm.get("PC", "0.0"),
                "miss_distance": cdm.get("MIN_RNG", "0.0")
            }
            if r["id"] != "UNKNOWN":
                 new_risks.append(r)
        
        # Update Cache
        if new_risks:
            CDM_CACHE["25544"] = {
                "risks": new_risks,
                "timestamp": time.time()
            }
            return new_risks
    except Exception as e:
        print(f"Error checking risks: {e}")
        
    return []

# --- TLE Cache (In-Memory) ---
TLE_CACHE = {}
CACHE_DURATION = 3600 # 1 hour

@app.get("/tle/{norad_id}")
def get_latest_tle(norad_id: int):
    """
    Fetch TLE from Cache or Space-Track (Auto-Caching).
    """
    nid_str = str(norad_id)
    now = time.time()
    
    # 1. Check Cache
    if nid_str in TLE_CACHE:
        entry = TLE_CACHE[nid_str]
        age = now - entry['timestamp']
        if age < CACHE_DURATION:
            print(f"Returning Cached TLE for {nid_str} (Age: {int(age)}s)")
            return entry
            
    # 2. Fetch Live
    if SPACETRACK_USER and SPACETRACK_PASSWORD:
        try:
            print(f"Fetching real TLE for {norad_id}...")
            # ... (Session/Request logic largely same, but condensed)
            session = requests.Session()
            login_url = "https://www.space-track.org/ajaxauth/login"
            query = f"https://www.space-track.org/basicspacedata/query/class/tle_latest/NORAD_CAT_ID/{norad_id}/orderby/ORDINAL desc/format/json"
            
            resp = session.post(login_url, data={"identity": SPACETRACK_USER, "password": SPACETRACK_PASSWORD})
            if resp.status_code == 200:
                resp = session.get(query)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 0 and 'TLE_LINE1' in data[0]:
                        sat = data[0]
                        cache_entry = {
                            "line1": sat["TLE_LINE1"],
                            "line2": sat["TLE_LINE2"],
                            "name": sat["OBJECT_NAME"],
                            "epoch": sat["EPOCH"],
                            "source": "SPACE-TRACK-LIVE",
                            "timestamp": now
                        }
                        # Update Cache
                        TLE_CACHE[nid_str] = cache_entry
                        return cache_entry
                    elif 'error' in data:
                        print(f"Space-Track API Error: {data}")
                        
        except Exception as e:
            print(f"Error fetching real TLE: {e}")
            
    # 3. Fallback to Stale Cache if Live Failed (e.g. Rate Limit)
    if nid_str in TLE_CACHE:
        print(f"Live Fetch Failed. Using Stale Cache for {nid_str}.")
        entry = TLE_CACHE[nid_str]
        entry['source'] = "CACHE-STALE"
        return entry
        
            
    # Real Mode: No fallback.
    raise HTTPException(status_code=404, detail="Satellite ID not found in Space-Track query.")

class TleRequest(BaseModel):
    line1: str
    line2: str
    target_time: datetime
    solar_flux: float = 150.0
    kp_index: float = 3.0

@app.post("/predict")
def predict_orbit(request: TleRequest):
    return _predict_single(request)

class BatchRequest(BaseModel):
    line1: str
    line2: str
    start_time: str
    minutes_duration: int
    step_minutes: int
    solar_flux: float
    kp_index: float

@app.post("/predict_batch")
def predict_batch(request: BatchRequest):
    """Efficiently predict a sequence of points."""
    try:
        start_dt = datetime.fromisoformat(request.start_time.replace("Z", "+00:00"))
        steps = int(request.minutes_duration / request.step_minutes)
        timestamps = [start_dt + timedelta(minutes=i*request.step_minutes) for i in range(steps)]
        
        results = []
        results = []
        for ts in timestamps:
            try:
                # Re-use logic (refactored slightly for speed?)
                # Just calling internal function to avoid HTTP overhead
                pos_physics = propagator.get_position(request.line1, request.line2, ts)
                
                # Validate physics output
                if np.any(np.isnan(pos_physics)) or np.any(np.isinf(pos_physics)):
                    print(f"Skipping point {ts}: Physics propagation returned NaN/Inf")
                    continue
                
                # NORMALIZED INPUTS
                input_tensor = torch.tensor([[
                    request.solar_flux / 300.0, 
                    request.kp_index / 10.0, 
                    pos_physics[0] / 10000.0, 
                    pos_physics[1] / 10000.0, 
                    pos_physics[2] / 10000.0
                ]], dtype=torch.float32)
                
                with torch.no_grad():
                    # Output: 6 values (Mean + LogVar)
                    output = model(input_tensor).numpy()[0]
                
                correction = output[:3]
                log_var = output[3:]
                
                # Validate output for NaN/Inf (untrained model protection)
                if np.any(np.isnan(correction)) or np.any(np.isinf(correction)):
                    print(f"Warning: Model output NaN/Inf at {ts}. Using zero correction.")
                    correction = np.zeros(3)
                    
                if np.any(np.isnan(log_var)) or np.any(np.isinf(log_var)):
                    log_var = np.full(3, -5.0)  # Small uncertainty
                
                std_dev = np.sqrt(np.exp(np.clip(log_var, -10, 10)))  # Clip to prevent overflow
                uncert = float(np.linalg.norm(std_dev))
                
                # Validate uncertainty
                if np.isnan(uncert) or np.isinf(uncert):
                    uncert = 10.0  # Default 10km uncertainty
                    
                final_pos = pos_physics + correction
                
                # Final validation before JSON
                if np.any(np.isnan(final_pos)) or np.any(np.isinf(final_pos)):
                    print(f"Skipping point {ts}: Final position is NaN/Inf")
                    continue
                
                results.append({
                    "ts": ts.isoformat(),
                    "x": float(final_pos[0]), "y": float(final_pos[1]), "z": float(final_pos[2]),
                    "physics_x": float(pos_physics[0]), "physics_y": float(pos_physics[1]), "physics_z": float(pos_physics[2]),
                    "uncertainty_km": uncert
                })
            except ValueError as e:
                # SGP4 Error (e.g. decay). Skip this point but allow others if possible.
                print(f"Skipping point {ts} due to SGP4 error: {e}")
                continue
            except Exception as e:
                print(f"Skipping point {ts} due to error: {e}")
                continue

        if not results:
             raise HTTPException(status_code=400, detail="Prediction failed: No valid points generated.")
             
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def _predict_single(request: TleRequest):
    try:
        # 1. Physics Prediction (SGP4)
        pos_physics = propagator.get_position(request.line1, request.line2, request.target_time)
        
        # 2. ML Correction (Probabilistic)
        # NORMALIZATION: Flux/300, Kp/10, Pos/10000
        input_tensor = torch.tensor([[
            request.solar_flux / 300.0, 
            request.kp_index / 10.0, 
            pos_physics[0] / 10000.0, 
            pos_physics[1] / 10000.0, 
            pos_physics[2] / 10000.0
        ]], dtype=torch.float32)
        
        with torch.no_grad():
            output = model(input_tensor).numpy()[0] # [mu_x, mu_y, mu_z, logvar_x, logvar_y, logvar_z]
            
        correction = output[:3]
        log_var = output[3:]
        
        # Validate output for NaN/Inf (untrained model protection)
        if np.any(np.isnan(correction)) or np.any(np.isinf(correction)):
            print(f"Warning: Model output NaN/Inf. Using zero correction.")
            correction = np.zeros(3)
            
        if np.any(np.isnan(log_var)) or np.any(np.isinf(log_var)):
            log_var = np.full(3, -5.0)  # Small uncertainty
        
        variance = np.exp(np.clip(log_var, -10, 10))  # Clip to prevent overflow
        std_dev = np.sqrt(variance)
        
        uncertainty_scalar = float(np.linalg.norm(std_dev)) # Radius of uncertainty sphere

        # 3. Final Prediction
        final_pos = pos_physics + correction
        
        return {
            "x": float(final_pos[0]), "y": float(final_pos[1]), "z": float(final_pos[2]),
            "physics_x": float(pos_physics[0]), "physics_y": float(pos_physics[1]), "physics_z": float(pos_physics[2]),
            "correction_x": float(correction[0]), "correction_y": float(correction[1]), "correction_z": float(correction[2]),
            "uncertainty_km": uncertainty_scalar,
            "source": "HYBRID_PHYSICS_ML_PROBABILISTIC"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class RiskAnalysisRequest(BaseModel):
    sat_id: str
    debris_id: str
    tca: str

class ManeuverRequest(BaseModel):
    sat_tle: dict  # {"line1": ..., "line2": ...}
    debris_tle: dict
    tca: str  # ISO format

@app.post("/plan_maneuver")
def plan_maneuver(request: ManeuverRequest):
    """
    DeepDebris 3.0: Autonomous Maneuver Planning
    
    Uses trained RL agent to generate optimal collision avoidance maneuver.
    Returns thrust direction, timing, fuel cost, and new trajectory.
    """
    if maneuver_agent is None:
        raise HTTPException(
            status_code=503,
            detail="RL Maneuver Agent not loaded. Train with: python rl/train_maneuver_agent.py"
        )
    
    try:
        # Create environment with current scenario
        env = SpaceGym(
            sat_tle=request.sat_tle,
            debris_tle=request.debris_tle,
            tca=request.tca
        )
        
        # Get initial observation
        obs, _ = env.reset()
        
        # Agent predicts optimal action
        action, _states = maneuver_agent.predict(obs, deterministic=True)
        action = int(action)
        
        # Simulate maneuver to get new trajectory
        new_trajectory, fuel_cost, miss_distance = env.simulate_maneuver(action)
        
        # Convert action to human-readable format
        thrust_dir_name = action_to_vector(action)
        burn_duration = calculate_burn_time(action)
        exec_time = calculate_optimal_time(request.tca)
        
        return {
            "thrust_direction": thrust_dir_name,
            "thrust_action": action,
            "burn_duration_seconds": burn_duration,
            "execution_time_utc": exec_time,
            "fuel_cost_percent": fuel_cost,
            "new_miss_distance_km": miss_distance / 1000,
            "new_trajectory": [
                {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
                for pos in new_trajectory
            ],
            "status": "success"
        }
    except Exception as e:
        print(f"Maneuver Planning Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_risk")
def analyze_risk(request: RiskAnalysisRequest):
    """
    Perform deep analysis of a potential collision.
    Compare SGP4 (Physics) vs ResidualNet (AI) predictions.
    """
    try:
        # 1. Get TLEs
        sat_tle = get_latest_tle(int(request.sat_id)) # Auto-caches
        deb_axis = get_latest_tle(int(request.debris_id))
        
        tca_dt = datetime.fromisoformat(request.tca.replace("Z", "+00:00"))
        
        # 2. Physics Prediction (SGP4)
        sat_pos = propagator.get_position(sat_tle['line1'], sat_tle['line2'], tca_dt)
        deb_pos = propagator.get_position(deb_axis['line1'], deb_axis['line2'], tca_dt)
        
        physics_dist = np.linalg.norm(sat_pos - deb_pos)
        
        # 3. AI Prediction (Apply Correction)
        # We need flux/kp. Use defaults or fetch live.
        wx = weather_service.get_live_weather()
        # Fix: Match keys from weather_service.py ('flux', 'kp')
        flux = wx.get('flux', 150.0)
        kp = wx.get('kp', 3.0) 
        
        # Correct Satellite (normalize inputs, extract only correction)
        input_sat = torch.tensor([[
            flux / 300.0,
            kp / 10.0,
            sat_pos[0] / 10000.0,
            sat_pos[1] / 10000.0,
            sat_pos[2] / 10000.0
        ]], dtype=torch.float32)
        with torch.no_grad():
            output_sat = model(input_sat).numpy()[0]
        corr_sat = output_sat[:3]  # Extract only mean correction (first 3 values)
        sat_pos_ai = sat_pos + corr_sat
        
        # Correct Debris
        input_deb = torch.tensor([[
            flux / 300.0,
            kp / 10.0,
            deb_pos[0] / 10000.0,
            deb_pos[1] / 10000.0,
            deb_pos[2] / 10000.0
        ]], dtype=torch.float32)
        with torch.no_grad():
            output_deb = model(input_deb).numpy()[0]
        corr_deb = output_deb[:3]  # Extract only mean correction (first 3 values)
        deb_pos_ai = deb_pos + corr_deb
        
        ai_dist = np.linalg.norm(sat_pos_ai - deb_pos_ai)
        
        # 4. Assessment
        diff = physics_dist - ai_dist
        risk_reduction = 0.0
        if physics_dist > 0:
            risk_reduction = ((physics_dist - ai_dist) / physics_dist) * 100
            
        rec = "MONITOR"
        if ai_dist < 1.0: rec = "MANEUVER REQUIRED (CRITICAL)"
        elif ai_dist < 10.0: rec = "HIGH ALERT"
        elif ai_dist > physics_dist: rec = "FALSE ALARM (AI Cleared)"
        
        return {
            "tca": request.tca,
            "physics_miss_distance_km": float(physics_dist),
            "ai_miss_distance_km": float(ai_dist),
            "risk_reduction_percent": float(risk_reduction),
            "recommendation": rec
        }
        
    except Exception as e:
        print(f"Risk Analysis Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Static Files (Mount LAST to avoid masking API routes)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Start Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



