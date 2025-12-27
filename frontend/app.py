import streamlit as st
import pydeck as pdk
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="DeepDebris Dashboard", layout="wide")

st.title("🛰️ DeepDebris: Physics-Informed Space Debris Tracking")

st.markdown("""
### 🌍 The Problem: Space Junk & Solar Storms
There are over **30,000 pieces of debris** orbiting Earth. When the Sun gets angry (Solar Storms), the atmosphere "puffs up" and drags them down.

*   **The Issue**: Standard physics models (SGP4) *cannot predict* this sudden drag accurately. They get the position wrong by kilometers!
*   **The Risk**: If we don't know where debris is, it can **smash into the ISS** or destroy GPS satellites.

### 🧠 The Solution: DeepDebris
We use AI to **learn** from past storms. By combining standard physics with a Neural Network, we correct the error in real-time.

**Legend**:
*   🔴 **Red Line (Physics)**: "Where the math *thinks* the satellite is going." (Often wrong during storms)
*   🟢 **Green Line (AI)**: "Where the satellite is *actually* going." (Corrected for atmospheric drag)
""")

# Sidebar
st.sidebar.title("🎮 Control Panel")

st.sidebar.header("Step 1: Choose Target 🛰️")

# --- GLOBAL SETTINGS (Moved to top for visibility) ---
st.sidebar.markdown("---")

data_source = st.sidebar.radio("Data Source", ["Manual Input", "Live Space-Track Data"], label_visibility="collapsed")

if data_source == "Live Space-Track Data":
    sat_choice = st.sidebar.selectbox("Select Satellite", ["ISS (ZARYA)", "HUBBLE SPACE TELESCOPE"])
    sat_id = "25544" if "ISS" in sat_choice else "20580"
    
    if st.sidebar.button("Fetch TLE Data"):
        try:
            with st.spinner(f"Connecting to USSTRATCOM..."):
                # Call ML Service to get TLE (it handles Auth or Mock)
                resp = requests.get(f"http://localhost:8000/tle/{sat_id}")
                resp.raise_for_status()
                tle_data = resp.json()
                
                # Store in session state to persist
                st.session_state['line1'] = tle_data['line1']
                st.session_state['line2'] = tle_data['line2']
                st.session_state['source_info'] = f"Source: {tle_data.get('source', 'Unknown')} | Epoch: {tle_data.get('epoch', 'N/A')}"
                st.sidebar.success("Fetched!")
        except Exception as e:
            st.sidebar.error(f"Fetch failed: {e}")


    # Use session state or default
    def_line1 = st.session_state.get('line1', "1 25544U 98067A   23356.54321689  .00016717  00000+0  30283-3 0  9997")
    def_line2 = st.session_state.get('line2', "2 25544  51.6416  21.9684 0005432  35.2163  86.1264 15.49507156430342")
    
    if 'source_info' in st.session_state:
        st.sidebar.caption(st.session_state['source_info'])

else:
    def_line1 = "1 25544U 98067A   23356.54321689  .00016717  00000+0  30283-3 0  9997"
    def_line2 = "2 25544  51.6416  21.9684 0005432  35.2163  86.1264 15.49507156430342"

st.sidebar.markdown("---")
st.sidebar.header("Inputs")

# Tooltips for TLE
tle_line1 = st.sidebar.text_input(
    "TLE Line 1", 
    def_line1, 
    help="The satellite's 'Line 1' address. Contains ID and launch year."
)
tle_line2 = st.sidebar.text_input(
    "TLE Line 2", 
    def_line2, 
    help="The satellite's 'Line 2' address. Contains orbit shape and inclination."
)

st.sidebar.markdown("---")
st.sidebar.markdown("---")
st.sidebar.header("Step 2: Set Weather ☀️")

# Initialize Session State for Weather
if 'flux' not in st.session_state: st.session_state['flux'] = 150.0
if 'kp' not in st.session_state: st.session_state['kp'] = 3.0
if 'scenario' not in st.session_state: st.session_state['scenario'] = "Custom"

def update_weather_scenario():
    """Callback to update Flux/Kp based on Scenario"""
    scen = st.session_state['scenario']
    
    if scen == "Live NOAA Data (Real-Time)":
        try:
            with st.spinner("Fetching live NOAA data..."):
                resp = requests.get("http://localhost:8000/weather/live")
                data = resp.json()
                st.session_state['flux'] = data['flux']
                st.session_state['kp'] = data['kp']
                st.toast(f"Updated from NOAA: Flux={data['flux']}, Kp={data['kp']}")
        except Exception as e:
            st.error(f"NOAA Fetch Failed: {e}")
            
    elif scen == "Quiet Sun (Default)":
        st.session_state['flux'] = 70.0
        st.session_state['kp'] = 1.0
    elif scen == "Moderate Storm":
        st.session_state['flux'] = 180.0
        st.session_state['kp'] = 5.0
    elif scen == "Extreme Storm (The Problem)":
        st.session_state['flux'] = 250.0
        st.session_state['kp'] = 8.0

def reset_defaults():
    """Callback to reset everything"""
    st.session_state['flux'] = 150.0
    st.session_state['kp'] = 3.0
    st.session_state['scenario'] = "Custom"

# Scenario Selector
st.sidebar.selectbox(
    "Weather Scenario",
    ["Custom", "Live NOAA Data (Real-Time)", "Quiet Sun (Default)", "Moderate Storm", "Extreme Storm (The Problem)"],
    key="scenario",
    on_change=update_weather_scenario,
    help="Select a preset to see how weather affects the orbit."
)
# No inline 'if' logic here anymore

# Sliders (linked to session state)
flux = st.sidebar.slider(
    "Solar Flux (F10.7)", 
    50.0, 300.0, 
    key="flux",
    help="Energy from the Sun. Higher = Thicker Atmosphere = More Drag."
)

kp = st.sidebar.slider(
    "Geomagnetic Index (Kp)", 
    0.0, 9.0, 
    key="kp",
    help="Magnetic Storm Level. Higher = More Drag (0=Calm, 9=Apocalyptic)."
)

# Reset Button
# Reset Button
st.sidebar.button("↺ Reset to Default", on_click=reset_defaults)

if st.button("🚀 Step 3: Run Prediction (Next 3 Orbits)"):
    # Generate points for 3 orbits (~270 mins) with high resolution (3 min steps)
    
    progress_bar = st.progress(0)
    
    payload = {
        "line1": tle_line1,
        "line2": tle_line2,
        "start_time": datetime.utcnow().isoformat(),
        "minutes_duration": 270, # 4.5 hours
        "step_minutes": 3,
        "solar_flux": flux,
        "kp_index": kp
    }
    
    try:
        with st.spinner("Calculating high-resolution trajectory..."):
            response = requests.post("http://localhost:8000/predict_batch", json=payload)
            response.raise_for_status()
            batch_data = response.json()
            
            # Store in Session State
            st.session_state['prediction_data'] = batch_data
            st.session_state['sat_choice'] = sat_choice # Persist name too
                
        progress_bar.progress(100)
    except Exception as e:
        st.error(f"Error fetching prediction: {e}")

# Check if data exists in Session State to render
if 'prediction_data' in st.session_state and st.session_state['prediction_data']:
    batch_data = st.session_state['prediction_data']
    
    # Reconstruct lists from state
    points_physics = []
    points_corrected = []
    for pt in batch_data:
        points_physics.append({"x": pt["physics_x"], "y": pt["physics_y"], "z": pt["physics_z"]})
        points_corrected.append({"x": pt["x"], "y": pt["y"], "z": pt["z"]})

    # Plotly 3D Visualization
    import plotly.graph_objects as go

    phys_df = pd.DataFrame(points_physics)
    corr_df = pd.DataFrame(points_corrected)

    fig = go.Figure()

    # Physics Path (Dashed Red)
    fig.add_trace(go.Scatter3d(
        x=phys_df['x'], y=phys_df['y'], z=phys_df['z'],
        mode='lines', # No markers, just smooth line
        name='Physics (SGP4)',
        line=dict(color='red', width=5, dash='dash'), # Dashed line
    ))

    # Corrected Path (Solid Green)
    fig.add_trace(go.Scatter3d(
        x=corr_df['x'], y=corr_df['y'], z=corr_df['z'],
        mode='lines',
        name='ML-Corrected',
        line=dict(color='#00FF00', width=6),
        hovertemplate="<b>Time</b>: %{text}<br><b>Pos</b>: %{x:.1f}, %{y:.1f}, %{z:.1f} km<extra></extra>",
        text=[pt.get('ts', 'N/A') for pt in batch_data] # Pass timestamps for hover
    ))

    # Current Position Marker (Start)
    fig.add_trace(go.Scatter3d(
        x=[corr_df['x'].iloc[0]], y=[corr_df['y'].iloc[0]], z=[corr_df['z'].iloc[0]],
        mode='markers',
        name='Current Position',
        marker=dict(size=10, color='yellow', symbol='diamond'),
        hovertemplate="<b>Current Position</b><br>%{x:.1f}, %{y:.1f}, %{z:.1f} km<extra></extra>"
    ))

    # --- 3. Uncertainty Ellipsoid (Visual Polish) ---
    
    # --- REALISTIC EARTH (Texture Mapping Fix) ---
    r_earth = 6371
    
    # 1. Load Earth Image
    from PIL import Image
    import numpy as np
    
    @st.cache_data
    def load_texture(filepath):
        try:
            img = Image.open(filepath)
            return img
        except:
            return None

    img_raw = load_texture("assets/earth.jpg")
    
    # Use proper resolution for the texture
    N_PHI = 200
    N_THETA = 100
    
    if img_raw:
        img_resized = img_raw.resize((N_PHI, N_THETA))
        img_gray = img_resized.convert('L')
        surface_color = np.array(img_gray)
        
        # Colorscale: Dark Blue -> Blue -> Green -> Brown -> White
        earth_colorscale = [
            [0.0, 'rgb(0,0,20)'], 
            [0.2, 'rgb(0,0,100)'],
            [0.4, 'rgb(0,100,50)'], 
            [0.5, 'rgb(150,150,100)'], 
            [1.0, 'rgb(255,255,255)']
        ]
    else:
        surface_color = np.zeros((N_THETA, N_PHI))
        earth_colorscale = 'Viridis'

    # 2. Sphere Mesh
    phi = np.linspace(0, 2*np.pi, N_PHI)
    theta = np.linspace(0, np.pi, N_THETA)
    phi, theta = np.meshgrid(phi, theta)
    
    x_earth = r_earth * np.cos(phi) * np.sin(theta)
    y_earth = r_earth * np.sin(phi) * np.sin(theta)
    z_earth = r_earth * np.cos(theta)

    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        surfacecolor=surface_color,
        colorscale=earth_colorscale,
        showscale=False,
        opacity=1.0,
        lighting=dict(ambient=0.4, diffuse=0.6, specular=0.1, roughness=0.5),
        name='Earth'
    ))

    # --- STARFIELD ---
    import random
    r_stars = 30000 
    star_x = [random.uniform(-r_stars, r_stars) for _ in range(500)]
    star_y = [random.uniform(-r_stars, r_stars) for _ in range(500)]
    star_z = [random.uniform(-r_stars, r_stars) for _ in range(500)]

    fig.add_trace(go.Scatter3d(
        x=star_x, y=star_y, z=star_z,
        mode='markers',
        marker=dict(size=2, color='white', opacity=0.8),
        hoverinfo='none',
        name='Stars'
    ))

    # --- ANIMATION FRAMES (Client-Side / No Blinking) ---
    # Instead of a Python loop, we build frames for Plotly JS to play.
    
    frames = []
    # Subsample points for performance (every 2nd point)
    anim_points = points_corrected[::2] 
    
    for k, pt in enumerate(anim_points):
        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=[pt['x']], y=[pt['y']], z=[pt['z']]) # Updates Trace 2 (Marker)
            ],
            traces=[2], # IMPORTANT: Only update the 3rd trace (Index 2)
            name=f"frame{k}",
            layout=go.Layout(title_text=f"Time: {pt.get('ts', 'N/A')}")
        ))
    
    # --- VISUALIZATION CONTROLS ---
    st.markdown("### 🎮 Simulation Mode")
    mode_cols = st.columns([2, 1, 1])
    with mode_cols[0]:
        sim_mode = st.radio(
            "Select Mode:", 
            ["🔮 Prediction Preview (Fast)", "🔴 Live Real-Time (1x)"], 
            horizontal=True,
            label_visibility="collapsed"
        )
    
    real_time_tracking = "Live" in sim_mode

    # --- SIMULATION BUTTON (Client-Side) ---
    # Only show the "Play" button if we are NOT in Real-Time mode (to avoid conflicts)
    if not real_time_tracking:
        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0.1, x=0.1, xanchor="right", yanchor="top",
                buttons=[dict(
                    label="▶ Fast-Forward Orbit",
                    method="animate",
                    args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]
                )]
            )]
        )
    else:
        # In real-time mode, remove the play button
        fig.update_layout(updatemenus=[])

    # --- METRICS & LAYOUT ---
    diff = np.sqrt(
        (phys_df['x'] - corr_df['x'])**2 + 
        (phys_df['y'] - corr_df['y'])**2 + 
        (phys_df['z'] - corr_df['z'])**2
    )
    max_correction_km = diff.max()
    avg_correction_km = diff.mean()

    # UI Layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Max Correction Applied", f"{max_correction_km:.2f} km", delta="AI Impact")
    with col2:
        st.metric("Avg Deviation from Physics", f"{avg_correction_km:.2f} km")
    with col3:
        st.metric("Prediction Window", "4.5 Hours", "Forward Prop")

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(r=0, b=0, l=0, t=50),
        legend=dict(
            yanchor="top", y=0.95, 
            xanchor="left", x=0.02,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white", size=14)
        ),
        title=dict(
            text=f"Orbit Visualization: {sat_choice} ({'Real-Time' if real_time_tracking else 'Prediction View'})",
            y=0.9, x=0.5, xanchor='center', yanchor='top'
        )
    )

    # --- RENDERING LOGIC ---
    # We use st.rerun() for the Real-Time loop to avoid 'DuplicateElementKey' errors 
    # and to ensure the camera state is preserved via the stable key.
    
    if real_time_tracking:
        from datetime import datetime, timezone
        import time
        
        # Helper to parse Start Time from data
        try:
            start_ts_str = points_corrected[0]['ts'].replace('Z', '+00:00')
            start_ts = datetime.fromisoformat(start_ts_str)
        except:
            start_ts = datetime.now(timezone.utc)

        # 1. Get Time
        now = datetime.now(timezone.utc)
        
        # 2. Find Index with Interpolation
        seconds_per_step = 3 * 60 # 180s
        elapsed_seconds = (now - start_ts).total_seconds()
        
        # Base index
        idx = int(elapsed_seconds / seconds_per_step)
        
        # Fraction for interpolation
        alpha = (elapsed_seconds % seconds_per_step) / seconds_per_step
        
        # Cyclic Logic
        idx = idx % len(points_corrected)
        next_idx = (idx + 1) % len(points_corrected)
        
        p1 = points_corrected[idx]
        p2 = points_corrected[next_idx]
        
        # Linear Interpolation (LERP)
        curr_pt = {
            'x': p1['x'] * (1 - alpha) + p2['x'] * alpha,
            'y': p1['y'] * (1 - alpha) + p2['y'] * alpha,
            'z': p1['z'] * (1 - alpha) + p2['z'] * alpha
        }
        
        # Clone figure for this frame
        fig_rt = go.Figure(fig)
        
        # Update Marker (Trace 2)
        fig_rt.data[2].x = [curr_pt['x']]
        fig_rt.data[2].y = [curr_pt['y']]
        fig_rt.data[2].z = [curr_pt['z']]
        
        # Update Title
        fig_rt.layout.title.text = f"LIVE TRACKING | {sat_choice} | {now.strftime('%H:%M:%S UTC')}"
        
        # Preserve State
        fig_rt.update_layout(uirevision="RealTimeLock")
        
        # RENDER with STABLE KEY
        st.plotly_chart(fig_rt, use_container_width=True, key="live_chart_stable")
        
        # Loop via Rerun
        time.sleep(1.0)
        st.rerun()
            
    else:
        # Static Rendering (with Play Button enabled)
        st.plotly_chart(fig, use_container_width=True, key="static_chart_stable")
        st.caption("ℹ️ **Mode: Prediction View**. Use the **'▶ Fast-Forward Orbit'** button above to preview the path.")

    # --- Metrics & Details ---
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    radius = 40.0 # 40km uncertainty radius (example)
    
    # Center at the last predicted point
    cx, cy, cz = corr_df['x'].iloc[-1], corr_df['y'].iloc[-1], corr_df['z'].iloc[-1]
    
    xx = cx + radius * np.outer(np.cos(u), np.sin(v))
    yy = cy + radius * np.outer(np.sin(u), np.sin(v))
    zz = cz + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz,
        opacity=0.3,
        colorscale='Reds',
        showscale=False,
        name='Uncertainty (95% CI)'
    ))
    
    # Calculate difference magnitude
    diff = np.sqrt(
        (phys_df['x'] - corr_df['x'])**2 + 
        (phys_df['y'] - corr_df['y'])**2 + 
        (phys_df['z'] - corr_df['z'])**2
    )
    max_correction_km = diff.max()
    avg_correction_km = diff.mean()

    # --- UI Layout ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Max Correction Applied", f"{max_correction_km:.2f} km", delta="AI Impact")
    with col2:
        st.metric("Avg Deviation from Physics", f"{avg_correction_km:.2f} km")
    with col3:
        st.metric("Prediction Window", "3 Days", "Forward Prop")

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data' # Important to keep sphere spherical
        ),
        margin=dict(r=0, b=0, l=0, t=50),
        legend=dict(
            yanchor="top", y=0.95, 
            xanchor="left", x=0.02,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white", size=14)
        ),
        title=dict(
            text=f"Orbit Visualization: {sat_choice}",
            y=0.9, x=0.5, xanchor='center', yanchor='top'
        )
    )

    # REMOVED: Redundant static chart render was deleted here to prevent "2 Graphs" bug.
    
    with st.expander("ℹ️ Technical Details (Click to Expand)"):
        st.markdown(f"""
        - **Target Satellite**: {sat_choice} (NORAD ID: {sat_id})
        - **Solar Flux (F10.7)**: {flux} sfu (Higher flux = More drag)
        - **Geomagnetic Index (Kp)**: {kp} (Higher Kp = Geomagnetic storm)
        - **Model Architecture**: Residual Neural Network (PyTorch) trained on historical residuals.
        - **Input Features**: `[Solar Flux, Kp, Physics_X, Physics_Y, Physics_Z]`
        - **Output**: `[Delta_X, Delta_Y, Delta_Z]` correction vector.
        """)
    
    st.success("Trajectory prediction complete!")
    

else:
    st.info("Click 'Predict Trajectory' to see the visualization.")

# --- 4. OrbitGPT Section ---
st.markdown("---")
st.header("🤖 OrbitGPT: Flight Dynamics Assistant")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask OrbitGPT (e.g., 'Summarize high risk alerts')..."):
    # User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # AI Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner("Analyzing CDMs..."):
                # Call RAG Endpoint
                resp = requests.post("http://localhost:8000/chat", json={"query": prompt})
                full_response = resp.json().get("response", "Error: No response from OrbitGPT")
        except Exception as e:
            full_response = f"Error connecting to OrbitGPT: {e}"
            
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
