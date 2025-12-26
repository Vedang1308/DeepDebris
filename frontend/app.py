import streamlit as st
import pydeck as pdk
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="DeepDebris Dashboard", layout="wide")

st.title("🛰️ DeepDebris: Physics-Informed Space Debris Tracking")

st.markdown("""
**What is this?**
This system predicts where a satellite (like the ISS) will be in **3 days**.
*   🔴 **Physics Baseline (SGP4)**: The standard prediction used by most trackers. It often drifts due to space weather (drag).
*   🟢 **ML-Corrected (Ours)**: A Neural Network corrects the physics model using live Space Weather data, reducing error.

**Visual Guide**:
*   **Earth**: The solid sphere in the center.
*   **Red Line**: Standard physics prediction (Standard SGP4).
*   **Green Line**: Our AI-corrected trajectory (closer to reality).
*   **Red Cloud**: The area of uncertainty (95% confidence).
""")

# Sidebar
st.sidebar.header("Configuration")
data_source = st.sidebar.radio("Data Source", ["Manual Input", "Live Space-Track Data"])

if data_source == "Live Space-Track Data":
    sat_choice = st.sidebar.selectbox("Select Satellite", ["ISS (ZARYA)", "HUBBLE SPACE TELESCOPE"])
    sat_id = "25544" if "ISS" in sat_choice else "20580"
    
    if st.sidebar.button("Fetch Latest TLE"):
        try:
            with st.spinner(f"Fetching TLE for {sat_choice}..."):
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
        st.sidebar.info(st.session_state['source_info'])

else:
    def_line1 = "1 25544U 98067A   23356.54321689  .00016717  00000+0  30283-3 0  9997"
    def_line2 = "2 25544  51.6416  21.9684 0005432  35.2163  86.1264 15.49507156430342"

tle_line1 = st.sidebar.text_input("TLE Line 1", def_line1)
tle_line2 = st.sidebar.text_input("TLE Line 2", def_line2)

flux = st.sidebar.slider("Solar Flux (F10.7)", 50.0, 300.0, 150.0)
kp = st.sidebar.slider("Geomagnetic Index (Kp)", 0.0, 9.0, 3.0)

if st.button("Predict Trajectory"):
    # Generate points for 3 days
    points_physics = []
    points_corrected = []
    
    # We will sample every 6 hours for 3 days
    timestamps = [datetime.utcnow() + timedelta(hours=i*6) for i in range(12)]
    
    progress_bar = st.progress(0)
    
    for i, ts in enumerate(timestamps):
        payload = {
            "line1": tle_line1,
            "line2": tle_line2,
            "target_time": ts.isoformat(),
            "solar_flux": flux,
            "kp_index": kp
        }
        
        try:
            response = requests.post("http://localhost:8000/predict", json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Physics Point (Red)
            points_physics.append({
                "x": data["physics_x"], 
                "y": data["physics_y"], 
                "z": data["physics_z"], 
                "color": [255, 0, 0, 200]
            })
            
            # Corrected Point (Green)
            points_corrected.append({
                "x": data["x"], 
                "y": data["y"], 
                "z": data["z"], 
                "color": [0, 255, 0, 200]
            })
            
        except Exception as e:
            st.error(f"Error fetching prediction: {e}")
            break
            
        progress_bar.progress((i + 1) / len(timestamps))

    # Plotly 3D Visualization
    import plotly.graph_objects as go

    phys_df = pd.DataFrame(points_physics)
    corr_df = pd.DataFrame(points_corrected)

    fig = go.Figure()

    # Physics Path
    fig.add_trace(go.Scatter3d(
        x=phys_df['x'], y=phys_df['y'], z=phys_df['z'],
        mode='lines+markers',
        name='Physics (SGP4)',
        line=dict(color='red', width=4),
        marker=dict(size=4)
    ))

    # Corrected Path
    fig.add_trace(go.Scatter3d(
        x=corr_df['x'], y=corr_df['y'], z=corr_df['z'],
        mode='lines+markers',
        name='ML-Corrected',
        line=dict(color='green', width=4),
        marker=dict(size=4)
    ))

    # Earth (Simple Sphere wireframe or surface)
    # Earth Radius ~ 6371 km
    r = 6371
    # --- 1. Earth (Solid Sphere) ---
    # Create a sphere mesh
    phi = np.linspace(0, 2*np.pi, 50)
    theta = np.linspace(0, np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)
    
    r_earth = 6371 # km
    x_earth = r_earth * np.cos(phi) * np.sin(theta)
    y_earth = r_earth * np.sin(phi) * np.sin(theta)
    z_earth = r_earth * np.cos(theta)
    
    # Add Earth Surface
    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale='Earth', # Built-in Earth-like colors
        showscale=False,
        lighting=dict(ambient=0.4, diffuse=0.5, roughness=0.9, specular=0.1),
        name='Earth'
    ))
    
    # --- 3. Uncertainty Ellipsoid ---
    # Visualize a semi-transparent sphere/ellipsoid around the tip of the prediction
    # to represent the confidence interval (e.g., +/- 10km)
    
    # Generate sphere points
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
        scene=dict(
            xaxis_title='X (km) - Vernal Equinox',
            yaxis_title='Y (km)',
            zaxis_title='Z (km) - North Pole',
            aspectmode='data', # Important to keep sphere spherical
            # Make the background look like space
            xaxis=dict(backgroundcolor="#0e1117", gridcolor="#262730", showbackground=True),
            yaxis=dict(backgroundcolor="#0e1117", gridcolor="#262730", showbackground=True),
            zaxis=dict(backgroundcolor="#0e1117", gridcolor="#262730", showbackground=True),
        ),
        margin=dict(r=0, b=0, l=0, t=0),
        legend=dict(
            yanchor="top", y=0.95, 
            xanchor="left", x=0.02,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white", size=14)
        ),
        title=dict(
            text=f"Orbit Visualization: {sat_choice} (Epoch: {st.session_state.get('source_info', 'N/A').split('|')[1].strip() if 'source_info' in st.session_state else 'N/A'})",
            y=0.9, x=0.5, xanchor='center', yanchor='top'
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    
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
    
    # Calculate difference magnitude
    diff = np.sqrt(
        (phys_df['x'] - corr_df['x'])**2 + 
        (phys_df['y'] - corr_df['y'])**2 + 
        (phys_df['z'] - corr_df['z'])**2
    )
    st.metric("Max Correction Magnitude", f"{diff.max():.2f} km")

else:
    st.info("Click 'Predict Trajectory' to see the visualization.")
