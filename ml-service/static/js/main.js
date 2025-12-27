import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'https://esm.sh/three/addons/renderers/CSS2DRenderer.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';

// --- Debugging ---
console.log("DeepDebris 2.0 Starting...");

// --- Scene Setup ---
const scene = new THREE.Scene();
window.scene = scene; // Expose for debugging

// Camera
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100000); // Far clip for stars
camera.position.set(12000, 5000, 12000); // View from high altitude

// Renderer (WebGL)
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
document.getElementById('canvas-container').appendChild(renderer.domElement);

// Renderer (Labels)
const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(window.innerWidth, window.innerHeight);
labelRenderer.domElement.style.position = 'absolute';
labelRenderer.domElement.style.top = '0px';
labelRenderer.domElement.style.pointerEvents = 'none'; // Passthrough to allow canvas interaction
document.body.appendChild(labelRenderer.domElement);

// Controls (Orbit) - Bind to WebGL Canvas
const controls = new OrbitControls(camera, renderer.domElement); // Bind to canvas underneath
controls.enableDamping = true; // Smooth momentum
// ... (controls config) ...
controls.dampingFactor = 0.05;
controls.screenSpacePanning = false;
controls.minDistance = 6400;
controls.maxDistance = 50000;
controls.zoomSpeed = 0.6;
controls.rotateSpeed = 0.8;

// ... (Lights & Meshes unchanged) ...

// --- Global UI Elements (Hoisted) ---
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');
const chatHistory = document.getElementById('chat-history');
const chatToggle = document.getElementById('chat-toggle');
const btnFetch = document.getElementById('btn-fetch-tle');
const selector = document.getElementById('sat-selector');
const inputs = {
    l1: document.getElementById('tle1'),
    l2: document.getElementById('tle2'),
    alt: document.getElementById('alt-val'),
    vel: document.getElementById('vel-val')
};
const weatherSelect = document.getElementById('weather-scenario');
const fluxSlider = document.getElementById('flux-slider');
const kpSlider = document.getElementById('kp-slider');
let predictionLine;
let debrisPredictionLine;
let simulatedTime = new Date(); // Hoisted critical variable
let timeScale = 1.0;
let isLive = true;

// Helper: Convert SGP4 ECI (Z-up) to Three.js (Y-up)
function toVector3(p) {
    return new THREE.Vector3(p.x, p.z, -p.y);
}

// --- Label Helper ---
function createLabel(text, type = 'default') {
    const div = document.createElement('div');
    div.className = 'label-tag';
    if (type === 'prediction') div.classList.add('label-prediction');
    if (type === 'debris') div.classList.add('label-debris');
    div.textContent = text;
    return new CSS2DObject(div);
}

// ... (Rest of file) ...

// (Garbage removed)

// ...

if (weatherSelect) {
    weatherSelect.addEventListener('change', async (e) => {
        const val = e.target.value;
        let newFlux = 150;
        let newKp = 3;

        // "LIVE" Logic
        if (val === 'LIVE') {
            try {
                const resp = await fetch('/weather/live');
                if (resp.ok) {
                    const data = await resp.json();
                    newFlux = data.flux;
                    newKp = data.kp;
                }
            } catch (err) { console.warn("Live weather fetch failed", err); }
        }
        else if (val === 'MINIMUM') { newFlux = 65; newKp = 0; }
        else if (val === 'QUIET') { newFlux = 70; newKp = 1; }
        else if (val === 'NORMAL') { newFlux = 150; newKp = 3; }
        else if (val === 'MINOR') { newFlux = 180; newKp = 5; }
        else if (val === 'STORM') { newFlux = 300; newKp = 7; }
        else if (val === 'EXTREME') { newFlux = 400; newKp = 9; }

        if (fluxSlider) {
            fluxSlider.value = newFlux;
            document.getElementById('flux-disp').innerText = newFlux;
        }
        if (kpSlider) {
            kpSlider.value = newKp;
            document.getElementById('kp-disp').innerText = newKp;
        }
        // Force update prediction on scenario change
        const l1 = inputs.l1.value;
        const l2 = inputs.l2.value;
        if (l1 && l2) drawPredictionPath(l1, l2);
    });
}
// ...
const ambientLight = new THREE.AmbientLight(0xffffff, 1.5); // BRIGHTER for visibility
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xffffff, 2.0);
sunLight.position.set(50000, 10000, 20000);
scene.add(sunLight);

// ... (Existing Three.js Setup) ...

// --- UI Logic: Chat & TLE ---
// (Moved to top)

// Chat Toggle
chatToggle.addEventListener('click', () => {
    document.getElementById('chat-widget').classList.toggle('minimized');
});

// Send Message
async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    // Add User Message
    addMessage(text, 'user');
    chatInput.value = '';

    // Call Backend (OrbitGPT)
    // Call Backend (OrbitGPT)
    try {
        // Show Thinking Indicator
        const thinkingDiv = document.createElement('div');
        thinkingDiv.className = 'msg ai thinking';
        thinkingDiv.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Analyzing...';
        chatHistory.appendChild(thinkingDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text })
        });
        const data = await response.json();

        // Remove thinking indicator
        chatHistory.removeChild(thinkingDiv);

        addMessage(data.response, 'ai');
    } catch (err) {
        // cleanup
        const thinking = chatHistory.querySelector('.thinking');
        if (thinking) chatHistory.removeChild(thinking);

        addMessage("⚠️ System Offline. OrbitGPT Unavailable.", 'ai');
    }
}

function addMessage(text, sender) {
    const div = document.createElement('div');
    div.className = `msg ${sender}`;
    div.innerText = text;
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

chatSend.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// (Mock TLE Listener Removed)

// --- Earth ---
const textureLoader = new THREE.TextureLoader();
const earthRadius = 6371;
const earthGeo = new THREE.SphereGeometry(earthRadius, 64, 64);
// Restoration: Use PhongMaterial for realistic lighting
const earthMat = new THREE.MeshPhongMaterial({
    map: textureLoader.load('/assets/earth.jpg'),
    specular: 0x333333,
    shininess: 5
});
const earth = new THREE.Mesh(earthGeo, earthMat);
scene.add(earth);


// ... (Atmosphere & Stars remain same) ...

// --- UI Logic: Chat & TLE ---
// ... (Chat logic remains same) ...

// Fetch TLE (REAL)
btnFetch.addEventListener('click', async () => {
    const btn = btnFetch;
    const originalText = btn.innerHTML;
    const satId = document.getElementById('sat-selector').value;

    if (satId === 'CUSTOM') {
        alert("Cannot fetch for Custom ID. Select ISS or Hubble.");
        return;
    }

    btn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Fetching...`;

    try {
        const response = await fetch(`/tle/${satId}`);
        if (!response.ok) throw new Error("API Error");
        const data = await response.json();

        // Update Inputs
        inputs.l1.value = data.line1;
        inputs.l2.value = data.line2;

        // Trigger Engine Update
        updateSatelliteEngine();

        btn.innerHTML = `<i class="fas fa-check"></i> Updated`;
        setTimeout(() => btn.innerHTML = originalText, 2000);
    } catch (err) {
        console.error(err);
        btn.innerHTML = `<i class="fas fa-exclamation-triangle"></i> Failed`;
    }
});

// --- Atmosphere Halo (Simple) ---
const atmoGeo = new THREE.SphereGeometry(earthRadius + 100, 64, 64);
const atmoMat = new THREE.MeshBasicMaterial({
    color: 0x4ca6ff,
    transparent: true,
    opacity: 0.15,
    side: THREE.BackSide,
    blending: THREE.AdditiveBlending
});
const atmosphere = new THREE.Mesh(atmoGeo, atmoMat);
scene.add(atmosphere);

// --- Stars ---
function createStars() {
    const geometry = new THREE.BufferGeometry();
    const vertices = [];
    for (let i = 0; i < 10000; i++) { // More stars
        const x = THREE.MathUtils.randFloatSpread(200000);
        const y = THREE.MathUtils.randFloatSpread(200000);
        const z = THREE.MathUtils.randFloatSpread(200000);
        vertices.push(x, y, z);
    }
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    const material = new THREE.PointsMaterial({ color: 0xffffff, size: 150 });
    const stars = new THREE.Points(geometry, material);
    scene.add(stars);
}
createStars();

// --- Satellite Setup ---
const satGeo = new THREE.SphereGeometry(150, 16, 16);
const satMat = new THREE.MeshBasicMaterial({ color: 0xFF00FF }); // Magenta to match Physics/Legend
const satelliteMesh = new THREE.Mesh(satGeo, satMat);
scene.add(satelliteMesh);

// --- Ghost Satellite (AI Prediction) ---
const ghostGeo = new THREE.SphereGeometry(150, 16, 16);
const ghostMat = new THREE.MeshBasicMaterial({
    color: 0x00FFFF,
    transparent: true,
    opacity: 0.4,
    wireframe: true
});
const ghostMesh = new THREE.Mesh(ghostGeo, ghostMat);
scene.add(ghostMesh);

// Ghost Label
const ghostLabel = createLabel("AI PREDICTION", "warning");
ghostLabel.element.style.opacity = "0.7";
ghostLabel.element.style.fontSize = "10px";
ghostMesh.add(ghostLabel);
scene.add(ghostLabel); // Revert: Add to scene for proper scaling? No, typically add to mesh.
// Actually, if added to scene, we must update position. Only Mesh.add works for auto-tracking.
scene.remove(ghostLabel);
ghostMesh.add(ghostLabel); // Ensure it's child

// Prediction Data Store
let predictionData = [];
let predictionStartTime = null;



// --- Orbit Line (Trajectory) ---
let orbitLine;
function updateOrbitPath(satrec) {
    if (!satrec) return; // Guard
    if (orbitLine) scene.remove(orbitLine);

    const points = [];
    const now = new Date();
    // Calculate path for next 95 mins (1 orbit)
    for (let i = 0; i <= 95; i++) {
        const t = new Date(now.getTime() + i * 60000); // +1 min per step
        const positionAndVelocity = satellite.propagate(satrec, t);
        if (!positionAndVelocity.position) continue;
        const p = positionAndVelocity.position;
        if (!isNaN(p.x)) points.push(toVector3(p));
    }

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineDashedMaterial({
        color: 0xFF00FF, // Magenta (SGP4 Physics)
        dashSize: 500, // Larger dashes to be visible
        gapSize: 300,
        linewidth: 2,
        opacity: 0.8,
        transparent: true
    });
    orbitLine = new THREE.Line(geometry, material);
    orbitLine.computeLineDistances(); // REQUIRED for dashes
    scene.add(orbitLine);

    // Label Logic
    if (orbitLine.userData.label) scene.remove(orbitLine.userData.label);
    if (points.length > 0) {
        // Add label to end
    }
}
// (Orphaned logic cleaned up)

// ... 

// (Orphaned logic moved to drawPredictionPath)

// ...

// (Orphaned debris logic moved to drawDebrisOrbits)
// ...
// predictionLine hoisted
async function drawPredictionPath(l1, l2) {
    if (predictionLine) scene.remove(predictionLine);

    try {
        const now = new Date(simulatedTime.getTime()); // Sync with SIMULATION TIME
        const req = {
            line1: l1,
            line2: l2,
            start_time: now.toISOString(),
            minutes_duration: 95,
            step_minutes: 1,
            solar_flux: (val => isNaN(val) ? 150 : val)(parseFloat(fluxSlider ? fluxSlider.value : 150)),
            kp_index: (val => isNaN(val) ? 3 : val)(parseFloat(kpSlider ? kpSlider.value : 3))
        };

        const resp = await fetch('/predict_batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req)
        });
        const data = await resp.json();

        if (!Array.isArray(data)) {
            console.error("Prediction API Error:", data);
            return;
        }

        // Store for Ghost Interpolation
        predictionData = data;
        predictionStartTime = new Date(req.start_time);

        const points = [];
        const positions = [];
        data.forEach(pt => {
            const v = toVector3(pt); // Fix alignment
            points.push(v);
            positions.push(v.x, v.y, v.z);
        });

        const geometry = new LineGeometry();
        geometry.setPositions(positions); // Flat array [x,y,z, x,y,z...]

        const material = new LineMaterial({
            color: 0x00FFFF,
            linewidth: 5, // Pixels
            resolution: new THREE.Vector2(window.innerWidth, window.innerHeight), // Needed for screenspace width
            dashed: false,
            alphaToCoverage: true, // Anti-aliasing
        });

        predictionLine = new Line2(geometry, material);
        predictionLine.computeLineDistances();
        scene.add(predictionLine);

        // Label Logic: Prediction
        if (predictionLine.userData.label) scene.remove(predictionLine.userData.label);
        if (points.length > 0) {
            const endPt = points[points.length - 1];
            // ... (rest of label logic same, attach to a dummy because Line2 might not support add() same way? It inherits Object3D so it should)
            const label = createLabel("AI Predicted (90m)", 'prediction');
            const dummy = new THREE.Object3D();
            dummy.position.copy(endPt);
            scene.add(dummy); // Add to scene, not line, to be safe with Line2
            dummy.add(label);
            predictionLine.userData.label = dummy;
        }

        console.log("Prediction Path Drawn (Fat Line)");
    } catch (err) { console.error(err); }
}

// --- Debris Visualization (Dynamic) ---
const debrisObjects = []; // Store { line, mesh, satrec }
// debrisPredictionLine hoisted
let currentRisks = []; // Store latest risks for button access

async function drawDebrisOrbits() {
    // Clear old
    debrisObjects.forEach(obj => {
        scene.remove(obj.line);
        scene.remove(obj.mesh);
    });
    debrisObjects.length = 0;
    if (debrisPredictionLine) scene.remove(debrisPredictionLine);

    try {
        const resp = await fetch('/risks');
        currentRisks = await resp.json(); // Update global

        if (!Array.isArray(currentRisks)) {
            console.error("Risks API returned non-array:", currentRisks);
            currentRisks = []; // Safe fallback
            return;
        }

        for (const debris of currentRisks) {
            try {
                const tleResp = await fetch(`/tle/${debris.id}`);
                if (!tleResp.ok) continue;
                const tleData = await tleResp.json();
                const satrec = satellite.twoline2satrec(tleData.line1, tleData.line2);

                // Initial Line Draw
                const material = new THREE.LineBasicMaterial({ color: 0xFFFF00, transparent: true, opacity: 1.0 }); // Yellow
                const geometry = new THREE.BufferGeometry();
                const line = new THREE.Line(geometry, material);
                scene.add(line);

                // Initial Mesh
                const mesh = new THREE.Mesh(
                    new THREE.SphereGeometry(80, 8, 8),
                    new THREE.MeshBasicMaterial({ color: 0xFFFF00 }) // Yellow Match
                );
                mesh.userData = {
                    satrec: satrec,
                    id: debris.id,
                    tca: debris.tca,
                    line1: tleData.line1,
                    line2: tleData.line2,
                    name: debris.name
                };
                scene.add(mesh);

                // Label Logic: Debris
                const label = createLabel(`☄ ${debris.name}`, 'debris');
                label.position.set(0, 400, 0);
                mesh.add(label);

                debrisObjects.push({ line, mesh, satrec, geometry });

                // Force first update
                updateDebrisLineGeometry(line, geometry, satrec, new Date());

            } catch (e) { console.warn(e); }
        }

        // Add "Analyze Risk" button if not exists
        if (!document.getElementById('btn-analyze')) {
            const btnAnalyze = document.createElement('button');
            btnAnalyze.id = 'btn-analyze';
            btnAnalyze.innerHTML = '<i class="fas fa-search-location"></i> Analyze Collision Risks';
            btnAnalyze.className = 'action-btn';
            btnAnalyze.style.marginTop = '10px';
            btnAnalyze.style.background = 'linear-gradient(45deg, #ff4500, #ff8c00)';

            // Append to Mission Control
            document.querySelector('.controls').appendChild(btnAnalyze);

            btnAnalyze.addEventListener('click', async () => {
                const btn = document.getElementById('btn-analyze');
                btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Screening...';

                // Analyze the first high-risk object found
                if (currentRisks.length > 0) {
                    const target = currentRisks[0];
                    // Find object to get TLE
                    const targetObj = debrisObjects.find(d => d.mesh.userData.id === target.id);

                    if (targetObj) {
                        try {
                            // 1. Draw Debris Prediction (Green Line)
                            await drawDebrisPredictionPath(targetObj.mesh.userData.line1, targetObj.mesh.userData.line2);

                            // 2. Perform Analysis
                            const satId = document.getElementById('sat-selector').value;
                            const req = {
                                sat_id: satId,
                                debris_id: target.id,
                                tca: target.tca
                            };

                            const resp = await fetch('/analyze_risk', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify(req)
                            });
                            const result = await resp.json();

                            // const result = await resp.json(); (Already declared above)

                            if (result.physics_miss_distance_km !== undefined) {
                                alert(
                                    `RISK ANALYSIS REPORT\n\n` +
                                    `Target Debris: ${target.name}\n` +
                                    `TCA: ${result.tca}\n\n` +
                                    `🔴 Standard Physics Miss: ${result.physics_miss_distance_km.toFixed(3)} km (ALARM)\n` +
                                    `🟢 AI-Corrected Miss: ${result.ai_miss_distance_km.toFixed(3)} km\n\n` +
                                    `Analysis: ${result.recommendation}\n` +
                                    `Risk Reduction: ${result.risk_reduction_percent.toFixed(1)}%`
                                );
                            } else {
                                throw new Error(result.detail || "Invalid response from server");
                            }

                        } catch (e) {
                            alert("Analysis Failed: " + e.message);
                        }
                    } else {
                        // Error handling for missing TLE
                        alert(`Cannot analyze risk: TLE data missing for debris ${target.name} (ID: ${target.id}).\nThe backend failed to fetch its orbit data from Space-Track.`);
                    }
                } else {
                    alert("No active collision risks to analyze.");
                }
                btn.innerHTML = '<i class="fas fa-search-location"></i> Analyze Collision Risks';
            });
        }

        // Auto-Trigger Analysis removed per user request
        // setTimeout(() => {
        //     const btn = document.getElementById('btn-analyze');
        //     if (btn) btn.click();
        // }, 1000);

    } catch (e) { console.error(e); }
}

// Helper: Re-calculate line points based on current time
function updateDebrisLineGeometry(line, geometry, satrec, baseTime) {
    const points = [];
    // 3 orbits = 300 mins
    for (let i = 0; i < 300; i += 5) {
        const t = new Date(baseTime.getTime() + i * 60000);
        const pv = satellite.propagate(satrec, t);
        if (pv.position && !isNaN(pv.position.x)) {
            points.push(toVector3(pv.position));
        }
    }
    geometry.setFromPoints(points);
    geometry.attributes.position.needsUpdate = true;
}

// --- Debris Prediction Helper ---
async function drawDebrisPredictionPath(l1, l2) {
    if (debrisPredictionLine) scene.remove(debrisPredictionLine);

    try {
        const now = new Date(simulatedTime.getTime()); // Sync with SIMULATION TIME
        const req = {
            line1: l1,
            line2: l2,
            start_time: now.toISOString(),
            minutes_duration: 95,
            step_minutes: 1,
            solar_flux: 150, // Use scenario defaults
            kp_index: 3
        };

        const resp = await fetch('/predict_batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req)
        });
        const data = await resp.json();

        const points = [];
        data.forEach(pt => {
            points.push(toVector3(pt));
        });

        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        // Bright Green for Debris Prediction
        const material = new THREE.LineBasicMaterial({ color: 0x00FF00, linewidth: 3 });
        debrisPredictionLine = new THREE.Line(geometry, material);
        scene.add(debrisPredictionLine);
        console.log("Debris Prediction Path Drawn");

    } catch (err) { console.error("Debris Prediction Error", err); }
}

// --- TLE Data (Hardcoded for Demo speed) ---
// --- Real-Time Mode: No Local DB ---
// TLE_DB Removed. All data must come from API.

let currentSatrec = null;

// --- Logic: Selection ---
// (Moved to top)

if (weatherSelect) {
    weatherSelect.addEventListener('change', (e) => {
        const val = e.target.value;
        let newFlux = 150;
        let newKp = 3;

        if (val === 'QUIET') { newFlux = 70; newKp = 1; }
        if (val === 'STORM') { newFlux = 300; newKp = 9; }

        if (fluxSlider) {
            fluxSlider.value = newFlux;
            document.getElementById('flux-disp').innerText = newFlux;
        }
        if (kpSlider) {
            kpSlider.value = newKp;
            document.getElementById('kp-disp').innerText = newKp;
        }
        // Force update prediction on scenario change
        updateSatelliteEngine();
    });

    // Real-time slider listeners
    fluxSlider.addEventListener('input', (e) => {
        document.getElementById('flux-disp').innerText = e.target.value;
        const l1 = inputs.l1.value;
        const l2 = inputs.l2.value;
        drawPredictionPath(l1, l2); // Debounce could be good, but direct call is okay for now
    });

    kpSlider.addEventListener('input', (e) => {
        document.getElementById('kp-disp').innerText = e.target.value;
        const l1 = inputs.l1.value;
        const l2 = inputs.l2.value;
        drawPredictionPath(l1, l2);
    });
}


// Handle Change: Trigger Live Fetch
selector.addEventListener('change', (e) => {
    // When selection changes, just click the fetch button to get fresh data
    if (btnFetch) btnFetch.click();
});

function updateSatelliteEngine() {
    const l1 = inputs.l1.value;
    const l2 = inputs.l2.value;
    try {
        // Parse TLE using satellite.js
        currentSatrec = satellite.twoline2satrec(l1, l2);
        updateOrbitPath(currentSatrec);

        // Trigger New Visualizations
        drawPredictionPath(l1, l2);
        drawDebrisOrbits(); // Refresh debris context

        console.log("Satellite Initialized");
    } catch (err) {
        console.error("TLE Error", err);
    }
}

// Init
updateSatelliteEngine();


// --- Time Control Logic ---
// timeScale, isLive, simulatedTime hoisted to top
let lastFrameTime = Date.now();

const btnLive = document.getElementById('btn-live');
const btnSim = document.getElementById('btn-sim');

function setMode(mode) {
    if (mode === 'LIVE') {
        isLive = true;
        timeScale = 1.0;
        btnLive.classList.add('active');
        btnSim.classList.remove('active');
        // Reset to now
        simulatedTime = new Date();
    } else {
        isLive = false;
        timeScale = 1000.0; // 1000x Speed
        btnLive.classList.remove('active');
        btnSim.classList.add('active');
    }
}

btnLive.addEventListener('click', () => setMode('LIVE'));
btnSim.addEventListener('click', () => setMode('SIM'));


// --- Animation Loop ---
let frameCount = 0;
let isPaused = false; // For future pause button

function animate() {
    requestAnimationFrame(animate);
    frameCount++;
    controls.update();

    const currentFrameTime = Date.now();
    const deltaTime = (currentFrameTime - lastFrameTime); // ms
    lastFrameTime = currentFrameTime;

    // Time Propagation
    if (isLive) {
        simulatedTime = new Date(); // Lock to system clock
    } else {
        // Advance simulated time
        // deltaTime (ms) * timeScale
        simulatedTime = new Date(simulatedTime.getTime() + deltaTime * timeScale);
    }

    // 1. Earth Rotation
    earth.rotation.y += 0.00007;
    atmosphere.rotation.y += 0.00007;

    // 2. Satellite Physics
    if (currentSatrec) {
        const positionAndVelocity = satellite.propagate(currentSatrec, simulatedTime);

        const positionEci = positionAndVelocity.position;
        const velocityEci = positionAndVelocity.velocity;

        if (positionEci && !isNaN(positionEci.x)) {
            // Update Mesh
            satelliteMesh.position.copy(toVector3(positionEci));

            // Stats
            const gmst = satellite.gstime(simulatedTime);
            const positionGd = satellite.eciToGeodetic(positionEci, gmst);

            inputs.alt.innerText = positionGd.height.toFixed(1) + " km";
            const vel = Math.sqrt(velocityEci.x ** 2 + velocityEci.y ** 2 + velocityEci.z ** 2);
            inputs.vel.innerText = vel.toFixed(2) + " km/s";
        }
    }

    // 3. Debris Physics (Dynamic Alignment)
    // Move Markers (Every Frame)
    debrisObjects.forEach(obj => {
        const pv = satellite.propagate(obj.satrec, simulatedTime);
        if (pv.position && !isNaN(pv.position.x)) {
            obj.mesh.position.copy(toVector3(pv.position));
        }
    });

    // --- Ghost Update (AI Path) ---
    if (ghostMesh && predictionData.length > 0 && predictionStartTime) {
        // Calculate minutes elapsed since prediction start
        const elapsed = (simulatedTime - predictionStartTime) / 60000;
        if (elapsed >= 0 && elapsed < predictionData.length - 1) {
            const idx = Math.floor(elapsed);
            const alpha = elapsed - idx;

            const p1 = predictionData[idx];
            const p2 = predictionData[idx + 1];

            // Linear Interpolation
            const x = p1.x + (p2.x - p1.x) * alpha;
            const y = p1.y + (p2.y - p1.y) * alpha;
            const z = p1.z + (p2.z - p1.z) * alpha;

            ghostMesh.position.copy(toVector3({ x, y, z }));
            ghostMesh.visible = true;
        } else {
            ghostMesh.visible = false; // Hide if out of range
        }
    }


    // Re-calculate Lines (Every ~5 seconds)
    if (frameCount % 300 === 0 && !isLive) { // Only needed in Sim Mode where drift is fast
        debrisObjects.forEach(obj => {
            updateDebrisLineGeometry(obj.line, obj.geometry, obj.satrec, simulatedTime);
        });
    }

    renderer.render(scene, camera);
    if (window.labelRenderer) window.labelRenderer.render(scene, camera);
}

// Handle Resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    labelRenderer.setSize(window.innerWidth, window.innerHeight);

    // Update Line2 Resolutions
    if (predictionLine && predictionLine.material && predictionLine.material.resolution) {
        predictionLine.material.resolution.set(window.innerWidth, window.innerHeight);
    }
});

// Start
animate();

// AUTO-START
window.addEventListener('load', () => {
    if (btnFetch) {
        console.log("Auto-Fetching initial TLE...");
        btnFetch.click();
    }
});

// --- INTERACTIVITY: Raycaster & Info Box ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

// Create Info Box Overlay (if not exists)
let infoBox = document.getElementById('info-overlay');
if (!infoBox) {
    infoBox = document.createElement('div');
    infoBox.id = 'info-overlay';
    infoBox.style.cssText = `
        position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
        background: rgba(0, 20, 40, 0.9); border: 2px solid #00d2ff;
        color: white; padding: 15px; border-radius: 10px;
        font-family: 'Orbitron', monospace; display: none; z-index: 1000;
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.4); text-align: center;
        min-width: 300px;
    `;
    document.body.appendChild(infoBox);
}

// Click Listener
window.addEventListener('pointerdown', (event) => {
    // Only raycast if left click (to avoid confusing with drag/rotate)
    if (event.button !== 0) return;

    // Normalize Mouse
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);

    // Collect Intersectable Objects
    const targets = [earth];
    if (satelliteMesh) targets.push(satelliteMesh);
    debrisObjects.forEach(d => targets.push(d.mesh));

    // Raycast explicitly against visible objects
    const intersects = raycaster.intersectObjects(targets);

    if (intersects.length > 0) {
        // ... (Interaction logic) ...
        const hit = intersects[0].object;
        let content = "";

        if (hit === earth) {
            content = "<h3>🌍 PLANET EARTH</h3><p>Status: Habitable</p><p>Radius: 6,371 km</p>";
        } else if (hit === satelliteMesh) {
            content = "<h3>🛰️ FOCUS: ISS (ZARYA)</h3><p>Type: Space Station</p><p>Orbit: Low Earth Orbit</p><p>Crew: Active</p>";
        } else if (hit.userData.id) {
            // Debris
            const d = hit.userData;
            content = `<h3>☄️ DEBRIS OBJECT: ${d.id}</h3><p>TCA: ${d.tca}</p><p>Risk Level: HIGH</p>`;
        }

        infoBox.innerHTML = content;
        infoBox.style.display = 'block';

        // Auto-hide after 5 seconds
        setTimeout(() => { infoBox.style.display = 'none'; }, 5000);
    } else {
        // Clicked empty space -> Close info
        infoBox.style.display = 'none';
    }
});
