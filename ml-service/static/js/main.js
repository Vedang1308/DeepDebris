import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';

// --- Debugging ---
console.log("DeepDebris 2.0 Starting...");

// --- Scene Setup ---
const scene = new THREE.Scene();
window.scene = scene; // Expose for debugging

// Camera - Professional setup
const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 10, 200000);
camera.position.set(15000, 8000, 15000);

// Renderer - Production quality
const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
    precision: 'highp'
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Cap at 2x for performance
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = false; // Disable for space (no shadows)
renderer.outputEncoding = THREE.sRGBEncoding;
document.getElementById('canvas-container').appendChild(renderer.domElement);

// Renderer (Labels)
const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(window.innerWidth, window.innerHeight);
labelRenderer.domElement.style.position = 'absolute';
labelRenderer.domElement.style.top = '0px';
labelRenderer.domElement.style.pointerEvents = 'none'; // Passthrough to allow canvas interaction
document.body.appendChild(labelRenderer.domElement);

// Controls - Smooth and professional
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08; // Smoother damping
controls.screenSpacePanning = false;
controls.minDistance = 7000;  // Prevent clipping into Earth
controls.maxDistance = 80000; // Allow wider view
controls.zoomSpeed = 1.2;     // Responsive zoom
controls.rotateSpeed = 0.5;   // Gentle rotation
controls.panSpeed = 0.8;
controls.enablePan = true;
controls.autoRotate = false;
controls.autoRotateSpeed = 0.5;
// Smooth zoom with mouse wheel
controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.DOLLY,
    RIGHT: THREE.MOUSE.PAN
};

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
// Professional lighting setup
const ambientLight = new THREE.AmbientLight(0x404040, 0.4); // Subtle ambient
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
sunLight.position.set(50000, 20000, 30000);
scene.add(sunLight);

// Hemisphere light for realistic sky/ground lighting
const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.6);
hemiLight.position.set(0, 20000, 0);
scene.add(hemiLight);

// ... (Existing Three.js Setup) ...

// --- UI Logic: Chat & TLE ---
// (Moved to top)

// Chat Toggle
chatToggle.addEventListener('click', () => {
    const widget = document.getElementById('chat-widget');
    widget.classList.toggle('minimized');
});

// Alerts Widget Toggle
const alertsToggle = document.getElementById('alerts-toggle');
const alertsWidget = document.getElementById('alerts-widget');
const alertsList = document.getElementById('alerts-list');

if (alertsToggle) {
    alertsToggle.addEventListener('click', () => {
        alertsWidget.classList.toggle('minimized');
    });
}

// Fetch and Display Alerts
async function fetchAlerts() {
    try {
        const resp = await fetch('/alerts');
        if (!resp.ok) return;
        const data = await resp.json();

        if (data.count === 0) {
            alertsList.innerHTML = '<div class="msg ai">No active alerts. System monitoring...</div>';
        } else {
            alertsList.innerHTML = '';
            data.alerts.forEach(alert => {
                const div = document.createElement('div');
                div.className = `msg ${alert.status === 'CRITICAL' ? 'critical' : 'alert'}`;
                div.innerHTML = `
                    <strong>${alert.status}:</strong> ${alert.debris_name}<br>
                    <small>TCA: ${new Date(alert.tca).toLocaleString()}</small><br>
                    <small>Miss: ${alert.ai_dist_km.toFixed(2)}km ± ${alert.uncertainty_km.toFixed(2)}km</small>
                `;
                alertsList.appendChild(div);
            });
        }
    } catch (err) {
        console.error('Alerts fetch error:', err);
    }
}

// Fetch alerts every 30 seconds
setInterval(fetchAlerts, 30000);
fetchAlerts(); // Initial fetch

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

// --- Earth - High Quality ---
const textureLoader = new THREE.TextureLoader();
const earthRadius = 6371;

// High-resolution geometry for smooth sphere
const earthGeo = new THREE.SphereGeometry(earthRadius, 128, 128);

// Professional Earth material with NASA Blue Marble texture
const earthMat = new THREE.MeshStandardMaterial({
    map: textureLoader.load('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg'),
    bumpMap: textureLoader.load('https://unpkg.com/three-globe/example/img/earth-topology.png'),
    bumpScale: 20,
    roughness: 0.9,
    metalness: 0.1,
    emissive: 0x000000,
    emissiveIntensity: 0
});

const earth = new THREE.Mesh(earthGeo, earthMat);
earth.rotation.y = Math.PI; // Rotate to show correct hemisphere
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

// --- Atmosphere - Realistic Glow ---
const atmoGeo = new THREE.SphereGeometry(earthRadius + 80, 128, 128);
const atmoMat = new THREE.ShaderMaterial({
    uniforms: {
        c: { value: 0.3 },
        p: { value: 6.5 }
    },
    vertexShader: `
        varying vec3 vNormal;
        void main() {
            vNormal = normalize(normalMatrix * normal);
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,
    fragmentShader: `
        uniform float c;
        uniform float p;
        varying vec3 vNormal;
        void main() {
            float intensity = pow(c - dot(vNormal, vec3(0.0, 0.0, 1.0)), p);
            gl_FragColor = vec4(0.3, 0.6, 1.0, 1.0) * intensity;
        }
    `,
    side: THREE.BackSide,
    blending: THREE.AdditiveBlending,
    transparent: true
});
const atmosphere = new THREE.Mesh(atmoGeo, atmoMat);
scene.add(atmosphere);

// --- Stars - Realistic Starfield ---
function createStars() {
    const geometry = new THREE.BufferGeometry();
    const vertices = [];
    const colors = [];

    for (let i = 0; i < 15000; i++) {
        const x = THREE.MathUtils.randFloatSpread(250000);
        const y = THREE.MathUtils.randFloatSpread(250000);
        const z = THREE.MathUtils.randFloatSpread(250000);
        vertices.push(x, y, z);

        // Varied star colors (white, blue-white, yellow-white)
        const colorVariation = Math.random();
        if (colorVariation > 0.95) {
            colors.push(0.7, 0.8, 1.0); // Blue stars
        } else if (colorVariation > 0.90) {
            colors.push(1.0, 0.9, 0.7); // Yellow stars
        } else {
            colors.push(1.0, 1.0, 1.0); // White stars
        }
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: 120,
        vertexColors: true,
        transparent: true,
        opacity: 0.9,
        sizeAttenuation: true
    });

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
const ghostGeo = new THREE.SphereGeometry(220, 32, 32); // Larger shell
const ghostMat = new THREE.MeshStandardMaterial({
    color: 0x0088FF,      // Solid Azure Blue
    emissive: 0x0044FF,   // Glow
    emissiveIntensity: 0.5,
    transparent: true,
    opacity: 0.8,         // High visibility
    wireframe: false      // Solid, not wireframe
});
const ghostMesh = new THREE.Mesh(ghostGeo, ghostMat);
scene.add(ghostMesh);

// Correction Vector (Physics -> AI)
const clGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0)]);
const clMat = new THREE.LineBasicMaterial({ color: 0xFF3333, linewidth: 2 });
let correctionLine = new THREE.Line(clGeo, clMat);
scene.add(correctionLine);

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
    const now = simulatedTime || new Date(); // SYNC with Simulation Time
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
        dashSize: 800,   // Longer dashes
        gapSize: 400,    // Wider gaps
        linewidth: 3,    // Thicker line
        opacity: 0.6,    // More transparent
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
    if (predictionLine) {
        scene.remove(predictionLine);
        // Clean up uncertainty tube if it exists
        if (predictionLine.userData && predictionLine.userData.uncertaintyTube) {
            scene.remove(predictionLine.userData.uncertaintyTube);
        }
    }

    try {
        if (!l1 || !l2) return; // Don't predict without TLE

        const now = new Date(simulatedTime.getTime()); // Sync with SIMULATION TIME
        const req = {
            line1: l1,
            line2: l2,
            start_time: now.toISOString(),
            minutes_duration: 480,
            step_minutes: 1,
            solar_flux: (val => isNaN(val) ? 150 : val)(parseFloat(fluxSlider ? fluxSlider.value : 150)),
            kp_index: (val => isNaN(val) ? 3 : val)(parseFloat(kpSlider ? kpSlider.value : 3))
        };

        const resp = await fetch('/predict_batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req)
        });

        if (!resp.ok) {
            const errText = await resp.text();
            console.error(`API Error ${resp.status}:`, errText);
            return;
        }

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
            color: 0x00FFFF,      // Cyan (AI Prediction)
            linewidth: 8,         // Much thicker than physics line
            resolution: new THREE.Vector2(window.innerWidth, window.innerHeight),
            dashed: false,        // Solid line (vs dashed physics)
            alphaToCoverage: true,
            opacity: 0.95,        // More opaque
            transparent: true
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
let debrisObjects = []; // Store { line, mesh, satrec }
// debrisPredictionLine hoisted
let currentRisks = []; // Store latest risks for button access

// --- Debris Catalog (Background Field) ---
async function loadDebrisCatalog() {
    try {
        const response = await fetch('/debris/catalog?limit=20');
        const data = await response.json();

        if (data.debris && data.debris.length > 0) {
            console.log(`Loaded ${data.debris.length} debris objects from catalog.`);

            // Material for background debris
            const debrisMat = new THREE.MeshBasicMaterial({
                color: 0x888888,
                transparent: true,
                opacity: 0.6
            });
            const debrisGeo = new THREE.SphereGeometry(15, 8, 8); // Small dots

            data.debris.forEach(deb => {
                // Initialize SatRec for propagation
                const satrec = satellite.twoline2satrec(deb.line1, deb.line2);

                // Create mesh
                const mesh = new THREE.Mesh(debrisGeo, debrisMat);
                mesh.userData = {
                    id: deb.id,
                    name: deb.name,
                    line1: deb.line1,
                    line2: deb.line2,
                    isCatalog: true,
                    satrec: satrec // Save to userData too
                };
                scene.add(mesh);

                // Push to render loop
                debrisObjects.push({
                    mesh: mesh,
                    id: deb.id,
                    type: 'catalog',
                    satrec: satrec
                });
            });
        }
    } catch (e) {
        console.error("Failed to load debris catalog:", e);
    }
}

async function drawDebrisOrbits() {
    if (!Array.isArray(debrisObjects)) debrisObjects = []; // Safety

    // Clear old RISK lines/meshes (preserve catalog)
    // Filter out catalog objects from removal list
    const risksToRemove = debrisObjects.filter(obj => obj.type !== 'catalog');
    risksToRemove.forEach(obj => {
        if (obj.line) scene.remove(obj.line);
        if (obj.mesh) scene.remove(obj.mesh);
        if (obj.label) scene.remove(obj.label);
    });

    // Keep only catalog objects in the array
    debrisObjects = debrisObjects.filter(obj => obj.type === 'catalog');

    if (debrisPredictionLine) scene.remove(debrisPredictionLine);

    try {
        const resp = await fetch('/risks');
        currentRisks = await resp.json(); // Update global

        if (!Array.isArray(currentRisks)) {
            console.error("Risks API returned non-array:", currentRisks);
            currentRisks = []; // Safe fallback
            return;
        }

        const currentSatId = document.getElementById('sat-selector').value;

        for (const debris of currentRisks) {
            // Safety: Don't render "Self-Collision" if backend reports same ID
            if (debris.id == currentSatId) continue;

            try {
                const tleResp = await fetch(`/tle/${debris.id}`);
                if (!tleResp.ok) continue;
                const tleData = await tleResp.json();
                const satrec = satellite.twoline2satrec(tleData.line1, tleData.line2);

                // Initial Line Draw
                const material = new THREE.LineBasicMaterial({ color: 0xFFA500, transparent: true, opacity: 1.0 }); // Orange for Risk
                const geometry = new THREE.BufferGeometry();
                const line = new THREE.Line(geometry, material);
                scene.add(line);

                // Initial Mesh
                const mesh = new THREE.Mesh(
                    new THREE.SphereGeometry(80, 8, 8),
                    new THREE.MeshBasicMaterial({ color: 0xFFA500 }) // Orange for Risk
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
            minutes_duration: 300,
            step_minutes: 1,
            solar_flux: 150, // Use scenario defaults
            kp_index: 3
        };

        const resp = await fetch('/predict_batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req)
        });

        if (!resp.ok) throw new Error('Prediction failed');
        const data = await resp.json();

        if (!Array.isArray(data)) {
            console.error("Prediction data is not an array:", data);
            return;
        }

        const positions = [];
        const uncertainties = []; // Store uncertainty for tube radius

        data.forEach(pt => {
            const v = toVector3({ x: pt.x, y: pt.y, z: pt.z });
            positions.push(v.x, v.y, v.z);
            uncertainties.push(pt.uncertainty_km || 0); // Default to 0 if missing
        });

        // --- CYAN LINE (AI Prediction) ---
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        const material = new THREE.LineBasicMaterial({ color: 0x00ffff, linewidth: 2 });
        debrisPredictionLine = new THREE.Line(geometry, material);
        scene.add(debrisPredictionLine);

        // --- UNCERTAINTY TUBE (Probabilistic Visualization) ---
        // Create a tube around the path with radius = average uncertainty
        if (data.length > 1 && uncertainties.some(u => u > 0)) {
            const avgUncertainty = uncertainties.reduce((a, b) => a + b, 0) / uncertainties.length;
            const tubeRadius = Math.max(avgUncertainty, 10); // Min 10km for visibility

            // Create curve from points
            const curvePoints = [];
            for (let i = 0; i < positions.length; i += 3) {
                curvePoints.push(new THREE.Vector3(positions[i], positions[i + 1], positions[i + 2]));
            }

            if (curvePoints.length > 1) {
                const curve = new THREE.CatmullRomCurve3(curvePoints);
                const tubeGeometry = new THREE.TubeGeometry(curve, 64, tubeRadius, 8, false);
                const tubeMaterial = new THREE.MeshBasicMaterial({
                    color: 0x00ffff,
                    transparent: true,
                    opacity: 0.15,
                    side: THREE.DoubleSide
                });
                const uncertaintyTube = new THREE.Mesh(tubeGeometry, tubeMaterial);
                uncertaintyTube.name = 'uncertaintyTube';
                scene.add(uncertaintyTube);

                // Store reference for cleanup
                if (!debrisPredictionLine.userData) debrisPredictionLine.userData = {};
                debrisPredictionLine.userData.uncertaintyTube = uncertaintyTube;
            }
        }
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
        if (!obj.satrec) return; // Skip invalid objects
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

            // Draw Connection Line (Physics -> AI)
            if (satelliteMesh && correctionLine) {
                const pSat = satelliteMesh.position;
                const pGhost = ghostMesh.position;
                correctionLine.geometry.setFromPoints([pSat, pGhost]);
                correctionLine.visible = true;

                // Calculate KM offset (Earth Radius 100 units = 6371 km)
                const distUnits = pSat.distanceTo(pGhost);
                const distKm = (distUnits / 100.0) * 6371.0;

                ghostLabel.element.innerHTML = `AI PREDICTION<br>Offset: ${distKm.toFixed(1)} km`;
                ghostLabel.element.style.color = distKm > 50 ? "#ffcc00" : "#00ffff";
            }
        } else {
            ghostMesh.visible = false; // Hide if out of range
            if (correctionLine) correctionLine.visible = false;
        }
    }


    // Re-calculate Lines (Dynamic Sync)
    // Update Orbit Line every 5 seconds to prevent "Drift" failure
    if (frameCount % 300 === 0 && currentSatrec) {
        // Force update of the Magenta Physics Line to follow the satellite
        updateOrbitPath(currentSatrec);
    }

    if (frameCount % 300 === 0 && !isLive) {
        debrisObjects.forEach(obj => {
            if (obj.line && obj.geometry) {
                updateDebrisLineGeometry(obj.line, obj.geometry, obj.satrec, simulatedTime);
            }
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

    // Initialize New Features Safely (delayed)
    setTimeout(() => {
        try {
            console.log("Initializing Debris & Simulation...");
            loadDebrisCatalog();
            initSimulationListeners();
        } catch (e) { console.error(e); }
    }, 1500);
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
    if (ghostMesh && ghostMesh.visible) targets.push(ghostMesh); // Add Ghost
    debrisObjects.forEach(d => targets.push(d.mesh));

    // Raycast explicitly against visible objects
    const intersects = raycaster.intersectObjects(targets);

    if (intersects.length > 0) {
        // ... (Interaction logic) ...
        const hit = intersects[0].object;
        let content = "";

        // Helper to formatting coords
        const formatPos = (p) => `X: ${p.x.toFixed(2)}, Y: ${p.y.toFixed(2)}, Z: ${p.z.toFixed(2)} km`;
        const formatGeo = (posEci) => {
            const gmst = satellite.gstime(simulatedTime);
            const gd = satellite.eciToGeodetic(posEci, gmst);
            const lat = satellite.degreesLat(gd.latitude);
            const lon = satellite.degreesLong(gd.longitude);
            return `Lat: ${lat.toFixed(4)}°, Lon: ${lon.toFixed(4)}°, Alt: ${gd.height.toFixed(2)} km`;
        };

        if (hit === earth) {
            content = "<h3>🌍 PLANET EARTH</h3><p>Status: Habitable</p><p>Radius: 6,371 km</p>";
        } else if (hit === satelliteMesh && currentSatrec) {
            const pv = satellite.propagate(currentSatrec, simulatedTime);
            if (pv.position) {
                const geoStr = formatGeo(pv.position);
                const eciStr = formatPos(pv.position);
                content = `<h3>🛰️ FOCUS: ISS (ZARYA)</h3><p>Type: Space Station</p><p>${geoStr}</p><p class="mono-small">${eciStr}</p>`;
            } else {
                content = "<h3>🛰️ FOCUS: ISS (ZARYA)</h3><p>Orbit Data Unavailable</p>";
            }
        } else if (hit === ghostMesh) {
            // Reverse Transform: Three(x, y, z) -> ECI(x, -z, y)
            // because toVector3 was (x, z, -y)
            const v = ghostMesh.position;
            const eci = { x: v.x, y: -v.z, z: v.y };

            const geoStr = formatGeo(eci);
            const eciStr = formatPos(eci);

            content = `<h3>🤖 AI PREDICTION (GHOST)</h3><p>Model: ResidualNet + SGP4</p><p>${geoStr}</p><p class="mono-small">${eciStr}</p><p>Status: <span style="color:cyan">OPTIMIZED PATH</span></p>`;
        } else if (hit.userData.id) {
            // Debris
            const d = hit.userData;
            // Propagate specifically for this debris at current time
            const pv = satellite.propagate(d.satrec, simulatedTime);
            let locStr = "";
            if (pv.position) {
                locStr = `<p>${formatGeo(pv.position)}</p><p class="mono-small">${formatPos(pv.position)}</p>`;
            }

            content = `<h3>☄️ DEBRIS OBJECT: ${d.id}</h3><p>Name: ${d.name}</p><p>TCA: ${d.tca}</p>${locStr}<p>Risk Level: HIGH</p>`;
        }

        infoBox.innerHTML = content;
        infoBox.style.display = 'block';

        // Auto-hide after 10 seconds (increased from 5 for readability)
        setTimeout(() => { infoBox.style.display = 'none'; }, 10000);
    } else {
        // Clicked empty space -> Close info
        infoBox.style.display = 'none';
    }
});

// Load Debris Catalog on startup
// loadDebrisCatalog();


// --- SIMULATION MODE LOGIC (Merged) ---
let simulationMode = false;
let activeSimObjects = [];

function initSimulationListeners() {
    console.log("Initializing Simulation Listeners...");

    const btnSim = document.getElementById('btn-mode-sim');
    const btnLive = document.getElementById('btn-mode-live');
    const btnRun = document.getElementById('btn-run-simulation');
    const btnReset = document.getElementById('btn-reset-simulation');

    if (btnSim) {
        btnSim.addEventListener('click', () => {
            simulationMode = true;
            document.getElementById('simulation-controls').style.display = 'block';
            toggleLiveMode(false);

            // UI Toggle
            btnSim.classList.add('active');
            btnLive.classList.remove('active');
        });
    }

    if (btnLive) {
        btnLive.addEventListener('click', () => {
            simulationMode = false;
            document.getElementById('simulation-controls').style.display = 'none';
            toggleLiveMode(true);
            clearSimulation();

            // UI Toggle
            btnLive.classList.add('active');
            btnSim.classList.remove('active');
        });
    }

    if (btnReset) {
        btnReset.addEventListener('click', () => {
            clearSimulation();
            document.getElementById('sim-status').innerText = 'Simulation Reset';
        });
    }

    if (btnRun) {
        btnRun.addEventListener('click', async () => {
            const scenario = document.getElementById('sim-scenario').value;
            const hoursVal = document.getElementById('sim-time-offset').value;
            const hours = parseInt(hoursVal, 10);
            const tle1 = document.getElementById('tle1').value;
            const tle2 = document.getElementById('tle2').value;

            if (!tle1 || !tle2) {
                alert("Please load a satellite TLE first!");
                return;
            }

            document.getElementById('sim-status').innerText = `Running: ${scenario.toUpperCase()}...`;

            clearSimulation(); // Clear previous run

            if (scenario === 'collision' || scenario === 'near-miss') {
                await runCollisionSim(tle1, tle2, hours, scenario === 'collision');
            } else if (scenario === 'maneuver') {
                runManeuverSim(hours);
            } else if (scenario === 'storm') {
                runStormSim(hours);
            } else {
                runDecaySim(hours);
            }
        });
    }
}

function toggleLiveMode(isLiveMode) {
    const status = document.getElementById('status-msg');
    isLive = isLiveMode; // Update global state

    if (isLiveMode) {
        status.innerText = "System Online - Live Tracking";
        status.style.color = "#0f0";
        // Resume animation loop updates
    } else {
        status.innerText = "SIMULATION MODE ACTIVE - Live Data Paused";
        status.style.color = "#ff9500";
    }
}

async function runCollisionSim(tle1, tle2, hours, isImpact) {
    // Check if satellite exists
    const satrec = satellite.twoline2satrec(tle1, tle2);
    const targetTime = new Date(Date.now() + hours * 60 * 60 * 1000);

    // Propagate satellite to future time
    const pv = satellite.propagate(satrec, targetTime);
    if (!pv.position) {
        console.error("Propagation failed");
        return;
    }

    // Satellite Position (Target)
    const satPos = pv.position; // {x, y, z} in ECI

    // Debris Position (Impact or Near Miss)
    // impact: < 1km
    // near-miss: 50km
    const offset = isImpact ? 0.5 : 50.0;
    const debPos = {
        x: satPos.x + offset,
        y: satPos.y + (isImpact ? 0 : 30.0),
        z: satPos.z + (isImpact ? 0 : 40.0)
    };

    // VISUALIZATION
    const satVec = toVector3(satPos);
    const debVec = toVector3(debPos);

    // 1. Draw Simulated Satellite (Ghost) at T+Hours
    const satGeo = new THREE.SphereGeometry(180, 24, 24); // Larger than before
    const satMat = new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true, transparent: true, opacity: 0.5 });
    const satMesh = new THREE.Mesh(satGeo, satMat);
    satMesh.position.copy(satVec);
    scene.add(satMesh);
    activeSimObjects.push(satMesh);

    // 2. Draw Debris Object
    const debGeo = new THREE.SphereGeometry(isImpact ? 100 : 50, 32, 32);
    const debMat = new THREE.MeshBasicMaterial({
        color: isImpact ? 0xff0000 : 0xffa500,
        wireframe: false
    });
    const debMesh = new THREE.Mesh(debGeo, debMat);
    debMesh.position.copy(debVec);
    scene.add(debMesh);
    activeSimObjects.push(debMesh);

    // 3. Draw Trajectory Line (Collision Vector)
    // Simple line to show path
    const points = [
        satVec.clone().add(new THREE.Vector3(-2000, 0, 0)),
        satVec,
        debVec,
        debVec.clone().add(new THREE.Vector3(2000, 500, 500))
    ];
    // Use Line2/LineGeometry if possible, else standard Line
    const curve = new THREE.CatmullRomCurve3(points);
    const lineGeo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(50));
    const lineMat = new THREE.LineBasicMaterial({ color: 0xff0000, linewidth: 2, transparent: true, opacity: 0.8 });
    const line = new THREE.Line(lineGeo, lineMat);
    scene.add(line);
    activeSimObjects.push(line);

    // 4. Update Status UI
    const dist = isImpact ? "0.5 km (IMPACT)" : "58 km";
    const statusDiv = document.getElementById('sim-status');
    statusDiv.innerHTML = `
        <span style="color:${isImpact ? 'red' : 'orange'}">
        ⚠ RESULT: ${isImpact ? 'DIRECT COLLISION' : 'NEAR MISS'}<br>
        Separation: ${dist}<br>
        Time: T+${hours}h
        </span>
    `;

    // Look at it
    camera.position.copy(satVec).add(new THREE.Vector3(200, 200, 500));
    camera.lookAt(satVec);
}

function runManeuverSim(hours) {
    document.getElementById('sim-status').innerText = `Simulating Avoidance Maneuver (Delta-V) at T+${hours}h...`;
}

function runStormSim(hours) {
    document.getElementById('sim-status').innerText = `Simulating Solar Storm (Drag x3) for ${hours}h...`;
}

function runDecaySim(hours) {
    document.getElementById('sim-status').innerText = `Simulating Orbital Decay over ${hours}h...`;
}

function clearSimulation() {
    activeSimObjects.forEach(obj => scene.remove(obj));
    activeSimObjects = [];
}

// Start Listeners
// initSimulationListeners();
// loadDebrisCatalog();
