/**
 * DeepDebris 3.0: Maneuver Planning Module
 * 
 * Handles autonomous collision avoidance maneuver planning using RL agent.
 */

// Global state
let currentFuelPercent = 100.0;
let maneuverLine = null;
let thrustArrow = null;
let currentManeuverPlan = null;

/**
 * Generate optimal maneuver plan using RL agent
 */
async function generateManeuverPlan() {
    const statusMsg = document.getElementById('status-msg');
    const maneuverPanel = document.getElementById('maneuver-panel');

    try {
        // Get current satellite TLE
        const l1 = document.getElementById('tle1').value.trim();
        const l2 = document.getElementById('tle2').value.trim();

        if (!l1 || !l2) {
            statusMsg.textContent = 'Error: TLE data missing';
            statusMsg.style.color = '#ff0000';
            return;
        }

        // Get current risk (debris TLE)
        if (!currentRisks || currentRisks.length === 0) {
            statusMsg.textContent = 'No collision risks detected. Run risk analysis first.';
            statusMsg.style.color = '#ffaa00';
            return;
        }

        const topRisk = currentRisks[0];

        // Fetch debris TLE
        statusMsg.textContent = 'Fetching debris TLE...';
        const debrisResp = await fetch(`/tle/${topRisk.id}`);
        if (!debrisResp.ok) {
            throw new Error('Failed to fetch debris TLE');
        }
        const debrisTLE = await debrisResp.json();

        // Call maneuver planning API
        statusMsg.textContent = 'AI Agent calculating optimal maneuver...';
        const resp = await fetch('/plan_maneuver', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sat_tle: { line1: l1, line2: l2 },
                debris_tle: { line1: debrisTLE.line1, line2: debrisTLE.line2 },
                tca: topRisk.tca
            })
        });

        if (!resp.ok) {
            const error = await resp.json();
            throw new Error(error.detail || 'Maneuver planning failed');
        }

        const plan = await resp.json();
        currentManeuverPlan = plan;

        // Update UI
        document.getElementById('thrust-dir').textContent = plan.thrust_direction;
        document.getElementById('burn-time').textContent = plan.burn_duration_seconds.toFixed(1);
        document.getElementById('exec-time').textContent = new Date(plan.execution_time_utc).toLocaleString();
        document.getElementById('fuel-cost').textContent = plan.fuel_cost_percent.toFixed(3);
        document.getElementById('new-miss').textContent = plan.new_miss_distance_km.toFixed(2);

        // Show panel
        maneuverPanel.style.display = 'block';

        // Draw maneuver trajectory
        drawManeuverTrajectory(plan.new_trajectory);

        // Update fuel gauge
        updateFuelGauge(currentFuelPercent - plan.fuel_cost_percent);

        statusMsg.textContent = '✓ Maneuver plan generated';
        statusMsg.style.color = '#00ff00';

    } catch (error) {
        console.error('Maneuver planning error:', error);
        statusMsg.textContent = `Error: ${error.message}`;
        statusMsg.style.color = '#ff0000';
        maneuverPanel.style.display = 'none';
    }
}

/**
 * Draw maneuver trajectory on 3D globe
 */
function drawManeuverTrajectory(positions) {
    // Remove old maneuver line if exists
    if (maneuverLine) {
        scene.remove(maneuverLine);
        maneuverLine.geometry.dispose();
        maneuverLine.material.dispose();
    }

    if (thrustArrow) {
        scene.remove(thrustArrow);
    }

    // Create white dashed line for new trajectory
    const points = positions.map(p => toVector3([p.x, p.y, p.z]));
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineDashedMaterial({
        color: 0xFFFFFF,
        dashSize: 500,
        gapSize: 200,
        linewidth: 3
    });

    maneuverLine = new THREE.Line(geometry, material);
    maneuverLine.computeLineDistances();
    scene.add(maneuverLine);

    // Add thrust vector arrow at satellite position
    if (satelliteMesh && currentManeuverPlan) {
        // Map action to direction vector
        const actionVectors = {
            1: [1, 0, 0],   // Prograde
            2: [-1, 0, 0],  // Retrograde
            3: [0, 1, 0],   // Normal
            4: [0, -1, 0],  // Anti-Normal
            5: [0, 0, 1],   // Radial
            6: [0, 0, -1]   // Anti-Radial
        };

        const dirVec = actionVectors[currentManeuverPlan.thrust_action] || [1, 0, 0];
        const thrustDir = new THREE.Vector3(...dirVec).normalize();

        thrustArrow = new THREE.ArrowHelper(
            thrustDir,
            satelliteMesh.position,
            2000,  // Length
            0xFFFF00,  // Yellow
            500,   // Head length
            300    // Head width
        );
        scene.add(thrustArrow);
    }

    console.log(`Maneuver trajectory drawn: ${positions.length} points`);
}

/**
 * Update fuel gauge display
 */
function updateFuelGauge(fuelPercent) {
    currentFuelPercent = Math.max(0, Math.min(100, fuelPercent));

    const fuelBar = document.getElementById('fuel-level');
    const fuelText = document.getElementById('fuel-percent');

    fuelBar.style.width = currentFuelPercent + '%';
    fuelText.textContent = currentFuelPercent.toFixed(1);

    // Color coding
    if (currentFuelPercent < 20) {
        fuelBar.style.background = 'linear-gradient(90deg, #ff0000, #cc0000)';
    } else if (currentFuelPercent < 50) {
        fuelBar.style.background = 'linear-gradient(90deg, #ffaa00, #ff8800)';
    } else {
        fuelBar.style.background = 'linear-gradient(90deg, #00ff00, #00cc00)';
    }
}

/**
 * Execute maneuver (simulation mode)
 */
function executeManeuver() {
    if (!currentManeuverPlan) {
        alert('No maneuver plan available');
        return;
    }

    const statusMsg = document.getElementById('status-msg');
    statusMsg.textContent = '🚀 Executing maneuver (simulation)...';
    statusMsg.style.color = '#00ffff';

    // Deduct fuel
    updateFuelGauge(currentFuelPercent - currentManeuverPlan.fuel_cost_percent);

    // Animate satellite along maneuver trajectory
    // (This would require more complex animation logic)

    setTimeout(() => {
        statusMsg.textContent = '✓ Maneuver executed successfully';
        statusMsg.style.color = '#00ff00';
        alert(`Maneuver executed!\n\nNew Miss Distance: ${currentManeuverPlan.new_miss_distance_km.toFixed(2)} km\nFuel Used: ${currentManeuverPlan.fuel_cost_percent.toFixed(3)}%`);
    }, 2000);
}

// Export functions for use in main.js
if (typeof window !== 'undefined') {
    window.generateManeuverPlan = generateManeuverPlan;
    window.executeManeuver = executeManeuver;
    window.updateFuelGauge = updateFuelGauge;
}
