"""
Space Gym: Custom Gymnasium Environment for Satellite Collision Avoidance

This environment trains a Deep RL agent to plan optimal collision avoidance maneuvers
by balancing safety (miss distance) and fuel efficiency.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import math

class MockSatrec:
    """Mock Satellite object for simplified physics training."""
    def __init__(self, r, v, epoch_offset=0.0):
        self.r = np.array(r, dtype=np.float64)
        self.v = np.array(v, dtype=np.float64)
        self.epoch_offset = epoch_offset
        self.is_mock = True
        
    def update_state(self, current_time_offset, new_vel):
        """Update state (re-epoch) to apply a velocity change at a specific time."""
        # Calculate position at current time using OLD velocity
        dt = current_time_offset - self.epoch_offset
        current_r = self.r + self.v * dt
        
        # Update state: New epoch starts now, at current_r, with new_vel
        self.r = current_r
        self.v = np.array(new_vel, dtype=np.float64)
        self.epoch_offset = current_time_offset


class SpaceGym(gym.Env):
    """
    Custom Gymnasium environment for satellite collision avoidance.
    
    Observation Space:
        - Relative position (x, y, z) in km
        - Relative velocity (vx, vy, vz) in km/s
        - Time to closest approach (TCA) in seconds
        - Fuel remaining (percentage)
    
    Action Space (Discrete):
        0: Wait (no thrust)
        1: Thrust Prograde (+velocity direction)
        2: Thrust Retrograde (-velocity direction)
        3: Thrust Normal (+orbit plane perpendicular)
        4: Thrust Anti-Normal (-orbit plane perpendicular)
        5: Thrust Radial (+away from Earth)
        6: Thrust Anti-Radial (-toward Earth)
    
    Reward Function:
        +100 if miss distance > 10km (safe)
        -50 if miss distance < 1km (danger)
        -10 * fuel_used_percent (efficiency penalty)
        -1000 if collision (game over)
        -20 if time_to_tca < 60s (late action penalty)
    """
    
    metadata = {'render_modes': []}
    
    def __init__(self, sat_tle=None, debris_tle=None, tca=None, max_fuel=100.0):
        super().__init__()
        
        # Action space: 7 discrete actions (wait + 6 thrust directions)
        self.action_space = spaces.Discrete(7)
        
        # Observation space: [rel_pos(3), rel_vel(3), time_to_tca(1), fuel(1)]
        self.observation_space = spaces.Box(
            low=np.array([-10000, -10000, -10000, -10, -10, -10, 0, 0]),
            high=np.array([10000, 10000, 10000, 10, 10, 10, 86400, 100]),
            dtype=np.float32
        )
        
        # Environment state
        self.sat_tle = sat_tle
        self.debris_tle = debris_tle
        self.tca = tca
        self.max_fuel = max_fuel
        self.fuel_remaining = max_fuel
        self.current_step = 0
        self.max_steps = 100
        
        # Thrust parameters
        self.delta_v_per_action = 0.01  # km/s per thrust action
        self.fuel_cost_per_action = 0.05  # % fuel per thrust
        
        # State tracking
        self.sat_satrec = None
        self.debris_satrec = None
        self.initial_tca_time = None
        
        # Simulation Parameters
        self.dt = 10.0  # 10 seconds per step
        self.sim_elapsed = 0.0
        self.start_utc = None
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # If no TLE provided, generate random collision scenario
        if self.sat_tle is None or self.debris_tle is None:
            self.sat_tle, self.debris_tle, self.tca = self._generate_random_scenario()
        
        # Parse TLEs
        self.sat_satrec = self._parse_tle(self.sat_tle)
        self.debris_satrec = self._parse_tle(self.debris_tle)
        
        # Reset state
        self.fuel_remaining = self.max_fuel
        self.current_step = 0
        self.initial_tca_time = self._parse_time(self.tca)
        
        # Reset Sim Time
        self.sim_elapsed = 0.0
        self.start_utc = datetime.utcnow()
        
        # Get initial observation
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """Execute one timestep within the environment."""
        # Clean action input (handle numpy types)
        if hasattr(action, 'item'):
            action = int(action.item())
            
        self.current_step += 1
        self.sim_elapsed += self.dt
        
        # Apply action (thrust maneuver)
        if action > 0:  # action 0 is "wait"
            self._apply_thrust(action)
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward, done, info = self._calculate_reward()
        
        # Check if max steps reached
        if self.current_step >= self.max_steps:
            done = True
        
        truncated = False
        return obs, reward, done, truncated, info
    
    def _get_observation(self):
        """Get current observation state."""
        # Get current positions and velocities (Using Sim Time)
        sat_pos, sat_vel = self._propagate_satellite(self.sat_satrec, self.sim_elapsed)
        debris_pos, debris_vel = self._propagate_satellite(self.debris_satrec, self.sim_elapsed)
        
        # Calculate relative state
        rel_pos = np.array(debris_pos) - np.array(sat_pos)
        rel_vel = np.array(debris_vel) - np.array(sat_vel)
        
        # Time to TCA (in seconds)
        current_sim_time = self.start_utc + timedelta(seconds=self.sim_elapsed)
        time_to_tca = (self.initial_tca_time - current_sim_time).total_seconds()
        
        # Construct observation
        obs = np.array([
            rel_pos[0], rel_pos[1], rel_pos[2],
            rel_vel[0], rel_vel[1], rel_vel[2],
            time_to_tca,
            self.fuel_remaining
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_reward(self):
        """Calculate reward based on current state."""
        # Get current miss distance (Predicted minimum)
        miss_distance = self._calculate_miss_distance()
        
        # Get time to TCA (Sim Time)
        current_sim_time = self.start_utc + timedelta(seconds=self.sim_elapsed)
        time_to_tca = (self.initial_tca_time - current_sim_time).total_seconds()
        
        reward = 0
        done = False
        info = {'miss_distance_km': miss_distance / 1000}
        
        # 1. SUCCESS: Safe Separation (> 10km)
        if miss_distance > 10000:
            reward = 1000
            done = True
            info['status'] = 'success_safe'
            return reward, done, info
            
        # 2. DANGER: Collision Course
        if miss_distance < 100:
             reward -= 10 # Penalty per step for being on collision course
             info['status'] = 'collision_course'
             # DO NOT TERMINATE - Give agent chance to maneuver
        elif miss_distance < 1000:
             reward -= 1
             info['status'] = 'danger_zone'
        else:
             reward += 1 # Shaping encouragement
             info['status'] = 'nominal'
             
        # 3. FAILURE: Time Expired (Collision or Unsafe)
        if time_to_tca <= 0:
             done = True
             if miss_distance < 1000: # Still unsafe at TCA
                 reward -= 1000 # Major failure
                 info['status'] = 'collision_failure'
             else:
                 # Safeish but not 10km?
                 reward -= 100
                 info['status'] = 'timeout_unsafe'
        
        # 4. FAILURE: Out of Fuel
        if self.fuel_remaining <= 0:
             reward -= 1000
             done = True
             info['status'] = 'out_of_fuel'
             
        return reward, done, info
    
    def _apply_thrust(self, action):
        """Apply thrust maneuver based on action."""
        # Get current satellite state (using Sim Time)
        sat_pos, sat_vel = self._propagate_satellite(self.sat_satrec, self.sim_elapsed)
        
        # Calculate thrust direction based on action
        thrust_vector = self._action_to_thrust_vector(action, sat_pos, sat_vel)
        
        # Apply delta-V
        new_vel = np.array(sat_vel) + thrust_vector * self.delta_v_per_action
        
        # Update satellite TLE with new velocity
        if hasattr(self.sat_satrec, 'is_mock'):
            self.sat_satrec.update_state(self.sim_elapsed, new_vel)
        else:
            self.sat_satrec = self._update_satrec_velocity(self.sat_satrec, new_vel)
        
        # Deduct fuel
        self.fuel_remaining -= self.fuel_cost_per_action
    
    def _action_to_thrust_vector(self, action, pos, vel):
        """Convert discrete action to thrust vector."""
        # Normalize velocity for prograde/retrograde
        vel_norm = np.array(vel) / np.linalg.norm(vel)
        
        # Calculate radial (away from Earth)
        pos_norm = np.array(pos) / np.linalg.norm(pos)
        
        # Calculate normal (perpendicular to orbit plane)
        normal = np.cross(pos, vel)
        normal_norm = normal / np.linalg.norm(normal)
        
        # Map action to thrust direction
        action_map = {
            1: vel_norm,          # Prograde
            2: -vel_norm,         # Retrograde
            3: normal_norm,       # Normal
            4: -normal_norm,      # Anti-Normal
            5: pos_norm,          # Radial
            6: -pos_norm          # Anti-Radial
        }
        
        return action_map.get(action, np.array([0, 0, 0]))
    
    def _propagate_satellite(self, satrec, time_offset_seconds=0):
        """Propagate satellite position and velocity."""
        # Handle Mock Satellites (Linear Physics)
        if hasattr(satrec, 'is_mock'):
            # Linear propagation relative to its epoch
            dt = time_offset_seconds - satrec.epoch_offset
            r = satrec.r + satrec.v * dt
            v = satrec.v # Velocity constant (linear drift)
            return r, v

        # Get current time + offset
        now = datetime.utcnow() + timedelta(seconds=time_offset_seconds)
        jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)
        
        # Propagate
        e, r, v = satrec.sgp4(jd, fr)
        
        if e != 0:
            # Propagation error, return last known state
            return [0, 0, 0], [0, 0, 0]
        
        return r, v
    
    def _calculate_miss_distance(self):
        """Calculate minimum distance between satellite and debris near TCA."""
        # Optimization: Check +/- 30 mins around initial TCA
        current_time = datetime.utcnow()
        tca_offset = (self.initial_tca_time - current_time).total_seconds()
        
        # Define window (seconds relative to now)
        start_t = int(tca_offset - 1800)  # -30 mins
        end_t = int(tca_offset + 1800)    # +30 mins
        
        # Ensure we don't look too far in past
        start_t = max(0, start_t)
        
        if start_t >= end_t:
             return float('inf') # TCA passed long ago
             
        # Sample positions
        min_distance = float('inf')
        
        # Step size 10s for accuracy (vs 60s) since window is small
        for t in range(start_t, end_t, 10): 
            sat_pos, _ = self._propagate_satellite(self.sat_satrec, t)
            debris_pos, _ = self._propagate_satellite(self.debris_satrec, t)
            
            distance = np.linalg.norm(np.array(sat_pos) - np.array(debris_pos))
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _parse_tle(self, tle):
        """Parse TLE into Satrec object (or pass through Mock)."""
        if isinstance(tle, MockSatrec):
            return tle
            
        if isinstance(tle, dict):
            line1 = tle.get('line1', '')
            line2 = tle.get('line2', '')
        else:
            # Assume it's already a tuple/list
            line1, line2 = tle[0], tle[1]
        
        return Satrec.twoline2rv(line1, line2)
    
    def _parse_time(self, time_str):
        """Parse time string to datetime."""
        if isinstance(time_str, datetime):
            return time_str
        # Assume ISO format
        return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
    
    def _update_satrec_velocity(self, satrec, new_vel):
        """Update Satrec with new velocity."""
        # Handle Mock Satellite
        if hasattr(satrec, 'is_mock'):
            satrec.v = np.array(new_vel, dtype=np.float64)
            return satrec

        # Real SGP4 Satrec (Complexity Warning)
        # SGP4 objects are complex. Updating V directly breaks orbital elements B*, Mean Motion, etc.
        # For 'Real' training, we would need to solve the Lambert problem or osculating elements conversion.
        # For now, we return unchanged (No-Op for Real TLEs).
        # This implies training MUST use Mock Mode for now.
        return satrec
    
    def _generate_random_scenario(self):
        """Generate deterministic collision scenario using Mock Physics."""
        # Scenario: Collision in 15 minutes (900s)
        t_collision = 900
        
        # Satellite: LEO orbit (Simplified Linear)
        # Position: [7000 km, 0, 0]
        # Velocity: [0, 7.5 km/s, 0] (Along Y axis)
        sat_r = [7000.0, 0.0, 0.0]
        sat_v = [0.0, 7.5, 0.0]
        
        # Debris: Impact course
        # Starts 10km away in Z axis (Cross-track)
        # MUST arrive at Satellite's future position at T=900
        
        # Sat Position @ T=900:
        target_r = [
            sat_r[0] + sat_v[0] * t_collision,
            sat_r[1] + sat_v[1] * t_collision,
            sat_r[2] + sat_v[2] * t_collision
        ] # [7000, 6750, 0]
        
        # Debris Start Position:
        # Same X/Y, but Z is offset by 10km
        deb_r = [7000.0, 0.0, 10.0]
        
        # Debris Required Velocity to hit Target:
        # V = (Target - Start) / T
        deb_v = [
            (target_r[0] - deb_r[0]) / t_collision,
            (target_r[1] - deb_r[1]) / t_collision,
            (target_r[2] - deb_r[2]) / t_collision
        ]
        
        # Create Mock Objects
        # Note: We return MockSatrec objects directly?
        # But reset() expects TLE dictionary or calls this.
        # This function signature returns (sat_tle, deb_tle, tca).
        # We need to hack it to store the objects OR return Mock objects as "TLEs" 
        # and handle them in reset.
        
        # But wait, reset() calls _parse_tle. 
        # We should modify reset logic? 
        # Or make _parse_tle handle Mock objects passed through?
        
        # Hack: Return MockSatrec as the "TLE". 
        # And update _parse_tle to pass it through.
        
        sat_obj = MockSatrec(sat_r, sat_v)
        deb_obj = MockSatrec(deb_r, deb_v)
        
        tca_dt = datetime.utcnow() + timedelta(seconds=t_collision)
        
        return sat_obj, deb_obj, tca_dt.isoformat()
    
    def simulate_maneuver(self, action):
        """
        Simulate a maneuver and return the resulting trajectory.
        Used by the API endpoint for visualization.
        """
        # Apply the action
        if action > 0:
            self._apply_thrust(action)
        
        # Generate new trajectory (next 8 hours)
        trajectory = []
        for t in range(0, 28800, 300):  # Every 5 minutes for 8 hours
            pos, _ = self._propagate_satellite(self.sat_satrec, t)
            trajectory.append(pos)
        
        # Calculate fuel cost
        fuel_cost = (self.max_fuel - self.fuel_remaining) / self.max_fuel * 100
        
        # Calculate new miss distance
        miss_distance = self._calculate_miss_distance()
        
        return trajectory, fuel_cost, miss_distance


# Helper function for API integration
def action_to_vector(action):
    """Convert action index to human-readable thrust direction."""
    action_names = {
        0: "Wait (No Thrust)",
        1: "Prograde (+Velocity)",
        2: "Retrograde (-Velocity)",
        3: "Normal (+Orbit Plane)",
        4: "Anti-Normal (-Orbit Plane)",
        5: "Radial (+Away from Earth)",
        6: "Anti-Radial (-Toward Earth)"
    }
    return action_names.get(action, "Unknown")


def calculate_burn_time(action, delta_v=0.01, thrust_acceleration=0.001):
    """Calculate burn duration in seconds."""
    if action == 0:
        return 0
    # Simplified: burn_time = delta_v / acceleration
    return delta_v / thrust_acceleration


def calculate_optimal_time(tca, lead_time_hours=2):
    """Calculate optimal execution time (before TCA)."""
    tca_dt = datetime.fromisoformat(tca.replace('Z', '+00:00'))
    optimal_time = tca_dt - timedelta(hours=lead_time_hours)
    return optimal_time.isoformat() + 'Z'
