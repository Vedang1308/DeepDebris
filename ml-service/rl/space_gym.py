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
        
        # Get initial observation
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """Execute one timestep within the environment."""
        self.current_step += 1
        
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
        # Get current positions and velocities
        sat_pos, sat_vel = self._propagate_satellite(self.sat_satrec)
        debris_pos, debris_vel = self._propagate_satellite(self.debris_satrec)
        
        # Calculate relative state
        rel_pos = np.array(debris_pos) - np.array(sat_pos)
        rel_vel = np.array(debris_vel) - np.array(sat_vel)
        
        # Time to TCA (in seconds)
        current_time = datetime.utcnow()
        time_to_tca = (self.initial_tca_time - current_time).total_seconds()
        
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
        # Get current miss distance
        miss_distance = self._calculate_miss_distance()
        
        # Get time to TCA
        current_time = datetime.utcnow()
        time_to_tca = (self.initial_tca_time - current_time).total_seconds()
        
        reward = 0
        done = False
        info = {'miss_distance_km': miss_distance / 1000}
        
        # Safety reward (primary objective)
        if miss_distance > 10000:  # 10km safe
            reward += 100
            done = True  # Mission success
            info['status'] = 'safe'
        elif miss_distance < 1000:  # 1km danger
            reward -= 50
            info['status'] = 'danger'
        
        # Collision check
        if miss_distance < 100:  # 100m = collision
            reward -= 1000
            done = True
            info['status'] = 'collision'
        
        # Time penalty (encourage early action)
        if time_to_tca < 60 and time_to_tca > 0:
            reward -= 20
        
        # Fuel efficiency penalty (applied on every action)
        fuel_used_percent = (self.max_fuel - self.fuel_remaining) / self.max_fuel * 100
        reward -= 10 * fuel_used_percent
        
        # Out of fuel check
        if self.fuel_remaining <= 0:
            reward -= 500
            done = True
            info['status'] = 'out_of_fuel'
        
        return reward, done, info
    
    def _apply_thrust(self, action):
        """Apply thrust maneuver based on action."""
        # Get current satellite state
        sat_pos, sat_vel = self._propagate_satellite(self.sat_satrec)
        
        # Calculate thrust direction based on action
        thrust_vector = self._action_to_thrust_vector(action, sat_pos, sat_vel)
        
        # Apply delta-V
        new_vel = np.array(sat_vel) + thrust_vector * self.delta_v_per_action
        
        # Update satellite TLE with new velocity (simplified)
        # In reality, you'd need to convert velocity change to TLE elements
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
        """Calculate minimum distance between satellite and debris."""
        # Sample positions over next 24 hours
        min_distance = float('inf')
        
        for t in range(0, 86400, 60):  # Sample every minute
            sat_pos, _ = self._propagate_satellite(self.sat_satrec, t)
            debris_pos, _ = self._propagate_satellite(self.debris_satrec, t)
            
            distance = np.linalg.norm(np.array(sat_pos) - np.array(debris_pos))
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _parse_tle(self, tle):
        """Parse TLE into Satrec object."""
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
        """Update Satrec with new velocity (simplified)."""
        # This is a simplified version - in reality you'd need to:
        # 1. Convert ECI velocity to Keplerian elements
        # 2. Update TLE elements
        # 3. Recreate Satrec
        # For now, we'll just return the same satrec (placeholder)
        return satrec
    
    def _generate_random_scenario(self):
        """Generate random collision scenario for training."""
        # Generate random ISS-like orbit
        sat_tle = {
            'line1': '1 25544U 98067A   25361.50000000  .00002182  00000-0  41420-4 0  9990',
            'line2': '2 25544  51.6461 339.8014 0002571  85.5211 274.6305 15.48919393123456'
        }
        
        # Generate random debris with collision course
        debris_tle = {
            'line1': '1 99999U 99999A   25361.50000000  .00000000  00000-0  00000-0 0  9999',
            'line2': '2 99999  51.6500 339.8000 0002600  85.5000 274.6000 15.48900000000001'
        }
        
        # Random TCA within next 24 hours
        tca = datetime.utcnow() + timedelta(hours=np.random.uniform(1, 24))
        
        return sat_tle, debris_tle, tca.isoformat()
    
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
