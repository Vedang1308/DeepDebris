import numpy as np

class VisualServoController:
    """
    Classical Control Layer (PID) for 'The Last 10 Meters'.
    Uses Computer Vision estimates to align Chaser with Target.
    """
    def __init__(self, kp_spin=0.5, kd_spin=0.1):
        self.kp_spin = kp_spin # Proportional Gain
        self.kd_spin = kd_spin # Derivative Gain
        
        self.prev_error_w = np.array([0.0, 0.0, 0.0])
        
    def compute_control(self, current_tumble, target_tumble):
        """
        Calculate control torque to match spin.
        Args:
            current_tumble (list): [wx, wy, wz] of Chaser (Usually 0 if stabilized)
            target_tumble (list): [wx, wy, wz] estimated from Vision
        Returns:
            dict: { 'torque': [tau_x, tau_y, tau_z], 'status': 'MATCHING' }
        """
        w_chaser = np.array(current_tumble)
        w_target = np.array(target_tumble)
        
        # Error: Difference in angular velocity
        error_w = w_target - w_chaser
        
        # PID Control Law (PD for rate matching)
        # Torque = Kp * error + Kd * (error - prev_error)
        d_error = error_w - self.prev_error_w
        torque = self.kp_spin * error_w + self.kd_spin * d_error
        
        self.prev_error_w = error_w
        
        # Status Logic
        norm_error = np.linalg.norm(error_w)
        status = "LOCKED" if norm_error < 0.01 else "MATCHING SPIN"
        
        return {
            "torque": torque.tolist(),
            "status": status,
            "error_norm": float(norm_error)
        }

if __name__ == "__main__":
    controller = VisualServoController()
    raw_output = controller.compute_control([0,0,0], [0.1, 0.0, -0.05])
    print(f"Test Output: {raw_output}")
