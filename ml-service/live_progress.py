#!/usr/bin/env python3
"""
DeepDebris 3.0: Live Training Progress Monitor
Shows real-time progress bar with metrics
"""

import os
import sys
import time
import re
from datetime import timedelta

def get_training_progress():
    """Parse training logs to get current progress"""
    log_dirs = [
        "logs/training",
        "logs/tensorboard/PPO_1"
    ]
    
    max_timesteps = 1_000_000
    current_timesteps = 0
    
    # Try to find latest timestep from any log file
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            for file in os.listdir(log_dir):
                if file.endswith('.csv') or file.endswith('.log'):
                    try:
                        with open(os.path.join(log_dir, file), 'r') as f:
                            content = f.read()
                            # Look for total_timesteps
                            matches = re.findall(r'total_timesteps[,\s]+(\d+)', content)
                            if matches:
                                current_timesteps = max(current_timesteps, int(matches[-1]))
                    except:
                        pass
    
    return current_timesteps, max_timesteps

def format_time(seconds):
    """Format seconds into human-readable time"""
    return str(timedelta(seconds=int(seconds)))

def draw_progress_bar(percent, width=50):
    """Draw a fancy progress bar"""
    filled = int(width * percent / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f'[{bar}]'

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')

def main():
    print("DeepDebris 3.0: Live Training Monitor")
    print("Press Ctrl+C to exit\n")
    
    start_time = time.time()
    last_timesteps = 0
    last_update = time.time()
    
    try:
        while True:
            current_timesteps, max_timesteps = get_training_progress()
            
            if current_timesteps > 0:
                # Calculate metrics
                percent = (current_timesteps / max_timesteps) * 100
                elapsed = time.time() - start_time
                
                # Calculate speed (timesteps per second)
                if current_timesteps > last_timesteps:
                    time_diff = time.time() - last_update
                    speed = (current_timesteps - last_timesteps) / time_diff if time_diff > 0 else 0
                    last_timesteps = current_timesteps
                    last_update = time.time()
                else:
                    speed = 0
                
                # Estimate remaining time
                if speed > 0:
                    remaining_timesteps = max_timesteps - current_timesteps
                    eta_seconds = remaining_timesteps / speed
                    eta = format_time(eta_seconds)
                else:
                    eta = "Calculating..."
                
                # Clear and redraw
                clear_screen()
                print("=" * 70)
                print("🚀 DeepDebris 3.0: Autonomous Maneuver Agent Training")
                print("=" * 70)
                print()
                
                # Progress bar
                print(f"Progress: {draw_progress_bar(percent, 50)}")
                print()
                
                # Metrics
                print(f"  Timesteps:     {current_timesteps:,} / {max_timesteps:,}")
                print(f"  Completion:    {percent:.2f}%")
                print(f"  Speed:         {speed:.0f} steps/sec")
                print(f"  Elapsed:       {format_time(elapsed)}")
                print(f"  ETA:           {eta}")
                print()
                
                # Status
                if percent < 25:
                    status = "🟡 Early Training - Agent exploring environment"
                elif percent < 50:
                    status = "🟢 Learning Phase - Policy improving"
                elif percent < 75:
                    status = "🔵 Refinement - Fine-tuning strategies"
                else:
                    status = "🟣 Final Phase - Convergence"
                
                print(f"  Status:        {status}")
                print()
                print("=" * 70)
                print("TensorBoard: http://localhost:6006/")
                print("Press Ctrl+C to exit")
                print("=" * 70)
            else:
                clear_screen()
                print("=" * 70)
                print("⏳ Waiting for training to start...")
                print("=" * 70)
                print("\nNo training data detected yet.")
                print("The agent may still be initializing the environment.")
            
            time.sleep(2)  # Update every 2 seconds
            
    except KeyboardInterrupt:
        print("\n\n✓ Monitor stopped")
        sys.exit(0)

if __name__ == "__main__":
    main()
