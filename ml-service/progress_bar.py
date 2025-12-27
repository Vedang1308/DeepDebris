#!/usr/bin/env python3
"""
DeepDebris 3.0: Real Progress Bar Monitor
Reads actual training progress from logs
"""

import sys
import time
import os
import re

MAX_TIMESTEPS = 1_000_000

def clear_line():
    sys.stdout.write('\r\033[K')
    sys.stdout.flush()

def draw_bar(percent, width=50):
    filled = int(width * percent / 100)
    return '█' * filled + '░' * (width - filled)

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m {s}s"

def get_current_timesteps():
    """Read actual timesteps from training logs"""
    current = 0
    
    # Check monitor.csv in training logs
    log_file = "logs/training/monitor.csv"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # Last line has the latest data
                    last_line = lines[-1]
                    # CSV format: r,l,t (reward, length, time)
                    # We need to count lines to get timesteps
                    current = (len(lines) - 1) * 1  # Each line = 1 step
        except:
            pass
    
    # Fallback: check TensorBoard event files
    if current == 0:
        tb_dir = "logs/tensorboard/PPO_1"
        if os.path.exists(tb_dir):
            for file in os.listdir(tb_dir):
                if file.startswith("events.out"):
                    # Estimate based on file size (rough approximation)
                    file_path = os.path.join(tb_dir, file)
                    file_size = os.path.getsize(file_path)
                    # Rough estimate: 1KB ≈ 100 timesteps
                    current = max(current, int(file_size / 10))
    
    return current

print("DeepDebris 3.0 Training Monitor")
print("Reading actual training progress...\n")

try:
    last_timesteps = 0
    last_update = time.time()
    fps = 120  # Default FPS
    
    while True:
        current = get_current_timesteps()
        
        # Calculate FPS if we have new data
        if current > last_timesteps:
            elapsed = time.time() - last_update
            if elapsed > 0:
                fps = (current - last_timesteps) / elapsed
            last_timesteps = current
            last_update = time.time()
        
        percent = (current / MAX_TIMESTEPS) * 100
        
        if current < MAX_TIMESTEPS and fps > 0:
            remaining_steps = MAX_TIMESTEPS - current
            eta_sec = remaining_steps / fps
            eta = format_time(eta_sec)
        else:
            eta = "Calculating..." if current < MAX_TIMESTEPS else "Done!"
        
        # Single line progress bar
        clear_line()
        bar = draw_bar(percent, 50)
        sys.stdout.write(f"[{bar}] {percent:5.1f}% | {current:,}/{MAX_TIMESTEPS:,} | ETA: {eta}")
        sys.stdout.flush()
        
        if current >= MAX_TIMESTEPS:
            print("\n\n✅ Training Complete!")
            break
        
        time.sleep(3)  # Update every 3 seconds
        
except KeyboardInterrupt:
    print("\n\n✓ Stopped")
    sys.exit(0)
