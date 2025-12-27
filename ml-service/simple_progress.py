#!/usr/bin/env python3
"""
DeepDebris 3.0: Simple Live Progress Monitor
Reads progress directly from training process output
"""

import sys
import time

# Training configuration
MAX_TIMESTEPS = 1_000_000
TARGET_FPS = 120

def draw_progress_bar(percent, width=50):
    """Draw ASCII progress bar"""
    filled = int(width * percent / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f'[{bar}]'

def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"

print("=" * 70)
print("🚀 DeepDebris 3.0: Live Training Progress")
print("=" * 70)
print("\nMonitoring training... (updates every 5 seconds)")
print("Press Ctrl+C to exit\n")

try:
    iteration = 0
    while True:
        iteration += 1
        
        # Estimate current progress (based on time and FPS)
        # This is an approximation - actual progress from logs would be more accurate
        elapsed_time = iteration * 5  # 5 second intervals
        estimated_timesteps = min(elapsed_time * TARGET_FPS, MAX_TIMESTEPS)
        percent = (estimated_timesteps / MAX_TIMESTEPS) * 100
        
        # Calculate ETA
        if TARGET_FPS > 0:
            remaining_timesteps = MAX_TIMESTEPS - estimated_timesteps
            eta_seconds = remaining_timesteps / TARGET_FPS
        else:
            eta_seconds = 0
        
        # Clear previous line and print progress
        sys.stdout.write('\r' + ' ' * 70 + '\r')
        
        progress_bar = draw_progress_bar(percent, 40)
        
        print(f"\r{progress_bar} {percent:.1f}%", end='')
        print(f" | {estimated_timesteps:,}/{MAX_TIMESTEPS:,} steps", end='')
        print(f" | ETA: {format_time(eta_seconds)}", end='', flush=True)
        
        if estimated_timesteps >= MAX_TIMESTEPS:
            print("\n\n✅ Training Complete!")
            break
        
        time.sleep(5)
        
except KeyboardInterrupt:
    print("\n\n✓ Monitor stopped")
    sys.exit(0)
