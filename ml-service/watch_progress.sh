#!/bin/bash
# Simple progress monitor for macOS (no 'watch' needed)

echo "DeepDebris 3.0 Training Monitor"
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "🚀 Training Progress"
    echo "=========================================="
    echo ""
    
    # Get latest timesteps from training process
    if pgrep -f "train_maneuver_agent.py" > /dev/null; then
        echo "Status: ✅ TRAINING ACTIVE"
        echo ""
        
        # Try to get progress from monitor.csv
        if [ -f "logs/training/monitor.csv" ]; then
            lines=$(wc -l < logs/training/monitor.csv)
            timesteps=$((lines * 100))  # Rough estimate
            percent=$(echo "scale=1; $timesteps / 10000" | bc)
            echo "Estimated Progress: ~${percent}%"
            echo "Timesteps: ~${timesteps}"
        fi
        
        echo ""
        echo "View detailed graphs: http://localhost:6006/"
    else
        echo "Status: ⚠️  Training not detected"
    fi
    
    echo ""
    echo "=========================================="
    echo "Last updated: $(date '+%H:%M:%S')"
    echo "=========================================="
    
    sleep 5
done
