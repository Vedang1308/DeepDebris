#!/bin/bash
# Live Training Monitor for DeepDebris 3.0

echo "=========================================="
echo "DeepDebris 3.0: Training Monitor"
echo "=========================================="
echo ""

# Check if training is running
if pgrep -f "train_maneuver_agent.py" > /dev/null; then
    echo "✓ Training is ACTIVE"
    echo ""
    
    # Show latest checkpoint
    if [ -d "checkpoints" ]; then
        latest=$(ls -t checkpoints/*.zip 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            echo "Latest checkpoint: $(basename $latest)"
            echo "Checkpoint size: $(du -h $latest | cut -f1)"
        fi
    fi
    
    echo ""
    echo "Monitoring logs (Ctrl+C to exit)..."
    echo "=========================================="
    
    # Tail the training logs
    if [ -d "logs/training" ]; then
        tail -f logs/training/monitor.csv 2>/dev/null || echo "Waiting for logs..."
    else
        echo "Log directory not found. Training may still be initializing..."
    fi
else
    echo "⚠ Training is NOT running"
    echo ""
    echo "To start training:"
    echo "  python3 rl/train_maneuver_agent.py --timesteps 1000000"
fi
