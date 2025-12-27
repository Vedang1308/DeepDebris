#!/bin/bash
# DeepDebris 3.0: Quick Start Script
# This script trains the RL agent and starts the application

echo "=========================================="
echo "DeepDebris 3.0: Autonomous Maneuver Agent"
echo "=========================================="
echo ""

# Check if model exists
if [ -f "rl/models/maneuver_agent.zip" ]; then
    echo "✓ Trained model found: rl/models/maneuver_agent.zip"
    echo ""
    read -p "Do you want to retrain the agent? (y/N): " retrain
    if [[ $retrain =~ ^[Yy]$ ]]; then
        echo ""
        echo "[1/2] Training RL Agent (this may take 2-4 hours)..."
        python3 rl/train_maneuver_agent.py --timesteps 1000000
    fi
else
    echo "⚠ No trained model found. Training required."
    echo ""
    read -p "Start training now? (Y/n): " start_training
    if [[ ! $start_training =~ ^[Nn]$ ]]; then
        echo ""
        echo "[1/2] Training RL Agent (this may take 2-4 hours)..."
        python3 rl/train_maneuver_agent.py --timesteps 1000000
    else
        echo ""
        echo "Skipping training. The /plan_maneuver endpoint will return 503 errors."
        echo "To train later, run: python3 rl/train_maneuver_agent.py"
    fi
fi

echo ""
echo "[2/2] Starting DeepDebris Application..."
echo "Navigate to: http://localhost:8000"
echo ""
python3 main.py
