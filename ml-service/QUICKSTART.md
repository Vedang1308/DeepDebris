# DeepDebris 3.0: Quick Start Guide

## Installation Complete ✅

All dependencies have been successfully installed:
- ✅ `gymnasium==0.29.1`
- ✅ `stable-baselines3==2.2.1`
- ✅ `matplotlib` (for training visualization)
- ✅ `torch` (already installed)

---

## ⚠️ CRITICAL: Space-Track Credentials Required

**This application REQUIRES valid Space-Track.org credentials** to function. There is NO demo mode or fallback data.

### Setup Instructions:

1. Register for a free account at **[https://www.space-Track.org/auth/createAccount](https://www.space-track.org/auth/createAccount)**

2. Create `ml-service/.env` file:
```bash
SPACETRACK_USER=your_email@example.com
SPACETRACK_PASSWORD=your_password
```

3. **The application will NOT start without these credentials.**

---

## Next Steps

### Option 1: Quick Start (Recommended)
```bash
cd ml-service
./start_deepdebris3.sh
```

This script will:
1. Check if a trained model exists
2. Offer to train the agent (or skip if already trained)
3. Start the DeepDebris application

---

### Option 2: Manual Training

#### Train the RL Agent
```bash
cd ml-service

# Quick training (100k steps, ~30 min on M1/M2 Mac)
python3 rl/train_maneuver_agent.py --timesteps 100000

# Full training (1M steps, ~2-4 hours)
python3 rl/train_maneuver_agent.py --timesteps 1000000
```

**Output**: `rl/models/maneuver_agent.zip`

#### Test the Agent
```bash
python3 rl/test_agent.py --model rl/models/maneuver_agent.zip --scenarios 100
```

#### Start the Application
```bash
python3 main.py
```

Navigate to: **http://localhost:8000**

---

## Using the Maneuver Planning Feature

1. **Select a satellite** (ISS or Hubble)
2. **Click "Fetch Latest TLE"** to load orbital data
3. **Click "Analyze Collision Risks"** to detect threats
4. **Click "Generate Maneuver Plan"** to get AI recommendation
5. Review the optimal maneuver:
   - Thrust direction (e.g., "Prograde +Velocity")
   - Burn duration (seconds)
   - Execution time (UTC)
   - Fuel cost (%)
   - New miss distance (km)
6. **Click "Execute Maneuver (Simulation)"** to visualize

---

## Training Notes

- **Device**: Auto-detects Apple Silicon MPS, CUDA, or CPU
- **Checkpoints**: Saved every 10,000 steps in `checkpoints/`
- **Best Model**: Saved in `logs/best_model/`
- **TensorBoard**: View training progress with `tensorboard --logdir logs/tensorboard`

---

## Troubleshooting

### Model Not Found Error
If you see "RL Maneuver Agent not found" in the console:
```bash
python3 rl/train_maneuver_agent.py --timesteps 100000
```

### Import Errors
Ensure you're using the correct Python environment:
```bash
which python3
pip3 list | grep gymnasium
```

### Training Too Slow
Reduce timesteps for faster testing:
```bash
python3 rl/train_maneuver_agent.py --timesteps 10000
```

---

## System Status

| Component | Status |
|:----------|:-------|
| RL Environment | ✅ Ready |
| Training Scripts | ✅ Ready |
| Backend API | ✅ Ready |
| Frontend UI | ✅ Ready |
| Dependencies | ✅ Installed |
| Trained Model | ⏳ Pending (run training) |

---

## What's Next?

After training completes, the system will be a **fully autonomous Level 3/4 platform** capable of:
- Real-time collision detection
- AI-powered trajectory prediction
- **Autonomous maneuver planning** (NEW!)
- Fuel-optimized collision avoidance

Enjoy DeepDebris 3.0! 🚀
