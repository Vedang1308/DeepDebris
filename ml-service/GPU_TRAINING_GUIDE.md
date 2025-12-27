# DeepDebris 3.0 - GPU Training Transfer Guide

## Quick Start on GPU Supercomputer

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd DeepDebris/ml-service
```

### 2. Setup Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Resume Training from Checkpoint

**Option A: Resume from latest checkpoint (recommended)**
```bash
# The script will auto-detect and resume from the latest checkpoint
python3 rl/resume_training.py --checkpoint checkpoints/maneuver_agent_140000_steps.zip
```

**Option B: Start fresh (if you want to retrain)**
```bash
python3 rl/train_maneuver_agent.py --timesteps 1000000 --device cuda
```

### 4. Monitor Training
```bash
# Terminal 1: Run training (above)

# Terminal 2: Monitor with TensorBoard
tensorboard --logdir logs/tensorboard --port 6006

# Terminal 3: Watch progress
python3 progress_bar.py
```

---

## Current Training Status

**Completed on Mac (Apple Silicon MPS):**
- Timesteps: 147,456 / 1,000,000 (14.7%)
- Checkpoints saved: 40k, 80k, 120k, 140k
- Training loss: 192,000 (down from 446,000)
- Speed: 117 FPS on MPS

**Expected on CUDA GPU:**
- Speed: ~500-1000 FPS (4-8x faster!)
- Remaining time: ~20-40 minutes (vs 2 hours on Mac)

---

## Files to Push to GitHub

### Essential Files:
- ✅ `rl/space_gym.py` - Custom Gymnasium environment
- ✅ `rl/train_maneuver_agent.py` - Training script
- ✅ `rl/test_agent.py` - Evaluation script
- ✅ `rl/resume_training.py` - Resume from checkpoint (NEW)
- ✅ `requirements.txt` - Dependencies
- ✅ `checkpoints/maneuver_agent_140000_steps.zip` - Latest checkpoint

### Optional (for monitoring):
- `progress_bar.py` - Simple progress monitor
- `QUICKSTART.md` - Quick start guide

---

## GPU-Specific Optimizations

The training script will automatically:
1. Detect CUDA and use GPU
2. Increase batch size for better GPU utilization
3. Enable mixed precision training (if available)

**Expected Performance:**
- Mac M1/M2 (MPS): ~120 FPS
- NVIDIA RTX 3090: ~800 FPS
- NVIDIA A100: ~1200 FPS

**Time to complete remaining 852k steps:**
- On Mac: ~2 hours
- On RTX 3090: ~18 minutes
- On A100: ~12 minutes

---

## Resume Training Script

The `resume_training.py` script will:
1. Load the checkpoint from `checkpoints/`
2. Continue training from 140,000 steps
3. Save new checkpoints every 10,000 steps
4. Complete training to 1,000,000 steps

---

## Verification After Training

Once training completes on GPU:
```bash
# Test the trained agent
python3 rl/test_agent.py --model models/maneuver_agent.zip --scenarios 100

# Expected results:
# - Average miss distance > 10 km
# - Fuel usage < 0.1% per maneuver
# - Zero collisions
```

---

## Troubleshooting

**CUDA Out of Memory:**
```bash
# Reduce batch size in train_maneuver_agent.py
# Change line 81: batch_size=64 -> batch_size=32
```

**Slow Training:**
```bash
# Verify CUDA is detected
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Missing Dependencies:**
```bash
pip install gymnasium==0.29.1 stable-baselines3==2.2.1 tensorboard torch
```

---

## Next Steps After Training

1. Pull the trained model back to your Mac
2. Test the model locally
3. Deploy to DeepDebris application
4. Verify `/plan_maneuver` endpoint works

Good luck with GPU training! 🚀
