#!/bin/bash
# Quick script to push DeepDebris 3.0 to GitHub for GPU training

echo "=========================================="
echo "DeepDebris 3.0: GitHub Push Script"
echo "=========================================="
echo ""

cd /Users/vedangavaghade/Desktop/LEO/DeepDebris

# Check if git repo exists
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
fi

# Add all files
echo "Adding files to Git..."
git add .

# Create commit
echo "Creating commit..."
git commit -m "DeepDebris 3.0: Add RL training infrastructure and checkpoint at 140k steps

- Added custom Gymnasium environment (space_gym.py)
- Added PPO training script (train_maneuver_agent.py)
- Added resume training script (resume_training.py)
- Added test/evaluation script (test_agent.py)
- Included checkpoint at 140,000 steps (14% complete)
- Added GPU training guide
- Updated requirements.txt with RL dependencies

Ready for GPU training continuation on supercomputer.
Current progress: 140k/1M steps, Loss: 192k (57% reduction)"

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Add remote (if not already added):"
echo "   git remote add origin <your-repo-url>"
echo ""
echo "2. Push to GitHub:"
echo "   git push -u origin main"
echo ""
echo "3. On GPU supercomputer:"
echo "   git clone <your-repo-url>"
echo "   cd DeepDebris/ml-service"
echo "   pip install -r requirements.txt"
echo "   python3 rl/resume_training.py --checkpoint checkpoints/maneuver_agent_140000_steps.zip"
echo "=========================================="
