#!/bin/bash
# Push training progress from GPU supercomputer to GitHub

echo "=========================================="
echo "Pushing Training Progress to GitHub"
echo "=========================================="
echo ""

# Navigate to project directory
cd ~/DeepDebris/DeepDebris/ml-service

# Add all new checkpoints
echo "Adding checkpoints..."
git add checkpoints/

# Add any logs (optional - they're large)
# git add logs/best_model/

# Check what will be committed
echo ""
echo "Files to commit:"
git status --short

echo ""
read -p "Continue with commit? (Y/n): " confirm
if [[ ! $confirm =~ ^[Nn]$ ]]; then
    # Get latest checkpoint info
    latest=$(ls -t checkpoints/*.zip | head -1)
    steps=$(echo $latest | grep -oP '\d+(?=_steps)')
    
    # Commit with progress info
    git commit -m "Training progress: checkpoint at ${steps} steps on A100 GPU

- Trained on NVIDIA A100-SXM4-80GB
- Checkpoints saved every 10,000 steps
- Ready for continuation on Mac or other system"
    
    # Push to GitHub
    echo ""
    echo "Pushing to GitHub..."
    git push origin main
    
    echo ""
    echo "=========================================="
    echo "✓ Training progress pushed to GitHub!"
    echo "=========================================="
    echo ""
    echo "Latest checkpoint: $latest"
    echo "Steps completed: $steps"
else
    echo "Cancelled."
fi
