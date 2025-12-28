# Training Transfer Guide: GPU → Mac

## On GPU Supercomputer

```bash
cd ~/DeepDebris/DeepDebris/ml-service
git add checkpoints/
git commit -m "Training progress from A100 GPU"
git push origin main
```

## On Your Mac

```bash
cd /Users/vedangavaghade/Desktop/LEO/DeepDebris
git pull origin main
cd ml-service
python3 rl/resume_training.py
```

The script will automatically detect and resume from the latest checkpoint!
