#!/usr/bin/env python3
"""
Resume Training from Checkpoint
Continues training from a saved checkpoint on GPU
"""

import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
from space_gym import SpaceGym


def resume_training(checkpoint_path, total_timesteps=1_000_000, device="auto"):
    """
    Resume training from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        total_timesteps: Total timesteps to train to
        device: Device to use ('auto', 'cuda', 'mps', 'cpu')
    """
    print("=" * 60)
    print("DeepDebris 3.0: Resuming Training from Checkpoint")
    print("=" * 60)
    
    # Auto-detect device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("✓ Using Apple Silicon MPS")
        else:
            device = "cpu"
            print("⚠ Using CPU (training will be slow)")
    
    # Load checkpoint
    print(f"\n[1/4] Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = PPO.load(checkpoint_path, device=device)
    
    # Get current timesteps from checkpoint name
    import re
    match = re.search(r'(\d+)_steps', checkpoint_path)
    if match:
        current_steps = int(match.group(1))
        print(f"✓ Loaded checkpoint at {current_steps:,} steps")
    else:
        current_steps = 0
        print("⚠ Could not determine current steps from filename")
    
    # Create vectorized environment
    print("\n[2/4] Creating training environment...")
    env = make_vec_env(
        SpaceGym,
        n_envs=4,
        monitor_dir="logs/training"
    )
    
    # Update model's environment
    model.set_env(env)
    
    # Create evaluation environment
    eval_env = Monitor(SpaceGym())
    
    # Create callbacks
    print("[3/4] Setting up callbacks...")
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="maneuver_agent"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/eval/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Calculate remaining timesteps
    remaining_steps = total_timesteps - current_steps
    
    print(f"\n[4/4] Resuming training...")
    print(f"  Current:   {current_steps:,} steps")
    print(f"  Target:    {total_timesteps:,} steps")
    print(f"  Remaining: {remaining_steps:,} steps")
    print("=" * 60)
    
    # Continue training
    model.learn(
        total_timesteps=remaining_steps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
        reset_num_timesteps=False  # Important: don't reset timestep counter
    )
    
    # Save final model
    final_path = "models/maneuver_agent"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    model.save(final_path)
    print(f"\n✓ Final model saved to: {final_path}.zip")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nTo test the agent, run:")
    print(f"  python test_agent.py --model {final_path}.zip")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Resume PPO training from checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., checkpoints/maneuver_agent_140000_steps.zip)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps to reach (default: 1,000,000)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/training", exist_ok=True)
    os.makedirs("logs/eval", exist_ok=True)
    os.makedirs("logs/best_model", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Resume training
    resume_training(
        checkpoint_path=args.checkpoint,
        total_timesteps=args.timesteps,
        device=args.device
    )


if __name__ == "__main__":
    main()
