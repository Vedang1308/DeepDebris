"""
Train Maneuver Agent: PPO Training Script for Collision Avoidance

This script trains a Deep RL agent using Proximal Policy Optimization (PPO)
to learn optimal collision avoidance maneuvers.

Usage:
    python train_maneuver_agent.py --timesteps 1000000 --save_path models/maneuver_agent
"""

import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
from space_gym import SpaceGym


def train_agent(total_timesteps=1_000_000, save_path="models/maneuver_agent", device="auto"):
    """
    Train PPO agent for collision avoidance.
    
    Args:
        total_timesteps: Total training timesteps
        save_path: Path to save trained model
        device: Device to use ('auto', 'cuda', 'mps', 'cpu')
    """
    print("=" * 60)
    print("DeepDebris 3.0: Autonomous Maneuver Agent Training")
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
    
    # Create vectorized environment (4 parallel environments for faster training)
    print("\n[1/4] Creating training environment...")
    env = make_vec_env(
        SpaceGym,
        n_envs=4,
        monitor_dir="logs/training"
    )
    
    # Create evaluation environment
    eval_env = Monitor(SpaceGym())
    
    # Create callbacks
    print("[2/4] Setting up callbacks...")
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
    
    # Initialize PPO agent
    print("[3/4] Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,          # Discount factor
        gae_lambda=0.95,     # GAE lambda
        clip_range=0.2,      # PPO clip range
        ent_coef=0.01,       # Entropy coefficient (exploration)
        vf_coef=0.5,         # Value function coefficient
        max_grad_norm=0.5,   # Gradient clipping
        verbose=1,
        device=device,
        tensorboard_log="./logs/tensorboard/"
    )
    
    print("\nModel Architecture:")
    print(model.policy)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # Train the agent
    print(f"\n[4/4] Training for {total_timesteps:,} timesteps...")
    print("=" * 60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\n✓ Model saved to: {save_path}.zip")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nTo test the agent, run:")
    print(f"  python test_agent.py --model {save_path}.zip")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for collision avoidance")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1,000,000)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="models/maneuver_agent",
        help="Path to save trained model (default: models/maneuver_agent)"
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
    
    # Train
    train_agent(
        total_timesteps=args.timesteps,
        save_path=args.save_path,
        device=args.device
    )


if __name__ == "__main__":
    main()
