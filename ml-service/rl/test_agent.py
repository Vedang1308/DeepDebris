"""
Test Maneuver Agent: Evaluation Script

Tests the trained RL agent on random collision scenarios and reports performance metrics.

Usage:
    python test_agent.py --model models/maneuver_agent.zip --scenarios 100
"""

import argparse
import numpy as np
from stable_baselines3 import PPO
from space_gym import SpaceGym
from tqdm import tqdm


def test_agent(model_path, num_scenarios=100):
    """
    Test trained agent on random scenarios.
    
    Args:
        model_path: Path to trained model
        num_scenarios: Number of test scenarios
    """
    print("=" * 60)
    print("DeepDebris 3.0: Agent Evaluation")
    print("=" * 60)
    
    # Load trained model
    print(f"\n[1/3] Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    print("[2/3] Creating test environment...")
    env = SpaceGym()
    
    # Test metrics
    results = {
        'miss_distances': [],
        'fuel_costs': [],
        'collisions': 0,
        'successes': 0,
        'actions_taken': []
    }
    
    print(f"[3/3] Running {num_scenarios} test scenarios...")
    print("=" * 60)
    
    for i in tqdm(range(num_scenarios), desc="Testing"):
        obs, info = env.reset()
        done = False
        episode_actions = []
        
        while not done:
            # Agent predicts action
            action, _states = model.predict(obs, deterministic=True)
            episode_actions.append(int(action))
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            
            if done or truncated:
                # Record results
                miss_dist_km = info.get('miss_distance_km', 0)
                results['miss_distances'].append(miss_dist_km)
                
                fuel_cost = (env.max_fuel - env.fuel_remaining) / env.max_fuel * 100
                results['fuel_costs'].append(fuel_cost)
                
                if info.get('status') == 'collision':
                    results['collisions'] += 1
                elif info.get('status') == 'safe':
                    results['successes'] += 1
                
                results['actions_taken'].append(episode_actions)
                break
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    print(f"\nTotal Scenarios: {num_scenarios}")
    print(f"Successes (>10km): {results['successes']} ({results['successes']/num_scenarios*100:.1f}%)")
    print(f"Collisions (<100m): {results['collisions']} ({results['collisions']/num_scenarios*100:.1f}%)")
    
    print(f"\nMiss Distance:")
    print(f"  Average: {np.mean(results['miss_distances']):.2f} km")
    print(f"  Median: {np.median(results['miss_distances']):.2f} km")
    print(f"  Min: {np.min(results['miss_distances']):.2f} km")
    print(f"  Max: {np.max(results['miss_distances']):.2f} km")
    
    print(f"\nFuel Cost:")
    print(f"  Average: {np.mean(results['fuel_costs']):.3f}%")
    print(f"  Median: {np.median(results['fuel_costs']):.3f}%")
    print(f"  Max: {np.max(results['fuel_costs']):.3f}%")
    
    # Action distribution
    all_actions = [a for episode in results['actions_taken'] for a in episode]
    action_counts = {i: all_actions.count(i) for i in range(7)}
    
    print(f"\nAction Distribution:")
    action_names = [
        "Wait",
        "Prograde",
        "Retrograde",
        "Normal",
        "Anti-Normal",
        "Radial",
        "Anti-Radial"
    ]
    for i, name in enumerate(action_names):
        count = action_counts.get(i, 0)
        pct = count / len(all_actions) * 100 if all_actions else 0
        print(f"  {name:15s}: {count:4d} ({pct:5.1f}%)")
    
    print("\n" + "=" * 60)
    
    # Success criteria
    avg_miss = np.mean(results['miss_distances'])
    avg_fuel = np.mean(results['fuel_costs'])
    
    print("\nSuccess Criteria:")
    print(f"  ✓ Avg miss distance > 5km: {'PASS' if avg_miss > 5 else 'FAIL'}")
    print(f"  ✓ Avg fuel usage < 0.1%: {'PASS' if avg_fuel < 0.1 else 'FAIL'}")
    print(f"  ✓ Zero collisions: {'PASS' if results['collisions'] == 0 else 'FAIL'}")
    
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Test trained maneuver agent")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.zip file)"
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=100,
        help="Number of test scenarios (default: 100)"
    )
    
    args = parser.parse_args()
    
    test_agent(args.model, args.scenarios)


if __name__ == "__main__":
    main()
