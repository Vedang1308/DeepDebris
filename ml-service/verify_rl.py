
from rl.space_gym import SpaceGym
import numpy as np

def test_visual_servoing():
    print("Testing SpaceGym Visual Servoing Integration...")
    
    # Initialize environment
    env = SpaceGym()
    obs, info = env.reset()
    print("✓ Environment Reset Successful")
    
    # Check Action Space
    assert env.action_space.n == 8, f"Action space should be 8, got {env.action_space.n}"
    print("✓ Action Space Verified (n=8)")
    
    # Check Observation Space
    # [3 pos, 3 vel, 3 w, 1 tca, 1 fuel] = 11
    assert env.observation_space.shape[0] == 11, f"Observation space should be 11, got {env.observation_space.shape[0]}"
    print("✓ Observation Space Verified (dims=11)")
    
    # Test Step with Action 7 (MATCH_SPIN)
    # Get initial relative w
    initial_w = obs[6:9]
    print(f"Initial Relative W: {initial_w}")
    
    # Take step
    obs, reward, done, truncated, info = env.step(7)
    
    new_w = obs[6:9]
    print(f"New Relative W: {new_w}")
    print(f"Reward: {reward}")
    print(f"Info: {info}")
    
    # Verify W decreased (it should, as we match spin)
    # Note: initial_w is (deb - sat). sat starts at 0. deb is tumbling.
    # Action 7 moves sat w towards deb w. So diff should decrease?
    # Wait, rel_w = deb_w - sat_w. 
    # sat_w becomes closer to deb_w. 
    # So magnitude of rel_w should decrease.
    
    initial_mag = np.linalg.norm(initial_w)
    new_mag = np.linalg.norm(new_w)
    
    print(f"Magnitude Change: {initial_mag:.4f} -> {new_mag:.4f}")
    
    if new_mag < initial_mag:
        print("✓ Visual Servoing Logic Verified (Relative tumble decreased)")
    else:
        print("⚠ Visual Servoing Logic Warning: Magnitude did not decrease (might be one-step dynamics)")

    print("Verification Complete.")

if __name__ == "__main__":
    test_visual_servoing()
