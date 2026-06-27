"""
Smoke test: verifies that sumo-rl environment runs without errors.
Run this before any training to catch integration issues early.
"""
import sumo_rl
import yaml


def run_smoke_test(n_steps: int = 200) -> None:
    with open("config/env_config.yaml") as f:
        env_cfg = yaml.safe_load(f)

    env = sumo_rl.SumoEnvironment(
        net_file=env_cfg["net_file"],
        route_file=env_cfg["route_file_plain"],
        use_gui=False,
        num_seconds=env_cfg["num_seconds"],
        min_green=env_cfg["min_green"],
        delta_time=env_cfg["delta_time"],
        yellow_time=env_cfg["yellow_time"],
        single_agent=True,
    )

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
            print(f"  Episode ended at step {step}, reset done")

    env.close()
    print("Smoke test passed.")


if __name__ == "__main__":
    run_smoke_test()