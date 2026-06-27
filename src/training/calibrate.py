"""
Calibration script: measures steps per second to estimate total training time.
Run this before any long training to size n_steps correctly.
"""
import argparse
import time
import yaml
import sumo_rl


def calibrate(n_steps: int = 1000, traffic: str = "plain") -> float:
    """Run n_steps and return steps per second."""
    with open("config/env_config.yaml") as f:
        env_cfg = yaml.safe_load(f)

    env = sumo_rl.SumoEnvironment(
        net_file=env_cfg["net_file"],
        route_file=env_cfg[f"route_file_{traffic}"],
        use_gui=False,
        num_seconds=env_cfg["num_seconds"],
        min_green=env_cfg["min_green"],
        delta_time=env_cfg["delta_time"],
        single_agent=True,
    )

    obs, _ = env.reset()
    start = time.perf_counter()

    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    elapsed = time.perf_counter() - start
    env.close()

    steps_per_second = n_steps / elapsed
    return steps_per_second


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic", default="plain", choices=["plain", "peak"])
    args = parser.parse_args()

    print(f"Running calibration (1000 steps, traffic={args.traffic})...")
    sps = calibrate(1000, traffic=args.traffic)
    print(f"Steps per second: {sps:.1f}")

    for total_steps in [100_000, 200_000, 500_000]:
        estimated_hours = total_steps / sps / 3600
        print(f"  {total_steps:>7,} steps: ~{estimated_hours:.1f}h per run")
