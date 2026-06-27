"""
Training script for PPO agent variants.
Supports three configurations:
  - baseline: fixed-time controller (no training)
  - ppo_no_priority: PPO without ambulance prioritization
  - ppo_priority: PPO with ambulance prioritization (proposed method)

Usage:
  python src/training/train_ppo.py --config ppo_priority --seed 42 --traffic plain
  python src/training/train_ppo.py --config ppo_priority --seed 42 --traffic plain --device cpu
"""
import argparse
import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sumo_rl

from src.environment.sumo_env import AmbulancePriorityEnv
from src.training.callbacks import MLflowCallback
from src.utils.seeding import set_global_seed


def load_configs():
    with open("config/env_config.yaml") as f:
        env_cfg = yaml.safe_load(f)
    with open("config/ppo_config.yaml") as f:
        ppo_cfg = yaml.safe_load(f)
    with open("config/experiment_config.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    return env_cfg, ppo_cfg, exp_cfg


def make_base_env(env_cfg: dict, traffic: str):
    """Creates sumo-rl environment without ambulance priority."""
    return sumo_rl.SumoEnvironment(
        net_file=env_cfg["net_file"],
        route_file=env_cfg[f"route_file_{traffic}"],
        use_gui=False,
        num_seconds=env_cfg["num_seconds"],
        min_green=env_cfg["min_green"],
        delta_time=env_cfg["delta_time"],
        yellow_time=env_cfg["yellow_time"],
        single_agent=True,
    )


def make_priority_env(env_cfg: dict, exp_cfg: dict, ppo_cfg: dict, traffic: str):
    """Creates ambulance priority environment."""
    return AmbulancePriorityEnv(
        alpha=exp_cfg["alpha"],
        beta=exp_cfg["beta"],
        gamma=ppo_cfg["gamma"],
        net_file=env_cfg["net_file"],
        route_file=env_cfg[f"route_file_{traffic}"],
        use_gui=False,
        num_seconds=env_cfg["num_seconds"],
        min_green=env_cfg["min_green"],
        delta_time=env_cfg["delta_time"],
        yellow_time=env_cfg["yellow_time"],
        single_agent=True,
    )


def train(config: str, seed: int, traffic: str, device: str = "cpu") -> str:
    """
    Trains PPO agent and returns the path to saved model.
    config: 'ppo_no_priority' or 'ppo_priority'
    traffic: 'plain' or 'peak'
    device: 'cpu' or 'cuda' (cpu recommended for MlpPolicy)
    """
    set_global_seed(seed)
    env_cfg, ppo_cfg, exp_cfg = load_configs()

    if config == "ppo_no_priority":
        env_fn = lambda: make_base_env(env_cfg, traffic)
    elif config == "ppo_priority":
        env_fn = lambda: make_priority_env(env_cfg, exp_cfg, ppo_cfg, traffic)
    else:
        raise ValueError(f"Unknown config: {config}")

    # norm_obs=False: ambulance features have fixed [0,1] semantics; normalizing them
    # would destroy their meaning. norm_reward=True: stabilizes gradient magnitudes
    # across plain/peak traffic without retuning alpha/beta.
    vec_env = DummyVecEnv([env_fn])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    os.makedirs("results/models", exist_ok=True)
    model_path = f"results/models/{config}_{traffic}_seed{seed}"

    run_name = f"{config}_{traffic}_seed{seed}"
    callback = MLflowCallback(
        experiment_name=exp_cfg["mlflow_experiment_name"],
        run_name=run_name,
        model_path=model_path,
        track_ambulance=(config == "ppo_priority"),
        tags={
            "config": config,
            "traffic": traffic,
            "seed": str(seed),
            "device": device,
        },
        params={
            "total_timesteps": exp_cfg["total_timesteps"],
            "alpha": exp_cfg.get("alpha", "N/A"),
            "beta": exp_cfg.get("beta", "N/A"),
        },
        env_params={
            "num_seconds": env_cfg["num_seconds"],
            "min_green": env_cfg["min_green"],
            "delta_time": env_cfg["delta_time"],
            "yellow_time": env_cfg["yellow_time"],
            "max_ambulance_distance": env_cfg["max_ambulance_distance"],
        },
    )

    policy_kwargs = ppo_cfg.pop("policy_kwargs", {})

    model = PPO(
        policy=ppo_cfg["policy"],
        env=vec_env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1,
        device=device,
    )

    model.learn(
        total_timesteps=exp_cfg["total_timesteps"],
        callback=callback,
    )

    vecnormalize_path = f"{model_path}_vecnormalize.pkl"
    vec_env.save(vecnormalize_path)
    print(f"VecNormalize stats saved to {vecnormalize_path}")

    vec_env.close()
    print(f"Model saved to {model_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=["ppo_no_priority", "ppo_priority"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--traffic", required=True, choices=["plain", "peak"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device for PPO training. CPU is faster for MlpPolicy (default: cpu)")
    args = parser.parse_args()

    train(config=args.config, seed=args.seed, traffic=args.traffic, device=args.device)