"""
Evaluation script for all three configurations.
Runs multiple episodes and logs metrics to MLflow.

Usage:
  python src/evaluation/evaluate.py --config fixed_time
  python src/evaluation/evaluate.py --config ppo_no_priority --model results/models/ppo_no_priority_seed42
  python src/evaluation/evaluate.py --config ppo_priority --model results/models/ppo_priority_seed42
"""
import argparse
import json
import math
import os
import yaml
import mlflow
import numpy as np
import traci
import sumo_rl
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.evaluation.metrics import EpisodeMetrics
from src.environment.sumo_env import AmbulancePriorityEnv
from src.utils.seeding import set_global_seed
from src.evaluation.stats import run_pairwise_stats, format_stats_table


def load_configs():
    with open("config/env_config.yaml") as f:
        env_cfg = yaml.safe_load(f)
    with open("config/experiment_config.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    with open("config/ppo_config.yaml") as f:
        ppo_cfg = yaml.safe_load(f)
    return env_cfg, exp_cfg, ppo_cfg


def evaluate_fixed_time(env_cfg: dict, exp_cfg: dict, n_episodes: int, traffic: str, seed: int | None = None) -> list:
    """Evaluates fixed-time controller (alternating phases matching the static signal program)."""
    env = sumo_rl.SumoEnvironment(
        net_file=env_cfg["net_file"],
        route_file=env_cfg[f"route_file_{traffic}"],
        use_gui=False,
        single_agent=True,
        num_seconds=env_cfg["num_seconds"],
        min_green=env_cfg["min_green"],
        delta_time=env_cfg["delta_time"],
        yellow_time=env_cfg["yellow_time"],
    )

    _CYCLE_STEPS = math.ceil(env_cfg["green_time"] / env_cfg["delta_time"])

    results = []
    for ep in range(n_episodes):
        ep_seed = (seed + ep) if seed is not None else None
        obs, _ = env.reset(seed=ep_seed)
        metrics = EpisodeMetrics(num_seconds=env_cfg["num_seconds"])
        terminated, truncated = False, False

        _phase = 0
        _phase_steps = 0

        while not (terminated or truncated):
            if _phase_steps >= _CYCLE_STEPS:
                _phase = 1 - _phase
                _phase_steps = 0
            action = _phase
            _phase_steps += 1

            metrics.record_action(action)
            obs, reward, terminated, truncated, info = env.step(action)
            metrics.update_step(traci.simulation.getTime(), env_cfg["ambulance_type_id"])

        metrics.finalize_vehicle_delays()
        results.append(metrics.summarize())

    env.close()
    return results


def evaluate_ppo(model_path: str, config: str, env_cfg: dict, exp_cfg: dict, ppo_cfg: dict, n_episodes: int, traffic: str, seed: int | None = None) -> list:
    """Evaluates a trained PPO model.

    Loads VecNormalize statistics from {model_path}_vecnormalize.pkl if present
    (produced by train_ppo.py v1+). During evaluation norm_reward is disabled so
    raw episode metrics are comparable to fixed-time baseline numbers. norm_obs was
    False at training time, so observations are always in the original scale.
    """
    model = PPO.load(model_path)

    if config == "ppo_priority":
        env = AmbulancePriorityEnv(
            alpha=exp_cfg["alpha"],
            beta=exp_cfg["beta"],
            gamma=ppo_cfg["gamma"],
            net_file=env_cfg["net_file"],
            route_file=env_cfg[f"route_file_{traffic}"],
            use_gui=False,
            single_agent=True,
            num_seconds=env_cfg["num_seconds"],
            min_green=env_cfg["min_green"],
            delta_time=env_cfg["delta_time"],
            yellow_time=env_cfg["yellow_time"],
        )
    else:
        env = sumo_rl.SumoEnvironment(
            net_file=env_cfg["net_file"],
            route_file=env_cfg[f"route_file_{traffic}"],
            use_gui=False,
            single_agent=True,
            num_seconds=env_cfg["num_seconds"],
            min_green=env_cfg["min_green"],
            delta_time=env_cfg["delta_time"],
            yellow_time=env_cfg["yellow_time"],
        )

    vecnormalize_path = f"{model_path}_vecnormalize.pkl"
    if os.path.exists(vecnormalize_path):
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False  # keep raw rewards comparable to fixed-time baseline
        eval_env = vec_env
        use_vecenv = True
        print(f"Loaded VecNormalize stats from {vecnormalize_path}")
    else:
        eval_env = env
        use_vecenv = False

    results = []
    for ep in range(n_episodes):
        ep_seed = (seed + ep) if seed is not None else None
        if use_vecenv:
            obs_arr = eval_env.reset()
            obs = (obs_arr[0] if isinstance(obs_arr, tuple) else obs_arr)[0]
        else:
            obs, _ = eval_env.reset(seed=ep_seed)
        metrics = EpisodeMetrics(num_seconds=env_cfg["num_seconds"])
        terminated, truncated = False, False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            metrics.record_action(int(action))
            if use_vecenv:
                step_action = np.array([action])
                obs_arr, _, done_arr, info_arr = eval_env.step(step_action)
                obs = obs_arr[0]
                terminated = bool(done_arr[0])
                truncated = False
            else:
                obs, reward, terminated, truncated, info = eval_env.step(action)
            metrics.update_step(traci.simulation.getTime(), env_cfg["ambulance_type_id"])

        metrics.finalize_vehicle_delays()
        results.append(metrics.summarize())

    eval_env.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=["fixed_time", "ppo_no_priority", "ppo_priority"])
    parser.add_argument("--model", type=str, help="Path to .zip model (required for PPO configs)")
    parser.add_argument("--traffic", required=True, choices=["plain", "peak"])
    parser.add_argument("--seed", type=int, help="Seed used during training (for PPO configs; included in output names)")
    args = parser.parse_args()

    env_cfg, exp_cfg, ppo_cfg = load_configs()
    n_episodes = exp_cfg["eval_episodes"]

    if args.seed is not None:
        set_global_seed(args.seed)

    mlflow.set_experiment(exp_cfg["mlflow_experiment_name"])

    seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
    run_name = f"eval_{args.config}_{args.traffic}{seed_suffix}"

    with mlflow.start_run(run_name=run_name):
        if args.config == "fixed_time":
            results = evaluate_fixed_time(env_cfg, exp_cfg, n_episodes, args.traffic, seed=args.seed)
        else:
            assert args.model, "--model required for PPO configs"
            results = evaluate_ppo(args.model, args.config, env_cfg, exp_cfg, ppo_cfg, n_episodes, args.traffic, seed=args.seed)

        # Aggregate results across episodes and log to MLflow
        print(f"\n--- Results: {args.config} ({args.traffic}{seed_suffix}) ---")
        keys = results[0].keys()
        for key in keys:
            values = [r[key] for r in results if r.get(key) is not None]
            if values:
                mlflow.log_metric(f"eval_{key}_mean", float(np.mean(values)))
                mlflow.log_metric(f"eval_{key}_std", float(np.std(values)))
                print(f"  {key}: {np.mean(values):.3f} ± {np.std(values):.3f}")

        os.makedirs("results/eval", exist_ok=True)
        out_path = f"results/eval/{args.config}_{args.traffic}{seed_suffix}_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Raw results saved to {out_path}")

        all_configs = ["fixed_time", "ppo_no_priority", "ppo_priority"]
        results_by_config: dict[str, list[dict]] = {}

        for cfg in all_configs:
            candidates = [
                f"results/eval/{cfg}_{args.traffic}{seed_suffix}_results.json",
                f"results/eval/{cfg}_{args.traffic}_results.json",
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    with open(cand) as fh:
                        results_by_config[cfg] = json.load(fh)
                    break

        if len(results_by_config) == 3:
            print("\n--- Pairwise Statistical Tests (Mann-Whitney + Bootstrap CI 95% + Holm-Bonferroni) ---")
            for metric_key in ["mean_ambulance_transit_s", "mean_vehicle_delay_all_s", "vehicle_throughput_veh_h"]:
                print(f"\nMetric: {metric_key}")
                pstats = run_pairwise_stats(results_by_config, metric=metric_key, seed=args.seed or 0)
                print(format_stats_table(pstats))

                for pair_key, pr in pstats.items():
                    prefix = f"stats_{metric_key}_{pair_key}"
                    for field_name in ("p_raw", "p_corrected", "mean_a", "mean_b", "u_stat"):
                        val = pr.get(field_name)
                        if val is not None and not (isinstance(val, float) and np.isnan(val)):
                            mlflow.log_metric(f"{prefix}_{field_name}", float(val))

            h2_threshold = exp_cfg.get("h2_max_throughput_drop_pct", 5.0)
            tp_priority   = float(np.mean([r["vehicle_throughput_veh_h"] for r in results_by_config.get("ppo_priority", []) if r.get("vehicle_throughput_veh_h") is not None])) if results_by_config.get("ppo_priority") else None
            tp_no_priority = float(np.mean([r["vehicle_throughput_veh_h"] for r in results_by_config.get("ppo_no_priority", []) if r.get("vehicle_throughput_veh_h") is not None])) if results_by_config.get("ppo_no_priority") else None

            if tp_priority is not None and tp_no_priority is not None and tp_no_priority > 0:
                drop_pct = (tp_no_priority - tp_priority) / tp_no_priority * 100
                h2_pass = drop_pct <= h2_threshold
                print(f"\nH2: throughput drop = {drop_pct:.2f}% (threshold ≤{h2_threshold}%) → {'PASS ✓' if h2_pass else 'FAIL ✗'}")
                mlflow.log_metric("h2_throughput_drop_pct", drop_pct)
                mlflow.log_param("h2_pass", str(h2_pass))
        else:
            missing = set(all_configs) - set(results_by_config.keys())
            print(f"\n[stats] Aguardando resultados de: {missing} — testes estatísticos serão executados quando todos os 3 configs estiverem disponíveis.")

        h3_threshold_ms = exp_cfg.get("h3_max_p95_latency_ms", 50.0)
        latency_candidates = [
            "results/edge/latency_constrained.json",
            "results/edge/latency_unconstrained.json",
        ]
        for lat_file in latency_candidates:
            if os.path.exists(lat_file):
                with open(lat_file) as fh:
                    lat_data = json.load(fh)
                p95 = lat_data.get("p95_latency_ms")
                if p95 is not None:
                    constrained_label = "constrained" if lat_data.get("constrained") else "unconstrained"
                    h3_pass = p95 <= h3_threshold_ms
                    print(f"\nH3 ({constrained_label}): P95 latency = {p95:.3f} ms (threshold ≤{h3_threshold_ms} ms) → {'PASS ✓' if h3_pass else 'FAIL ✗'}")
                    mlflow.log_metric(f"h3_p95_latency_ms_{constrained_label}", p95)
                    mlflow.log_param(f"h3_pass_{constrained_label}", str(h3_pass))
