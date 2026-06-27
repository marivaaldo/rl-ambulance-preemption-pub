"""
Visual demo: runs a single episode in sumo-gui with the trained PPO model.

Usage:
  # PPO com prioridade (modelo treinado)
  python src/evaluation/visualize.py --config ppo_priority --model results/models/ppo_priority_42.zip

  # PPO sem prioridade
  python src/evaluation/visualize.py --config ppo_no_priority --model results/models/ppo_no_priority_42.zip

  # Fixed-time (baseline)
  python src/evaluation/visualize.py --config fixed_time
"""
import argparse
import time
import yaml
import traci
import sumo_rl
from stable_baselines3 import PPO

from src.environment.sumo_env import AmbulancePriorityEnv


def load_configs():
    with open("config/env_config.yaml") as f:
        env_cfg = yaml.safe_load(f)
    with open("config/experiment_config.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    with open("config/ppo_config.yaml") as f:
        ppo_cfg = yaml.safe_load(f)
    return env_cfg, exp_cfg, ppo_cfg


def make_env(config: str, env_cfg: dict, exp_cfg: dict, ppo_cfg: dict, traffic: str = "plain"):
    common = dict(
        net_file=env_cfg["net_file"],
        route_file=env_cfg[f"route_file_{traffic}"],
        use_gui=True,
        single_agent=True,
        num_seconds=env_cfg["num_seconds"],
        min_green=env_cfg["min_green"],
        delta_time=env_cfg["delta_time"],
        yellow_time=env_cfg["yellow_time"],
        sumo_warnings=False,
    )
    if config == "ppo_priority":
        return AmbulancePriorityEnv(
            alpha=exp_cfg["alpha"],
            beta=exp_cfg["beta"],
            gamma=ppo_cfg["gamma"],
            **common,
        )
    return sumo_rl.SumoEnvironment(**common)


def run_episode(env, model=None, ambulance_type_id: str = "", delay: int = 0):
    obs, _ = env.reset()

    try:
        traci.gui.loadView("sumo/viewsettings.xml", "View #0")
        traci.gui.setZoom("View #0", 800)
    except Exception:
        pass

    delay_s = delay / 1000.0

    terminated, truncated = False, False
    step = 0
    while not (terminated or truncated):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = 0

        obs, reward, terminated, truncated, info = env.step(action)
        if delay_s > 0:
            time.sleep(delay_s)

        if ambulance_type_id:
            vehs = traci.vehicle.getIDList()
            amb = [v for v in vehs if traci.vehicle.getTypeID(v) == ambulance_type_id]
            if amb:
                dist = traci.vehicle.getLanePosition(amb[0])
                print(f"[step {step:4d}] ambulance {amb[0]} | pos={dist:.1f}m | reward={reward:.3f}")

        step += 1

    env.close()
    print(f"\nEpisódio concluído em {step} steps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=["fixed_time", "ppo_no_priority", "ppo_priority"])
    parser.add_argument("--model", type=str, help="Caminho para o .zip do modelo (obrigatório para configs PPO)")
    parser.add_argument("--traffic", default="plain", choices=["plain", "peak"])
    parser.add_argument("--delay", type=int, default=0, help="Delay entre steps no SUMO-GUI em ms (padrão: 0)")
    args = parser.parse_args()

    env_cfg, exp_cfg, ppo_cfg = load_configs()

    model = None
    if args.config != "fixed_time":
        assert args.model, "--model é obrigatório para configs PPO"
        model = PPO.load(args.model)

    env = make_env(args.config, env_cfg, exp_cfg, ppo_cfg, traffic=args.traffic)
    run_episode(env, model=model, ambulance_type_id=env_cfg.get("ambulance_type_id", ""), delay=args.delay)