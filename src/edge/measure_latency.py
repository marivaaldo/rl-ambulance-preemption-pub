"""
Measures PPO inference latency with and without resource constraints.

Two execution modes are supported:

  A) cgroup mode (x86 dev machine with systemd):
     Both runs MUST use systemd-run --scope so that cgroup isolation is
     identical and does not confound results. The only variable that differs
     is whether resource quotas (CPUQuota, MemoryMax) are applied.

     Run TWICE:
       1. Unconstrained (inside cgroup, no quota):
          systemd-run --scope \
            python src/edge/measure_latency.py --device cpu

       2. Constrained (inside cgroup, edge quotas):
          systemd-run --scope -p CPUQuota=100% -p MemoryMax=768M \
            python src/edge/measure_latency.py --constrained --device cpu

  B) Real-hardware mode (Raspberry Pi 3B+ or similar edge device):
     Physical constraints replace cgroup limits. Pass --no-cgroups to skip
     all cgroup validation. Both runs go directly via Python:

       python src/edge/measure_latency.py --no-cgroups --device cpu
       python src/edge/measure_latency.py --no-cgroups --constrained --device cpu

     Hardware metadata (CPU model, available RAM) is captured automatically
     and saved to the output JSON for article reproducibility.
"""
import argparse
import time
import json
import os
import platform
import subprocess
import sys

# Ensure the repo root is on sys.path so `src.*` imports resolve regardless
# of how the script is invoked (python src/edge/measure_latency.py or
# python -m src.edge.measure_latency).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import yaml
import mlflow
from stable_baselines3 import PPO
import sumo_rl

from src.environment.sumo_env import AmbulancePriorityEnv


def _get_cgroup_cpu_quota() -> tuple[int | None, int | None]:
    """Returns (quota_us, period_us) from cgroup v2 cpu.max, or (None, None) if unlimited."""
    try:
        cgroup_rel = ""
        with open("/proc/self/cgroup") as f:
            for line in f:
                parts = line.strip().split(":", 2)
                if parts[0] == "0":  # cgroup v2 unified hierarchy
                    cgroup_rel = parts[2].lstrip("/")
                    break

        # Walk up from the process's own scope to parents until a quota is found.
        candidates = []
        parts = cgroup_rel.split("/") if cgroup_rel else []
        for i in range(len(parts), -1, -1):
            sub = "/".join(parts[:i])
            cpu_max_path = os.path.join("/sys/fs/cgroup", sub, "cpu.max") if sub else "/sys/fs/cgroup/cpu.max"
            candidates.append(cpu_max_path)

        for cpu_max_path in candidates:
            if not os.path.exists(cpu_max_path):
                continue
            content = open(cpu_max_path).read().strip()
            quota_str, period_str = content.split()
            if quota_str == "max":
                continue  # No limit at this level; check parent
            return int(quota_str), int(period_str)
    except Exception:
        pass
    return None, None


def _is_inside_systemd_scope() -> bool:
    """Returns True if launched via 'systemd-run --scope' (detects '.scope' in cgroup path)."""
    try:
        with open("/proc/self/cgroup") as f:
            for line in f:
                if ".scope" in line:
                    return True
    except Exception:
        pass
    return False


def _warn_if_not_in_scope(constrained: bool) -> None:
    if _is_inside_systemd_scope():
        return
    label = "constrained" if constrained else "unconstrained"
    run_cmd = (
        "systemd-run --scope -p CPUQuota=100% -p MemoryMax=768M \\\n"
        "    python src/edge/measure_latency.py --constrained --device cpu"
        if constrained else
        "systemd-run --scope \\\n"
        "    python src/edge/measure_latency.py --device cpu"
    )
    print(
        f"\n[WARN] Run '{label}' não está dentro de um scope systemd.\n"
        "       Ambos os runs devem usar 'systemd-run --scope' para que o isolamento\n"
        "       de cgroup não seja uma variável confundidora na comparação.\n"
        f"\n       Comando recomendado:\n\n    {run_cmd}\n"
        "\n       Para hardware real sem cgroups, passe --no-cgroups.\n",
        file=sys.stderr,
    )


def _validate_cgroup_constraints(cfg: dict) -> None:
    """Aborts if --constrained was passed but no CPU quota cgroup limit is active."""
    quota_us, period_us = _get_cgroup_cpu_quota()

    if quota_us is None:
        print(
            "\n[ERROR] --constrained flag detectado, mas este processo NÃO está rodando\n"
            "        sob restrição de CPU via cgroup.\n"
            "\nOpções:\n"
            "  1. cgroup (máquina de desenvolvimento):\n"
            "     systemd-run --scope -p CPUQuota=100% -p MemoryMax=768M \\\n"
            "         python src/edge/measure_latency.py --constrained --device cpu\n"
            "\n  2. Hardware real (Raspberry Pi ou similar):\n"
            "     python src/edge/measure_latency.py --no-cgroups --constrained --device cpu\n"
            "\nAbortando. Use --no-cgroups para hardware real ou systemd-run para cgroup.",
            file=sys.stderr,
        )
        sys.exit(1)

    effective_pct = 100.0 * quota_us / period_us
    expected_pct = float(cfg.get("cpu_quota", "100%").rstrip("%"))
    print(f"[OK] cgroup CPU quota ativa: {effective_pct:.1f}% (esperado ≤{expected_pct:.0f}%)")


def _collect_hardware_info() -> dict:
    """Returns platform metadata for reproducibility in real-hardware runs."""
    info: dict = {
        "node": platform.node(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("Model name") or line.startswith("model name") or line.startswith("Model"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["total_ram_mb"] = round(int(line.split()[1]) / 1024, 1)
                elif line.startswith("MemAvailable:"):
                    info["available_ram_mb"] = round(int(line.split()[1]) / 1024, 1)
    except Exception:
        pass

    return info


def measure_latency(model_path: str, n_trials: int, constrained: bool, device: str = "cpu", no_cgroups: bool = False) -> dict:
    env_cfg = yaml.safe_load(open("config/env_config.yaml"))
    exp_cfg = yaml.safe_load(open("config/experiment_config.yaml"))

    # PPO.load() appends ".zip" automatically — strip to avoid double extension.
    model_path_stripped = model_path.removesuffix(".zip")
    model = PPO.load(model_path_stripped, device=device)

    env = AmbulancePriorityEnv(
        alpha=exp_cfg["alpha"],
        beta=exp_cfg["beta"],
        net_file=env_cfg["net_file"],
        route_file=env_cfg["route_file_plain"],
        use_gui=False,
        num_seconds=env_cfg["num_seconds"],
        min_green=env_cfg["min_green"],
        delta_time=env_cfg["delta_time"],
        yellow_time=env_cfg["yellow_time"],
        single_agent=True,
    )

    obs, _ = env.reset()
    torch_device = torch.device(device)
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(torch_device)

    use_cuda = torch_device.type == "cuda" and torch.cuda.is_available()

    # Warmup: stabilize caches and clock before measuring.
    for _ in range(10):
        with torch.no_grad():
            model.policy(obs_tensor)
        if use_cuda:
            torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(n_trials):
        if use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            model.policy(obs_tensor)

        if use_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000.0)

    env.close()

    results: dict = {
        "constrained": constrained,
        "no_cgroups": no_cgroups,
        "device": device,
        "n_trials": n_trials,
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "std_latency_ms": float(np.std(latencies_ms)),
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
        "raw_latencies_ms": latencies_ms,
    }

    if no_cgroups:
        results["hardware_info"] = _collect_hardware_info()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Measure PPO inference latency.\n\n"
            "cgroup mode (dev machine):\n"
            "  Unconstrained: systemd-run --scope "
            "python src/edge/measure_latency.py --device cpu\n"
            "  Constrained:   systemd-run --scope -p CPUQuota=100%% -p MemoryMax=768M "
            "python src/edge/measure_latency.py --constrained --device cpu\n\n"
            "Real-hardware mode (Raspberry Pi 3B+ or similar):\n"
            "  python src/edge/measure_latency.py --no-cgroups --device cpu\n"
            "  python src/edge/measure_latency.py --no-cgroups --constrained --device cpu"
        )
    )
    parser.add_argument(
        "--constrained", action="store_true",
        help=(
            "Mark this run as 'constrained' (affects output filename and MLflow run name). "
            "In cgroup mode, also validates that a CPU quota is active. "
            "In --no-cgroups mode, physical hardware limits are assumed."
        ),
    )
    parser.add_argument(
        "--no-cgroups", action="store_true", dest="no_cgroups",
        help=(
            "Real-hardware mode: skip all cgroup/systemd-run validation. "
            "Use on Raspberry Pi or any edge device where physical constraints apply. "
            "Hardware metadata (CPU, RAM) is captured and saved to the output JSON."
        ),
    )
    parser.add_argument("--model", default="results/models/ppo_priority_plain_seed42.zip")
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help=(
            "Device for inference (default: cpu). Edge devices have no GPU; "
            "MlpPolicy is also faster on CPU than GPU due to transfer overhead. "
            "Use 'cuda' only to benchmark GPU inference explicitly."
        ),
    )
    args = parser.parse_args()

    exp_cfg = yaml.safe_load(open("config/experiment_config.yaml"))

    if args.no_cgroups:
        print("[INFO] Modo hardware real (--no-cgroups): validações de cgroup desativadas.")
    else:
        # Warn if not inside a systemd scope (applies to both runs in cgroup mode)
        _warn_if_not_in_scope(args.constrained)
        if args.constrained:
            _validate_cgroup_constraints(exp_cfg)

    print(f"[INFO] Device de inferência: {args.device}")

    results = measure_latency(
        model_path=args.model,
        n_trials=exp_cfg["edge_n_trials"],
        constrained=args.constrained,
        device=args.device,
        no_cgroups=args.no_cgroups,
    )

    label = "constrained" if args.constrained else "unconstrained"
    print(f"\n--- Latency ({label}) ---")
    print(f"Mean: {results['mean_latency_ms']:.3f} ms")
    print(f"P95:  {results['p95_latency_ms']:.3f} ms")
    print(f"P99:  {results['p99_latency_ms']:.3f} ms")

    os.makedirs("results/edge", exist_ok=True)
    with open(f"results/edge/latency_{label}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Log to MLflow
    mlflow.set_experiment(exp_cfg["mlflow_experiment_name"])
    with mlflow.start_run(run_name=f"edge_{label}"):
        mlflow.log_param("device", args.device)
        mlflow.log_metrics({
            "mean_latency_ms": results["mean_latency_ms"],
            "p99_latency_ms": results["p99_latency_ms"],
        })