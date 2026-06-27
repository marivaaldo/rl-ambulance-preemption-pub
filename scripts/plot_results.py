"""Gera gráficos comparativos das três configurações (Fixed-Time, PPO s/ Prior., PPO c/ Prior.)
em ambos os cenários (plain, peak). Lê os JSONs em results/eval/ e salva PNGs em results/eval/figs/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = Path(__file__).resolve().parents[1] / "results" / "eval"
OUT_DIR = EVAL_DIR / "figs"
OUT_DIR.mkdir(exist_ok=True)

CONFIGS = {
    "Fixed-Time": {
        "plain": ["fixed_time_plain_results.json"],
        "peak": ["fixed_time_peak_results.json"],
    },
    "PPO s/ Prior.": {
        "plain": ["ppo_no_priority_plain_seed42_results.json", "ppo_no_priority_plain_seed7_results.json"],
        "peak": ["ppo_no_priority_peak_seed42_results.json", "ppo_no_priority_peak_seed7_results.json"],
    },
    "PPO c/ Prior.": {
        "plain": ["ppo_priority_plain_seed42_results.json", "ppo_priority_plain_seed7_results.json"],
        "peak": ["ppo_priority_peak_seed42_results.json", "ppo_priority_peak_seed7_results.json"],
    },
}

COLORS = {"Fixed-Time": "#888888", "PPO s/ Prior.": "#d95f02", "PPO c/ Prior.": "#1b9e77"}


def _extract(payload) -> dict[str, list[float]]:
    """Retorna listas de métricas por episódio."""
    episodes = payload if isinstance(payload, list) else payload.get("episodes", [])
    out = {"amb_transit": [], "queue": [], "throughput": []}
    for ep in episodes:
        if "mean_ambulance_transit_s" in ep:
            out["amb_transit"].append(float(ep["mean_ambulance_transit_s"]))
        if "mean_queue_length" in ep:
            out["queue"].append(float(ep["mean_queue_length"]))
        if "vehicle_throughput" in ep:
            out["throughput"].append(float(ep["vehicle_throughput"]))
    return out


def load_metrics() -> dict[str, dict[str, dict[str, list[float]]]]:
    data: dict = {}
    for cfg, scenarios in CONFIGS.items():
        data[cfg] = {}
        for scen, files in scenarios.items():
            merged = {"amb_transit": [], "queue": [], "throughput": []}
            for fname in files:
                path = EVAL_DIR / fname
                if not path.exists():
                    continue
                with path.open() as f:
                    payload = json.load(f)
                vals = _extract(payload)
                for k in merged:
                    merged[k].extend(vals[k])
                # fallback: usar agregados se não houver per-episode
                if not any(vals.values()):
                    agg = payload.get("aggregated", payload)
                    for k_local, k_agg in [
                        ("amb_transit", "ambulance_transit_time_mean"),
                        ("queue", "avg_queue_length_mean"),
                        ("throughput", "throughput_mean"),
                    ]:
                        if k_agg in agg:
                            merged[k_local].append(float(agg[k_agg]))
            data[cfg][scen] = merged
    return data


def _bar_with_err(ax, labels, means, stds, colors, ylabel, title):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=colors, capsize=4, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + (s if s else 0) + max(means) * 0.01, f"{m:.1f}", ha="center", fontsize=9)


def plot_metric(data, metric_key, ylabel, fname, title_metric):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, scen in zip(axes, ["plain", "peak"]):
        labels, means, stds, colors = [], [], [], []
        for cfg in CONFIGS:
            vals = data[cfg][scen][metric_key]
            if not vals:
                continue
            labels.append(cfg)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
            colors.append(COLORS[cfg])
        _bar_with_err(ax, labels, means, stds, colors, ylabel, f"{title_metric} — {scen}")
    fig.suptitle(title_metric, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_grouped_overview(data):
    """Painel 3x1: ambulância, fila, throughput lado a lado com plain/peak agrupados."""
    metrics = [
        ("amb_transit", "Trânsito ambulância (s)", "menor é melhor"),
        ("queue", "Comp. fila médio", "menor é melhor"),
        ("throughput", "Throughput (veículos)", "maior é melhor"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    scenarios = ["plain", "peak"]
    x = np.arange(len(CONFIGS))
    width = 0.38
    for ax, (key, ylabel, hint) in zip(axes, metrics):
        for i, scen in enumerate(scenarios):
            means = [float(np.mean(data[cfg][scen][key])) if data[cfg][scen][key] else 0 for cfg in CONFIGS]
            stds = [float(np.std(data[cfg][scen][key])) if data[cfg][scen][key] else 0 for cfg in CONFIGS]
            offset = -width / 2 if i == 0 else width / 2
            bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=scen,
                          color=["#888", "#d95f02", "#1b9e77"], alpha=0.7 if scen == "plain" else 1.0,
                          edgecolor="black", linewidth=0.5)
            for b, m in zip(bars, means):
                if m > 0:
                    ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(list(CONFIGS.keys()), rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}\n({hint})", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="cenário")
    fig.suptitle("Comparativo geral — Fixed-Time vs PPO s/ Prior. vs PPO c/ Prior.",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "overview_comparativo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    data = load_metrics()
    plot_metric(data, "amb_transit", "Trânsito da ambulância (s)",
                "trans_ambulancia.png", "Trânsito da ambulância")
    plot_metric(data, "queue", "Comprimento médio da fila",
                "fila.png", "Comprimento de fila")
    plot_metric(data, "throughput", "Throughput (veículos)",
                "throughput.png", "Throughput")
    plot_grouped_overview(data)


if __name__ == "__main__":
    main()
