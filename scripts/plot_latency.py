"""
Gera figura de distribuição de latência de inferência (H3).

Saída: results/edge/figs/latency_distribution.pdf  (e .png)

Uso:
    python scripts/plot_latency.py [--edge-dir results/edge] [--out-dir results/edge/figs]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── paleta consistente com plot_results.py ─────────────────────────────────────
COLOR_CON  = "#1b9e77"   # verde — constrained (o método "exigente")
COLOR_UNCO = "#d95f02"   # laranja — unconstrained

DECISION_WINDOW_MS = 5_000   # delta_time = 5 s


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def percentile_line(ax, value: float, color: str, label: str, ls: str = "--") -> None:
    ax.axvline(value, color=color, linestyle=ls, linewidth=1.2, label=label, zorder=3)


def _label_constrained(data: dict) -> str:
    hw = data.get("hardware_info")
    if hw:
        cpu = hw.get("cpu_model", hw.get("machine", "ARM"))
        ram = hw.get("total_ram_mb")
        ram_str = f", {ram:.0f} MB RAM" if ram else ""
        node = hw.get("node", "")
        node_str = f"\n({node})" if node else ""
        return f"Raspberry Pi 3B+\n{cpu}{ram_str}{node_str}"
    return "Constrained\n(1 CPU, 768 MB RAM)"


def _label_unconstrained(_data: dict) -> str:
    return "Dev machine\n(sem restrição)"


def plot_latency(edge_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    con  = load(edge_dir / "latency_constrained.json")
    unco = load(edge_dir / "latency_unconstrained.json")

    raw_con  = np.array(con["raw_latencies_ms"])
    raw_unco = np.array(unco["raw_latencies_ms"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    datasets = [
        (axes[0], raw_con,  con,  COLOR_CON,  _label_constrained(con)),
        (axes[1], raw_unco, unco, COLOR_UNCO, _label_unconstrained(unco)),
    ]

    # Determina limites de x comuns para facilitar comparação visual
    x_lo = 0.55
    x_hi = max(raw_unco.max(), raw_con.max()) * 1.05

    for ax, raw, stats, color, title in datasets:
        bins = np.arange(x_lo, x_hi + 0.05, 0.05)
        counts, edges = np.histogram(raw, bins=bins, density=False)
        freq = counts / counts.sum()

        ax.bar(
            edges[:-1], freq,
            width=np.diff(edges),
            align="edge",
            color=color,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
            label="Frequência relativa",
        )

        # Linhas verticais para P50, P95, P99
        p50 = stats["p50_latency_ms"]
        p95 = stats["p95_latency_ms"]
        p99 = stats["p99_latency_ms"]

        ax.axvline(p50, color=color, linestyle=":",  linewidth=1.4,
                   label=f"P50 = {p50:.3f} ms", zorder=4)
        ax.axvline(p95, color=color, linestyle="-.", linewidth=1.4,
                   label=f"P95 = {p95:.3f} ms", zorder=4)
        ax.axvline(p99, color=color, linestyle="--", linewidth=1.8,
                   label=f"P99 = {p99:.3f} ms", zorder=4)

        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel("Latência de inferência (ms)", fontsize=10)
        ax.set_ylabel("Frequência relativa", fontsize=10)
        ax.set_xlim(x_lo, x_hi)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)

        # Anotação: % da janela de decisão
        pct = p99 / DECISION_WINDOW_MS * 100
        ax.annotate(
            f"P99 = {pct:.2f}%\nda janela (5 s)",
            xy=(p99, ax.get_ylim()[1] * 0.6),
            xytext=(p99 + 0.08, ax.get_ylim()[1] * 0.6),
            fontsize=7.5,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
        )

    fig.suptitle(
        "Distribuição de Latência de Inferência PPO — Validação H3",
        fontsize=12, y=1.02,
    )

    for ext in ("pdf", "png"):
        out_path = out_dir / f"latency_distribution.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=180)
        print(f"Salvo: {out_path}")

    plt.close(fig)


# ── figura alternativa: boxplot lado-a-lado ────────────────────────────────────
def plot_boxplot(edge_dir: Path, out_dir: Path) -> None:
    con  = load(edge_dir / "latency_constrained.json")
    unco = load(edge_dir / "latency_unconstrained.json")

    raw_con  = np.array(con["raw_latencies_ms"])
    raw_unco = np.array(unco["raw_latencies_ms"])

    fig, ax = plt.subplots(figsize=(5, 4.5))

    bp = ax.boxplot(
        [raw_con, raw_unco],
        labels=[_label_constrained(con), _label_unconstrained(unco)],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
        widths=0.4,
    )

    bp["boxes"][0].set_facecolor(COLOR_CON  + "99")
    bp["boxes"][1].set_facecolor(COLOR_UNCO + "99")
    bp["boxes"][0].set_edgecolor(COLOR_CON)
    bp["boxes"][1].set_edgecolor(COLOR_UNCO)

    # Linha de referência: limiar de 2 ms (headroom visual)
    ax.axhline(2.0, color="red", linestyle="--", linewidth=1.2,
               label="Limiar 2 ms", zorder=3)

    ax.set_ylabel("Latência de inferência (ms)", fontsize=10)
    ax.set_title("Latência PPO — Boxplot (1 000 amostras)", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="upper left")

    for ext in ("pdf", "png"):
        out_path = out_dir / f"latency_boxplot.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=180)
        print(f"Salvo: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-dir", default="results/edge",
                        help="Diretório com os JSONs de latência (default: results/edge)")
    parser.add_argument("--out-dir",  default="results/edge/figs",
                        help="Diretório de saída das figuras (default: results/edge/figs)")
    args = parser.parse_args()

    edge = Path(args.edge_dir)
    out  = Path(args.out_dir)

    plot_latency(edge, out)
    plot_boxplot(edge, out)
