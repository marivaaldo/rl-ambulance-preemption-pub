"""
Statistical tests for evaluation results.

Computes:
- Mann-Whitney U test between pairs of configurations (non-parametric, no normality assumption).
- 95% confidence intervals via bootstrap (10 000 resamples).
- Holm-Bonferroni correction for familywise error rate across 3 comparisons.

Comparison pairs (k=3):
  A: baseline × ppo_no_priority
  B: baseline × ppo_priority
  C: ppo_no_priority × ppo_priority

Usage:
    from src.evaluation.stats import run_pairwise_stats
    stats = run_pairwise_stats(results_by_config, metric="mean_ambulance_transit_s")
    # stats is a dict with keys like "baseline_vs_ppo_priority" → {u_stat, p_raw, p_corrected, ...}
"""
from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


def bootstrap_mean_ci(
    data: list[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Returns (lower, upper) bootstrap CI for the mean."""
    if len(data) == 0:
        return (float("nan"), float("nan"))
    arr = np.asarray(data, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_resamples)
    ])
    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Step-down Holm-Bonferroni correction. Returns adjusted p-values in input order."""
    k = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [None] * k
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = min(p * (k - rank), 1.0)
        running_max = max(running_max, adjusted)
        corrected[orig_idx] = running_max
    return corrected


PAIRS: list[tuple[str, str]] = [
    ("baseline",        "ppo_no_priority"),
    ("baseline",        "ppo_priority"),
    ("ppo_no_priority", "ppo_priority"),
]


def run_pairwise_stats(
    results_by_config: dict[str, list[dict]],
    metric: str,
    bootstrap_n: int = 10_000,
    seed: int = 0,
) -> dict[str, dict]:
    """Mann-Whitney U + bootstrap CI for all 3 pairwise config comparisons, with Holm-Bonferroni correction."""
    rng = np.random.default_rng(seed)

    def _extract(config: str) -> list[float]:
        episodes = results_by_config.get(config, [])
        return [float(r[metric]) for r in episodes if r.get(metric) is not None]

    pair_results: dict[str, dict] = {}
    raw_pvalues: list[float] = []

    for cfg_a, cfg_b in PAIRS:
        key = f"{cfg_a}_vs_{cfg_b}"
        samples_a = _extract(cfg_a)
        samples_b = _extract(cfg_b)

        result: dict = {
            "config_a": cfg_a,
            "config_b": cfg_b,
            "metric": metric,
            "n_a": len(samples_a),
            "n_b": len(samples_b),
            "mean_a": float(np.mean(samples_a)) if samples_a else float("nan"),
            "mean_b": float(np.mean(samples_b)) if samples_b else float("nan"),
            "ci95_a": bootstrap_mean_ci(samples_a, n_resamples=bootstrap_n, rng=rng),
            "ci95_b": bootstrap_mean_ci(samples_b, n_resamples=bootstrap_n, rng=rng),
        }

        if len(samples_a) >= 2 and len(samples_b) >= 2:
            u_stat, p_raw = scipy_stats.mannwhitneyu(samples_a, samples_b, alternative="two-sided")
            result["u_stat"] = float(u_stat)
            result["p_raw"] = float(p_raw)
        else:
            result["u_stat"] = float("nan")
            result["p_raw"] = float("nan")

        raw_pvalues.append(result.get("p_raw", float("nan")))
        pair_results[key] = result

    corrected = holm_bonferroni(raw_pvalues)
    for (key, result), p_corr in zip(pair_results.items(), corrected):
        result["p_corrected"] = p_corr
        result["significant"] = bool(p_corr < 0.05) if not np.isnan(p_corr) else False

    return pair_results


def format_stats_table(stats: dict[str, dict]) -> str:
    """Returns a human-readable table of pairwise comparison results."""
    lines = [
        f"{'Comparison':<35} {'n_a':>4} {'n_b':>4} {'mean_a':>10} {'mean_b':>10} "
        f"{'p_raw':>8} {'p_holm':>8} {'sig':>5}",
        "-" * 90,
    ]
    for key, r in stats.items():
        sig = "✓" if r.get("significant") else " "
        lines.append(
            f"{key:<35} {r['n_a']:>4} {r['n_b']:>4} "
            f"{r['mean_a']:>10.3f} {r['mean_b']:>10.3f} "
            f"{r['p_raw']:>8.4f} {r['p_corrected']:>8.4f} {sig:>5}"
        )
    return "\n".join(lines)
