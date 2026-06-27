#!/usr/bin/env bash
# Gera Markdown estruturado para o artigo:
#   1. Tabela principal com seeds agregados (20 eps/config), agrupada por cenário.
#   2. Tabela de deltas relativos (%) para evidenciar H1/H2.
# Uso: bash scripts/eval_to_md_artigo.sh [results/eval/]

set -euo pipefail

EVAL_DIR="${1:-results/eval}"

python3 - "$EVAL_DIR" <<'PYEOF'
import json, sys, math
from pathlib import Path

eval_dir = Path(sys.argv[1])

# ── utilitários ────────────────────────────────────────────────────────────────
def load(files):
    """Carrega e concatena episódios de uma lista de arquivos."""
    eps = []
    for fname in files:
        p = eval_dir / fname
        if p.exists():
            eps.extend(json.load(open(p)))
    return eps

def agg(episodes, key):
    vals = [e[key] for e in episodes if e.get(key) is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
    return m, s

def fmt(m, s, dec=1):
    if m is None:
        return "—"
    return f"{m:.{dec}f} ± {s:.{dec}f}"

def bold(text, condition):
    return f"**{text}**" if condition else text

# ── definição dos grupos ───────────────────────────────────────────────────────
GROUPS = {
    "plain": {
        "Fixed-Time":   ["fixed_time_plain_results.json"],
        "PPO s/ Prior.": ["ppo_no_priority_plain_seed42_results.json",
                          "ppo_no_priority_plain_seed7_results.json"],
        "PPO c/ Prior.": ["ppo_priority_plain_seed42_results.json",
                          "ppo_priority_plain_seed7_results.json"],
    },
    "peak": {
        "Fixed-Time":   ["fixed_time_peak_results.json"],
        "PPO s/ Prior.": ["ppo_no_priority_peak_seed42_results.json",
                          "ppo_no_priority_peak_seed7_results.json"],
        "PPO c/ Prior.": ["ppo_priority_peak_seed42_results.json",
                          "ppo_priority_peak_seed7_results.json"],
    },
}

CONFIGS = ["Fixed-Time", "PPO s/ Prior.", "PPO c/ Prior."]

# ── carregar todos os dados ────────────────────────────────────────────────────
data = {}   # data[scen][cfg] = {m_transit, s_transit, m_queue, s_queue, m_thr, s_thr}
for scen, cfgs in GROUPS.items():
    data[scen] = {}
    for cfg, files in cfgs.items():
        eps = load(files)
        m_t, s_t = agg(eps, "mean_ambulance_transit_s")
        m_q, s_q = agg(eps, "mean_queue_length")
        m_p, s_p = agg(eps, "vehicle_throughput")
        data[scen][cfg] = dict(m_t=m_t, s_t=s_t, m_q=m_q, s_q=s_q, m_p=m_p, s_p=s_p)

# ── Tabela 1: resultados principais ───────────────────────────────────────────
print("## Tabela 1 — Resultados por configuração e cenário\n")
print("> Valores: média ± desvio padrão entre episódios (plain: 10 ep; peak: 20 ep com seeds 42 e 7).")
print("> **Negrito** indica melhor valor na coluna dentro do cenário. ↓ menor é melhor · ↑ maior é melhor.\n")

print("| Cenário | Configuração | T_amb (s) ↓ | Fila ↓ | Throughput (veic.) ↑ |")
print("|:-------:|:-------------|:-----------:|:------:|:--------------------:|")

for scen in ["plain", "peak"]:
    # achar mínimos/máximos para negrito
    ts  = {c: data[scen][c]["m_t"] for c in CONFIGS if data[scen][c]["m_t"] is not None}
    qs  = {c: data[scen][c]["m_q"] for c in CONFIGS if data[scen][c]["m_q"] is not None}
    ps  = {c: data[scen][c]["m_p"] for c in CONFIGS if data[scen][c]["m_p"] is not None}
    best_t = min(ts, key=ts.get)
    best_q = min(qs, key=qs.get)
    best_p = max(ps, key=ps.get)

    first = True
    for cfg in CONFIGS:
        d = data[scen][cfg]
        scen_label = f"**{scen}**" if first else ""
        first = False

        t_str = bold(fmt(d["m_t"], d["s_t"]), cfg == best_t)
        q_str = bold(fmt(d["m_q"], d["s_q"], dec=2), cfg == best_q)
        p_str = bold(fmt(d["m_p"], d["s_p"], dec=1), cfg == best_p)
        print(f"| {scen_label} | {cfg} | {t_str} | {q_str} | {p_str} |")

# ── Tabela 2: deltas relativos ─────────────────────────────────────────────────
print()
print("## Tabela 2 — Variação relativa (%) de PPO c/ Prior. em relação às baselines\n")
print("> Sinal: − = redução (bom para T_amb e fila) · + = aumento (bom para throughput).\n")

print("| Comparação | ΔT_amb (%) | ΔFila (%) | ΔThroughput (%) |")
print("|:-----------|:----------:|:---------:|:---------------:|")

def delta(new, ref):
    if new is None or ref is None or ref == 0:
        return "—"
    d = (new - ref) / ref * 100
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.0f}%"

comparisons = [
    ("PPO c/ Prior. vs PPO s/ Prior.", "PPO s/ Prior.", "PPO c/ Prior."),
    ("PPO c/ Prior. vs Fixed-Time",    "Fixed-Time",    "PPO c/ Prior."),
]

for label, ref_cfg, new_cfg in comparisons:
    for scen in ["plain", "peak"]:
        ref = data[scen][ref_cfg]
        new = data[scen][new_cfg]
        dt = delta(new["m_t"], ref["m_t"])
        dq = delta(new["m_q"], ref["m_q"])
        dp = delta(new["m_p"], ref["m_p"])
        print(f"| {label} ({scen}) | {dt} | {dq} | {dp} |")
    print(f"|  |  |  |  |")   # linha em branco entre grupos

print()
print("> Fonte: avaliação determinística com seeds 42 e 7. Episódios por configuração: 10 (Fixed-Time), 20 (PPO).")

PYEOF
