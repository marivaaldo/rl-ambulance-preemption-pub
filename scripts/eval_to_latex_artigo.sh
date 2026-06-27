#!/usr/bin/env bash
# Gera LaTeX estruturado para o artigo:
#   1. Tabela principal com seeds agregados, agrupada por cenário (multirow).
#   2. Tabela de deltas relativos (%) para H1/H2.
# Uso: bash scripts/eval_to_latex_artigo.sh [results/eval/]
# Requer: \usepackage{booktabs,multirow,siunitx} no preâmbulo.

set -euo pipefail

EVAL_DIR="${1:-results/eval}"

python3 - "$EVAL_DIR" <<'PYEOF'
import json, sys, math
from pathlib import Path

eval_dir = Path(sys.argv[1])

# ── utilitários ────────────────────────────────────────────────────────────────
def load(files):
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
        return r"\multicolumn{1}{c}{---}"
    return rf"${m:.{dec}f} \pm {s:.{dec}f}$"

def bfmt(m, s, dec, is_best):
    cell = fmt(m, s, dec)
    if is_best and m is not None:
        inner = rf"{m:.{dec}f} \pm {s:.{dec}f}"
        return rf"$\mathbf{{{inner}}}$"
    return cell

def dpct(new, ref):
    if new is None or ref is None or ref == 0:
        return r"\multicolumn{1}{c}{---}"
    d = (new - ref) / ref * 100
    sign = "+" if d >= 0 else ""
    val = rf"{sign}{d:.0f}\%"
    # colorir: redução boa para T e fila, aumento bom para throughput
    return val

# ── grupos ─────────────────────────────────────────────────────────────────────
GROUPS = {
    "plain": {
        "Fixed-Time":    ["fixed_time_plain_results.json"],
        r"PPO s/ Prior.": ["ppo_no_priority_plain_seed42_results.json",
                           "ppo_no_priority_plain_seed7_results.json"],
        r"PPO c/ Prior.": ["ppo_priority_plain_seed42_results.json",
                           "ppo_priority_plain_seed7_results.json"],
    },
    "peak": {
        "Fixed-Time":    ["fixed_time_peak_results.json"],
        r"PPO s/ Prior.": ["ppo_no_priority_peak_seed42_results.json",
                           "ppo_no_priority_peak_seed7_results.json"],
        r"PPO c/ Prior.": ["ppo_priority_peak_seed42_results.json",
                           "ppo_priority_peak_seed7_results.json"],
    },
}

CONFIGS = ["Fixed-Time", r"PPO s/ Prior.", r"PPO c/ Prior."]

data = {}
for scen, cfgs in GROUPS.items():
    data[scen] = {}
    for cfg, files in cfgs.items():
        eps = load(files)
        m_t, s_t = agg(eps, "mean_ambulance_transit_s")
        m_q, s_q = agg(eps, "mean_queue_length")
        m_p, s_p = agg(eps, "vehicle_throughput")
        data[scen][cfg] = dict(m_t=m_t, s_t=s_t, m_q=m_q, s_q=s_q, m_p=m_p, s_p=s_p)

# ══════════════════════════════════════════════════════════════════════════════
# Tabela 1: resultados principais
# ══════════════════════════════════════════════════════════════════════════════
print(r"""% -------------------------------------------------------
% Tabela 1 — gerada por scripts/eval_to_latex_artigo.sh
% Requer: booktabs, multirow, siunitx
% -------------------------------------------------------
\begin{table}[ht]
\centering
\caption{Resultados de avaliação por configuração e cenário de tráfego.
Valores representam média~$\pm$~desvio padrão entre episódios
(\textit{plain}: 10~ep.; \textit{peak}: 20~ep., \textit{seeds} 42 e 7).
\textbf{Negrito} indica melhor valor na coluna dentro do cenário
($\downarrow$~menor é melhor; $\uparrow$~maior é melhor).}
\label{tab:eval_results}
\setlength{\tabcolsep}{8pt}
\begin{tabular}{llrrr}
\toprule
\textbf{Cenário}
  & \textbf{Configuração}
  & \textbf{\makecell{$T_{\mathrm{amb}}$ (s)\\$\downarrow$}}
  & \textbf{\makecell{Fila média\\$\downarrow$}}
  & \textbf{\makecell{Throughput\\(veic.)\ $\uparrow$}} \\
\midrule""")

SCEN_LABELS = {"plain": r"\textit{plain}", "peak": r"\textit{peak}"}

for scen in ["plain", "peak"]:
    ts = {c: data[scen][c]["m_t"] for c in CONFIGS if data[scen][c]["m_t"] is not None}
    qs = {c: data[scen][c]["m_q"] for c in CONFIGS if data[scen][c]["m_q"] is not None}
    ps = {c: data[scen][c]["m_p"] for c in CONFIGS if data[scen][c]["m_p"] is not None}
    best_t = min(ts, key=ts.get)
    best_q = min(qs, key=qs.get)
    best_p = max(ps, key=ps.get)

    n = len(CONFIGS)
    for i, cfg in enumerate(CONFIGS):
        d = data[scen][cfg]
        scen_cell = rf"\multirow{{{n}}}{{*}}{{{SCEN_LABELS[scen]}}}" if i == 0 else ""
        t_cell = bfmt(d["m_t"], d["s_t"], 1, cfg == best_t)
        q_cell = bfmt(d["m_q"], d["s_q"], 2, cfg == best_q)
        p_cell = bfmt(d["m_p"], d["s_p"], 1, cfg == best_p)
        print(rf"  {scen_cell} & {cfg} & {t_cell} & {q_cell} & {p_cell} \\")

    if scen != "peak":
        print(r"\midrule")

print(r"""\bottomrule
\end{tabular}
\end{table}

""")

# ══════════════════════════════════════════════════════════════════════════════
# Tabela 2: deltas relativos
# ══════════════════════════════════════════════════════════════════════════════
print(r"""% -------------------------------------------------------
% Tabela 2 — variação relativa (%)
% -------------------------------------------------------
\begin{table}[ht]
\centering
\caption{Variação relativa (\%) de \textbf{PPO c/ Prioridade} em relação às configurações de referência.
Sinal negativo em $T_{\mathrm{amb}}$ e Fila indica \emph{melhora}; positivo em Throughput indica \emph{melhora}.}
\label{tab:eval_deltas}
\begin{tabular}{llrrr}
\toprule
\textbf{Referência} & \textbf{Cenário}
  & $\Delta T_{\mathrm{amb}}$ (\%)
  & $\Delta\text{Fila}$ (\%)
  & $\Delta\text{Throughput}$ (\%) \\
\midrule""")

comparisons = [
    (r"PPO s/ Prior.", r"PPO s/ Prior.", r"PPO c/ Prior."),
    (r"Fixed-Time",    r"Fixed-Time",    r"PPO c/ Prior."),
]

for label, ref_cfg, new_cfg in comparisons:
    for j, scen in enumerate(["plain", "peak"]):
        ref = data[scen][ref_cfg]
        new = data[scen][new_cfg]
        dt = dpct(new["m_t"], ref["m_t"])
        dq = dpct(new["m_q"], ref["m_q"])
        dp = dpct(new["m_p"], ref["m_p"])
        lbl_cell = rf"\multirow{{2}}{{*}}{{{label}}}" if j == 0 else ""
        print(rf"  {lbl_cell} & \textit{{{scen}}} & {dt} & {dq} & {dp} \\")
    print(r"\midrule")

print(r"""\bottomrule
\end{tabular}
\end{table}""")

PYEOF
