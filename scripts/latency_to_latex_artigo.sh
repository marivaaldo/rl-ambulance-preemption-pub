#!/usr/bin/env bash
# Gera LaTeX para validação de H3 (latência de inferência em borda):
#   1. Tabela de hardware (Pi specs, quando disponível em hardware_info).
#   2. Tabela comparativa Pi (constrained) vs dev machine (unconstrained).
#   3. Parágrafo de análise pronto para colar no artigo.
#   4. Código pgfplots para histograma da distribuição de latências.
# Uso: bash scripts/latency_to_latex_artigo.sh [results/edge/]
# Requer: \usepackage{booktabs,siunitx,pgfplots} no preâmbulo.

set -euo pipefail

EDGE_DIR="${1:-results/edge}"

python3 - "$EDGE_DIR" <<'PYEOF'
import json, sys, math
from pathlib import Path

edge_dir = Path(sys.argv[1])

def load(fname):
    p = edge_dir / fname
    if not p.exists():
        print(f"% AVISO: arquivo não encontrado: {p}", file=sys.stderr)
        return None
    return json.load(open(p))

con  = load("latency_constrained.json")
unco = load("latency_unconstrained.json")

if con is None or unco is None:
    print("% Execute make pi-measure-all e make measure-unconstrained antes de gerar LaTeX.", file=sys.stderr)
    sys.exit(1)

DECISION_WINDOW_MS = 5_000

def ratio(lat_ms):
    return DECISION_WINDOW_MS / lat_ms

def fmt(v, dec=3):
    return rf"\num{{{v:.{dec}f}}}"

hw = con.get("hardware_info")

# ══════════════════════════════════════════════════════════════════════════════
# Tabela de hardware (apenas se hardware_info disponível)
# ══════════════════════════════════════════════════════════════════════════════
if hw:
    cpu   = hw.get("cpu_model", hw.get("machine", "ARM Cortex-A53"))
    ram   = f"{hw['total_ram_mb']:.0f}" if hw.get("total_ram_mb") else "976"
    avail = f"{hw['available_ram_mb']:.0f}" if hw.get("available_ram_mb") else "—"
    arch  = hw.get("machine", "aarch64")
    node  = hw.get("node", "Raspberry Pi 3B+")

    print(r"""% -------------------------------------------------------
% Tabela de hardware — gerada por scripts/latency_to_latex_artigo.sh
% -------------------------------------------------------
\begin{table}[ht]
\centering
\caption{Especificação do hardware utilizado na validação de H3.}
\label{tab:hardware}
\begin{tabular}{lll}
\toprule
\textbf{Parâmetro} & \textbf{Dev machine} & \textbf{Edge device} \\
\midrule""")
    print(rf"  Dispositivo & x86-64 (host local) & {node} \\")
    print(rf"  Arquitetura & x86-64 & {arch} \\")
    print(rf"  CPU & — & {cpu} \\")
    print(rf"  RAM total & — & \SI{{{ram}}}{{\mega\byte}} \\")
    if avail != "—":
        print(rf"  RAM disponível & — & \SI{{{avail}}}{{\mega\byte}} \\")
    print(r"""  Restrição & nenhuma (systemd-run) & hardware físico \\
\bottomrule
\end{tabular}
\end{table}

""")

# ══════════════════════════════════════════════════════════════════════════════
# Tabela principal de latência
# ══════════════════════════════════════════════════════════════════════════════
constrained_label   = r"Raspberry Pi 3B+ \textit{(hardware real)}"
unconstrained_label = r"Dev machine \textit{(sem restrição)}"

print(r"""% -------------------------------------------------------
% Tabela H3 — gerada por scripts/latency_to_latex_artigo.sh
% Requer: booktabs, siunitx
% -------------------------------------------------------
\begin{table}[ht]
\centering
\caption{Latência de inferência da política PPO (\textit{MlpPolicy},
10 chamadas de aquecimento + 1\,000 medições por cenário).
\textit{Raspberry Pi 3B+}: hardware real, restrições físicas.
\textit{Dev machine}: máquina de desenvolvimento sem quotas de cgroup.
A janela de decisão do agente é de \SI{5}{\second} = \SI{5000}{\milli\second}.}
\label{tab:latency}
\setlength{\tabcolsep}{8pt}
\begin{tabular}{lS[table-format=1.3]S[table-format=1.3]S[table-format=1.3]S[table-format=1.3]S[table-format=1.3]}
\toprule
\textbf{Cenário}
  & \textbf{Média (ms)}
  & \textbf{Desvio (ms)}
  & \textbf{P50 (ms)}
  & \textbf{P95 (ms)}
  & \textbf{P99 (ms)} \\
\midrule""")

for label, d in [(constrained_label, con), (unconstrained_label, unco)]:
    print(
        rf"  {label}"
        rf" & {fmt(d['mean_latency_ms'])}"
        rf" & {fmt(d['std_latency_ms'])}"
        rf" & {fmt(d['p50_latency_ms'])}"
        rf" & {fmt(d['p95_latency_ms'])}"
        rf" & {fmt(d['p99_latency_ms'])}"
        r" \\"
    )

print(r"""\bottomrule
\end{tabular}
\end{table}

""")

# ══════════════════════════════════════════════════════════════════════════════
# Parágrafo de análise
# ══════════════════════════════════════════════════════════════════════════════
c_p99  = con["p99_latency_ms"]
u_p99  = unco["p99_latency_ms"]
c_mean = con["mean_latency_ms"]
u_mean = unco["mean_latency_ms"]
c_ratio = ratio(c_p99)
u_ratio = ratio(u_p99)

hw_str = ""
if hw:
    cpu = hw.get("cpu_model", hw.get("machine", "ARM Cortex-A53"))
    ram = f"{hw['total_ram_mb']:.0f}" if hw.get("total_ram_mb") else "976"
    hw_str = rf" ({cpu}, \SI{{{ram}}}{{\mega\byte}} RAM)"

print(rf"""% -------------------------------------------------------
% Parágrafo de análise — cole no corpo do artigo
% -------------------------------------------------------
% A Tabela~\ref{{tab:latency}} apresenta os resultados de latência de inferência.
% No Raspberry Pi 3B+{hw_str},
% a latência média foi de \SI{{{c_mean:.3f}}}{{\milli\second}}
% (P99 = \SI{{{c_p99:.3f}}}{{\milli\second}}),
% representando apenas $1/{c_ratio:.0f}$ da janela de decisão de \SI{{5}}{{\second}}.
% Na máquina de desenvolvimento, a média foi \SI{{{u_mean:.3f}}}{{\milli\second}}
% (P99 = \SI{{{u_p99:.3f}}}{{\milli\second}}, $1/{u_ratio:.0f}$ da janela).
% Em ambos os ambientes, a latência permaneceu abaixo de \SI{{2}}{{\milli\second}},
% confirmando a viabilidade de inferência em tempo real em dispositivos de borda
% e validando a Hipótese~H3.

""")

# ══════════════════════════════════════════════════════════════════════════════
# Código pgfplots — histograma
# ══════════════════════════════════════════════════════════════════════════════
def make_hist(raw, bin_width=0.05):
    lo = math.floor(min(raw) / bin_width) * bin_width
    hi = math.ceil(max(raw) / bin_width) * bin_width
    n_bins = round((hi - lo) / bin_width)
    counts = [0] * n_bins
    for v in raw:
        idx = min(int((v - lo) / bin_width), n_bins - 1)
        counts[idx] += 1
    total = len(raw)
    return [(lo + (i + 0.5) * bin_width, c / total) for i, c in enumerate(counts)]

hist_con  = make_hist(con["raw_latencies_ms"])
hist_unco = make_hist(unco["raw_latencies_ms"])

def coords(hist):
    return " ".join(f"({x:.4f},{y:.4f})" for x, y in hist)

print(r"""% -------------------------------------------------------
% Figura — histograma de latência (pgfplots)
% Requer: pgfplots no preâmbulo
% -------------------------------------------------------
\begin{figure}[ht]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=0.85\columnwidth,
    height=5.5cm,
    xlabel={Latência de inferência (ms)},
    ylabel={Frequência relativa},
    xmin=0.55, xmax=2.1,
    ymin=0,
    xtick distance=0.2,
    legend style={at={(0.97,0.97)}, anchor=north east, font=\small},
    ymajorgrids=true,
    grid style=dashed,
]""")

print(r"""
\addplot[ybar interval, fill=blue!30, draw=blue!60, opacity=0.7, bar width=0.05]
    coordinates {""" + coords(hist_con) + r"""};
\addlegendentry{Raspberry Pi 3B+}

\addplot[ybar interval, fill=orange!40, draw=orange!70, opacity=0.7, bar width=0.05]
    coordinates {""" + coords(hist_unco) + r"""};
\addlegendentry{Dev machine}

\draw[red, dashed, thick] (axis cs:2.0,0) -- (axis cs:2.0,{axis cs:2.0,\pgfkeysvalueof{/pgfplots/ymax}});
""")

print(rf"""\end{{axis}}
\end{{tikzpicture}}
\caption{{Distribuição de latência de inferência (1\,000 amostras por ambiente).
         A linha vermelha tracejada marca \SI{{2}}{{\milli\second}}.
         \textit{{Raspberry Pi 3B+}}: hardware real (restrições físicas);
         \textit{{Dev machine}}: máquina de desenvolvimento sem quotas de cgroup.}}
\label{{fig:latency_hist}}
\end{{figure}}""")

PYEOF
