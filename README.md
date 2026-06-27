# rl-ambulance-preemption

Replication package for the paper **"Adaptive Traffic Signal Control with Emergency Vehicle Prioritization using PPO under Edge Computational Constraints"** (PPGETI/UFC).

Implements PPO-based adaptive traffic signal control with ambulance prioritization at a single signalized intersection, validated under edge hardware constraints (Raspberry Pi 3B+).

---

## Research hypotheses

| Hypothesis | Claim |
|------------|-------|
| **H1** | PPO with priority reduces mean ambulance transit time vs. PPO without priority |
| **H2** | Aggregate throughput degradation stays below threshold vs. PPO without priority |
| **H3** | Inference latency is feasible under 1 CPU core + 768 MB RAM (Raspberry Pi 3B+) |

Ambulance detection is **oracle-based** (binary signal via TraCI vehicle type ID). Computer vision (YOLOv8) is deferred to future work.

---

## Prerequisites

- Ubuntu 22.04 / 24.04 (or WSL2 with Ubuntu 24.04)
- Miniconda or Anaconda
- Python 3.10 (managed by conda)

### Install SUMO

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

Add to `~/.bashrc`:

```bash
export SUMO_HOME="/usr/share/sumo"
```

### Create conda environment

```bash
conda env create -f enviroment.yml
conda activate rl-ambulance-preemption
```

> Note: the filename `enviroment.yml` is missing one 'n' — use exactly that name.

---

## Project structure

```
├── config/                  # Hyperparameters and experiment settings
│   ├── env_config.yaml      # SUMO paths, ambulance detection distance
│   ├── ppo_config.yaml      # SB3 PPO hyperparameters
│   └── experiment_config.yaml  # Training steps, seeds, reward coefficients
├── sumo/                    # SUMO simulation files
│   ├── intersection.net.xml
│   ├── routes_plain.rou.xml # Plain traffic (~300 veh/h)
│   ├── routes_peak.rou.xml  # Peak traffic (~700 veh/h)
│   └── simulation.*.sumocfg
├── src/
│   ├── environment/         # Custom Gym environment
│   │   ├── sumo_env.py      # AmbulancePriorityEnv
│   │   ├── observation.py   # Adds ambulance_present + ambulance_distance features
│   │   ├── reward.py        # Conditional reward with potential-based shaping
│   │   └── smoke_test.py    # Quick sanity check (200 steps, no training)
│   ├── training/
│   │   ├── train_ppo.py     # Entry point for single training run
│   │   ├── calibrate.py     # Estimate training wall time (steps/sec)
│   │   └── callbacks.py     # MLflow callback for SB3
│   ├── evaluation/
│   │   ├── evaluate.py      # 10-episode deterministic evaluation
│   │   ├── metrics.py       # EpisodeMetrics dataclass (TraCI collection)
│   │   ├── stats.py         # Mann-Whitney U, bootstrap CI, Holm-Bonferroni
│   │   └── visualize.py     # Visual demo in sumo-gui
│   ├── edge/
│   │   └── measure_latency.py  # Inference latency measurement (with/without cgroups)
│   └── utils/
│       └── seeding.py       # set_global_seed(): pins random/numpy/torch
├── scripts/
│   ├── train_all.sh         # Train all 8 model variants (2 configs × 2 traffic × 2 seeds)
│   ├── evaluate_all.sh      # Evaluate all 10 configurations
│   ├── eval_to_md_artigo.sh     # H1/H2 result tables in Markdown
│   ├── eval_to_latex_artigo.sh  # H1/H2 result tables in LaTeX (booktabs)
│   ├── latency_to_md_artigo.sh  # H3 latency report in Markdown
│   ├── latency_to_latex_artigo.sh  # H3 latency table + pgfplots in LaTeX
│   ├── plot_results.py      # H1/H2 comparison figures → results/eval/figs/
│   └── plot_latency.py      # H3 latency histogram + boxplot → results/edge/figs/
├── enviroment.yml           # Conda environment (full reproducibility)
├── requirements-edge.txt    # Minimal deps for Raspberry Pi edge inference
└── Makefile                 # Convenience targets (see make help)
```

All scripts must be run from the **repo root** — imports are relative and there are no `__init__.py` files.

---

## Reproducing the results

### 1. Sanity check

```bash
python src/environment/smoke_test.py
```

### 2. (Optional) Calibrate simulation speed

```bash
python src/training/calibrate.py --traffic plain
```

### 3. Train all models

```bash
bash scripts/train_all.sh
```

This trains 8 variants: `{ppo_priority, ppo_no_priority}` × `{plain, peak}` × `{seed 42, seed 123}`.  
Models are saved to `results/models/`.

Or train a single run:

```bash
python src/training/train_ppo.py \
  --config ppo_priority \
  --traffic plain \
  --seed 42 \
  --device cpu
```

### 4. Evaluate

```bash
bash scripts/evaluate_all.sh
```

Results are saved to `results/eval/` as JSON (one file per configuration).

### 5. Generate tables and figures

```bash
bash scripts/eval_to_md_artigo.sh       # H1/H2 tables in Markdown
bash scripts/eval_to_latex_artigo.sh    # H1/H2 tables in LaTeX
python scripts/plot_results.py          # H1/H2 comparison figures
```

### 6. Edge latency (H3)

**Development machine (baseline):**

```bash
systemd-run --scope python src/edge/measure_latency.py --device cpu
```

**Raspberry Pi 3B+ (hardware):**

```bash
python src/edge/measure_latency.py --no-cgroups --constrained --device cpu
```

```bash
bash scripts/latency_to_md_artigo.sh    # H3 report in Markdown
bash scripts/latency_to_latex_artigo.sh # H3 table in LaTeX
python scripts/plot_latency.py          # H3 distribution figures
```

---

## Experiment configurations

| Config | Description |
|--------|-------------|
| `baseline` / `fixed_time` | Fixed-time control (phase 0 always on) |
| `ppo_no_priority` | Standard PPO, no ambulance awareness |
| `ppo_priority` | PPO with ambulance-aware observation and reward shaping |

---

## Tracking experiments

MLflow is used for all metric logging:

```bash
mlflow ui
```

Open `http://localhost:5000` — experiment name: `rl-ambulance-preemption-artigo`.

---

## Edge deployment (Raspberry Pi)

See `requirements-edge.txt` for minimal dependencies. The `Makefile` has targets to sync, run, and fetch results from the Pi:

```bash
make pi-measure-all RPI_IP=<ip> RPI_USER=<user> RPI_PASSWORD=<password>
```

---

## Reference hardware

- **Training/evaluation**: Intel Core i7-10750H, 16 GB RAM (Ubuntu 24.04 / WSL2)
- **Edge validation**: Raspberry Pi 3B+ (1 CPU core, 768 MB RAM)
