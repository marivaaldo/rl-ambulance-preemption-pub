# =============================================================================
# Raspberry Pi — edge inference via rsync + SSH
# =============================================================================
# H3 measurement strategy:
#   - latency_unconstrained.json → dev machine, systemd-run sem quotas (baseline)
#   - latency_constrained.json   → Raspberry Pi, hardware real (a restrição de borda)
#
# Usage:
#   export RPI_IP=192.168.1.42 RPI_USER=pi RPI_PASSWORD=raspberry
#   make pi-measure-all   # pipeline completo no Pi
#
# Or inline:
#   make pi-measure-all RPI_IP=192.168.1.42 RPI_USER=pi RPI_PASSWORD=raspberry

RPI_IP       ?= 192.168.1.100
RPI_USER     ?= pi
RPI_PASSWORD ?= raspberry
RPI_DIR      ?= ~/rl-ambulance-preemption

# sshpass -e reads password from SSHPASS env var to avoid it appearing in
# the process list (ps aux). We set SSHPASS per-command via env.
_SSH   = SSHPASS=$(RPI_PASSWORD) sshpass -e ssh \
           -o StrictHostKeyChecking=no \
           -o UserKnownHostsFile=/dev/null \
           $(RPI_USER)@$(RPI_IP)
_RSYNC = SSHPASS=$(RPI_PASSWORD) sshpass -e rsync -avz \
           -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

.PHONY: check-sshpass sync fetch-results \
        pi-setup pi-smoke-test pi-measure pi-measure-all \
        measure-unconstrained \
        setup smoke-test calibrate train eval \
        tables figures artifacts \
        mlflow backup visualize

# =============================================================================
# Raspberry Pi targets
# =============================================================================

check-sshpass:
	@command -v sshpass >/dev/null 2>&1 || \
	  { echo "[ERROR] sshpass não encontrado. Instale com: sudo apt install sshpass"; exit 1; }

# Sync repo to Pi (exclude git history, caches, and large result files)
sync: check-sshpass
	$(_SSH) "mkdir -p $(RPI_DIR)"
	$(_RSYNC) \
	  --exclude='.git/' \
	  --exclude='__pycache__/' \
	  --exclude='*.pyc' \
	  --exclude='.mypy_cache/' \
	  --exclude='mlruns/' \
	  --exclude='results/eval/' \
	  --exclude='results/edge/figs/' \
	  . $(RPI_USER)@$(RPI_IP):$(RPI_DIR)/

# Install edge dependencies on Pi (SUMO must already be installed via apt)
pi-setup: check-sshpass sync
	$(_SSH) "cd $(RPI_DIR) && pip install -r requirements-edge.txt"

# Sanity check: verify SUMO + SB3 + model load work on Pi before measuring
pi-smoke-test: check-sshpass
	$(_SSH) "cd $(RPI_DIR) && python src/environment/smoke_test.py"

# Measure inference latency on Pi (hardware real = a restrição; --constrained é o label do artigo)
pi-measure: check-sshpass
	$(_SSH) "cd $(RPI_DIR) && python src/edge/measure_latency.py --no-cgroups --constrained --device cpu"

# Fetch result JSONs back to the local machine
fetch-results: check-sshpass
	mkdir -p results/edge
	$(_RSYNC) $(RPI_USER)@$(RPI_IP):$(RPI_DIR)/results/edge/*.json results/edge/

# Full Pi pipeline: sync → measure → fetch
pi-measure-all: sync pi-measure fetch-results

# =============================================================================
# Local development targets
# =============================================================================

# Create conda environment from environment.yml
setup:
	conda env create -f enviroment.yml
	@echo "[OK] Ambiente criado. Ative com: conda activate rl-ambulance-preemption"

# Sanity check: 200 random steps in the SUMO environment
smoke-test:
	python src/environment/smoke_test.py

# Estimate training wall time before starting a long run
calibrate:
	python src/training/calibrate.py --traffic plain
	python src/training/calibrate.py --traffic peak

# Train all 8 model combinations (2 configs × 2 traffic × 2 seeds, 2 parallel jobs)
train:
	bash scripts/train_all.sh

# Evaluate all 10 configurations (fixed_time + PPO × traffic × seeds)
eval:
	bash scripts/evaluate_all.sh

# Baseline de latência na máquina de desenvolvimento (sem quotas de cgroup)
# Produz latency_unconstrained.json para comparação com o resultado do Pi
measure-unconstrained:
	systemd-run --scope \
	  python src/edge/measure_latency.py --device cpu

# =============================================================================
# Article artifact targets
# =============================================================================

# Generate H1/H2 and H3 tables in Markdown and LaTeX
tables:
	@echo "=== H1/H2 — Markdown ===" && bash scripts/eval_to_md_artigo.sh
	@echo "=== H1/H2 — LaTeX ===" && bash scripts/eval_to_latex_artigo.sh
	@echo "=== H3 — Markdown ===" && bash scripts/latency_to_md_artigo.sh
	@echo "=== H3 — LaTeX ===" && bash scripts/latency_to_latex_artigo.sh

# Generate all comparison figures (H1/H2 bar/box plots + H3 latency histogram/boxplot)
figures:
	python scripts/plot_results.py
	python scripts/plot_generalization.py
	python scripts/plot_latency.py

# Generate all article artifacts (tables + figures) in one shot
artifacts: tables figures

# =============================================================================
# Utilities
# =============================================================================

# Start MLflow UI (open http://localhost:5000)
mlflow:
	mlflow ui --port 5000

# Archive results/ to a timestamped tarball
backup:
	bash scripts/backup_results.sh

# Visualize a trained PPO policy in sumo-gui
# Usage: make visualize CONFIG=ppo_priority TRAFFIC=plain SEED=42
CONFIG  ?= ppo_priority
TRAFFIC ?= plain
SEED    ?= 42
visualize:
	python src/evaluation/visualize.py \
	  --config $(CONFIG) \
	  --traffic $(TRAFFIC) \
	  --model results/models/$(CONFIG)_$(TRAFFIC)_seed$(SEED).zip \
	  --delay 200
