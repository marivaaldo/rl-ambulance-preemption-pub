#!/usr/bin/env bash
# Run all PPO training combinations in parallel:
#   configs: ppo_no_priority, ppo_priority
#   traffic: plain, peak
#   seeds:   42, 7
# Total: 8 runs
#
# Usage:
#   ./scripts/train_all.sh               # 2 jobs in parallel (default)
#   PARALLEL=4 ./scripts/train_all.sh    # 4 jobs in parallel
#   DEVICE=cuda ./scripts/train_all.sh   # override device
#
# Summary log: results/logs/train_all.log

set -euo pipefail

export PYTHONPATH="$(pwd)"
DEVICE="${DEVICE:-cpu}"
PARALLEL="${PARALLEL:-2}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-5}"  # minutos entre batimentos
RAM_MIN_MB="${RAM_MIN_MB:-1500}"               # aviso se RAM < este valor (MB)
DISK_MIN_GB="${DISK_MIN_GB:-2}"                # aviso se disco < este valor (GB)
LOG_DIR="results/logs"
SUMMARY_LOG="$LOG_DIR/train_all.log"
ACTIVE_FILE="$LOG_DIR/.active_runs_$$"        # arquivo temporário com runs ativos
mkdir -p "$LOG_DIR"
> "$ACTIVE_FILE"

CONFIGS=(ppo_no_priority ppo_priority)
TRAFFIC=(plain peak)
SEEDS=(42 7)

TOTAL=$(( ${#CONFIGS[@]} * ${#TRAFFIC[@]} * ${#SEEDS[@]} ))
RUN=0
DONE=0
SKIPPED=0
ACTIVE=0
TRAINED=0          # runs treinados com sucesso (para cálculo de ETA)
TOTAL_ELAPSED=0    # soma dos tempos de runs completos (segundos)
declare -A PID_TAG    # pid -> tag
declare -A PID_START  # pid -> epoch de início
FAILED=()
HEARTBEAT_PID=""

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY_LOG"; }

cleanup() {
    [[ -n "$HEARTBEAT_PID" ]] && kill "$HEARTBEAT_PID" 2>/dev/null || true
    rm -f "$ACTIVE_FILE"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Checks iniciais
# ---------------------------------------------------------------------------

check_disk() {
    local avail_gb
    avail_gb=$(df -BG . | awk 'NR==2{gsub(/G/,"",$4); print $4}')
    if (( avail_gb < DISK_MIN_GB )); then
        log "⚠  Disco disponível: ${avail_gb} GB — abaixo do mínimo recomendado (${DISK_MIN_GB} GB)"
    else
        log "   Disco disponível: ${avail_gb} GB ✓"
    fi
}

check_ram_global() {
    local avail_mb avail_gb warn=""
    avail_mb=$(free -m | awk 'NR==2{print $7}')
    avail_gb=$(awk "BEGIN{printf \"%.1f\", $avail_mb/1024}")
    (( avail_mb < RAM_MIN_MB )) && warn=" ⚠ (abaixo do recomendado para $PARALLEL paralelos)"
    log "   RAM disponível:  ${avail_gb} GB${warn}"
}

# ---------------------------------------------------------------------------
# Helpers de status
# ---------------------------------------------------------------------------

show_active() {
    (( ACTIVE == 0 )) && return
    [[ ! -s "$ACTIVE_FILE" ]] && return
    local now htag hstart helapsed hmins info=""
    now=$(date +%s)
    while IFS=' ' read -r htag hstart; do
        helapsed=$(( now - hstart ))
        hmins=$(( helapsed / 60 ))
        info+="${htag}(~${hmins}m) "
    done < "$ACTIVE_FILE"
    log "           Ativos ($ACTIVE): $info"
}

show_eta() {
    local remaining=$(( TOTAL - DONE ))
    (( TRAINED == 0 || remaining == 0 )) && return
    local avg=$(( TOTAL_ELAPSED / TRAINED ))
    local slots=$(( PARALLEL < remaining ? PARALLEL : remaining ))
    local eta_mins=$(( (avg * remaining) / (slots * 60) ))
    log "           ETA restante: ~${eta_mins}m ($remaining runs × ~$((avg/60))m médio ÷ ${slots} paralelos)"
}

ram_inline() {
    local avail_mb avail_gb warn=""
    avail_mb=$(free -m | awk 'NR==2{print $7}')
    avail_gb=$(awk "BEGIN{printf \"%.1f\", $avail_mb/1024}")
    (( avail_mb < RAM_MIN_MB )) && warn=" ⚠"
    echo "RAM: ${avail_gb} GB${warn}"
}

# ---------------------------------------------------------------------------
# Heartbeat: processo background que registra runs ativos a cada N minutos
# ---------------------------------------------------------------------------
(
    trap '' INT TERM
    while true; do
        sleep $(( HEARTBEAT_INTERVAL * 60 ))
        [[ -f "$ACTIVE_FILE" ]] || exit 0
        count=$(wc -l < "$ACTIVE_FILE" 2>/dev/null || echo 0)
        (( count == 0 )) && continue
        now=$(date +%s)
        info=""
        while IFS=' ' read -r htag hstart; do
            helapsed=$(( now - hstart ))
            hmins=$(( helapsed / 60 ))
            info+="${htag}(~${hmins}m) "
        done < "$ACTIVE_FILE"
        echo "[$(date '+%H:%M:%S')] ♥  aguardando ${count} job(s) | ${info}" \
            | tee -a "$SUMMARY_LOG"
    done
) &
HEARTBEAT_PID=$!

# ---------------------------------------------------------------------------
# Espera um job de treino terminar; registra falhas e atualiza métricas.
# ---------------------------------------------------------------------------
reap_one() {
    local finished_pid rc=0
    # Ignora o heartbeat se ele sair inesperadamente
    while true; do
        wait -n -p finished_pid 2>/dev/null && rc=0 || rc=$?
        [[ "$finished_pid" != "${HEARTBEAT_PID:-}" ]] && break
        rc=0
    done

    local tag="${PID_TAG[$finished_pid]:-unknown}"
    local start_ts="${PID_START[$finished_pid]:-0}"
    unset "PID_TAG[$finished_pid]"
    unset "PID_START[$finished_pid]"
    sed -i "/^${tag} /d" "$ACTIVE_FILE" 2>/dev/null || true

    ACTIVE=$(( ACTIVE - 1 ))
    DONE=$(( DONE + 1 ))

    local elapsed=$(( $(date +%s) - start_ts ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    if (( rc == 0 )); then
        TRAINED=$(( TRAINED + 1 ))
        TOTAL_ELAPSED=$(( TOTAL_ELAPSED + elapsed ))
        log "[$DONE/$TOTAL] ✓ OK    $tag  (${mins}m ${secs}s)"
    else
        log "[$DONE/$TOTAL] ✗ FAIL  $tag  (exit $rc, ${mins}m ${secs}s) — ver $LOG_DIR/${tag}.log"
        FAILED+=("$tag")
    fi
    show_active
    show_eta
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
log "====== INÍCIO DO TREINO — $(date) ======"
log "Configurações: ${CONFIGS[*]}"
log "Tráfego:       ${TRAFFIC[*]}"
log "Seeds:         ${SEEDS[*]}"
log "Total de runs: $TOTAL  |  PARALLEL=$PARALLEL  |  DEVICE=$DEVICE"
check_disk
check_ram_global
log ""

# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------
for config in "${CONFIGS[@]}"; do
    for traffic in "${TRAFFIC[@]}"; do
        for seed in "${SEEDS[@]}"; do
            RUN=$(( RUN + 1 ))
            TAG="${config}_${traffic}_seed${seed}"
            LOG="$LOG_DIR/${TAG}.log"

            MODEL_FILE="results/models/${TAG}.zip"
            if [[ -f "$MODEL_FILE" ]]; then
                log "[$RUN/$TOTAL] ! EXISTE  $TAG ($MODEL_FILE)"
                read -r -p "    Sobrescrever? [y/N] " REPLY
                if [[ "${REPLY,,}" != "y" ]]; then
                    log "[$RUN/$TOTAL] ↷ SKIP   $TAG"
                    DONE=$(( DONE + 1 ))
                    SKIPPED=$(( SKIPPED + 1 ))
                    continue
                fi
            fi

            # Aguarda slot livre
            while (( ACTIVE >= PARALLEL )); do
                reap_one
            done

            log "[$RUN/$TOTAL] → START  $TAG  (slot $((ACTIVE + 1))/$PARALLEL) | $(ram_inline) | log: $LOG"

            python src/training/train_ppo.py \
                --config "$config" \
                --traffic "$traffic" \
                --seed "$seed" \
                --device "$DEVICE" \
                > "$LOG" 2>&1 &
            new_pid=$!
            PID_TAG[$new_pid]="$TAG"
            PID_START[$new_pid]=$(date +%s)
            echo "$TAG $(date +%s)" >> "$ACTIVE_FILE"
            ACTIVE=$(( ACTIVE + 1 ))
        done
    done
done

# Drena os jobs restantes
while (( ACTIVE > 0 )); do
    reap_one
done

# ---------------------------------------------------------------------------
# Rodapé
# ---------------------------------------------------------------------------
log ""
log "====== FIM DO TREINO — $(date) ======"
log "Runs concluídos: $DONE / $TOTAL  ($SKIPPED pulados, $(( DONE - SKIPPED )) treinados)"

if (( ${#FAILED[@]} > 0 )); then
    log "FALHAS (${#FAILED[@]}):"
    for t in "${FAILED[@]}"; do log "  - $t"; done
    exit 1
else
    log "Todos os runs concluídos com sucesso."
fi
