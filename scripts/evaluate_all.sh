#!/usr/bin/env bash
# Run all evaluation combinations:
#   fixed_time: plain, peak (no model/seed)
#   ppo_no_priority, ppo_priority: plain, peak x seeds 42, 7
# Total: 10 runs

set -euo pipefail

export PYTHONPATH="$(pwd)"
LOG_DIR="results/logs/eval"
MODEL_DIR="results/models"
mkdir -p "$LOG_DIR"

PPO_CONFIGS=(ppo_no_priority ppo_priority)
TRAFFIC=(plain peak)
SEEDS=(42 7)

TOTAL=$(( 2 + ${#PPO_CONFIGS[@]} * ${#TRAFFIC[@]} * ${#SEEDS[@]} ))
RUN=0
SKIPPED=0
FAILED=()

# fixed_time baseline (no model, no seed)
for traffic in "${TRAFFIC[@]}"; do
    RUN=$(( RUN + 1 ))
    TAG="fixed_time_${traffic}"
    LOG="$LOG_DIR/${TAG}.log"
    EVAL_FILE="results/eval/fixed_time_${traffic}_results.json"

    echo ""
    echo "[$RUN/$TOTAL] $TAG"

    if [[ -f "$EVAL_FILE" ]]; then
        printf "  ! %s — resultado já existe (%s)\n" "$TAG" "$EVAL_FILE"
        read -r -p "    Sobrescrever? [y/N] " REPLY
        if [[ "${REPLY,,}" != "y" ]]; then
            printf "  ↷ Pulando %s\n" "$TAG"
            SKIPPED=$(( SKIPPED + 1 ))
            continue
        fi
    fi

    echo "  log → $LOG"

    if python src/evaluation/evaluate.py \
            --config fixed_time \
            --traffic "$traffic" \
            2>&1 | tee "$LOG"; then
        echo "  ✓ done"
    else
        echo "  ✗ FAILED (exit $?)"
        FAILED+=("$TAG")
    fi
done

# PPO configs
for config in "${PPO_CONFIGS[@]}"; do
    for traffic in "${TRAFFIC[@]}"; do
        for seed in "${SEEDS[@]}"; do
            RUN=$(( RUN + 1 ))
            TAG="${config}_${traffic}_seed${seed}"
            MODEL="$MODEL_DIR/${TAG}.zip"
            LOG="$LOG_DIR/${TAG}.log"

            echo ""
            echo "[$RUN/$TOTAL] $TAG"

            if [ ! -f "$MODEL" ]; then
                echo "  ✗ SKIPPED — model not found: $MODEL"
                FAILED+=("$TAG (missing model)")
                continue
            fi

            EVAL_FILE="results/eval/${config}_${traffic}_seed${seed}_results.json"
            if [[ -f "$EVAL_FILE" ]]; then
                printf "  ! %s — resultado já existe (%s)\n" "$TAG" "$EVAL_FILE"
                read -r -p "    Sobrescrever? [y/N] " REPLY
                if [[ "${REPLY,,}" != "y" ]]; then
                    printf "  ↷ Pulando %s\n" "$TAG"
                    SKIPPED=$(( SKIPPED + 1 ))
                    continue
                fi
            fi

            echo "  model → $MODEL"
            echo "  log   → $LOG"

            if python src/evaluation/evaluate.py \
                    --config "$config" \
                    --traffic "$traffic" \
                    --seed "$seed" \
                    --model "$MODEL" \
                    2>&1 | tee "$LOG"; then
                echo "  ✓ done"
            else
                echo "  ✗ FAILED (exit $?)"
                FAILED+=("$TAG")
            fi
        done
    done
done

echo ""
echo "========================================"
printf "Completed %d/%d runs (%d skipped, %d evaluated)\n" \
       "$RUN" "$TOTAL" "$SKIPPED" "$(( RUN - SKIPPED - ${#FAILED[@]} ))"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED/SKIPPED (${#FAILED[@]}):"
    for t in "${FAILED[@]}"; do echo "  - $t"; done
    exit 1
else
    echo "All runs succeeded."
fi
