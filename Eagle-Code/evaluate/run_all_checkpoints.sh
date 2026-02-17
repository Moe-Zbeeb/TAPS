#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# Wrapper: run run_eval_all.sh for 4 checkpoints, one per GPU
# ──────────────────────────────────────────────────────────────

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EVAL_SCRIPT="evaluate/run_eval_all.sh"
BASE_MODEL_PATH="${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to your base model (e.g. /path/to/Meta-Llama-3-8B-Instruct)}"

declare -A MODELS
MODELS=(
  [0]="MathInstruct|checkpoints/Eagle-MathInstruct_20epochs"
  [1]="ShareGPT|checkpoints/Eagle-ShareGPT_20epochs"
  [2]="ShareGPT-MathInstruct|checkpoints/Eagle-Sharegpt-Mathinstruct-20epochs"
  [3]="Averaged|checkpoints/Eagle-Averaged-Checkpoint"
)

PIDS=()
for gpu in 0 1 2 3; do
  IFS='|' read -r model_id ea_path <<< "${MODELS[$gpu]}"

  echo "Launching ${model_id} on GPU ${gpu}..."

  CUDA_VISIBLE_DEVICES="${gpu}" \
  EA_MODEL_PATH="${ea_path}" \
  BASE_MODEL_PATH="${BASE_MODEL_PATH}" \
  MODEL_ID="${model_id}" \
    bash "${EVAL_SCRIPT}" > "results/${model_id}.log" 2>&1 &

  PIDS+=($!)
done

echo ""
echo "All 4 launched. PIDs: ${PIDS[*]}"
echo "Logs: results/{MathInstruct,ShareGPT,ShareGPT-MathInstruct,Averaged}.log"
echo ""
echo "Waiting for all to finish..."

FAIL=0
for i in 0 1 2 3; do
  IFS='|' read -r model_id _ <<< "${MODELS[$i]}"
  if wait "${PIDS[$i]}"; then
    echo "  [done] ${model_id} (GPU ${i})"
  else
    echo "  [FAIL] ${model_id} (GPU ${i}) — check results/${model_id}.log"
    FAIL=1
  fi
done

if [ "${FAIL}" -ne 0 ]; then
  echo ""
  echo "WARNING: Some evaluations failed. Check logs above."
fi

echo ""
echo "All finished. Results in results/tau_summary.json"
