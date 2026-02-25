#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_PATH="${1:-}"
if [[ -z "${BASE_MODEL_PATH}" ]]; then
  echo "Usage: $0 <base_model_path> [gpu_list]"
  echo "Example: $0 /path/to/base 0,1,2,3"
  exit 1
fi

GPU_LIST="${2:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "gpu_list must contain exactly 4 GPUs, e.g. 0,1,2,3"
  exit 1
fi

mkdir -p results/confidence

EA_PATHS=(
  "checkpoints/Eagle-MathInstruct_20epochs"
  "checkpoints/Eagle-ShareGPT_20epochs"
  "checkpoints/Eagle-Sharegpt-Mathinstruct-20epochs"
  "checkpoints/Eagle-Averaged-Checkpoint"
)

MODEL_IDS=(
  "MathInstruct"
  "ShareGPT"
  "ShareGPT-MathInstruct"
  "Averaged"
)

BENCHES=(mt_bench gsm8k math_500 svamp)

PIDS=()
for i in 0 1 2 3; do
  gpu="${GPUS[$i]}"
  ea_path="${EA_PATHS[$i]}"
  model_id="${MODEL_IDS[$i]}"
  log_file="results/confidence/${model_id}.log"

  echo "Launching ${model_id} on GPU ${gpu} -> ${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python -m eagle.evaluation.gen_confidence_analysis \
      --base-model-path "${BASE_MODEL_PATH}" \
      --ea-model-paths "${ea_path}" \
      --model-ids "${model_id}" \
      --bench-name "${BENCHES[@]}" \
      --output-dir results/confidence \
      > "${log_file}" 2>&1 &
  PIDS+=("$!")
done

FAIL=0
for i in 0 1 2 3; do
  pid="${PIDS[$i]}"
  model_id="${MODEL_IDS[$i]}"
  if wait "${pid}"; then
    echo "${model_id}: done"
  else
    echo "${model_id}: failed (see results/confidence/${model_id}.log)"
    FAIL=1
  fi
done

python scripts/print_confidence_results.py \
  --output-dir results/confidence \
  --model-ids "${MODEL_IDS[@]}" \
  --bench-name "${BENCHES[@]}" \
  | tee results/confidence/summary_table.txt

if [[ "${FAIL}" -ne 0 ]]; then
  exit 1
fi
