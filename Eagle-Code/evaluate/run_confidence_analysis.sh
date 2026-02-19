#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_PATH="${1:-}"
if [[ -z "${BASE_MODEL_PATH}" ]]; then
  echo "Usage: $0 <base_model_path>"
  exit 1
fi

python -m eagle.evaluation.gen_confidence_analysis \
  --base-model-path "${BASE_MODEL_PATH}" \
  --ea-model-paths \
    checkpoints/Eagle-MathInstruct_20epochs \
    checkpoints/Eagle-ShareGPT_20epochs \
    checkpoints/Eagle-Sharegpt-Mathinstruct-20epochs \
    checkpoints/Eagle-Averaged-Checkpoint \
  --model-ids MathInstruct ShareGPT ShareGPT-MathInstruct Averaged \
  --bench-name mt_bench gsm8k math_500 svamp \
  --output-dir results/confidence
