#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# Run EAGLE-2 entropy analysis for all 4 checkpoints in parallel
# Each checkpoint runs on its own GPU (0–3), iterating over all
# specified benchmarks sequentially.
# ──────────────────────────────────────────────────────────────

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to your base model (e.g. /path/to/Meta-Llama-3-8B-Instruct)}"
COMMON_ARGS="--total-token 60 --depth 8 --top-k 10 --temperature 0.0 --output-dir results/entropy"

mkdir -p results/entropy

# GPU 0: MathInstruct
CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_entropy_analysis \
  --ea-model-path checkpoints/Eagle-MathInstruct_20epochs \
  --base-model-path "$BASE_MODEL" --model-id MathInstruct \
  --bench-name mt_bench gsm8k math_500 svamp $COMMON_ARGS \
  > results/entropy/MathInstruct.log 2>&1 &

# GPU 1: ShareGPT
CUDA_VISIBLE_DEVICES=1 python -m eagle.evaluation.gen_entropy_analysis \
  --ea-model-path checkpoints/Eagle-ShareGPT_20epochs \
  --base-model-path "$BASE_MODEL" --model-id ShareGPT \
  --bench-name mt_bench gsm8k math_500 svamp $COMMON_ARGS \
  > results/entropy/ShareGPT.log 2>&1 &

# GPU 2: ShareGPT-MathInstruct
CUDA_VISIBLE_DEVICES=2 python -m eagle.evaluation.gen_entropy_analysis \
  --ea-model-path checkpoints/Eagle-Sharegpt-Mathinstruct-20epochs \
  --base-model-path "$BASE_MODEL" --model-id ShareGPT-MathInstruct \
  --bench-name mt_bench gsm8k math_500 svamp $COMMON_ARGS \
  > results/entropy/ShareGPT-MathInstruct.log 2>&1 &

# GPU 3: Averaged
CUDA_VISIBLE_DEVICES=3 python -m eagle.evaluation.gen_entropy_analysis \
  --ea-model-path checkpoints/Eagle-Averaged-Checkpoint \
  --base-model-path "$BASE_MODEL" --model-id Averaged \
  --bench-name mt_bench gsm8k math_500 svamp $COMMON_ARGS \
  > results/entropy/Averaged.log 2>&1 &

echo "All 4 checkpoints launched in parallel."
echo "Logs: results/entropy/{MathInstruct,ShareGPT,ShareGPT-MathInstruct,Averaged}.log"
echo ""
echo "Waiting for all to finish..."
wait
echo "All finished."
echo ""

python evaluate/summarize_entropy.py
