#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Routed Multi-Head EA Evaluation Script
# Routes between general (ShareGPT) and math (MathInstruct) heads
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'USAGE'
Run routed evaluation between multiple EA heads on all benchmarks.

Usage:
  bash scripts/eval_routed.sh [OPTIONS]

Required:
  --base-model PATH      Base LLaMA model path

Optional:
  --head-general PATH    Path to general/ShareGPT head (default: uses original EAGLE checkpoint)
  --head-math PATH       Path to math/fine-tuned head
  --benchmarks LIST      Space-separated benchmarks (default: mt_bench gsm8k math_500 svamp)
  --output-dir DIR       Output directory (default: routed_eval_results)
  --gpu-index IDX        CUDA device (default: 0)
  --max-questions N      Limit questions per benchmark (for debugging)
  --temperature F        Decoding temperature (default: 0.0)

Example:
  bash scripts/eval_routed.sh \
    --base-model /path/to/Meta-Llama-3-8B-Instruct \
    --head-general checkpoints/ShareGPT_20epochs \
    --head-math checkpoints/MathInstruct_20epochs

USAGE
}

# Defaults
BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to your base model (e.g. /path/to/Meta-Llama-3-8B-Instruct)}"
HEAD_GENERAL="${HEAD_GENERAL:-checkpoints/ShareGPT_20epochs}"
HEAD_MATH="${HEAD_MATH:-checkpoints/MathInstruct_20epochs/state_final}"
BENCHMARKS="mt_bench gsm8k math_500 svamp"
OUTPUT_DIR="routed_eval_results"
GPU_INDEX="0"
MAX_QUESTIONS=""
TEMPERATURE="0.0"
TOTAL_TOKEN=60
DEPTH=5
TOPK=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --head-general) HEAD_GENERAL="$2"; shift 2 ;;
    --head-math) HEAD_MATH="$2"; shift 2 ;;
    --benchmarks) BENCHMARKS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --max-questions) MAX_QUESTIONS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --total-token) TOTAL_TOKEN="$2"; shift 2 ;;
    --depth) DEPTH="$2"; shift 2 ;;
    --top-k) TOPK="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

# Validate required args
if [[ -z "$BASE_MODEL" ]]; then
  echo "Error: --base-model is required"
  usage
  exit 1
fi

# Validate head paths exist
if [[ ! -d "$HEAD_GENERAL" ]]; then
  echo "Error: General head not found at $HEAD_GENERAL"
  exit 1
fi

if [[ ! -d "$HEAD_MATH" ]]; then
  echo "Error: Math head not found at $HEAD_MATH"
  exit 1
fi

echo "=============================================="
echo "Routed Multi-Head EA Evaluation"
echo "=============================================="
echo "Base model:    $BASE_MODEL"
echo "General head:  $HEAD_GENERAL"
echo "Math head:     $HEAD_MATH"
echo "Benchmarks:    $BENCHMARKS"
echo "Output dir:    $OUTPUT_DIR"
echo "GPU:           $GPU_INDEX"
echo "Temperature:   $TEMPERATURE"
echo "=============================================="

cd "$REPO_ROOT"

# Build command
CMD="CUDA_VISIBLE_DEVICES=$GPU_INDEX python -m evaluation.routed_eval \
  --base-model-path $BASE_MODEL \
  --head-general $HEAD_GENERAL \
  --head-math $HEAD_MATH \
  --benchmarks $BENCHMARKS \
  --output-dir $OUTPUT_DIR \
  --temperature $TEMPERATURE \
  --total-token $TOTAL_TOKEN \
  --depth $DEPTH \
  --top-k $TOPK"

if [[ -n "$MAX_QUESTIONS" ]]; then
  CMD="$CMD --max-questions $MAX_QUESTIONS"
fi

echo ""
echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo "=============================================="
