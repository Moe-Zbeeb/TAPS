#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Sweep merged checkpoints with different ShareGPT/MathInstruct weight ratios
# and benchmark on SVAMP, GSM8K, MATH500, and MT-Bench
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Paths
BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to your base model (e.g. /path/to/Meta-Llama-3-8B-Instruct)}"
CKPT_SHAREGPT="${REPO_ROOT}/checkpoints/ShareGPT_20epochs"
CKPT_MATH="${REPO_ROOT}/checkpoints/MathInstruct_20epochs/state_final"
MERGED_DIR="${REPO_ROOT}/checkpoints/merged_sweep"
RESULTS_DIR="${REPO_ROOT}/merge_sweep_results"

# Benchmarks to run
BENCHMARKS=("gsm8k" "math_500" "svamp" "mt_bench")

# Weight sweep: ShareGPT weight from 0.1 to 0.9 (MathInstruct = 1 - ShareGPT)
WEIGHTS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# GPU settings
GPU_INDEX="${GPU_INDEX:-0}"

usage() {
  cat <<'EOF'
Sweep merged checkpoint weights and benchmark on multiple datasets.

Usage: bash scripts/run_merge_sweep.sh [options]

Options:
  --base-model PATH     Base LLaMA model path (set via BASE_MODEL env var)
  --gpu INDEX           GPU index to use (default: 0)
  --skip-merge          Skip checkpoint merging (use existing merged checkpoints)
  --skip-eval           Skip evaluation (only merge checkpoints)
  --weights "W1 W2..."  Custom weights to sweep (default: "0.1 0.2 ... 0.9")
  --benchmarks "B1 B2"  Custom benchmarks (default: "gsm8k math_500 svamp mt_bench")
  -h, --help            Show this help

Example:
  bash scripts/run_merge_sweep.sh --gpu 0
  bash scripts/run_merge_sweep.sh --weights "0.3 0.5 0.7" --benchmarks "gsm8k svamp"
EOF
}

SKIP_MERGE=false
SKIP_EVAL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --gpu) GPU_INDEX="$2"; shift 2 ;;
    --skip-merge) SKIP_MERGE=true; shift ;;
    --skip-eval) SKIP_EVAL=true; shift ;;
    --weights) IFS=' ' read -ra WEIGHTS <<< "$2"; shift 2 ;;
    --benchmarks) IFS=' ' read -ra BENCHMARKS <<< "$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$MERGED_DIR" "$RESULTS_DIR"

# =============================================================================
# Step 1: Create merged checkpoints
# =============================================================================
if [[ "$SKIP_MERGE" == false ]]; then
  echo "=============================================="
  echo "Step 1: Creating merged checkpoints"
  echo "=============================================="
  echo "ShareGPT checkpoint: $CKPT_SHAREGPT"
  echo "MathInstruct checkpoint: $CKPT_MATH"
  echo ""

  for w_sharegpt in "${WEIGHTS[@]}"; do
    w_math=$(echo "1 - $w_sharegpt" | bc -l | sed 's/^\./0./')
    ckpt_name="merged_sharegpt${w_sharegpt}_math${w_math}"
    out_dir="${MERGED_DIR}/${ckpt_name}"

    if [[ -f "${out_dir}/pytorch_model.bin" ]]; then
      echo "[SKIP] ${ckpt_name} already exists"
      continue
    fi

    echo "[MERGE] ShareGPT=${w_sharegpt}, MathInstruct=${w_math} -> ${ckpt_name}"
    python "${REPO_ROOT}/scripts/merge_hass_checkpoints.py" \
      --ckpt-a "$CKPT_SHAREGPT" \
      --ckpt-b "$CKPT_MATH" \
      --out "$out_dir" \
      --weight-a "$w_sharegpt" \
      --weight-b "$w_math"
    echo ""
  done
  echo "Merged checkpoints saved to: $MERGED_DIR"
  echo ""
fi

# =============================================================================
# Step 2: Evaluate each merged checkpoint on all benchmarks
# =============================================================================
if [[ "$SKIP_EVAL" == false ]]; then
  echo "=============================================="
  echo "Step 2: Evaluating merged checkpoints"
  echo "=============================================="
  echo "Benchmarks: ${BENCHMARKS[*]}"
  echo "Base model: $BASE_MODEL"
  echo ""

  # Create summary file header
  SUMMARY_FILE="${RESULTS_DIR}/summary.csv"
  echo "checkpoint,sharegpt_weight,math_weight,benchmark,acceptance_length,num_samples" > "$SUMMARY_FILE"

  for w_sharegpt in "${WEIGHTS[@]}"; do
    w_math=$(echo "1 - $w_sharegpt" | bc -l | sed 's/^\./0./')
    ckpt_name="merged_sharegpt${w_sharegpt}_math${w_math}"
    ckpt_path="${MERGED_DIR}/${ckpt_name}"

    if [[ ! -f "${ckpt_path}/pytorch_model.bin" ]]; then
      echo "[ERROR] Checkpoint not found: ${ckpt_path}"
      continue
    fi

    for bench in "${BENCHMARKS[@]}"; do
      model_id="sweep_${ckpt_name}_${bench}"
      output_jsonl="${REPO_ROOT}/${bench}/${model_id}-temperature-0.0.jsonl"

      echo "----------------------------------------"
      echo "[EVAL] ${ckpt_name} on ${bench}"
      echo "----------------------------------------"

      # Skip if already evaluated
      if [[ -f "$output_jsonl" ]]; then
        echo "[SKIP] Output already exists: $output_jsonl"
      else
        # Run EA evaluation
        CUDA_VISIBLE_DEVICES="$GPU_INDEX" python -m evaluation.gen_ea_answer_llama3chat \
          --ea-model-path "$ckpt_path" \
          --base-model-path "$BASE_MODEL" \
          --model-id "$model_id" \
          --bench-name "$bench" \
          --temperature 0.0 \
          --total-token 60 \
          --depth 5 \
          --top-k 10 \
          --num-gpus-per-model 1 \
          --num-gpus-total 1
      fi

      # Extract acceptance length
      if [[ -f "$output_jsonl" ]]; then
        acc_result=$(python -m evaluation.acceptance_length --input_file "$output_jsonl" 2>&1)
        num_samples=$(echo "$acc_result" | grep "num of samples" | awk '{print $NF}')
        acc_length=$(echo "$acc_result" | grep "acceptance length" | awk '{print $NF}')
        echo "  -> Acceptance length: ${acc_length} (n=${num_samples})"
        echo "${ckpt_name},${w_sharegpt},${w_math},${bench},${acc_length},${num_samples}" >> "$SUMMARY_FILE"
      else
        echo "  -> [ERROR] Output file not found"
        echo "${ckpt_name},${w_sharegpt},${w_math},${bench},ERROR,0" >> "$SUMMARY_FILE"
      fi
      echo ""
    done
  done

  echo "=============================================="
  echo "Results saved to: $SUMMARY_FILE"
  echo "=============================================="
fi

# =============================================================================
# Step 3: Generate summary report
# =============================================================================
echo ""
echo "=============================================="
echo "Summary Report"
echo "=============================================="

if [[ -f "${RESULTS_DIR}/summary.csv" ]]; then
  # Create a nice formatted table
  python3 - <<'PYEND'
import pandas as pd
import sys

try:
    df = pd.read_csv("merge_sweep_results/summary.csv")

    # Pivot table: rows = weight ratio, columns = benchmark
    pivot = df.pivot_table(
        index=['sharegpt_weight', 'math_weight'],
        columns='benchmark',
        values='acceptance_length',
        aggfunc='first'
    ).round(3)

    print("\nAcceptance Length by Weight Ratio and Benchmark:")
    print("=" * 70)
    print(pivot.to_string())
    print("\n")

    # Find best weights per benchmark
    print("Best weight ratio per benchmark:")
    print("-" * 40)
    for bench in df['benchmark'].unique():
        bench_df = df[df['benchmark'] == bench]
        if 'ERROR' not in bench_df['acceptance_length'].values:
            bench_df_numeric = bench_df.copy()
            bench_df_numeric['acceptance_length'] = pd.to_numeric(bench_df_numeric['acceptance_length'], errors='coerce')
            best = bench_df_numeric.loc[bench_df_numeric['acceptance_length'].idxmax()]
            print(f"  {bench}: ShareGPT={best['sharegpt_weight']}, Math={best['math_weight']:.1f} -> {best['acceptance_length']:.3f}")

except Exception as e:
    print(f"Could not generate summary: {e}")
    sys.exit(0)
PYEND
fi

echo ""
echo "Done! Full results in: ${RESULTS_DIR}/"
