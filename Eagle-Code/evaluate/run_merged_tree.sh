#!/usr/bin/env bash
# Evaluate merged-tree dual-head EAGLE-2.
# Both heads' draft trees are merged into one combined tree per step and
# submitted to the verifier in a single pass.
#
# Usage:
#   BASE_MODEL_PATH=/path/to/Meta-Llama-3-8B-Instruct bash evaluate/run_merged_tree.sh
#
# Override any variable on the command line, e.g.
#   BENCHES="gsm8k math_500" bash evaluate/run_merged_tree.sh

set -euo pipefail

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/zbibm/MOSS---Mixture-of-Speculative-Samplers/base}"
EA_MODEL_PATH1="${EA_MODEL_PATH1:-checkpoints/Eagle-MathInstruct_20epochs}"
EA_MODEL_PATH2="${EA_MODEL_PATH2:-checkpoints/Eagle-ShareGPT_20epochs}"
HEAD1_NAME="${HEAD1_NAME:-MathInstruct}"
HEAD2_NAME="${HEAD2_NAME:-ShareGPT}"
MODEL_ID="${MODEL_ID:-merged_tree_eagle2}"
TOTAL_TOKEN="${TOTAL_TOKEN:-60}"
DEPTH="${DEPTH:-5}"
TOP_K="${TOP_K:-10}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-1024}"

RESULTS_DIR="results/merged"
LOGS_DIR="results/merged/logs"
mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

BENCHES=(mt_bench gsm8k math_500 svamp)
GPUS=(0 1 2 3)

echo "========================================================"
echo "Merged-tree dual-head EAGLE-2 evaluation  [4 GPUs, parallel]"
echo "  Head 1 : ${HEAD1_NAME}  (${EA_MODEL_PATH1})"
echo "  Head 2 : ${HEAD2_NAME}  (${EA_MODEL_PATH2})"
echo "  Base   : ${BASE_MODEL_PATH}"
echo "  GPU assignment:"
for i in "${!BENCHES[@]}"; do
    echo "    GPU ${GPUS[$i]} -> ${BENCHES[$i]}"
done
echo "========================================================"

PIDS=()
for i in "${!BENCHES[@]}"; do
    bench="${BENCHES[$i]}"
    gpu="${GPUS[$i]}"
    out_file="${RESULTS_DIR}/${bench}_${MODEL_ID}.jsonl"
    log_file="${LOGS_DIR}/${bench}.log"

    echo "Launching ${bench} on GPU ${gpu}  (log: ${log_file})"

    CUDA_VISIBLE_DEVICES="${gpu}" python -m eagle.evaluation.gen_merged_tree_eval \
        --base-model-path   "${BASE_MODEL_PATH}" \
        --ea-model-path1    "${EA_MODEL_PATH1}" \
        --ea-model-path2    "${EA_MODEL_PATH2}" \
        --head1-name        "${HEAD1_NAME}" \
        --head2-name        "${HEAD2_NAME}" \
        --model-id          "${MODEL_ID}" \
        --bench-name        "${bench}" \
        --answer-file       "${out_file}" \
        --total-token       "${TOTAL_TOKEN}" \
        --depth             "${DEPTH}" \
        --top-k             "${TOP_K}" \
        --temperature       "${TEMPERATURE}" \
        --max-new-token     "${MAX_NEW_TOKEN}" \
        > "${log_file}" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All 4 jobs running. Waiting for completion..."

FAILED=0
for i in "${!BENCHES[@]}"; do
    bench="${BENCHES[$i]}"
    if wait "${PIDS[$i]}"; then
        echo "  [done] ${bench}"
    else
        echo "  [FAILED] ${bench}  -- check ${LOGS_DIR}/${bench}.log"
        FAILED=1
    fi
done

if [ "${FAILED}" -eq 1 ]; then
    echo "One or more benchmarks failed. Aborting summary."
    exit 1
fi

echo ""
python evaluate/merged_tree_table.py \
    --results-dir "${RESULTS_DIR}" \
    --model-id    "${MODEL_ID}"
