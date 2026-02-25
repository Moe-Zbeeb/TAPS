#!/usr/bin/env bash
# Evaluate dual-head EAGLE-2 (MathInstruct + ShareGPT, step-level confidence routing).
# Outputs τ = total_new_tokens / total_decode_steps for each benchmark.
#
# Usage:
#   BASE_MODEL_PATH=/path/to/Meta-Llama-3-8B-Instruct bash scripts/run_dual_head_eval.sh
#
# Override any variable on the command line, e.g.
#   BENCHES="gsm8k math_500" bash scripts/run_dual_head_eval.sh

set -euo pipefail

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/zbibm/MOSS---Mixture-of-Speculative-Samplers/base}"
EA_MODEL_PATH1="${EA_MODEL_PATH1:-checkpoints/Eagle-MathInstruct_20epochs}"
EA_MODEL_PATH2="${EA_MODEL_PATH2:-checkpoints/Eagle-ShareGPT_20epochs}"
HEAD1_NAME="${HEAD1_NAME:-MathInstruct}"
HEAD2_NAME="${HEAD2_NAME:-ShareGPT}"
MODEL_ID="${MODEL_ID:-dual_eagle2_math_sharegpt}"
TOTAL_TOKEN="${TOTAL_TOKEN:-60}"
DEPTH="${DEPTH:-5}"
TOP_K="${TOP_K:-10}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-1024}"

RESULTS_DIR="results/dual_head"
LOGS_DIR="results/dual_head/logs"
mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

# One benchmark per GPU — assigned in order
BENCHES=(mt_bench gsm8k math_500 svamp)
GPUS=(0 1 2 3)

echo "========================================================"
echo "Dual-head EAGLE-2 evaluation  [4 GPUs, parallel]"
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

    CUDA_VISIBLE_DEVICES="${gpu}" python -m eagle.evaluation.gen_dual_head_eval \
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

# Wait for all and report any failures
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

# Print final summary row alongside baselines
echo ""
python3 - <<'PYEOF'
import json
import os
from pathlib import Path

results_dir = Path(os.environ.get("RESULTS_DIR", "results/dual_head"))
model_id    = os.environ.get("MODEL_ID", "dual_eagle2_math_sharegpt")
head_label  = os.environ.get("HEAD1_NAME", "MathInstruct") + "+" + os.environ.get("HEAD2_NAME", "ShareGPT") + " (dual)"
benches     = ["mt_bench", "gsm8k", "math_500", "svamp"]

BASELINE = {
    "MathInstruct": {"mt_bench": 2.541, "gsm8k": 4.923, "math_500": 5.284, "svamp": 4.812},
    "ShareGPT":     {"mt_bench": 3.606, "gsm8k": 3.835, "math_500": 3.968, "svamp": 3.778},
}

taus = {}
for bench in benches:
    p = results_dir / f"{bench}_{model_id}.jsonl"
    if not p.exists():
        taus[bench] = "?"
        continue
    total_tokens, total_steps = 0, 0
    with open(p) as f:
        for line in f:
            rec = json.loads(line)
            for choice in rec.get("choices", []):
                total_tokens += sum(choice.get("new_tokens", []))
                total_steps  += sum(choice.get("idxs", []))
    taus[bench] = round(total_tokens / total_steps, 3) if total_steps else 0.0

col_w = 12
header = f"{'Head':<28}" + "".join(f"{b:>{col_w}}" for b in benches)
sep    = "-" * len(header)
print(header)
print(sep)
for head, row in BASELINE.items():
    print(f"{head:<28}" + "".join(f"{row.get(b, '-'):>{col_w}}" for b in benches))
print(sep)
print(f"{head_label:<28}" + "".join(f"{taus[b]:>{col_w}}" for b in benches))
PYEOF
