#!/usr/bin/env bash
# Evaluate all HASS checkpoints at temperature=1 across 4 GPUs.
# GPU 0: MathInstruct   GPU 1: ShareGPT
# GPU 2: Mixed          GPU 3: Averaged
# Then runs step-level confidence routed eval (MathInstruct+ShareGPT) on GPU 0.
# Prints a tau table at the end matching the T=0 paper table format.
set -euo pipefail

HASS_DIR=/home/zbibm/MOSS---Mixture-of-Speculative-Samplers/Hass-Code
BASE=/home/zbibm/MOSS---Mixture-of-Speculative-Samplers/base
CKPT=$HASS_DIR/checkpoints
OUT=$HASS_DIR/results/temp1
LOGDIR=$OUT/logs
BENCHMARKS=(mt_bench gsm8k math_500 svamp)
TEMP=1.0

mkdir -p "$OUT" "$LOGDIR"

# Must run python -m from Hass-Code root
cd "$HASS_DIR"

# -----------------------------------------------------------------------
# Single-head eval: run all 4 benchmarks sequentially on one GPU
# -----------------------------------------------------------------------
run_single_head() {
    local GPU=$1 KEY=$2 CKPT_PATH=$3
    export CUDA_VISIBLE_DEVICES=$GPU
    for BENCH in "${BENCHMARKS[@]}"; do
        local OUTFILE="$OUT/${KEY}_${BENCH}.jsonl"
        if [[ -f "$OUTFILE" ]]; then
            echo "[GPU$GPU] $KEY/$BENCH already exists, skipping."
            continue
        fi
        echo "[GPU$GPU] $KEY / $BENCH ..."
        python -m evaluation.gen_ea_answer_llama3chat \
            --base-model-path "$BASE" \
            --ea-model-path   "$CKPT_PATH" \
            --model-id        "$KEY" \
            --bench-name      "$BENCH" \
            --answer-file     "$OUTFILE" \
            --temperature     $TEMP \
            --total-token     60 \
            --depth           5 \
            --top-k           10 \
            --num-choices     1 \
            --max-new-token   1024 \
            --seed            42
    done
}

# Launch 4 heads in parallel, one per GPU
echo "=== Launching HASS single-head evals in parallel ==="
run_single_head 0 MathInstruct "$CKPT/MathInstruct_20epochs/state_final"   > "$LOGDIR/MathInstruct.log" 2>&1 &
run_single_head 1 ShareGPT     "$CKPT/ShareGPT_20epochs"                    > "$LOGDIR/ShareGPT.log"     2>&1 &
run_single_head 2 Mixed        "$CKPT/Sharegpt-Mathinstruct-20epochs"       > "$LOGDIR/Mixed.log"        2>&1 &
run_single_head 3 Averaged     "$CKPT/Averaged-Checkpoint"                  > "$LOGDIR/Averaged.log"     2>&1 &

echo "Waiting for all 4 GPU jobs..."
wait
echo "Single-head evals complete."

# -----------------------------------------------------------------------
# Routed eval: DualEaModel (MathInstruct + ShareGPT) on GPU 0
# -----------------------------------------------------------------------
echo "=== Running confidence-routed eval (GPU 0) ==="
export CUDA_VISIBLE_DEVICES=0
for BENCH in "${BENCHMARKS[@]}"; do
    OUTFILE="$OUT/RoutedConf_${BENCH}.jsonl"
    if [[ -f "$OUTFILE" ]]; then
        echo "[GPU0] RoutedConf/$BENCH already exists, skipping."
        continue
    fi
    echo "[GPU0] RoutedConf / $BENCH ..."
    python -m evaluation.gen_dual_head_eval \
        --base-model-path  "$BASE" \
        --ea-model-path1   "$CKPT/MathInstruct_20epochs/state_final" \
        --ea-model-path2   "$CKPT/ShareGPT_20epochs" \
        --head1-name       MathInstruct \
        --head2-name       ShareGPT \
        --model-id         routed_conf \
        --bench-name       "$BENCH" \
        --answer-file      "$OUTFILE" \
        --temperature      $TEMP \
        --total-token      60 \
        --depth            5 \
        --top-k            10 \
        --max-new-token    1024 \
        --seed             42
done 2>&1 | tee "$LOGDIR/RoutedConf.log"

# -----------------------------------------------------------------------
# Print summary table
# -----------------------------------------------------------------------
echo ""
echo "=== Computing tau and printing table ==="
python3 - "$OUT" <<'PYEOF'
import sys
import json
from pathlib import Path

out_dir = Path(sys.argv[1])

MODELS = [
    ("MathInstruct", "MathInstruct"),
    ("ShareGPT",     "ShareGPT"),
    ("Mixed Data",   "Mixed"),
    ("Averaged",     "Averaged"),
    ("Routed Conf",  "RoutedConf"),
]
BENCHES    = ["mt_bench", "gsm8k", "math_500", "svamp"]
BENCH_HDRS = ["MT-Bench", "GSM8K",  "MATH-500", "SVAMP"]


def tau_from_jsonl(path):
    total_new = total_idx = 0
    try:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                for ch in rec.get("choices", []):
                    total_new += sum(ch.get("new_tokens", []))
                    total_idx  += sum(ch.get("idxs", []))
    except FileNotFoundError:
        return None
    return round(total_new / total_idx, 2) if total_idx else None


header = (
    f"{'Model Variant':<14} | "
    + " | ".join(f"{h:>8}" for h in BENCH_HDRS)
    + " | {'Average':>8}"
)
sep = "-" * len(header)

lines = [
    "",
    "Temperature = 1.0  —  τ (average accepted tokens / step)",
    sep,
    header,
    sep,
]

for label, key in MODELS:
    taus = [tau_from_jsonl(out_dir / f"{key}_{b}.jsonl") for b in BENCHES]
    cells = " | ".join(
        f"{t:>8.2f}" if t is not None else f"{'N/A':>8}" for t in taus
    )
    valid = [t for t in taus if t is not None]
    avg   = round(sum(valid) / len(valid), 2) if valid else None
    avg_s = f"{avg:>8.2f}" if avg is not None else f"{'N/A':>8}"
    lines.append(f"{label:<14} | {cells} | {avg_s}")

lines.append(sep)

output = "\n".join(lines)
print(output)

out_file = out_dir / "tau_table_temp1.txt"
with open(out_file, "w") as f:
    f.write(output + "\n")
print(f"\nTable saved to {out_file}")
PYEOF
