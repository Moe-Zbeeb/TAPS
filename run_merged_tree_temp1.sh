#!/usr/bin/env bash
# Combined merged-tree evaluation at temperature=1 for both Eagle and HASS.
#
# Phase 1 (parallel, 4 GPUs): single-head baselines
#   GPU 0 — Eagle MathInstruct   (4 benchmarks sequential)
#   GPU 1 — Eagle ShareGPT       (4 benchmarks sequential)
#   GPU 2 — HASS  MathInstruct   (4 benchmarks sequential)
#   GPU 3 — HASS  ShareGPT       (4 benchmarks sequential)
#
# Phase 2 (parallel, 4 GPUs): Eagle merged-tree T=1
# Phase 3 (parallel, 4 GPUs): HASS  merged-tree T=1
# Phase 4: combined table
#
# Usage:
#   bash run_merged_tree_temp1.sh

set -euo pipefail

MOSS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EAGLE_DIR="${MOSS_ROOT}/Eagle-Code"
HASS_DIR="${MOSS_ROOT}/Hass-Code"
BASE="${MOSS_ROOT}/base"

EAGLE_PY="/home/zbibm/miniconda3/envs/eagle_env/bin/python"
HASS_PY="/home/zbibm/miniconda3/envs/HASS/bin/python"

EAGLE_CKPT1="${EAGLE_DIR}/checkpoints/Eagle-MathInstruct_20epochs"
EAGLE_CKPT2="${EAGLE_DIR}/checkpoints/Eagle-ShareGPT_20epochs"
HASS_CKPT1="${HASS_DIR}/checkpoints/MathInstruct_20epochs/state_final"
HASS_CKPT2="${HASS_DIR}/checkpoints/ShareGPT_20epochs"

TEMP=1.0
TOTAL_TOKEN=60
DEPTH=5
TOP_K=10
MAX_NEW_TOKEN=1024

EAGLE_OUT="${EAGLE_DIR}/results/merged_temp1"
HASS_OUT="${HASS_DIR}/results/merged_temp1"
LOGDIR="${MOSS_ROOT}/logs/merged_temp1"

BENCHES=(mt_bench gsm8k math_500 svamp)

mkdir -p "${EAGLE_OUT}" "${HASS_OUT}" "${LOGDIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

run_eagle_single() {
    local GPU=$1 KEY=$2 CKPT=$3 OUT_DIR=$4
    export CUDA_VISIBLE_DEVICES=${GPU}
    cd "${EAGLE_DIR}"
    for bench in "${BENCHES[@]}"; do
        local outfile="${OUT_DIR}/${KEY}_${bench}.jsonl"
        [[ -f "${outfile}" ]] && echo "[GPU${GPU}] Eagle/${KEY}/${bench} exists, skip" && continue
        echo "[GPU${GPU}] Eagle/${KEY}/${bench} ..."
        "${EAGLE_PY}" -m eagle.evaluation.gen_ea_answer_llama3chat \
            --base-model-path "${BASE}" \
            --ea-model-path   "${CKPT}" \
            --model-id        "${KEY}" \
            --bench-name      "${bench}" \
            --answer-file     "${outfile}" \
            --temperature     "${TEMP}" \
            --total-token     "${TOTAL_TOKEN}" \
            --depth           "${DEPTH}" \
            --top-k           "${TOP_K}" \
            --max-new-token   "${MAX_NEW_TOKEN}"
    done
}

run_hass_single() {
    local GPU=$1 KEY=$2 CKPT=$3 OUT_DIR=$4
    export CUDA_VISIBLE_DEVICES=${GPU}
    cd "${HASS_DIR}"
    for bench in "${BENCHES[@]}"; do
        local outfile="${OUT_DIR}/${KEY}_${bench}.jsonl"
        [[ -f "${outfile}" ]] && echo "[GPU${GPU}] HASS/${KEY}/${bench} exists, skip" && continue
        echo "[GPU${GPU}] HASS/${KEY}/${bench} ..."
        "${HASS_PY}" -m evaluation.gen_ea_answer_llama3chat \
            --base-model-path "${BASE}" \
            --ea-model-path   "${CKPT}" \
            --model-id        "${KEY}" \
            --bench-name      "${bench}" \
            --answer-file     "${outfile}" \
            --temperature     "${TEMP}" \
            --total-token     "${TOTAL_TOKEN}" \
            --depth           "${DEPTH}" \
            --top-k           "${TOP_K}" \
            --max-new-token   "${MAX_NEW_TOKEN}"
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: single-head baselines (4 GPUs in parallel)
# ─────────────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Phase 1: single-head baselines at T=${TEMP}"
echo "  GPU 0 — Eagle MathInstruct"
echo "  GPU 1 — Eagle ShareGPT"
echo "  GPU 2 — HASS  MathInstruct"
echo "  GPU 3 — HASS  ShareGPT"
echo "========================================================"

run_eagle_single 0 EagleMathInstruct "${EAGLE_CKPT1}" "${EAGLE_OUT}" \
    > "${LOGDIR}/eagle_math.log" 2>&1 &
PID0=$!

run_eagle_single 1 EagleShareGPT "${EAGLE_CKPT2}" "${EAGLE_OUT}" \
    > "${LOGDIR}/eagle_share.log" 2>&1 &
PID1=$!

run_hass_single 2 HassMathInstruct "${HASS_CKPT1}" "${HASS_OUT}" \
    > "${LOGDIR}/hass_math.log" 2>&1 &
PID2=$!

run_hass_single 3 HassShareGPT "${HASS_CKPT2}" "${HASS_OUT}" \
    > "${LOGDIR}/hass_share.log" 2>&1 &
PID3=$!

wait ${PID0} && echo "  [done] Eagle MathInstruct" || echo "  [FAILED] Eagle MathInstruct"
wait ${PID1} && echo "  [done] Eagle ShareGPT"     || echo "  [FAILED] Eagle ShareGPT"
wait ${PID2} && echo "  [done] HASS MathInstruct"  || echo "  [FAILED] HASS MathInstruct"
wait ${PID3} && echo "  [done] HASS ShareGPT"      || echo "  [FAILED] HASS ShareGPT"

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Eagle merged-tree T=1 (4 GPUs in parallel)
# ─────────────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Phase 2: Eagle merged-tree at T=${TEMP}"
echo "========================================================"

cd "${EAGLE_DIR}"
PIDS=()
for i in "${!BENCHES[@]}"; do
    bench="${BENCHES[$i]}"
    gpu=$i
    outfile="${EAGLE_OUT}/eagle_merged_${bench}.jsonl"
    logfile="${LOGDIR}/eagle_merged_${bench}.log"
    [[ -f "${outfile}" ]] && echo "  ${bench} exists, skip" && PIDS+=(0) && continue
    echo "  Launching Eagle/${bench} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES=${gpu} "${EAGLE_PY}" -m eagle.evaluation.gen_merged_tree_eval \
        --base-model-path "${BASE}" \
        --ea-model-path1  "${EAGLE_CKPT1}" \
        --ea-model-path2  "${EAGLE_CKPT2}" \
        --head1-name      MathInstruct \
        --head2-name      ShareGPT \
        --model-id        merged_tree_eagle2 \
        --bench-name      "${bench}" \
        --answer-file     "${outfile}" \
        --temperature     "${TEMP}" \
        --total-token     "${TOTAL_TOKEN}" \
        --depth           "${DEPTH}" \
        --top-k           "${TOP_K}" \
        --max-new-token   "${MAX_NEW_TOKEN}" \
        > "${logfile}" 2>&1 &
    PIDS+=($!)
done

for i in "${!BENCHES[@]}"; do
    bench="${BENCHES[$i]}"
    pid="${PIDS[$i]}"
    [[ "${pid}" == "0" ]] && continue
    wait "${pid}" && echo "  [done] Eagle/${bench}" || echo "  [FAILED] Eagle/${bench} -- see ${LOGDIR}/eagle_merged_${bench}.log"
done

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: HASS merged-tree T=1 (4 GPUs in parallel)
# ─────────────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Phase 3: HASS merged-tree at T=${TEMP}"
echo "========================================================"

cd "${HASS_DIR}"
PIDS=()
for i in "${!BENCHES[@]}"; do
    bench="${BENCHES[$i]}"
    gpu=$i
    outfile="${HASS_OUT}/hass_merged_${bench}.jsonl"
    logfile="${LOGDIR}/hass_merged_${bench}.log"
    [[ -f "${outfile}" ]] && echo "  ${bench} exists, skip" && PIDS+=(0) && continue
    echo "  Launching HASS/${bench} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES=${gpu} "${HASS_PY}" -m evaluation.gen_merged_tree_eval \
        --base-model-path "${BASE}" \
        --ea-model-path1  "${HASS_CKPT1}" \
        --ea-model-path2  "${HASS_CKPT2}" \
        --head1-name      MathInstruct \
        --head2-name      ShareGPT \
        --model-id        merged_tree_hass \
        --bench-name      "${bench}" \
        --answer-file     "${outfile}" \
        --temperature     "${TEMP}" \
        --total-token     "${TOTAL_TOKEN}" \
        --depth           "${DEPTH}" \
        --top-k           "${TOP_K}" \
        --max-new-token   "${MAX_NEW_TOKEN}" \
        > "${logfile}" 2>&1 &
    PIDS+=($!)
done

for i in "${!BENCHES[@]}"; do
    bench="${BENCHES[$i]}"
    pid="${PIDS[$i]}"
    [[ "${pid}" == "0" ]] && continue
    wait "${pid}" && echo "  [done] HASS/${bench}" || echo "  [FAILED] HASS/${bench} -- see ${LOGDIR}/hass_merged_${bench}.log"
done

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: combined table
# ─────────────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Results (temperature=${TEMP})"
echo "========================================================"

python3 - <<PYEOF
import json
from pathlib import Path

BENCHES = ["mt_bench", "gsm8k", "math_500", "svamp"]

eagle_out = Path("${EAGLE_OUT}")
hass_out  = Path("${HASS_OUT}")

def tau(path):
    path = Path(path)
    if not path.exists():
        return "?"
    total_tok, total_steps = 0, 0
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            for ch in rec.get("choices", []):
                total_tok   += sum(ch.get("new_tokens", []))
                total_steps += sum(ch.get("idxs", []))
    return round(total_tok / total_steps, 3) if total_steps else 0.0

ROWS = [
    ("Eagle MathInstruct (single)", [eagle_out / f"EagleMathInstruct_{b}.jsonl" for b in BENCHES]),
    ("Eagle ShareGPT (single)",     [eagle_out / f"EagleShareGPT_{b}.jsonl"     for b in BENCHES]),
    ("Eagle Merged-tree",           [eagle_out / f"eagle_merged_{b}.jsonl"      for b in BENCHES]),
    None,
    ("HASS  MathInstruct (single)", [hass_out  / f"HassMathInstruct_{b}.jsonl"  for b in BENCHES]),
    ("HASS  ShareGPT (single)",     [hass_out  / f"HassShareGPT_{b}.jsonl"      for b in BENCHES]),
    ("HASS  Merged-tree",           [hass_out  / f"hass_merged_{b}.jsonl"       for b in BENCHES]),
]

col_w   = 12
label_w = 32
header  = f"{'Model':<{label_w}}" + "".join(f"{b:>{col_w}}" for b in BENCHES)
sep     = "-" * len(header)

lines = [
    f"Merged-tree evaluation  (temperature=${TEMP})",
    sep, header, sep,
]

for row in ROWS:
    if row is None:
        lines.append(sep)
        continue
    label, paths = row
    taus = [tau(p) for p in paths]
    cells = "".join(f"{t:>{col_w}}" if t != "?" else f"{'?':>{col_w}}" for t in taus)
    lines.append(f"{label:<{label_w}}{cells}")

lines.append(sep)
table = "\n".join(lines)
print(table)

out = Path("${MOSS_ROOT}/results_merged_temp1.txt")
out.parent.mkdir(exist_ok=True)
out.write_text(table + "\n")
print(f"\nSaved to {out}")
PYEOF
