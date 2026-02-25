#!/usr/bin/env bash
set -euo pipefail

EA_MODEL_PATH="${EA_MODEL_PATH:-checkpoints/eagle2_llama3_8b_mathinstruct/state_20_ea}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to your base model (e.g. /path/to/Meta-Llama-3-8B-Instruct)}"
MODEL_ID="${MODEL_ID:-eagle2_mathinstruct_state_20}"
TOTAL_TOKEN="${TOTAL_TOKEN:-60}"
DEPTH="${DEPTH:-8}"
TOP_K="${TOP_K:-10}"
TEMPERATURE="${TEMPERATURE:-0.0}"

mkdir -p results
SUMMARY_JSON="results/tau_summary.json"
TMP_SUMMARY="results/.tau_summary.tmp.json"

python - <<'PY'
import json
from pathlib import Path

path = Path("results/tau_summary.json")
if path.exists():
    data = json.loads(path.read_text())
else:
    data = {}

# Write an empty/placeholder file to ensure it's valid JSON
path.write_text(json.dumps(data, indent=2))
PY

run_eval() {
  local bench_name="$1"
  local out_file="results/${bench_name}_${MODEL_ID}.jsonl"

  python -m eagle.evaluation.gen_ea_answer_llama3chat \
    --ea-model-path "${EA_MODEL_PATH}" \
    --base-model-path "${BASE_MODEL_PATH}" \
    --bench-name "${bench_name}" \
    --model-id "${MODEL_ID}" \
    --total-token "${TOTAL_TOKEN}" \
    --depth "${DEPTH}" \
    --top-k "${TOP_K}" \
    --temperature "${TEMPERATURE}" \
    --answer-file "${out_file}"

  python scripts/compute_tau.py "${out_file}"

  # Append tau stats into a single JSON summary
  python - <<PY
import json
from pathlib import Path

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def compute_tau(records):
    total_new_tokens = 0
    total_idxs = 0
    total_wall_time = 0
    num_questions = 0
    for rec in records:
        choices = rec.get("choices", [])
        if not choices:
            continue
        for choice in choices:
            total_new_tokens += sum(choice.get("new_tokens", []))
            total_idxs += sum(choice.get("idxs", []))
            total_wall_time += sum(choice.get("wall_time", []))
        num_questions += 1
    tau = total_new_tokens / total_idxs if total_idxs else 0
    tokens_per_sec = total_new_tokens / total_wall_time if total_wall_time else 0
    return {
        "num_questions": num_questions,
        "total_new_tokens": total_new_tokens,
        "total_iterations": total_idxs,
        "total_wall_time": round(total_wall_time, 2),
        "tau": round(tau, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
    }

out_file = Path("${out_file}")
summary_path = Path("${SUMMARY_JSON}")

stats = compute_tau(load_jsonl(out_file))

summary = {}
if summary_path.exists():
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        summary = {}

bench = "${bench_name}"
model = "${MODEL_ID}"
summary.setdefault(bench, {})[model] = stats
summary_path.write_text(json.dumps(summary, indent=2))
PY
}

run_eval mt_bench
run_eval gsm8k
run_eval math_500
run_eval svamp
