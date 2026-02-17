#!/bin/bash
#
# Run Full Multi-Head Entropy Analysis (4 heads x 4 benchmarks)
# Each checkpoint runs on its own GPU in parallel.
#

set -e

# Default paths
BASE_MODEL_PATH="${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to your base model (e.g. /path/to/Meta-Llama-3-8B-Instruct)}"
OUTPUT_DIR="${OUTPUT_DIR:-entropy_analysis_results}"
BENCHMARKS="${BENCHMARKS:-mt_bench gsm8k math_500 svamp}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ENTRY="$REPO_ROOT/evaluation/entropy_analysis.py"

# Forward extra args (e.g. --max-questions 2)
EXTRA_ARGS="${@}"

# ── checkpoint name, GPU id, path ──
HEADS=(
  "mathinstruct  0  ${HEAD_MATHINSTRUCT:-checkpoints/MathInstruct_20epochs/state_final}"
  "sharegpt      1  ${HEAD_SHAREGPT:-checkpoints/ShareGPT_20epochs}"
  "mixed         2  ${HEAD_MIXED:-checkpoints/Sharegpt-Mathinstruct-20epochs}"
  "averaged      3  ${HEAD_AVERAGED:-checkpoints/Averaged-Checkpoint}"
)

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Multi-Head Entropy Analysis (4 GPUs parallel)"
echo "=============================================="
echo "Base model:  $BASE_MODEL_PATH"
echo "Benchmarks:  $BENCHMARKS"
echo "Output dir:  $OUTPUT_DIR"
echo "Extra args:  $EXTRA_ARGS"
for entry in "${HEADS[@]}"; do
  read -r name gpu path <<< "$entry"
  printf "  GPU %s → %-14s  %s\n" "$gpu" "$name" "$path"
done
echo "=============================================="

# ── Launch one process per checkpoint ──
PIDS=()
for entry in "${HEADS[@]}"; do
  read -r name gpu path <<< "$entry"
  sub_dir="$OUTPUT_DIR/per_head/$name"
  mkdir -p "$sub_dir/per_benchmark"

  echo "[GPU $gpu] Starting $name ..."
  CUDA_VISIBLE_DEVICES="$gpu" python "$ENTRY" \
      --base-model-path "$BASE_MODEL_PATH" \
      --head "$name" "$path" \
      --benchmarks $BENCHMARKS \
      --output-dir "$sub_dir" \
      --skip-visualizations \
      $EXTRA_ARGS \
      > "$sub_dir/log.txt" 2>&1 &

  PIDS+=($!)
done

echo ""
echo "Waiting for ${#PIDS[@]} jobs: ${PIDS[*]}"

# ── Wait and report ──
FAILED=0
for i in "${!PIDS[@]}"; do
  read -r name gpu path <<< "${HEADS[$i]}"
  if wait "${PIDS[$i]}"; then
    echo "  ✓ $name (GPU $gpu) finished"
  else
    echo "  ✗ $name (GPU $gpu) FAILED — see $OUTPUT_DIR/per_head/$name/log.txt"
    FAILED=1
  fi
done

if [ "$FAILED" -eq 1 ]; then
  echo "Some jobs failed. Check logs above."
  exit 1
fi

# ── Merge traces ──
echo ""
echo "Merging results ..."
mkdir -p "$OUTPUT_DIR/per_benchmark"

> "$OUTPUT_DIR/detailed_traces.jsonl"
for entry in "${HEADS[@]}"; do
  read -r name gpu path <<< "$entry"
  cat "$OUTPUT_DIR/per_head/$name/detailed_traces.jsonl" >> "$OUTPUT_DIR/detailed_traces.jsonl"
  # Copy per-benchmark stats
  cp "$OUTPUT_DIR/per_head/$name/per_benchmark/"*.json "$OUTPUT_DIR/per_benchmark/" 2>/dev/null || true
done

# ── Generate markdown summary table ──
SUMMARY_OUTPUT_DIR="$OUTPUT_DIR" python3 << 'PYEOF'
import json, os

output_dir = os.environ.get("SUMMARY_OUTPUT_DIR", "entropy_analysis_results")

traces = []
with open(f"{output_dir}/detailed_traces.jsonl") as f:
    for line in f:
        traces.append(json.loads(line))

heads = sorted(set(t["head_name"] for t in traces))
benchmarks = sorted(set(t["benchmark"] for t in traces))

output = []
output.append("# EA Entropy Analysis")
output.append("")
output.append("| Checkpoint | Benchmark | Accepted | Rejected | Δ |")
output.append("|---|---|---|---|---|")

for head in heads:
    for bench in benchmarks:
        ea_acc = [r["ea_entropy"] for t in traces if t["head_name"]==head and t["benchmark"]==bench for r in t["records"] if r.get("ea_entropy") and r["accepted"]]
        ea_rej = [r["ea_entropy"] for t in traces if t["head_name"]==head and t["benchmark"]==bench for r in t["records"] if r.get("ea_entropy") and not r["accepted"]]
        m_acc = sum(ea_acc)/len(ea_acc) if ea_acc else 0
        m_rej = sum(ea_rej)/len(ea_rej) if ea_rej else 0
        output.append(f"| {head} | {bench} | {m_acc:.4f} | {m_rej:.4f} | {m_rej-m_acc:+.4f} |")

print("\n")
for line in output:
    print(line)

with open(f"{output_dir}/ea_entropy_table.md", "w") as f:
    f.write("\n".join(output) + "\n")

print(f"\nTable saved to: {output_dir}/ea_entropy_table.md")
PYEOF

echo ""
echo "=============================================="
echo "Done! Results saved to: $OUTPUT_DIR/"
echo "=============================================="
