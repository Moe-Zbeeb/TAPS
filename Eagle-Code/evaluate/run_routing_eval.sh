#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# Run EAGLE-2 entropy-based head routing evaluation.
# Loads all 4 EA heads (one per GPU 0-3), probes each per sample,
# routes to the lowest-entropy head, and generates with that head.
# ──────────────────────────────────────────────────────────────

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to your base model (e.g. /path/to/Meta-Llama-3-8B-Instruct)}"

mkdir -p results/routing

python -m eagle.evaluation.gen_routing_eval \
  --base-model-path "$BASE_MODEL" \
  --bench-name mt_bench gsm8k math_500 svamp \
  --output-dir results/routing \
  --total-token 60 --depth 8 --top-k 10 --temperature 0.0 \
  2>&1 | tee results/routing/routing_eval.log

echo ""
echo "Done. Results in results/routing/"
