#\!/bin/bash
set -e
BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to your base model (e.g. /path/to/Meta-Llama-3-8B-Instruct)}"
OUTPUT_DIR="results/routing_v2"
mkdir -p "$OUTPUT_DIR"
echo "=== EAGLE-2 Routing Evaluation ==="
python -m eagle.evaluation.gen_routing_eval --base-model-path "$BASE_MODEL" --bench-name mt_bench gsm8k math_500 svamp --output-dir "$OUTPUT_DIR" --total-token 60 --depth 8 --top-k 10 --temperature 0.0 2>&1 | tee "$OUTPUT_DIR/routing_eval.log"
echo "=== Results ==="
python3 scripts/print_routing_results.py "$OUTPUT_DIR"
