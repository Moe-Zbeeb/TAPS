#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'USAGE'
Evaluate EA/HASS vs baseline on a specified dataset.

Required:
  --base-model PATH        Base LLM path or HF id
  --ea-ckpt DIR            EA/HASS weights directory (state_* or a folder with model file)
  --model-family NAME      llama3 | llama2chat (selects eval scripts and prompt template)
  --dataset ID             HF dataset id used to build questions (e.g., openai/gsm8k)

Optional:
  --subset NAME            HF subset/config (e.g., main)
  --split NAME             Split to use (default: test)
  --limit N                Limit number of questions
  --bench-name NAME        Folder under data/ for questions (default: derived from dataset)
  --gpu-index IDX          CUDA device to use (default: 0)
  --temperature F          Decoding temperature (default: 0.0)
  --total-token N          HASS tree total tokens (default: 60)
  --depth N                HASS tree depth (default: 5)
  --top-k N                HASS candidate top-k (default: 10)
  --model-id-prefix STR    Prefix for output model id (default: eval)

Example:
  bash scripts/eval_hass.sh \
    --base-model ../models/meta-llama/Meta-LLaMA-3-8B-Instruct \
    --ea-ckpt checkpoints/hass_llama3/state_10 \
    --model-family llama3 \
    --dataset openai/gsm8k --subset main --split test \
    --bench-name gsm8k --gpu-index 0
USAGE
}

BASE_MODEL=""; EA_CKPT=""; MODEL_FAMILY=""; DATASET=""; SUBSET=""; SPLIT="test"; LIMIT=""
BENCH_NAME=""; GPU_INDEX="0"; TEMPERATURE=0.0; TOTAL_TOKEN=60; DEPTH=5; TOPK=10; MODEL_ID_PREFIX="eval"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --ea-ckpt) EA_CKPT="$2"; shift 2 ;;
    --model-family) MODEL_FAMILY="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --subset) SUBSET="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --bench-name) BENCH_NAME="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --total-token) TOTAL_TOKEN="$2"; shift 2 ;;
    --depth) DEPTH="$2"; shift 2 ;;
    --top-k) TOPK="$2"; shift 2 ;;
    --model-id-prefix) MODEL_ID_PREFIX="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[[ -z "$BASE_MODEL" || -z "$EA_CKPT" || -z "$MODEL_FAMILY" || -z "$DATASET" ]] && { usage; exit 1; }
case "$MODEL_FAMILY" in
  llama3|llama2chat) ;;
  *) echo "--model-family must be llama3 or llama2chat"; exit 1 ;;
esac

# 1) Create data/<bench-name>/question.jsonl from HF dataset
if [[ -z "$BENCH_NAME" ]]; then
  # derive a short bench name from dataset id
  BENCH_NAME=$(basename "$DATASET")
fi

QUEST_DIR="$REPO_ROOT/data/$BENCH_NAME"
mkdir -p "$QUEST_DIR"

python - <<PY
import json
from datasets import load_dataset
from pathlib import Path
import itertools

ds = load_dataset("$DATASET"${SUBSET:+, '$SUBSET'})
split = "$SPLIT" if "$SPLIT" in ds else list(ds.keys())[0]
data = ds[split]
if "$LIMIT":
    data = data.select(range(min(int("$LIMIT"), len(data))))

out_path = Path("$QUEST_DIR") / "question.jsonl"
with out_path.open("w") as f:
    for i, ex in enumerate(data):
        q = None
        for key in ("question", "instruction", "input"):
            if key in ex and ex[key]:
                q = str(ex[key]).strip(); break
        if q is None:
            # take the first non-empty string field
            for k, v in ex.items():
                if isinstance(v, str) and v.strip():
                    q = v.strip(); break
        if q is None:
            q = ""
        obj = {"question_id": i, "category": "$BENCH_NAME", "turns": [q]}
        f.write(json.dumps(obj) + "\n")
print("Wrote", out_path)
PY

# 2) Run HASS (EA) decoding and baseline decoding
if [[ "$MODEL_FAMILY" == "llama3" ]]; then
  HASS_GEN=evaluation.gen_ea_answer_llama3chat
  BASE_GEN=evaluation.gen_baseline_answer_llama3chat
  EXTRA_ARGS=( --total-token "$TOTAL_TOKEN" --depth "$DEPTH" --top-k "$TOPK" )
else
  HASS_GEN=evaluation.gen_ea_answer_llama2chat
  BASE_GEN=evaluation.gen_baseline_answer_llama2chat
  EXTRA_ARGS=( --total-token "$TOTAL_TOKEN" --depth "$DEPTH" --top-k "$TOPK" )
fi

MODEL_ID_HASS="${MODEL_ID_PREFIX}_${BENCH_NAME}_hass"
MODEL_ID_BASE="${MODEL_ID_PREFIX}_${BENCH_NAME}_naive"

cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES="$GPU_INDEX" python -m $HASS_GEN \
  --ea-model-path "$EA_CKPT" \
  --base-model-path "$BASE_MODEL" \
  --model-id "$MODEL_ID_HASS" \
  --bench-name "$BENCH_NAME" \
  --temperature "$TEMPERATURE" \
  --num-gpus-per-model 4 \
  --num-gpus-total 4 \
  "${EXTRA_ARGS[@]}"

# CUDA_VISIBLE_DEVICES="$GPU_INDEX" python -m $BASE_GEN \
#   --ea-model-path "$EA_CKPT" \
#   --base-model-path "$BASE_MODEL" \
#   --model-id "$MODEL_ID_BASE" \
#   --bench-name "$BENCH_NAME" \
#   --temperature "$TEMPERATURE" \
#   --num-gpus-per-model 4 \
#   --num-gpus-total 4

# 3) Report acceptance length and speedup
HASS_JSON="$REPO_ROOT/$BENCH_NAME/${MODEL_ID_HASS}-temperature-${TEMPERATURE}.jsonl"
BASE_JSON="$REPO_ROOT/$BENCH_NAME/${MODEL_ID_BASE}-temperature-${TEMPERATURE}.jsonl"

echo ""
echo "=== Acceptance length (HASS) ==="
python -m evaluation.acceptance_length --input_file "$HASS_JSON"



# echo ""
# echo "=== Speedup ratio (HASS vs baseline) ==="
# python -m evaluation.speed \
#   --model_path "$BASE_MODEL" \
#   --baseline_json "$BASE_JSON" \
#   --hass_json     "$HASS_JSON"

# echo ""
# echo "Done. Outputs saved under $BENCH_NAME/."
# 3.871924416187054
