#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'USAGE'
Build HASS training features from a HF dataset and a base LLM.

Required:
  --dataset ID            HF dataset id (e.g., openai/gsm8k)
  --model-path PATH       Base LLM path or HF id
  --model-family NAME     One of: llama3, llama2chat
  --outdir DIR            Output directory for feature ckpts

Optional:
  --subset NAME           HF subset/config (e.g., main)
  --split NAME            Split (train|test|validation), default: train
  --limit N               Limit number of records
  --preset NAME           Mapping preset: gsm8k|alpaca|dolly|passthrough
  --human-field KEY       Custom human/source field (if no preset)
  --assistant-field KEY   Custom assistant/target field (if no preset)
  --gpu-index IDX         CUDA GPU index for feature extraction, default: 0
  --sharegpt-out FILE     Where to save ShareGPT JSON (default: <outdir>/sharegpt.json)
  --batch-size N          Batch size for feature extraction (ge_data), default: 1

Example:
  bash scripts/build_hass_dataset.sh \
    --dataset openai/gsm8k --subset main --split train --preset gsm8k \
    --model-path ../models/meta-llama/Meta-Llama-3-8B-Instruct \
    --model-family llama3 \
    --outdir data/gsm8k_hass_feats \
    --gpu-index 0
USAGE
}

DATASET=""
SUBSET=""
SPLIT="train"
LIMIT=""
PRESET=""
HUMAN_FIELD=""
ASSISTANT_FIELD=""
MODEL_PATH=""
MODEL_FAMILY=""
OUTDIR=""
GPU_INDEX="0"
SHAREGPT_OUT=""
BATCH_SIZE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --subset) SUBSET="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --human-field) HUMAN_FIELD="$2"; shift 2 ;;
    --assistant-field) ASSISTANT_FIELD="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --model-family) MODEL_FAMILY="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --sharegpt-out) SHAREGPT_OUT="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[[ -z "$DATASET" || -z "$MODEL_PATH" || -z "$MODEL_FAMILY" || -z "$OUTDIR" ]] && { usage; exit 1; }

case "$MODEL_FAMILY" in
  llama3|llama2chat) ;;
  *) echo "--model-family must be one of: llama3, llama2chat"; exit 1 ;;
esac

mkdir -p "$OUTDIR"
if [[ -z "$SHAREGPT_OUT" ]]; then
  SHAREGPT_OUT="${OUTDIR%/}/sharegpt.json"
fi

echo "[1/3] Converting HF dataset to ShareGPT: $DATASET ($SPLIT) -> $SHAREGPT_OUT"

PY_ARGS=(
  --dataset "$DATASET"
  --split "$SPLIT"
  --out "$SHAREGPT_OUT"
)
[[ -n "$SUBSET" ]] && PY_ARGS+=( --subset "$SUBSET" )
[[ -n "$LIMIT" ]] && PY_ARGS+=( --limit "$LIMIT" )
[[ -n "$PRESET" ]] && PY_ARGS+=( --preset "$PRESET" )
[[ -n "$HUMAN_FIELD" ]] && PY_ARGS+=( --human-field "$HUMAN_FIELD" )
[[ -n "$ASSISTANT_FIELD" ]] && PY_ARGS+=( --assistant-field "$ASSISTANT_FIELD" )

python "${REPO_ROOT}/scripts/hf_to_sharegpt.py" "${PY_ARGS[@]}"

echo "[2/3] Counting examples in ShareGPT JSON"
COUNT=$(python - "$SHAREGPT_OUT" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    obj = json.load(f)
print(len(obj))
PY
)

if [[ "$COUNT" == "" || "$COUNT" == "0" ]]; then
  echo "No examples found in $SHAREGPT_OUT"; exit 1
fi
echo "Found $COUNT examples"

echo "[3/3] Generating HASS features with base model: $MODEL_PATH"

GEN_SCRIPT=""
case "$MODEL_FAMILY" in
  llama3) GEN_SCRIPT="${REPO_ROOT}/ge_data/ge_data_all_llama3.py" ;;
  llama2chat) GEN_SCRIPT="${REPO_ROOT}/ge_data/ge_data_all_llama2chat.py" ;;
esac

GEN_ARGS=(
  --start 0
  --end "$COUNT"
  --index 0
  --gpu_index "$GPU_INDEX"
  --outdir "$OUTDIR"
  --data_path "$SHAREGPT_OUT"
  --model_path "$MODEL_PATH"
)
[[ -n "$BATCH_SIZE" ]] && GEN_ARGS+=( --batch-size "$BATCH_SIZE" )

CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
python "$GEN_SCRIPT" "${GEN_ARGS[@]}"

echo "Done. Feature files saved under: ${OUTDIR}/0"
