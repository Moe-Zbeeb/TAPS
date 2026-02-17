#!/usr/bin/env bash
set -euo pipefail

# One-shot pipeline using the helper shell scripts:
# 1) Build ShareGPT + features from HF dataset (train split)
# 2) Train HASS for 10 epochs on the train split features
# 3) Evaluate acceptance (and speed) on the test split

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'USAGE'
Run HASS train+eval using the repo's shell helpers.

Required:
  --dataset ID              HF dataset id (e.g., openai/gsm8k)
  --model-path PATH         Base LLM path or HF id
  --model-family NAME       llama3 | llama2chat
  --ea-ckpt DIR             Initial EA/HASS weights directory
  --workdir DIR             Working dir (will contain feats/, checkpoints/, logs/)

Optional:
  --subset NAME             HF subset/config (e.g., main)
  --limit N                 Limit train examples for feature build
  --preset NAME             Mapping preset: gsm8k|alpaca|dolly|passthrough
  --human-field KEY         Custom human field (if no preset)
  --assistant-field KEY     Custom assistant field (if no preset)
  --gpu-index IDX           CUDA GPU for feature/eval, default: 0
  --train-gpus LIST         CUDA devices for training (default: 0)
  --configpath DIR          HASS config dir (default: /Users/mohammadzbeeb/Research/Speculative Decoding/DHASS/train/EAGLE-LLaMA3-Instruct-8B)
  --bs N                    Batch size per device (default: 2)
  --lr F                    Learning rate (default: 1e-5)
  --topk N                  Top-k loss focus (default: 10)
  --topk-w F                Weight for top-k loss (default: 1)
  --forward-num-total N     Draft refinement steps (default: 3)

Notes:
- Always trains for 10 epochs on the train split features.
- Always evaluates acceptance/speed on the test split.
USAGE
}

# Defaults
DATASET=""; SUBSET=""; LIMIT=""; PRESET=""; HUMAN_FIELD=""; ASSISTANT_FIELD=""
BASE_MODEL=""; MODEL_FAMILY=""; EA_CKPT=""; CONFIGPATH="/Users/mohammadzbeeb/Research/Speculative Decoding/DHASS/train/EAGLE-LLaMA3-Instruct-8B"; WORKDIR=""
GPU_INDEX="0"; TRAIN_GPUS="0"; BS=2; LR=1e-5; TOPK=10; TOPK_W=1; FWD_NUM=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --subset) SUBSET="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --human-field) HUMAN_FIELD="$2"; shift 2 ;;
    --assistant-field) ASSISTANT_FIELD="$2"; shift 2 ;;
    --model-path) BASE_MODEL="$2"; shift 2 ;;
    --model-family) MODEL_FAMILY="$2"; shift 2 ;;
    --ea-ckpt) EA_CKPT="$2"; shift 2 ;;
    --configpath) CONFIGPATH="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --train-gpus) TRAIN_GPUS="$2"; shift 2 ;;
    --bs) BS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --topk-w) TOPK_W="$2"; shift 2 ;;
    --forward-num-total) FWD_NUM="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[[ -z "$DATASET" || -z "$BASE_MODEL" || -z "$MODEL_FAMILY" || -z "$EA_CKPT" || -z "$WORKDIR" ]] && { usage; exit 1; }
case "$MODEL_FAMILY" in
  llama3|llama2chat) ;;
  *) echo "--model-family must be llama3 or llama2chat"; exit 1 ;;
esac

mkdir -p "$WORKDIR" "$WORKDIR/logs" "$WORKDIR/feats" "$WORKDIR/checkpoints"

echo "[1/3] Building ShareGPT + features on train split"
bash "${REPO_ROOT}/scripts/build_hass_dataset.sh" \
  --dataset "$DATASET" \
  ${SUBSET:+--subset "$SUBSET"} \
  --split train \
  ${LIMIT:+--limit "$LIMIT"} \
  ${PRESET:+--preset "$PRESET"} \
  ${HUMAN_FIELD:+--human-field "$HUMAN_FIELD"} \
  ${ASSISTANT_FIELD:+--assistant-field "$ASSISTANT_FIELD"} \
  --model-path "$BASE_MODEL" \
  --model-family "$MODEL_FAMILY" \
  --outdir "$WORKDIR/feats" \
  --gpu-index "$GPU_INDEX" 2>&1 | tee "$WORKDIR/logs/01_build_data.log"

echo "[2/3] Training HASS for 10 epochs on train features"
bash "${REPO_ROOT}/scripts/train_hass.sh" \
  --base-model "$BASE_MODEL" \
  --ea-ckpt "$EA_CKPT" \
  --data-dir "$WORKDIR/feats" \
  --epochs 10 \
  --outdir "$WORKDIR/checkpoints" \
  --configpath "$CONFIGPATH" \
  --gpus "$TRAIN_GPUS" \
  --bs "$BS" \
  --lr "$LR" \
  --topk "$TOPK" \
  --topk-w "$TOPK_W" \
  --fwd "$FWD_NUM" 2>&1 | tee "$WORKDIR/logs/02_train.log"

# Find latest checkpoint dir named state_*
EA_CKPT_DIR=$(ls -1d "$WORKDIR"/checkpoints/state_* 2>/dev/null | sort -V | tail -n1 || true)
if [[ -z "$EA_CKPT_DIR" ]]; then
  echo "Warning: no state_* checkpoint found after training. Using provided --ea-ckpt for eval."
  EA_CKPT_DIR="$EA_CKPT"
fi
echo "EA weights path: $EA_CKPT_DIR"

echo "[3/3] Evaluating acceptance/speed on test split"
bash "${REPO_ROOT}/scripts/eval_hass.sh" \
  --base-model "$BASE_MODEL" \
  --ea-ckpt "$EA_CKPT_DIR" \
  --model-family "$MODEL_FAMILY" \
  --dataset "$DATASET" \
  ${SUBSET:+--subset "$SUBSET"} \
  --split test \
  --gpu-index "$GPU_INDEX" 2>&1 | tee "$WORKDIR/logs/03_eval.log"

echo "Pipeline finished. Check logs in $WORKDIR/logs and outputs under data/<bench>/."
