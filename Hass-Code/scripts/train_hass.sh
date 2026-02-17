#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'USAGE'
Train HASS draft model with minimal required inputs.

Required:
  --base-model PATH     Base LLM path or HF id (for head + config)
  --ea-ckpt DIR         Initial HASS/EA weights directory (contains pytorch_model.bin or model.safetensors)
  --data-dir DIR        Directory of feature ckpts (from ge_data), e.g., feats/ or feats/0/
  --epochs N            Number of training epochs

Optional:
  --outdir DIR          Checkpoint output directory (default: checkpoints/hass_train)
  --configpath PATH     HASS config dir (default: /Users/mohammadzbeeb/Research/Speculative Decoding/DHASS/train/EAGLE-LLaMA3-Instruct-8B)
  --gpus LIST           CUDA devices for training, e.g., 0 or 0,1,2,3 (default: 0)
  --bs N                Batch size per device (default: 2)
  --lr F                Learning rate (default: 1e-5)
  --topk N              Top-k loss focus (default: 10)
  --topk-w F            Weight for top-k loss (default: 1)
  --fwd N               forward_num_total refinement steps (default: 3)

Example:
  bash scripts/train_hass.sh \
    --base-model ../models/meta-llama/Meta-LLaMA-3-8B-Instruct \
    --ea-ckpt ../models/HArmonizedSS/HASS-LLaMA3-Instruct-8B \
    --data-dir runs/llama3_gsm8k/feats \
    --epochs 10 \
    --outdir runs/llama3_gsm8k/checkpoints
USAGE
}

BASE_MODEL=""; EA_CKPT=""; DATA_DIR=""; EPOCHS=""
OUTDIR="checkpoints/hass_train"; CONFIGPATH="/Users/mohammadzbeeb/Research/Speculative Decoding/DHASS/train/EAGLE-LLaMA3-Instruct-8B"
GPUS="0"; BS=2; LR=1e-5; TOPK=10; TOPK_W=1; FWD=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --ea-ckpt) EA_CKPT="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --configpath) CONFIGPATH="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --bs) BS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --topk-w) TOPK_W="$2"; shift 2 ;;
    --fwd) FWD="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[[ -z "$BASE_MODEL" || -z "$EA_CKPT" || -z "$DATA_DIR" || -z "$EPOCHS" ]] && { usage; exit 1; }

# Resolve tmpdir: accept parent dir (with shard subfolders) or direct shard dir
TMPDIR="$DATA_DIR"
if [[ -d "$DATA_DIR/0" ]]; then
  TMPDIR="$DATA_DIR"
fi

mkdir -p "$OUTDIR" "$REPO_ROOT/logs"

# Count data points (all *.ckpt recursively)
DATA_NUM=$(find "$TMPDIR" -type f -name "*.ckpt" | wc -l | tr -d ' ')
if [[ "$DATA_NUM" == "0" ]]; then
  echo "No .ckpt files found under $TMPDIR"; exit 1
fi
echo "Training on $DATA_NUM feature files from $TMPDIR"

export WANDB_MODE=offline
export WANDB_DISABLED=true

cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch -m --mixed_precision=bf16 train.main_hass \
  --basepath "$BASE_MODEL" \
  --tmpdir   "$TMPDIR" \
  --cpdir    "$OUTDIR" \
  --configpath "$CONFIGPATH" \
  --epoch "$EPOCHS" \
  --lr "$LR" \
  --bs "$BS" \
  --topk "$TOPK" \
  --topk_w "$TOPK_W" \
  --forward_num_total "$FWD" \
  --ckpt_path "$EA_CKPT" \
  --data_num "$DATA_NUM" 2>&1 | tee "$REPO_ROOT/logs/train_hass.log"

echo "Training complete. Checkpoints saved under: $OUTDIR"
