#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# GSM8K-specific configuration
DATASET="openai/gsm8k"
PRESET="gsm8k"
LIMIT=""
OUTDIR="./sharegpt_data"

show_help() {
    cat <<EOF
Convert GSM8K dataset to ShareGPT JSON format (train and test splits)

Usage: bash hf_to_sharegpt.sh [options]

Options:
  --outdir <dir>              Output directory (default: ./sharegpt_data)
  --limit <n>                 Limit number of records to convert per split
  -h, --help                  Show this help message

Examples:
  bash hf_to_sharegpt.sh --outdir ./data
  bash hf_to_sharegpt.sh --outdir ./data --limit 100

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --outdir)
            OUTDIR="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

mkdir -p "$OUTDIR"

echo "Converting GSM8K dataset to ShareGPT format..."
echo "Dataset: $DATASET"
echo "Preset: $PRESET"
echo "Output directory: $OUTDIR"
echo ""

# Convert train split
TRAIN_OUTPUT="${OUTDIR}/openai_gsm8k_train.json"
TRAIN_CMD="python ${REPO_ROOT}/scripts/hf_to_sharegpt.py"
TRAIN_CMD="$TRAIN_CMD --dataset $DATASET"
TRAIN_CMD="$TRAIN_CMD --split train"
TRAIN_CMD="$TRAIN_CMD --preset $PRESET"
TRAIN_CMD="$TRAIN_CMD --out $TRAIN_OUTPUT"
[[ -n "$LIMIT" ]] && TRAIN_CMD="$TRAIN_CMD --limit $LIMIT"

echo "Converting train split..."
eval "$TRAIN_CMD"

# Convert test split
TEST_OUTPUT="${OUTDIR}/openai_gsm8k_test.json"
TEST_CMD="python ${REPO_ROOT}/scripts/hf_to_sharegpt.py"
TEST_CMD="$TEST_CMD --dataset $DATASET"
TEST_CMD="$TEST_CMD --split test"
TEST_CMD="$TEST_CMD --preset $PRESET"
TEST_CMD="$TEST_CMD --out $TEST_OUTPUT"
[[ -n "$LIMIT" ]] && TEST_CMD="$TEST_CMD --limit $LIMIT"

echo ""
echo "Converting test split..."
eval "$TEST_CMD"

echo ""
echo "GSM8K conversion complete!"
echo "Train data: $TRAIN_OUTPUT"
echo "Test data: $TEST_OUTPUT"
