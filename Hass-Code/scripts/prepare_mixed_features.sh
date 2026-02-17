#!/bin/bash

# Configuration
MIXED_DATA_DIR="${REPO_ROOT:-./}/mixed_train_70k"
OUTPUT_DIR="${REPO_ROOT:-./}/runs/mixed_train_70k"
SHAREGPT_JSON="${OUTPUT_DIR}/mixed_train_sharegpt.json"
FEATS_DIR="${OUTPUT_DIR}/feats/0"
CHECKPOINTS_DIR="${OUTPUT_DIR}/checkpoints"
MODEL_PATH="${BASE_MODEL:?Set BASE_MODEL to your base model (e.g. /path/to/Meta-Llama-3-8B-Instruct)}"

# GPU configuration for feature generation
GPU_INDICES="0,1,2,3"
BATCH_SIZE=4  # Adjust based on your GPU memory
START_IDX=0
END_IDX=68600  # Total samples in your dataset

echo "=========================================="
echo "HASS Mixed Data Feature Generation Pipeline"
echo "=========================================="
echo ""

# Step 1: Create output directories
echo "[Step 1] Creating output directories..."
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${FEATS_DIR}"
mkdir -p "${CHECKPOINTS_DIR}"
echo "✓ Directories created"
echo ""

# Step 2: Convert to ShareGPT format
echo "[Step 2] Converting mixed data to ShareGPT format..."
if [ -f "${SHAREGPT_JSON}" ]; then
    echo "⚠ ShareGPT JSON already exists at: ${SHAREGPT_JSON}"
    read -p "Do you want to regenerate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 ${REPO_ROOT:-./}/scripts/convert_mixed_to_sharegpt.py \
            --input_dir "${MIXED_DATA_DIR}" \
            --output_file "${SHAREGPT_JSON}"
    fi
else
    python3 ${REPO_ROOT:-./}/scripts/convert_mixed_to_sharegpt.py \
        --input_dir "${MIXED_DATA_DIR}" \
        --output_file "${SHAREGPT_JSON}"
fi
echo "✓ Conversion complete"
echo ""

# Step 3: Count examples
echo "[Step 3] Counting examples..."
TOTAL_EXAMPLES=$(python3 -c "import json; data = json.load(open('${SHAREGPT_JSON}')); print(len(data))")
echo "Total examples: ${TOTAL_EXAMPLES}"
echo ""

# Step 4: Generate features using base model
echo "[Step 4] Generating features from base model..."
echo "This will extract hidden states from all layers of the base model."
echo "Configuration:"
echo "  - Model: ${MODEL_PATH}"
echo "  - GPUs: ${GPU_INDICES}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Samples: ${START_IDX} to ${END_IDX}"
echo "  - Output: ${FEATS_DIR}"
echo ""

read -p "Start feature generation? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 -m ge_data.ge_data_all_llama3 \
        --start ${START_IDX} \
        --end ${END_IDX} \
        --gpu_index ${GPU_INDICES//,/ } \
        --batch_size ${BATCH_SIZE} \
        --outdir "${FEATS_DIR}" \
        --data_path "${SHAREGPT_JSON}" \
        --model_path "${MODEL_PATH}"

    echo "✓ Feature generation complete"
else
    echo "⚠ Skipping feature generation"
fi
echo ""

# Step 5: Display training command
echo "=========================================="
echo "Feature preparation complete!"
echo "=========================================="
echo ""
echo "To train the HASS model, run:"
echo ""
echo "CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch -m --mixed_precision=bf16 train.main_hass \\"
echo "    --basepath ${MODEL_PATH} \\"
echo "    --tmpdir ${FEATS_DIR} \\"
echo "    --cpdir ${CHECKPOINTS_DIR} \\"
echo "    --configpath ${REPO_ROOT:-./}/train/EAGLE-LLaMA3-Instruct-8B \\"
echo "    --epoch 20"
echo ""
