
📄 **original Paper:** [arXiv:2408.15766](https://arxiv.org/abs/2408.15766)

## Quick Start (GSM8K)

The scripts now natively handle GSM8K via ShareGPT-formatted data. Below are copy‑pasteable commands to prepare data, build features, train, and evaluate on GSM8K.

### 1) Install requirements

```bash
pip install -r requirements.txt
```

### 2) Convert GSM8K to ShareGPT JSON

Option A — convenience script (produces both train and test files):

```bash
bash scripts/hf_to_sharegpt.sh --outdir data
```

Option B — direct calls (explicit splits):

```bash
# Train split
python scripts/hf_to_sharegpt.py \
  --dataset openai/gsm8k --split train --preset gsm8k \
  --out data/gsm8k_sharegpt_train.json

# Test split
python scripts/hf_to_sharegpt.py \
  --dataset openai/gsm8k --split test --preset gsm8k \
  --out data/gsm8k_sharegpt_test.json
```

Notes:
- `--preset gsm8k` maps `question -> human`, `answer -> gpt`.
- The converter auto‑sets `--subset main` for `openai/gsm8k` if omitted.

### 3) Build HASS features on GSM8K (train split)

```bash
bash scripts/build_hass_dataset.sh \
  --dataset openai/gsm8k --subset main --split train --preset gsm8k \
  --model-path /path/to/Meta-Llama-3-8B-Instruct \
  --model-family llama3 \
  --outdir runs/gsm8k_llama3/feats \
  --gpu-index 0
```

This produces a ShareGPT JSON under the outdir and ge_data feature ckpts under `runs/gsm8k_llama3/feats/`.

### 4) Train HASS on GSM8K features

```bash
bash scripts/train_hass.sh \
  --base-model /path/to/Meta-Llama-3-8B-Instruct \
  --ea-ckpt   /path/to/HASS-LLaMA3-Instruct-8B \
  --data-dir  runs/gsm8k_llama3/feats \
  --epochs    10 \
  --outdir    runs/gsm8k_llama3/checkpoints \
  --configpath train/EAGLE-LLaMA3-Instruct-8B \
  --gpus      0,1,2,3
```

For single GPU training, use `--gpus 0`.

### 4b) Train HASS from scratch (no pretrained EA checkpoint)

```bash
bash scripts/train_hass_scratch.sh \
  --base-model /path/to/Meta-Llama-3-8B-Instruct \
  --data-dir  runs/gsm8k_llama3/feats \
  --epochs    40 \
  --outdir    runs/gsm8k_llama3/checkpoints \
  --configpath train/EAGLE-LLaMA3-Instruct-8B \
  --gpus      0,1,2,3
```

For single GPU training, use `--gpus 0`.

### 5) Evaluate on GSM8K test split

```bash
bash scripts/eval_hass.sh \
  --base-model /path/to/Meta-Llama-3-8B-Instruct \
  --ea-ckpt    runs/gsm8k_llama3/checkpoints/state_10 \
  --model-family llama3 \
  --dataset openai/gsm8k --subset main --split test \
  --bench-name gsm8k --gpu-index 0
```

If `state_10` does not exist, point `--ea-ckpt` to the latest `state_*` under `runs/gsm8k_llama3/checkpoints/` or to a provided HASS weights folder.

### 6) One‑shot GSM8K pipeline (build + train + eval)

```bash
bash scripts/run_hass_pipeline.sh \
  --dataset openai/gsm8k --subset main \
  --model-path /path/to/Meta-Llama-3-8B-Instruct \
  --model-family llama3 \
  --ea-ckpt /path/to/HASS-LLaMA3-Instruct-8B \
  --configpath train/EAGLE-LLaMA3-Instruct-8B \
  --workdir runs/gsm8k_llama3 \
  --gpu-index 0 --train-gpus 0,1,2,3
```

For single GPU training, use `--train-gpus 0`.

> Tip: Run any script with `-h` to see all available options.

## Reference

```bibtex
@inproceedings{zhang2025learning,
  title={Learning Harmonized Representations for Speculative Sampling},
  author={Zhang, Lefan and Wang, Xiaodan and Huang, Yanhua and Xu, Ruiwen},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```
