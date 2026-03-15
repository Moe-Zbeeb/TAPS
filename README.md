# TAPS: Task-Aware Proposal Distributions for Speculative Sampling

![Speculative decoding pipeline](Taps-draft1/figures/illustrative.png)
![Merged-tree verification](Taps-draft1/figures/mergdTrees.png)

Date: March 2026

Authors: Mohamad Zbib, Mohamad Bazzi, Ammar Mohanna, Bernard Ghanem*, Hasan Abed Al Kader Hammoud* (*equal advising authors)

Affiliations: King Abdullah University of Science and Technology (KAUST), American University of Beirut (AUB)

Contacts: [mbz02@mail.aub.edu](mailto:mbz02@mail.aub.edu), [hasanabedalkader.hammoud@kaust.edu.sa](mailto:hasanabedalkader.hammoud@kaust.edu.sa)

---

## What is TAPS?
TAPS studies how the training distribution of lightweight draft models shapes speculative decoding quality. Using Meta-Llama-3-8B-Instruct as the verifier and lightweight LLaMA-style drafters (HASS and EAGLE-2), the project compares single-domain training, mixed-domain training, and inference-time composition strategies (confidence routing and merged-tree verification).

## Key findings (paper highlights)
- Task-specific drafts specialize strongly: ShareGPT-trained drafts lead on MT-Bench (e.g., HASS 3.98 vs. 2.90 acceptance length for MathInstruct), while MathInstruct drafts dominate GSM8K/MATH-500/SVAMP.
- Mixed-data training broadens coverage but is not monotonic: under HASS, Mixed 70k+70k is best at temperature 0 (avg 5.18) but drops at temperature 1 (3.69), where Mixed 35k+35k is steadier.
- Composition beats weight merging: weight-space averaging is weakest (≈2.4–2.6); confidence routing improves to ~4.8, and merged-tree verification is strongest (HASS 5.11, EAGLE-2 5.02 at temperature 0).
- Routing signals: confidence cleanly separates workloads (e.g., MathInstruct chosen for 90.8% of GSM8K vs. ShareGPT for 81.2% of MT-Bench), while entropy is mainly diagnostic; acceptance declines with depth but specialization persists.

## Repository layout
- `Taps-draft1/` – LaTeX source for the manuscript (`main.tex` + `sections/`, figures, macros). Build with `latexmk -pdf -interaction=nonstopmode main.tex`.
- `Hass-Code/` – HASS-based drafter training and evaluation scripts (feature building, training, confidence routing, merged-tree verification).
- `Eagle-Code/` – EAGLE-2/3-style drafter code, training configs, evaluation utilities, and a Gradio demo web UI.
- `docs/` – supporting PDFs (routing and merged-tree technical reports).

## Assets
- Model weights: https://huggingface.co/collections/zbeeb/taps
- Datasets: https://huggingface.co/datasets/zbeeb/TAPS-Datasets
- Paper source: `Taps-draft1/main.tex` (build to obtain the full PDF).

## Setup
- Python 3.10+ recommended; GPU with enough memory for Meta-Llama-3-8B-Instruct and draft checkpoints (paper experiments used 4×A100).
- Install per sub-project to avoid dependency clashes.

HASS install:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r Hass-Code/requirements.txt
```

EAGLE install:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r Eagle-Code/requirements.txt
pip install -e Eagle-Code
```

## Usage (HASS)
- One-shot train + eval pipeline (feature build → 10-epoch train → acceptance eval):
```bash
bash Hass-Code/scripts/run_hass_pipeline.sh \
  --dataset <hf_dataset_id> \
  --model-path <base_llama3_or_llama2chat> \
  --model-family llama3 \
  --ea-ckpt <init_ea_weights_dir> \
  --workdir <workdir> \
  --preset <gsm8k|alpaca|dolly|passthrough> \
  --gpu-index 0 --train-gpus 0
```
- Merged-tree dual-head evaluation (runs MT-Bench/GSM8K/MATH-500/SVAMP in parallel across 4 GPUs):
```bash
BASE_MODEL_PATH=/path/to/Meta-Llama-3-8B-Instruct \
EA_MODEL_PATH1=/path/to/MathInstruct_ckpt \
EA_MODEL_PATH2=/path/to/ShareGPT_ckpt \
RESULTS_DIR=results/merged \
bash Hass-Code/scripts/run_merged_tree.sh
```
  Environment variables also control `TOTAL_TOKEN`, `DEPTH`, `TOP_K`, `TEMPERATURE`, and `MAX_NEW_TOKEN`.
- Other helpers: `scripts/run_routed.sh` (confidence routing), `scripts/run_entropy_analysis.sh` and `scripts/run_full_entropy_analysis.sh` (diagnostics), `scripts/run_merge_sweep.sh` (weight averaging sweep).

## Usage (EAGLE)
- Launch demo web UI:
```bash
cd Eagle-Code
python eagle/application/webui.py --base-model-path <base> --ea-model-path <eagle3_ckpt> --model-type vicuna [--load-in-4bit|--load-in-8bit]
```
- Train a drafter with a config:
```bash
cd Eagle-Code
python -m eagle.train.main --basepath <hf_or_local_model> \
  --configpath eagle/train/vicuna_13B_config.json \
  --tmpdir <data_dir> --cpdir <checkpoint_dir>
```
  Use `main_deepspeed.py` for multi-GPU coordination.
- Merged-tree dual-head evaluation (4 GPUs, per-benchmark workers):
```bash
cd Eagle-Code
BASE_MODEL_PATH=/path/to/Meta-Llama-3-8B-Instruct \
EA_MODEL_PATH1=checkpoints/Eagle-MathInstruct_20epochs \
EA_MODEL_PATH2=checkpoints/Eagle-ShareGPT_20epochs \
bash scripts/run_merged_tree.sh
```
- Entropy-based routing evaluation (loads four EA heads, routes to lowest entropy):
```bash
cd Eagle-Code
BASE_MODEL=/path/to/Meta-Llama-3-8B-Instruct bash scripts/run_routing_eval.sh
```
- Additional analyses: `scripts/run_confidence_analysis*.sh`, `scripts/run_entropy_analysis.sh`, `scripts/run_dual_head_eval.sh`.

## Reproducing paper numbers (overview)
- Verifier: Meta-Llama-3-8B-Instruct.
- Draft architectures: HASS and EAGLE-2 with a single transformer layer (≈0.8B params) sharing tokenizer/vocabulary with the verifier.
- Training data: MathInstruct (reasoning), ShareGPT (chat), and mixed 35k+35k / 70k+70k variants; 20 epochs, lr 3e-5, batch size 8, grad accumulation 1.
- Benchmarks: MT-Bench, GSM8K, MATH-500, SVAMP; temperatures 0 and 1.
- Metric: acceptance length (average accepted draft tokens per verifier call) under the lossless speculative decoding constraint.

## Citation
If you use TAPS, please cite:
```bibtex
@article{zbib2026taps,
  title={TAPS: Task Aware Proposal Distributions for Speculative Sampling},
  author={Zbib, Mohamad and Bazzi, Mohamad and Mohanna, Ammar and Ghanem, Bernard and Hammoud, Hasan Abed Al Kader},
  year={2026},
  note={Technical report}
}
```

## License
- `Eagle-Code` is released under the Apache 2.0 License (see `Eagle-Code/LICENSE`).
- HASS scripts inherit their upstream licenses; add your own license file before redistribution.
