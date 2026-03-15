# TAPS: Task-Aware Proposal Distributions for Speculative Sampling

**Research-grade README · March 2026**

Mohamad Zbib, Mohamad Bazzi, Ammar Mohanna, **Bernard Ghanem**\*, **Hasan Abed Al Kader Hammoud**\*  (\*equal advising)  
KAUST · AUB — Contacts: [mbz02@mail.aub.edu](mailto:mbz02@mail.aub.edu), [hasanabedalkader.hammoud@kaust.edu.sa](mailto:hasanabedalkader.hammoud@kaust.edu.sa)

---

## Quick links
- Paper source: `Taps-draft1/main.tex` (build with `latexmk -pdf -interaction=nonstopmode main.tex`)
- Weights: https://huggingface.co/collections/zbeeb/taps
- Datasets: https://huggingface.co/datasets/zbeeb/TAPS-Datasets
- Code: `Hass-Code/` (HASS drafts), `Eagle-Code/` (EAGLE-2/3 drafts), supporting PDFs (`docs/ROUTING_TECHNICAL_REPORT.pdf`, `docs/MERGED_TREE_DUAL_DRAFTING.pdf`)

## Contents
- [Overview](#overview)
- [Abstract](#abstract)
- [Highlights](#highlights)
- [Figures](#figures)
- [Results snapshot](#results-snapshot)
- [Repository map](#repository-map)
- [Setup](#setup)
- [Run recipes — HASS](#run-recipes--hass)
- [Run recipes — EAGLE](#run-recipes--eagle)
- [Reproducing the paper](#reproducing-the-paper)
- [Citation](#citation)
- [License](#license)

---

## Overview
TAPS studies how the draft training distribution shapes speculative decoding quality. Using Meta-Llama-3-8B-Instruct as the verifier and lightweight LLaMA-style drafters (HASS and EAGLE-2, ~0.8B params, shared tokenizer), the work compares single-domain training, mixed-domain training, and inference-time composition (confidence routing, merged-tree verification).

### At a glance
- **Workloads:** MT-Bench (chat), GSM8K, MATH-500, SVAMP (reasoning).
- **Metric:** acceptance length (avg accepted draft tokens per verifier call), lossless speculative decoding.
- **Verifier:** Meta-Llama-3-8B-Instruct.
- **Drafters:** HASS and EAGLE-2, single-layer decoder (~0.8B params), shared tokenizer with verifier.
- **Compute:** single node, 4×A100 (paper experiments).
- **Core question:** How much does draft training data—and how we combine specialists—shape speculative decoding quality?

## Abstract
> Speculative decoding speeds up autoregressive generation by letting a lightweight drafter propose tokens that a larger verifier checks in parallel. We study how much draft quality depends on the training distribution using HASS and EAGLE-2 drafts trained on MathInstruct, ShareGPT, and mixed variants, evaluated on MT-Bench, GSM8K, MATH-500, and SVAMP. Task-matched drafts specialize; mixed data aids robustness but is not uniformly dominant across temperatures. Among composition strategies, weight averaging underperforms, confidence routing improves, and merged-tree verification attains the highest acceptance length. Confidence is a stronger routing signal than entropy. Results show speculative decoding quality hinges on both draft architecture and the alignment between draft training data and downstream workload.

## Highlights
- **Task-aware specialization:** ShareGPT drafts lead MT-Bench (e.g., HASS 3.98 vs. 2.90 for MathInstruct), while MathInstruct drafts dominate GSM8K/MATH-500/SVAMP.
- **Mixed data is nuanced:** Mixed 70k+70k (HASS) peaks at temperature 0 with average acceptance length 5.18 but drops at temperature 1 (3.69), where Mixed 35k+35k is steadier.
- **Composition wins:** Weight-space averaging is weakest (≈2.4–2.6). Confidence routing reaches ~4.8 average acceptance length; merged-tree verification is strongest (HASS 5.11, EAGLE-2 5.02 at temperature 0).
- **Routing signal:** Confidence cleanly separates workloads (MathInstruct chosen for 90.8% of GSM8K; ShareGPT for 81.2% of MT-Bench). Entropy is diagnostic but weaker for routing.

## Figures
<table>
  <tr>
    <td align="center">
      <img src="Taps-draft1/figures/illustrative.png" alt="Speculative decoding pipeline with draft proposals verified by the target model" height="185">
    </td>
    <td align="center">
      <img src="Taps-draft1/figures/mergdTrees.png" alt="Merged-tree verification combining MathInstruct and ShareGPT draft trees" height="185">
    </td>
  </tr>
  <tr>
    <td align="center"><em>Speculative decoding pipeline.</em></td>
    <td align="center"><em>Merged-tree verification packs MathInstruct and ShareGPT trees for one-pass verification.</em></td>
  </tr>
</table>

<p align="center">
  <img src="Taps-draft1/figures/averaging_effect.png" alt="Interpolation sweep for checkpoint averaging between MathInstruct and ShareGPT drafts" width="780">
  <br>
  <em>Checkpoint averaging is unstable across interpolation weights and remains weaker than inference-time composition (source: 5434×1294 px).</em>
</p>

<table>
  <tr>
    <td align="center">
      <img src="Taps-draft1/figures/depth_tables_grid.png" alt="Acceptance by speculative depth across backbones, benchmarks, and temperatures" width="480">
    </td>
    <td align="center">
      <img src="Taps-draft1/figures/draft_entropy_eagle2.png" alt="Accepted vs. rejected token entropy for EAGLE-2 drafts" width="480">
    </td>
  </tr>
  <tr>
    <td align="center"><em>Acceptance declines with depth but specialization persists (source: 4046×3665 px).</em></td>
    <td align="center"><em>Rejected tokens show higher entropy; confidence remains the stronger routing signal (source: 7059×1554 px).</em></td>
  </tr>
</table>

## Results snapshot
Average acceptance length (higher is better), temperature 0:

| Variant (HASS) | Avg | Variant (EAGLE-2) | Avg |
| --- | --- | --- | --- |
| Mixed 70k+70k | 5.18 | Mixed 70k+70k | 4.48 |
| Merged Trees | 5.11 | Merged Trees | 5.02 |
| Confidence Routed | 4.80 | Confidence Routed | 4.63 |
| Weight Averaged | 2.59 | Weight Averaged | 2.42 |

Benchmarks: MT-Bench, GSM8K, MATH-500, SVAMP. Metric: acceptance length (lossless speculative decoding constraint).

Benchmark-level routing decisions (confidence-based, EAGLE-2):

| Benchmark | MathInstruct | ShareGPT |
| --- | --- | --- |
| MT-Bench | 18.8% | 81.2% |
| GSM8K | 90.8% | 9.2% |
| MATH-500 | 97.0% | 3.0% |
| SVAMP | 93.0% | 7.0% |

## Repository map
- `Taps-draft1/` — LaTeX manuscript (sections, figures, macros).
- `Hass-Code/` — HASS draft training/eval scripts (feature build, training, routing, merged-tree).
- `Eagle-Code/` — EAGLE-2/3 draft code, training configs, eval utilities, Gradio demo.
- `docs/` — routing and merged-tree technical reports.

## Setup
- Python 3.10+; GPU memory sufficient for Meta-Llama-3-8B-Instruct plus draft checkpoints (paper used 4×A100).
- Prefer per-project virtualenvs to avoid dependency clashes.

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

## Run recipes — HASS
- One-shot pipeline (feature build → 10-epoch train → acceptance eval):
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
- Merged-tree dual-head evaluation (per-benchmark jobs across 4 GPUs):
```bash
BASE_MODEL_PATH=/path/to/Meta-Llama-3-8B-Instruct \
EA_MODEL_PATH1=/path/to/MathInstruct_ckpt \
EA_MODEL_PATH2=/path/to/ShareGPT_ckpt \
RESULTS_DIR=results/merged \
bash Hass-Code/scripts/run_merged_tree.sh
```
  Key env knobs: `TOTAL_TOKEN`, `DEPTH`, `TOP_K`, `TEMPERATURE`, `MAX_NEW_TOKEN`.
- Diagnostics and sweeps: `scripts/run_routed.sh` (confidence routing), `scripts/run_entropy_analysis.sh`, `scripts/run_full_entropy_analysis.sh`, `scripts/run_merge_sweep.sh`.

## Run recipes — EAGLE
- Demo web UI:
```bash
cd Eagle-Code
python eagle/application/webui.py --base-model-path <base> --ea-model-path <eagle3_ckpt> --model-type vicuna [--load-in-4bit|--load-in-8bit]
```
- Train a drafter:
```bash
cd Eagle-Code
python -m eagle.train.main --basepath <hf_or_local_model> \
  --configpath eagle/train/vicuna_13B_config.json \
  --tmpdir <data_dir> --cpdir <checkpoint_dir>
```
  For multi-GPU, use `main_deepspeed.py`.
- Merged-tree dual-head evaluation (4 GPUs):
```bash
cd Eagle-Code
BASE_MODEL_PATH=/path/to/Meta-Llama-3-8B-Instruct \
EA_MODEL_PATH1=checkpoints/Eagle-MathInstruct_20epochs \
EA_MODEL_PATH2=checkpoints/Eagle-ShareGPT_20epochs \
bash scripts/run_merged_tree.sh
```
- Entropy-based routing eval:
```bash
cd Eagle-Code
BASE_MODEL=/path/to/Meta-Llama-3-8B-Instruct bash scripts/run_routing_eval.sh
```
- More analyses: `scripts/run_confidence_analysis*.sh`, `scripts/run_entropy_analysis.sh`, `scripts/run_dual_head_eval.sh`.

## Reproducing the paper
- Verifier: Meta-Llama-3-8B-Instruct.
- Drafts: HASS and EAGLE-2, single transformer layer (~0.8B), shared tokenizer/vocab.
- Training: 20 epochs, lr 3e-5, batch size 8, grad accumulation 1.
- Data: MathInstruct (reasoning), ShareGPT (chat), mixed 35k+35k and 70k+70k.
- Benchmarks: MT-Bench, GSM8K, MATH-500, SVAMP at temperatures 0 and 1.
- Metric: acceptance length (avg accepted draft tokens per verifier call) with lossless speculative decoding.
- Hardware: single node, 4×A100 (per paper).
- Suggested order:
  1) Train single-domain drafts (MathInstruct, ShareGPT) for both backbones.  
  2) Train mixed 35k+35k and 70k+70k.  
  3) Run acceptance evals (temps 0 and 1) and collect JSON outputs.  
  4) Run confidence routing and merged-tree evals.  
  5) Tabulate with `scripts/merged_tree_table.py` / `scripts/routing_table.py` as appropriate.

## Citation
```bibtex
@article{zbib2026taps,
  title={TAPS: Task Aware Proposal Distributions for Speculative Sampling},
  author={Zbib, Mohamad and Bazzi, Mohamad and Mohanna, Ammar and Ghanem, Bernard and Hammoud, Hasan Abed Al Kader},
  year={2026},
  note={Technical report}
}
```

## License
- `Eagle-Code` is Apache 2.0 (see `Eagle-Code/LICENSE`).
- HASS scripts follow their upstream licenses; add a license file before redistribution.
