# TAPS: Task-Aware Proposal Distributions for Speculative Sampling

**Project Page · March 2026**

Mohamad Zbib, Mohamad Bazzi, Ammar Mohanna, **Bernard Ghanem**\*, **Hasan Abed Al Kader Hammoud**\*  (\*equal advising)  
KAUST · AUB — Contacts: [mbz02@mail.aub.edu](mailto:mbz02@mail.aub.edu), [hasanabedalkader.hammoud@kaust.edu.sa](mailto:hasanabedalkader.hammoud@kaust.edu.sa)

---

> Task-aware drafts that speed up speculative decoding without sacrificing quality.

## Quick links
[![HF Weights](https://img.shields.io/badge/HF_HuggingFace-Weights-orange)](https://huggingface.co/collections/zbeeb/taps)
[![HF Datasets](https://img.shields.io/badge/HF_HuggingFace-Datasets-blue)](https://huggingface.co/datasets/zbeeb/TAPS-Datasets)

## Contents
<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#abstract">Abstract</a> •
  <a href="#highlights">Highlights</a> •
  <a href="#figures">Figures</a> •
  <a href="#results-snapshot">Results snapshot</a> •
  <a href="#repository-map">Repository map</a> •
  <a href="#setup">Setup</a> •
  <a href="#citation">Citation</a> •
  <a href="#license">License</a>
</p>

---

## Overview
TAPS studies how the draft training distribution shapes speculative decoding quality. Using Meta-Llama-3-8B-Instruct as the verifier and lightweight LLaMA-style drafters (HASS and EAGLE-2, ~0.8B params, shared tokenizer), the work compares single-domain training, mixed-domain training, and inference-time composition (confidence routing, merged-tree verification).

### TL;DR (At a glance)
- Goal: Quantify how draft training data and specialist composition affect speculative decoding.
- Verifier: Meta-Llama-3-8B-Instruct.
- Drafts: HASS and EAGLE-2 (single-layer, ~0.8B params, shared tokenizer).
- Workloads: MT-Bench (chat), GSM8K, MATH-500, SVAMP (reasoning).
- Metric: acceptance length (lossless speculative decoding).
- Compute: single node, 4×A100.

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
      <img src="Taps-draft1/figures/illustrative.png" alt="Speculative decoding pipeline with draft proposals verified by the target model" height="170">
    </td>
    <td align="center">
      <img src="Taps-draft1/figures/mergdTrees.png" alt="Merged-tree verification combining MathInstruct and ShareGPT draft trees" height="170">
    </td>
  </tr>
  <tr>
    <td align="center"><em>Speculative decoding pipeline.</em></td>
    <td align="center"><em>Merged-tree verification packs MathInstruct and ShareGPT trees for one-pass verification.</em></td>
  </tr>
</table>

<p align="center">
  <img src="Taps-draft1/figures/averaging_effect.png" alt="Interpolation sweep for checkpoint averaging between MathInstruct and ShareGPT drafts" width="760">
  <br>
  <em>Checkpoint averaging is unstable across interpolation weights and remains weaker than inference-time composition.</em>
</p>

<table>
  <tr>
    <td align="center">
      <img src="Taps-draft1/figures/depth_tables_grid.png" alt="Acceptance by speculative depth across backbones, benchmarks, and temperatures" width="460">
    </td>
    <td align="center">
      <img src="Taps-draft1/figures/draft_entropy_eagle2.png" alt="Accepted vs. rejected token entropy for EAGLE-2 drafts" width="460">
    </td>
  </tr>
  <tr>
    <td align="center"><em>Acceptance declines with depth but specialization persists.</em></td>
    <td align="center"><em>Rejected tokens show higher entropy; confidence remains the stronger routing signal.</em></td>
  </tr>
</table>

## Results snapshot
Benchmarks: MT-Bench (chat), GSM8K, MATH-500, SVAMP. Metric: average acceptance length (higher is better) under lossless speculative decoding. Tables stay inline (no collapses).

<p><strong>Temperature&nbsp;0</strong></p>
<table>
  <thead>
    <tr><th>Model Variant</th><th>Method</th><th>MT-Bench</th><th>GSM8K</th><th>MATH-500</th><th>SVAMP</th><th>Average</th></tr>
  </thead>
  <tbody>
    <tr><td>MathInstruct</td><td>HASS</td><td>2.90</td><td>5.02</td><td>5.35</td><td>3.13</td><td>4.10</td></tr>
    <tr><td>MathInstruct</td><td>EAGLE-2</td><td>2.54</td><td>5.04</td><td>5.28</td><td>4.81</td><td>4.42</td></tr>
    <tr><td>ShareGPT</td><td>HASS</td><td>3.98</td><td>4.09</td><td>3.98</td><td>4.44</td><td>4.12</td></tr>
    <tr><td>ShareGPT</td><td>EAGLE-2</td><td>3.57</td><td>3.72</td><td>3.81</td><td>3.71</td><td>3.70</td></tr>
    <tr><td>Mixed 35k+35k</td><td>HASS</td><td>3.92</td><td>4.77</td><td>5.02</td><td>4.15</td><td>4.47</td></tr>
    <tr><td>Mixed 35k+35k</td><td>EAGLE-2</td><td>3.37</td><td>4.12</td><td>4.44</td><td>4.16</td><td>4.02</td></tr>
    <tr><td>Mixed 70k+70k</td><td>HASS</td><td>4.13</td><td>5.53</td><td>5.67</td><td>5.38</td><td>5.18</td></tr>
    <tr><td>Mixed 70k+70k</td><td>EAGLE-2</td><td>3.75</td><td>4.68</td><td>4.85</td><td>4.64</td><td>4.48</td></tr>
    <tr><td>Averaged</td><td>HASS</td><td>2.29</td><td>2.80</td><td>3.12</td><td>2.13</td><td>2.59</td></tr>
    <tr><td>Averaged</td><td>EAGLE-2</td><td>2.07</td><td>2.53</td><td>2.57</td><td>2.50</td><td>2.42</td></tr>
    <tr><td>Confidence Routed</td><td>HASS</td><td>3.93</td><td>5.01</td><td>5.37</td><td>4.89</td><td>4.80</td></tr>
    <tr><td>Confidence Routed</td><td>EAGLE-2</td><td>3.63</td><td>4.91</td><td>5.25</td><td>4.71</td><td>4.63</td></tr>
    <tr><td>Merged Trees</td><td>HASS</td><td>4.05</td><td>5.42</td><td>5.65</td><td>5.31</td><td>5.11</td></tr>
    <tr><td>Merged Trees</td><td>EAGLE-2</td><td>3.93</td><td>5.32</td><td>5.63</td><td>5.25</td><td>5.02</td></tr>
  </tbody>
</table>

<p><strong>Temperature&nbsp;1</strong></p>
<table>
  <thead>
    <tr><th>Model Variant</th><th>Method</th><th>MT-Bench</th><th>GSM8K</th><th>MATH-500</th><th>SVAMP</th><th>Average</th></tr>
  </thead>
  <tbody>
    <tr><td>MathInstruct</td><td>HASS</td><td>2.31</td><td>4.75</td><td>4.63</td><td>2.46</td><td>3.54</td></tr>
    <tr><td>MathInstruct</td><td>EAGLE-2</td><td>2.43</td><td>4.71</td><td>4.61</td><td>4.53</td><td>4.07</td></tr>
    <tr><td>ShareGPT</td><td>HASS</td><td>3.50</td><td>4.03</td><td>3.61</td><td>3.95</td><td>3.77</td></tr>
    <tr><td>ShareGPT</td><td>EAGLE-2</td><td>3.38</td><td>3.72</td><td>3.43</td><td>3.65</td><td>3.54</td></tr>
    <tr><td>Mixed 35k+35k</td><td>HASS</td><td>3.46</td><td>4.66</td><td>4.47</td><td>4.57</td><td>4.29</td></tr>
    <tr><td>Mixed 35k+35k</td><td>EAGLE-2</td><td>3.10</td><td>4.08</td><td>4.02</td><td>4.03</td><td>3.81</td></tr>
    <tr><td>Mixed 70k+70k</td><td>HASS</td><td>3.17</td><td>4.16</td><td>3.42</td><td>4.01</td><td>3.69</td></tr>
    <tr><td>Mixed 70k+70k</td><td>EAGLE-2</td><td>2.99</td><td>3.76</td><td>3.20</td><td>3.08</td><td>3.26</td></tr>
    <tr><td>Averaged</td><td>HASS</td><td>2.10</td><td>2.78</td><td>2.90</td><td>2.69</td><td>2.62</td></tr>
    <tr><td>Averaged</td><td>EAGLE-2</td><td>2.01</td><td>2.49</td><td>2.42</td><td>2.45</td><td>2.34</td></tr>
    <tr><td>Confidence Routed</td><td>HASS</td><td>3.51</td><td>4.72</td><td>4.55</td><td>4.71</td><td>4.37</td></tr>
    <tr><td>Confidence Routed</td><td>EAGLE-2</td><td>3.36</td><td>4.65</td><td>4.62</td><td>4.46</td><td>4.27</td></tr>
    <tr><td>Merged Trees</td><td>HASS</td><td>3.76</td><td>5.21</td><td>4.98</td><td>5.05</td><td>4.75</td></tr>
    <tr><td>Merged Trees</td><td>EAGLE-2</td><td>3.55</td><td>5.01</td><td>4.79</td><td>4.93</td><td>4.57</td></tr>
  </tbody>
</table>

## Repository map
| Folder | What lives here |
| --- | --- |
| `Taps-draft1/` | Paper figures and assets |
| `Hass-Code/` | HASS draft training/eval scripts (feature build, training, routing, merged-tree) |
| `Eagle-Code/` | EAGLE-2/3 draft code, training configs, eval utilities, Gradio demo |

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

## Citation
```bibtex
@article{zbib2026taps,
  title={TAPS: Task Aware Proposal Distributions for Speculative Sampling},
  author={Zbib, Mohamad and Bazzi, Mohamad and Mohanna, Ammar and Hammoud, Hasan Abed Al Kader and Ghanem, Bernard},
  journal={arXiv preprint arXiv:2603.27027},
  year={2026}
}
```
