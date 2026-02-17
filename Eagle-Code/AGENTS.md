# Repository Guidelines

## Project Structure & Module Organization
- Core package lives in `eagle/` with submodules for training (`train/`, `traineagle3/`), evaluation utilities (`evaluation/`), application entry points (`application/`), and model components (`model/`, `modeling_eagle.py`).
- Reference configs for common bases (Vicuna, LLaMA, Mixtral) sit in `eagle/train/*.json`; adjust paths before running.
- Sample datasets for quick checks are under `eagle/data/`; synthetic and visualization helpers for micro-benchmarks reside in `eagle/testbug/`.
- Assets and figures stay in `figs/`; packaging metadata lives at the repo root (`setup.py`, `requirements*.txt`).

## Build, Test, and Development Commands
- Create an isolated environment and install dependencies: `python -m venv .venv && source .venv/bin/activate`, then `pip install -r requirements.txt` (use `requirements-rocm.txt` for ROCm setups) and `pip install -e .` for editable work.
- Launch the Gradio demo with explicit model locations: `python eagle/application/webui.py --base-model-path <base> --ea-model-path <eagle3> [--load-in-4bit|--load-in-8bit] --model-type vicuna`.
- Run the training driver with a config: `python -m eagle.train.main --basepath <hf_or_local_model> --configpath eagle/train/vicuna_13B_config.json --tmpdir <data_dir> --cpdir <checkpoint_dir>`; use `main_deepspeed.py` when coordinating multi-GPU jobs.
- Quick regression check: `python eagle/testbug/testbbug.py` generates cached outputs and timings in the working directory.

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indentation; prefer `snake_case` for functions and variables, `PascalCase` for classes, and descriptive module names (e.g., `ea_model.py`).
- Keep tensors and model artifacts on GPU-aware paths only when needed; gate device moves behind explicit flags and avoid hardcoded device indices.
- Use type hints where practical and keep functions small and single-purpose; reuse helpers in `eagle/model/` instead of duplicating logic.

## Testing Guidelines
- Add new unit-style tests under `eagle/testbug/` or a `tests/` folder, using `test_*.py` naming so `pytest` discovery works if introduced; ensure deterministic seeds when benchmarking.
- For performance-sensitive changes, compare outputs against saved tensors in `eagle/testbug/` and record token-per-second deltas.
- Prefer lightweight smoke checks (`python eagle/testbug/testbbug.py`) before running heavier evaluation suites; document any required hardware assumptions.

## Commit & Pull Request Guidelines
- Write imperative, concise commit subjects (e.g., "Add eagle3 webui flag"), mirroring the short summaries already in the history; include scope prefixes when touching multiple areas ("train:" or "model:").
- In pull requests, describe the motivation, key changes, and validation commands; link related issues and include screenshots for UI-facing updates (e.g., web UI screenshots from `application/webui.py`).
- Call out performance impacts, new dependencies, and configuration changes (paths, precision flags) in the PR description to aid reviewers.

## Security & Configuration Tips
- Keep credentials (API keys, weights that require auth) out of the repo; load them via environment variables and `.gitignore` any local overrides.
- Verify license compatibility before adding new datasets or checkpoints; prefer Hugging Face IDs that allow redistribution.
- When sharing configs, avoid embedding absolute paths—use placeholders (`<path_to_weights>`) to keep instructions portable across environments.
