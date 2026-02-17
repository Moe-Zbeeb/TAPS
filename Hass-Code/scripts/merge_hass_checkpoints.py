#!/usr/bin/env python3
"""
Merge two HASS/EA checkpoints via weighted averaging.

Example:
  python scripts/merge_hass_checkpoints.py \
    --ckpt-a runs/mathinstruct_llama3/checkpoints/state_final \
    --ckpt-b runs/vicuna_sharegpt/checkpoints/state_final \
    --out    runs/merged_hass_math_vicuna \
    --weight-a 0.5 --weight-b 0.5
"""
import argparse
import shutil
from pathlib import Path

import torch


def merge_shard(ckpt_a: Path, ckpt_b: Path, out_dir: Path, name: str, w_a: float, w_b: float) -> None:
    """Average one checkpoint shard and save it to out_dir/name."""
    path_a = ckpt_a / name
    path_b = ckpt_b / name
    if not path_a.exists() or not path_b.exists():
        raise FileNotFoundError(f"Missing shard {name} in {ckpt_a} or {ckpt_b}")

    state_a = torch.load(path_a, map_location="cpu")
    state_b = torch.load(path_b, map_location="cpu")

    if set(state_a.keys()) != set(state_b.keys()):
        raise ValueError(f"Key mismatch between shards for {name}")

    merged = {}
    for key in state_a:
        if state_a[key].shape != state_b[key].shape:
            raise ValueError(f"Shape mismatch for {key}: {state_a[key].shape} vs {state_b[key].shape}")
        merged[key] = (state_a[key].float() * w_a + state_b[key].float() * w_b).to(state_a[key].dtype)

    torch.save(merged, out_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two HASS/EA checkpoints with weighted averaging.")
    parser.add_argument("--ckpt-a", required=True, type=Path, help="Path to first checkpoint directory")
    parser.add_argument("--ckpt-b", required=True, type=Path, help="Path to second checkpoint directory")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for merged checkpoint")
    parser.add_argument("--weight-a", type=float, default=0.5, help="Weight for checkpoint A (default: 0.5)")
    parser.add_argument("--weight-b", type=float, default=0.5, help="Weight for checkpoint B (default: 0.5)")
    parser.add_argument("--config", type=str, default="config.json", help="Config filename to copy (default: config.json)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Merging shards with weights A={args.weight_a} B={args.weight_b}")
    for shard in ("pytorch_model.bin", "pytorch_model_1.bin"):
        merge_shard(args.ckpt_a, args.ckpt_b, args.out, shard, args.weight_a, args.weight_b)
        print(f"  saved {shard}")

    # Copy config for completeness
    cfg_src = args.ckpt_a / args.config
    cfg_dst = args.out / args.config
    if cfg_src.exists():
        shutil.copy(cfg_src, cfg_dst)
        print(f"  copied {cfg_src} -> {cfg_dst}")
    else:
        print(f"Warning: config file {cfg_src} not found; skipped copy")

    print(f"Done. Merged checkpoint saved to {args.out}")


if __name__ == "__main__":
    main()
