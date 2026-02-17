#!/usr/bin/env python3
"""
Diagnostic script to check for corrupted .ckpt files in a data directory.

This script:
- Recursively finds all .ckpt files in the specified directory
- Sorts them the same way os.walk encounters them (filesystem order)
- Attempts to load each file with torch.load()
- Reports corrupted files with their index position and error details

Usage:
    python check_ckpt_files.py /path/to/data/directory
    python check_ckpt_files.py /path/to/data/directory --verbose
    python check_ckpt_files.py /path/to/data/directory --check-keys
"""

import argparse
import os
import sys
from typing import List, Tuple


def list_files(path: str) -> List[str]:
    """
    List all files in a directory tree using os.walk.
    This matches the behavior of the training script's list_files function.
    """
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


def check_ckpt_file(filepath: str, check_keys: bool = False) -> Tuple[bool, str]:
    """
    Check if a .ckpt file can be loaded successfully.

    Args:
        filepath: Path to the .ckpt file
        check_keys: If True, also verify expected keys exist in the checkpoint

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    import torch

    try:
        # Use weights_only=False to match training behavior, but this is for diagnostics
        # Map to CPU to avoid GPU memory issues when checking many files
        data = torch.load(filepath, map_location='cpu')

        if check_keys:
            # Check for expected keys based on the training script
            expected_keys = ['hidden_state', 'input_ids', 'loss_mask']
            missing_keys = [k for k in expected_keys if k not in data]
            if missing_keys:
                return False, f"Missing expected keys: {missing_keys}. Found keys: {list(data.keys())}"

            # Basic shape validation
            hidden_state = data.get('hidden_state')
            input_ids = data.get('input_ids')
            loss_mask = data.get('loss_mask')

            if hidden_state is not None and len(hidden_state.shape) != 2:
                return False, f"hidden_state has unexpected shape: {hidden_state.shape}"
            if input_ids is not None and len(input_ids.shape) != 1:
                return False, f"input_ids has unexpected shape: {input_ids.shape}"
            if loss_mask is not None and len(loss_mask.shape) != 1:
                return False, f"loss_mask has unexpected shape: {loss_mask.shape}"

        return True, ""

    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Check .ckpt files for corruption in a data directory"
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing .ckpt files (can have subdirectories)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress for each file checked"
    )
    parser.add_argument(
        "--check-keys",
        action="store_true",
        help="Also verify expected keys (hidden_state, input_ids, loss_mask) exist"
    )
    parser.add_argument(
        "--train-split-only",
        action="store_true",
        help="Only check the training split (first 95%% of files)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start checking from this index (useful to resume or focus on problem area)"
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Stop checking at this index"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: {args.data_dir} is not a valid directory", file=sys.stderr)
        sys.exit(1)

    # Import torch here to fail fast if not available
    try:
        import torch
    except ImportError:
        print("Error: PyTorch is required. Install with: pip install torch", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning directory: {args.data_dir}")

    # Get all files using the same method as the training script
    all_files = list_files(args.data_dir)

    # Filter to only .ckpt files
    ckpt_files = [f for f in all_files if f.endswith('.ckpt')]

    if not ckpt_files:
        print(f"No .ckpt files found in {args.data_dir}")
        sys.exit(0)

    print(f"Found {len(ckpt_files)} .ckpt files")

    # Apply train split filter if requested
    if args.train_split_only:
        train_count = int(len(ckpt_files) * 0.95)
        ckpt_files = ckpt_files[:train_count]
        print(f"Checking training split only: {len(ckpt_files)} files (95%)")

    # Apply index range
    start_idx = args.start_index
    end_idx = args.end_index if args.end_index is not None else len(ckpt_files)

    if start_idx > 0 or end_idx < len(ckpt_files):
        ckpt_files = ckpt_files[start_idx:end_idx]
        print(f"Checking files from index {start_idx} to {end_idx}")

    # Track results
    corrupted_files = []
    checked_count = 0

    print(f"\nChecking {len(ckpt_files)} files...")
    print("-" * 60)

    for i, filepath in enumerate(ckpt_files):
        actual_index = start_idx + i

        if args.verbose:
            print(f"[{actual_index}/{start_idx + len(ckpt_files) - 1}] Checking: {filepath}")

        success, error = check_ckpt_file(filepath, check_keys=args.check_keys)
        checked_count += 1

        if not success:
            corrupted_files.append((actual_index, filepath, error))
            print(f"\n*** CORRUPTED FILE FOUND ***")
            print(f"Index: {actual_index}")
            print(f"Path: {filepath}")
            print(f"Error: {error}")
            print()
        elif not args.verbose and checked_count % 100 == 0:
            print(f"Checked {checked_count}/{len(ckpt_files)} files...")

    # Summary
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Total files checked: {checked_count}")
    print(f"  Corrupted files: {len(corrupted_files)}")

    if corrupted_files:
        print(f"\nCorrupted file details:")
        for idx, path, error in corrupted_files:
            print(f"\n  Index {idx}:")
            print(f"    Path: {path}")
            print(f"    Error: {error}")

        print(f"\n\nCorrupted file paths (copy-paste friendly):")
        for _, path, _ in corrupted_files:
            print(path)

        sys.exit(1)
    else:
        print("\nAll files loaded successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
