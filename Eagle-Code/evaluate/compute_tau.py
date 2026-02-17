#!/usr/bin/env python3
"""
Compute τ (average acceptance length) from EAGLE evaluation results.
τ = total_new_tokens / total_iterations

This matches the metric reported in the EAGLE-3 paper Table 1.
"""

import json
import argparse
import os
from pathlib import Path


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_tau(records):
    total_new_tokens = 0
    total_idxs = 0
    total_wall_time = 0
    num_questions = 0

    for rec in records:
        choices = rec.get("choices", [])
        if not choices:
            continue

        for choice in choices:
            idxs_list = choice.get("idxs", [])
            new_tokens_list = choice.get("new_tokens", [])
            wall_time_list = choice.get("wall_time", [])

            total_new_tokens += sum(new_tokens_list)
            total_idxs += sum(idxs_list)
            total_wall_time += sum(wall_time_list)

        num_questions += 1

    tau = total_new_tokens / total_idxs if total_idxs > 0 else 0
    tokens_per_sec = total_new_tokens / total_wall_time if total_wall_time > 0 else 0

    return {
        "num_questions": num_questions,
        "total_new_tokens": total_new_tokens,
        "total_iterations": total_idxs,
        "total_wall_time": round(total_wall_time, 2),
        "tau": round(tau, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute τ from EAGLE results")
    parser.add_argument("files", nargs="+", help="JSONL result files")
    args = parser.parse_args()

    results = {}

    print("\n" + "=" * 70)
    print(f"{'Model':<40} {'τ':>8} {'Tok/s':>10} {'Questions':>10}")
    print("=" * 70)

    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"Warning: {filepath} not found")
            continue

        records = load_jsonl(path)
        stats = compute_tau(records)
        results[path.stem] = stats

        print(f"{path.stem:<40} {stats['tau']:>8.2f} {stats['tokens_per_sec']:>10.2f} {stats['num_questions']:>10}")

    print("=" * 70)

    # Group by benchmark
    if len(results) > 1:
        print("\nComparison Summary:")
        benchmarks = {}
        for name, stats in results.items():
            parts = name.split("_")
            if len(parts) >= 2:
                bench = parts[0]  # mt_bench, gsm8k, etc.
                model = "_".join(parts[1:])  # math, sharegpt, etc.
                if bench not in benchmarks:
                    benchmarks[bench] = {}
                benchmarks[bench][model] = stats["tau"]

        for bench, models in benchmarks.items():
            print(f"\n{bench}:")
            for model, tau in models.items():
                print(f"  {model}: τ = {tau}")


if __name__ == "__main__":
    main()
