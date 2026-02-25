"""Print a routing-distribution table from pre-computed confidence data.

For each benchmark × question we pick the head whose draft tokens had the
highest mean overall confidence (accepted + rejected combined).  Higher
confidence ↔ lower entropy, so this is equivalent to the entropy-min routing
used in gen_routing_eval.py.

Usage:
    python evaluate/routing_table.py
    python evaluate/routing_table.py --output-dir results/confidence \
        --bench-name mt_bench gsm8k math_500 svamp
"""

import argparse
import json
import os


HEADS = ["MathInstruct", "ShareGPT"]
BENCHMARKS = ["mt_bench", "gsm8k", "math_500", "svamp"]


def load_questions(path):
    """Return {question_id: record} for one head/bench jsonl file."""
    data = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            data[rec["question_id"]] = rec
    return data


def overall_mean_conf(rec):
    """Mean confidence across all draft tokens (accepted + rejected)."""
    all_conf = rec.get("accepted_confidences", []) + rec.get("rejected_confidences", [])
    return sum(all_conf) / len(all_conf) if all_conf else 0.0


def compute_routing(output_dir, benchmarks, fallback_dir=None):
    results = {}  # bench -> {head: count}

    for bench in benchmarks:
        # Load per-head data
        head_data = {}
        for head in HEADS:
            path = os.path.join(output_dir, f"{head}_{bench}.jsonl")
            if not os.path.exists(path) and fallback_dir:
                path = os.path.join(fallback_dir, f"{head}_{bench}.jsonl")
            if not os.path.exists(path):
                print(f"  WARNING: missing {head}_{bench}.jsonl")
                continue
            head_data[head] = load_questions(path)

        if not head_data:
            continue

        # Use the intersection of question IDs present in all heads
        qids = set.intersection(*(set(d.keys()) for d in head_data.values()))
        counts = {h: 0 for h in HEADS}

        for qid in qids:
            # Route to head with highest mean confidence (= min entropy)
            best_head = max(
                head_data.keys(),
                key=lambda h: overall_mean_conf(head_data[h][qid]),
            )
            counts[best_head] += 1

        results[bench] = counts

    return results


def build_table(results):
    col_w = 22
    bench_w = 12
    head_labels = ["MathInstruct", "ShareGPT"]
    header = f"{'Benchmark':<{bench_w}}" + "".join(f"{h:^{col_w}}" for h in head_labels) + f"{'Total':>8}"
    sep = "-" * len(header)

    lines = [sep, header, sep]
    for bench, counts in results.items():
        total = sum(counts.values())
        row = f"{bench:<{bench_w}}"
        for head in HEADS:
            n = counts.get(head, 0)
            pct = 100.0 * n / total if total else 0.0
            cell = f"{n} ({pct:.1f}%)"
            row += f"{cell:^{col_w}}"
        row += f"{total:>8}"
        lines.append(row)
    lines.append(sep)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/confidence")
    parser.add_argument("--fallback-dir", default=None,
                        help="Secondary dir to search if a file is missing in --output-dir.")
    parser.add_argument("--bench-name", nargs="+", default=BENCHMARKS)
    parser.add_argument("--save-path", default="results/confidence/routing_table.txt")
    args = parser.parse_args()

    header_line = "Entropy routing simulation (max overall draft confidence = min entropy)"
    print(f"\n{header_line}")
    print(f"Data directory: {args.output_dir}" + (f" + {args.fallback_dir}" if args.fallback_dir else "") + "\n")

    results = compute_routing(args.output_dir, args.bench_name, fallback_dir=args.fallback_dir)
    table = build_table(results)

    print(table)

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    with open(args.save_path, "w") as f:
        f.write(header_line + "\n\n")
        f.write(table + "\n")
    print(f"\nSaved to {args.save_path}")


if __name__ == "__main__":
    main()
