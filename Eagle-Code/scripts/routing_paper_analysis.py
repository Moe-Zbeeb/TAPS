"""Routing analysis for paper figures and tables.

Reads existing JSONL results (no GPU needed) and produces:
1. Mixed-workload table (80 samples per domain, equal representation)
2. Routing accuracy table (entropy-chosen head vs oracle-best head)
3. Routing decision stacked bar chart (PDF + PNG)
4. Per-domain breakdown table

Outputs printed to stdout and saved to results/routing/routing_paper_results.md.
Figure saved to results/routing/routing_decisions.{pdf,png}.
"""

import json
import os
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HEADS = ["MathInstruct", "ShareGPT", "ShareGPT-MathInstruct", "Averaged"]
BENCHMARKS = ["mt_bench", "gsm8k", "math_500", "svamp"]
ENTROPY_DIR = "results/entropy"
ROUTING_DIR = "results/routing"
SEED = 42
MIXED_N = 80  # samples per benchmark in mixed workload


def load_entropy_data():
    """Load per-sample acceptance ratios for all heads x benchmarks.

    Returns: {bench: {head: {qid: acceptance_ratio}}}
    """
    data = {}
    for bench in BENCHMARKS:
        data[bench] = {}
        for head in HEADS:
            path = os.path.join(ENTROPY_DIR, f"{head}_{bench}.jsonl")
            if not os.path.exists(path):
                print(f"  Warning: missing {path}")
                continue
            samples = {}
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    qid = rec["question_id"]
                    na = rec["num_accepted"]
                    nr = rec["num_rejected"]
                    total = na + nr
                    samples[qid] = na / total if total > 0 else 0.0
            data[bench][head] = samples
    return data


def load_routing_data():
    """Load per-sample routing decisions and acceptance ratios.

    Returns: {bench: [{qid, chosen_head, acceptance_ratio}, ...]}
    """
    data = {}
    for bench in BENCHMARKS:
        path = os.path.join(ROUTING_DIR, f"{bench}.jsonl")
        if not os.path.exists(path):
            print(f"  Warning: missing {path}")
            continue
        records = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                records.append({
                    "qid": rec["question_id"],
                    "chosen_head": rec["chosen_head"],
                    "acceptance_ratio": rec["acceptance_ratio"],
                })
        data[bench] = records
    return data


def get_mixed_sample_ids(entropy_data):
    """Select 80 question IDs per benchmark for balanced mixed workload.

    mt_bench has exactly 80; subsample others with fixed seed.
    """
    rng = random.Random(SEED)
    selected = {}
    for bench in BENCHMARKS:
        # Get question IDs present in ALL heads for this benchmark
        head_qids = [set(entropy_data[bench][h].keys())
                     for h in HEADS if h in entropy_data[bench]]
        if not head_qids:
            selected[bench] = []
            continue
        common_qids = sorted(set.intersection(*head_qids))
        if len(common_qids) <= MIXED_N:
            selected[bench] = common_qids
        else:
            selected[bench] = sorted(rng.sample(common_qids, MIXED_N))
    return selected


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def compute_mixed_workload_table(entropy_data, routing_data, sample_ids):
    """Output 1: Mixed-workload acceptance ratio table."""
    lines = []
    lines.append("## Table 1: Mixed-Workload Routing (80 samples per domain)\n")

    # Header
    header = f"| {'Strategy':<25} |"
    for b in BENCHMARKS:
        header += f" {b:>9} |"
    header += f" {'Mean':>7} |"
    lines.append(header)

    sep = f"|:{'-'*24}-|"
    for _ in BENCHMARKS:
        sep += f"-{'-'*8}:-|"
    sep += f"-{'-'*6}:-|"
    lines.append(sep)

    # Per-head fixed strategies
    head_means_overall = {}
    for head in HEADS:
        row = f"| {head:<25} |"
        bench_means = []
        for bench in BENCHMARKS:
            ids = sample_ids[bench]
            if head in entropy_data[bench] and ids:
                vals = [entropy_data[bench][head].get(qid, 0.0) for qid in ids]
                m = mean(vals)
            else:
                m = 0.0
            bench_means.append(m)
            row += f" {m:>9.4f} |"
        overall = mean(bench_means)
        head_means_overall[head] = overall
        row += f" {overall:>7.4f} |"
        lines.append(row)

    # Entropy routing
    row = f"| {'Entropy routing':<25} |"
    routing_bench_means = []
    for bench in BENCHMARKS:
        ids_set = set(sample_ids[bench])
        if bench in routing_data:
            vals = [r["acceptance_ratio"] for r in routing_data[bench]
                    if r["qid"] in ids_set]
            m = mean(vals) if vals else 0.0
        else:
            m = 0.0
        routing_bench_means.append(m)
        row += f" {m:>9.4f} |"
    routing_overall = mean(routing_bench_means)
    row += f" {routing_overall:>7.4f} |"
    lines.append(row)

    # Oracle routing (per-sample best across heads)
    row = f"| {'Oracle routing':<25} |"
    oracle_bench_means = []
    for bench in BENCHMARKS:
        ids = sample_ids[bench]
        oracle_vals = []
        for qid in ids:
            best = max(
                entropy_data[bench][h].get(qid, 0.0)
                for h in HEADS if h in entropy_data[bench]
            )
            oracle_vals.append(best)
        m = mean(oracle_vals) if oracle_vals else 0.0
        oracle_bench_means.append(m)
        row += f" {m:>9.4f} |"
    oracle_overall = mean(oracle_bench_means)
    row += f" {oracle_overall:>7.4f} |"
    lines.append(row)

    # % of oracle gap closed
    best_fixed = max(head_means_overall.values())
    best_fixed_name = max(head_means_overall, key=head_means_overall.get)
    gap = oracle_overall - best_fixed
    if gap > 0:
        pct = (routing_overall - best_fixed) / gap * 100
    else:
        pct = 100.0 if routing_overall >= oracle_overall else 0.0
    lines.append("")
    lines.append(f"Best fixed head: **{best_fixed_name}** (mean {best_fixed:.4f})")
    lines.append(f"Oracle gap closed: **{pct:.1f}%** "
                 f"(routing {routing_overall:.4f} vs best-fixed {best_fixed:.4f} "
                 f"vs oracle {oracle_overall:.4f})")

    return "\n".join(lines)


def compute_routing_accuracy_table(entropy_data, routing_data):
    """Output 2: Per-benchmark routing accuracy (chosen == oracle)."""
    lines = []
    lines.append("## Table 2: Routing Accuracy\n")
    lines.append(f"| {'Benchmark':<12} | {'Accuracy':>8} | {'N':>6} |")
    lines.append(f"|:{'-'*11}-|-{'-'*7}:|-{'-'*5}:|")

    total_correct = 0
    total_n = 0

    for bench in BENCHMARKS:
        if bench not in routing_data:
            continue
        correct = 0
        n = 0
        for rec in routing_data[bench]:
            qid = rec["qid"]
            chosen = rec["chosen_head"]
            # Find oracle head
            best_head = None
            best_val = -1
            for h in HEADS:
                if h in entropy_data[bench]:
                    v = entropy_data[bench][h].get(qid, 0.0)
                    if v > best_val:
                        best_val = v
                        best_head = h
            if chosen == best_head:
                correct += 1
            n += 1
        acc = correct / n * 100 if n > 0 else 0.0
        lines.append(f"| {bench:<12} | {acc:>7.1f}% | {n:>5} |")
        total_correct += correct
        total_n += n

    overall_acc = total_correct / total_n * 100 if total_n > 0 else 0.0
    lines.append(f"|:{'-'*11}-|-{'-'*7}:|-{'-'*5}:|")
    lines.append(f"| {'**Overall**':<12} | {overall_acc:>7.1f}% | {total_n:>5} |")

    return "\n".join(lines)


def compute_per_domain_table(entropy_data, routing_data):
    """Per-domain breakdown: routing choice and resulting acceptance ratio
    vs each fixed head."""
    lines = []
    lines.append("## Table 3: Per-Domain Routing Breakdown\n")
    lines.append("For each benchmark, the head chosen most often by the router, "
                 "and acceptance ratios.\n")

    for bench in BENCHMARKS:
        if bench not in routing_data:
            continue
        lines.append(f"### {bench}\n")

        # Routing head distribution
        head_counts = Counter(r["chosen_head"] for r in routing_data[bench])
        total = len(routing_data[bench])
        routing_mean_ar = mean([r["acceptance_ratio"] for r in routing_data[bench]])

        lines.append(f"| {'Head':<25} | {'Routed %':>8} | {'Fixed AR':>8} |")
        lines.append(f"|:{'-'*24}-|-{'-'*7}:|-{'-'*7}:|")

        for head in HEADS:
            pct = head_counts.get(head, 0) / total * 100 if total > 0 else 0.0
            if head in entropy_data[bench]:
                fixed_ar = mean(list(entropy_data[bench][head].values()))
            else:
                fixed_ar = 0.0
            lines.append(f"| {head:<25} | {pct:>7.1f}% | {fixed_ar:>8.4f} |")

        lines.append(f"| {'**Entropy routing**':<25} | {'100.0%':>8} | {routing_mean_ar:>8.4f} |")
        lines.append("")

    return "\n".join(lines)


def make_routing_decision_figure(routing_data):
    """Output 3: Stacked bar chart of routing decisions per benchmark."""
    fig, ax = plt.subplots(figsize=(8, 5))

    head_colors = {
        "MathInstruct": "#2196F3",
        "ShareGPT": "#FF9800",
        "ShareGPT-MathInstruct": "#4CAF50",
        "Averaged": "#9C27B0",
    }

    bench_labels = []
    head_pcts = {h: [] for h in HEADS}

    for bench in BENCHMARKS:
        if bench not in routing_data:
            continue
        bench_labels.append(bench)
        total = len(routing_data[bench])
        counts = Counter(r["chosen_head"] for r in routing_data[bench])
        for head in HEADS:
            head_pcts[head].append(counts.get(head, 0) / total * 100 if total > 0 else 0.0)

    x = range(len(bench_labels))
    bottom = [0.0] * len(bench_labels)

    for head in HEADS:
        bars = ax.bar(x, head_pcts[head], bottom=bottom,
                      label=head, color=head_colors[head], width=0.6)
        # Add percentage labels for segments >= 10%
        for i, (val, bot) in enumerate(zip(head_pcts[head], bottom)):
            if val >= 10:
                ax.text(i, bot + val / 2, f"{val:.0f}%",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white")
        bottom = [b + v for b, v in zip(bottom, head_pcts[head])]

    ax.set_xticks(x)
    ax.set_xticklabels(bench_labels, fontsize=11)
    ax.set_ylabel("% of samples routed", fontsize=12)
    ax.set_title("Entropy Router Head Selection by Benchmark", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    os.makedirs(ROUTING_DIR, exist_ok=True)
    pdf_path = os.path.join(ROUTING_DIR, "routing_decisions.pdf")
    png_path = os.path.join(ROUTING_DIR, "routing_decisions.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {png_path}")


def main():
    print("Loading entropy data...")
    entropy_data = load_entropy_data()
    print("Loading routing data...")
    routing_data = load_routing_data()

    # Sample IDs for mixed workload
    sample_ids = get_mixed_sample_ids(entropy_data)
    for bench in BENCHMARKS:
        print(f"  {bench}: {len(sample_ids[bench])} samples selected")

    sections = []

    # Output 1
    print("\n" + "=" * 70)
    t1 = compute_mixed_workload_table(entropy_data, routing_data, sample_ids)
    print(t1)
    sections.append(t1)

    # Output 2
    print("\n" + "=" * 70)
    t2 = compute_routing_accuracy_table(entropy_data, routing_data)
    print(t2)
    sections.append(t2)

    # Output 3 (figure)
    print("\n" + "=" * 70)
    make_routing_decision_figure(routing_data)

    # Per-domain breakdown
    print("\n" + "=" * 70)
    t3 = compute_per_domain_table(entropy_data, routing_data)
    print(t3)
    sections.append(t3)

    # Save markdown
    os.makedirs(ROUTING_DIR, exist_ok=True)
    md_path = os.path.join(ROUTING_DIR, "routing_paper_results.md")
    full_md = "# Routing Analysis for Paper\n\n" + "\n\n".join(sections) + "\n"
    with open(md_path, "w") as f:
        f.write(full_md)
    print(f"\nSaved markdown: {md_path}")


if __name__ == "__main__":
    main()
