"""Summarize entropy analysis results into markdown tables.

Reads all JSONL files from results/entropy/ and produces two tables:
1. Draft (EA Head) entropy — accepted vs rejected
2. Verifier (Target LLM) entropy — accepted vs rejected

Printed to stdout and saved as results/entropy/summary_table.md.
"""
import json
import os
import glob


def main():
    results_dir = "results/entropy"
    jsonl_files = sorted(glob.glob(os.path.join(results_dir, "*.jsonl")))

    if not jsonl_files:
        print(f"No JSONL files found in {results_dir}/")
        return

    known_benches = ["mt_bench", "gsm8k", "math_500", "svamp"]
    rows = []

    for filepath in jsonl_files:
        filename = os.path.basename(filepath)
        name_no_ext = filename.rsplit(".", 1)[0]
        checkpoint = None
        bench = None
        for b in known_benches:
            if name_no_ext.endswith(f"_{b}"):
                checkpoint = name_no_ext[: -(len(b) + 1)]
                bench = b
                break
        if checkpoint is None:
            checkpoint = name_no_ext
            bench = "unknown"

        all_d_acc = []
        all_d_rej = []
        all_v_acc = []
        all_v_rej = []

        with open(filepath, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                all_d_acc.extend(rec.get("draft_accepted_entropies", []))
                all_d_rej.extend(rec.get("draft_rejected_entropies", []))
                all_v_acc.extend(rec.get("verifier_accepted_entropies", []))
                all_v_rej.extend(rec.get("verifier_rejected_entropies", []))

        mean_d_acc = sum(all_d_acc) / len(all_d_acc) if all_d_acc else 0.0
        mean_d_rej = sum(all_d_rej) / len(all_d_rej) if all_d_rej else 0.0
        mean_v_acc = sum(all_v_acc) / len(all_v_acc) if all_v_acc else 0.0
        mean_v_rej = sum(all_v_rej) / len(all_v_rej) if all_v_rej else 0.0

        rows.append({
            "checkpoint": checkpoint,
            "benchmark": bench,
            "draft_accepted": mean_d_acc,
            "draft_rejected": mean_d_rej,
            "draft_delta": mean_d_rej - mean_d_acc,
            "num_d_acc": len(all_d_acc),
            "num_d_rej": len(all_d_rej),
            "verifier_accepted": mean_v_acc,
            "verifier_rejected": mean_v_rej,
            "verifier_delta": mean_v_rej - mean_v_acc,
            "num_v_acc": len(all_v_acc),
            "num_v_rej": len(all_v_rej),
        })

    # --- Draft (EA Head) Entropy Table ---
    lines = ["# EAGLE-2 Entropy Analysis Summary\n"]
    lines.append("## Draft (EA Head) Entropy\n")
    header = "| Checkpoint             | Benchmark | Draft Accepted | Draft Rejected | \u0394 Draft  |"
    sep =    "|:-----------------------|:----------|---------------:|---------------:|--------:|"
    lines.extend([header, sep])
    for r in rows:
        lines.append(
            f"| {r['checkpoint']:<22} | {r['benchmark']:<9} | "
            f"{r['draft_accepted']:>14.4f} | {r['draft_rejected']:>14.4f} | "
            f"{r['draft_delta']:>+7.4f} |"
        )
    # Draft overall
    total_d_acc_w = sum(r["draft_accepted"] * r["num_d_acc"] for r in rows)
    total_d_rej_w = sum(r["draft_rejected"] * r["num_d_rej"] for r in rows)
    total_d_acc_n = sum(r["num_d_acc"] for r in rows)
    total_d_rej_n = sum(r["num_d_rej"] for r in rows)
    overall_d_acc = total_d_acc_w / total_d_acc_n if total_d_acc_n > 0 else 0.0
    overall_d_rej = total_d_rej_w / total_d_rej_n if total_d_rej_n > 0 else 0.0
    lines.append(sep)
    lines.append(
        f"| {'**Overall**':<22} | {'**all**':<9} | "
        f"{overall_d_acc:>14.4f} | {overall_d_rej:>14.4f} | "
        f"{overall_d_rej - overall_d_acc:>+7.4f} |"
    )

    # --- Verifier (Target LLM) Entropy Table ---
    lines.append("\n## Verifier (Target LLM) Entropy\n")
    header = "| Checkpoint             | Benchmark | Verifier Accepted | Verifier Rejected | \u0394 Verifier |"
    sep =    "|:-----------------------|:----------|------------------:|------------------:|-----------:|"
    lines.extend([header, sep])
    for r in rows:
        lines.append(
            f"| {r['checkpoint']:<22} | {r['benchmark']:<9} | "
            f"{r['verifier_accepted']:>17.4f} | {r['verifier_rejected']:>17.4f} | "
            f"{r['verifier_delta']:>+10.4f} |"
        )
    # Verifier overall
    total_v_acc_w = sum(r["verifier_accepted"] * r["num_v_acc"] for r in rows)
    total_v_rej_w = sum(r["verifier_rejected"] * r["num_v_rej"] for r in rows)
    total_v_acc_n = sum(r["num_v_acc"] for r in rows)
    total_v_rej_n = sum(r["num_v_rej"] for r in rows)
    overall_v_acc = total_v_acc_w / total_v_acc_n if total_v_acc_n > 0 else 0.0
    overall_v_rej = total_v_rej_w / total_v_rej_n if total_v_rej_n > 0 else 0.0
    lines.append(sep)
    lines.append(
        f"| {'**Overall**':<22} | {'**all**':<9} | "
        f"{overall_v_acc:>17.4f} | {overall_v_rej:>17.4f} | "
        f"{overall_v_rej - overall_v_acc:>+10.4f} |"
    )

    full_output = "\n".join(lines) + "\n"

    print(full_output)

    output_path = os.path.join(results_dir, "summary_table.md")
    with open(output_path, "w") as f:
        f.write(full_output)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
