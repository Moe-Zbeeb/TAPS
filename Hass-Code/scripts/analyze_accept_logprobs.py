#!/usr/bin/env python
"""
Summarize acceptance-step traces produced by trace_accept_logprobs.py.

Computes match/mismatch rates, logprob diffs, entropy, and accept-length stats.
Optionally dumps mismatches and saves plots.
"""

import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


def load_traces(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Analyze acceptance-step logprob traces.")
    parser.add_argument("--input", required=True, help="JSONL produced by trace_accept_logprobs.py")
    parser.add_argument("--save-summary", help="Optional path to save summary JSON")
    parser.add_argument("--dump-mismatches", help="Optional path to save per-step mismatches (jsonl)")
    parser.add_argument("--plot-dir", help="Optional directory to save plots")
    args = parser.parse_args()

    traces = load_traces(Path(args.input))
    if not traces:
        print("No data found.")
        return

    step_records = []
    accept_lengths = []
    total_tokens = 0
    for t in traces:
        step_records.extend(t.get("steps", []))
        total_tokens += t.get("tokens_generated", 0)
        accept_lengths.extend([s.get("accept_length", 0) for s in t.get("steps", [])])

    total_q = len(traces)
    total_steps = len(step_records)
    matches = [s for s in step_records if s.get("ea_token") == s.get("verifier_token")]
    mismatches = [s for s in step_records if s.get("ea_token") != s.get("verifier_token")]

    diffs_all = [s.get("logprob_diff", 0.0) for s in step_records]
    diffs_match = [s.get("logprob_diff", 0.0) for s in matches]
    diffs_mismatch = [s.get("logprob_diff", 0.0) for s in mismatches]

    logprob_ea = [s.get("logprob_ea_token", 0.0) for s in step_records]
    logprob_ver = [s.get("logprob_verifier_token", 0.0) for s in step_records]
    logprob_ea_prop = [s.get("ea_logprob") for s in step_records if s.get("ea_logprob") is not None]
    logprob_ea_prop_match = [s.get("ea_logprob") for s in matches if s.get("ea_logprob") is not None]
    logprob_ea_prop_mismatch = [s.get("ea_logprob") for s in mismatches if s.get("ea_logprob") is not None]
    ea_ver_gap_match = [s.get("ea_logprob") - s.get("logprob_verifier_token") for s in matches if s.get("ea_logprob") is not None]
    ea_ver_gap_mismatch = [s.get("ea_logprob") - s.get("logprob_verifier_token") for s in mismatches if s.get("ea_logprob") is not None]

    ent_all = [s.get("entropy", 0.0) for s in step_records]
    ent_match = [s.get("entropy", 0.0) for s in matches]
    ent_mismatch = [s.get("entropy", 0.0) for s in mismatches]
    ent_ea_all = [s.get("ea_entropy") for s in step_records if s.get("ea_entropy") is not None]
    ent_ea_match = [s.get("ea_entropy") for s in matches if s.get("ea_entropy") is not None]
    ent_ea_mismatch = [s.get("ea_entropy") for s in mismatches if s.get("ea_entropy") is not None]

    accept_dist = Counter(accept_lengths)

    summary = {
        "total_questions": total_q,
        "total_steps": total_steps,
        "total_tokens_generated": total_tokens,
        "avg_steps_per_question": total_steps / total_q,
        "avg_tokens_per_question": total_tokens / total_q,
        "match_rate": len(matches) / total_steps if total_steps else 0.0,
        "mismatch_rate": len(mismatches) / total_steps if total_steps else 0.0,
        "logprob_diff": {
            "mean_all": float(np.mean(diffs_all)),
            "std_all": float(np.std(diffs_all)),
            "mean_match": float(np.mean(diffs_match)) if diffs_match else 0.0,
            "std_match": float(np.std(diffs_match)) if diffs_match else 0.0,
            "mean_mismatch": float(np.mean(diffs_mismatch)) if diffs_mismatch else 0.0,
            "std_mismatch": float(np.std(diffs_mismatch)) if diffs_mismatch else 0.0,
        },
        "logprob_ea": {
            "mean": float(np.mean(logprob_ea)),
            "std": float(np.std(logprob_ea)),
        },
        "logprob_verifier": {
            "mean": float(np.mean(logprob_ver)),
            "std": float(np.std(logprob_ver)),
        },
        "logprob_ea_proposal": {
            "mean": float(np.mean(logprob_ea_prop)) if logprob_ea_prop else 0.0,
            "std": float(np.std(logprob_ea_prop)) if logprob_ea_prop else 0.0,
            "mean_match": float(np.mean(logprob_ea_prop_match)) if logprob_ea_prop_match else 0.0,
            "std_match": float(np.std(logprob_ea_prop_match)) if logprob_ea_prop_match else 0.0,
            "mean_mismatch": float(np.mean(logprob_ea_prop_mismatch)) if logprob_ea_prop_mismatch else 0.0,
            "std_mismatch": float(np.std(logprob_ea_prop_mismatch)) if logprob_ea_prop_mismatch else 0.0,
        },
        "logprob_gap_ea_minus_verifier": {
            "mean_match": float(np.mean(ea_ver_gap_match)) if ea_ver_gap_match else 0.0,
            "std_match": float(np.std(ea_ver_gap_match)) if ea_ver_gap_match else 0.0,
            "mean_mismatch": float(np.mean(ea_ver_gap_mismatch)) if ea_ver_gap_mismatch else 0.0,
            "std_mismatch": float(np.std(ea_ver_gap_mismatch)) if ea_ver_gap_mismatch else 0.0,
        },
        "entropy": {
            "mean_all": float(np.mean(ent_all)),
            "std_all": float(np.std(ent_all)),
            "mean_match": float(np.mean(ent_match)) if ent_match else 0.0,
            "std_match": float(np.std(ent_match)) if ent_match else 0.0,
            "mean_mismatch": float(np.mean(ent_mismatch)) if ent_mismatch else 0.0,
            "std_mismatch": float(np.std(ent_mismatch)) if ent_mismatch else 0.0,
        },
        "entropy_ea": {
            "mean": float(np.mean(ent_ea_all)) if ent_ea_all else 0.0,
            "std": float(np.std(ent_ea_all)) if ent_ea_all else 0.0,
            "mean_match": float(np.mean(ent_ea_match)) if ent_ea_match else 0.0,
            "std_match": float(np.std(ent_ea_match)) if ent_ea_match else 0.0,
            "mean_mismatch": float(np.mean(ent_ea_mismatch)) if ent_ea_mismatch else 0.0,
            "std_mismatch": float(np.std(ent_ea_mismatch)) if ent_ea_mismatch else 0.0,
        },
        "accept_length": {
            "mean": float(np.mean(accept_lengths)) if accept_lengths else 0.0,
            "median": float(np.median(accept_lengths)) if accept_lengths else 0.0,
            "min": int(min(accept_lengths)) if accept_lengths else 0,
            "max": int(max(accept_lengths)) if accept_lengths else 0,
            "distribution": dict(sorted(accept_dist.items())),
        },
    }

    print("\n=== Key Metrics ===")
    print(f"Match rate:          {summary['match_rate']*100:.2f}%")
    print(f"Mismatch rate:       {summary['mismatch_rate']*100:.2f}%")
    print("\nVerifier logprob diff (ea - verifier top):")
    print(f"  Match:             mean {summary['logprob_diff']['mean_match']:.4f}, std {summary['logprob_diff']['std_match']:.4f}")
    print(f"  Mismatch:          mean {summary['logprob_diff']['mean_mismatch']:.4f}, std {summary['logprob_diff']['std_mismatch']:.4f}")
    print("\nEA vs Verifier (logprob on EA token):")
    if logprob_ea_prop:
        print(f"  EA proposal logprob:")
        print(f"    Match:           mean {summary['logprob_ea_proposal']['mean_match']:.4f}, std {summary['logprob_ea_proposal']['std_match']:.4f}")
        print(f"    Mismatch:        mean {summary['logprob_ea_proposal']['mean_mismatch']:.4f}, std {summary['logprob_ea_proposal']['std_mismatch']:.4f}")
    gap = summary["logprob_gap_ea_minus_verifier"]
    print(f"\nEA - Verifier logprob gap (EA proposal - verifier) on EA token:")
    print(f"  Match:             mean {gap['mean_match']:.4f}, std {gap['std_match']:.4f}")
    print(f"  Mismatch:          mean {gap['mean_mismatch']:.4f}, std {gap['std_mismatch']:.4f}")
    print("\nEntropy:")
    print(f"  Verifier entropy:")
    print(f"    Match:           mean {summary['entropy']['mean_match']:.4f}, std {summary['entropy']['std_match']:.4f}")
    print(f"    Mismatch:        mean {summary['entropy']['mean_mismatch']:.4f}, std {summary['entropy']['std_mismatch']:.4f}")
    if ent_ea_all:
        print(f"  EA proposal entropy:")
        print(f"    Match:           mean {summary['entropy_ea']['mean_match']:.4f}, std {summary['entropy_ea']['std_match']:.4f}")
        print(f"    Mismatch:        mean {summary['entropy_ea']['mean_mismatch']:.4f}, std {summary['entropy_ea']['std_mismatch']:.4f}")

    if args.save_summary:
        out_path = Path(args.save_summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to {out_path}")

    if args.dump_mismatches and mismatches:
        dump_path = Path(args.dump_mismatches)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w") as f:
            for m in mismatches:
                f.write(json.dumps(m) + "\n")
        print(f"Mismatches written to {dump_path}")

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Logprob diff over steps
        plt.figure(figsize=(8, 4))
        plt.plot(diffs_all, alpha=0.6)
        plt.axhline(y=0, color="red", linestyle="--", linewidth=1)
        plt.title("Verifier log-prob difference (ea - verifier top) across steps")
        plt.xlabel("Step index (flattened over questions)")
        plt.ylabel("Log-prob difference")
        plt.tight_layout()
        plt.savefig(plot_dir / "logprob_diff_over_steps.png", dpi=150)
        plt.close()

        # Histogram of logprob diffs
        plt.figure(figsize=(6, 4))
        plt.hist(diffs_all, bins=50, alpha=0.7, color="steelblue")
        plt.axvline(x=np.mean(diffs_all), color="orange", linestyle="--", label="Mean")
        plt.title("Histogram of log-prob differences")
        plt.xlabel("Log-prob difference (ea - verifier top)")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "logprob_diff_hist.png", dpi=150)
        plt.close()

        # Entropy ECDF for match vs mismatch
        if ent_match or ent_mismatch:
            plt.style.use("seaborn-v0_8-whitegrid")
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            colors = {"match": "#1b9e77", "mismatch": "#d95f02"}

            def ecdf(data):
                x = np.sort(data)
                y = np.arange(1, len(x) + 1) / len(x)
                return x, y

            med_match = med_mismatch = None
            if ent_match:
                x_m, y_m = ecdf(ent_match)
                ax.step(x_m, y_m, where="post", color=colors["match"], linewidth=2.0, label=f"match (n={len(ent_match)})")
                med_match = np.median(ent_match)
                iqr_lo_m, iqr_hi_m = np.percentile(ent_match, [25, 75])
                ax.axvspan(iqr_lo_m, iqr_hi_m, color=colors["match"], alpha=0.08)
                ax.axvline(med_match, color=colors["match"], linewidth=1.6)

            if ent_mismatch:
                x_mm, y_mm = ecdf(ent_mismatch)
                ax.step(x_mm, y_mm, where="post", color=colors["mismatch"], linewidth=2.0, label=f"mismatch (n={len(ent_mismatch)})")
                med_mismatch = np.median(ent_mismatch)
                iqr_lo_mm, iqr_hi_mm = np.percentile(ent_mismatch, [25, 75])
                ax.axvspan(iqr_lo_mm, iqr_hi_mm, color=colors["mismatch"], alpha=0.08)
                ax.axvline(med_mismatch, color=colors["mismatch"], linewidth=1.6)

            ax.set_title("Verifier entropy vs EA–verifier agreement", fontsize=14)
            ax.set_xlabel("Verifier entropy (nats)", fontsize=13)
            ax.set_ylabel("ECDF", fontsize=12)
            ax.set_xlim(0, 2.5)
            ax.set_ylim(0, 1.0)
            ax.tick_params(labelsize=11)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(loc="lower right", frameon=False, fontsize=11)

            if med_match is not None and med_mismatch is not None:
                delta_med = med_mismatch - med_match
                y_bracket = 0.6
                ax.plot([med_match, med_mismatch], [y_bracket, y_bracket], color="#4d4d4d", linewidth=1.2)
                ax.plot([med_match, med_match], [y_bracket - 0.02, y_bracket + 0.02], color="#4d4d4d", linewidth=1.0)
                ax.plot([med_mismatch, med_mismatch], [y_bracket - 0.02, y_bracket + 0.02], color="#4d4d4d", linewidth=1.0)
                ax.text(
                    (med_match + med_mismatch) / 2,
                    y_bracket + 0.04,
                    f"Δ median = {delta_med:.2f} nats",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    color="#4d4d4d",
                )

            p_match_lo = np.mean(np.array(ent_match) < 0.05) if ent_match else 0.0
            p_mismatch_lo = np.mean(np.array(ent_mismatch) < 0.05) if ent_mismatch else 0.0
            ax.text(
                0.0,
                -0.18,
                f"P(entropy < 0.05): match={p_match_lo:.3f}, mismatch={p_mismatch_lo:.3f}",
                transform=ax.transAxes,
                fontsize=10,
            )

            plt.tight_layout()
            plt.savefig(plot_dir / "entropy_match_vs_mismatch.png", dpi=300, bbox_inches="tight")
            plt.close()

        print(f"Plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
