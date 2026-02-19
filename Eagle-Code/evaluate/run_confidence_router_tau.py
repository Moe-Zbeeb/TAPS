#!/usr/bin/env python3
"""
Route between MathInstruct and ShareGPT using confidence, with tie-break to
ShareGPT-MathInstruct, then compute tau using evaluate/compute_tau.py logic.

Routing rule:
1) Compare overall confidence score for MathInstruct vs ShareGPT.
2) If abs(delta) < eps, choose ShareGPT-MathInstruct.
3) Else choose the higher-confidence model.

Input files expected in input_dir:
  - MathInstruct_<bench>.jsonl
  - ShareGPT_<bench>.jsonl
  - ShareGPT-MathInstruct_<bench>.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


BENCHMARKS = ["mt_bench", "gsm8k", "math_500", "svamp"]
HEADS = ["MathInstruct", "ShareGPT", "ShareGPT-MathInstruct"]


def load_jsonl_by_qid(path: Path) -> Dict[int, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows[int(rec["question_id"])] = rec
    return rows


def overall_confidence(rec: dict) -> float:
    na = int(rec.get("num_accepted", 0))
    nr = int(rec.get("num_rejected", 0))
    total = na + nr
    if total <= 0:
        return 0.0
    ma = float(rec.get("mean_accepted_confidence", 0.0))
    mr = float(rec.get("mean_rejected_confidence", 0.0))
    return (ma * na + mr * nr) / total


def to_compute_tau_record(qid: int, bench: str, chosen_head: str, chosen_rec: dict, score_mi: float, score_sg: float, eps: float) -> dict:
    new_tokens = int(chosen_rec.get("new_tokens", 0))
    steps = int(chosen_rec.get("steps", 0))
    return {
        "question_id": qid,
        "benchmark": bench,
        "model_id": "confidence_router_mi_sg_tie_sm",
        "router_meta": {
            "rule": "max_confidence_math_vs_sharegpt_tie_sharegpt_mathinstruct",
            "eps": eps,
            "score_mathinstruct": score_mi,
            "score_sharegpt": score_sg,
            "chosen_head": chosen_head,
        },
        "choices": [
            {
                "index": 0,
                "turns": [],
                "idxs": [steps],
                "new_tokens": [new_tokens],
                # Not available in confidence logs; keep 0 for tau-only usage.
                "wall_time": [0.0],
            }
        ],
    }


def compute_tau(records: List[dict]) -> dict:
    total_new_tokens = 0
    total_idxs = 0
    total_wall_time = 0.0
    num_questions = 0
    for rec in records:
        for choice in rec.get("choices", []):
            total_new_tokens += sum(choice.get("new_tokens", []))
            total_idxs += sum(choice.get("idxs", []))
            total_wall_time += sum(choice.get("wall_time", []))
        num_questions += 1
    tau = total_new_tokens / total_idxs if total_idxs > 0 else 0.0
    tps = total_new_tokens / total_wall_time if total_wall_time > 0 else 0.0
    return {
        "num_questions": num_questions,
        "total_new_tokens": total_new_tokens,
        "total_iterations": total_idxs,
        "tau": round(tau, 4),
        "tokens_per_sec": round(tps, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="results/confidence")
    parser.add_argument("--output-dir", type=str, default="results/confidence_routing")
    parser.add_argument("--bench-name", nargs="+", default=BENCHMARKS)
    parser.add_argument("--eps", type=float, default=0.01)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    print("\n" + "=" * 78)
    print(f"{'Benchmark':<14} {'Tau':>8} {'Questions':>10} {'MI':>8} {'SG':>8} {'SM(tie)':>10}")
    print("=" * 78)

    for bench in args.bench_name:
        per_head = {}
        for head in HEADS:
            path = input_dir / f"{head}_{bench}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Missing input file: {path}")
            per_head[head] = load_jsonl_by_qid(path)

        common_qids = sorted(set(per_head["MathInstruct"]).intersection(
            set(per_head["ShareGPT"]),
            set(per_head["ShareGPT-MathInstruct"]),
        ))

        out_records = []
        pick_counts = {"MathInstruct": 0, "ShareGPT": 0, "ShareGPT-MathInstruct": 0}

        for qid in common_qids:
            rec_mi = per_head["MathInstruct"][qid]
            rec_sg = per_head["ShareGPT"][qid]
            rec_sm = per_head["ShareGPT-MathInstruct"][qid]

            score_mi = overall_confidence(rec_mi)
            score_sg = overall_confidence(rec_sg)

            if abs(score_mi - score_sg) < args.eps:
                chosen_head = "ShareGPT-MathInstruct"
                chosen_rec = rec_sm
            elif score_mi > score_sg:
                chosen_head = "MathInstruct"
                chosen_rec = rec_mi
            else:
                chosen_head = "ShareGPT"
                chosen_rec = rec_sg

            pick_counts[chosen_head] += 1
            out_records.append(
                to_compute_tau_record(
                    qid=qid,
                    bench=bench,
                    chosen_head=chosen_head,
                    chosen_rec=chosen_rec,
                    score_mi=score_mi,
                    score_sg=score_sg,
                    eps=args.eps,
                )
            )

        out_path = output_dir / f"{bench}_confidence_router.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in out_records:
                f.write(json.dumps(rec) + "\n")

        stats = compute_tau(out_records)
        summary[bench] = {
            **stats,
            "selection_counts": pick_counts,
            "output_file": str(out_path),
        }
        print(
            f"{bench:<14} {stats['tau']:>8.4f} {stats['num_questions']:>10d} "
            f"{pick_counts['MathInstruct']:>8d} {pick_counts['ShareGPT']:>8d} {pick_counts['ShareGPT-MathInstruct']:>10d}"
        )

    print("=" * 78)

    summary_path = output_dir / "tau_summary_confidence_router.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    # Helper command for exact compatibility with existing tool.
    print("\nRun compute_tau.py on generated files:")
    files = " ".join(
        str(output_dir / f"{b}_confidence_router.jsonl") for b in args.bench_name
    )
    print(f"python evaluate/compute_tau.py {files}")


if __name__ == "__main__":
    main()

