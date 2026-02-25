import argparse
import json
import os


def summarize_file(path):
    accepted = []
    rejected = []
    if not os.path.exists(path):
        return 0.0, 0.0, 0, 0
    with open(path, "r") as fin:
        for line in fin:
            rec = json.loads(line)
            accepted.extend(rec.get("accepted_confidences", []))
            rejected.extend(rec.get("rejected_confidences", []))
    mean_acc = sum(accepted) / len(accepted) if accepted else 0.0
    mean_rej = sum(rejected) / len(rejected) if rejected else 0.0
    return mean_acc, mean_rej, len(accepted), len(rejected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="results/confidence")
    parser.add_argument("--model-ids", nargs="+", required=True)
    parser.add_argument(
        "--bench-name",
        nargs="+",
        default=["mt_bench", "gsm8k", "math_500", "svamp"],
    )
    args = parser.parse_args()

    print(f"{'Model':<24} {'Benchmark':<12} {'Accepted':>12} {'Rejected':>12} {'Delta':>12} {'N_acc':>10} {'N_rej':>10}")
    for model_id in args.model_ids:
        for bench in args.bench_name:
            path = os.path.join(args.output_dir, f"{model_id}_{bench}.jsonl")
            mean_acc, mean_rej, n_acc, n_rej = summarize_file(path)
            print(
                f"{model_id:<24} {bench:<12} "
                f"{mean_acc:>12.6f} {mean_rej:>12.6f} {(mean_acc - mean_rej):>12.6f} "
                f"{n_acc:>10} {n_rej:>10}"
            )


if __name__ == "__main__":
    main()
