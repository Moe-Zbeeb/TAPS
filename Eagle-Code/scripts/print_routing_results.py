import json, os, sys

output_dir = sys.argv[1] if len(sys.argv) > 1 else "results/routing_v2"
benchmarks = ["mt_bench", "gsm8k", "math_500", "svamp"]

print("Benchmark      Mean Accepted Length    N")
print("-" * 45)
all_tokens, all_steps = 0, 0
for bench in benchmarks:
    path = os.path.join(output_dir, bench + ".jsonl")
    if not os.path.exists(path):
        print(f"{bench:<12}   (missing)")
        continue
    tokens, steps, n = 0, 0, 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            tokens += rec["new_tokens"]
            steps += rec["steps"]
            n += 1
    mal = tokens / steps if steps > 0 else 0
    all_tokens += tokens
    all_steps += steps
    print(f"{bench:<12}   {mal:>8.2f}              {n:>5}")
overall = all_tokens / all_steps if all_steps > 0 else 0
print("-" * 45)
print(f"Overall        {overall:>8.2f}")
