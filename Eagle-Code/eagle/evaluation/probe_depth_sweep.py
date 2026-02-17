"""Sweep probe tree size to measure routing accuracy at different depths.

For each total_token value in [5, 10, 20, 30, 40, 50, 60], probes 2 heads
(MathInstruct + ShareGPT) and records the routing decision. Compares against
oracle from existing entropy data.

Loads both heads on each GPU, runs one benchmark per GPU in parallel (4 GPUs).

Usage:
python -m eagle.evaluation.probe_depth_sweep \
    --base-model-path /path/to/Meta-Llama-3-8B-Instruct \
    --bench-name mt_bench gsm8k math_500 svamp
"""
import argparse
import json
import os
import torch.multiprocessing as mp

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)

from accelerate.utils import set_seed

import torch
from fastchat.llm_judge.common import load_questions

try:
    from ..model.ea_model import EaModel
    from ..model.kv_cache import initialize_past_key_values
    from ..model.utils import *
except:
    from eagle.model.ea_model import EaModel
    from eagle.model.kv_cache import initialize_past_key_values
    from eagle.model.utils import *


HEAD_CONFIGS = [
    ("MathInstruct", "checkpoints/Eagle-MathInstruct_20epochs"),
    ("ShareGPT", "checkpoints/Eagle-ShareGPT_20epochs"),
]

SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe.  Your answers should not include "
    "any harmful, unethical, racist, sexist, toxic, dangerous, or illegal "
    "content. Please ensure that your responses are socially unbiased and "
    "positive in nature.\n\nIf a question does not make any sense, or is not "
    "factually coherent, explain why instead of answering something not correct. "
    "If you don't know the answer to a question, please don't share false "
    "information."
)

TOTAL_TOKEN_VALUES = [5, 10, 20, 30, 40, 50, 60]
HEADS = [name for name, _ in HEAD_CONFIGS]
ALL_HEADS = ["MathInstruct", "ShareGPT", "ShareGPT-MathInstruct", "Averaged"]


def load_oracle_data():
    """Load oracle (best head per sample) from existing entropy data."""
    entropy_dir = "results/entropy"
    oracle = {}
    benchmarks = ["mt_bench", "gsm8k", "math_500", "svamp"]
    for bench in benchmarks:
        oracle[bench] = {}
        head_data = {}
        for head in ALL_HEADS:
            path = os.path.join(entropy_dir, f"{head}_{bench}.jsonl")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    qid = rec["question_id"]
                    na = rec["num_accepted"]
                    nr = rec["num_rejected"]
                    total = na + nr
                    ar = na / total if total > 0 else 0.0
                    if qid not in head_data:
                        head_data[qid] = {}
                    head_data[qid][head] = ar
        for qid, hd in head_data.items():
            oracle[bench][qid] = max(hd, key=hd.get)
    return oracle


def run_one_benchmark(gpu_id, bench_name, base_model_path, depth, top_k, use_eagle3, oracle, result_dict):
    """Run probe sweep for one benchmark on one GPU."""
    set_seed(0)
    from tqdm import tqdm

    print(f"[GPU {gpu_id}] Loading MathInstruct + ShareGPT for {bench_name}...")
    models = {}
    for name, ea_path in HEAD_CONFIGS:
        model = EaModel.from_pretrained(
            base_model_path=base_model_path,
            ea_model_path=ea_path,
            total_token=max(TOTAL_TOKEN_VALUES),
            depth=depth,
            top_k=top_k,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": gpu_id},
            use_eagle3=use_eagle3,
        )
        model.eval()
        models[name] = model

    tokenizer = next(iter(models.values())).get_tokenizer()

    # Warmup
    dummy_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hello"},
    ]
    dummy_prompt = tokenizer.apply_chat_template(
        dummy_messages, tokenize=False, add_generation_prompt=True
    )
    dummy_ids = tokenizer([dummy_prompt], add_special_tokens=False).input_ids
    for name, model in models.items():
        for _ in range(3):
            torch.manual_seed(0)
            model.eagenerate(
                torch.as_tensor(dummy_ids).to(f"cuda:{gpu_id}"),
                temperature=0.0, log=True, is_llama3=True, max_new_tokens=32,
            )
    print(f"[GPU {gpu_id}] Warmup done for {bench_name}")

    question_file = f"{parent_dir}/data/{bench_name}/question.jsonl"
    questions = load_questions(question_file, None, None)
    bench_results = {tt: {"correct": 0, "total": 0} for tt in TOTAL_TOKEN_VALUES}

    with torch.inference_mode():
        for question in tqdm(questions, desc=f"[GPU{gpu_id}] {bench_name}"):
            torch.manual_seed(0)
            qid = question["question_id"]
            oracle_head = oracle.get(bench_name, {}).get(qid)
            if oracle_head is None:
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question["turns"][0]},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer([prompt], add_special_tokens=False).input_ids

            for tt in TOTAL_TOKEN_VALUES:
                for name, model in models.items():
                    model.ea_layer.total_tokens = tt

                probe_entropies = {}
                for name, model in models.items():
                    ent = model.probe_entropy(
                        torch.as_tensor(input_ids).to(f"cuda:{gpu_id}"),
                        temperature=0.0,
                    )
                    probe_entropies[name] = ent

                chosen = min(probe_entropies, key=probe_entropies.get)
                bench_results[tt]["total"] += 1
                if chosen == oracle_head:
                    bench_results[tt]["correct"] += 1

    result_dict[bench_name] = bench_results
    print(f"[GPU {gpu_id}] Done: {bench_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--bench-name", type=str, nargs="+",
                        default=["mt_bench", "gsm8k", "math_500", "svamp"])
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--use_eagle3", action="store_true")
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}={v}")

    oracle = load_oracle_data()

    bench_names = args.bench_name
    num_gpus = min(len(bench_names), torch.cuda.device_count())
    print(f"Running {len(bench_names)} benchmarks across {num_gpus} GPUs")

    # Use mp.Manager for shared result dict
    manager = mp.Manager()
    result_dict = manager.dict()

    processes = []
    for i, bench in enumerate(bench_names):
        gpu_id = i % num_gpus
        p = mp.Process(
            target=run_one_benchmark,
            args=(gpu_id, bench, args.base_model_path,
                  args.depth, args.top_k, args.use_eagle3,
                  oracle, result_dict),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results = dict(result_dict)

    # Print results table
    print(f"\n{'='*70}")
    print("Routing Accuracy vs Probe Tree Size (total_token)")
    print(f"Heads: MathInstruct + ShareGPT only")
    print(f"{'='*70}")
    header = f"{'total_token':>12}"
    for bench in bench_names:
        header += f"  {bench:>10}"
    header += f"  {'Overall':>10}"
    print(header)
    print("-" * len(header))

    for tt in TOTAL_TOKEN_VALUES:
        row = f"{tt:>12}"
        total_c, total_n = 0, 0
        for bench in bench_names:
            if bench not in results:
                row += f"  {'N/A':>10}"
                continue
            key = tt if tt in results[bench] else str(tt)
            c = results[bench][key]["correct"]
            n = results[bench][key]["total"]
            acc = c / n * 100 if n > 0 else 0
            row += f"  {acc:>9.1f}%"
            total_c += c
            total_n += n
        overall = total_c / total_n * 100 if total_n > 0 else 0
        row += f"  {overall:>9.1f}%"
        print(row)

    # Save
    output_dir = "results/routing_v2"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "probe_depth_sweep.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
