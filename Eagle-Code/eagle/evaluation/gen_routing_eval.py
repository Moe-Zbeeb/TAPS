"""EAGLE-2 entropy-based head routing evaluation.

Loads 4 EA heads (one per GPU), probes each per sample to measure draft
entropy, routes to the lowest-entropy head, and generates with that head.
Logs per-sample routing decisions and prints a summary table.

Usage:
python -m eagle.evaluation.gen_routing_eval \
    --base-model-path /path/to/Meta-Llama-3-8B-Instruct \
    --bench-name mt_bench gsm8k math_500 svamp \
    --output-dir results/routing
"""
import argparse
import json
import os
import time

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)

from accelerate.utils import set_seed
set_seed(0)

import torch
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

try:
    from ..model.ea_model import EaModel
    from ..model.kv_cache import initialize_past_key_values
    from ..model.utils import *
except:
    from eagle.model.ea_model import EaModel
    from eagle.model.kv_cache import initialize_past_key_values
    from eagle.model.utils import *


HEAD_CONFIGS = [
    ("MathInstruct", "checkpoints/Eagle-MathInstruct_20epochs", 0),
    ("ShareGPT", "checkpoints/Eagle-ShareGPT_20epochs", 1),
    ("ShareGPT-MathInstruct", "checkpoints/Eagle-Sharegpt-Mathinstruct-20epochs", 2),
    ("Averaged", "checkpoints/Eagle-Averaged-Checkpoint", 3),
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


def load_all_models(base_model_path, args):
    models = {}
    for name, ea_path, gpu in HEAD_CONFIGS:
        print(f"Loading {name} on GPU {gpu}...")
        model = EaModel.from_pretrained(
            base_model_path=base_model_path,
            ea_model_path=ea_path,
            total_token=args.total_token,
            depth=args.depth,
            top_k=args.top_k,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": gpu},
            use_eagle3=args.use_eagle3,
        )
        model.eval()
        models[name] = model
    return models


def decode_output(output_ids, tokenizer):
    stop_token_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    stop_indices = [i for i, t in enumerate(output_ids) if t in stop_token_ids]
    if stop_indices:
        output_ids = output_ids[: stop_indices[0]]
    text = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            for tok in special_token:
                text = text.replace(tok, "")
        else:
            text = text.replace(special_token, "")
    return text.strip()


@torch.inference_mode()
def run_routing_eval(base_model_path, bench_names, output_dir, temperature, args):
    models = load_all_models(base_model_path, args)
    # Use tokenizer from first model (all share the same base)
    first_model = next(iter(models.values()))
    tokenizer = first_model.get_tokenizer()

    # Warmup all models
    print("Warming up all models...")
    dummy_text = "Hello, how are you?"
    dummy_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": dummy_text},
    ]
    dummy_prompt = tokenizer.apply_chat_template(
        dummy_messages, tokenize=False, add_generation_prompt=True
    )
    dummy_ids = tokenizer([dummy_prompt], add_special_tokens=False).input_ids
    for name, model in models.items():
        device = next(model.base_model.parameters()).device
        for _ in range(3):
            torch.manual_seed(0)
            model.eagenerate(
                torch.as_tensor(dummy_ids).to(device),
                temperature=temperature,
                log=True,
                is_llama3=True,
                max_new_tokens=32,
            )
    print("Warmup done")

    os.makedirs(output_dir, exist_ok=True)

    all_bench_results = {}

    for bench_name in bench_names:
        print(f"\n{'='*60}")
        print(f"Routing eval: bench={bench_name}")
        print(f"{'='*60}")

        question_file = f"{parent_dir}/data/{bench_name}/question.jsonl"
        questions = load_questions(question_file, None, None)

        output_file = os.path.join(output_dir, f"{bench_name}.jsonl")
        # Clear previous results for this benchmark
        if os.path.exists(output_file):
            os.remove(output_file)

        bench_results = []

        for question in tqdm(questions, desc=bench_name):
            torch.manual_seed(0)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            all_draft_accepted = []
            all_draft_rejected = []
            total_new_tokens = 0
            total_steps = 0

            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                messages.append({"role": "user", "content": qs})
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids

                # Probe all heads
                probe_entropies = {}
                for name, model in models.items():
                    device = next(model.base_model.parameters()).device
                    ent = model.probe_entropy(
                        torch.as_tensor(input_ids).to(device),
                        temperature=temperature,
                    )
                    probe_entropies[name] = ent

                # Route to lowest entropy head
                chosen = min(probe_entropies, key=probe_entropies.get)
                chosen_model = models[chosen]
                chosen_device = next(chosen_model.base_model.parameters()).device

                # Full generation with chosen head
                output_ids, new_token, idx, acc_ent, rej_ent, v_acc_ent, v_rej_ent = chosen_model.eagenerate(
                    torch.as_tensor(input_ids).to(chosen_device),
                    temperature=temperature,
                    log=True,
                    is_llama3=True,
                    entropy_log=True,
                )

                all_draft_accepted.extend(acc_ent)
                all_draft_rejected.extend(rej_ent)
                total_new_tokens += int(new_token)
                total_steps += int(idx)

                # Decode for multi-turn context
                raw_output_ids = output_ids[0][len(input_ids[0]):]
                output_text = decode_output(raw_output_ids, tokenizer)
                messages.append({"role": "assistant", "content": output_text})

            num_accepted = len(all_draft_accepted)
            num_rejected = len(all_draft_rejected)
            total = num_accepted + num_rejected
            acc_ratio = num_accepted / total if total > 0 else 0.0
            mean_accepted_length = total_new_tokens / total_steps if total_steps > 0 else 0.0

            result = {
                "question_id": question["question_id"],
                "chosen_head": chosen,
                "probe_entropies": probe_entropies,
                "num_accepted": num_accepted,
                "num_rejected": num_rejected,
                "acceptance_ratio": acc_ratio,
                "new_tokens": total_new_tokens,
                "steps": total_steps,
                "mean_accepted_length": round(mean_accepted_length, 4),
            }
            bench_results.append(result)

            with open(output_file, "a") as fout:
                fout.write(json.dumps(result) + "\n")

        all_bench_results[bench_name] = bench_results

    # Print summary
    print_summary(all_bench_results, output_dir)


def print_summary(all_bench_results, output_dir):
    header = f"{'Benchmark':<15} {'Strategy':<25} {'Acc. Ratio':>10} {'Head Distribution':<40}"
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for bench_name, results in all_bench_results.items():
        if not results:
            continue

        # Per-head fixed baselines from routing data
        head_counts = {}
        routing_acc_total = 0
        routing_rej_total = 0

        for r in results:
            head = r["chosen_head"]
            head_counts[head] = head_counts.get(head, 0) + 1
            routing_acc_total += r["num_accepted"]
            routing_rej_total += r["num_rejected"]

        routing_total = routing_acc_total + routing_rej_total
        routing_ratio = routing_acc_total / routing_total if routing_total > 0 else 0.0

        # Head distribution string
        abbrevs = {"MathInstruct": "MI", "ShareGPT": "SG",
                    "ShareGPT-MathInstruct": "SM", "Averaged": "AV"}
        dist_parts = []
        for name, _, _ in HEAD_CONFIGS:
            abbr = abbrevs.get(name, name[:2])
            count = head_counts.get(name, 0)
            dist_parts.append(f"{abbr}={count}")
        dist_str = " ".join(dist_parts)

        lines.append(
            f"{bench_name:<15} {'Entropy routing':<25} {routing_ratio:>10.4f} {dist_str:<40}"
        )
        lines.append("")

    lines.append(sep)
    summary = "\n".join(lines)
    print(f"\n{summary}")

    summary_file = os.path.join(output_dir, "routing_summary.txt")
    with open(summary_file, "w") as f:
        f.write(summary + "\n")
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model-path", type=str, required=True,
        help="Path to base model.",
    )
    parser.add_argument(
        "--bench-name", type=str, nargs="+", default=["mt_bench"],
        help="Benchmark name(s). E.g. --bench-name mt_bench gsm8k math_500 svamp",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/routing",
        help="Directory for routing evaluation output files.",
    )
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use_eagle3", action="store_true")

    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}={v}")

    run_routing_eval(
        args.base_model_path,
        args.bench_name,
        args.output_dir,
        args.temperature,
        args,
    )
