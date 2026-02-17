"""Entropy analysis for EAGLE-2 speculative decoding.

Measures Shannon entropy at two levels:
1. Draft (EA head) — entropy of the draft model's logit distributions
2. Verifier (target LLM) — entropy of the target model's logit distributions

For each draft token, classifies as accepted or rejected and compares
mean entropy between the two groups at both levels.

Usage:
python -m eagle.evaluation.gen_entropy_analysis \
    --ea-model-path checkpoints/Eagle-MathInstruct_20epochs \
    --base-model-path /path/to/Meta-Llama-3-8B-Instruct \
    --model-id MathInstruct \
    --bench-name mt_bench gsm8k math_500 svamp \
    --output-dir results/entropy
"""
import argparse
import json
import os

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)

from accelerate.utils import set_seed
set_seed(0)

import time
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


@torch.inference_mode()
def run_entropy_analysis(
        base_model_path,
        ea_model_path,
        model_id,
        bench_names,
        output_dir,
        max_new_token,
        temperature,
        args,
):
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_eagle3=args.use_eagle3,
    )

    tokenizer = model.get_tokenizer()
    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    # Warmup with first benchmark's first question
    first_bench = bench_names[0]
    question_file = f"{parent_dir}/data/{first_bench}/question.jsonl"
    warmup_questions = load_questions(question_file, None, None)
    if warmup_questions:
        question = warmup_questions[0]
        for _ in range(3):
            torch.manual_seed(0)
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                messages.append({"role": "user", "content": qs})
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
                output_ids, _, _ = model.eagenerate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    log=True,
                    is_llama3=True,
                )
                output_ids = output_ids[0][len(input_ids[0]):]
                stop_token_ids = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                if stop_token_ids:
                    stop_token_ids_index = [
                        i for i, id in enumerate(output_ids) if id in stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]
                output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()
                messages.append({"role": "assistant", "content": output})
        print('Warmup done')

    os.makedirs(output_dir, exist_ok=True)

    for bench_name in bench_names:
        print(f"\n{'='*60}")
        print(f"Running entropy analysis: model={model_id}, bench={bench_name}")
        print(f"{'='*60}")

        question_file = f"{parent_dir}/data/{bench_name}/question.jsonl"
        questions = load_questions(question_file, None, None)

        output_file = os.path.join(output_dir, f"{model_id}_{bench_name}.jsonl")

        for question in tqdm(questions, desc=f"{model_id}/{bench_name}"):
            torch.manual_seed(0)
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]

            all_draft_accepted = []
            all_draft_rejected = []
            all_verifier_accepted = []
            all_verifier_rejected = []
            total_new_tokens = 0
            total_steps = 0

            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                messages.append({"role": "user", "content": qs})
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids

                output_ids, new_token, idx, acc_ent, rej_ent, v_acc_ent, v_rej_ent = model.eagenerate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    log=True,
                    is_llama3=True,
                    entropy_log=True,
                )

                all_draft_accepted.extend(acc_ent)
                all_draft_rejected.extend(rej_ent)
                all_verifier_accepted.extend(v_acc_ent)
                all_verifier_rejected.extend(v_rej_ent)
                total_new_tokens += int(new_token)
                total_steps += int(idx)

                # Decode output for multi-turn conversation context
                output_ids = output_ids[0][len(input_ids[0]):]
                stop_token_ids = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                if stop_token_ids:
                    stop_token_ids_index = [
                        i for i, id in enumerate(output_ids) if id in stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]
                output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()
                messages.append({"role": "assistant", "content": output})

            mean_draft_acc = sum(all_draft_accepted) / len(all_draft_accepted) if all_draft_accepted else 0.0
            mean_draft_rej = sum(all_draft_rejected) / len(all_draft_rejected) if all_draft_rejected else 0.0
            mean_verifier_acc = sum(all_verifier_accepted) / len(all_verifier_accepted) if all_verifier_accepted else 0.0
            mean_verifier_rej = sum(all_verifier_rejected) / len(all_verifier_rejected) if all_verifier_rejected else 0.0

            mean_accepted_length = total_new_tokens / total_steps if total_steps > 0 else 0.0

            result = {
                "question_id": question["question_id"],
                "draft_accepted_entropies": all_draft_accepted,
                "draft_rejected_entropies": all_draft_rejected,
                "verifier_accepted_entropies": all_verifier_accepted,
                "verifier_rejected_entropies": all_verifier_rejected,
                "mean_draft_accepted": mean_draft_acc,
                "mean_draft_rejected": mean_draft_rej,
                "mean_verifier_accepted": mean_verifier_acc,
                "mean_verifier_rejected": mean_verifier_rej,
                "num_accepted": len(all_draft_accepted),
                "num_rejected": len(all_draft_rejected),
                "new_tokens": total_new_tokens,
                "steps": total_steps,
                "mean_accepted_length": round(mean_accepted_length, 4),
            }

            with open(output_file, "a") as fout:
                fout.write(json.dumps(result) + "\n")

        # Print per-benchmark summary
        all_d_acc = []
        all_d_rej = []
        all_v_acc = []
        all_v_rej = []
        with open(output_file, "r") as fin:
            for line in fin:
                rec = json.loads(line)
                all_d_acc.extend(rec["draft_accepted_entropies"])
                all_d_rej.extend(rec["draft_rejected_entropies"])
                all_v_acc.extend(rec["verifier_accepted_entropies"])
                all_v_rej.extend(rec["verifier_rejected_entropies"])

        mean_da = sum(all_d_acc) / len(all_d_acc) if all_d_acc else 0.0
        mean_dr = sum(all_d_rej) / len(all_d_rej) if all_d_rej else 0.0
        mean_va = sum(all_v_acc) / len(all_v_acc) if all_v_acc else 0.0
        mean_vr = sum(all_v_rej) / len(all_v_rej) if all_v_rej else 0.0
        print(f"\n[{model_id} / {bench_name}]")
        print(f"  Draft:    Accepted={mean_da:.4f} ({len(all_d_acc)}) | Rejected={mean_dr:.4f} ({len(all_d_rej)}) | Delta={mean_dr - mean_da:+.4f}")
        print(f"  Verifier: Accepted={mean_va:.4f} ({len(all_v_acc)}) | Rejected={mean_vr:.4f} ({len(all_v_rej)}) | Delta={mean_vr - mean_va:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea-model-path", type=str, required=True,
        help="Path to EAGLE model weights.",
    )
    parser.add_argument(
        "--base-model-path", type=str, required=True,
        help="Path to base model.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name", type=str, nargs="+", default=["mt_bench"],
        help="Benchmark name(s). E.g. --bench-name mt_bench gsm8k math_500 svamp",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/entropy",
        help="Directory for entropy analysis output files.",
    )
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use_eagle3", action="store_true")

    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}={v}")

    run_entropy_analysis(
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        args.bench_name,
        args.output_dir,
        args.max_new_token,
        args.temperature,
        args,
    )
