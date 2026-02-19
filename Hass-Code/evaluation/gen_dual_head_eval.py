"""Evaluate dual-head HASS (step-level confidence routing) and record τ.

Produces a JSONL file with the same schema as gen_ea_answer_llama3chat.py
(choices[].idxs, choices[].new_tokens) so τ = sum(new_tokens)/sum(idxs).
"""
import argparse
import json
import os
import random
import time

import numpy as np
import shortuuid
import torch
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)

from model.ea_model_dual import DualEaModel
from model.kv_cache import initialize_past_key_values
from model.utils import prepare_logits_processor


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


@torch.inference_mode()
def get_model_answers(
    base_model_path, ea_model_path1, ea_model_path2,
    head1_name, head2_name, model_id,
    questions, answer_file,
    max_new_token, num_choices, temperature, args,
):
    model = DualEaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path1=ea_model_path1,
        ea_model_path2=ea_model_path2,
        head1_name=head1_name,
        head2_name=head2_name,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = model.get_tokenizer()
    model.eval()
    print("Model loaded.")

    # warmup
    question = questions[0]
    for _ in range(3):
        torch.manual_seed(0)
        messages = [{"role": "system", "content": (
            "You are a helpful, respectful and honest assistant."
        )}]
        for j in range(len(question["turns"])):
            messages.append({"role": "user", "content": question["turns"][j]})
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
            output_ids, new_token, idx, _, _ = model.dual_eagenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=temperature,
                log=True,
                is_llama3=True,
                max_new_tokens=32,
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
            messages.append({"role": "assistant", "content": output.strip()})
    print("Warmup done")

    stop_token_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    for question in tqdm(questions):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = [{"role": "system", "content": (
                "You are a helpful, respectful and honest assistant. Always answer as "
                "helpfully as possible, while being safe.  Your answers should not include "
                "any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                "If a question does not make any sense, or is not factually coherent, explain why "
                "instead of answering something not correct. If you don't know the answer to a "
                "question, please don't share false information."
            )}]
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            accept_length_lists = []

            for j in range(len(question["turns"])):
                messages.append({"role": "user", "content": question["turns"][j]})
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids

                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, idx, accept_length_list, _ = model.dual_eagenerate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    log=True,
                    is_llama3=True,
                    max_new_tokens=max_new_token,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time

                output_ids = output_ids[0][len(input_ids[0]):]
                if stop_token_ids:
                    stop_idx = [
                        k for k, tid in enumerate(output_ids) if tid in stop_token_ids
                    ]
                    if stop_idx:
                        output_ids = output_ids[: stop_idx[0]]

                output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for st in special_token:
                            output = output.replace(st, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                accept_length_lists += accept_length_list
                messages.append({"role": "assistant", "content": output})

            choices.append({
                "index": i,
                "turns": turns,
                "idxs": idxs,
                "new_tokens": new_tokens,
                "wall_time": wall_time,
                "accept_length": accept_length_lists,
            })

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            fout.write(json.dumps({
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--ea-model-path1", type=str, required=True)
    parser.add_argument("--ea-model-path2", type=str, required=True)
    parser.add_argument("--head1-name", type=str, default="head1")
    parser.add_argument("--head2-name", type=str, default="head2")
    parser.add_argument("--model-id", type=str, default="dual_hass")
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--answer-file", type=str, default=None)
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_seed(args.seed)

    question_file = os.path.join(parent_dir, "data", args.bench_name, "question.jsonl")
    answer_file = args.answer_file or (
        f"results/dual_head/{args.bench_name}_{args.model_id}.jsonl"
    )
    print(f"Output to {answer_file}")

    questions = load_questions(question_file, None, None)
    get_model_answers(
        base_model_path=args.base_model_path,
        ea_model_path1=args.ea_model_path1,
        ea_model_path2=args.ea_model_path2,
        head1_name=args.head1_name,
        head2_name=args.head2_name,
        model_id=args.model_id,
        questions=questions,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        temperature=args.temperature,
        args=args,
    )
