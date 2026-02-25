"""Evaluate merged-tree dual-head EAGLE-2.

Both draft heads' trees are merged into a single combined tree and submitted
to the verifier in one pass.  The best accepted path from either head wins.
Outputs a JSONL file compatible with evaluate/compute_tau.py.

Usage:
    python -m eagle.evaluation.gen_merged_tree_eval \
        --base-model-path /path/to/Meta-Llama-3-8B-Instruct \
        --ea-model-path1 checkpoints/Eagle-MathInstruct_20epochs \
        --ea-model-path2 checkpoints/Eagle-ShareGPT_20epochs \
        --head1-name MathInstruct \
        --head2-name ShareGPT \
        --bench-name mt_bench \
        --answer-file results/merged/mt_bench.jsonl
"""

import argparse
import json
import os
import time

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)

from accelerate.utils import set_seed
set_seed(0)

import shortuuid
import torch
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

try:
    from ..model.ea_model_dual import DualEaModel
    from ..model.utils import prepare_logits_processor
except Exception:
    from eagle.model.ea_model_dual import DualEaModel
    from eagle.model.utils import prepare_logits_processor


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
def get_model_answers(model, tokenizer, questions, answer_file,
                      max_new_token, num_choices, temperature, args):
    device = next(model.base_model.parameters()).device

    question = questions[0]
    # Warmup
    for _ in range(3):
        torch.manual_seed(0)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for j in range(len(question["turns"])):
            messages.append({"role": "user", "content": question["turns"][j]})
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
            torch.cuda.synchronize()
            model.merged_eagenerate(
                torch.as_tensor(input_ids).to(device),
                temperature=temperature,
                log=True,
                is_llama3=True,
                max_new_tokens=32,
            )
            torch.cuda.synchronize()
            messages.append({"role": "assistant", "content": ""})
    print("Warmup done")

    for question in tqdm(questions):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []

            for j in range(len(question["turns"])):
                messages.append({"role": "user", "content": question["turns"][j]})
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids

                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, idx = model.merged_eagenerate(
                    torch.as_tensor(input_ids).to(device),
                    temperature=temperature,
                    log=True,
                    is_llama3=True,
                    max_new_tokens=max_new_token,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time

                raw_output_ids = output_ids[0][len(input_ids[0]):]
                output = decode_output(raw_output_ids.tolist(), tokenizer)

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                messages.append({"role": "assistant", "content": output})

            choices.append({
                "index": i,
                "turns": turns,
                "idxs": idxs,
                "new_tokens": new_tokens,
                "wall_time": wall_time,
            })

        os.makedirs(os.path.dirname(os.path.abspath(answer_file)), exist_ok=True)
        with open(answer_file, "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": args.model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    answers = {}
    with open(answer_file, "r") as fin:
        for line in fin:
            qid = json.loads(line)["question_id"]
            answers[qid] = line
    with open(answer_file, "w") as fout:
        for qid in sorted(answers):
            fout.write(answers[qid])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--ea-model-path1", type=str, required=True,
                        help="Path to first draft head (e.g. MathInstruct).")
    parser.add_argument("--ea-model-path2", type=str, required=True,
                        help="Path to second draft head (e.g. ShareGPT).")
    parser.add_argument("--head1-name", type=str, default="head1")
    parser.add_argument("--head2-name", type=str, default="head2")
    parser.add_argument("--model-id", type=str, default="merged-tree-eagle2")
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--question-begin", type=int, default=None)
    parser.add_argument("--question-end", type=int, default=None)
    parser.add_argument("--answer-file", type=str, default=None)
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"  {k}={v}")

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)

    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    answer_file = args.answer_file or f"{args.bench_name}/{args.model_id}.jsonl"
    print(f"Output to {answer_file}")

    questions = load_questions(question_file, args.question_begin, args.question_end)

    print("Loading DualEaModel (merged-tree mode) …")
    model = DualEaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path1=args.ea_model_path1,
        ea_model_path2=args.ea_model_path2,
        head1_name=args.head1_name,
        head2_name=args.head2_name,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    tokenizer = model.get_tokenizer()
    print("Model loaded.")

    get_model_answers(
        model, tokenizer, questions, answer_file,
        args.max_new_token, args.num_choices, args.temperature, args,
    )
    reorg_answer_file(answer_file)


if __name__ == "__main__":
    main()
