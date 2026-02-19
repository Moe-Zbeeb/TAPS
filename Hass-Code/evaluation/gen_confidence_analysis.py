"""Confidence analysis for HASS draft tokens.

Computes mean draft confidence for accepted vs rejected draft tokens across
benchmarks and checkpoints, then prints a summary table.
"""

import argparse
import json
import os

from accelerate.utils import set_seed
set_seed(0)

import torch
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

from model.ea_model import EaModel
from model.kv_cache import initialize_past_key_values
from model.utils import (
    prepare_logits_processor,
    initialize_tree,
    tree_decoding,
    evaluate_posterior,
    reset_tree_mode,
)


SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe.  Your answers should not include "
    "any harmful, unethical, racist, sexist, toxic, dangerous, or illegal "
    "content. Please ensure that your responses are socially unbiased and "
    "positive in nature.\\n\\nIf a question does not make any sense, or is not "
    "factually coherent, explain why instead of answering something not correct. "
    "If you don't know the answer to a question, please don't share false "
    "information."
)

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


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


def summarize_file(path):
    all_acc = []
    all_rej = []
    if not os.path.exists(path):
        return 0.0, 0.0, 0, 0
    with open(path, "r") as fin:
        for line in fin:
            rec = json.loads(line)
            all_acc.extend(rec["accepted_confidences"])
            all_rej.extend(rec["rejected_confidences"])
    mean_acc = sum(all_acc) / len(all_acc) if all_acc else 0.0
    mean_rej = sum(all_rej) / len(all_rej) if all_rej else 0.0
    return mean_acc, mean_rej, len(all_acc), len(all_rej)


@torch.no_grad()
def _update_inference_inputs_with_logprobs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values_data_list,
        current_length_data,
        model,
        hidden_state_new,
        sample_p,
):
    prev_input_len = input_ids.shape[1]

    select_indices = retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)],
        dim=-1,
    )

    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        dst.copy_(tgt, non_blocking=True)

    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]

    if logits_processor is not None:
        token = torch.multinomial(sample_p, 1)[None]
    else:
        token = torch.argmax(sample_p)[None, None]

    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, draft_logprobs = model.ea_layer.topK_genrate(
        accept_hidden_state_new,
        input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
        head=model.base_model.lm_head,
        logits_processor=logits_processor,
        return_logprobs=True,
    )

    new_token += int(accept_length) + 1
    return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, token, draft_logprobs


@torch.inference_mode()
def eagenerate_with_confidence(
        model,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        is_llama3=True,
):
    if is_llama3:
        stop_token_id = model.tokenizer.convert_tokens_to_ids("<|eot_id|>")

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
    else:
        logits_processor = None

    padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
    input_ids = input_ids.clone()
    model.ea_layer.reset_kv()

    accepted_confidences = []
    rejected_confidences = []

    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    reset_tree_mode(model)

    init = initialize_tree(
        input_ids,
        model,
        past_key_values,
        logits_processor,
        return_logprobs=True,
    )
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _, draft_logprobs = init

    new_token = 0
    max_length = max_length - model.ea_layer.total_tokens - 10

    for idx in range(max_length):
        model.base_model.model.tree_mask = tree_mask

        draft_tokens = draft_tokens.to(input_ids.device)
        logits, hidden_state_new, _ = tree_decoding(
            model,
            draft_tokens,
            past_key_values,
            tree_position_ids,
            input_ids,
            retrieve_indices,
        )

        draft_tokens = torch.cat((draft_tokens, padding), dim=1)
        candidates = draft_tokens[0, retrieve_indices]

        best_candidate, accept_length, sample_p = evaluate_posterior(logits, candidates, logits_processor)

        bc = int(best_candidate)
        al = int(accept_length)

        if draft_logprobs is not None and draft_logprobs.shape[0] > bc:
            ea_lp_path = draft_logprobs[bc]
            num_pred_pos = min(ea_lp_path.shape[0], candidates.shape[1] - 1)
            for pos in range(num_pred_pos):
                tok = int(candidates[bc, pos + 1])
                if tok < 0:
                    continue
                conf = float(torch.exp(ea_lp_path[pos, tok]))
                if pos < al:
                    accepted_confidences.append(conf)
                else:
                    rejected_confidences.append(conf)

        input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, _, draft_logprobs = _update_inference_inputs_with_logprobs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            retrieve_indices,
            logits_processor,
            new_token,
            past_key_values_data,
            current_length_data,
            model,
            hidden_state_new,
            sample_p,
        )

        if is_llama3 and stop_token_id in input_ids[0, input_len:].tolist():
            break
        if model.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > max_new_tokens:
            break
        if input_ids.shape[1] > max_length:
            break

    return input_ids, new_token, idx, accepted_confidences, rejected_confidences


@torch.inference_mode()
def run_confidence_analysis_for_model(base_model_path, ea_model_path, model_id, args):
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    tokenizer = model.get_tokenizer()
    device = next(model.base_model.parameters()).device

    os.makedirs(args.output_dir, exist_ok=True)
    summary_rows = []

    for bench_name in args.bench_name:
        question_file = os.path.join(REPO_ROOT, "data", bench_name, "question.jsonl")
        questions = load_questions(question_file, None, None)
        out_file = os.path.join(args.output_dir, f"{model_id}_{bench_name}.jsonl")
        if os.path.exists(out_file):
            os.remove(out_file)

        for question in tqdm(questions, desc=f"{model_id}/{bench_name}"):
            torch.manual_seed(0)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            all_acc = []
            all_rej = []
            total_new_tokens = 0
            total_steps = 0

            for turn in question["turns"]:
                messages.append({"role": "user", "content": turn})
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids

                output_ids, new_token, idx, acc_conf, rej_conf = eagenerate_with_confidence(
                    model,
                    torch.as_tensor(input_ids).to(device),
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_token,
                    is_llama3=True,
                )

                all_acc.extend(acc_conf)
                all_rej.extend(rej_conf)
                total_new_tokens += int(new_token)
                total_steps += int(idx)

                raw_output_ids = output_ids[0][len(input_ids[0]):]
                output_text = decode_output(raw_output_ids, tokenizer)
                messages.append({"role": "assistant", "content": output_text})

            mean_acc = sum(all_acc) / len(all_acc) if all_acc else 0.0
            mean_rej = sum(all_rej) / len(all_rej) if all_rej else 0.0
            mean_accepted_length = total_new_tokens / total_steps if total_steps > 0 else 0.0

            rec = {
                "question_id": question["question_id"],
                "accepted_confidences": all_acc,
                "rejected_confidences": all_rej,
                "mean_accepted_confidence": mean_acc,
                "mean_rejected_confidence": mean_rej,
                "num_accepted": len(all_acc),
                "num_rejected": len(all_rej),
                "new_tokens": total_new_tokens,
                "steps": total_steps,
                "mean_accepted_length": round(mean_accepted_length, 4),
            }
            with open(out_file, "a") as fout:
                fout.write(json.dumps(rec) + "\n")

        mean_acc, mean_rej, n_acc, n_rej = summarize_file(out_file)
        summary_rows.append((bench_name, mean_acc, mean_rej, n_acc, n_rej))
        print(
            f"[{model_id} / {bench_name}] "
            f"Accepted={mean_acc:.6f} ({n_acc}) | "
            f"Rejected={mean_rej:.6f} ({n_rej}) | "
            f"Delta={mean_acc - mean_rej:+.6f}"
        )

    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--ea-model-paths", nargs="+", required=True)
    parser.add_argument("--model-ids", nargs="+", default=None)
    parser.add_argument(
        "--bench-name",
        type=str,
        nargs="+",
        default=["mt_bench", "gsm8k", "math_500", "svamp"],
    )
    parser.add_argument("--output-dir", type=str, default="results/confidence")
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    if args.model_ids is not None and len(args.model_ids) != len(args.ea_model_paths):
        raise ValueError("--model-ids length must match --ea-model-paths length")

    model_ids = args.model_ids
    if model_ids is None:
        model_ids = [os.path.basename(os.path.normpath(p)) for p in args.ea_model_paths]

    full_summary = {}
    for model_id, ea_path in zip(model_ids, args.ea_model_paths):
        print(f"\\n{'='*60}")
        print(f"Running confidence analysis for {model_id}")
        print(f"EA path: {ea_path}")
        print(f"{'='*60}")
        summary_rows = run_confidence_analysis_for_model(args.base_model_path, ea_path, model_id, args)
        full_summary[model_id] = summary_rows

    print(f"\\n{'='*80}")
    print("Summary (mean draft confidence)")
    print(f"{'='*80}")
    print(f"{'Model':<32} {'Benchmark':<12} {'Accepted':>12} {'Rejected':>12} {'Delta':>12}")
    for model_id, rows in full_summary.items():
        for bench_name, mean_acc, mean_rej, _, _ in rows:
            print(
                f"{model_id:<32} {bench_name:<12} "
                f"{mean_acc:>12.6f} {mean_rej:>12.6f} {(mean_acc - mean_rej):>12.6f}"
            )


if __name__ == "__main__":
    main()
