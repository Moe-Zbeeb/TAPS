#!/usr/bin/env python
"""
Trace verifier vs EA token choices at each acceptance step on GSM8K-style questions.

Per compared position we log:
- accepted flag (prefix vs first rejection)
- EA token, verifier top token
- logprob difference under the verifier (ea - verifier top)
- entropy of verifier distribution
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.ea_model import EaModel  # noqa: E402
from model.kv_cache import initialize_past_key_values  # noqa: E402
from model.utils import (  # noqa: E402
    prepare_logits_processor,
    initialize_tree,
    tree_decoding,
    evaluate_posterior,
    update_inference_inputs,
    reset_tree_mode,
)


SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
    "while being safe.  Your answers should not include any harmful, unethical, racist, sexist, "
    "toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased "
    "and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of "
    "answering something not correct. If you don't know the answer to a question, please don't share "
    "false information."
)


def load_questions(path: Path, n_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    with path.open() as f:
        data = [json.loads(line) for line in f if line.strip()]
    random.seed(seed)
    random.shuffle(data)
    return data[:n_samples]


def prepare_inputs(tokenizer, question: Dict[str, Any], device: torch.device) -> torch.Tensor:
    prompt = question["turns"][0] if isinstance(question.get("turns"), list) else question.get("question", "")
    template = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer([template], add_special_tokens=False, return_tensors="pt")
    return encoded.input_ids.to(device)


@torch.inference_mode()
def trace_accept_steps(
    model: EaModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    device = input_ids.device
    padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(device)
    logits_processor = prepare_logits_processor(temperature=temperature) if temperature > 1e-5 else None

    # Reset caches
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

    if hasattr(model, "ea_layer") and hasattr(model.ea_layer, "stable_kv"):
        model.ea_layer.stable_kv = None

    reset_tree_mode(model)
    input_len = input_ids.shape[1]
    max_length = 2048 - model.ea_layer.total_tokens - 10

    init = initialize_tree(input_ids, model, past_key_values, logits_processor, return_logprobs=True)
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token, draft_logprobs = init

    new_token = 0
    step_logs: List[Dict[str, Any]] = []

    for step in range(max_length):
        model.base_model.model.tree_mask = tree_mask
        draft_tokens = draft_tokens.to(device)

        logits, hidden_state_new, outputs = tree_decoding(
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

        best_candidate_idx = int(best_candidate)
        accept_len_int = int(accept_length)

        cand_seq = candidates[best_candidate_idx]
        log_probs = F.log_softmax(logits[best_candidate_idx], dim=-1)
        positions = min(log_probs.shape[0], cand_seq.shape[0] - 1)  # logits align with cand_seq[1:]

        for pos in range(positions):
            ea_tok = int(cand_seq[pos + 1])
            top_tok = int(torch.argmax(log_probs[pos]))
            accepted = pos < accept_len_int
            prob_pos = torch.exp(log_probs[pos])
            entropy = float(-(prob_pos * log_probs[pos]).sum())
            # EA proposal logprob/entropy (aligned with retrieve_indices ordering)
            if draft_logprobs is not None and draft_logprobs.shape[0] > best_candidate_idx:
                ea_pos_lp = draft_logprobs[best_candidate_idx]
                if pos < ea_pos_lp.shape[0]:
                    ea_log_probs_pos = ea_pos_lp[pos]
                    ea_entropy = float(-(torch.exp(ea_log_probs_pos) * ea_log_probs_pos).sum())
                    ea_logprob = float(ea_log_probs_pos[ea_tok])
                else:
                    ea_entropy = None
                    ea_logprob = None
            else:
                ea_entropy = None
                ea_logprob = None
            step_logs.append(
                {
                    "step": step,
                    "pos": pos,
                    "accepted": accepted,
                    "ea_token": ea_tok,
                    "verifier_token": top_tok,
                    "logprob_ea_token": float(log_probs[pos, ea_tok]),
                    "logprob_verifier_token": float(log_probs[pos, top_tok]),
                    "logprob_diff": float(log_probs[pos, ea_tok] - log_probs[pos, top_tok]),
                    "accept_length": accept_len_int,
                    "entropy": entropy,
                    "ea_entropy": ea_entropy,
                    "ea_logprob": ea_logprob,
                }
            )

        input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
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

        if model.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > max_new_tokens:
            break
        if input_ids.shape[1] > max_length:
            break

    return {
        "tokens_generated": int(new_token),
        "steps": step_logs,
    }


def main():
    parser = argparse.ArgumentParser(description="Trace verifier vs EA logprobs on GSM8K acceptance steps.")
    parser.add_argument(
        "--ea-ckpt",
        default=None,
        help="Path to HASS/EA checkpoint (state_* folder).",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model path or HF id.",
    )
    parser.add_argument(
        "--questions",
        default="data/gsm8k/question.jsonl",
        help="Path to question.jsonl.",
    )
    parser.add_argument("--num-questions", type=int, default=100, help="Number of questions to trace.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max tokens to generate per question.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--out", default="logs/gsm8k_accept_logprobs.jsonl", help="Output JSONL path.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading EA model from {args.ea_ckpt}")
    model = EaModel.from_pretrained(
        base_model_path=args.base_model,
        ea_model_path=args.ea_ckpt,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    if hasattr(model.ea_layer, "gradient_checkpointing"):
        model.ea_layer.gradient_checkpointing = False
    tokenizer = model.get_tokenizer()

    questions = load_questions(Path(args.questions), args.num_questions)
    print(f"Loaded {len(questions)} questions from {args.questions}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as fout:
        for q in tqdm(questions, desc="Tracing"):
            input_ids = prepare_inputs(tokenizer, q, device)
            trace = trace_accept_steps(
                model,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            fout.write(
                json.dumps(
                    {
                        "question_id": q.get("question_id"),
                        "question": q.get("turns", [""])[0] if q.get("turns") else q.get("question", ""),
                        **trace,
                    }
                )
                + "\n"
            )

    print(f"Done. Traces saved to {out_path}")


if __name__ == "__main__":
    main()
