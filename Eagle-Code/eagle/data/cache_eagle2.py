import argparse
import json
import os
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_tokens_and_mask(tokenizer, conversations: List[dict], max_len: int):
    messages = []
    tokens = []
    loss_mask = []
    for item in conversations:
        role = item.get("from")
        content = item.get("value") or ""
        if role not in ["human", "gpt"] or not content:
            continue
        messages.append({"role": "user" if role == "human" else "assistant", "content": content})
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        new_tokens = encoded[len(tokens):]
        tokens = encoded
        mask_value = 1 if role == "gpt" else 0
        loss_mask.extend([mask_value] * len(new_tokens))
        if len(tokens) >= max_len:
            tokens = tokens[:max_len]
            loss_mask = loss_mask[:max_len]
            break
    return tokens, loss_mask


def process_file(input_path, outdir, tokenizer, model, max_len, limit):
    os.makedirs(outdir, exist_ok=True)
    count = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if limit and count >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            conversations = row.get("conversations") or []
            if not conversations or conversations[0].get("from") != "human":
                continue
            tokens, loss_mask = build_tokens_and_mask(tokenizer, conversations, max_len)
            if len(tokens) < 2:
                continue
            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
                hidden_state = outputs.hidden_states[-1].cpu()
            loss_mask[-1] = 0
            save_obj = {
                "input_ids": input_ids.cpu().squeeze(0),
                "hidden_state": hidden_state.squeeze(0),
                "loss_mask": loss_mask,
            }
            out_path = os.path.join(outdir, f"sample_{count}.pt")
            torch.save(save_obj, out_path)
            count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="cache_eagle2")
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=70000)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    args = parser.parse_args()

    dtype_map = {"auto": None, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = dtype_map.get(args.torch_dtype, None)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch_dtype, device_map="auto")
    process_file(args.input, args.outdir, tokenizer, model, args.max_len, args.limit)


if __name__ == "__main__":
    main()
