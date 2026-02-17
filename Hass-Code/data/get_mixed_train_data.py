from datasets import load_dataset, concatenate_datasets
import random

SEED = 42
N = 35_000

def _pick_conversation_field(example: dict):
    """
    ShareGPT variants usually store the conversation as a list of dicts.
    We try common field names. If none exist, raise.
    """
    for k in ("conversations", "conversation", "messages", "items", "data"):
        v = example.get(k, None)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return k
    raise KeyError(f"Couldn't find a conversation list field in keys={list(example.keys())[:30]}")

def sharegpt_to_messages(example: dict):
    conv_key = _pick_conversation_field(example)
    conv = example[conv_key]

    out = []
    for m in conv:
        frm = (m.get("from") or m.get("role") or "").lower().strip()
        content = m.get("value") or m.get("content") or ""
        if not isinstance(content, str):
            content = str(content)

        if frm in ("human", "user"):
            out.append({"role": "user", "content": content})
        elif frm in ("gpt", "assistant"):
            out.append({"role": "assistant", "content": content})
        else:
            # unknown role -> skip
            continue

    # Keep only conversations with at least one user+assistant exchange
    # and starting with user (helps training stability)
    if len(out) < 2:
        return {"messages": []}
    if out[0]["role"] != "user":
        # trim until first user
        first_user = next((i for i, x in enumerate(out) if x["role"] == "user"), None)
        out = out[first_user:] if first_user is not None else []

    # Ensure at least one assistant reply exists
    if not any(x["role"] == "assistant" for x in out):
        return {"messages": []}

    return {"messages": out}

def mathinstruct_to_messages(example: dict):
    instr = example.get("instruction", "")
    out = example.get("output", "")
    if not isinstance(instr, str):
        instr = str(instr)
    if not isinstance(out, str):
        out = str(out)

    # You can also add a system prompt here if your trainer expects it.
    return {
        "messages": [
            {"role": "user", "content": instr},
            {"role": "assistant", "content": out},
        ]
    }

# 1) Load
share = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
math  = load_dataset("TIGER-Lab/MathInstruct", split="train")

# 2) Sample 35k from each (deterministic)
share_s = share.shuffle(seed=SEED).select(range(min(N, len(share))))
math_s  = math.shuffle(seed=SEED).select(range(min(N, len(math))))

# 3) Convert to unified chat format
share_s = share_s.map(sharegpt_to_messages, remove_columns=share_s.column_names)
math_s  = math_s.map(mathinstruct_to_messages, remove_columns=math_s.column_names)

# 4) Drop any empties from ShareGPT parsing
share_s = share_s.filter(lambda x: isinstance(x["messages"], list) and len(x["messages"]) >= 2)

# 5) Merge + shuffle
mixed = concatenate_datasets([share_s, math_s]).shuffle(seed=SEED)

print("ShareGPT sampled:", len(share_s))
print("MathInstruct sampled:", len(math_s))
print("Mixed total:", len(mixed))
print("Example:", mixed[0]["messages"][:2])

# 6) Optional: train/eval split (e.g., 98/2)
splits = mixed.train_test_split(test_size=0.02, seed=SEED)
train_ds, eval_ds = splits["train"], splits["test"]

# 7) Optional: save locally (fast reload later)
train_ds.save_to_disk("mixed_train_70k")
eval_ds.save_to_disk("mixed_eval_70k")
