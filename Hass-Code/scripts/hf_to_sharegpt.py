import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from datasets import load_dataset, load_from_disk


def to_sharegpt_single_turn(
    records: List[Dict[str, Any]],
    human_field: str,
    assistant_field: str,
    human_prefix: Optional[str] = None,
    assistant_prefix: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out = []
    for i, ex in enumerate(records):
        if human_field not in ex or assistant_field not in ex:
            # skip rows missing fields
            continue
        human_text = str(ex[human_field]).strip()
        assistant_text = str(ex[assistant_field]).strip()
        if human_prefix:
            human_text = f"{human_prefix}{human_text}".strip()
        if assistant_prefix:
            assistant_text = f"{assistant_prefix}{assistant_text}".strip()
        out.append(
            {
                "id": i,
                "conversations": [
                    {"from": "human", "value": human_text},
                    {"from": "gpt", "value": assistant_text},
                ],
            }
        )
    return out


def to_sharegpt_alpaca(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # tatsu-lab/alpaca style: instruction, input, output
    out = []
    for i, ex in enumerate(records):
        instruction = str(ex.get("instruction", "")).strip()
        input_ = str(ex.get("input", "")).strip()
        output = str(ex.get("output", "")).strip()

        if not instruction and not input_:
            continue

        if input_:
            user = f"{instruction}\n\nInput: {input_}".strip()
        else:
            user = instruction

        out.append(
            {
                "id": i,
                "conversations": [
                    {"from": "human", "value": user},
                    {"from": "gpt", "value": output},
                ],
            }
        )
    return out


def to_sharegpt_dolly(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # databricks/databricks-dolly-15k: instruction, context, response
    out = []
    for i, ex in enumerate(records):
        instruction = str(ex.get("instruction", "")).strip()
        context = str(ex.get("context", "")).strip()
        response = str(ex.get("response", "")).strip()

        if not instruction and not context:
            continue

        if context:
            user = f"{instruction}\n\nContext: {context}".strip()
        else:
            user = instruction

        out.append(
            {
                "id": i,
                "conversations": [
                    {"from": "human", "value": user},
                    {"from": "gpt", "value": response},
                ],
            }
        )
    return out


def to_sharegpt_passthrough(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # If dataset already has ShareGPT-like schema, normalize minimal fields.
    out = []
    for i, ex in enumerate(records):
        conv = ex.get("conversations")
        # Some datasets store chat turns under "messages"
        if conv is None and "messages" in ex:
            conv = ex.get("messages")
        if not isinstance(conv, list):
            continue
        # normalize roles and values
        norm = []
        for msg in conv:
            role = msg.get("from") or msg.get("role")
            content = msg.get("value") or msg.get("content")
            if role in ("human", "user"):
                role = "human"
            elif role in ("gpt", "assistant"):
                role = "gpt"
            else:
                # skip unknown roles
                continue
            if content is None:
                continue
            norm.append({"from": role, "value": str(content)})
        if not norm:
            continue
        out.append({"id": ex.get("id", i), "conversations": norm})
    return out


def to_sharegpt_messages(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Generic OpenAI-style chat format: {"messages": [{"role": "...", "content": "..."}]}
    out = []
    for i, ex in enumerate(records):
        msgs = ex.get("messages")
        if not isinstance(msgs, list):
            continue
        norm = []
        for msg in msgs:
            role = msg.get("role")
            content = msg.get("content") or msg.get("value")
            if role in ("human", "user", "customer"):
                role = "human"
            elif role in ("assistant", "gpt", "model"):
                role = "gpt"
            else:
                continue
            if content is None:
                continue
            norm.append({"from": role, "value": str(content)})
        if not norm:
            continue
        out.append({"id": ex.get("id", i), "conversations": norm})
    return out


def build_sharegpt(
    dataset: str,
    subset: Optional[str],
    split: str,
    limit: Optional[int],
    preset: Optional[str],
    human_field: Optional[str],
    assistant_field: Optional[str],
) -> List[Dict[str, Any]]:
    # Check if dataset is a local JSON file
    dataset_path = Path(dataset)
    if dataset_path.is_file() and dataset_path.suffix.lower() == ".json":
        with dataset_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {dataset}, got {type(data).__name__}")
        if limit is not None:
            data = data[:limit]
    else:
        # Auto-detect subset for known datasets that require it
        if subset is None and dataset == "openai/gsm8k":
            subset = "main"

        if dataset_path.is_dir():
            ds = load_from_disk(str(dataset_path))
            # load_from_disk can return a Dataset or DatasetDict; handle both
            if hasattr(ds, "keys"):
                if split not in ds:
                    split = list(ds.keys())[0]
                records = ds[split]
            else:
                records = ds
        else:
            ds = load_dataset(dataset, subset) if subset else load_dataset(dataset)
            if split not in ds:
                raise ValueError(f"Split '{split}' not found in dataset {dataset}.")
            records = ds[split]

        if limit is not None:
            records = records.select(range(min(limit, len(records))))

        data = list(records)

    if preset:
        p = preset.lower()
        if p == "gsm8k":
            # openai/gsm8k: question, answer
            return to_sharegpt_single_turn(data, "question", "answer")
        elif p == "alpaca":
            return to_sharegpt_alpaca(data)
        elif p in ("dolly", "databricks-dolly-15k", "databricks/databricks-dolly-15k"):
            return to_sharegpt_dolly(data)
        elif p == "passthrough":
            return to_sharegpt_passthrough(data)
        else:
            raise ValueError(
                f"Unknown preset '{preset}'. Use one of: gsm8k, alpaca, dolly, passthrough."
            )

    # Light-weight field inference for common schemas
    if not human_field and not assistant_field and data:
        sample = data[0]
        if "instruction" in sample and "output" in sample:
            human_field, assistant_field = "instruction", "output"

    # Generic mapping path
    if human_field and assistant_field:
        return to_sharegpt_single_turn(data, human_field, assistant_field)

    # Try passthrough if conversations exist
    if data and ("conversations" in data[0] or "messages" in data[0]):
        if "messages" in data[0]:
            return to_sharegpt_messages(data)
        return to_sharegpt_passthrough(data)

    raise ValueError(
        "Cannot infer mapping. Provide --preset or --human-field/--assistant-field."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fetch a HF dataset and convert to ShareGPT format."
    )
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset id (e.g., openai/gsm8k)")
    parser.add_argument("--subset", type=str, default=None, help="Optional dataset subset/config (e.g., main)")
    parser.add_argument("--split", type=str, default="train", help="Split name (train/test/validation)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records")
    parser.add_argument("--preset", type=str, default=None, help="Preset mapping: gsm8k, alpaca, dolly, passthrough")
    parser.add_argument("--human-field", type=str, default=None, help="Generic mapping: human/source field name")
    parser.add_argument("--assistant-field", type=str, default=None, help="Generic mapping: assistant/target field name")
    parser.add_argument("--out", type=str, required=True, help="Output file path (JSON)")

    args = parser.parse_args()

    data = build_sharegpt(
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
        limit=args.limit,
        preset=args.preset,
        human_field=args.human_field,
        assistant_field=args.assistant_field,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Wrote {len(data)} examples to {out_path}")


if __name__ == "__main__":
    main()
