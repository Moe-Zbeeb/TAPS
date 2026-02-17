import argparse
import json
import os
from datasets import load_dataset


def normalize_message(message):
    for key in ["value", "text", "markdown"]:
        content = message.get(key)
        if content:
            return content
    return ""


def prepare_sharegpt(split, output_path):
    ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split=split, trust_remote_code=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            convs = []
            for item in row.get("conversations", []):
                role = item.get("from")
                content = normalize_message(item)
                if not role or not content:
                    continue
                convs.append({"from": role, "value": content})
            if not convs or convs[0].get("from") != "human":
                continue
            rec = {"id": row.get("id", ""), "conversations": convs}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def prepare_mathinstruct(split, output_path):
    ds = load_dataset("TIGER-Lab/MathInstruct", split=split)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            instruction = row.get("instruction") or ""
            output = row.get("output") or ""
            if not instruction or not output:
                continue
            convs = [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output},
            ]
            rec = {"id": row.get("id") or f"mathinstruct_{idx}", "conversations": convs}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="prepared_data")
    parser.add_argument("--sharegpt", action="store_true")
    parser.add_argument("--mathinstruct", action="store_true")
    args = parser.parse_args()

    if not args.sharegpt and not args.mathinstruct:
        args.sharegpt = True
        args.mathinstruct = True

    os.makedirs(args.outdir, exist_ok=True)

    if args.sharegpt:
        sharegpt_path = os.path.join(args.outdir, "sharegpt_vicuna_unfiltered_train.jsonl")
        prepare_sharegpt("train", sharegpt_path)

    if args.mathinstruct:
        math_path = os.path.join(args.outdir, "mathinstruct_train.jsonl")
        prepare_mathinstruct("train", math_path)


if __name__ == "__main__":
    main()
