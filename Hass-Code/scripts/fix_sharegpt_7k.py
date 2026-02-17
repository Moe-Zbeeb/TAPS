#!/usr/bin/env python3
"""Fix ShareGPT dataset to ensure strict alternating human/gpt roles."""

import json
from pathlib import Path

# Read from the already-processed file
input_path = Path("data/sharegpt_7k_fixed.json")
output_path = Path("data/sharegpt_7k_fixed2.json")

with input_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

fixed = []
dropped = 0

for ex in data:
    convs = ex.get("conversations", [])
    if not convs:
        dropped += 1
        continue

    # Merge consecutive same-role messages
    merged = []
    for turn in convs:
        role = turn.get("from", "")
        value = turn.get("value", "").strip()

        if not value:
            continue

        if merged and merged[-1]["from"] == role:
            # Merge with previous message
            merged[-1]["value"] += "\n\n" + value
        else:
            merged.append({"from": role, "value": value})

    if not merged:
        dropped += 1
        continue

    # Must start with human
    if merged[0]["from"] != "human":
        merged = merged[1:]

    if not merged:
        dropped += 1
        continue

    # Must have alternating pattern and at least human+gpt
    if len(merged) < 2:
        dropped += 1
        continue

    # Verify strict alternation
    valid = True
    expected = "human"
    for turn in merged:
        if turn["from"] != expected:
            valid = False
            break
        expected = "gpt" if expected == "human" else "human"

    if not valid:
        dropped += 1
        continue

    ex["conversations"] = merged
    fixed.append(ex)

with output_path.open("w", encoding="utf-8") as f:
    json.dump(fixed, f, ensure_ascii=False, indent=2)

print(f"Original examples: {len(data)}")
print(f"Kept examples:     {len(fixed)}")
print(f"Dropped examples:  {dropped}")
print(f"Wrote cleaned dataset to: {output_path}")
