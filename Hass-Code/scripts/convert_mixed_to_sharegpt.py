#!/usr/bin/env python3
"""
Convert HuggingFace dataset with 'messages' format to ShareGPT JSON format
for HASS feature generation.
"""
import json
import argparse
from datasets import load_from_disk
from tqdm import tqdm


def convert_to_sharegpt_format(dataset, output_path):
    """
    Convert messages format to ShareGPT format.

    Input format:
        messages: [{"role": "user/assistant", "content": "..."}]

    Output format:
        conversations: [{"from": "human/gpt", "value": "..."}]
    """
    role_mapping = {
        "user": "human",
        "assistant": "gpt",
        "system": "system"  # Keep system messages as-is
    }

    converted_data = []

    for idx, example in enumerate(tqdm(dataset, desc="Converting")):
        conversations = []

        for message in example["messages"]:
            role = message["role"]
            content = message["content"]

            # Map roles
            from_role = role_mapping.get(role, role)

            conversations.append({
                "from": from_role,
                "value": content
            })

        converted_data.append({
            "id": idx,
            "conversations": conversations
        })

    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    print(f"\nConversion complete!")
    print(f"Total samples: {len(converted_data)}")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert HF dataset to ShareGPT JSON')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to HuggingFace dataset directory')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to output JSON file')

    args = parser.parse_args()

    print(f"Loading dataset from: {args.input_dir}")
    dataset = load_from_disk(args.input_dir)

    print(f"Dataset size: {len(dataset)}")
    print(f"Sample format: {dataset[0].keys()}")

    convert_to_sharegpt_format(dataset, args.output_file)


if __name__ == "__main__":
    main()
