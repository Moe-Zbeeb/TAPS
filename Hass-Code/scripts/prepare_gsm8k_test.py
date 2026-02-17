"""
Download GSM8K test split from HuggingFace and convert to question.jsonl format
"""
import json
from datasets import load_dataset

# Load GSM8K test split
print("Loading GSM8K test split from HuggingFace...")
dataset = load_dataset("openai/gsm8k", "main", split="test")

print(f"Loaded {len(dataset)} test examples")

# Convert to question.jsonl format
output_file = "data/gsm8k/question_test.jsonl"
print(f"Converting to {output_file}...")

with open(output_file, 'w') as f:
    for idx, example in enumerate(dataset):
        question_data = {
            "question_id": idx,
            "category": "gsm8k",
            "turns": [example["question"]]
        }
        f.write(json.dumps(question_data) + '\n')

print(f"Successfully created {output_file} with {len(dataset)} questions")
