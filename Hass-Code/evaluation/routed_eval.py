"""
Routed Multi-Head EA Evaluation

This script loads multiple EA heads and routes between them based on
a naive keyword-based router. It evaluates on all benchmarks and outputs
statistics on acceptance rates and head selection.

Usage:
python evaluation/routed_eval.py \
    --base-model-path /path/to/llama3 \
    --head-general /path/to/original_checkpoint \
    --head-math /path/to/finetuned_checkpoint \
    --benchmarks mt_bench gsm8k math_500 svamp
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from accelerate.utils import set_seed
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

# Add parent directory to path
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
import sys
sys.path.insert(0, parent_dir)

from model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from model.cnets import Model
from model.configs import EConfig
from model.kv_cache import initialize_past_key_values
from model.utils import *


# ============================================================================
# Naive Keyword Router
# ============================================================================

MATH_KEYWORDS = [
    # Direct math terms
    "calculate", "compute", "solve", "equation", "formula", "algebra",
    "arithmetic", "mathematical", "mathematics", "math",
    # Operations
    "add", "subtract", "multiply", "divide", "sum", "difference", "product",
    "quotient", "remainder", "percent", "percentage", "ratio", "proportion",
    # Numbers and units
    "how many", "how much", "total", "average", "mean", "median",
    "dollars", "cents", "price", "cost", "profit", "loss", "discount",
    "miles", "kilometers", "meters", "feet", "inches", "hours", "minutes",
    # Math symbols (as words or patterns)
    "x =", "= x", "+ x", "- x", "×", "÷", "²", "³",
    # Problem types
    "word problem", "find the value", "what is the", "express",
    # GSM8K / Math specific patterns
    "per day", "per hour", "per week", "per month", "per year",
    "each day", "each hour", "each week", "every day", "every week",
    "times as", "twice as", "half of", "third of", "quarter of",
]

MATH_PATTERNS = [
    r'\d+\s*[\+\-\*\/\%]\s*\d+',  # arithmetic expressions like "5 + 3"
    r'\$\d+',                      # dollar amounts
    r'\d+%',                       # percentages
    r'\d+\s*(dollars|cents|miles|km|meters|hours|minutes|seconds)',  # units
    r'how (many|much)',            # quantity questions
]


def is_math_prompt(prompt: str) -> Tuple[bool, List[str]]:
    """
    Determine if a prompt is math-related using keyword and pattern matching.

    Returns:
        Tuple of (is_math, matched_keywords)
    """
    prompt_lower = prompt.lower()
    matched = []

    # Check keywords
    for keyword in MATH_KEYWORDS:
        if keyword.lower() in prompt_lower:
            matched.append(keyword)

    # Check patterns
    for pattern in MATH_PATTERNS:
        if re.search(pattern, prompt_lower):
            matched.append(f"pattern:{pattern[:20]}")

    return len(matched) > 0, matched


# ============================================================================
# Multi-Head EA Model
# ============================================================================

@dataclass
class HeadStats:
    """Statistics for a single head"""
    name: str
    triggers: int = 0
    total_accept_length: float = 0.0
    total_tokens: int = 0
    accept_lengths: List[float] = field(default_factory=list)

    @property
    def mean_accept_length(self) -> float:
        if not self.accept_lengths:
            return 0.0
        return sum(self.accept_lengths) / len(self.accept_lengths)


class MultiHeadEaModel(nn.Module):
    """
    EA Model that supports multiple heads with routing.
    Loads the base model once and swaps EA heads as needed.
    """

    def __init__(
        self,
        base_model,
        base_model_path: str,
        heads: Dict[str, nn.Module],
        head_configs: Dict[str, str],
        total_token: int = 60,
        depth: int = 5,
        top_k: int = 10,
        threshold: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_path = base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

        # Store multiple heads
        self.heads = nn.ModuleDict(heads)
        self.head_configs = head_configs
        self.current_head_name = None
        self.ea_layer = None  # Will be set when switching heads

        # Model parameters
        self.total_token = total_token
        self.depth = depth
        self.top_k = top_k
        self.threshold = threshold

        # KV cache (shared across heads)
        self.past_key_values = None
        self.past_key_values_data = None
        self.current_length_data = None

    def switch_head(self, head_name: str):
        """Switch to a different EA head"""
        if head_name not in self.heads:
            raise ValueError(f"Unknown head: {head_name}. Available: {list(self.heads.keys())}")

        if self.current_head_name == head_name:
            return  # Already using this head

        self.ea_layer = self.heads[head_name]
        self.current_head_name = head_name
        self.ea_layer.reset_kv()

    def get_tokenizer(self):
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        base_model_path: str,
        head_paths: Dict[str, str],  # {"general": path, "math": path}
        total_token: int = 60,
        depth: int = 5,
        top_k: int = 10,
        threshold: float = 1.0,
        **kwargs,
    ):
        """Load base model and multiple EA heads"""

        print(f"Loading base model from {base_model_path}...")
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(base_model_path, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {Type}")

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device

        # Load each head
        heads = {}
        head_configs = {}

        for head_name, head_path in head_paths.items():
            print(f"Loading EA head '{head_name}' from {head_path}...")

            config_path = os.path.join(head_path, "config.json")
            config = EConfig.from_pretrained(config_path)
            head_configs[head_name] = config_path

            # Check for bias setting
            with open(config_path, "r") as f:
                con = json.loads(f.read())
            bias = con.get("bias", True)

            # Create head model
            head = Model(
                config,
                bias=bias,
                total_tokens=total_token,
                depth=depth,
                top_k=top_k,
                threshold=threshold
            )

            # Load weights
            load_path = os.path.join(head_path, "pytorch_model.bin")
            if os.path.exists(load_path):
                state_dict = torch.load(load_path, map_location=device)
            else:
                from safetensors.torch import load_file
                load_path = os.path.join(head_path, "model.safetensors")
                state_dict = load_file(load_path)

            head.load_state_dict(state_dict, strict=True)

            # Setup device handling
            if device != base_model.lm_head.weight.device:
                head.diff_device = True
                head.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                head.diff_device = False

            head.to(base_model.dtype).to(device)
            head.init_tree()

            heads[head_name] = head

        model = cls(
            base_model,
            base_model_path,
            heads,
            head_configs,
            total_token,
            depth,
            top_k,
            threshold,
        )

        # Set first head as default
        first_head = list(heads.keys())[0]
        model.switch_head(first_head)

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
    ):
        with torch.inference_mode():
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def eagenerate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
        is_llama3=False,
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        max_length = max_length - self.ea_layer.total_tokens - 10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize KV cache
        if hasattr(self, "past_key_values") and self.past_key_values is not None:
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)

        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )

        new_token = 0
        accept_length_list = []

        for idx in range(max_length):
            self.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)

            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )

            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            try:
                accept_length_list.append(accept_length.item())
            except:
                accept_length_list.append(accept_length)

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
                self,
                hidden_state_new,
                sample_p
            )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx, accept_length_list, {}


# ============================================================================
# Evaluation Functions
# ============================================================================

def run_routed_eval(
    model: MultiHeadEaModel,
    questions: List[dict],
    benchmark_name: str,
    temperature: float = 0.0,
    max_new_token: int = 1024,
) -> Tuple[Dict[str, HeadStats], List[dict]]:
    """
    Run evaluation with routing between heads.

    Returns:
        Tuple of (head_stats, detailed_results)
    """
    tokenizer = model.get_tokenizer()

    # Initialize stats for each head
    head_stats = {name: HeadStats(name=name) for name in model.heads.keys()}
    detailed_results = []

    # System message
    system_message = {
        "role": "system",
        "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    }

    for question in tqdm(questions, desc=f"Evaluating {benchmark_name}"):
        question_id = question["question_id"]
        turns = question["turns"]

        # Combine all turns for routing decision
        full_prompt = " ".join(turns)

        # Route to appropriate head
        is_math, matched_keywords = is_math_prompt(full_prompt)
        selected_head = "math" if is_math else "general"

        # Switch head
        model.switch_head(selected_head)

        # Run generation
        messages = [system_message]
        all_accept_lengths = []
        total_new_tokens = 0

        for turn_idx, turn_content in enumerate(turns):
            messages.append({"role": "user", "content": turn_content})

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer([prompt], add_special_tokens=False).input_ids

            torch.cuda.synchronize()
            start_time = time.time()

            output_ids, new_token, idx, accept_length_list, _ = model.eagenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=temperature,
                log=True,
                is_llama3=True,
                max_new_tokens=max_new_token,
            )

            torch.cuda.synchronize()
            total_time = time.time() - start_time

            # Process output
            output_ids = output_ids[0][len(input_ids[0]):]
            stop_token_ids = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            if stop_token_ids:
                stop_token_ids_index = [
                    i for i, id in enumerate(output_ids)
                    if id in stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[:stop_token_ids_index[0]]

            output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for tok in special_token:
                        output = output.replace(tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            messages.append({"role": "assistant", "content": output})
            all_accept_lengths.extend(accept_length_list)
            total_new_tokens += new_token

        # Update stats
        stats = head_stats[selected_head]
        stats.triggers += 1
        if all_accept_lengths:
            mean_accept = sum(all_accept_lengths) / len(all_accept_lengths)
            stats.accept_lengths.append(mean_accept)
            stats.total_accept_length += sum(all_accept_lengths)
            stats.total_tokens += len(all_accept_lengths)

        # Store detailed result
        detailed_results.append({
            "question_id": question_id,
            "benchmark": benchmark_name,
            "selected_head": selected_head,
            "is_math": is_math,
            "matched_keywords": matched_keywords[:5],  # Top 5 matches
            "accept_lengths": all_accept_lengths,
            "mean_accept_length": mean_accept if all_accept_lengths else 0,
            "total_new_tokens": total_new_tokens,
        })

    return head_stats, detailed_results


def print_results_table(all_results: Dict[str, Tuple[Dict[str, HeadStats], List[dict]]]):
    """Print formatted results tables"""

    print("\n" + "=" * 80)
    print("ROUTED EVALUATION RESULTS")
    print("=" * 80)

    # Table 1: Overall Acceptance Rate per Benchmark
    print("\n### Overall Acceptance Rate (Mean Accept Length) ###\n")
    print(f"{'Benchmark':<15} {'Routed Avg':<15} {'General Head':<15} {'Math Head':<15}")
    print("-" * 60)

    summary_data = []
    for bench_name, (head_stats, detailed) in all_results.items():
        # Calculate overall routed average
        all_accepts = [r["mean_accept_length"] for r in detailed if r["mean_accept_length"] > 0]
        routed_avg = sum(all_accepts) / len(all_accepts) if all_accepts else 0

        general_avg = head_stats["general"].mean_accept_length
        math_avg = head_stats["math"].mean_accept_length

        print(f"{bench_name:<15} {routed_avg:<15.4f} {general_avg:<15.4f} {math_avg:<15.4f}")
        summary_data.append({
            "benchmark": bench_name,
            "routed_avg": routed_avg,
            "general_avg": general_avg,
            "math_avg": math_avg,
        })

    # Table 2: Head Trigger Counts per Benchmark
    print("\n### Head Selection Distribution ###\n")
    print(f"{'Benchmark':<15} {'Total':<10} {'General':<12} {'Math':<12} {'General %':<12} {'Math %':<12}")
    print("-" * 75)

    for bench_name, (head_stats, detailed) in all_results.items():
        total = len(detailed)
        general_count = head_stats["general"].triggers
        math_count = head_stats["math"].triggers
        general_pct = (general_count / total * 100) if total > 0 else 0
        math_pct = (math_count / total * 100) if total > 0 else 0

        print(f"{bench_name:<15} {total:<10} {general_count:<12} {math_count:<12} {general_pct:<12.1f} {math_pct:<12.1f}")

    # Table 3: Per-Head Performance on Each Benchmark (when triggered)
    print("\n### Per-Head Performance (When Selected) ###\n")
    print(f"{'Benchmark':<15} {'Head':<12} {'Triggers':<10} {'Mean Accept':<15}")
    print("-" * 55)

    for bench_name, (head_stats, detailed) in all_results.items():
        for head_name, stats in head_stats.items():
            if stats.triggers > 0:
                print(f"{bench_name:<15} {head_name:<12} {stats.triggers:<10} {stats.mean_accept_length:<15.4f}")
        print("-" * 55)

    return summary_data


def tensor_to_python(obj):
    """Recursively convert tensors to Python types for JSON serialization"""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_python(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def save_results(
    all_results: Dict[str, Tuple[Dict[str, HeadStats], List[dict]]],
    output_dir: str
):
    """Save all results to files"""
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed per-question results
    all_detailed = []
    for bench_name, (_, detailed) in all_results.items():
        all_detailed.extend(detailed)

    with open(os.path.join(output_dir, "detailed_results.jsonl"), "w") as f:
        for item in all_detailed:
            # Convert any tensors to Python types
            item_clean = tensor_to_python(item)
            f.write(json.dumps(item_clean) + "\n")

    # Save summary
    summary = {}
    for bench_name, (head_stats, detailed) in all_results.items():
        all_accepts = [r["mean_accept_length"] for r in detailed if r["mean_accept_length"] > 0]
        summary[bench_name] = {
            "routed_mean_accept": sum(all_accepts) / len(all_accepts) if all_accepts else 0,
            "total_questions": len(detailed),
            "heads": {
                name: {
                    "triggers": stats.triggers,
                    "mean_accept_length": stats.mean_accept_length,
                    "trigger_percentage": stats.triggers / len(detailed) * 100 if detailed else 0,
                }
                for name, stats in head_stats.items()
            }
        }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Routed Multi-Head EA Evaluation")

    # Model paths
    parser.add_argument("--base-model-path", type=str, required=True,
                        help="Path to base LLaMA model")
    parser.add_argument("--head-general", type=str, required=True,
                        help="Path to general/ShareGPT EA head checkpoint")
    parser.add_argument("--head-math", type=str, required=True,
                        help="Path to math/MathInstruct EA head checkpoint")

    # Benchmarks
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["mt_bench", "gsm8k", "math_500", "svamp"],
                        help="Benchmarks to evaluate on")

    # Generation params
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)

    # Output
    parser.add_argument("--output-dir", type=str, default="routed_eval_results",
                        help="Directory to save results")

    # Debug
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Max questions per benchmark (for debugging)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    # Load model with multiple heads
    print("Loading multi-head model...")
    model = MultiHeadEaModel.from_pretrained(
        base_model_path=args.base_model_path,
        head_paths={
            "general": args.head_general,
            "math": args.head_math,
        },
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    print("Model loaded successfully!")

    # Warmup
    print("Warming up...")
    tokenizer = model.get_tokenizer()
    warmup_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello, how are you?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    warmup_ids = tokenizer([warmup_prompt], add_special_tokens=False).input_ids
    for head_name in model.heads.keys():
        model.switch_head(head_name)
        for _ in range(2):
            _ = model.eagenerate(
                torch.as_tensor(warmup_ids).cuda(),
                temperature=0.0,
                log=True,
                is_llama3=True,
                max_new_tokens=50,
            )
    print("Warmup complete!")

    # Run evaluation on each benchmark
    all_results = {}

    for bench_name in args.benchmarks:
        print(f"\n{'='*60}")
        print(f"Evaluating on {bench_name}")
        print(f"{'='*60}")

        question_file = os.path.join(parent_dir, "data", bench_name, "question.jsonl")
        if not os.path.exists(question_file):
            print(f"Warning: Question file not found: {question_file}")
            continue

        questions = load_questions(question_file, None, None)

        if args.max_questions:
            questions = questions[:args.max_questions]

        print(f"Loaded {len(questions)} questions")

        head_stats, detailed = run_routed_eval(
            model=model,
            questions=questions,
            benchmark_name=bench_name,
            temperature=args.temperature,
            max_new_token=args.max_new_token,
        )

        all_results[bench_name] = (head_stats, detailed)

    # Print and save results
    print_results_table(all_results)
    save_results(all_results, args.output_dir)


if __name__ == "__main__":
    main()
