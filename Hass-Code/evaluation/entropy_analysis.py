#!/usr/bin/env python
"""
Multi-Head Entropy Analysis

Traces entropy at accept/reject decisions for multiple EA heads
across multiple benchmarks. Enables comparison of head behavior.

Key difference from existing scripts:
- Runs ALL heads on EVERY question (not routing) to compare head behavior
- Tracks entropy at decision points for BOTH heads on same input
- Enables side-by-side comparison of entropy patterns

Usage:
    python evaluation/entropy_analysis.py \
        --base-model-path /path/to/llama3 \
        --head mathinstruct /path/to/mathinstruct_ckpt \
        --head sharegpt /path/to/sharegpt_ckpt \
        --head mixed /path/to/mixed_ckpt \
        --head averaged /path/to/averaged_ckpt \
        --benchmarks mt_bench gsm8k math_500 svamp \
        --output-dir entropy_analysis_results
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import set_seed
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

# Add parent directory to path
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from model.cnets import Model
from model.configs import EConfig
from model.kv_cache import initialize_past_key_values
from model.utils import (
    prepare_logits_processor,
    initialize_tree,
    tree_decoding,
    evaluate_posterior,
    update_inference_inputs,
    reset_tree_mode,
)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class EntropyRecord:
    """Record of entropy at a single decision point"""
    position: int                    # Position within generation
    step: int                        # Decoding step
    accepted: bool                   # Whether token was accepted
    verifier_entropy: float          # Entropy of base model distribution
    ea_entropy: Optional[float]      # Entropy of EA head distribution
    ea_token: int                    # Token proposed by EA
    verifier_token: int              # Token preferred by verifier
    logprob_diff: float              # EA logprob - verifier top logprob (under verifier)
    logprob_ea_token: float          # Verifier's logprob of EA token
    logprob_verifier_token: float    # Verifier's logprob of its top token
    ea_logprob: Optional[float]      # EA's logprob of its own token
    accept_length: int               # Total accept length for this step


@dataclass
class QuestionTrace:
    """Trace for one question with one head"""
    question_id: Any
    benchmark: str
    head_name: str
    tokens_generated: int
    records: List[EntropyRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "benchmark": self.benchmark,
            "head_name": self.head_name,
            "tokens_generated": self.tokens_generated,
            "records": [asdict(r) for r in self.records],
        }


@dataclass
class EntropyStats:
    """Aggregated entropy statistics"""
    head_name: str
    benchmark: str
    total_positions: int = 0
    total_accepted: int = 0
    total_rejected: int = 0

    # Verifier entropy stats
    verifier_entropy_accept: List[float] = field(default_factory=list)
    verifier_entropy_reject: List[float] = field(default_factory=list)

    # EA entropy stats
    ea_entropy_accept: List[float] = field(default_factory=list)
    ea_entropy_reject: List[float] = field(default_factory=list)

    # Logprob stats
    logprob_diff_accept: List[float] = field(default_factory=list)
    logprob_diff_reject: List[float] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        if self.total_positions == 0:
            return 0.0
        return self.total_accepted / self.total_positions

    def compute_summary(self) -> dict:
        """Compute summary statistics"""
        def safe_stats(data: List[float]) -> dict:
            if not data:
                return {"mean": 0.0, "std": 0.0, "median": 0.0, "count": 0}
            arr = np.array(data)
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "count": len(data),
            }

        return {
            "head_name": self.head_name,
            "benchmark": self.benchmark,
            "total_positions": self.total_positions,
            "acceptance_rate": self.acceptance_rate,
            "verifier_entropy": {
                "accept": safe_stats(self.verifier_entropy_accept),
                "reject": safe_stats(self.verifier_entropy_reject),
                "delta": (
                    np.mean(self.verifier_entropy_reject) - np.mean(self.verifier_entropy_accept)
                    if self.verifier_entropy_accept and self.verifier_entropy_reject else 0.0
                ),
            },
            "ea_entropy": {
                "accept": safe_stats(self.ea_entropy_accept),
                "reject": safe_stats(self.ea_entropy_reject),
                "delta": (
                    np.mean(self.ea_entropy_reject) - np.mean(self.ea_entropy_accept)
                    if self.ea_entropy_accept and self.ea_entropy_reject else 0.0
                ),
            },
            "logprob_diff": {
                "accept": safe_stats(self.logprob_diff_accept),
                "reject": safe_stats(self.logprob_diff_reject),
            },
        }


# ============================================================================
# Multi-Head EA Model (adapted from routed_eval.py)
# ============================================================================

class MultiHeadEaModel(nn.Module):
    """
    EA Model that supports multiple heads for entropy analysis.
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
        self.ea_layer = None

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
            return

        self.ea_layer = self.heads[head_name]
        self.current_head_name = head_name
        self.ea_layer.reset_kv()

    def get_tokenizer(self):
        return self.tokenizer

    @torch.inference_mode()
    def reset_kv_cache(self):
        """Reset the KV cache for new generation"""
        if hasattr(self, "current_length_data") and self.current_length_data is not None:
            self.current_length_data.zero_()
        if hasattr(self, "ea_layer") and self.ea_layer is not None:
            self.ea_layer.reset_kv()

    @classmethod
    def from_pretrained(
        cls,
        base_model_path: str,
        head_paths: Dict[str, str],
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

        heads = {}
        head_configs = {}

        for head_name, head_path in head_paths.items():
            print(f"Loading EA head '{head_name}' from {head_path}...")

            config_path = os.path.join(head_path, "config.json")
            config = EConfig.from_pretrained(config_path)
            head_configs[head_name] = config_path

            with open(config_path, "r") as f:
                con = json.loads(f.read())
            bias = con.get("bias", True)

            head = Model(
                config,
                bias=bias,
                total_tokens=total_token,
                depth=depth,
                top_k=top_k,
                threshold=threshold
            )

            load_path = os.path.join(head_path, "pytorch_model.bin")
            if os.path.exists(load_path):
                state_dict = torch.load(load_path, map_location=device)
            else:
                from safetensors.torch import load_file
                load_path = os.path.join(head_path, "model.safetensors")
                state_dict = load_file(load_path)

            head.load_state_dict(state_dict, strict=True)

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


# ============================================================================
# Entropy Tracing Functions
# ============================================================================

def compute_entropy(log_probs: torch.Tensor) -> float:
    """Compute Shannon entropy from log probabilities"""
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum()
    return float(entropy)


@torch.inference_mode()
def trace_single_generation(
    model: MultiHeadEaModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> Tuple[List[EntropyRecord], int]:
    """
    Trace entropy at each decision point for the current head.

    Returns:
        Tuple of (list of EntropyRecord, tokens_generated)
    """
    device = input_ids.device
    padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(device)
    logits_processor = prepare_logits_processor(temperature=temperature) if temperature > 1e-5 else None

    # Reset/initialize caches
    if hasattr(model, "past_key_values") and model.past_key_values is not None:
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

    # Initialize tree with logprobs for EA entropy tracking
    init = initialize_tree(input_ids, model, past_key_values, logits_processor, return_logprobs=True)
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token, draft_logprobs = init

    new_token = 0
    records: List[EntropyRecord] = []
    position_counter = 0

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
        positions = min(log_probs.shape[0], cand_seq.shape[0] - 1)

        for pos in range(positions):
            ea_tok = int(cand_seq[pos + 1])
            top_tok = int(torch.argmax(log_probs[pos]))
            accepted = pos < accept_len_int

            # Verifier entropy
            prob_pos = torch.exp(log_probs[pos])
            verifier_entropy = float(-(prob_pos * log_probs[pos]).sum())

            # EA entropy (if available)
            ea_entropy = None
            ea_logprob = None
            if draft_logprobs is not None and draft_logprobs.shape[0] > best_candidate_idx:
                ea_pos_lp = draft_logprobs[best_candidate_idx]
                if pos < ea_pos_lp.shape[0]:
                    ea_log_probs_pos = ea_pos_lp[pos]
                    ea_entropy = float(-(torch.exp(ea_log_probs_pos) * ea_log_probs_pos).sum())
                    ea_logprob = float(ea_log_probs_pos[ea_tok])

            record = EntropyRecord(
                position=position_counter,
                step=step,
                accepted=accepted,
                verifier_entropy=verifier_entropy,
                ea_entropy=ea_entropy,
                ea_token=ea_tok,
                verifier_token=top_tok,
                logprob_diff=float(log_probs[pos, ea_tok] - log_probs[pos, top_tok]),
                logprob_ea_token=float(log_probs[pos, ea_tok]),
                logprob_verifier_token=float(log_probs[pos, top_tok]),
                ea_logprob=ea_logprob,
                accept_length=accept_len_int,
            )
            records.append(record)
            position_counter += 1

        # Update for next iteration
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

        # Check termination conditions
        if model.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        # Check for llama3 stop token
        try:
            stop_token_id = model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if stop_token_id in input_ids[0, input_len:].tolist():
                break
        except:
            pass
        if new_token > max_new_tokens:
            break
        if input_ids.shape[1] > max_length:
            break

    return records, int(new_token)


def trace_all_heads(
    model: MultiHeadEaModel,
    input_ids: torch.Tensor,
    heads_to_trace: List[str],
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> Dict[str, Tuple[List[EntropyRecord], int]]:
    """
    Run all heads on the same input and return entropy traces per head.

    Returns:
        Dict mapping head_name -> (list of EntropyRecord, tokens_generated)
    """
    results = {}

    for head_name in heads_to_trace:
        # Switch to this head
        model.switch_head(head_name)

        # Reset KV cache between heads
        model.reset_kv_cache()

        # Run tracing
        records, tokens_gen = trace_single_generation(
            model,
            input_ids.clone(),  # Clone to ensure fresh input
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        results[head_name] = (records, tokens_gen)

    return results


# ============================================================================
# Benchmark Loading
# ============================================================================

SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
    "while being safe. Your answers should not include any harmful, unethical, racist, sexist, "
    "toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased "
    "and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of "
    "answering something not correct. If you don't know the answer to a question, please don't share "
    "false information."
)


def load_questions(benchmark_name: str, data_dir: str, max_questions: Optional[int] = None) -> List[dict]:
    """Load questions from a benchmark"""
    question_file = os.path.join(data_dir, benchmark_name, "question.jsonl")

    if not os.path.exists(question_file):
        raise FileNotFoundError(f"Question file not found: {question_file}")

    questions = []
    with open(question_file, "r") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if max_questions is not None:
        questions = questions[:max_questions]

    return questions


def prepare_inputs(tokenizer, question: dict, device: torch.device) -> torch.Tensor:
    """Prepare input_ids from a question"""
    # Handle different question formats
    if "turns" in question and isinstance(question["turns"], list):
        prompt = question["turns"][0]
    elif "question" in question:
        prompt = question["question"]
    else:
        prompt = str(question)

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


# ============================================================================
# Statistics and Analysis
# ============================================================================

def aggregate_stats(
    traces: List[QuestionTrace],
    head_name: str,
    benchmark: str,
) -> EntropyStats:
    """Aggregate records into statistics"""
    stats = EntropyStats(head_name=head_name, benchmark=benchmark)

    for trace in traces:
        if trace.head_name != head_name or trace.benchmark != benchmark:
            continue

        for record in trace.records:
            stats.total_positions += 1

            if record.accepted:
                stats.total_accepted += 1
                stats.verifier_entropy_accept.append(record.verifier_entropy)
                if record.ea_entropy is not None:
                    stats.ea_entropy_accept.append(record.ea_entropy)
                stats.logprob_diff_accept.append(record.logprob_diff)
            else:
                stats.total_rejected += 1
                stats.verifier_entropy_reject.append(record.verifier_entropy)
                if record.ea_entropy is not None:
                    stats.ea_entropy_reject.append(record.ea_entropy)
                stats.logprob_diff_reject.append(record.logprob_diff)

    return stats


# ============================================================================
# Visualization
# ============================================================================

def generate_visualizations(
    all_stats: Dict[Tuple[str, str], EntropyStats],
    output_dir: str,
):
    """Generate visualization plots"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("Warning: matplotlib not available, skipping visualizations")
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    def ecdf(data):
        """Compute empirical CDF"""
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    # Gather unique heads and benchmarks
    heads = sorted(set(k[0] for k in all_stats.keys()))
    benchmarks = sorted(set(k[1] for k in all_stats.keys()))

    # 1. ECDF plots per head (verifier entropy, accept vs reject)
    for head in heads:
        plt.figure(figsize=(8, 5))
        colors = {"accept": "#1b9e77", "reject": "#d95f02"}

        all_accept = []
        all_reject = []

        for bench in benchmarks:
            key = (head, bench)
            if key in all_stats:
                stats = all_stats[key]
                all_accept.extend(stats.verifier_entropy_accept)
                all_reject.extend(stats.verifier_entropy_reject)

        if all_accept:
            x_a, y_a = ecdf(all_accept)
            plt.step(x_a, y_a, where="post", color=colors["accept"],
                     linewidth=2, label=f"Accept (n={len(all_accept)})")
            med_accept = np.median(all_accept)
            plt.axvline(med_accept, color=colors["accept"], linestyle="--", alpha=0.7)

        if all_reject:
            x_r, y_r = ecdf(all_reject)
            plt.step(x_r, y_r, where="post", color=colors["reject"],
                     linewidth=2, label=f"Reject (n={len(all_reject)})")
            med_reject = np.median(all_reject)
            plt.axvline(med_reject, color=colors["reject"], linestyle="--", alpha=0.7)

        plt.xlabel("Verifier Entropy (nats)")
        plt.ylabel("ECDF")
        plt.title(f"Verifier Entropy Distribution - {head} head")
        plt.legend(loc="lower right")
        plt.xlim(0, max(3.0, plt.xlim()[1]))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"entropy_ecdf_{head}.png"), dpi=150)
        plt.close()

    # 2. Heatmap: mean entropy delta (reject - accept) by head x benchmark
    if len(heads) > 1 and len(benchmarks) > 1:
        delta_matrix = np.zeros((len(heads), len(benchmarks)))
        for i, head in enumerate(heads):
            for j, bench in enumerate(benchmarks):
                key = (head, bench)
                if key in all_stats:
                    summary = all_stats[key].compute_summary()
                    delta_matrix[i, j] = summary["verifier_entropy"]["delta"]

        plt.figure(figsize=(10, 6))
        im = plt.imshow(delta_matrix, cmap="RdYlGn_r", aspect="auto")
        plt.colorbar(im, label="Entropy Delta (reject - accept)")
        plt.xticks(range(len(benchmarks)), benchmarks, rotation=45, ha="right")
        plt.yticks(range(len(heads)), heads)
        plt.xlabel("Benchmark")
        plt.ylabel("Head")
        plt.title("Verifier Entropy Delta: Reject - Accept")

        # Add value annotations
        for i in range(len(heads)):
            for j in range(len(benchmarks)):
                plt.text(j, i, f"{delta_matrix[i, j]:.3f}",
                         ha="center", va="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "head_comparison_heatmap.png"), dpi=150)
        plt.close()

    # 3. Acceptance rate comparison bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(benchmarks))
    width = 0.8 / len(heads)
    colors_heads = plt.cm.Set2(np.linspace(0, 1, len(heads)))

    for i, head in enumerate(heads):
        rates = []
        for bench in benchmarks:
            key = (head, bench)
            if key in all_stats:
                rates.append(all_stats[key].acceptance_rate * 100)
            else:
                rates.append(0)
        ax.bar(x + i * width - (len(heads) - 1) * width / 2, rates,
               width, label=head, color=colors_heads[i])

    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Acceptance Rate (%)")
    ax.set_title("Token Acceptance Rate by Head and Benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "acceptance_rate_comparison.png"), dpi=150)
    plt.close()

    # 4. EA vs Verifier entropy scatter (if EA entropy available)
    for head in heads:
        ea_ent_accept = []
        ver_ent_accept = []
        ea_ent_reject = []
        ver_ent_reject = []

        for bench in benchmarks:
            key = (head, bench)
            if key in all_stats:
                stats = all_stats[key]
                ea_ent_accept.extend(stats.ea_entropy_accept)
                ver_ent_accept.extend(stats.verifier_entropy_accept[:len(stats.ea_entropy_accept)])
                ea_ent_reject.extend(stats.ea_entropy_reject)
                ver_ent_reject.extend(stats.verifier_entropy_reject[:len(stats.ea_entropy_reject)])

        if ea_ent_accept or ea_ent_reject:
            plt.figure(figsize=(8, 6))

            if ea_ent_accept:
                plt.scatter(ver_ent_accept, ea_ent_accept, alpha=0.3, c="#1b9e77",
                           label=f"Accept (n={len(ea_ent_accept)})", s=10)
            if ea_ent_reject:
                plt.scatter(ver_ent_reject, ea_ent_reject, alpha=0.3, c="#d95f02",
                           label=f"Reject (n={len(ea_ent_reject)})", s=10)

            max_val = max(
                max(ver_ent_accept + ver_ent_reject) if ver_ent_accept or ver_ent_reject else 3,
                max(ea_ent_accept + ea_ent_reject) if ea_ent_accept or ea_ent_reject else 3
            )
            plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label="y=x")

            plt.xlabel("Verifier Entropy (nats)")
            plt.ylabel("EA Entropy (nats)")
            plt.title(f"EA vs Verifier Entropy - {head} head")
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"ea_vs_verifier_scatter_{head}.png"), dpi=150)
            plt.close()

    # 5. Per-benchmark ECDF comparison between heads
    for bench in benchmarks:
        plt.figure(figsize=(8, 5))
        colors_iter = iter(plt.cm.tab10(np.linspace(0, 1, len(heads))))

        for head in heads:
            key = (head, bench)
            if key not in all_stats:
                continue

            stats = all_stats[key]
            all_entropy = stats.verifier_entropy_accept + stats.verifier_entropy_reject

            if all_entropy:
                color = next(colors_iter)
                x, y = ecdf(all_entropy)
                plt.step(x, y, where="post", color=color, linewidth=2,
                         label=f"{head} (accept={stats.acceptance_rate:.1%})")

        plt.xlabel("Verifier Entropy (nats)")
        plt.ylabel("ECDF")
        plt.title(f"Entropy Distribution Comparison - {bench}")
        plt.legend(loc="lower right")
        plt.xlim(0, max(3.0, plt.xlim()[1]))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"entropy_ecdf_benchmark_{bench}.png"), dpi=150)
        plt.close()

    print(f"Visualizations saved to {plots_dir}/")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Head Entropy Analysis for EA Accept/Reject Decisions"
    )

    # Model paths
    parser.add_argument("--base-model-path", type=str, required=True,
                        help="Path to base LLaMA model")
    parser.add_argument("--head", type=str, nargs=2, action="append", default=[],
                        metavar=("NAME", "PATH"),
                        help="EA head: --head NAME PATH (repeatable)")

    # Data
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing benchmark data (default: REPO/data)")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["mt_bench", "gsm8k", "math_500", "svamp"],
                        help="Benchmarks to evaluate on")

    # Generation params
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)

    # Output
    parser.add_argument("--output-dir", type=str, default="entropy_analysis_results",
                        help="Directory to save results")

    # Debug
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Max questions per benchmark (for testing)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-visualizations", action="store_true",
                        help="Skip generating visualization plots")

    args = parser.parse_args()

    if not args.head:
        parser.error("At least one --head NAME PATH pair is required")
    head_paths = {name: path for name, path in args.head}

    # Set data directory
    if args.data_dir is None:
        args.data_dir = os.path.join(parent_dir, "data")

    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "per_benchmark"), exist_ok=True)

    # Load model with multiple heads
    print("Loading multi-head model...")
    model = MultiHeadEaModel.from_pretrained(
        base_model_path=args.base_model_path,
        head_paths=head_paths,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    print("Model loaded successfully!")

    device = next(model.base_model.parameters()).device
    tokenizer = model.get_tokenizer()
    heads_to_trace = list(model.heads.keys())

    # Warmup
    print("Warming up...")
    warmup_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello, how are you?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    warmup_ids = tokenizer([warmup_prompt], add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    for head_name in heads_to_trace:
        model.switch_head(head_name)
        model.reset_kv_cache()
        _ = trace_single_generation(model, warmup_ids, max_new_tokens=50, temperature=0.0)
    print("Warmup complete!")

    # Storage for all traces
    all_traces: List[QuestionTrace] = []
    all_stats: Dict[Tuple[str, str], EntropyStats] = {}

    # Process each benchmark
    for benchmark_name in args.benchmarks:
        print(f"\n{'='*60}")
        print(f"Processing benchmark: {benchmark_name}")
        print(f"{'='*60}")

        try:
            questions = load_questions(benchmark_name, args.data_dir, args.max_questions)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

        print(f"Loaded {len(questions)} questions")

        benchmark_traces: List[QuestionTrace] = []

        for question in tqdm(questions, desc=f"Tracing {benchmark_name}"):
            question_id = question.get("question_id", question.get("id", "unknown"))

            # Prepare input
            input_ids = prepare_inputs(tokenizer, question, device)

            # Trace ALL heads on this question
            head_results = trace_all_heads(
                model,
                input_ids,
                heads_to_trace,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            # Store results for each head
            for head_name, (records, tokens_gen) in head_results.items():
                trace = QuestionTrace(
                    question_id=question_id,
                    benchmark=benchmark_name,
                    head_name=head_name,
                    tokens_generated=tokens_gen,
                    records=records,
                )
                benchmark_traces.append(trace)
                all_traces.append(trace)

        # Compute per-benchmark statistics
        for head_name in heads_to_trace:
            stats = aggregate_stats(benchmark_traces, head_name, benchmark_name)
            all_stats[(head_name, benchmark_name)] = stats

            summary = stats.compute_summary()
            print(f"\n  {head_name} head on {benchmark_name}:")
            print(f"    Acceptance rate: {summary['acceptance_rate']:.2%}")
            print(f"    EA entropy (accept): {summary['ea_entropy']['accept']['mean']:.4f}")
            print(f"    EA entropy (reject): {summary['ea_entropy']['reject']['mean']:.4f}")
            print(f"    EA entropy delta (reject-accept): {summary['ea_entropy']['delta']:.4f}")

            # Save per-benchmark stats
            stats_path = os.path.join(args.output_dir, "per_benchmark", f"{benchmark_name}_{head_name}_stats.json")
            with open(stats_path, "w") as f:
                json.dump(summary, f, indent=2)

    # Save detailed traces
    print("\nSaving detailed traces...")
    traces_path = os.path.join(args.output_dir, "detailed_traces.jsonl")
    with open(traces_path, "w") as f:
        for trace in all_traces:
            f.write(json.dumps(trace.to_dict()) + "\n")
    print(f"Saved {len(all_traces)} traces to {traces_path}")

    # Compute and save overall summary
    print("\nComputing overall summary...")
    overall_summary = {}
    for (head_name, benchmark_name), stats in all_stats.items():
        if benchmark_name not in overall_summary:
            overall_summary[benchmark_name] = {}
        overall_summary[benchmark_name][head_name] = stats.compute_summary()

    summary_path = os.path.join(args.output_dir, "summary_stats.json")
    with open(summary_path, "w") as f:
        json.dump(overall_summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    # Print final results table
    print("\n" + "=" * 80)
    print("ENTROPY ANALYSIS RESULTS")
    print("=" * 80)

    print("\n### Acceptance Rate by Head x Benchmark ###\n")
    benchmarks = sorted(set(k[1] for k in all_stats.keys()))
    heads = sorted(set(k[0] for k in all_stats.keys()))

    header = f"{'Benchmark':<15}" + "".join(f"{h:<15}" for h in heads)
    print(header)
    print("-" * len(header))

    for bench in benchmarks:
        row = f"{bench:<15}"
        for head in heads:
            key = (head, bench)
            if key in all_stats:
                rate = all_stats[key].acceptance_rate
                row += f"{rate:.2%}".ljust(15)
            else:
                row += "N/A".ljust(15)
        print(row)

    print("\n### EA Entropy (Accept) ###\n")
    print(header)
    print("-" * len(header))

    for bench in benchmarks:
        row = f"{bench:<15}"
        for head in heads:
            key = (head, bench)
            if key in all_stats:
                summary = all_stats[key].compute_summary()
                ea_accept = summary["ea_entropy"]["accept"]["mean"]
                row += f"{ea_accept:.4f}".ljust(15)
            else:
                row += "N/A".ljust(15)
        print(row)

    print("\n### EA Entropy (Reject) ###\n")
    print(header)
    print("-" * len(header))

    for bench in benchmarks:
        row = f"{bench:<15}"
        for head in heads:
            key = (head, bench)
            if key in all_stats:
                summary = all_stats[key].compute_summary()
                ea_reject = summary["ea_entropy"]["reject"]["mean"]
                row += f"{ea_reject:.4f}".ljust(15)
            else:
                row += "N/A".ljust(15)
        print(row)

    # Generate visualizations
    if not args.skip_visualizations:
        print("\nGenerating visualizations...")
        generate_visualizations(all_stats, args.output_dir)

    print(f"\nAll results saved to {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
