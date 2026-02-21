"""Dual-head EAGLE-2 model for confidence-based step-level routing.

At each decode step, both draft heads generate a candidate tree.  The tree
whose tokens carry higher mean cumulative confidence is submitted to the
verifier.  Both heads always stay in sync because they are called with the
same accepted hidden states every step.
"""

import json
import os
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
from .utils import prepare_logits_processor, reset_tree_mode, tree_decoding, evaluate_posterior
from .kv_cache import initialize_past_key_values
from .cnets1 import Model as Model1
from .configs import EConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_ea_weights(ea_model_path, device):
    """Return (configpath, state_dict) for one EA checkpoint."""
    configpath = os.path.join(ea_model_path, "config.json")
    if not os.path.exists(configpath):
        configpath = hf_hub_download(ea_model_path, "config.json")

    try:
        load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
        if not os.path.exists(load_model_path):
            load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
        state_dict = torch.load(load_model_path, map_location=device)
    except Exception:
        from safetensors.torch import load_file
        load_model_path = os.path.join(ea_model_path, "model.safetensors")
        if not os.path.exists(load_model_path):
            load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
        state_dict = load_file(load_model_path)

    return configpath, state_dict


def _build_ea_layer(configpath, state_dict, total_token, depth, top_k, threshold,
                    base_model_path, dtype, device):
    """Instantiate and load one EAGLE-2 draft head."""
    config = EConfig.from_pretrained(configpath)
    with open(configpath, "r") as f:
        con = json.loads(f.read())
    bias = con.get("bias", True)

    ea_layer = Model1(
        config,
        bias=bias,
        total_tokens=total_token,
        depth=depth,
        top_k=top_k,
        threshold=threshold,
        path=base_model_path,
        load_emb=True,
    )
    ea_layer.diff_device = False
    ea_layer.load_state_dict(state_dict, strict=False)
    ea_layer.to(dtype).to(device)
    ea_layer.init_tree()
    return ea_layer


# ---------------------------------------------------------------------------
# DualEaModel
# ---------------------------------------------------------------------------

class DualEaModel(nn.Module):
    """EAGLE-2 speculative decoding with two draft heads.

    Each decode step:
      1. Both heads generate a draft tree (with confidence scores).
      2. The tree with higher mean token confidence is submitted to the verifier.
      3. The verifier accepts/rejects tokens normally.
      4. Both heads advance together using the same accepted hidden states.

    Args:
        base_model: the loaded verifier LLM (KVLlama / KVQwen2 / …).
        base_model_name_or_path: path used to load the tokenizer.
        ea_layer1: first draft head (e.g. MathInstruct).
        ea_layer2: second draft head (e.g. ShareGPT).
        head1_name: label used in routing statistics.
        head2_name: label used in routing statistics.
    """

    def __init__(self, base_model, base_model_name_or_path,
                 ea_layer1, ea_layer2,
                 head1_name="head1", head2_name="head2"):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path, use_fast=False
        )
        self.use_eagle3 = False  # always EAGLE-2

        self.ea_layer1 = ea_layer1
        self.ea_layer2 = ea_layer2
        self.head1_name = head1_name
        self.head2_name = head2_name

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        base_model_path,
        ea_model_path1,
        ea_model_path2,
        head1_name="head1",
        head2_name="head2",
        total_token=60,
        depth=5,
        top_k=10,
        threshold=1.0,
        **kwargs,
    ):
        """Load base model once, then build two EAGLE-2 draft heads.

        ``**kwargs`` are forwarded to the base-model ``from_pretrained``
        (e.g. ``torch_dtype``, ``device_map``, ``low_cpu_mem_usage``).
        """
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == "LlamaForCausalLM":
            base_model = KVLlamaForCausalLM.from_pretrained(base_model_path, **kwargs)
        elif Type == "Qwen2ForCausalLM":
            base_model = KVQwen2ForCausalLM.from_pretrained(base_model_path, **kwargs)
        elif Type == "Qwen3ForCausalLM":
            base_model = KVQwen3ForCausalLM.from_pretrained(base_model_path, **kwargs)
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(base_model_path, **kwargs)

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        dtype = base_model.dtype

        configpath1, state_dict1 = _load_ea_weights(ea_model_path1, device)
        configpath2, state_dict2 = _load_ea_weights(ea_model_path2, device)

        ea_layer1 = _build_ea_layer(
            configpath1, state_dict1, total_token, depth, top_k, threshold,
            base_model_path, dtype, device,
        )
        ea_layer2 = _build_ea_layer(
            configpath2, state_dict2, total_token, depth, top_k, threshold,
            base_model_path, dtype, device,
        )

        return cls(base_model, base_model_path, ea_layer1, ea_layer2,
                   head1_name=head1_name, head2_name=head2_name)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                output_orig=False, position_ids=None):
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
        return outputs, hidden_states

    # ------------------------------------------------------------------
    # Core: step-level confidence routing
    # ------------------------------------------------------------------

    def _draft_best(self, hidden_states, input_ids, logits_processor):
        """Run both heads; return the draft tree with higher mean confidence.

        Returns:
            (draft_tokens, retrieve_indices, tree_mask, tree_position_ids,
             chosen_name)
        """
        r1 = self.ea_layer1.topK_genrate(
            hidden_states, input_ids, self.base_model.lm_head,
            logits_processor, return_confidence=True,
        )
        r2 = self.ea_layer2.topK_genrate(
            hidden_states, input_ids, self.base_model.lm_head,
            logits_processor, return_confidence=True,
        )
        draft_tokens1, retrieve_indices1, tree_mask1, tree_pos1, confs1 = r1
        draft_tokens2, retrieve_indices2, tree_mask2, tree_pos2, confs2 = r2

        if confs1.float().mean() >= confs2.float().mean():
            return draft_tokens1, retrieve_indices1, tree_mask1, tree_pos1, self.head1_name
        return draft_tokens2, retrieve_indices2, tree_mask2, tree_pos2, self.head2_name

    @torch.no_grad()
    def dual_eagenerate(
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
        """Speculative decoding with step-level head routing.

        Returns:
            input_ids                         (log=False)
            input_ids, new_token, idx         (log=True)
        """
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        logits_processor = None
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(
                temperature=temperature, top_p=top_p, top_k=top_k
            )

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer1.reset_kv()
        self.ea_layer2.reset_kv()

        # --- KV cache (shared verifier cache) ---
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            past_key_values, past_key_values_data, current_length_data = (
                initialize_past_key_values(self.base_model, max_length=max_length)
            )
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)

        # --- Prefill ---
        # input_ids stays at length L here; sample_token is kept separate.
        # This matches the initialize_tree / update_inference_inputs contract so
        # that topK_genrate's stable_kv slice is always consistent with
        # accept_hidden_states (both covering exactly the newly accepted tokens).
        outputs, orig, hidden_states = self.forward(
            input_ids, past_key_values=past_key_values, output_orig=True
        )

        # Sample first token but do NOT append to input_ids yet.
        if logits_processor is not None:
            first_logits = logits_processor(None, orig[:, -1])
            token = torch.multinomial(torch.softmax(first_logits, dim=-1), 1)
        else:
            token = torch.argmax(orig[:, -1])[None, None]

        # --- Initial draft: pass cat(input_ids, token) but keep input_ids at length L ---
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _ = (
            self._draft_best(
                hidden_states,
                torch.cat((input_ids, token.to(input_ids.device)), dim=1),
                logits_processor,
            )
        )

        new_token = 0
        loop_max = max_length - max(self.ea_layer1.total_tokens, self.ea_layer2.total_tokens) - 10

        for idx in range(loop_max):
            self.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)

            # --- Verify chosen draft ---
            # input_ids is still length L on first iteration; tree positions
            # start at L so there is no positional gap in the KV cache.
            logits, hidden_state_new, _ = tree_decoding(
                self, draft_tokens, past_key_values,
                tree_position_ids, input_ids, retrieve_indices,
            )

            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            # --- Update shared KV cache ---
            # candidates[best_candidate, 0] is the sample_token from the
            # previous step (position 0 of every tree), so appending
            # candidates[..., :accept_length+1] both commits the sample_token
            # and the newly accepted draft tokens.
            prev_input_len = input_ids.shape[1]
            select_indices = (
                retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
            )
            input_ids = torch.cat(
                [input_ids,
                 candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)],
                dim=-1,
            )
            for pkv_data in past_key_values_data:
                tgt = pkv_data[..., select_indices.to(pkv_data.device), :]
                dst = pkv_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
                dst.copy_(tgt, non_blocking=True)
            current_length_data.fill_(prev_input_len + tgt.shape[-2])

            # --- Accepted hidden states ---
            accept_hidden = (
                hidden_state_new[:, retrieve_indices]
                [:, best_candidate, : accept_length + 1]
            )

            # --- Sample next token ---
            if logits_processor is not None:
                next_token = torch.multinomial(sample_p, 1)
            else:
                next_token = torch.argmax(sample_p)[None, None]

            new_token += accept_length + 1

            # --- Stop conditions ---
            if is_llama3 and stop_token_id in input_ids[0, input_len:].tolist():
                break
            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

            # --- Next draft: pass cat(input_ids, next_token) without updating input_ids ---
            # next_token becomes candidates[0] in the next tree (position 0 of
            # draft_tokens), which will be committed in the next iteration's
            # KV update, keeping the stable_kv slice exactly accept_hidden sized.
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _ = (
                self._draft_best(
                    accept_hidden,
                    torch.cat((input_ids, next_token.to(input_ids.device)), dim=1),
                    logits_processor,
                )
            )

        if log:
            return input_ids, new_token, idx
        return input_ids
