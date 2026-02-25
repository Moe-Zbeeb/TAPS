#!/usr/bin/env python3
"""Visualize the confidence-routing decision for one decode step.

Generates a side-by-side figure of the two draft trees (head1 vs head2)
with nodes coloured by cumulative confidence.  The winning tree (higher
mean confidence) is labelled as the one submitted to the verifier.

Usage (from Eagle-Code root, with eagle env active):
    python scripts/visualize_routing.py \
        --base-model   /path/to/base \
        --ckpt1        checkpoints/Eagle-MathInstruct_20epochs \
        --ckpt2        checkpoints/Eagle-ShareGPT_20epochs \
        --head1-name   MathInstruct \
        --head2-name   ShareGPT \
        --prompt       "Solve: what is the integral of x^2 from 0 to 1?" \
        --output       routing_tree.pdf
"""

import argparse
import torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"text.usetex": False, "mathtext.default": "regular"})
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from eagle.model.ea_model_dual import DualEaModel
from eagle.model.kv_cache import initialize_past_key_values
from eagle.model.utils import reset_tree_mode


# ---------------------------------------------------------------------------
# Tree reconstruction
# ---------------------------------------------------------------------------

def build_tree(draft_tokens, retrieve_indices, draft_confidences, tokenizer):
    """Build a networkx DiGraph from topK_genrate outputs.

    Returns:
        G           DiGraph with nodes 0..N  (0 = root)
        node_conf   dict {node_idx -> confidence in [0,1]}
        node_label  dict {node_idx -> display string (token text + conf)}
    """
    tokens = draft_tokens[0]          # [N+1]
    N = tokens.shape[0] - 1          # number of draft tokens

    # confidence: root gets 1.0; node i (1-indexed) -> draft_confidences[i-1]
    node_conf = {0: 1.0}
    for i in range(1, N + 1):
        node_conf[i] = float(draft_confidences[i - 1].float())

    # parent map from retrieve_indices (each row = path root→leaf, -1 = pad)
    parent = {}
    for path in retrieve_indices:
        path = [p.item() for p in path if p.item() >= 0]
        for d in range(1, len(path)):
            child = path[d]
            if child not in parent:
                parent[child] = path[d - 1]

    G = nx.DiGraph()
    G.add_node(0)
    for child, par in parent.items():
        G.add_edge(par, child)

    def decode(idx):
        tid = tokens[idx].item()
        text = tokenizer.decode([tid], skip_special_tokens=False)
        text = text.replace("\n", "/n").replace("\r", "").replace(" ", "_")
        # Escape chars that trigger matplotlib mathtext
        text = text.replace("$", "S").replace("_", "-").replace("^", "")
        text = text.replace("{", "(").replace("}", ")")
        return text.strip() or f"[{tid}]"

    node_label = {}
    for n in G.nodes():
        txt = decode(n)
        if n == 0:
            node_label[n] = f"{txt}\n[root]"
        else:
            node_label[n] = f"{txt}\n{node_conf[n]:.2f}"

    return G, node_conf, node_label


# ---------------------------------------------------------------------------
# Tree layout (Reingold-Tilford style, pure Python)
# ---------------------------------------------------------------------------

def _tree_layout(G, root=0):
    """Return {node: (x, y)} with root at top, leaves at bottom."""
    # BFS order
    depths = {root: 0}
    queue = [root]
    while queue:
        nxt = []
        for n in queue:
            for c in G.successors(n):
                if c not in depths:
                    depths[c] = depths[n] + 1
                    nxt.append(c)
        queue = nxt

    max_depth = max(depths.values()) if depths else 0

    # Group nodes per depth and assign x positions
    from collections import defaultdict
    by_depth = defaultdict(list)
    for n, d in depths.items():
        by_depth[d].append(n)

    pos = {}
    for d, nodes in by_depth.items():
        for i, n in enumerate(nodes):
            pos[n] = (i - (len(nodes) - 1) / 2.0, -d)

    return pos


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_tree(ax, G, node_conf, node_label, title, cmap, norm, is_winner):
    pos = _tree_layout(G)
    node_list = list(G.nodes())

    colors = [cmap(norm(node_conf.get(n, 0.0))) for n in node_list]

    # edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#aaaaaa", width=1.2,
        arrows=True, arrowsize=10,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.0",
    )

    # nodes
    border_color = "#1a7a3a" if is_winner else "#888888"
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=node_list,
        node_color=colors,
        node_size=900,
        edgecolors=border_color,
        linewidths=2.0 if is_winner else 1.0,
    )

    # labels
    nx.draw_networkx_labels(
        G, pos, labels=node_label, ax=ax,
        font_size=5.5, font_color="black",
    )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10,
                 color="#1a7a3a" if is_winner else "#555555")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",  required=True)
    parser.add_argument("--ckpt1",       required=True, help="Draft head 1 checkpoint")
    parser.add_argument("--ckpt2",       required=True, help="Draft head 2 checkpoint")
    parser.add_argument("--head1-name",  default="Head 1")
    parser.add_argument("--head2-name",  default="Head 2")
    parser.add_argument("--prompt",      required=True)
    parser.add_argument("--output",      default="routing_tree.pdf")
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth",       type=int, default=5)
    parser.add_argument("--top-k",       type=int, default=10)
    args = parser.parse_args()

    print("Loading model …")
    model = DualEaModel.from_pretrained(
        base_model_path=args.base_model,
        ea_model_path1=args.ckpt1,
        ea_model_path2=args.ckpt2,
        head1_name=args.head1_name,
        head2_name=args.head2_name,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = model.get_tokenizer()

    # Tokenize
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(next(model.base_model.parameters()).device)

    print("Prefilling …")
    past_key_values, _, _ = initialize_past_key_values(model.base_model, max_length=2048)
    model.past_key_values = past_key_values
    reset_tree_mode(model)

    with torch.no_grad():
        outputs, orig, hidden_states = model.forward(
            input_ids, past_key_values=past_key_values, output_orig=True
        )

    # Greedy first token (temperature=0 for reproducibility)
    token = torch.argmax(orig[:, -1])[None, None].to(input_ids.device)
    context_ids = torch.cat((input_ids, token), dim=1)

    print("Generating draft trees …")
    with torch.no_grad():
        r1 = model.ea_layer1.topK_genrate(
            hidden_states, context_ids, model.base_model.lm_head,
            logits_processor=None, return_confidence=True,
        )
        r2 = model.ea_layer2.topK_genrate(
            hidden_states, context_ids, model.base_model.lm_head,
            logits_processor=None, return_confidence=True,
        )

    dt1, ri1, _, _, confs1 = r1
    dt2, ri2, _, _, confs2 = r2

    mean1 = confs1.float().mean().item()
    mean2 = confs2.float().mean().item()
    head1_wins = mean1 >= mean2

    print(f"  {args.head1_name:20s}  mean conf = {mean1:.4f}  {'← SUBMITTED' if head1_wins else ''}")
    print(f"  {args.head2_name:20s}  mean conf = {mean2:.4f}  {'← SUBMITTED' if not head1_wins else ''}")

    G1, nc1, nl1 = build_tree(dt1, ri1, confs1, tokenizer)
    G2, nc2, nl2 = build_tree(dt2, ri2, confs2, tokenizer)

    # Shared colour scale across both trees
    all_confs = list(nc1.values()) + list(nc2.values())
    norm = mcolors.Normalize(vmin=min(all_confs), vmax=max(all_confs))
    cmap = cm.RdYlGn

    tag1 = "[SUBMITTED TO VERIFIER]" if head1_wins else "[discarded]"
    tag2 = "[SUBMITTED TO VERIFIER]" if not head1_wins else "[discarded]"
    title1 = f"{args.head1_name}   mean conf = {mean1:.3f}   {tag1}"
    title2 = f"{args.head2_name}   mean conf = {mean2:.3f}   {tag2}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))
    fig.suptitle(
        f"Confidence-Routed Draft Trees\n\"{args.prompt[:100]}\"",
        fontsize=12, y=1.02,
    )

    draw_tree(ax1, G1, nc1, nl1, title1, cmap, norm, is_winner=head1_wins)
    draw_tree(ax2, G2, nc2, nl2, title2, cmap, norm, is_winner=not head1_wins)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.55, pad=0.02, aspect=30)
    cbar.set_label("Cumulative Token Confidence", fontsize=10)

    plt.tight_layout()
    out_path = args.output.replace(".pdf", ".png") if args.output.endswith(".pdf") else args.output
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
