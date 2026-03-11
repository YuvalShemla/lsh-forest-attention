"""
Attention Entropy & Top-K Concentration Verification.

Reproduces Andoni's two diagnostic plots:
1. Top-K attention concentration vs query position
2. Attention entropy vs query position (with 10%/50% reference curves)

Runs on multiple examples and averages, to verify the single-example results.
Saves per-example + averaged plots.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from algorithms.base import softmax

# ── Hyperparameters ──
DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/attention_vectors_long_bench_llama_8b.jsonl')
NUM_EXAMPLES = 20
HEAD_DIM = 128
SEED = 42
TOPK_VALUES = [10, 50, 100, 200, 500]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../results/attention_entropy')

# Subsample query positions to keep runtime reasonable (every 8th position)
QUERY_STRIDE = 8


def compute_entropy_and_topk(Q, K, head_dim, stride=1):
    """
    For each query position, compute full causal attention weights,
    then entropy and top-K mass.

    Returns:
        positions: [num_queries] query positions evaluated
        entropies: [num_queries]
        topk_masses: dict of {k: [num_queries]} for each k in TOPK_VALUES
    """
    seq_len = Q.shape[0]
    positions = list(range(0, seq_len, stride))
    entropies = np.zeros(len(positions))
    topk_masses = {k: np.zeros(len(positions)) for k in TOPK_VALUES}

    for idx, qpos in enumerate(positions):
        q = Q[qpos]
        valid_keys = K[:qpos + 1]
        n_valid = qpos + 1

        # Compute attention weights
        logits = (q @ valid_keys.T) / np.sqrt(head_dim)
        weights = softmax(logits)

        # Entropy
        log_weights = np.log(weights + 1e-12)
        entropies[idx] = -np.sum(weights * log_weights)

        # Top-K mass
        sorted_weights = np.sort(weights)[::-1]
        for k in TOPK_VALUES:
            topk_masses[k][idx] = sorted_weights[:min(k, n_valid)].sum()

    return np.array(positions), entropies, topk_masses


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect results per example per layer
    all_results = {layer: [] for layer in ['first_layer', 'last_layer']}

    print(f"Loading {NUM_EXAMPLES} examples, stride={QUERY_STRIDE}")
    print(f"Top-K values: {TOPK_VALUES}\n")

    with open(DATA_PATH, 'r') as f:
        for ex_idx, line in enumerate(f):
            if ex_idx >= NUM_EXAMPLES:
                break
            example = json.loads(line)
            domain = example.get('domain', 'unknown')
            seq_len = example.get('sequence_length', 0)
            print(f"  [{ex_idx:3d}] {domain} (seq_len={seq_len})")

            for layer_name in ['first_layer', 'last_layer']:
                Q = np.array(example[layer_name]['Q'], dtype=np.float64)
                K = np.array(example[layer_name]['K'], dtype=np.float64)

                positions, entropies, topk_masses = compute_entropy_and_topk(
                    Q, K, HEAD_DIM, stride=QUERY_STRIDE
                )

                all_results[layer_name].append({
                    'positions': positions,
                    'entropies': entropies,
                    'topk_masses': topk_masses,
                    'example_idx': ex_idx,
                    'domain': domain,
                })

            del example

    # ═══════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════

    layer_labels = {'first_layer': 'First Layer (0)', 'last_layer': 'Last Layer (31)'}
    topk_colors = {10: '#e74c3c', 50: '#e67e22', 100: '#2ecc71', 200: '#3498db', 500: '#9b59b6'}

    # ── Plot 1: Single example (#0) — matches Andoni's plots ──
    for layer_name in ['first_layer', 'last_layer']:
        r = all_results[layer_name][0]
        positions = r['positions']
        entropies = r['entropies']
        topk_masses = r['topk_masses']

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Top-K concentration
        ax = axes[0]
        for k in TOPK_VALUES:
            ax.plot(positions, 100 * topk_masses[k], linewidth=0.8, color=topk_colors[k],
                    label=f'Top-{k}', alpha=0.8)
        ax.set_ylabel('% of Total Attention Mass')
        ax.set_title(f'Top-K Attention Concentration — {layer_labels[layer_name]} (Example #0)')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.2)

        # Entropy
        ax = axes[1]
        ax.plot(positions, entropies, linewidth=0.8, color='green', alpha=0.8, label='Entropy')

        # Reference curves: uniform over 10% and 50% of keys
        n_valid = positions + 1  # number of valid keys at each position
        ref_50 = np.log(np.maximum(1, 0.5 * n_valid))
        ref_10 = np.log(np.maximum(1, 0.1 * n_valid))
        ax.plot(positions, ref_50, '--', color='gray', linewidth=1.2, alpha=0.7,
                label='Uniform over 50% keys')
        ax.plot(positions, ref_10, '--', color='blue', linewidth=1.2, alpha=0.7,
                label='Uniform over 10% keys')

        ax.set_xlabel('Query Position')
        ax.set_ylabel('Entropy (nats)')
        ax.set_title(f'Attention Entropy — {layer_labels[layer_name]} (Example #0)')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.2)

        plt.tight_layout()
        short = 'first' if layer_name == 'first_layer' else 'last'
        plt.savefig(os.path.join(OUTPUT_DIR, f'example0_{short}_layer.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: example0_{short}_layer.png")

    # ── Plot 2: Averaged over all examples (mean ± std band) ──
    for layer_name in ['first_layer', 'last_layer']:
        # All examples should have the same positions (same seq_len, same stride)
        positions = all_results[layer_name][0]['positions']
        n_pos = len(positions)

        # Stack entropies and topk
        entropy_stack = np.array([r['entropies'] for r in all_results[layer_name]])  # [N_ex, n_pos]
        topk_stacks = {k: np.array([r['topk_masses'][k] for r in all_results[layer_name]])
                       for k in TOPK_VALUES}

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Top-K concentration (mean ± std)
        ax = axes[0]
        for k in TOPK_VALUES:
            mean_vals = 100 * topk_stacks[k].mean(axis=0)
            std_vals = 100 * topk_stacks[k].std(axis=0)
            ax.plot(positions, mean_vals, linewidth=1.2, color=topk_colors[k], label=f'Top-{k}')
            ax.fill_between(positions, mean_vals - std_vals, mean_vals + std_vals,
                            color=topk_colors[k], alpha=0.15)
        ax.set_ylabel('% of Total Attention Mass')
        ax.set_title(f'Top-K Attention Concentration — {layer_labels[layer_name]} '
                     f'(mean ± std, {NUM_EXAMPLES} examples)')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.2)

        # Entropy (mean ± std)
        ax = axes[1]
        ent_mean = entropy_stack.mean(axis=0)
        ent_std = entropy_stack.std(axis=0)
        ax.plot(positions, ent_mean, linewidth=1.2, color='green', label='Entropy (mean)')
        ax.fill_between(positions, ent_mean - ent_std, ent_mean + ent_std,
                        color='green', alpha=0.15)

        # Reference curves
        n_valid = positions + 1
        ref_50 = np.log(np.maximum(1, 0.5 * n_valid))
        ref_10 = np.log(np.maximum(1, 0.1 * n_valid))
        ax.plot(positions, ref_50, '--', color='gray', linewidth=1.2, alpha=0.7,
                label='Uniform over 50% keys')
        ax.plot(positions, ref_10, '--', color='blue', linewidth=1.2, alpha=0.7,
                label='Uniform over 10% keys')

        ax.set_xlabel('Query Position')
        ax.set_ylabel('Entropy (nats)')
        ax.set_title(f'Attention Entropy — {layer_labels[layer_name]} '
                     f'(mean ± std, {NUM_EXAMPLES} examples)')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.2)

        plt.tight_layout()
        short = 'first' if layer_name == 'first_layer' else 'last'
        plt.savefig(os.path.join(OUTPUT_DIR, f'averaged_{short}_layer.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: averaged_{short}_layer.png")

    # ── Plot 3: All examples overlaid (spaghetti plot) ──
    for layer_name in ['first_layer', 'last_layer']:
        positions = all_results[layer_name][0]['positions']

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Entropy spaghetti
        ax = axes[1]
        for r in all_results[layer_name]:
            ax.plot(positions, r['entropies'], linewidth=0.4, alpha=0.3, color='green')
        n_valid = positions + 1
        ref_50 = np.log(np.maximum(1, 0.5 * n_valid))
        ref_10 = np.log(np.maximum(1, 0.1 * n_valid))
        ax.plot(positions, ref_50, '--', color='gray', linewidth=1.5, alpha=0.8,
                label='Uniform 50%')
        ax.plot(positions, ref_10, '--', color='blue', linewidth=1.5, alpha=0.8,
                label='Uniform 10%')
        ax.set_xlabel('Query Position')
        ax.set_ylabel('Entropy (nats)')
        ax.set_title(f'Attention Entropy — {layer_labels[layer_name]} '
                     f'(all {NUM_EXAMPLES} examples overlaid)')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.2)

        # Top-100 spaghetti (pick one representative K)
        ax = axes[0]
        for r in all_results[layer_name]:
            ax.plot(positions, 100 * r['topk_masses'][100], linewidth=0.4, alpha=0.3,
                    color=topk_colors[100])
        ax.set_ylabel('Top-100 Mass (%)')
        ax.set_title(f'Top-100 Attention Concentration — {layer_labels[layer_name]} '
                     f'(all {NUM_EXAMPLES} examples)')
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.2)

        plt.tight_layout()
        short = 'first' if layer_name == 'first_layer' else 'last'
        plt.savefig(os.path.join(OUTPUT_DIR, f'spaghetti_{short}_layer.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: spaghetti_{short}_layer.png")

    # ── Print summary stats for late queries (last 100 positions) ──
    print("\n" + "=" * 70)
    print("ENTROPY SUMMARY — Last 100 positions (late queries)")
    print("=" * 70)
    for layer_name in ['first_layer', 'last_layer']:
        positions = all_results[layer_name][0]['positions']
        # Find indices where position >= seq_len - 100
        late_mask = positions >= (positions[-1] - 100)
        late_entropies = []
        for r in all_results[layer_name]:
            late_entropies.extend(r['entropies'][late_mask].tolist())
        late_entropies = np.array(late_entropies)
        n_valid_late = positions[late_mask] + 1
        ref_50_late = np.log(0.5 * n_valid_late).mean()
        ref_10_late = np.log(0.1 * n_valid_late).mean()
        eff_support = np.exp(late_entropies) / (positions[-1] + 1) * 100  # as % of full context

        print(f"\n{layer_labels[layer_name]}:")
        print(f"  Entropy:  mean={np.mean(late_entropies):.3f}  std={np.std(late_entropies):.3f}  "
              f"[{np.min(late_entropies):.3f}, {np.max(late_entropies):.3f}]")
        print(f"  Ref 50%:  {ref_50_late:.3f}")
        print(f"  Ref 10%:  {ref_10_late:.3f}")
        print(f"  Effective support: mean={np.mean(eff_support):.1f}%  "
              f"[{np.min(eff_support):.1f}%, {np.max(eff_support):.1f}%]")

    # ── Save JSON summary ──
    summary = {}
    for layer_name in ['first_layer', 'last_layer']:
        positions = all_results[layer_name][0]['positions']
        late_mask = positions >= (positions[-1] - 100)
        all_late_ent = []
        per_ex = []
        for r in all_results[layer_name]:
            late_ent = r['entropies'][late_mask]
            all_late_ent.extend(late_ent.tolist())
            per_ex.append({
                'example_idx': r['example_idx'],
                'domain': r['domain'],
                'late_entropy_mean': float(np.mean(late_ent)),
                'late_entropy_std': float(np.std(late_ent)),
                'full_entropy_mean': float(np.mean(r['entropies'])),
            })
        all_late_ent = np.array(all_late_ent)
        summary[layer_name] = {
            'late_entropy': {
                'mean': float(np.mean(all_late_ent)),
                'std': float(np.std(all_late_ent)),
                'min': float(np.min(all_late_ent)),
                'max': float(np.max(all_late_ent)),
                'effective_support_pct_mean': float(np.mean(np.exp(all_late_ent) / (positions[-1]+1) * 100)),
            },
            'per_example': per_ex,
        }

    json_path = os.path.join(OUTPUT_DIR, 'entropy_stats.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {json_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
