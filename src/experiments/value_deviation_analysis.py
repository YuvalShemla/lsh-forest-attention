#!/usr/bin/env python3
"""
Value Deviation Analysis (MagicPIG Figure 10 Replication)

Visualizes why approximating attention weights matters more than value differences.
For each query, plots two lines across the sequence:
  1. Pre-softmax attention scores: q·k_i^T / sqrt(d)  (wildly varying)
  2. Value deviation from output: log||v_i - o||       (relatively flat)

If the paper's intuition holds, attention scores spike wildly while value deviations
stay mostly horizontal — justifying the focus on attention weight approximation.

Results saved to: results/value_deviation_analysis/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from algorithms.base import compute_ground_truth_attention
from visualization.plot_utils import setup_style, save_figure

# ============================================================================
# HYPERPARAMETERS - MODIFY HERE
# ============================================================================

NUM_EXAMPLES = 100            # Number of JSONL examples to process
LAYERS_TO_TEST = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42

# Query positions to visualize (picked from the last portion of the sequence
# so there are enough keys to attend to). We select a few spread-out positions.
NUM_QUERY_POSITIONS = 5       # Number of individual query plots per example

# For the aggregated statistics plot, average over many queries
NUM_QUERIES_AGGREGATE = 100

DATA_PATH = '../../data/attention_vectors_long_bench_llama_8b.jsonl'
OUTPUT_DIR = '../../results/value_deviation_analysis'

# ============================================================================
# END HYPERPARAMETERS
# ============================================================================


def compute_value_deviation_metrics(q, K, V, query_pos, head_dim):
    """
    Compute the two metrics for one query position.

    Returns:
        positions: array of position indices [0, ..., query_pos]
        scores: pre-softmax attention scores q·k_i^T / sqrt(d)
        log_deviations: log||v_i - o|| for each position
        output: the true attention output vector o
    """
    # Ground truth attention
    output, logits, weights, _ = compute_ground_truth_attention(
        q, K, V, query_pos, head_dim
    )

    valid_values = V[:query_pos + 1]
    positions = np.arange(query_pos + 1)

    # Pre-softmax scores (already computed as logits)
    scores = logits

    # Value deviation: log||v_i - o||
    diffs = valid_values - output[np.newaxis, :]  # [num_valid, head_dim]
    norms = np.linalg.norm(diffs, axis=1)         # [num_valid]
    # Clamp to avoid log(0)
    norms = np.maximum(norms, 1e-10)
    log_deviations = np.log(norms)

    return positions, scores, log_deviations, output


def plot_single_query(positions, scores, log_deviations, query_pos,
                      example_id, layer_name, output_dir):
    """Plot the two-line figure for a single query position."""
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Attention scores (left y-axis)
    color_scores = '#d95f02'
    ax1.plot(positions, scores, color=color_scores, alpha=0.7, linewidth=0.8,
             label=r'Attention score $q k_i^T / \sqrt{d}$')
    ax1.set_xlabel('Position $i$')
    ax1.set_ylabel(r'Attention Score $q k_i^T / \sqrt{d}$', color=color_scores)
    ax1.tick_params(axis='y', labelcolor=color_scores)

    # Value deviation (right y-axis)
    ax2 = ax1.twinx()
    color_dev = '#1b9e77'
    ax2.plot(positions, log_deviations, color=color_dev, alpha=0.7, linewidth=0.8,
             label=r'Value deviation $\log \|v_i - o\|$')
    ax2.set_ylabel(r'Value Deviation $\log \|v_i - o\|$', color=color_dev)
    ax2.tick_params(axis='y', labelcolor=color_dev)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
               framealpha=0.9, fontsize=9)

    layer_display = 'Layer 0' if 'first' in layer_name else 'Layer 31'
    ax1.set_title(
        f'Attention Scores vs Value Deviation — {layer_display}, '
        f'Query pos={query_pos} (Ex: {example_id})',
        fontsize=12
    )

    fig.tight_layout()
    fname = f'{example_id}_{layer_name}_qpos{query_pos}.png'
    save_figure(fig, output_dir / fname)


def plot_overlay_shared_axis(positions, scores, log_deviations, query_pos,
                             example_id, layer_name, output_dir):
    """
    Plot both metrics on the SAME y-axis (both are scalars, just different scales).
    This is closer to the original MagicPIG figure style.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(positions, scores, color='#d95f02', alpha=0.7, linewidth=0.8,
            label=r'Attention score $q k_i^T / \sqrt{d}$')
    ax.plot(positions, log_deviations, color='#1b9e77', alpha=0.7, linewidth=0.8,
            label=r'Value deviation $\log \|v_i - o\|$')

    ax.set_xlabel('Position $i$')
    ax.set_ylabel('Value')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

    layer_display = 'Layer 0' if 'first' in layer_name else 'Layer 31'
    ax.set_title(
        f'Attention Scores vs Value Deviation (Shared Axis) — {layer_display}, '
        f'Query pos={query_pos} (Ex: {example_id})',
        fontsize=12
    )

    fig.tight_layout()
    fname = f'{example_id}_{layer_name}_qpos{query_pos}_shared.png'
    save_figure(fig, output_dir / fname)


def plot_aggregated_stats(all_score_stds, all_dev_stds, layer_name,
                          output_dir, example_ids):
    """
    Plot aggregated statistics: std of attention scores vs std of value deviations
    across many queries, showing that scores vary much more than value deviations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: box plot of standard deviations
    ax = axes[0]
    data = [all_score_stds, all_dev_stds]
    bp = ax.boxplot(data, labels=[r'$\mathrm{std}(q k_i^T / \sqrt{d})$',
                                   r'$\mathrm{std}(\log \|v_i - o\|)$'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#d95f02')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('#1b9e77')
    bp['boxes'][1].set_alpha(0.6)
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Variability: Attention Scores vs Value Deviations')

    # Right: ratio histogram
    ax = axes[1]
    ratios = np.array(all_score_stds) / (np.array(all_dev_stds) + 1e-10)
    ax.hist(ratios, bins=30, color='#7570b3', alpha=0.7, edgecolor='white')
    ax.axvline(np.median(ratios), color='red', linestyle='--', linewidth=2,
               label=f'Median ratio: {np.median(ratios):.1f}x')
    ax.set_xlabel(r'$\mathrm{std}(\mathrm{scores}) \;/\; \mathrm{std}(\mathrm{value\ dev.})$')
    ax.set_ylabel('Count')
    ax.set_title('Ratio of Score Variability to Value Deviation Variability')
    ax.legend(fontsize=10)

    layer_display = 'Layer 0' if 'first' in layer_name else 'Layer 31'
    fig.suptitle(f'Aggregated Value Deviation Analysis — {layer_display}',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir / f'aggregated_{layer_name}.png')

    return np.median(ratios), np.mean(ratios)


def plot_summary_multi_query(all_positions, all_scores, all_log_devs,
                             query_positions, example_id, layer_name, output_dir):
    """
    Single figure with subplots for multiple query positions from the same example.
    """
    n = len(query_positions)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for idx, (pos_arr, sc, ld, qp) in enumerate(
        zip(all_positions, all_scores, all_log_devs, query_positions)
    ):
        ax1 = axes[idx]
        color_scores = '#d95f02'
        color_dev = '#1b9e77'

        ax1.plot(pos_arr, sc, color=color_scores, alpha=0.7, linewidth=0.6,
                 label=r'$q k_i^T / \sqrt{d}$')
        ax1.set_ylabel('Attn Score', color=color_scores, fontsize=9)
        ax1.tick_params(axis='y', labelcolor=color_scores, labelsize=8)

        ax2 = ax1.twinx()
        ax2.plot(pos_arr, ld, color=color_dev, alpha=0.7, linewidth=0.6,
                 label=r'$\log \|v_i - o\|$')
        ax2.set_ylabel('Value Dev.', color=color_dev, fontsize=9)
        ax2.tick_params(axis='y', labelcolor=color_dev, labelsize=8)

        ax1.set_title(f'Query position = {qp} (seq len = {qp + 1})', fontsize=10)

        if idx == 0:
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                       fontsize=8, framealpha=0.9)

    axes[-1].set_xlabel('Position $i$')
    layer_display = 'Layer 0' if 'first' in layer_name else 'Layer 31'
    fig.suptitle(
        f'MagicPIG Fig. 10 Replication — {layer_display} (Ex: {example_id})',
        fontsize=13, y=1.01
    )
    fig.tight_layout()
    save_figure(fig, output_dir / f'{example_id}_{layer_name}_multi_query.png')


def main():
    setup_style()
    np.random.seed(SEED)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(DATA_PATH)
    if not data_path.exists():
        # Try from script dir
        data_path = Path(__file__).parent / DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    print("=" * 70)
    print("Value Deviation Analysis (MagicPIG Figure 10 Replication)")
    print("=" * 70)
    print(f"  Examples: {NUM_EXAMPLES}")
    print(f"  Layers:   {LAYERS_TO_TEST}")
    print(f"  Query positions per example: {NUM_QUERY_POSITIONS}")
    print(f"  Queries for aggregation: {NUM_QUERIES_AGGREGATE}")
    print(f"  Output:   {output_dir}")
    print()

    all_results = {}

    with open(data_path, 'r') as f:
        for ex_idx, line in enumerate(f):
            if ex_idx >= NUM_EXAMPLES:
                break

            example = json.loads(line)
            example_id = example['example_id']
            seq_len = example['sequence_length']
            print(f"Example {ex_idx + 1}/{NUM_EXAMPLES}: {example_id} "
                  f"(seq_len={seq_len})")

            for layer_name in LAYERS_TO_TEST:
                layer_data = example[layer_name]
                Q = np.array(layer_data['Q'], dtype=np.float32)
                K = np.array(layer_data['K'], dtype=np.float32)
                V = np.array(layer_data['V'], dtype=np.float32)

                layer_dir = output_dir / layer_name
                layer_dir.mkdir(parents=True, exist_ok=True)

                # ============================================================
                # 1. Individual query position plots (MagicPIG Fig 10 style)
                # ============================================================
                print(f"  [{layer_name}] Plotting individual query positions...")

                # Pick query positions spread across the latter half of the sequence
                min_pos = max(50, seq_len // 4)
                max_pos = seq_len - 1
                query_positions = np.linspace(
                    min_pos, max_pos, NUM_QUERY_POSITIONS, dtype=int
                )
                # Deduplicate
                query_positions = sorted(set(query_positions.tolist()))

                multi_positions = []
                multi_scores = []
                multi_devs = []

                for qp in tqdm(query_positions, desc=f"    Queries ({layer_name})",
                               leave=False):
                    positions, scores, log_devs, output = \
                        compute_value_deviation_metrics(Q[qp], K, V, qp, HEAD_DIM)

                    multi_positions.append(positions)
                    multi_scores.append(scores)
                    multi_devs.append(log_devs)

                    # Individual dual-axis plot
                    plot_single_query(
                        positions, scores, log_devs, qp,
                        example_id, layer_name, layer_dir
                    )

                    # Individual shared-axis plot
                    plot_overlay_shared_axis(
                        positions, scores, log_devs, qp,
                        example_id, layer_name, layer_dir
                    )

                # Multi-query summary figure
                plot_summary_multi_query(
                    multi_positions, multi_scores, multi_devs,
                    query_positions, example_id, layer_name, layer_dir
                )

                # ============================================================
                # 2. Aggregated statistics across many queries
                # ============================================================
                print(f"  [{layer_name}] Computing aggregated statistics...")

                agg_positions = np.random.choice(
                    range(min_pos, max_pos + 1),
                    size=min(NUM_QUERIES_AGGREGATE, max_pos - min_pos + 1),
                    replace=False
                )

                score_stds = []
                dev_stds = []

                for qp in tqdm(agg_positions, desc=f"    Aggregating ({layer_name})",
                               leave=False):
                    positions, scores, log_devs, _ = \
                        compute_value_deviation_metrics(Q[qp], K, V, qp, HEAD_DIM)

                    score_stds.append(np.std(scores))
                    dev_stds.append(np.std(log_devs))

                median_ratio, mean_ratio = plot_aggregated_stats(
                    score_stds, dev_stds, layer_name, layer_dir,
                    [example_id]
                )

                key = f"{example_id}_{layer_name}"
                all_results[key] = {
                    'example_id': example_id,
                    'layer': layer_name,
                    'num_queries_aggregated': len(agg_positions),
                    'score_std_mean': float(np.mean(score_stds)),
                    'score_std_median': float(np.median(score_stds)),
                    'dev_std_mean': float(np.mean(dev_stds)),
                    'dev_std_median': float(np.median(dev_stds)),
                    'variability_ratio_median': float(median_ratio),
                    'variability_ratio_mean': float(mean_ratio),
                }

                print(f"    Score std (median): {np.median(score_stds):.4f}")
                print(f"    Value dev std (median): {np.median(dev_stds):.4f}")
                print(f"    Variability ratio (median): {median_ratio:.1f}x")
                print()

    # Save aggregated results
    results_path = output_dir / 'aggregated_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved aggregated results to {results_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for key, res in all_results.items():
        print(f"  {key}:")
        print(f"    Score variability (std):     {res['score_std_median']:.4f}")
        print(f"    Value dev variability (std): {res['dev_std_median']:.4f}")
        print(f"    Ratio:                       {res['variability_ratio_median']:.1f}x")
    print()
    print("Conclusion: If ratios >> 1, attention scores vary much more than")
    print("value deviations, confirming MagicPIG's insight that approximating")
    print("attention weights is the dominant source of error.")
    print("=" * 70)


if __name__ == '__main__':
    main()
