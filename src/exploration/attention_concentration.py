#!/usr/bin/env python3
"""
Attention Concentration Analysis

Analyzes:
1. Attention weight distributions for last queries
2. Top-K concentration curves (what % of mass do top-K keys capture?)
3. Query-Key cosine similarity distribution
4. Attention entropy across query positions
5. Key/Value vector norms

Output: PNG plots and printed statistics.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import argparse
from pathlib import Path
from algorithms.base import softmax
from visualization.plot_utils import setup_style, save_figure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# CONFIG
DATA_PATH = '../../data/attention_vectors_long_bench_llama_8b.jsonl'
OUTPUT_DIR = Path('../../results/exploration')
NUM_QUERIES = 1000
NUM_PERCENTILE_POINTS = 100
LAYERS = ['first_layer', 'last_layer']


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_attention_entropy(attn_weights):
    """Compute entropy of attention distribution"""
    eps = 1e-10
    return -np.sum(attn_weights * np.log(attn_weights + eps))


def compute_concentration_curve(attn_weights, num_percentile_points=100):
    """
    Compute concentration curve: what % of mass is captured by top X% of keys?

    Returns:
        percentiles: array of key percentages (e.g., [1, 2, 3, ..., 100])
        mass_captured: array of attention mass percentages at each percentile
    """
    # Sort attention weights in descending order
    sorted_weights = np.sort(attn_weights)[::-1]

    # Compute cumulative sum (normalized to percentage)
    cumsum = np.cumsum(sorted_weights)
    total = cumsum[-1]
    cumsum_pct = (cumsum / total) * 100  # Convert to percentage

    # Create percentile points for X-axis
    num_keys = len(attn_weights)
    percentiles = np.linspace(0, 100, num_percentile_points + 1)[1:]  # 1%, 2%, ..., 100%

    # For each percentile, find how much mass is captured
    mass_at_percentiles = []
    for pct in percentiles:
        # Top X% of keys
        num_keys_at_pct = max(1, int(np.ceil(num_keys * pct / 100)))
        if num_keys_at_pct <= len(cumsum_pct):
            mass_at_percentiles.append(cumsum_pct[num_keys_at_pct - 1])
        else:
            mass_at_percentiles.append(100.0)

    return percentiles, np.array(mass_at_percentiles)


# ============================================================================
# 5-PANEL EXPLORATION FIGURE (from explore_attention_data.py)
# ============================================================================

def create_exploration_figure(Q, K, V, layer_name, layer_idx, head_idx, example_id, domain):
    """
    Create the 5-panel exploration figure:
    1. Attention weights for last queries
    2. Top-K concentration
    3. Q-K similarity distribution
    4. Attention entropy
    5. Key/Value vector norms

    Returns: (fig, insights_dict)
    """

    seq_len, head_dim = Q.shape

    print("\n  Creating 5-panel exploration figure...")
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # ==========================================
    # Plot 1: Attention Weights for Last Queries
    # ==========================================
    print("     -> Plot 1: Last query attention weights...")
    ax1 = fig.add_subplot(gs[0, :2])

    num_last_queries = 5
    last_positions = list(range(seq_len - num_last_queries, seq_len))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, num_last_queries))

    for idx, i in enumerate(last_positions):
        q = Q[i]
        k = K[:i+1]
        scores = q @ k.T / np.sqrt(head_dim)
        attn_weights = softmax(scores)
        positions = np.arange(len(attn_weights))
        ax1.plot(positions, attn_weights, label=f'Query @ pos {i}',
                color=colors[idx], alpha=0.7, linewidth=2)

    ax1.set_xlabel('Key Position', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
    ax1.set_title(f'Attention Weight Distribution for Last {num_last_queries} Queries\n(Layer {layer_idx}, Head {head_idx})',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ==========================================
    # Plot 2: Top-K Concentration
    # ==========================================
    print("     -> Plot 2: Top-K attention concentration...")
    ax2 = fig.add_subplot(gs[1, 0])

    k_values = [10, 50, 100, 200, 500]
    colors_topk = plt.cm.plasma(np.linspace(0.2, 0.9, len(k_values)))

    sample_size = 100
    sample_idx = np.linspace(10, seq_len-1, sample_size, dtype=int)

    for k_idx, k_val in enumerate(k_values):
        mass_captured = []

        for i in sample_idx:
            q = Q[i]
            k = K[:i+1]
            scores = q @ k.T / np.sqrt(head_dim)
            attn_weights = softmax(scores)

            # Get top-k weights
            if len(attn_weights) >= k_val:
                top_k_weights = np.sort(attn_weights)[-k_val:]
                mass = top_k_weights.sum()
            else:
                mass = attn_weights.sum()  # All weights if fewer than k

            mass_captured.append(mass * 100)  # Convert to percentage

        ax2.plot(sample_idx, mass_captured, label=f'Top-{k_val}',
                color=colors_topk[k_idx], linewidth=2, alpha=0.8)

    ax2.set_xlabel('Query Position', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Percentage of Total Attention Mass (%)', fontsize=10, fontweight='bold')
    ax2.set_title('Top-K Attention Concentration\n(How much mass do top-K keys capture?)',
                  fontsize=11, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    # ==========================================
    # Plot 3: Query-Key Similarity
    # ==========================================
    print("     -> Plot 3: Q-K similarity distribution...")
    ax3 = fig.add_subplot(gs[0, 2])

    sample_size = 100
    sample_idx = np.linspace(10, seq_len-1, sample_size, dtype=int)
    all_sims = []

    for i in sample_idx:
        q_norm = Q[i] / (np.linalg.norm(Q[i]) + 1e-8)
        k_valid = K[:i+1]
        k_norm = k_valid / (np.linalg.norm(k_valid, axis=1, keepdims=True) + 1e-8)
        sims = k_norm @ q_norm
        all_sims.extend(sims)

    ax3.hist(all_sims, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(all_sims), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(all_sims):.3f}')
    ax3.set_xlabel('Cosine Similarity', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Query-Key Similarity Distribution', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ==========================================
    # Plot 4: Attention Entropy
    # ==========================================
    print("     -> Plot 4: Attention entropy...")
    ax4 = fig.add_subplot(gs[1, 1])

    entropies = []
    for i in sample_idx:
        q = Q[i]
        k = K[:i+1]
        scores = q @ k.T / np.sqrt(head_dim)
        attn_weights = softmax(scores)
        entropies.append(compute_attention_entropy(attn_weights))

    ax4.plot(sample_idx, entropies, color='darkgreen', linewidth=2)
    ax4.fill_between(sample_idx, entropies, alpha=0.3, color='green')
    ax4.set_xlabel('Query Position', fontsize=10)
    ax4.set_ylabel('Entropy (nats)', fontsize=10)
    ax4.set_title('Attention Entropy\n(Higher = More Diffuse)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # ==========================================
    # Plot 5: Key/Value Vector Norms
    # ==========================================
    print("     -> Plot 5: Key/Value vector norms...")
    ax5 = fig.add_subplot(gs[1, 2])

    k_norms = np.linalg.norm(K, axis=1)
    v_norms = np.linalg.norm(V, axis=1)

    # Subsample for plotting
    plot_idx = np.linspace(0, seq_len-1, min(1000, seq_len), dtype=int)
    ax5.plot(plot_idx, k_norms[plot_idx], label='Key', color='purple', alpha=0.7, linewidth=1.5)
    ax5.plot(plot_idx, v_norms[plot_idx], label='Value', color='orange', alpha=0.7, linewidth=1.5)
    ax5.set_xlabel('Position', fontsize=10)
    ax5.set_ylabel('L2 Norm', fontsize=10)
    ax5.set_title('Key and Value Vector Norms', fontsize=11, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Compute statistics for summary
    max_weights = []
    mean_weights = []

    for i in sample_idx:
        q = Q[i]
        k = K[:i+1]
        scores = q @ k.T / np.sqrt(head_dim)
        attn_weights = softmax(scores)
        max_weights.append(attn_weights.max())
        mean_weights.append(attn_weights.mean())

    plt.tight_layout()

    insights = {
        'max_weights': max_weights,
        'mean_weights': mean_weights,
        'all_sims': all_sims,
        'entropies': entropies,
        'k_norms': k_norms,
        'v_norms': v_norms,
        'seq_len': seq_len,
    }

    return fig, insights


# ============================================================================
# CONCENTRATION PERCENTILE FIGURE (from plot_concentration_statistics.py)
# ============================================================================

def create_concentration_figure(Q, K, layer_name, layer_idx, head_idx, domain,
                                num_queries=1000, num_percentile_points=100):
    """
    Create the concentration percentile figure with mean, p10, p50, p90, p99 curves.

    Returns: (fig, stats_dict)
    """

    seq_len, head_dim = Q.shape

    # Select last N queries (or all if less than N)
    num_queries = min(num_queries, seq_len - 100)  # Need at least some keys
    query_positions = list(range(seq_len - num_queries, seq_len))

    print(f"\n  Analyzing last {len(query_positions)} queries for concentration...")
    print(f"     Positions: {query_positions[0]} to {query_positions[-1]}")

    # Compute concentration curves for all queries
    all_curves = []

    for idx, query_pos in enumerate(query_positions):
        if (idx + 1) % 100 == 0:
            print(f"     Processing query {idx+1}/{len(query_positions)}...")

        # Get query and valid keys
        q = Q[query_pos]
        k = K[:query_pos+1]

        # Compute attention weights
        scores = q @ k.T / np.sqrt(head_dim)
        attn_weights = softmax(scores)

        # Compute concentration curve
        percentiles, mass_captured = compute_concentration_curve(
            attn_weights, num_percentile_points
        )
        all_curves.append(mass_captured)

    # Convert to array: [num_queries, num_percentile_points]
    all_curves = np.array(all_curves)

    print(f"  Computed concentration curves for {len(all_curves)} queries")

    # Compute statistics across queries
    print("  Computing statistics...")
    mean_curve = np.mean(all_curves, axis=0)
    p10_curve = np.percentile(all_curves, 10, axis=0)
    p50_curve = np.percentile(all_curves, 50, axis=0)  # Median
    p90_curve = np.percentile(all_curves, 90, axis=0)
    p99_curve = np.percentile(all_curves, 99, axis=0)

    # Create plot
    print("  Creating concentration plot...")
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot statistics as lines
    ax.plot(percentiles, mean_curve, label='Mean', color='black',
            linewidth=3, linestyle='-', alpha=0.9)
    ax.plot(percentiles, p50_curve, label='Median (p50)', color='blue',
            linewidth=2.5, linestyle='--', alpha=0.8)
    ax.plot(percentiles, p10_curve, label='p10', color='red',
            linewidth=2, linestyle=':', alpha=0.7)
    ax.plot(percentiles, p90_curve, label='p90', color='green',
            linewidth=2, linestyle=':', alpha=0.7)
    ax.plot(percentiles, p99_curve, label='p99', color='purple',
            linewidth=2, linestyle='-.', alpha=0.7)

    # Fill between p10 and p90 for visual reference
    ax.fill_between(percentiles, p10_curve, p90_curve, alpha=0.2,
                    color='gray', label='p10-p90 range')

    # Diagonal reference line (perfect uniform distribution)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1,
            label='Uniform (reference)')

    # Formatting
    ax.set_xlabel('Percentage of Keys (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage of Attention Mass Captured (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Attention Concentration Statistics\n'
                 f'Layer {layer_idx}, Head {head_idx} | {len(query_positions)} queries (positions {query_positions[0]}-{query_positions[-1]})\n'
                 f'Domain: {domain[:60]}...',
                 fontsize=15, fontweight='bold', pad=20)

    ax.set_xlim([0, 100])
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)

    # Add reference markers
    for x_val in [10, 25, 50, 75]:
        ax.axvline(x=x_val, color='gray', alpha=0.2, linewidth=0.8, linestyle=':')

    for y_val in [25, 50, 75]:
        ax.axhline(y=y_val, color='gray', alpha=0.2, linewidth=0.8, linestyle=':')

    # Add text annotations for key points
    # At 10% of keys
    mean_at_10 = mean_curve[int(10 * num_percentile_points / 100) - 1]
    ax.annotate(f'Mean at 10% keys:\n{mean_at_10:.1f}% mass',
                xy=(10, mean_at_10), xytext=(15, mean_at_10 - 10),
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=1))

    # At 50% of keys
    mean_at_50 = mean_curve[int(50 * num_percentile_points / 100) - 1]
    ax.annotate(f'Mean at 50% keys:\n{mean_at_50:.1f}% mass',
                xy=(50, mean_at_50), xytext=(55, mean_at_50 + 8),
                fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=1))

    plt.tight_layout()

    stats = {
        'percentiles': percentiles,
        'mean': mean_curve,
        'p10': p10_curve,
        'p50': p50_curve,
        'p90': p90_curve,
        'p99': p99_curve,
        'num_queries_analyzed': len(query_positions),
        'num_percentile_points': num_percentile_points,
    }

    return fig, stats


# ============================================================================
# PRINT FUNCTIONS
# ============================================================================

def print_exploration_insights(insights, layer_name):
    """Print insights from the exploration figure."""
    print("\n" + "="*70)
    print(f"KEY INSIGHTS ({layer_name})")
    print("="*70)
    print(f"\n1. Concentration: {'FOCUSED' if np.mean(insights['max_weights']) > 0.1 else 'DIFFUSE'}")
    print(f"   - Avg max weight: {np.mean(insights['max_weights']):.3f}")
    print(f"\n2. Q-K Similarity: {np.mean(insights['all_sims']):.4f}")
    print(f"\n3. Entropy: {np.mean(insights['entropies']):.3f} nats")
    print(f"   - Normalized: {np.mean(insights['entropies'])/np.log(insights['seq_len']/2):.2%} of max")
    print(f"\n4. Vector Norms:")
    print(f"   - K norm (mean): {np.mean(insights['k_norms']):.3f}")
    print(f"   - V norm (mean): {np.mean(insights['v_norms']):.3f}")
    print("="*70)


def print_concentration_stats(stats, layer_name, num_percentile_points):
    """Print detailed concentration statistics."""

    mean_curve = stats['mean']
    p10_curve = stats['p10']
    p50_curve = stats['p50']
    p90_curve = stats['p90']
    p99_curve = stats['p99']
    percentiles = stats['percentiles']

    print("\n" + "="*70)
    print(f"CONCENTRATION STATISTICS ({layer_name})")
    print("="*70)

    # At different percentages of keys
    for key_pct in [1, 5, 10, 25, 50, 75, 100]:
        idx = int(key_pct * num_percentile_points / 100) - 1
        if idx < 0:
            idx = 0

        print(f"\nTop {key_pct}% of keys:")
        print(f"  Mean mass captured:   {mean_curve[idx]:.2f}%")
        print(f"  Median (p50):         {p50_curve[idx]:.2f}%")
        print(f"  Range (p10-p90):      {p10_curve[idx]:.2f}% - {p90_curve[idx]:.2f}%")
        print(f"  p99:                  {p99_curve[idx]:.2f}%")

    # Efficiency metric: how many keys needed to capture X% of mass?
    print("\n" + "="*70)
    print("EFFICIENCY METRICS")
    print("="*70)

    for target_mass in [50, 80, 90, 95]:
        # Find percentage of keys needed (on average)
        idx_mean = np.searchsorted(mean_curve, target_mass)
        if idx_mean < len(percentiles):
            keys_needed_mean = percentiles[idx_mean]
            print(f"\nTo capture {target_mass}% of mass (on average):")
            print(f"  Need ~{keys_needed_mean:.1f}% of keys")

            # For median query
            idx_median = np.searchsorted(p50_curve, target_mass)
            if idx_median < len(percentiles):
                keys_needed_median = percentiles[idx_median]
                print(f"  Median query needs ~{keys_needed_median:.1f}% of keys")

    print("\n" + "="*70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Attention Concentration Analysis: 5-panel exploration + concentration percentile curves'
    )
    parser.add_argument('--file', type=str,
                       default=DATA_PATH,
                       help='Path to attention vectors JSONL file')
    parser.add_argument('--layer', type=str, default=None,
                       choices=['first_layer', 'last_layer'],
                       help='Which layer to analyze (default: both)')
    parser.add_argument('--num-queries', type=int, default=NUM_QUERIES,
                       help='Number of last queries to analyze for concentration (default: 1000)')

    args = parser.parse_args()

    # Resolve data path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, args.file) if not os.path.isabs(args.file) else args.file
    output_dir = Path(os.path.join(script_dir, str(OUTPUT_DIR))) if not OUTPUT_DIR.is_absolute() else OUTPUT_DIR

    layers_to_analyze = [args.layer] if args.layer else LAYERS

    setup_style()

    print("="*70)
    print("ATTENTION CONCENTRATION ANALYSIS")
    print("="*70)

    # Load first example from data file
    print(f"\nLoading: {data_path}")
    try:
        with open(data_path, 'r') as f:
            example = json.loads(f.readline())
        print("JSON loaded successfully")
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)

    print(f"  Example: {example['example_id']}")
    print(f"  Domain: {example['domain']}")

    for layer_name in layers_to_analyze:
        print(f"\n{'='*70}")
        print(f"ANALYZING: {layer_name}")
        print(f"{'='*70}")

        # Extract arrays
        print("  Converting to numpy arrays...")
        Q = np.array(example[layer_name]['Q'], dtype=np.float32)
        K = np.array(example[layer_name]['K'], dtype=np.float32)
        V = np.array(example[layer_name]['V'], dtype=np.float32)

        seq_len, head_dim = Q.shape
        layer_idx = example[layer_name]['layer_idx']
        head_idx = example[layer_name].get('head_idx',
                   example[layer_name].get('meta', {}).get('kv_head_idx', 0))

        print(f"  Layer: {layer_idx}, Head: {head_idx}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Head dim: {head_dim}")

        # --- 5-panel exploration figure ---
        fig_explore, insights = create_exploration_figure(
            Q, K, V, layer_name, layer_idx, head_idx,
            example['example_id'], example['domain']
        )

        output_path_explore = output_dir / f'attention_data_exploration_{layer_name}.png'
        save_figure(fig_explore, output_path_explore)

        print_exploration_insights(insights, layer_name)

        # --- Concentration percentile figure ---
        fig_conc, conc_stats = create_concentration_figure(
            Q, K, layer_name, layer_idx, head_idx,
            example['domain'],
            num_queries=args.num_queries,
            num_percentile_points=NUM_PERCENTILE_POINTS
        )

        output_path_conc = output_dir / f'concentration_statistics_{layer_name}.png'
        save_figure(fig_conc, output_path_conc)

        print_concentration_stats(conc_stats, layer_name, NUM_PERCENTILE_POINTS)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
