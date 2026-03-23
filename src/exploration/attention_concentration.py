#!/usr/bin/env python3
"""
Attention Concentration Analysis

Analyzes:
1. Attention weight distributions for last queries
2. Top-K concentration curves (what % of mass do top-K keys capture?)
3. Query-Key cosine similarity distribution
4. Query-Key dot product distribution (k · q)
5. Attention entropy across query positions
6. Key/Value vector norms

Output: PNG plots and printed statistics.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import argparse
import random
from typing import Optional
from pathlib import Path
from algorithms.base import softmax
from visualization.plot_utils import setup_style, save_figure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# CONFIG
DATA_PATH = '../../data/attention_vectors_infinitebench_math_calc_128k_with_rope.json'
OUTPUT_DIR = Path('../../results/exploration')
NUM_QUERIES = 1000
NUM_PERCENTILE_POINTS = 100
LAYERS = ['first_layer', 'last_layer']

def _scale_rows_to_target_norm(
    X: np.ndarray,
    target_norm: float,
    eps: float = 1e-8,
    preserve_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Rescale each row x_i by target_norm / (||x_i|| + eps).

    If preserve_indices is set, those rows are left unchanged (scale=1).
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    scales = target_norm / (norms + eps)
    if preserve_indices is not None and len(preserve_indices) > 0:
        scales[preserve_indices] = 1.0
    return X * scales


def _median_finite_positive(x: np.ndarray, eps: float = 1e-8) -> float:
    x = np.asarray(x)
    mask = np.isfinite(x) & (x > eps)
    if not np.any(mask):
        return float('nan')
    return float(np.median(x[mask]))


def normalize_qk_to_median_norm(
    Q: np.ndarray,
    K: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """
    Rescale each query/key vector to match the median L2 norm (per matrix).

    For each row vector x_i, we apply: x_i <- x_i * (median_norm / (||x_i|| + eps)).
    """
    q_norms = np.linalg.norm(Q, axis=1)
    k_norms = np.linalg.norm(K, axis=1)

    q_target = _median_finite_positive(q_norms, eps=eps)
    k_target = _median_finite_positive(k_norms, eps=eps)

    if np.isfinite(q_target):
        Q = _scale_rows_to_target_norm(Q, q_target, eps=eps)
    if np.isfinite(k_target):
        preserved_k_idx = np.array([0], dtype=int) if len(K) > 0 else np.array([], dtype=int)
        K = _scale_rows_to_target_norm(K, k_target, eps=eps, preserve_indices=preserved_k_idx)
    else:
        preserved_k_idx = np.array([], dtype=int)

    return Q, K, q_target, k_target, preserved_k_idx


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

def create_exploration_figure(
    Q,
    K,
    V,
    layer_name,
    layer_idx,
    head_idx,
    example_id,
    domain,
    kv_norm_desc: str,
    title_extra: str = "",
):
    """
    Create the exploration figure:
    1. Attention weights for last queries
    1a. Unnormalized attention scores exp(q·k/sqrt(d))
    1b. Query distance from mean(Q)
    2. Top-K concentration
    3. Q-K similarity distribution
    4. Q-K dot product distribution (k · q)
    4b. Q-Q similarity distribution
    5. Attention entropy
    6. Key/Value vector norms

    Returns: (fig, insights_dict)
    """

    seq_len, head_dim = Q.shape

    print("\n  Creating 9-panel exploration figure...")
    fig = plt.figure(figsize=(22, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(
        f"{example_id} | {domain}\n{layer_name} (Layer {layer_idx}, Head {head_idx}) | {kv_norm_desc}"
        + (f"\n{title_extra}" if title_extra else ""),
        fontsize=13,
        fontweight='bold',
        y=0.99,
    )

    # ==========================================
    # Plot 1: Attention Weights for Last Queries
    # ==========================================
    print("     -> Plot 1: Attention weights at 90% query position...")
    ax1 = fig.add_subplot(gs[0, :2])

    finite_q = np.all(np.isfinite(Q), axis=1)
    if np.any(finite_q):
        mean_q = Q[finite_q].mean(axis=0)
        q_dists = np.linalg.norm(Q[finite_q] - mean_q[np.newaxis, :], axis=1)
        q_dists = q_dists[np.isfinite(q_dists)]
    else:
        mean_q = np.full(head_dim, np.nan, dtype=Q.dtype)
        q_dists = np.array([], dtype=np.float64)

    query_pos = int(0.9 * max(seq_len - 1, 0))
    color = plt.cm.viridis(0.7)
    q = Q[query_pos]
    k = K[:query_pos + 1]
    scores = q @ k.T / np.sqrt(head_dim)
    attn_weights = softmax(scores)
    positions = np.arange(len(attn_weights))
    ax1.plot(positions, attn_weights, label=f'Query @ pos {query_pos} (90%)',
            color=color, alpha=0.8, linewidth=2)

    ax1.set_xlabel('Key Position', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    positive_weights = attn_weights[attn_weights > 0]
    if positive_weights.size > 0:
        ymin = max(float(np.min(positive_weights)) * 0.5, 1e-12)
        ymax = max(float(np.max(positive_weights)) * 1.1, ymin * 10)
        ax1.set_ylim(ymin, ymax)
    ax1.set_title(f'Attention Weight Distribution for Query @ pos {query_pos}   (Layer {layer_idx}, Head {head_idx})',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ==========================================
    # Plot 1a: Mean-query base vs full score
    # ==========================================
    print("     -> Plot 1a: Mean-query base vs full score...")
    ax1a = fig.add_subplot(gs[0, 2:])

    sqrt_d = np.sqrt(head_dim)
    mean_logits = (k @ mean_q) / sqrt_d
    delta_logits = (k @ (q - mean_q)) / sqrt_d
    base_scores = np.exp(mean_logits)
    full_scores = np.exp(mean_logits + delta_logits)

    base_scores = np.where(np.isfinite(base_scores), base_scores, np.nan)
    full_scores = np.where(np.isfinite(full_scores), full_scores, np.nan)
    positions = np.arange(len(base_scores))

    min_scores = np.minimum(base_scores, full_scores)
    max_scores = np.maximum(base_scores, full_scores)
    up_mask = np.isfinite(base_scores) & np.isfinite(full_scores) & (full_scores >= base_scores)

    ax1a.plot(
        positions,
        min_scores,
        label=r'$\min(M_k, M_kE_k)$',
        color='dimgray',
        alpha=0.95,
        linewidth=2,
    )
    ax1a.fill_between(
        positions,
        min_scores,
        max_scores,
        where=up_mask,
        interpolate=True,
        color='tab:green',
        alpha=0.45,
        label=r'$\max-\min$ where $M_kE_k > M_k$',
    )
    ax1a.set_xlabel('Key Position', fontsize=10, fontweight='bold')
    ax1a.set_ylabel('Unnormalized Score', fontsize=10, fontweight='bold')
    ax1a.set_title(
        f'Mean-Query Base vs Full Score for Query @ pos {query_pos}\n'
        r'$M_k=\exp(k\cdot\bar{q}/\sqrt{d}),\ \ E_k=\exp(k\cdot(q-\bar{q})/\sqrt{d})$',
        fontsize=11,
        fontweight='bold'
    )
    # Zoom y-axis to a robust range for readability.
    finite_scores = np.concatenate([base_scores[np.isfinite(base_scores)], full_scores[np.isfinite(full_scores)]])
    if finite_scores.size > 0:
        y_top = float(np.percentile(finite_scores, 90))
        if np.isfinite(y_top) and y_top > 0:
            ax1a.set_ylim(0, y_top)
    ax1a.legend(loc='upper left', fontsize=7)
    ax1a.grid(True, alpha=0.3)

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
    # Plot 5: Attention Entropy
    # ==========================================
    print("     -> Plot 5: Attention entropy...")
    ax5 = fig.add_subplot(gs[1, 1])

    entropies = []
    for i in sample_idx:
        q = Q[i]
        k = K[:i+1]
        scores = q @ k.T / np.sqrt(head_dim)
        attn_weights = softmax(scores)
        entropies.append(compute_attention_entropy(attn_weights))

    ax5.plot(sample_idx, entropies, color='darkgreen', linewidth=2)
    ax5.fill_between(sample_idx, entropies, alpha=0.3, color='green')

    # Reference: entropy if attention were uniform over a fraction of keys.
    # For uniform over m items: H = log(m).
    ref_fracs = [0.50, 0.10, 0.01]
    ref_colors = ['#444444', '#5555aa', '#aa5555']
    for frac, c in zip(ref_fracs, ref_colors):
        m_vals = np.maximum(1, np.ceil((sample_idx + 1) * frac)).astype(int)
        h_ref = np.log(m_vals.astype(np.float64))
        ax5.plot(
            sample_idx,
            h_ref,
            linestyle='--',
            linewidth=1.8,
            color=c,
            alpha=0.85,
            label=f'Uniform over {int(frac*100)}% keys'
        )
    # Reference: uniform over a fixed 10 entries (when available).
    # H = log(m) for a uniform distribution over m items.
    fixed_m = 10
    h_fixed = np.log(float(fixed_m))
    ax5.plot(
        sample_idx,
        np.full_like(sample_idx, h_fixed, dtype=np.float64),
        linestyle='-.',
        linewidth=1.8,
        color='#aa33aa',
        alpha=0.9,
        label=f'Uniform over {fixed_m} entries'
    )
    ax5.set_xlabel('Query Position', fontsize=10)
    ax5.set_ylabel('Entropy (nats)', fontsize=10)
    ax5.set_title('Attention Entropy\n(Higher = More Diffuse)', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8, framealpha=0.9, loc='best')

    # ==========================================
    # Plot 3: Query-Key Similarity
    # ==========================================
    print("     -> Plot 3: Q-K similarity distribution...")
    ax3 = fig.add_subplot(gs[1, 2])

    sample_size = 100
    sample_idx = np.linspace(10, seq_len-1, sample_size, dtype=int)
    all_sims = []
    key0_sims = []

    for i in sample_idx:
        q_norm = Q[i] / (np.linalg.norm(Q[i]) + 1e-8)
        k_valid = K[:i+1]
        k_norm = k_valid / (np.linalg.norm(k_valid, axis=1, keepdims=True) + 1e-8)
        sims = k_norm @ q_norm
        all_sims.extend(sims)
        if len(K) > 0:
            k0 = K[0]
            k0n = k0 / (np.linalg.norm(k0) + 1e-8)
            s0 = float(k0n @ q_norm)
            if np.isfinite(s0):
                key0_sims.append(s0)

    all_sims_arr = np.asarray(all_sims, dtype=np.float64)
    key0_sims_arr = np.asarray(key0_sims, dtype=np.float64)

    bins = np.histogram_bin_edges(all_sims_arr, bins=50)
    ax3.hist(all_sims_arr, bins=bins, density=True, color='steelblue', alpha=0.45, edgecolor='black',
             label='All keys (density)')
    if key0_sims_arr.size > 0:
        ax3.hist(key0_sims_arr, bins=bins, density=True, histtype='step', linewidth=2.5,
                 color='tab:orange', label='Key 0 only (density)')
        ax3.axvline(np.mean(key0_sims_arr), color='tab:orange', linestyle='--', linewidth=1.8,
                    alpha=0.9, label=f'Mean k0: {np.mean(key0_sims_arr):.3f}')

    ax3.axvline(np.mean(all_sims_arr), color='red', linestyle='--', linewidth=2,
                label=f'Mean all: {np.mean(all_sims_arr):.3f}')
    ax3.set_xlabel('Cosine Similarity', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.set_title('Query-Key Similarity Distribution', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ==========================================
    # Plot 4: Query-Key Dot Products (k · q)
    # ==========================================
    print("     -> Plot 4: Q-K dot product distribution...")
    ax4 = fig.add_subplot(gs[1, 3])

    all_dots = []
    key0_dots = []
    for i in sample_idx:
        q = Q[i]
        k_valid = K[:i+1]
        dots = (k_valid @ q) / np.sqrt(head_dim)  # scaled dot products (like attention logits)
        dots = dots[np.isfinite(dots)]
        all_dots.extend(dots.tolist())
        if len(K) > 0:
            d0 = float((K[0] @ q) / np.sqrt(head_dim))
            if np.isfinite(d0):
                key0_dots.append(d0)

    if len(all_dots) > 0:
        all_dots_arr = np.asarray(all_dots, dtype=np.float64)
        key0_arr = np.asarray(key0_dots, dtype=np.float64)

        bins = np.histogram_bin_edges(all_dots_arr, bins=50)

        # Plot as densities so key-0 isn't visually drowned out by the huge number
        # of samples in the "all keys" distribution.
        ax4.hist(
            all_dots_arr,
            bins=bins,
            density=True,
            color='slategray',
            alpha=0.45,
            edgecolor='black',
            label='All keys (density)',
        )
        if key0_arr.size > 0:
            ax4.hist(
                key0_arr,
                bins=bins,
                density=True,
                histtype='step',
                linewidth=2.5,
                color='tab:orange',
                label='Key 0 only (density)',
            )
            ax4.axvline(np.mean(key0_arr), color='tab:orange', linestyle='--', linewidth=1.8,
                        alpha=0.9, label=f"Mean k0: {np.mean(key0_arr):.2f}")

        ax4.axvline(np.mean(all_dots_arr), color='red', linestyle='--', linewidth=2,
                    label=f"Mean all: {np.mean(all_dots_arr):.2f}")
        ax4.set_xlabel('Scaled Dot Product (k · q / √d)', fontsize=10)
        ax4.set_ylabel('Density', fontsize=10)
        ax4.set_title('Query-Key Scaled Dot Product Distribution', fontsize=11, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No finite dot products', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_axis_off()

    # ==========================================
    # Plot 4b: Query-Query Similarity
    # ==========================================
    print("     -> Plot 4b: Q-Q similarity distribution...")
    ax4b = fig.add_subplot(gs[2, 0])

    q_sample = Q[sample_idx]
    q_sample_norm = q_sample / (np.linalg.norm(q_sample, axis=1, keepdims=True) + 1e-8)
    qq_sims = q_sample_norm @ q_sample_norm.T
    qq_mask = np.triu(np.ones_like(qq_sims, dtype=bool), k=1)
    qq_vals = qq_sims[qq_mask]
    qq_vals = qq_vals[np.isfinite(qq_vals)]

    if qq_vals.size > 0:
        bins = np.histogram_bin_edges(qq_vals, bins=50)
        ax4b.hist(qq_vals, bins=bins, density=True, color='mediumpurple', alpha=0.5, edgecolor='black')
        ax4b.axvline(np.mean(qq_vals), color='black', linestyle='--', linewidth=1.8,
                     label=f"Mean: {np.mean(qq_vals):.3f}")
        ax4b.set_xlabel('Cosine Similarity', fontsize=10)
        ax4b.set_ylabel('Density', fontsize=10)
        ax4b.set_title('Query-Query Similarity Distribution', fontsize=11, fontweight='bold')
        ax4b.legend(fontsize=8, framealpha=0.9, loc='best')
        ax4b.grid(True, alpha=0.3)
    else:
        ax4b.text(0.5, 0.5, 'No finite Q-Q similarities', ha='center', va='center', transform=ax4b.transAxes)
        ax4b.set_axis_off()

    # ==========================================
    # Plot 1b: Distribution of ||q - mean(Q)||_2
    # ==========================================
    print("     -> Plot 1b: ||q - mean(Q)|| distribution...")
    ax1b = fig.add_subplot(gs[2, 1])

    # Cosine of angle between mean(Q) and key #0 for caption
    cos_mean_q_k0 = np.nan
    if len(K) > 0 and np.all(np.isfinite(mean_q)) and np.all(np.isfinite(K[0])):
        n_mq = np.linalg.norm(mean_q)
        n_k0 = np.linalg.norm(K[0])
        if n_mq > 1e-12 and n_k0 > 1e-12:
            cos_mean_q_k0 = float(np.dot(mean_q, K[0]) / (n_mq * n_k0))

    if q_dists.size > 0:
        bins = np.histogram_bin_edges(q_dists, bins=50)
        ax1b.hist(q_dists, bins=bins, density=True, color='teal', alpha=0.5, edgecolor='black')
        ax1b.axvline(np.mean(q_dists), color='black', linestyle='--', linewidth=1.8,
                     label=f"Mean: {np.mean(q_dists):.2f}")
        ax1b.set_xlabel(r'$||q - \mathrm{mean}(Q)||_2$', fontsize=10)
        ax1b.set_ylabel('Density', fontsize=10)
        title_1b = r'Distribution of $||q - \mathrm{mean}(Q)||_2$'
        if not np.isnan(cos_mean_q_k0):
            title_1b += f'\ncos(mean(Q), K[0]) = {cos_mean_q_k0:.4f}'
        ax1b.set_title(title_1b, fontsize=11, fontweight='bold')
        ax1b.legend(fontsize=8, framealpha=0.9, loc='best')
        ax1b.grid(True, alpha=0.3)
    else:
        ax1b.text(0.5, 0.5, 'No finite queries', ha='center', va='center', transform=ax1b.transAxes)
        ax1b.set_axis_off()

    # ==========================================
    # Plot 6: Key/Value Vector Norms
    # ==========================================
    print("     -> Plot 6: Key/Value vector norms...")
    ax6 = fig.add_subplot(gs[2, 2:])

    k_norms = np.linalg.norm(K, axis=1)
    v_norms = np.linalg.norm(V, axis=1)

    # Subsample for plotting
    plot_idx = np.linspace(0, seq_len-1, min(1000, seq_len), dtype=int)
    ax6.plot(plot_idx, k_norms[plot_idx], label='Key', color='purple', alpha=0.7, linewidth=1.5)
    ax6.plot(plot_idx, v_norms[plot_idx], label='Value', color='orange', alpha=0.7, linewidth=1.5)
    ax6.set_xlabel('Position', fontsize=10)
    ax6.set_ylabel('L2 Norm', fontsize=10)
    ax6.set_title('Key and Value Vector Norms', fontsize=11, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

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

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    insights = {
        'max_weights': max_weights,
        'mean_weights': mean_weights,
        'all_sims': all_sims,
        'all_dots': all_dots,
        'qq_sims': qq_vals,
        'q_mean_dist': q_dists,
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
                                num_queries=1000, num_percentile_points=100, kv_norm_desc: str = ''):
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
                 f'Domain: {domain[:60]}...\n'
                 f'{kv_norm_desc}',
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
        description='Attention Concentration Analysis: 9-panel exploration + concentration percentile curves'
    )
    parser.add_argument('--file', type=str,
                       default=DATA_PATH,
                       help='Path to attention vectors JSONL file')
    parser.add_argument('--example-idx', type=int, default=None,
                        help='1-indexed example number to analyze from the JSONL (default: random line).')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed used only when --example-idx is not provided.')
    parser.add_argument('--layer', type=str, default=None,
                       choices=['first_layer', 'last_layer'],
                       help='Which layer to analyze (default: both)')
    parser.add_argument('--num-queries', type=int, default=NUM_QUERIES,
                       help='Number of last queries to analyze for concentration (default: 1000)')
    parser.add_argument('-qk', '--normalize-qk', action='store_true',
                        help='Rescale each query/key vector to the median L2 norm (per-layer). Values are unchanged.')
    # Backward-compatible alias (deprecated)
    parser.add_argument('--normalize-kv', dest='normalize_qk', action='store_true',
                        help=argparse.SUPPRESS)

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

    # Load selected example from data file (JSONL)
    print(f"\nLoading: {data_path}")
    try:
        selected_idx = args.example_idx
        if selected_idx is not None:
            if selected_idx < 1:
                raise ValueError(f"--example-idx must be >= 1 (got {selected_idx})")
            with open(data_path, 'r') as f:
                example_line = None
                for i, line in enumerate(f, start=1):
                    if i == selected_idx:
                        example_line = line
                        break
            if example_line is None:
                raise ValueError(f"File has fewer than {selected_idx} lines/examples")
            example = json.loads(example_line)
            print(f"JSON loaded successfully (example #{selected_idx})")
        else:
            rng = random.Random(args.seed)
            example_line = None
            selected_idx = None
            with open(data_path, 'r') as f:
                for i, line in enumerate(f, start=1):
                    # Reservoir sampling: each line is equally likely.
                    if rng.randrange(i) == 0:
                        example_line = line
                        selected_idx = i
            if example_line is None or selected_idx is None:
                raise ValueError("Input file is empty (no examples found).")
            example = json.loads(example_line)
            seed_note = f", seed={args.seed}" if args.seed is not None else ""
            print(f"JSON loaded successfully (random example #{selected_idx}{seed_note})")
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)

    print(f"  Example: {example['example_id']}")
    print(f"  Domain: {example['domain']}")
    rope_applied = bool(example.get('rope_applied', False))
    rope_cfg = example.get('rope_config', {}) if isinstance(example.get('rope_config', {}), dict) else {}
    if rope_applied:
        theta = rope_cfg.get('rope_theta', None)
        method = rope_cfg.get('method', 'unknown')
        if theta is not None:
            rope_desc = f"RoPE: applied (theta={theta:g}, {method})"
        else:
            rope_desc = f"RoPE: applied ({method})"
    else:
        rope_desc = "RoPE: not indicated in input JSON"
    print(f"  {rope_desc}")

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

        if args.normalize_qk:
            Q, K, q_target, k_target, preserved_k_idx = normalize_qk_to_median_norm(Q, K)
            kv_norm_desc = (
                f"{rope_desc} | "
                "QK normalized: each vector rescaled to median L2 norm "
                f"(Q→{q_target:.3f}, K→{k_target:.3f}, preserve K[0]); V raw"
            )
            file_suffix = "_qknorm_median"
            if preserved_k_idx.size > 0:
                shown = preserved_k_idx[:10].tolist()
                more = "" if preserved_k_idx.size <= 10 else f" (+{preserved_k_idx.size - 10} more)"
                print(f"  Left K unchanged for {preserved_k_idx.size} positions: {shown}{more}")
            else:
                print("  Left K unchanged for 0 positions")
        else:
            kv_norm_desc = f"{rope_desc} | QK raw (no normalization); V raw"
            file_suffix = ""

        is_random_example = args.example_idx is None
        if is_random_example:
            example_suffix = "_rand"
            example_title_tag = f"Example #{selected_idx} (random)"
        else:
            example_suffix = f"_ex{selected_idx}" if selected_idx != 1 else ""
            example_title_tag = f"Example #{selected_idx}"

        finite_q = np.all(np.isfinite(Q), axis=1)
        if np.any(finite_q):
            mean_q = Q[finite_q].mean(axis=0)
            mean_q_norm = float(np.linalg.norm(mean_q))
            mean_q_desc = f"||mean(Q)||={mean_q_norm:.3f} (over {int(finite_q.sum())} queries)"
        else:
            mean_q_desc = "||mean(Q)||=nan (no finite queries)"

        title_extra = f"{example_title_tag} | {mean_q_desc}"

        # --- 5-panel exploration figure ---
        fig_explore, insights = create_exploration_figure(
            Q, K, V, layer_name, layer_idx, head_idx,
            example['example_id'], example['domain'],
            kv_norm_desc=kv_norm_desc,
            title_extra=title_extra,
        )

        output_path_explore = output_dir / f'attention_data_exploration_{layer_name}{file_suffix}{example_suffix}.png'
        save_figure(fig_explore, output_path_explore)

        print_exploration_insights(insights, layer_name)

        # --- Concentration percentile figure ---
        fig_conc, conc_stats = create_concentration_figure(
            Q, K, layer_name, layer_idx, head_idx,
            example['domain'],
            num_queries=args.num_queries,
            num_percentile_points=NUM_PERCENTILE_POINTS,
            kv_norm_desc=f"{title_extra}\n{kv_norm_desc}",
        )

        output_path_conc = output_dir / f'concentration_statistics_{layer_name}{file_suffix}{example_suffix}.png'
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
