#!/usr/bin/env python3
"""
Exploration Dashboard: Attention Data Analysis + HTML Dashboard

Combined batched processing and visualization pipeline.
Processes examples in batches to avoid memory issues, aggregates statistics,
generates publication-quality plots, and builds an interactive HTML dashboard.

Combines logic from:
  - generate_dashboard_batched.py  (batch processing + aggregation)
  - generate_dashboard_visualize.py (plotting + HTML generation)

Usage:
    python3 exploration_dashboard.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from io import BytesIO
import base64
import pickle

from algorithms.base import softmax
from visualization.plot_utils import setup_style, save_figure

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = '../../data/attention_vectors_updated_long.jsonl'
OUTPUT_PATH = '../../results/attention_dashboard.html'
BATCH_DIR = Path('../../results/dashboard_batches')
NUM_EXAMPLES_TOTAL = 100
BATCH_SIZE = 5
NUM_QUERIES_PER_EXAMPLE = 1000
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42

# ============================================================================
# SETUP
# ============================================================================
np.random.seed(SEED)
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11


# ============================================================================
# HELPERS
# ============================================================================

def _softmax(x):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def fig_to_base64(fig, dpi=150):
    """Convert matplotlib figure to base64 PNG string for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"


# ============================================================================
# BATCH ANALYSIS
# ============================================================================

def analyze_batch(examples, layer_name, num_queries):
    """Analyze one batch of examples. Returns aggregated stats for the batch."""

    print(f"    Analyzing {len(examples)} examples for {layer_name}...")

    batch_data = {
        'concentration_curves': [],
        'q_norms': [],
        'k_norms': [],
        'v_norms': [],
        'key_query_dists': [],
        'key_query_cos': [],
        'top_k_masses': {10: [], 50: [], 100: [], 200: []},
        'logit_means': [],
        'logit_stds': [],
        'logit_ranges': [],
        'kv_corr_l2': [],
        'kv_corr_cos': [],
        'key_pw_dists_samples': [],
        'val_pw_dists_samples': [],
        'key_pw_cos_samples': [],
        'val_pw_cos_samples': [],
        'per_example_stats': {
            'mean_q_norm': [],
            'mean_k_norm': [],
            'mean_v_norm': [],
            'mean_conc_at_10pct': [],
            'mean_top10_mass': [],
            'mean_kv_corr_l2': [],
        },
        'num_queries': 0,
    }

    for ex_idx, example in enumerate(examples):
        Q = np.array(example[layer_name]['Q'], dtype=np.float32)
        K = np.array(example[layer_name]['K'], dtype=np.float32)
        V = np.array(example[layer_name]['V'], dtype=np.float32)
        seq_len = Q.shape[0]

        # Norms (all positions)
        q_norms_ex = np.linalg.norm(Q, axis=1)
        k_norms_ex = np.linalg.norm(K, axis=1)
        v_norms_ex = np.linalg.norm(V, axis=1)
        batch_data['q_norms'].extend(q_norms_ex.tolist())
        batch_data['k_norms'].extend(k_norms_ex.tolist())
        batch_data['v_norms'].extend(v_norms_ex.tolist())
        batch_data['per_example_stats']['mean_q_norm'].append(float(np.mean(q_norms_ex)))
        batch_data['per_example_stats']['mean_k_norm'].append(float(np.mean(k_norms_ex)))
        batch_data['per_example_stats']['mean_v_norm'].append(float(np.mean(v_norms_ex)))

        # Sample queries
        actual_num_queries = min(num_queries, seq_len - 100)
        query_positions = list(range(seq_len - actual_num_queries, seq_len))
        batch_data['num_queries'] += len(query_positions)

        example_conc_curves = []
        example_top10_masses = []
        example_kv_corrs = []

        for query_pos in query_positions:
            q = Q[query_pos]
            valid_keys = K[:query_pos + 1]
            valid_values = V[:query_pos + 1]
            n_keys = len(valid_keys)

            # Logits and weights
            logits = (q @ valid_keys.T) / np.sqrt(HEAD_DIM)
            weights = _softmax(logits)

            # Key-Query distances (subsample)
            sample_size = min(200, n_keys)
            sample_idx = np.random.choice(n_keys, sample_size, replace=False)
            kq_dists = np.linalg.norm(valid_keys[sample_idx] - q[None, :], axis=1)
            batch_data['key_query_dists'].extend(kq_dists.tolist())

            # Cosine similarity
            q_norm = q / (np.linalg.norm(q) + 1e-8)
            k_norm = valid_keys[sample_idx] / (
                np.linalg.norm(valid_keys[sample_idx], axis=1, keepdims=True) + 1e-8)
            cos = k_norm @ q_norm
            batch_data['key_query_cos'].extend(cos.tolist())

            # Concentration curve
            sorted_w = np.sort(weights)[::-1]
            cumsum = np.cumsum(sorted_w) * 100
            pct_points = np.linspace(0, 1, 101)[1:]
            curve = np.interp(pct_points * n_keys, np.arange(1, n_keys + 1), cumsum)
            batch_data['concentration_curves'].append(curve.tolist())
            example_conc_curves.append(curve)

            # Top-K masses
            for k in batch_data['top_k_masses']:
                if n_keys >= k:
                    mass = float(sorted_w[:k].sum() * 100)
                    batch_data['top_k_masses'][k].append(mass)
                    if k == 10:
                        example_top10_masses.append(mass)

            # Logit stats
            batch_data['logit_means'].append(float(logits.mean()))
            batch_data['logit_stds'].append(float(logits.std()))
            batch_data['logit_ranges'].append(float(logits.max() - logits.min()))

            # Key-Value correlation (for top-100)
            if n_keys >= 100:
                top_idx = np.argsort(logits)[-100:]
                k_top = valid_keys[top_idx]
                v_top = valid_values[top_idx]

                # L2 pairwise
                k_pw = np.linalg.norm(k_top[:, None, :] - k_top[None, :, :], axis=2)
                v_pw = np.linalg.norm(v_top[:, None, :] - v_top[None, :, :], axis=2)

                # Cosine pairwise
                k_top_norm = k_top / (np.linalg.norm(k_top, axis=1, keepdims=True) + 1e-8)
                v_top_norm = v_top / (np.linalg.norm(v_top, axis=1, keepdims=True) + 1e-8)
                k_cos_sim = k_top_norm @ k_top_norm.T
                v_cos_sim = v_top_norm @ v_top_norm.T
                k_cos_dist = 1.0 - k_cos_sim
                v_cos_dist = 1.0 - v_cos_sim

                # Upper triangle
                iu = np.triu_indices(100, k=1)
                k_dists = k_pw[iu]
                v_dists = v_pw[iu]
                k_cos_dists = k_cos_dist[iu]
                v_cos_dists = v_cos_dist[iu]

                # L2 correlation
                if k_dists.std() > 1e-6 and v_dists.std() > 1e-6:
                    corr_l2 = np.corrcoef(k_dists, v_dists)[0, 1]
                    batch_data['kv_corr_l2'].append(float(corr_l2))
                    example_kv_corrs.append(float(corr_l2))

                    if len(batch_data['key_pw_dists_samples']) < 2000:
                        idx_sample = np.random.choice(
                            len(k_dists), min(20, len(k_dists)), replace=False)
                        batch_data['key_pw_dists_samples'].extend(
                            k_dists[idx_sample].tolist())
                        batch_data['val_pw_dists_samples'].extend(
                            v_dists[idx_sample].tolist())

                # Cosine correlation
                if k_cos_dists.std() > 1e-6 and v_cos_dists.std() > 1e-6:
                    corr_cos = np.corrcoef(k_cos_dists, v_cos_dists)[0, 1]
                    batch_data['kv_corr_cos'].append(float(corr_cos))

                    if len(batch_data['key_pw_cos_samples']) < 2000:
                        idx_sample = np.random.choice(
                            len(k_cos_dists), min(20, len(k_cos_dists)), replace=False)
                        batch_data['key_pw_cos_samples'].extend(
                            k_cos_dists[idx_sample].tolist())
                        batch_data['val_pw_cos_samples'].extend(
                            v_cos_dists[idx_sample].tolist())

        # Per-example stats
        if len(example_conc_curves) > 0:
            mean_curve = np.mean(example_conc_curves, axis=0)
            batch_data['per_example_stats']['mean_conc_at_10pct'].append(
                float(mean_curve[9]))
        if len(example_top10_masses) > 0:
            batch_data['per_example_stats']['mean_top10_mass'].append(
                float(np.mean(example_top10_masses)))
        if len(example_kv_corrs) > 0:
            batch_data['per_example_stats']['mean_kv_corr_l2'].append(
                float(np.mean(example_kv_corrs)))

    print(f"      {batch_data['num_queries']} queries analyzed")
    return batch_data


# ============================================================================
# AGGREGATE BATCHES
# ============================================================================

def aggregate_batches(batch_files):
    """Combine multiple batch result files into final aggregated stats."""

    print(f"\nAggregating {len(batch_files)} batches...")

    aggregated = {
        'concentration_curves': [],
        'q_norms': [],
        'k_norms': [],
        'v_norms': [],
        'key_query_dists': [],
        'key_query_cos': [],
        'top_k_masses': {10: [], 50: [], 100: [], 200: []},
        'logit_means': [],
        'logit_stds': [],
        'logit_ranges': [],
        'kv_corr_l2': [],
        'kv_corr_cos': [],
        'key_pw_dists_samples': [],
        'val_pw_dists_samples': [],
        'key_pw_cos_samples': [],
        'val_pw_cos_samples': [],
        'per_example_stats': {
            'mean_q_norm': [],
            'mean_k_norm': [],
            'mean_v_norm': [],
            'mean_conc_at_10pct': [],
            'mean_top10_mass': [],
            'mean_kv_corr_l2': [],
        },
        'num_queries': 0,
    }

    for batch_file in batch_files:
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f)

        for key in ['q_norms', 'k_norms', 'v_norms', 'key_query_dists',
                     'key_query_cos', 'logit_means', 'logit_stds', 'logit_ranges',
                     'kv_corr_l2', 'kv_corr_cos',
                     'key_pw_dists_samples', 'val_pw_dists_samples',
                     'key_pw_cos_samples', 'val_pw_cos_samples',
                     'concentration_curves']:
            if key in batch:
                aggregated[key].extend(batch[key])

        for k in aggregated['top_k_masses']:
            aggregated['top_k_masses'][k].extend(batch['top_k_masses'][k])

        for key in aggregated['per_example_stats']:
            aggregated['per_example_stats'][key].extend(
                batch['per_example_stats'][key])

        aggregated['num_queries'] += batch['num_queries']

    # Convert concentration curves to array for percentile computation
    conc_curves_arr = np.array(aggregated['concentration_curves'])

    final = {
        'conc_x': np.linspace(0, 100, 101)[1:].tolist(),
        'conc_mean': np.mean(conc_curves_arr, axis=0).tolist(),
        'conc_p10': np.percentile(conc_curves_arr, 10, axis=0).tolist(),
        'conc_p50': np.percentile(conc_curves_arr, 50, axis=0).tolist(),
        'conc_p90': np.percentile(conc_curves_arr, 90, axis=0).tolist(),
        'conc_p99': np.percentile(conc_curves_arr, 99, axis=0).tolist(),
        'q_norms': aggregated['q_norms'],
        'k_norms': aggregated['k_norms'],
        'v_norms': aggregated['v_norms'],
        'key_query_dists': aggregated['key_query_dists'],
        'key_query_cos': aggregated['key_query_cos'],
        'top_k_masses': aggregated['top_k_masses'],
        'logit_means': aggregated['logit_means'],
        'logit_stds': aggregated['logit_stds'],
        'logit_ranges': aggregated['logit_ranges'],
        'kv_corr_l2': aggregated['kv_corr_l2'],
        'kv_corr_cos': aggregated['kv_corr_cos'],
        'key_pw_dists': aggregated['key_pw_dists_samples'],
        'val_pw_dists': aggregated['val_pw_dists_samples'],
        'key_pw_cos': aggregated['key_pw_cos_samples'],
        'val_pw_cos': aggregated['val_pw_cos_samples'],
        'per_example_stats': aggregated['per_example_stats'],
        'num_queries': aggregated['num_queries'],
    }

    print(f"  Aggregated {aggregated['num_queries']} queries")
    return final


# ============================================================================
# VISUALIZATION: CREATE ALL PLOTS
# ============================================================================

def create_all_plots(data_first, data_last):
    """Generate all publication-quality plots. Returns dict of base64 images."""

    print("Generating plots...")
    plots = {}

    # ============================================================
    # 1. CONCENTRATION CURVES (side by side)
    # ============================================================
    print("  1/7 Concentration curves...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, title in [(axes[0], data_first, 'First Layer (Layer 0)'),
                             (axes[1], data_last, 'Last Layer (Layer 31)')]:
        x = data['conc_x']
        ax.plot(x, data['conc_mean'], label='Mean', color='black',
                linewidth=3.5, alpha=0.9, zorder=5)
        ax.plot(x, data['conc_p50'], label='Median (p50)', color='#3b82f6',
                linewidth=2.5, linestyle='--', alpha=0.85, zorder=4)
        ax.plot(x, data['conc_p10'], label='p10', color='#ef4444',
                linewidth=2, linestyle=':', alpha=0.75, zorder=3)
        ax.plot(x, data['conc_p90'], label='p90', color='#22c55e',
                linewidth=2, linestyle=':', alpha=0.75, zorder=3)
        ax.plot(x, data['conc_p99'], label='p99', color='#a855f7',
                linewidth=1.8, linestyle='-.', alpha=0.7, zorder=2)
        ax.fill_between(x, data['conc_p10'], data['conc_p90'],
                         alpha=0.15, color='gray', zorder=1)
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.35, linewidth=1.5,
                label='Uniform', zorder=0)

        ax.set_xlabel('% of Keys (sorted by attention weight)',
                       fontweight='bold', fontsize=12)
        ax.set_ylabel('% of Attention Mass Captured',
                       fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', pad=12, fontsize=13)
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
        ax.legend(loc='lower right', framealpha=0.95, fontsize=10,
                  edgecolor='black')

        mean_at_10 = data['conc_mean'][9]
        ax.annotate(f'{mean_at_10:.1f}%\nat 10% keys',
                    xy=(10, mean_at_10), xytext=(22, mean_at_10 - 15),
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat',
                              alpha=0.85, edgecolor='black'),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
                    fontsize=10, fontweight='bold')

    fig.suptitle('Attention Concentration: How Many Keys Capture the Mass?',
                 fontsize=15, fontweight='bold', y=1.0)
    plt.tight_layout()
    plots['concentration'] = fig_to_base64(fig, dpi=200)

    # ============================================================
    # 2. VECTOR NORMS (3x2 grid)
    # ============================================================
    print("  2/7 Vector norms...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for row, (data, layer_title) in enumerate(
            [(data_first, 'First Layer'), (data_last, 'Last Layer')]):
        for col, (norm_key, label, color) in enumerate([
            ('q_norms', 'Query (Q)', '#f59e0b'),
            ('k_norms', 'Key (K)', '#8b5cf6'),
            ('v_norms', 'Value (V)', '#ec4899'),
        ]):
            ax = axes[row, col]
            norms = data[norm_key]
            ax.hist(norms, bins=100, color=color, alpha=0.75,
                    edgecolor='black', linewidth=0.6)
            mean_val = np.mean(norms)
            median_val = np.median(norms)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5,
                       label=f'Mean: {mean_val:.2f}', alpha=0.9)
            ax.axvline(median_val, color='blue', linestyle=':', linewidth=2.5,
                       label=f'Median: {median_val:.2f}', alpha=0.9)
            ax.set_xlabel('L2 Norm', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(f'{layer_title} -- {label}', fontweight='bold',
                         fontsize=11)
            ax.legend(framealpha=0.95, fontsize=9, edgecolor='black')
            ax.grid(True, alpha=0.2)

            std_val = np.std(norms)
            ax.text(0.98, 0.97, f'Std: {std_val:.3f}\nCV: {std_val / mean_val:.3f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                              edgecolor='gray'))

    fig.suptitle('Vector Normalization: Are Q, K, V Normalized?',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plots['norms'] = fig_to_base64(fig, dpi=200)

    # ============================================================
    # 3. KEY-QUERY DISTANCES (2x2 grid)
    # ============================================================
    print("  3/7 Key-query distances...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for row, (data, layer_title) in enumerate(
            [(data_first, 'First Layer'), (data_last, 'Last Layer')]):
        # L2 distance
        ax = axes[row, 0]
        dists = data['key_query_dists']
        ax.hist(dists, bins=100, color='#6366f1', alpha=0.75,
                edgecolor='black', linewidth=0.5)
        ax.axvline(np.mean(dists), color='red', linestyle='--', linewidth=2.5,
                   label=f"Mean: {np.mean(dists):.2f}")
        ax.axvline(np.median(dists), color='blue', linestyle=':', linewidth=2.5,
                   label=f"Median: {np.median(dists):.2f}")
        ax.set_xlabel('L2 Distance ||k - q||_2', fontweight='bold', fontsize=11)
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{layer_title} -- L2 Distance', fontweight='bold')
        ax.legend(framealpha=0.95, fontsize=9, edgecolor='black')
        ax.grid(True, alpha=0.2)

        # Cosine similarity
        ax = axes[row, 1]
        cos = data['key_query_cos']
        ax.hist(cos, bins=120, color='#22d3ee', alpha=0.75,
                edgecolor='black', linewidth=0.5)
        ax.axvline(np.mean(cos), color='red', linestyle='--', linewidth=2.5,
                   label=f"Mean: {np.mean(cos):.3f}")
        ax.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.6)
        ax.set_xlabel('Cosine Similarity (k . q)/(||k|| ||q||)',
                       fontweight='bold', fontsize=11)
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{layer_title} -- Cosine Similarity', fontweight='bold')
        ax.legend(framealpha=0.95, fontsize=9, edgecolor='black')
        ax.grid(True, alpha=0.2)

        ax.text(0.02, 0.97,
                f'Std: {np.std(cos):.3f}\np10: {np.percentile(cos, 10):.3f}\n'
                f'p90: {np.percentile(cos, 90):.3f}',
                transform=ax.transAxes, ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                          edgecolor='gray'))

    fig.suptitle('Key-Query Distance Distributions',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plots['key_query_dist'] = fig_to_base64(fig, dpi=200)

    # ============================================================
    # 4. KEY-VALUE CORRELATION (2x2: L2 and Cosine)
    # ============================================================
    print("  4/7 Key-value correlations...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for row, (data, layer_title) in enumerate(
            [(data_first, 'First Layer'), (data_last, 'Last Layer')]):
        # L2 correlation
        ax = axes[row, 0]
        if len(data['kv_corr_l2']) > 0:
            corrs = data['kv_corr_l2']
            ax.hist(corrs, bins=60, color='#f59e0b', alpha=0.75,
                    edgecolor='black', linewidth=0.6)
            ax.axvline(np.mean(corrs), color='red', linestyle='--',
                       linewidth=2.5, label=f"Mean: {np.mean(corrs):.3f}")
            ax.axvline(np.median(corrs), color='blue', linestyle=':',
                       linewidth=2.5, label=f"Median: {np.median(corrs):.3f}")
            ax.set_xlabel('Pearson r (L2 Key-Dist <-> L2 Value-Dist)',
                           fontweight='bold', fontsize=10)
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(f'{layer_title} -- L2 Correlation', fontweight='bold')
            ax.legend(framealpha=0.95, fontsize=9, edgecolor='black')
            ax.grid(True, alpha=0.2)

        # Cosine correlation
        ax = axes[row, 1]
        if len(data['kv_corr_cos']) > 0:
            corrs = data['kv_corr_cos']
            ax.hist(corrs, bins=60, color='#22d3ee', alpha=0.75,
                    edgecolor='black', linewidth=0.6)
            ax.axvline(np.mean(corrs), color='red', linestyle='--',
                       linewidth=2.5, label=f"Mean: {np.mean(corrs):.3f}")
            ax.axvline(np.median(corrs), color='blue', linestyle=':',
                       linewidth=2.5, label=f"Median: {np.median(corrs):.3f}")
            ax.set_xlabel('Pearson r (Cos Key-Dist <-> Cos Value-Dist)',
                           fontweight='bold', fontsize=10)
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(f'{layer_title} -- Cosine Correlation',
                         fontweight='bold')
            ax.legend(framealpha=0.95, fontsize=9, edgecolor='black')
            ax.grid(True, alpha=0.2)

    fig.suptitle('Key-Value Pairwise Distance Correlation (Top-100 Keys)',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plots['kv_correlation'] = fig_to_base64(fig, dpi=200)

    # ============================================================
    # 5. TOP-K MASS (side by side)
    # ============================================================
    print("  5/7 Top-K mass...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    k_vals = [10, 50, 100, 200]
    colors_topk = ['#ef4444', '#f59e0b', '#22c55e', '#3b82f6']

    for ax, data, layer_title in [
        (axes[0], data_first, 'First Layer'),
        (axes[1], data_last, 'Last Layer'),
    ]:
        positions = np.arange(len(k_vals))
        means = [np.mean(data['top_k_masses'][k]) for k in k_vals]
        stds = [np.std(data['top_k_masses'][k]) for k in k_vals]

        bars = ax.bar(positions, means, color=colors_topk, alpha=0.8,
                      edgecolor='black', linewidth=1.8, width=0.7)
        ax.errorbar(positions, means, yerr=stds, fmt='none', color='black',
                    linewidth=2.5, capsize=7, capthick=2, alpha=0.8)

        for i, (bar, mean_val) in enumerate(zip(bars, means)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + stds[i] + 3,
                    f'{mean_val:.1f}%', ha='center', va='bottom',
                    fontweight='bold', fontsize=11)

        ax.set_xticks(positions)
        ax.set_xticklabels([f'Top-{k}' for k in k_vals], fontsize=11)
        ax.set_ylabel('Attention Mass Captured (%)',
                       fontweight='bold', fontsize=12)
        ax.set_title(layer_title, fontweight='bold', pad=12, fontsize=13)
        ax.set_ylim([0, max(means) + max(stds) + 15])
        ax.grid(True, alpha=0.25, axis='y', linestyle='--')

    fig.suptitle('Top-K Mass: How Much Attention is Concentrated?',
                 fontsize=15, fontweight='bold', y=1.0)
    plt.tight_layout()
    plots['topk_mass'] = fig_to_base64(fig, dpi=200)

    # ============================================================
    # 6. PER-EXAMPLE VARIANCE
    # ============================================================
    print("  6/7 Per-example variance...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    metrics = [
        ('mean_q_norm', 'Mean Q Norm per Example', '#f59e0b'),
        ('mean_k_norm', 'Mean K Norm per Example', '#8b5cf6'),
        ('mean_v_norm', 'Mean V Norm per Example', '#ec4899'),
        ('mean_conc_at_10pct', 'Concentration at 10% Keys', '#6366f1'),
        ('mean_top10_mass', 'Top-10 Mass %', '#22c55e'),
        ('mean_kv_corr_l2', 'K-V L2 Correlation', '#f59e0b'),
    ]

    for idx, (metric_key, metric_label, color) in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        vals_first = data_first['per_example_stats'].get(metric_key, [])
        vals_last = data_last['per_example_stats'].get(metric_key, [])

        if len(vals_first) > 0:
            ax.hist(vals_first, bins=25, color='#3b82f6', alpha=0.6,
                    edgecolor='black', linewidth=0.6, label='First Layer')
        if len(vals_last) > 0:
            ax.hist(vals_last, bins=25, color='#ef4444', alpha=0.6,
                    edgecolor='black', linewidth=0.6, label='Last Layer')

        ax.set_xlabel(metric_label, fontweight='bold', fontsize=10)
        ax.set_ylabel('Frequency (# examples)', fontweight='bold')
        ax.set_title(metric_label, fontweight='bold', fontsize=11)
        ax.legend(framealpha=0.95, fontsize=9, edgecolor='black')
        ax.grid(True, alpha=0.2)

    fig.suptitle('Per-Example Variability: How Do Statistics Vary Across Examples?',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plots['per_example_variance'] = fig_to_base64(fig, dpi=200)

    # ============================================================
    # 7. SUMMARY TABLE (side by side)
    # ============================================================
    print("  7/7 Summary table...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, data, layer_title in [
        (axes[0], data_first, 'First Layer (Layer 0)'),
        (axes[1], data_last, 'Last Layer (Layer 31)'),
    ]:
        ax.axis('off')

        stats_data = [
            ['Metric', 'Mean', 'Median', 'Std', 'p10', 'p90'],
            ['Q Norm',
             f"{np.mean(data['q_norms']):.3f}",
             f"{np.median(data['q_norms']):.3f}",
             f"{np.std(data['q_norms']):.3f}",
             f"{np.percentile(data['q_norms'], 10):.3f}",
             f"{np.percentile(data['q_norms'], 90):.3f}"],
            ['K Norm',
             f"{np.mean(data['k_norms']):.3f}",
             f"{np.median(data['k_norms']):.3f}",
             f"{np.std(data['k_norms']):.3f}",
             f"{np.percentile(data['k_norms'], 10):.3f}",
             f"{np.percentile(data['k_norms'], 90):.3f}"],
            ['V Norm',
             f"{np.mean(data['v_norms']):.3f}",
             f"{np.median(data['v_norms']):.3f}",
             f"{np.std(data['v_norms']):.3f}",
             f"{np.percentile(data['v_norms'], 10):.3f}",
             f"{np.percentile(data['v_norms'], 90):.3f}"],
            ['K-Q L2 Dist',
             f"{np.mean(data['key_query_dists']):.2f}",
             f"{np.median(data['key_query_dists']):.2f}",
             f"{np.std(data['key_query_dists']):.2f}",
             f"{np.percentile(data['key_query_dists'], 10):.2f}",
             f"{np.percentile(data['key_query_dists'], 90):.2f}"],
            ['K-Q Cosine',
             f"{np.mean(data['key_query_cos']):.3f}",
             f"{np.median(data['key_query_cos']):.3f}",
             f"{np.std(data['key_query_cos']):.3f}",
             f"{np.percentile(data['key_query_cos'], 10):.3f}",
             f"{np.percentile(data['key_query_cos'], 90):.3f}"],
            ['Top-10 Mass %',
             f"{np.mean(data['top_k_masses'][10]):.1f}",
             f"{np.median(data['top_k_masses'][10]):.1f}",
             f"{np.std(data['top_k_masses'][10]):.1f}",
             f"{np.percentile(data['top_k_masses'][10], 10):.1f}",
             f"{np.percentile(data['top_k_masses'][10], 90):.1f}"],
            ['K-V Corr (L2)',
             f"{np.mean(data['kv_corr_l2']):.3f}" if len(data['kv_corr_l2']) > 0 else 'N/A',
             f"{np.median(data['kv_corr_l2']):.3f}" if len(data['kv_corr_l2']) > 0 else 'N/A',
             f"{np.std(data['kv_corr_l2']):.3f}" if len(data['kv_corr_l2']) > 0 else 'N/A',
             f"{np.percentile(data['kv_corr_l2'], 10):.3f}" if len(data['kv_corr_l2']) > 0 else 'N/A',
             f"{np.percentile(data['kv_corr_l2'], 90):.3f}" if len(data['kv_corr_l2']) > 0 else 'N/A'],
            ['K-V Corr (Cos)',
             f"{np.mean(data['kv_corr_cos']):.3f}" if len(data['kv_corr_cos']) > 0 else 'N/A',
             f"{np.median(data['kv_corr_cos']):.3f}" if len(data['kv_corr_cos']) > 0 else 'N/A',
             f"{np.std(data['kv_corr_cos']):.3f}" if len(data['kv_corr_cos']) > 0 else 'N/A',
             f"{np.percentile(data['kv_corr_cos'], 10):.3f}" if len(data['kv_corr_cos']) > 0 else 'N/A',
             f"{np.percentile(data['kv_corr_cos'], 90):.3f}" if len(data['kv_corr_cos']) > 0 else 'N/A'],
        ]

        table = ax.table(cellText=stats_data, cellLoc='center', loc='center',
                         colWidths=[0.24, 0.13, 0.13, 0.13, 0.13, 0.13])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # Style header
        for i in range(6):
            table[(0, i)].set_facecolor('#4a5568')
            table[(0, i)].set_text_props(weight='bold', color='white', size=11)

        # Alternate rows
        for i in range(1, 9):
            bg = '#f7fafc' if i % 2 == 0 else 'white'
            for j in range(6):
                table[(i, j)].set_facecolor(bg)
                table[(i, j)].set_edgecolor('#cbd5e0')
                table[(i, j)].set_linewidth(1)

        ax.set_title(layer_title, fontsize=13, fontweight='bold', pad=15)

    fig.suptitle('Comprehensive Statistics Summary',
                 fontsize=15, fontweight='bold', y=0.97)
    plt.tight_layout()
    plots['stats_table'] = fig_to_base64(fig, dpi=200)

    print("  All plots generated")
    return plots


# ============================================================================
# HTML DASHBOARD GENERATION
# ============================================================================

def generate_html_dashboard(layer_data, metadata, output_path):
    """Generate HTML dashboard from aggregated data."""

    plots = create_all_plots(layer_data['first_layer'], layer_data['last_layer'])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Attention Space Analysis Dashboard</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Arial', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    padding: 40px 20px;
  }}
  .container {{
    max-width: 1600px;
    margin: 0 auto;
    background: white;
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
  }}
  .header {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 45px 55px;
    border-radius: 20px 20px 0 0;
  }}
  .header h1 {{ font-size: 36px; margin-bottom: 12px; letter-spacing: -0.5px; }}
  .header .subtitle {{ font-size: 17px; opacity: 0.95; margin-top: 10px; line-height: 1.5; }}
  .header .meta {{ font-size: 14px; opacity: 0.85; margin-top: 14px; }}
  .content {{ padding: 50px 55px; }}
  .section {{
    margin-bottom: 70px;
    padding-bottom: 50px;
    border-bottom: 3px solid #e2e8f0;
  }}
  .section:last-child {{ border-bottom: none; }}
  .section h2 {{
    font-size: 26px;
    color: #2d3748;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 4px solid #667eea;
    display: inline-block;
  }}
  .description {{
    font-size: 14.5px;
    color: #4a5568;
    line-height: 1.8;
    margin: 18px 0 28px;
    background: #f7fafc;
    padding: 18px 24px;
    border-left: 5px solid #667eea;
    border-radius: 6px;
  }}
  .description strong {{ color: #2d3748; font-weight: 600; }}
  .description code {{
    background: #e2e8f0;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Monaco', 'Consolas', monospace;
    font-size: 13px;
  }}
  .plot-container {{
    background: white;
    border-radius: 12px;
    padding: 24px;
    margin: 24px 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
  }}
  .plot-container img {{
    width: 100%;
    height: auto;
    display: block;
    border-radius: 8px;
  }}
  .info-box {{
    background: linear-gradient(135deg, #edf2f7 0%, #e6fffa 100%);
    border-left: 5px solid #4299e1;
    padding: 20px 26px;
    margin: 24px 0;
    border-radius: 6px;
    font-size: 14px;
    color: #2d3748;
    line-height: 1.7;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }}
  .highlight {{
    background: #fef3c7;
    padding: 2px 7px;
    border-radius: 4px;
    font-weight: 600;
  }}
  footer {{
    text-align: center;
    padding: 35px;
    background: #f7fafc;
    border-radius: 0 0 20px 20px;
    font-size: 13px;
    color: #718096;
  }}
</style>
</head>
<body>

<div class="container">
  <div class="header">
    <h1>Attention Space Analysis</h1>
    <div class="subtitle">
      Comprehensive geometric analysis of attention distributions in Llama-3-8B<br>
      First Layer (Layer 0) vs. Last Layer (Layer 31) - Head 0 (KV-head 0)
    </div>
    <div class="meta">
      Generated: {metadata['timestamp']} |
      {metadata['num_examples']} examples |
      {metadata['num_queries']:,} queries per example |
      <strong>{metadata['total_queries']:,} total queries analyzed</strong> |
      Dimension: {metadata['head_dim']}
    </div>
  </div>

  <div class="content">

    <div class="info-box">
      <strong>Dataset:</strong> Query (Q), Key (K), and Value (V) vectors extracted from Llama-3-8B on
      long-context tasks from LongBench v2 (sequence length ~8192 tokens). We randomly sampled
      <span class="highlight">{metadata['num_examples']} examples</span> and analyzed the
      <span class="highlight">last {metadata['num_queries']:,} query positions</span> in each, totaling
      <span class="highlight">{metadata['total_queries']:,} queries</span>.
      All attention is causal. Vectors are 128-dimensional float32.
    </div>

    <div class="section">
      <h2>1. Attention Concentration Curve</h2>
      <div class="description">
        <strong>What:</strong> For each query, we sort attention weights in descending order and plot the cumulative
        mass captured as we include more keys. This is the definitive measure of attention concentration.<br>
        <strong>Method:</strong> Compute softmax weights for all {metadata['total_queries']:,} queries. For each,
        sort descending and compute cumulative sum. Report percentiles (p10, p50, p90, p99) across all queries.
        The diagonal represents uniform attention.<br>
        <strong>Interpretation:</strong> <strong>Curves above the diagonal</strong> = concentrated/spiky attention
        (few keys capture most mass). <strong>On the diagonal</strong> = diffuse/uniform attention.
        Higher percentiles (p90, p99) show the most concentrated queries.
      </div>
      <div class="plot-container">
        <img src="{plots['concentration']}" alt="Concentration Curve">
      </div>
    </div>

    <div class="section">
      <h2>2. Vector Normalization</h2>
      <div class="description">
        <strong>What:</strong> Distribution of L2 norms for all Query, Key, and Value vectors:
        <code>||v|| = sqrt(sum(v_i^2))</code>. Reveals whether vectors are normalized to a constant scale.<br>
        <strong>Method:</strong> Compute norms for every vector across all positions in all examples.
        Plot histograms showing the distribution, with mean and median lines.<br>
        <strong>Interpretation:</strong> <strong>Tight distribution</strong> (small std, low CV) = vectors are
        approximately normalized. <strong>Wide spread</strong> = non-normalized, varying scales across the sequence.
        CV (coefficient of variation) = std/mean quantifies relative spread.
      </div>
      <div class="plot-container">
        <img src="{plots['norms']}" alt="Vector Norms">
      </div>
    </div>

    <div class="section">
      <h2>3. Key-Query Distance Distributions</h2>
      <div class="description">
        <strong>What:</strong> Two complementary measures of separation between keys and queries in 128-D space.
        <strong>Left:</strong> L2 (Euclidean) distance <code>||k - q||_2</code>.
        <strong>Right:</strong> Cosine similarity <code>(k . q)/(||k|| ||q||)</code>, the normalized dot product.<br>
        <strong>Method:</strong> For each query, compute distances to all valid keys (causal masking).
        Subsample 200 per query for efficiency. Aggregate across all queries.<br>
        <strong>Interpretation:</strong> L2 shows absolute separation; large values = keys far from query.
        Cosine shows angular alignment: <strong>near 0</strong> = orthogonal (high-dimensional isotropy),
        <strong>near +/-1</strong> = strong (anti-)alignment. Compare first layer (token embeddings) vs.
        last layer (contextualized representations).
      </div>
      <div class="plot-container">
        <img src="{plots['key_query_dist']}" alt="Key-Query Distances">
      </div>
    </div>

    <div class="section">
      <h2>4. Key-Value Correlation: Do Similar Keys Have Similar Values?</h2>
      <div class="description">
        <strong>What:</strong> For the top-100 keys (by attention logit) in each query, we compute all pairwise
        distances among keys and among values, then measure their correlation. We test both L2 distance and
        cosine distance (1 - cosine_similarity).<br>
        <strong>Method:</strong> Per query, take top-100 keys. Compute all C(100,2)=4,950 pairwise distances
        for keys and for values. Compute Pearson correlation between the two distance vectors.
        Repeat for L2 and cosine metrics.<br>
        <strong>Interpretation:</strong> <strong>High correlation (r -> 1)</strong> means "similar keys have similar
        values" -- the value manifold is smooth and aligned with key space (easier to approximate).
        <strong>Low correlation (r -> 0)</strong> means key proximity does not predict value proximity -- complex,
        non-aligned geometry (harder to approximate).
      </div>
      <div class="plot-container">
        <img src="{plots['kv_correlation']}" alt="Key-Value Correlation">
      </div>
    </div>

    <div class="section">
      <h2>5. Top-K Mass Concentration</h2>
      <div class="description">
        <strong>What:</strong> Percentage of total attention mass captured by the top-K keys
        (K in {{10, 50, 100, 200}}), averaged across all queries. Error bars show standard deviation.<br>
        <strong>Method:</strong> For each query, sort attention weights descending and sum the top-K.
        Average across all queries. Error bars = 1 std.<br>
        <strong>Interpretation:</strong> <strong>High values</strong> (e.g., top-10 captures >40%) =
        concentrated/spiky attention. <strong>Low values</strong> (e.g., top-10 captures <20%) =
        diffuse/uniform attention. Compare layers to see how attention structure evolves.
      </div>
      <div class="plot-container">
        <img src="{plots['topk_mass']}" alt="Top-K Mass">
      </div>
    </div>

    <div class="section">
      <h2>6. Per-Example Variability</h2>
      <div class="description">
        <strong>What:</strong> How statistics vary <em>across examples</em> (not just queries within an example).
        Each histogram bar represents one example's average statistic.<br>
        <strong>Method:</strong> For each example, compute the mean of the metric across its
        {metadata['num_queries']:,} queries. Plot histogram of those {metadata['num_examples']} per-example means,
        comparing first layer (blue) vs. last layer (red).<br>
        <strong>Interpretation:</strong> <strong>Tight distribution</strong> = consistent behavior across different
        tasks/domains. <strong>Wide distribution</strong> = examples differ substantially -- attention structure is
        task-dependent or domain-dependent. Large layer differences = representations change significantly.
      </div>
      <div class="plot-container">
        <img src="{plots['per_example_variance']}" alt="Per-Example Variance">
      </div>
    </div>

    <div class="section">
      <h2>7. Summary Statistics</h2>
      <div class="description">
        Comprehensive numerical summary: mean, median, standard deviation, and percentiles (p10, p90)
        for all key metrics. Use this table for quantitative comparisons between layers.
      </div>
      <div class="plot-container">
        <img src="{plots['stats_table']}" alt="Statistics Table">
      </div>
    </div>

  </div>

  <footer>
    <p><strong>Attention Analysis Dashboard</strong> | Llama-3-8B | LongBench v2</p>
    <p style="margin-top: 10px; font-size: 12px;">
      {metadata['total_queries']:,} queries analyzed across {metadata['num_examples']} examples |
      Publication-quality matplotlib/seaborn visualizations
    </p>
  </footer>
</div>

</body>
</html>"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        f.write(html)

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"\nSaved dashboard: {output} ({size_mb:.1f} MB)")

    return output


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run batched analysis and generate HTML dashboard."""

    print("=" * 70)
    print("ATTENTION DASHBOARD - BATCHED PROCESSING + VISUALIZATION")
    print("=" * 70)
    print(f"Total examples: {NUM_EXAMPLES_TOTAL}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Queries per example: {NUM_QUERIES_PER_EXAMPLE}")
    print(f"Total batches: {int(np.ceil(NUM_EXAMPLES_TOTAL / BATCH_SIZE))}")
    print()

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    # First pass: count total examples
    print(f"Counting examples in: {DATA_PATH}")
    with open(DATA_PATH, 'r') as f:
        total_in_file = sum(1 for _ in f)
    print(f"Found {total_in_file} examples")

    # Select random indices
    selected_indices = sorted(
        np.random.choice(total_in_file, NUM_EXAMPLES_TOTAL, replace=False).tolist())
    print(f"Selected {NUM_EXAMPLES_TOTAL} random indices")

    # Second pass: load only selected examples
    print(f"Loading selected examples...")
    selected_indices_set = set(selected_indices)
    selected_examples = []
    with open(DATA_PATH, 'r') as f:
        for idx, line in enumerate(f):
            if idx in selected_indices_set:
                selected_examples.append(json.loads(line))
                if len(selected_examples) % 10 == 0:
                    print(f"  Loaded {len(selected_examples)}/{NUM_EXAMPLES_TOTAL}...",
                          end='\r')
            if len(selected_examples) >= NUM_EXAMPLES_TOTAL:
                break
    print(f"\nLoaded {len(selected_examples)} examples")

    # Process in batches
    num_batches = int(np.ceil(NUM_EXAMPLES_TOTAL / BATCH_SIZE))

    for layer_name in LAYERS:
        print(f"\n{'=' * 70}")
        print(f"Processing {layer_name}")
        print('=' * 70)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, NUM_EXAMPLES_TOTAL)
            batch_examples = selected_examples[start_idx:end_idx]

            print(f"\n  Batch {batch_idx + 1}/{num_batches} "
                  f"(examples {start_idx + 1}-{end_idx})")

            batch_result = analyze_batch(batch_examples, layer_name,
                                         NUM_QUERIES_PER_EXAMPLE)

            batch_file = BATCH_DIR / f'{layer_name}_batch_{batch_idx:03d}.pkl'
            with open(batch_file, 'wb') as f:
                pickle.dump(batch_result, f)
            print(f"    Saved: {batch_file}")

    print(f"\n{'=' * 70}")
    print("AGGREGATING BATCHES")
    print('=' * 70)

    final_data = {}
    for layer_name in LAYERS:
        print(f"\n{layer_name}:")
        batch_files = sorted(BATCH_DIR.glob(f'{layer_name}_batch_*.pkl'))
        final_data[layer_name] = aggregate_batches(batch_files)

    # Save aggregated data
    aggregated_file = BATCH_DIR / 'aggregated_data.pkl'
    with open(aggregated_file, 'wb') as f:
        pickle.dump(final_data, f)
    print(f"\nSaved aggregated data: {aggregated_file}")

    print(f"\n{'=' * 70}")
    print("GENERATING DASHBOARD")
    print('=' * 70)

    metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'num_examples': NUM_EXAMPLES_TOTAL,
        'num_queries': NUM_QUERIES_PER_EXAMPLE,
        'total_queries': (final_data['first_layer']['num_queries']
                          + final_data['last_layer']['num_queries']),
        'head_dim': HEAD_DIM,
    }

    output_path = generate_html_dashboard(final_data, metadata, OUTPUT_PATH)

    print(f"\n{'=' * 70}")
    print("COMPLETE")
    print('=' * 70)
    print(f"Dashboard: {output_path}")
    print(f"Open with: open {output_path}")


if __name__ == "__main__":
    main()
