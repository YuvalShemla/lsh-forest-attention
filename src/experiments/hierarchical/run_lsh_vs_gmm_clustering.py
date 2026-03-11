#!/usr/bin/env python3
"""
Hierarchical LSH v2 — L=1 Depth Sweep

Clean comparison of hierarchical tree-aggregation (count-weighted softmax over
LCP-depth groups) against simple baselines. Uses L=1 (single tree) averaged
across 50 random SimHash seeds per example.

Methods:
  - Hierarchical-AvgKey: K+1 mutually exclusive groups by LCP depth,
    count-weighted softmax over group representatives
  - TopK Attention @100: softmax over top-100 logits, weighted sum
  - Uniform @100: softmax over random-100 logits, weighted sum
  - Oracle @100: sample 100 by true attention weights, mean of values
  - TopKValues-avg @B: mean of top-B values by logit (B=10,50,100,500,1000)
  - mean(V): mean of all values

Output: results/hierarchical_lsh_v2/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from algorithms.base import softmax as stable_softmax
from algorithms.hierarchical_lsh import hierarchical_lsh_attention
from algorithms.gmm_attention import fit_gmm, gmm_attention

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = '../../../data/attention_vectors_long_bench_llama_8b.jsonl'
OUTPUT_DIR = Path('../../../results/hierarchical_lsh_v2')
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
NUM_EXAMPLES = 50         # Production: 50
NUM_QUERIES = 100          # Production: 100
NUM_SEEDS = 20             # Production: 20
MAX_DEPTH = 10
L = 1  # Number of trees (single tree experiment)

K_VALUES = [1, 5, 10]  # Hierarchical depth values to test

# GMM cluster counts to test
GMM_CLUSTERS = [1, 2, 10, 20, 50, 100, 200, 500, 1000]

# Baselines
TOPK_ATTN_BUDGET = 100
UNIFORM_BUDGET = 100
ORACLE_BUDGET = 100
TOPK_VALUES_BUDGETS = [10, 50, 100, 500, 1000]

# ============================================================================
# SETUP
# ============================================================================
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


# ============================================================================
# SINGLE-TREE SIMHASH (variable seed, centered)
# ============================================================================

class SingleTreeSimHash:
    """SimHash with a single table for variable-seed experiments."""

    def __init__(self, max_depth, head_dim, seed):
        self.max_depth = max_depth
        self.head_dim = head_dim
        rng = np.random.RandomState(seed)
        hp = rng.randn(max_depth, head_dim).astype(np.float32)
        self.hyperplanes = hp / np.linalg.norm(hp, axis=1, keepdims=True)
        self.key_mean = None
        self.key_codes = None

    def build_index(self, keys):
        self.key_mean = np.mean(keys, axis=0)
        c = (keys - self.key_mean).astype(np.float32)
        self.key_codes = (c @ self.hyperplanes.T > 0).astype(np.int8)

    def hash_queries(self, Q):
        """Returns [num_queries, max_depth]."""
        c = (Q - self.key_mean).astype(np.float32)
        return (c @ self.hyperplanes.T > 0).astype(np.int8)


# ============================================================================
# HELPERS
# ============================================================================

def softmax_1d(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# ============================================================================
# BASELINE COMPUTATIONS
# ============================================================================

def compute_baselines(query, keys, values, logits, full_output, out_norm, rng):
    """Compute all baseline errors for one query position."""
    nv = len(logits)
    results = {}

    # TopK Attention @100
    budget = min(TOPK_ATTN_BUDGET, nv)
    idx = np.argpartition(logits, -budget)[-budget:]
    w = softmax_1d(logits[idx])
    topk_out = w @ values[idx]
    results['topk_attn_100'] = float(np.linalg.norm(topk_out - full_output) / out_norm)

    # Uniform @100
    budget = min(UNIFORM_BUDGET, nv)
    idx = rng.choice(nv, size=budget, replace=False)
    w = softmax_1d(logits[idx])
    uni_out = w @ values[idx]
    results['uniform_100'] = float(np.linalg.norm(uni_out - full_output) / out_norm)

    # Oracle @100
    budget = min(ORACLE_BUDGET, nv)
    full_weights = softmax_1d(logits)
    idx = rng.choice(nv, size=budget, p=full_weights, replace=True)
    oracle_out = np.mean(values[idx], axis=0)
    results['oracle_100'] = float(np.linalg.norm(oracle_out - full_output) / out_norm)

    # Oracle Value-Weighted @100
    from algorithms.oracle_value_weighted import oracle_value_weighted
    ovw_out, _ = oracle_value_weighted(query, keys, values, logits, full_weights, budget)
    results['oracle_vw_100'] = float(np.linalg.norm(ovw_out - full_output) / out_norm)

    # TopKValues-avg @B
    for B in TOPK_VALUES_BUDGETS:
        b = min(B, nv)
        idx = np.argpartition(logits, -b)[-b:]
        topkv_out = np.mean(values[idx], axis=0)
        results[f'topkv_avg_{B}'] = float(np.linalg.norm(topkv_out - full_output) / out_norm)

    # mean(V)
    mean_v = np.mean(values, axis=0)
    results['mean_v'] = float(np.linalg.norm(mean_v - full_output) / out_norm)

    return results


# ============================================================================
# HIERARCHICAL COMPUTATION (uses algorithm file)
# ============================================================================
# Implementation in src/algorithms/hierarchical_lsh.py
# Using hierarchical_lsh_attention() with L parameter


# ============================================================================
# PER-EXAMPLE PROCESSING
# ============================================================================

def process_example(example, example_idx, rng):
    """Process one example: ground truths, baselines (once), hierarchical (50 seeds)."""
    results = {}

    for layer in LAYERS:
        Q = np.array(example[layer]['Q'], dtype=np.float32)
        K_mat = np.array(example[layer]['K'], dtype=np.float32)
        V = np.array(example[layer]['V'], dtype=np.float32)
        seq_len = Q.shape[0]

        query_positions = list(range(max(0, seq_len - NUM_QUERIES), seq_len))
        n_queries = len(query_positions)

        # Precompute all logits once
        all_logits = (Q @ K_mat.T) / np.sqrt(HEAD_DIM)

        # --- Phase 1: Ground truths + baselines (once) ---
        baseline_errors = {'topk_attn_100': [], 'uniform_100': [], 'oracle_100': [], 'oracle_vw_100': [], 'mean_v': []}
        for B in TOPK_VALUES_BUDGETS:
            baseline_errors[f'topkv_avg_{B}'] = []

        ground_truths = []
        out_norms = []

        for qpos in query_positions:
            nv = qpos + 1
            logits = all_logits[qpos, :nv]
            valid_keys = K_mat[:nv]
            valid_values = V[:nv]

            full_weights = softmax_1d(logits)
            full_output = full_weights @ valid_values
            out_norm = np.linalg.norm(full_output) + 1e-8

            ground_truths.append(full_output)
            out_norms.append(out_norm)

            bl = compute_baselines(Q[qpos], valid_keys, valid_values, logits,
                                   full_output, out_norm, rng)
            for k, v in bl.items():
                baseline_errors[k].append(v)

        # --- Phase 2: Hierarchical across seeds ---
        # hier_errors[K][seed_idx] = list of n_queries errors
        hier_errors = {K: [] for K in K_VALUES}

        for seed_idx in range(NUM_SEEDS):
            seed = SEED * 1000 + example_idx * 100 + seed_idx
            sh = SingleTreeSimHash(MAX_DEPTH, HEAD_DIM, seed)
            sh.build_index(K_mat)
            all_qhashes = sh.hash_queries(Q)  # [seq_len, MAX_DEPTH]

            seed_errors = {K: [] for K in K_VALUES}

            for qi, qpos in enumerate(query_positions):
                nv = qpos + 1
                valid_keys = K_mat[:nv]
                valid_values = V[:nv]
                full_output = ground_truths[qi]
                out_norm = out_norms[qi]

                # LCP: vectorized for single tree
                q_hash = all_qhashes[qpos]          # [MAX_DEPTH]
                key_codes = sh.key_codes[:nv]        # [nv, MAX_DEPTH]
                matches = (key_codes == q_hash[np.newaxis, :])   # [nv, MAX_DEPTH]
                cum_match = np.cumprod(matches, axis=1)
                lcp_full = np.sum(cum_match, axis=1).astype(np.int32)  # [nv]

                for K_depth in K_VALUES:
                    # Reshape LCP for single tree: [nv] -> [nv, 1]
                    lcp_reshaped = lcp_full[:, np.newaxis]
                    # Reshape query hash for single tree: [MAX_DEPTH] -> [1, MAX_DEPTH]
                    q_hash_reshaped = q_hash[np.newaxis, :]
                    # Reshape key codes for single tree: [nv, MAX_DEPTH] -> [nv, 1, MAX_DEPTH]
                    key_codes_reshaped = sh.key_codes[:nv, np.newaxis, :]
                    
                    h_out, _ = hierarchical_lsh_attention(
                        Q[qpos], valid_keys, valid_values, None, HEAD_DIM,
                        key_codes_reshaped, q_hash_reshaped, K_depth, L)
                    
                    err = float(np.linalg.norm(h_out - full_output) / out_norm)
                    seed_errors[K_depth].append(err)

            for K in K_VALUES:
                hier_errors[K].append(seed_errors[K])

        # Average across seeds per query -> [n_queries]
        hier_avg = {}
        for K in K_VALUES:
            arr = np.array(hier_errors[K])  # [NUM_SEEDS, n_queries]
            hier_avg[K] = np.nanmean(arr, axis=0).tolist()

        # --- Phase 3: GMM across cluster counts ---
        gmm_errors = {}
        for C in GMM_CLUSTERS:
            gmm_seed = SEED + example_idx * 100 + C
            resp = fit_gmm(K_mat, n_clusters=C, seed=gmm_seed)

            c_errors = []
            for qi, qpos in enumerate(query_positions):
                nv = qpos + 1
                valid_keys = K_mat[:nv]
                valid_values = V[:nv]
                valid_resp = resp[:nv]

                gmm_out, _ = gmm_attention(
                    Q[qpos], valid_keys, valid_values, None, HEAD_DIM, valid_resp)
                err = float(np.linalg.norm(gmm_out - ground_truths[qi]) / out_norms[qi])
                c_errors.append(err)

            gmm_errors[C] = c_errors

        results[layer] = {
            'baselines': baseline_errors,
            'hierarchical': hier_avg,
            'gmm': gmm_errors,
            'n_queries': n_queries,
        }

    return results


# ============================================================================
# PLOTTING
# ============================================================================

LAYER_LABELS = {'first_layer': 'First Layer', 'last_layer': 'Last Layer'}
HIER_COLOR = '#1f77b4'
GMM_COLOR = '#17becf'


def plot_per_example(example_results, example_idx, output_dir):
    """Per-example plot: error vs K for hierarchical, baselines as horizontal lines."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, layer in enumerate(LAYERS):
        ax = axes[ax_idx]
        data = example_results[layer]

        # Hierarchical curve
        means = [np.nanmean(data['hierarchical'][K]) for K in K_VALUES]
        stds = [np.nanstd(data['hierarchical'][K]) for K in K_VALUES]
        means_arr = np.array(means)
        stds_arr = np.array(stds)

        ax.plot(K_VALUES, means_arr, 'o-', color=HIER_COLOR, linewidth=2, markersize=6,
                label='Hierarchical-AvgKey', zorder=5)
        ax.fill_between(K_VALUES,
                         np.maximum(means_arr - stds_arr, 0),
                         means_arr + stds_arr,
                         color=HIER_COLOR, alpha=0.15)

        # GMM curve
        gmm_means = [np.nanmean(data['gmm'][C]) for C in GMM_CLUSTERS]
        ax.plot(GMM_CLUSTERS, gmm_means, 's-', color=GMM_COLOR, linewidth=2,
                markersize=5, label='GMM Soft Clustering', zorder=5)

        # Baseline horizontal lines
        for bname, color, label in [
            ('topk_attn_100', '#d62728', 'TopK Attn @100'),
            ('uniform_100', '#ff7f0e', 'Uniform @100'),
            ('oracle_100', '#2ca02c', 'Oracle @100'),
            ('oracle_vw_100', '#006400', 'Oracle VW @100'),
            ('mean_v', '#7f7f7f', 'mean(V)'),
        ]:
            val = np.nanmean(data['baselines'][bname])
            ax.axhline(y=val, color=color, linestyle='--', linewidth=1.5, alpha=0.7, label=label)

        # TopKValues-avg baselines with distinct colors
        topkv_colors = ['#9467bd', '#c5b0d5', '#d4a5d4', '#e7bcf3', '#f3d9fa']  # Purple shades
        for bi, B in enumerate(TOPK_VALUES_BUDGETS):
            val = np.nanmean(data['baselines'][f'topkv_avg_{B}'])
            color = topkv_colors[bi % len(topkv_colors)]
            ax.axhline(y=val, color=color, linestyle=':',
                       linewidth=1.5, alpha=0.8, label=f'TopKV-avg @{B}')

        ax.set_xlabel('Representatives (K for Hier / Clusters for GMM)', fontweight='bold')
        ax.set_ylabel('Mean Relative L2 Error', fontweight='bold')
        ax.set_title(f'{LAYER_LABELS[layer]} — Example {example_idx}', fontweight='bold')
        all_xticks = sorted(set(K_VALUES + GMM_CLUSTERS))
        ax.set_xticks(all_xticks)
        ax.tick_params(axis='x', labelsize=6)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best', framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_dir / f'example_{example_idx:02d}.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_averaged_error_vs_K(all_example_results, output_dir):
    """Error vs K averaged across examples, with std error bars and baseline bands."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, layer in enumerate(LAYERS):
        ax = axes[ax_idx]

        # Per-example mean errors for each K
        per_ex = {K: [] for K in K_VALUES}
        for ex_res in all_example_results:
            for K in K_VALUES:
                per_ex[K].append(np.nanmean(ex_res[layer]['hierarchical'][K]))

        means = [np.mean(per_ex[K]) for K in K_VALUES]
        stds = [np.std(per_ex[K]) for K in K_VALUES]
        ax.errorbar(K_VALUES, means, yerr=stds, fmt='o-', color=HIER_COLOR,
                    linewidth=2.5, markersize=8, capsize=4, capthick=1.5,
                    label='Hierarchical-AvgKey', zorder=5)

        # GMM curve
        gmm_per_ex = {C: [] for C in GMM_CLUSTERS}
        for ex_res in all_example_results:
            for C in GMM_CLUSTERS:
                gmm_per_ex[C].append(np.nanmean(ex_res[layer]['gmm'][C]))
        gmm_means = [np.mean(gmm_per_ex[C]) for C in GMM_CLUSTERS]
        gmm_stds = [np.std(gmm_per_ex[C]) for C in GMM_CLUSTERS]
        ax.errorbar(GMM_CLUSTERS, gmm_means, yerr=gmm_stds, fmt='s-', color=GMM_COLOR,
                    linewidth=2.5, markersize=7, capsize=4, capthick=1.5,
                    label='GMM Soft Clustering', zorder=5)

        # Baseline bands
        for bname, color, label in [
            ('topk_attn_100', '#d62728', 'TopK Attn @100'),
            ('uniform_100', '#ff7f0e', 'Uniform @100'),
            ('oracle_100', '#2ca02c', 'Oracle @100'),
            ('oracle_vw_100', '#006400', 'Oracle VW @100'),
            ('mean_v', '#7f7f7f', 'mean(V)'),
        ]:
            vals = [np.nanmean(ex[layer]['baselines'][bname]) for ex in all_example_results]
            m, s = np.mean(vals), np.std(vals)
            ax.axhline(y=m, color=color, linestyle='--', linewidth=2, alpha=0.8, label=label)
            ax.axhspan(m - s, m + s, color=color, alpha=0.08)

        # TopKValues-avg baselines with distinct colors
        topkv_colors = ['#9467bd', '#c5b0d5', '#d4a5d4', '#e7bcf3', '#f3d9fa']  # Purple gradient
        for bi, B in enumerate(TOPK_VALUES_BUDGETS):
            vals = [np.nanmean(ex[layer]['baselines'][f'topkv_avg_{B}']) for ex in all_example_results]
            m = np.mean(vals)
            color = topkv_colors[bi % len(topkv_colors)]
            ax.axhline(y=m, color=color, linestyle=':',
                       linewidth=1.8, alpha=0.85, label=f'TopKV-avg @{B}')

        ax.set_xlabel('K (Hierarchical) / Clusters (GMM)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Mean Relative L2 Error', fontweight='bold', fontsize=12)
        ax.set_title(f'{LAYER_LABELS[layer]} — Averaged over {len(all_example_results)} examples',
                     fontweight='bold', fontsize=13)
        all_xticks = sorted(set(K_VALUES + GMM_CLUSTERS))
        ax.set_xticks(all_xticks)
        ax.tick_params(axis='x', labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best', framealpha=0.95)

    plt.tight_layout()
    fig.savefig(output_dir / 'averaged_error_vs_K.png',
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: averaged_error_vs_K.png")


def plot_bar_chart(all_example_results, output_dir):
    """Bar chart of all methods at selected K values."""
    selected_K = [k for k in [1, 3, 5, 7, 10] if k in K_VALUES]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax_idx, layer in enumerate(LAYERS):
        ax = axes[ax_idx]
        names, vals, errs, colors = [], [], [], []

        # Hierarchical at selected K
        for K in selected_K:
            v = [np.nanmean(ex[layer]['hierarchical'][K]) for ex in all_example_results]
            names.append(f'Hier K={K}')
            vals.append(np.mean(v))
            errs.append(np.std(v))
            colors.append(HIER_COLOR)

        # Baselines
        for bname, label, color in [
            ('topk_attn_100', 'TopK @100', '#d62728'),
            ('uniform_100', 'Uniform @100', '#ff7f0e'),
            ('oracle_100', 'Oracle @100', '#2ca02c'),
            ('oracle_vw_100', 'Oracle VW @100', '#006400'),
            ('mean_v', 'mean(V)', '#7f7f7f'),
        ]:
            v = [np.nanmean(ex[layer]['baselines'][bname]) for ex in all_example_results]
            names.append(label)
            vals.append(np.mean(v))
            errs.append(np.std(v))
            colors.append(color)

        # TopKValues-avg with distinct colors
        topkv_colors = ['#9467bd', '#c5b0d5', '#d4a5d4', '#e7bcf3', '#f3d9fa']
        for bi, B in enumerate(TOPK_VALUES_BUDGETS):
            v = [np.nanmean(ex[layer]['baselines'][f'topkv_avg_{B}']) for ex in all_example_results]
            names.append(f'TopKV @{B}')
            vals.append(np.mean(v))
            errs.append(np.std(v))
            colors.append(topkv_colors[bi % len(topkv_colors)])

        # GMM at selected cluster counts
        selected_C = [c for c in [2, 8, 16, 32] if c in GMM_CLUSTERS]
        for C in selected_C:
            v = [np.nanmean(ex[layer]['gmm'][C]) for ex in all_example_results]
            names.append(f'GMM C={C}')
            vals.append(np.mean(v))
            errs.append(np.std(v))
            colors.append(GMM_COLOR)

        ax.bar(range(len(names)), vals, yerr=errs, color=colors, alpha=0.8,
               capsize=3, edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean Relative L2 Error', fontweight='bold')
        ax.set_title(f'{LAYER_LABELS[layer]} — All Methods', fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_dir / 'averaged_bar_chart.png',
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: averaged_bar_chart.png")


def plot_distribution_boxplot(all_example_results, output_dir):
    """Boxplot of per-example mean error distribution for selected K values and baselines."""
    selected_K = [k for k in [1, 3, 5, 7, 10] if k in K_VALUES]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, layer in enumerate(LAYERS):
        ax = axes[ax_idx]
        data_for_box, labels, face_colors = [], [], []

        for K in selected_K:
            v = [np.nanmean(ex[layer]['hierarchical'][K]) for ex in all_example_results]
            data_for_box.append(v)
            labels.append(f'Hier K={K}')
            face_colors.append(HIER_COLOR)

        # GMM at selected cluster counts
        selected_C = [c for c in [2, 8, 16, 32] if c in GMM_CLUSTERS]
        for C in selected_C:
            v = [np.nanmean(ex[layer]['gmm'][C]) for ex in all_example_results]
            data_for_box.append(v)
            labels.append(f'GMM C={C}')
            face_colors.append(GMM_COLOR)

        for bname, label, color in [
            ('topk_attn_100', 'TopK @100', '#d62728'),
            ('uniform_100', 'Uniform @100', '#ff7f0e'),
            ('oracle_100', 'Oracle @100', '#2ca02c'),
            ('oracle_vw_100', 'Oracle VW @100', '#006400'),
        ]:
            v = [np.nanmean(ex[layer]['baselines'][bname]) for ex in all_example_results]
            data_for_box.append(v)
            labels.append(label)
            face_colors.append(color)

        bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True, widths=0.6)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(face_colors[i])
            patch.set_alpha(0.6)

        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean Relative L2 Error (per example)', fontweight='bold')
        ax.set_title(f'{LAYER_LABELS[layer]} — Distribution across examples',
                     fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_dir / 'distribution_boxplot.png',
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: distribution_boxplot.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    individual_dir = OUTPUT_DIR / 'individual_runs'
    individual_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HIERARCHICAL LSH + GMM COMPARISON")
    print("=" * 70)
    print(f"Config: {NUM_EXAMPLES} examples, {NUM_QUERIES} queries/example, "
          f"{NUM_SEEDS} seeds/example")
    print(f"Hierarchical K depths: {K_VALUES}")
    print(f"L = {L} (single tree, averaged across {NUM_SEEDS} seeds)")
    print(f"GMM clusters: {GMM_CLUSTERS}")
    print(f"Layers: {LAYERS}")
    print(f"Baselines: TopK@{TOPK_ATTN_BUDGET}, Uniform@{UNIFORM_BUDGET}, "
          f"Oracle@{ORACLE_BUDGET}")
    print(f"  TopKValues-avg: {TOPK_VALUES_BUDGETS}")
    print(f"  mean(V)")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Select examples
    print(f"Counting examples in {DATA_PATH}...")
    with open(DATA_PATH, 'r') as f:
        total = sum(1 for _ in f)
    print(f"Found {total} examples")

    num_to_load = min(NUM_EXAMPLES, total)
    selected = sorted(rng.choice(total, num_to_load, replace=False).tolist())
    print(f"Selected {num_to_load} examples (indices: {selected[:5]}...)")

    # Process examples
    print("\nProcessing examples...")
    sel_set = set(selected)
    all_example_results = []

    loaded = 0
    with open(DATA_PATH, 'r') as f:
        for idx, line in enumerate(f):
            if idx not in sel_set:
                continue

            example = json.loads(line)
            loaded += 1

            t_ex = time.time()
            ex_results = process_example(example, loaded - 1, rng)
            dt = time.time() - t_ex

            domain = example.get('domain', '?')[:30]
            print(f"  [{loaded:3d}/{num_to_load}] {domain:<30s} ({dt:.1f}s)")

            plot_per_example(ex_results, loaded - 1, individual_dir)
            all_example_results.append(ex_results)

            if loaded >= num_to_load:
                break

    # Summary plots
    print("\nGenerating summary plots...")
    plot_averaged_error_vs_K(all_example_results, OUTPUT_DIR)
    plot_bar_chart(all_example_results, OUTPUT_DIR)
    plot_distribution_boxplot(all_example_results, OUTPUT_DIR)

    # Save JSON
    print("\nSaving results JSON...")
    all_baseline_names = ['topk_attn_100', 'uniform_100', 'oracle_100', 'oracle_vw_100', 'mean_v']
    all_baseline_names += [f'topkv_avg_{B}' for B in TOPK_VALUES_BUDGETS]

    json_results = {
        'metadata': {
            'num_examples': num_to_load,
            'num_queries': NUM_QUERIES,
            'num_seeds': NUM_SEEDS,
            'max_depth': MAX_DEPTH,
            'K_values': K_VALUES,
            'L': L,
            'layers': LAYERS,
            'seed': SEED,
            'total_time_seconds': time.time() - t0,
            'gmm_clusters': GMM_CLUSTERS,
            'baselines': {
                'topk_attn_budget': TOPK_ATTN_BUDGET,
                'uniform_budget': UNIFORM_BUDGET,
                'oracle_budget': ORACLE_BUDGET,
                'topkv_budgets': TOPK_VALUES_BUDGETS,
            },
        },
        'averaged': {},
    }

    for layer in LAYERS:
        layer_data = {'hierarchical': {}, 'gmm': {}, 'baselines': {}}

        for K in K_VALUES:
            vals = [np.nanmean(ex[layer]['hierarchical'][K])
                    for ex in all_example_results]
            layer_data['hierarchical'][str(K)] = {
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'std': float(np.std(vals)),
                'per_example_means': [float(v) for v in vals],
            }

        for C in GMM_CLUSTERS:
            vals = [np.nanmean(ex[layer]['gmm'][C])
                    for ex in all_example_results]
            layer_data['gmm'][str(C)] = {
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'std': float(np.std(vals)),
                'per_example_means': [float(v) for v in vals],
            }

        for bname in all_baseline_names:
            vals = [np.nanmean(ex[layer]['baselines'][bname])
                    for ex in all_example_results]
            layer_data['baselines'][bname] = {
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'std': float(np.std(vals)),
                'per_example_means': [float(v) for v in vals],
            }

        json_results['averaged'][layer] = layer_data

    json_path = OUTPUT_DIR / 'averaged_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved: {json_path}")

    # Print summary table
    elapsed = time.time() - t0
    print(f"\nDone! Total time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    print("\n" + "=" * 85)
    print("SUMMARY")
    print("=" * 85)

    for layer in LAYERS:
        print(f"\n{LAYER_LABELS[layer].upper()}")
        print(f"{'Method':<25} {'Mean Error':>12} {'Std':>10}")
        print("-" * 50)

        ld = json_results['averaged'][layer]

        for K in K_VALUES:
            d = ld['hierarchical'][str(K)]
            print(f"{'Hier K=' + str(K):<25} {d['mean']:>12.4f} {d['std']:>10.4f}")

        print()
        for C in GMM_CLUSTERS:
            d = ld['gmm'][str(C)]
            print(f"{'GMM C=' + str(C):<25} {d['mean']:>12.4f} {d['std']:>10.4f}")

        print()
        for bname in all_baseline_names:
            d = ld['baselines'][bname]
            label = bname.replace('_', ' ')
            print(f"{label:<25} {d['mean']:>12.4f} {d['std']:>10.4f}")

    print(f"\nResults: {OUTPUT_DIR}")
    print(f"  - averaged_results.json")
    print(f"  - averaged_error_vs_K.png")
    print(f"  - averaged_bar_chart.png")
    print(f"  - distribution_boxplot.png")
    print(f"  - individual_runs/ ({num_to_load} PNGs)")


if __name__ == "__main__":
    main()
