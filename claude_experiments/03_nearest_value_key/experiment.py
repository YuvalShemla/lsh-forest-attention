#!/usr/bin/env python3
"""
Experiment 3: Nearest-Value Key Selection (Strategy A')

Tests whether using the key of the nearest-to-centroid value beats
responsibility-weighted averaged keys in GMM attention.

Strategy A' (Nearest-Value Key):
  1. Compute the value centroid for each cluster: v_bar_c = sum_i r_ic * v_i / sum_i r_ic
  2. Find position i* whose value is nearest to v_bar_c (among r_ic > threshold)
  3. Use k_{i*} as representative key (instead of averaged key)
  4. Keep value representative as averaged value (same as standard GMM)

Three methods compared:
  1. Standard GMM        -- averaged keys, averaged values
  2. Nearest-Value Key   -- nearest-to-centroid-value key, averaged values (Strategy A')
  3. Exact Both (oracle) -- oracle weights and oracle values (upper bound)

Sweep: C = [10, 50, 100] clusters, both layers.
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: from claude_experiments/03_nearest_value_key/ -> ../../src/
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

from algorithms.base import softmax, compute_ground_truth_attention, relative_l2_error
from algorithms.gmm_attention import fit_gmm, gmm_attention

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

NUM_EXAMPLES = 10
NUM_QUERIES_PER_EXAMPLE = 50
CLUSTER_COUNTS = [10, 50, 100]
LAYERS_TO_TEST = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'attention_vectors_long_bench_llama_8b.jsonl')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')

# ============================================================================
# END HYPERPARAMETERS
# ============================================================================

METHODS = ['Standard GMM', 'Nearest-Value Key', 'Exact Both']

LAYER_TITLES = {
    'first_layer': 'First Layer (Layer 0)',
    'last_layer': 'Last Layer (Layer 31)',
}


# ---------------------------------------------------------------------------
# Strategy A': Nearest-Value Key Attention
# ---------------------------------------------------------------------------
def nearest_value_key_attention(query, keys, values, head_dim, resp):
    """
    Strategy A': use key of nearest-to-centroid value as cluster representative.

    For each cluster:
      1. Compute value centroid = responsibility-weighted average of values
      2. Among positions with r_ic > 0.01, find the one whose value is
         closest (L2) to the centroid
      3. Use that position's key as the representative key
      4. Value representative = responsibility-weighted average (same as standard GMM)

    Args:
        query: [head_dim] query vector
        keys: [nv, head_dim] valid key vectors
        values: [nv, head_dim] valid value vectors
        head_dim: dimension
        resp: [nv, n_clusters] GMM responsibilities

    Returns:
        output: [head_dim] approximate attention output
        n_active: number of active clusters used
    """
    nv = len(keys)
    n_clusters = resp.shape[1]

    effective_counts = resp.sum(axis=0)  # [C]
    active_mask = effective_counts > 1e-8

    if not active_mask.any():
        return np.zeros(head_dim), 0

    active_resp = resp[:, active_mask]          # [nv, A]
    active_counts = effective_counts[active_mask]  # [A]
    n_active = int(active_mask.sum())

    # Value centroids: responsibility-weighted average (same as standard GMM)
    avg_values = (active_resp.T @ values) / active_counts[:, np.newaxis]  # [A, d]

    # For each active cluster, find the key whose value is nearest to the centroid
    rep_keys = np.zeros((n_active, head_dim))
    for c in range(n_active):
        # Find positions with significant responsibility for this cluster
        mask = active_resp[:, c] > 0.01
        if not mask.any():
            # Fallback to responsibility-weighted averaged key
            rep_keys[c] = (active_resp[:, c] @ keys) / active_counts[c]
            continue
        candidate_values = values[mask]
        candidate_keys = keys[mask]
        # Find value nearest to centroid
        dists = np.linalg.norm(candidate_values - avg_values[c], axis=1)
        nearest_idx = np.argmin(dists)
        rep_keys[c] = candidate_keys[nearest_idx]

    # Softmax attention over representative keys -> averaged values
    sqrt_d = np.sqrt(head_dim)
    scores = (rep_keys @ query) / sqrt_d
    weights = softmax(scores)
    output = weights @ avg_values

    return output, n_active


# ---------------------------------------------------------------------------
# Exact Both (oracle upper bound) -- inline to avoid dependency on gmm_ablation
# ---------------------------------------------------------------------------
def gmm_exact_both(query, keys, values, logits, head_dim, resp, true_weights):
    """
    GMM with exact weights AND exact value representatives.

    Shows the irreducible error from the partition itself: even with perfect
    weights and representatives, the soft partition still loses information.

    Args:
        query: [head_dim] query vector (unused, kept for interface)
        keys: [nv, head_dim] valid key vectors (unused)
        values: [nv, head_dim] valid value vectors
        logits: unused
        head_dim: dimension
        resp: [nv, n_clusters] GMM responsibilities
        true_weights: [nv] true attention weights from ground truth

    Returns:
        output: [head_dim] approximate attention output
        n_active: number of active clusters
    """
    nv = len(keys)
    if nv == 0:
        return np.zeros(head_dim), 0

    effective_counts = resp.sum(axis=0)
    active_mask = effective_counts > 1e-8
    if not active_mask.any():
        return np.zeros(head_dim), 0

    active_resp = resp[:, active_mask]

    # Weights: exact cluster masses from true attention weights
    cluster_weights = active_resp.T @ true_weights
    total = cluster_weights.sum()
    if total < 1e-12:
        return np.zeros(head_dim), 0
    cluster_weights = cluster_weights / total

    # Value representatives: attention-weighted within-cluster means
    w_resp = active_resp * true_weights[:, np.newaxis]
    w_resp_sums = w_resp.sum(axis=0)
    safe_sums = np.maximum(w_resp_sums, 1e-12)
    avg_values = (w_resp.T @ values) / safe_sums[:, np.newaxis]

    output = cluster_weights @ avg_values
    return output, int(active_mask.sum())


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------
def evaluate_query(q, K, V, query_pos, head_dim, resp_dict):
    """Evaluate all three methods for one query across all cluster counts."""
    gt_output, gt_logits, gt_weights, _ = compute_ground_truth_attention(
        q, K, V, query_pos, head_dim
    )
    valid_keys = K[:query_pos + 1]
    valid_values = V[:query_pos + 1]
    nv = len(valid_keys)

    results = {}
    for C, full_resp in resp_dict.items():
        resp = full_resp[:nv]

        # 1. Standard GMM
        out_std, n_active = gmm_attention(
            q, valid_keys, valid_values, gt_logits, head_dim, resp)

        # 2. Nearest-Value Key (Strategy A')
        out_nvk, _ = nearest_value_key_attention(
            q, valid_keys, valid_values, head_dim, resp)

        # 3. Exact Both (oracle upper bound)
        out_eb, _ = gmm_exact_both(
            q, valid_keys, valid_values, gt_logits, head_dim, resp, gt_weights)

        results[C] = {
            'Standard GMM': float(relative_l2_error(out_std, gt_output)),
            'Nearest-Value Key': float(relative_l2_error(out_nvk, gt_output)),
            'Exact Both': float(relative_l2_error(out_eb, gt_output)),
            'n_active': int(n_active),
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(all_errors, output_dir):
    """Bar chart: methods grouped by cluster count, one subplot per layer."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    METHOD_COLORS = {
        'Standard GMM':      '#1f77b4',
        'Nearest-Value Key': '#ff7f0e',
        'Exact Both':        '#2ca02c',
    }

    fig, axes = plt.subplots(1, len(LAYERS_TO_TEST), figsize=(7 * len(LAYERS_TO_TEST), 6))
    if len(LAYERS_TO_TEST) == 1:
        axes = [axes]

    bar_width = 0.22
    x_base = np.arange(len(CLUSTER_COUNTS))

    for ax, layer_name in zip(axes, LAYERS_TO_TEST):
        for i, method in enumerate(METHODS):
            means = []
            stds = []
            for C in CLUSTER_COUNTS:
                errs = all_errors[layer_name][C][method]
                means.append(np.mean(errs) if errs else 0)
                stds.append(np.std(errs) if errs else 0)

            ax.bar(x_base + i * bar_width, means, bar_width,
                   yerr=stds, capsize=3,
                   label=method, color=METHOD_COLORS[method],
                   alpha=0.85, edgecolor='white')

        ax.set_xlabel('Number of Clusters', fontsize=12)
        ax.set_ylabel('Mean Relative L2 Error', fontsize=12)
        ax.set_title(f'Nearest-Value Key vs Standard GMM\n{LAYER_TITLES[layer_name]}',
                     fontsize=13)
        ax.set_xticks(x_base + bar_width * (len(METHODS) - 1) / 2)
        ax.set_xticklabels([str(c) for c in CLUSTER_COUNTS])
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'nearest_value_key_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {os.path.join(output_dir, 'nearest_value_key_comparison.png')}")


def plot_improvement(all_errors, output_dir):
    """Plot relative improvement of A' over Standard GMM per cluster count and layer."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(LAYERS_TO_TEST), figsize=(7 * len(LAYERS_TO_TEST), 5))
    if len(LAYERS_TO_TEST) == 1:
        axes = [axes]

    for ax, layer_name in zip(axes, LAYERS_TO_TEST):
        improvements = []
        labels = []
        colors = []
        for C in CLUSTER_COUNTS:
            std_errs = np.array(all_errors[layer_name][C]['Standard GMM'])
            nvk_errs = np.array(all_errors[layer_name][C]['Nearest-Value Key'])
            if len(std_errs) == 0:
                continue
            # Per-query relative improvement: (std - nvk) / std * 100
            # Positive = A' is better, negative = A' is worse
            safe_std = np.maximum(std_errs, 1e-12)
            pct_improvement = (std_errs - nvk_errs) / safe_std * 100
            improvements.append(pct_improvement)
            labels.append(f'C={C}')

        bp = ax.boxplot(improvements, labels=labels, patch_artist=True, widths=0.5)
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for patch, color in zip(bp['boxes'], palette[:len(improvements)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No improvement')
        ax.set_xlabel('Number of Clusters', fontsize=12)
        ax.set_ylabel('Relative Improvement (%)\n(positive = A\' better)', fontsize=11)
        ax.set_title(f'A\' Improvement over Standard GMM\n{LAYER_TITLES[layer_name]}',
                     fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'improvement_boxplot.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {os.path.join(output_dir, 'improvement_boxplot.png')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(SEED)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("Experiment 3: Nearest-Value Key Selection (Strategy A')", flush=True)
    print("=" * 70, flush=True)
    print(f"  Methods:    {METHODS}", flush=True)
    print(f"  Clusters:   {CLUSTER_COUNTS}", flush=True)
    print(f"  Examples:   {NUM_EXAMPLES}", flush=True)
    print(f"  Queries:    {NUM_QUERIES_PER_EXAMPLE} per example", flush=True)
    print(f"  Layers:     {LAYERS_TO_TEST}", flush=True)
    print(f"  Data:       {DATA_PATH}", flush=True)
    print(f"  Output:     {OUTPUT_DIR}", flush=True)
    print(flush=True)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        sys.exit(1)

    # Collect errors: {layer: {C: {method: [errors]}}}
    all_errors = {
        layer: {C: {m: [] for m in METHODS} for C in CLUSTER_COUNTS}
        for layer in LAYERS_TO_TEST
    }

    t0 = time.time()

    with open(DATA_PATH, 'r') as f:
        for ex_idx, line in enumerate(f):
            if ex_idx >= NUM_EXAMPLES:
                break

            t_load = time.time()
            example = json.loads(line)
            seq_len = example['sequence_length']
            domain = example.get('domain', '?')[:30]
            print(f"  [{ex_idx+1:3d}/{NUM_EXAMPLES}] {domain:<30s} (seq_len={seq_len}) [loaded in {time.time()-t_load:.1f}s]")
            sys.stdout.flush()

            for layer_name in LAYERS_TO_TEST:
                t_layer = time.time()
                layer_data = example[layer_name]
                Q = np.array(layer_data['Q'], dtype=np.float32)
                K = np.array(layer_data['K'], dtype=np.float32)
                V = np.array(layer_data['V'], dtype=np.float32)

                # Fit GMM once per cluster count (on all keys)
                resp_dict = {}
                for C in CLUSTER_COUNTS:
                    t_gmm = time.time()
                    resp_dict[C] = fit_gmm(K, C, seed=SEED)
                    print(f"    GMM fit C={C} in {time.time()-t_gmm:.1f}s")
                    sys.stdout.flush()

                # Pick query positions from the second half of the sequence
                min_pos = max(max(CLUSTER_COUNTS) + 1, seq_len // 2)
                max_pos = seq_len - 1
                n_queries = min(NUM_QUERIES_PER_EXAMPLE, max_pos - min_pos + 1)
                if n_queries <= 0:
                    print(f"    Skipping {layer_name}: not enough positions")
                    sys.stdout.flush()
                    continue

                query_positions = np.random.choice(
                    range(min_pos, max_pos + 1), size=n_queries, replace=False)

                for qpos in query_positions:
                    qr = evaluate_query(Q[qpos], K, V, qpos, HEAD_DIM, resp_dict)
                    for C, method_errors in qr.items():
                        for method in METHODS:
                            all_errors[layer_name][C][method].append(method_errors[method])

                print(f"    {layer_name}: {n_queries} queries done in {time.time()-t_layer:.1f}s")
                sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\nComputation done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")
    plot_results(all_errors, str(output_dir))
    plot_improvement(all_errors, str(output_dir))

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    json_results = {
        'metadata': {
            'experiment': 'Nearest-Value Key Selection (Strategy A\')',
            'cluster_counts': CLUSTER_COUNTS,
            'num_examples': NUM_EXAMPLES,
            'num_queries_per_example': NUM_QUERIES_PER_EXAMPLE,
            'layers': LAYERS_TO_TEST,
            'seed': SEED,
            'head_dim': HEAD_DIM,
            'elapsed_seconds': round(elapsed, 1),
        },
        'results': {},
    }

    for layer_name in LAYERS_TO_TEST:
        layer_out = {}
        for C in CLUSTER_COUNTS:
            cluster_out = {}
            for method in METHODS:
                errs = all_errors[layer_name][C][method]
                if errs:
                    cluster_out[method] = {
                        'mean': round(float(np.mean(errs)), 6),
                        'median': round(float(np.median(errs)), 6),
                        'std': round(float(np.std(errs)), 6),
                        'p25': round(float(np.percentile(errs, 25)), 6),
                        'p75': round(float(np.percentile(errs, 75)), 6),
                        'min': round(float(np.min(errs)), 6),
                        'max': round(float(np.max(errs)), 6),
                        'n': len(errs),
                    }
            layer_out[str(C)] = cluster_out
        json_results['results'][layer_name] = layer_out

    # Compute improvement statistics
    json_results['improvement'] = {}
    for layer_name in LAYERS_TO_TEST:
        layer_imp = {}
        for C in CLUSTER_COUNTS:
            std_errs = np.array(all_errors[layer_name][C]['Standard GMM'])
            nvk_errs = np.array(all_errors[layer_name][C]['Nearest-Value Key'])
            if len(std_errs) > 0:
                safe_std = np.maximum(std_errs, 1e-12)
                pct_imp = (std_errs - nvk_errs) / safe_std * 100
                layer_imp[str(C)] = {
                    'mean_pct_improvement': round(float(np.mean(pct_imp)), 2),
                    'median_pct_improvement': round(float(np.median(pct_imp)), 2),
                    'frac_improved': round(float(np.mean(pct_imp > 0)), 4),
                    'frac_degraded': round(float(np.mean(pct_imp < 0)), 4),
                    'mean_abs_error_reduction': round(float(np.mean(std_errs) - np.mean(nvk_errs)), 6),
                }
        json_results['improvement'][layer_name] = layer_imp

    json_path = output_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for layer_name in LAYERS_TO_TEST:
        print(f"\n{LAYER_TITLES[layer_name]}")
        print("-" * 65)

        for C in CLUSTER_COUNTS:
            print(f"\n  C = {C} clusters")
            header = f"    {'Method':<22s} {'Mean':>10s} {'Median':>10s} {'Std':>10s}"
            print(header)
            print("    " + "-" * 55)
            for method in METHODS:
                errs = all_errors[layer_name][C][method]
                if errs:
                    print(f"    {method:<22s} {np.mean(errs):>10.4f} "
                          f"{np.median(errs):>10.4f} {np.std(errs):>10.4f}")

            # Improvement summary
            std_mean = np.mean(all_errors[layer_name][C]['Standard GMM'])
            nvk_mean = np.mean(all_errors[layer_name][C]['Nearest-Value Key'])
            if std_mean > 1e-12:
                pct = (std_mean - nvk_mean) / std_mean * 100
                direction = "better" if pct > 0 else "worse"
                print(f"    --> A' is {abs(pct):.1f}% {direction} than Standard GMM (mean error)")

    print("\n" + "=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
