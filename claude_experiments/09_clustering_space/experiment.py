#!/usr/bin/env python3
"""
Experiment 9: Clustering Space Comparison + PCA Dimensionality Analysis

The paper assumes clustering in key space, but the error depends on value
homogeneity within clusters.  This raises the question: should we cluster in
key space, value space, or some combination?

Also tests the intrinsic dimensionality hypothesis from Section 7: LLM key
vectors may lie on a manifold of dimension d_eff << 128, which would explain
why segmentation (a form of quantization) outperforms sampling despite
operating in d=128 dimensions.

Five clustering approaches compared (all using k-means at various C):
  1. Key-space         — cluster K (current default)
  2. Value-space       — cluster V
  3. Joint KV-space    — cluster [K, V] concatenated (normalized)
  4. PCA-key-space     — cluster PCA(K, d_eff) in reduced dimensions
  5. Logit-projected   — cluster keys projected onto query direction

Also measures:
  - PCA explained variance curve for K and V at each layer
  - Effective dimensionality (90% and 95% variance thresholds)
  - Within-cluster value variance for each clustering approach

Run from: claude_experiments/09_clustering_space/
Results saved to: claude_experiments/09_clustering_space/results/
"""

import sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from algorithms.base import compute_ground_truth_attention, relative_l2_error, softmax
from visualization.plot_utils import setup_style, save_figure

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

NUM_EXAMPLES = 10
NUM_QUERIES_PER_EXAMPLE = 50
CLUSTERS = [10, 50, 100]
PCA_COMPONENTS = 16                    # Reduced dimension for PCA-key clustering
LAYERS_TO_TEST = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'attention_vectors_long_bench_llama_8b.jsonl')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')

# ============================================================================
# END HYPERPARAMETERS
# ============================================================================

CLUSTERING_METHODS = [
    'Key-Space',
    'Value-Space',
    'Joint-KV',
    'PCA-Key',
    'Logit-Projected',
]

METHOD_COLORS = {
    'Key-Space':        '#1f77b4',
    'Value-Space':      '#ff7f0e',
    'Joint-KV':         '#2ca02c',
    'PCA-Key':          '#9467bd',
    'Logit-Projected':  '#d62728',
}

LAYER_TITLES = {
    'first_layer': 'First Layer (Layer 0)',
    'last_layer': 'Last Layer (Layer 31)',
}


def cluster_and_compute_attention(query, keys, values, labels, n_clusters, head_dim):
    """
    Given cluster labels, compute k-means-style segmentation attention.
    Keys and values are averaged within clusters, softmax over centroid logits.
    """
    sqrt_d = np.sqrt(head_dim)
    key_centroids = np.zeros((n_clusters, head_dim))
    val_centroids = np.zeros((n_clusters, head_dim))
    counts = np.zeros(n_clusters)

    for c in range(n_clusters):
        mask = labels == c
        cnt = mask.sum()
        if cnt > 0:
            key_centroids[c] = keys[mask].mean(axis=0)
            val_centroids[c] = values[mask].mean(axis=0)
            counts[c] = cnt

    active = counts > 0
    if not active.any():
        return np.zeros(head_dim)

    scores = (key_centroids[active] @ query) / sqrt_d
    weights = softmax(scores)
    return weights @ val_centroids[active]


def compute_within_cluster_value_variance(values, labels, n_clusters, gt_weights):
    """
    Compute attention-weighted within-cluster value variance.
    sum_c W_c * sum_{i in S_c} (w_i/W_c) * ||v_i - mu_c||^2
    """
    total_var = 0.0
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() <= 1:
            continue
        W_c = gt_weights[mask].sum()
        if W_c < 1e-12:
            continue
        w_in = gt_weights[mask] / W_c
        mu_c = np.sum(w_in[:, np.newaxis] * values[mask], axis=0)
        var_c = np.sum(w_in * np.sum((values[mask] - mu_c)**2, axis=1))
        total_var += W_c * var_c
    return total_var


def fit_all_clusterings(K, V, n_clusters, seed, pca_model=None):
    """
    Fit all 5 clustering approaches on full key/value matrices.
    Returns dict of {method_name: labels}.
    Note: 'Logit-Projected' needs per-query fitting, so we return None here.
    """
    result = {}

    # 1. Key-space
    km_key = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=seed)
    km_key.fit(K)
    result['Key-Space'] = km_key.labels_

    # 2. Value-space
    km_val = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=seed)
    km_val.fit(V)
    result['Value-Space'] = km_val.labels_

    # 3. Joint KV-space (normalize each to unit variance before concat)
    scaler_k = StandardScaler()
    scaler_v = StandardScaler()
    K_norm = scaler_k.fit_transform(K)
    V_norm = scaler_v.fit_transform(V)
    KV = np.concatenate([K_norm, V_norm], axis=1)  # [N, 2d]
    km_kv = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=seed)
    km_kv.fit(KV)
    result['Joint-KV'] = km_kv.labels_

    # 4. PCA-key-space
    if pca_model is not None:
        K_pca = pca_model.transform(K)
    else:
        pca = PCA(n_components=PCA_COMPONENTS, random_state=seed)
        K_pca = pca.fit_transform(K)
    km_pca = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=seed)
    km_pca.fit(K_pca)
    result['PCA-Key'] = km_pca.labels_

    # 5. Logit-Projected — query-dependent, will be computed per query
    result['Logit-Projected'] = None

    return result


def logit_projected_clustering(keys, query, n_clusters, head_dim, seed):
    """
    Cluster keys by their 1D projection onto the query direction.
    This captures the dimension most relevant to attention weights.
    Uses 1D k-means (fast).
    """
    sqrt_d = np.sqrt(head_dim)
    projections = (keys @ query) / sqrt_d  # [N] — the actual logits
    # 1D k-means
    km = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=seed)
    km.fit(projections.reshape(-1, 1))
    return km.labels_


def compute_pca_stats(K, V, seed):
    """Compute PCA explained variance for keys and values."""
    pca_k = PCA(n_components=min(HEAD_DIM, len(K) - 1), random_state=seed)
    pca_k.fit(K)

    pca_v = PCA(n_components=min(HEAD_DIM, len(V) - 1), random_state=seed)
    pca_v.fit(V)

    cumvar_k = np.cumsum(pca_k.explained_variance_ratio_)
    cumvar_v = np.cumsum(pca_v.explained_variance_ratio_)

    # Effective dimensionality at various thresholds
    def d_eff(cumvar, threshold):
        idx = np.searchsorted(cumvar, threshold)
        return min(idx + 1, len(cumvar))

    stats = {
        'key_cumvar': cumvar_k.tolist(),
        'val_cumvar': cumvar_v.tolist(),
        'key_d_eff_90': d_eff(cumvar_k, 0.90),
        'key_d_eff_95': d_eff(cumvar_k, 0.95),
        'key_d_eff_99': d_eff(cumvar_k, 0.99),
        'val_d_eff_90': d_eff(cumvar_v, 0.90),
        'val_d_eff_95': d_eff(cumvar_v, 0.95),
        'val_d_eff_99': d_eff(cumvar_v, 0.99),
    }

    # Return the fitted PCA model for use in clustering
    pca_model = PCA(n_components=PCA_COMPONENTS, random_state=seed)
    pca_model.fit(K)

    return stats, pca_model


def evaluate_query(q, K, V, query_pos, head_dim, all_labels, n_clusters):
    """Evaluate all clustering approaches for one query."""
    gt_output, gt_logits, gt_weights, _ = compute_ground_truth_attention(
        q, K, V, query_pos, head_dim
    )
    valid_keys = K[:query_pos + 1]
    valid_values = V[:query_pos + 1]
    nv = query_pos + 1

    results = {}

    for method_name in CLUSTERING_METHODS:
        if method_name == 'Logit-Projected':
            # Query-dependent clustering
            labels = logit_projected_clustering(
                valid_keys, q, n_clusters, head_dim, SEED)
        else:
            labels = all_labels[method_name][:nv]

        out = cluster_and_compute_attention(
            q, valid_keys, valid_values, labels, n_clusters, head_dim)
        err = float(relative_l2_error(out, gt_output))

        # Within-cluster value variance
        wcvv = compute_within_cluster_value_variance(
            valid_values, labels, n_clusters, gt_weights)

        results[method_name] = {
            'error': err,
            'within_cluster_value_var': float(wcvv),
        }

    return results


def plot_comparison(all_errors, output_dir):
    """Bar chart: clustering methods × cluster counts."""
    fig, axes = plt.subplots(1, len(LAYERS_TO_TEST),
                             figsize=(7 * len(LAYERS_TO_TEST), 6))
    if len(LAYERS_TO_TEST) == 1:
        axes = [axes]

    bar_width = 0.14
    x_base = np.arange(len(CLUSTERS))

    for ax, layer_name in zip(axes, LAYERS_TO_TEST):
        for i, method in enumerate(CLUSTERING_METHODS):
            means = []
            for C in CLUSTERS:
                errs = all_errors[layer_name][C][method]
                means.append(np.mean(errs) if errs else 0)

            ax.bar(x_base + i * bar_width, means, bar_width,
                   label=method, color=METHOD_COLORS[method],
                   alpha=0.85, edgecolor='white')

        ax.set_xlabel('Number of Clusters', fontsize=12)
        ax.set_ylabel('Mean Relative L2 Error', fontsize=12)
        ax.set_title(f'Clustering Space — {LAYER_TITLES[layer_name]}', fontsize=13)
        ax.set_xticks(x_base + bar_width * (len(CLUSTERING_METHODS) - 1) / 2)
        ax.set_xticklabels([str(c) for c in CLUSTERS])
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    save_figure(fig, output_dir / 'clustering_space.png')


def plot_pca_curves(pca_stats, output_dir):
    """PCA explained variance curves for keys and values at each layer."""
    fig, axes = plt.subplots(1, len(LAYERS_TO_TEST),
                             figsize=(7 * len(LAYERS_TO_TEST), 5))
    if len(LAYERS_TO_TEST) == 1:
        axes = [axes]

    for ax, layer_name in zip(axes, LAYERS_TO_TEST):
        stats_list = pca_stats[layer_name]
        # Average across examples
        all_key_cumvar = [s['key_cumvar'] for s in stats_list]
        all_val_cumvar = [s['val_cumvar'] for s in stats_list]

        min_len_k = min(len(c) for c in all_key_cumvar)
        min_len_v = min(len(c) for c in all_val_cumvar)
        mean_k = np.mean([c[:min_len_k] for c in all_key_cumvar], axis=0)
        mean_v = np.mean([c[:min_len_v] for c in all_val_cumvar], axis=0)

        ax.plot(range(1, len(mean_k) + 1), mean_k, '-', color='#1f77b4',
                linewidth=2, label='Keys')
        ax.plot(range(1, len(mean_v) + 1), mean_v, '-', color='#ff7f0e',
                linewidth=2, label='Values')

        ax.axhline(y=0.90, color='gray', linestyle='--', alpha=0.5, label='90%')
        ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, label='95%')

        # Mark effective dimensions
        mean_d_eff_k90 = np.mean([s['key_d_eff_90'] for s in stats_list])
        mean_d_eff_v90 = np.mean([s['val_d_eff_90'] for s in stats_list])
        ax.axvline(x=mean_d_eff_k90, color='#1f77b4', linestyle='--', alpha=0.3)
        ax.axvline(x=mean_d_eff_v90, color='#ff7f0e', linestyle='--', alpha=0.3)

        ax.set_xlabel('Number of PCA Components', fontsize=12)
        ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
        ax.set_title(f'PCA Spectrum — {LAYER_TITLES[layer_name]}\n'
                     f'(d_eff@90%: keys={mean_d_eff_k90:.0f}, vals={mean_d_eff_v90:.0f})',
                     fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 80)

    fig.tight_layout()
    save_figure(fig, output_dir / 'pca_spectrum.png')


def main():
    setup_style()
    np.random.seed(SEED)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 9: Clustering Space Comparison + PCA Analysis")
    print("=" * 70)
    print(f"  Methods:  {CLUSTERING_METHODS}")
    print(f"  Clusters: {CLUSTERS}")
    print(f"  PCA dim:  {PCA_COMPONENTS}")
    print(f"  Examples: {NUM_EXAMPLES}")
    print(f"  Queries:  {NUM_QUERIES_PER_EXAMPLE} per example")
    print(f"  Layers:   {LAYERS_TO_TEST}")
    print(f"  Output:   {output_dir}")
    print()

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        sys.exit(1)

    all_errors = {
        layer: {C: {m: [] for m in CLUSTERING_METHODS} for C in CLUSTERS}
        for layer in LAYERS_TO_TEST
    }
    all_wcvv = {
        layer: {C: {m: [] for m in CLUSTERING_METHODS} for C in CLUSTERS}
        for layer in LAYERS_TO_TEST
    }
    pca_stats = {layer: [] for layer in LAYERS_TO_TEST}

    t0 = time.time()

    with open(DATA_PATH, 'r') as f:
        for ex_idx, line in enumerate(f):
            if ex_idx >= NUM_EXAMPLES:
                break

            example = json.loads(line)
            seq_len = example['sequence_length']
            domain = example.get('domain', '?')[:30]
            print(f"  [{ex_idx+1:3d}/{NUM_EXAMPLES}] {domain:<30s} (seq_len={seq_len})")

            for layer_name in LAYERS_TO_TEST:
                layer_data = example[layer_name]
                Q = np.array(layer_data['Q'], dtype=np.float32)
                K = np.array(layer_data['K'], dtype=np.float32)
                V = np.array(layer_data['V'], dtype=np.float32)

                # PCA analysis (once per example per layer)
                stats, pca_model = compute_pca_stats(K, V, SEED)
                pca_stats[layer_name].append(stats)

                # Fit all clusterings for each C
                all_labels = {}
                for C in CLUSTERS:
                    all_labels[C] = fit_all_clusterings(K, V, C, SEED, pca_model)

                # Query positions
                min_pos = max(max(CLUSTERS) + 1, seq_len // 4)
                max_pos = seq_len - 1
                n_queries = min(NUM_QUERIES_PER_EXAMPLE, max_pos - min_pos + 1)
                if n_queries <= 0:
                    continue
                query_positions = np.random.choice(
                    range(min_pos, max_pos + 1), size=n_queries, replace=False)

                for qpos in query_positions:
                    for C in CLUSTERS:
                        qr = evaluate_query(Q[qpos], K, V, qpos, HEAD_DIM,
                                            all_labels[C], C)
                        for method_name, res in qr.items():
                            all_errors[layer_name][C][method_name].append(res['error'])
                            all_wcvv[layer_name][C][method_name].append(
                                res['within_cluster_value_var'])

    elapsed = time.time() - t0
    print(f"\nComputation done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Plots
    print("\nGenerating plots...")
    plot_comparison(all_errors, output_dir)
    plot_pca_curves(pca_stats, output_dir)

    # Save JSON
    json_results = {
        'metadata': {
            'experiment': 'Clustering Space Comparison + PCA Analysis',
            'clustering_methods': CLUSTERING_METHODS,
            'clusters': CLUSTERS,
            'pca_components': PCA_COMPONENTS,
            'num_examples': NUM_EXAMPLES,
            'num_queries_per_example': NUM_QUERIES_PER_EXAMPLE,
            'layers': LAYERS_TO_TEST,
            'seed': SEED,
            'elapsed_seconds': elapsed,
        },
        'results': {},
        'within_cluster_value_variance': {},
        'pca_statistics': {},
    }

    for layer_name in LAYERS_TO_TEST:
        layer_out = {}
        wcvv_out = {}
        for C in CLUSTERS:
            c_out = {}
            w_out = {}
            for method in CLUSTERING_METHODS:
                errs = all_errors[layer_name][C][method]
                if errs:
                    c_out[method] = {
                        'mean': float(np.mean(errs)),
                        'median': float(np.median(errs)),
                        'std': float(np.std(errs)),
                        'n': len(errs),
                    }
                wcvvs = all_wcvv[layer_name][C][method]
                if wcvvs:
                    w_out[method] = {
                        'mean': float(np.mean(wcvvs)),
                        'median': float(np.median(wcvvs)),
                    }
            layer_out[str(C)] = c_out
            wcvv_out[str(C)] = w_out
        json_results['results'][layer_name] = layer_out
        json_results['within_cluster_value_variance'][layer_name] = wcvv_out

        # Aggregate PCA stats
        stats_list = pca_stats[layer_name]
        json_results['pca_statistics'][layer_name] = {
            'mean_key_d_eff_90': float(np.mean([s['key_d_eff_90'] for s in stats_list])),
            'mean_key_d_eff_95': float(np.mean([s['key_d_eff_95'] for s in stats_list])),
            'mean_key_d_eff_99': float(np.mean([s['key_d_eff_99'] for s in stats_list])),
            'mean_val_d_eff_90': float(np.mean([s['val_d_eff_90'] for s in stats_list])),
            'mean_val_d_eff_95': float(np.mean([s['val_d_eff_95'] for s in stats_list])),
            'mean_val_d_eff_99': float(np.mean([s['val_d_eff_99'] for s in stats_list])),
        }

    json_path = output_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved: {json_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for layer_name in LAYERS_TO_TEST:
        print(f"\n{LAYER_TITLES[layer_name]}")

        # PCA dimensionality
        stats_list = pca_stats[layer_name]
        print(f"\n  PCA Effective Dimensionality (mean across examples):")
        print(f"    Keys:   d_eff@90%={np.mean([s['key_d_eff_90'] for s in stats_list]):.0f}, "
              f"@95%={np.mean([s['key_d_eff_95'] for s in stats_list]):.0f}, "
              f"@99%={np.mean([s['key_d_eff_99'] for s in stats_list]):.0f}")
        print(f"    Values: d_eff@90%={np.mean([s['val_d_eff_90'] for s in stats_list]):.0f}, "
              f"@95%={np.mean([s['val_d_eff_95'] for s in stats_list]):.0f}, "
              f"@99%={np.mean([s['val_d_eff_99'] for s in stats_list]):.0f}")

        # Error comparison
        for C in CLUSTERS:
            print(f"\n  C = {C}")
            header = f"    {'Method':<20s} {'Mean Err':>10s} {'Median':>10s} {'WCVV':>10s}"
            print(header)
            print("    " + "-" * 52)

            # Sort by mean error
            method_means = []
            for method in CLUSTERING_METHODS:
                errs = all_errors[layer_name][C][method]
                wcvvs = all_wcvv[layer_name][C][method]
                if errs:
                    method_means.append((method, np.mean(errs), np.median(errs),
                                         np.mean(wcvvs) if wcvvs else 0))

            method_means.sort(key=lambda x: x[1])
            for method, mean_err, med_err, wcvv_mean in method_means:
                print(f"    {method:<20s} {mean_err:>10.4f} {med_err:>10.4f} {wcvv_mean:>10.4f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
