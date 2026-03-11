#!/usr/bin/env python3
"""
Experiment 6: Jensen Bias Correction for Cluster Weights

The ablation (Exp 5) showed weight distortion accounts for 84% (first layer) /
46% (last layer) of GMM error.  The weight distortion arises from Jensen's
inequality: softmax over centroid logits underestimates the true cluster mass.

The 2nd-order Taylor correction says the true unnormalized cluster mass is
approximately exp(z_bar_c) * (1 + sigma_c^2 / 2), where sigma_c^2 is the
within-cluster logit variance.

KEY IDEA: Estimate sigma_c^2 cheaply from precomputed per-cluster key
variances (diagonal approximation):
    sigma_c^2 ≈ (1/d) * sum_j q_j^2 * Var_c(k_{ij})
Cost: O(Cd) per query (same order as centroid softmax itself).

Five variants compared:
  1. Standard k-means attention  — baseline
  2. Diagonal correction         — cheap sigma_c^2 estimate from key variances
  3. Exact correction            — sigma_c^2 from true logits (oracle, O(Nd))
  4. Exact weights               — true W_c from attention weights (upper bound)
  5. Standard GMM                — soft clustering baseline

Also measures: gap closed by correction (% of standard→exact-weights gap).

Run from: claude_experiments/06_jensen_correction/
Results saved to: claude_experiments/06_jensen_correction/results/
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

from algorithms.base import compute_ground_truth_attention, relative_l2_error, softmax
from algorithms.gmm_attention import fit_gmm, gmm_attention
from visualization.plot_utils import setup_style, save_figure

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

NUM_EXAMPLES = 10
NUM_QUERIES_PER_EXAMPLE = 50
CLUSTERS = [10, 50, 100, 200]
LAYERS_TO_TEST = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'attention_vectors_long_bench_llama_8b.jsonl')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')

# ============================================================================
# END HYPERPARAMETERS
# ============================================================================

METHODS = [
    'KMeans Standard',
    'KMeans Diag Correction',
    'KMeans Exact Correction',
    'KMeans Exact Weights',
    'GMM Standard',
]

METHOD_STYLES = {
    'KMeans Standard':          {'color': '#1f77b4'},
    'KMeans Diag Correction':   {'color': '#ff7f0e'},
    'KMeans Exact Correction':  {'color': '#2ca02c'},
    'KMeans Exact Weights':     {'color': '#9467bd'},
    'GMM Standard':             {'color': '#d62728'},
}

LAYER_TITLES = {
    'first_layer': 'First Layer (Layer 0)',
    'last_layer': 'Last Layer (Layer 31)',
}


def fit_kmeans_with_stats(keys, n_clusters, seed=42):
    """
    Fit k-means and precompute per-cluster statistics needed for Jensen correction.

    Returns:
        labels: [N] hard cluster assignments
        centroids: [C, d] cluster centroids (mean keys)
        value_centroids: None (computed separately per query set)
        key_variances: [C, d] per-cluster per-dimension key variance
    """
    km = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=seed)
    km.fit(keys)
    labels = km.labels_
    centroids = km.cluster_centers_  # [C, d]

    # Per-cluster key variance (diagonal of covariance)
    key_variances = np.zeros((n_clusters, keys.shape[1]))
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() > 1:
            key_variances[c] = np.var(keys[mask], axis=0)

    return labels, centroids, key_variances


def kmeans_attention(query, keys, values, head_dim, labels, centroids, n_clusters):
    """Standard k-means attention: centroid softmax + unweighted value means."""
    sqrt_d = np.sqrt(head_dim)

    # Cluster value centroids (unweighted means)
    val_centroids = np.zeros((n_clusters, head_dim))
    counts = np.zeros(n_clusters)
    for c in range(n_clusters):
        mask = labels == c
        cnt = mask.sum()
        if cnt > 0:
            val_centroids[c] = values[mask].mean(axis=0)
            counts[c] = cnt

    active = counts > 0
    if not active.any():
        return np.zeros(head_dim)

    scores = (centroids[active] @ query) / sqrt_d
    weights = softmax(scores)
    return weights @ val_centroids[active]


def kmeans_corrected_attention(query, keys, values, head_dim, labels, centroids,
                               key_variances, n_clusters, use_exact_sigma=False,
                               gt_logits=None):
    """
    K-means attention with Jensen bias correction.

    The unnormalized weight for cluster c is corrected:
        W_c_corrected = exp(z_bar_c) * (1 + sigma_c^2 / 2)

    sigma_c^2 is estimated either from:
      - Diagonal approximation: (1/d) * sum_j q_j^2 * Var_c(k_j)  [cheap]
      - Exact logits: Var_{i in S_c}(q^T k_i / sqrt(d))             [oracle]
    """
    sqrt_d = np.sqrt(head_dim)

    val_centroids = np.zeros((n_clusters, head_dim))
    counts = np.zeros(n_clusters)
    sigma_sq = np.zeros(n_clusters)

    for c in range(n_clusters):
        mask = labels == c
        cnt = mask.sum()
        if cnt > 0:
            val_centroids[c] = values[mask].mean(axis=0)
            counts[c] = cnt

            if use_exact_sigma and gt_logits is not None:
                # Exact: variance of true logits within cluster
                sigma_sq[c] = np.var(gt_logits[mask])
            else:
                # Diagonal approximation: (1/d) * q^T diag(Sigma_c) q
                sigma_sq[c] = np.sum(query**2 * key_variances[c]) / head_dim

    active = counts > 0
    if not active.any():
        return np.zeros(head_dim)

    # Centroid logits
    z_bar = (centroids[active] @ query) / sqrt_d

    # Corrected unnormalized weights: exp(z_bar) * (1 + sigma^2/2)
    correction = 1.0 + sigma_sq[active] / 2.0
    unnorm = np.exp(z_bar - z_bar.max()) * correction

    weights = unnorm / unnorm.sum()
    return weights @ val_centroids[active]


def kmeans_exact_weights_attention(query, keys, values, head_dim, labels,
                                   true_weights, n_clusters):
    """K-means with exact cluster weights from true attention distribution."""
    val_centroids = np.zeros((n_clusters, head_dim))
    cluster_weights = np.zeros(n_clusters)

    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() > 0:
            val_centroids[c] = values[mask].mean(axis=0)
            cluster_weights[c] = true_weights[mask].sum()

    active = cluster_weights > 1e-12
    if not active.any():
        return np.zeros(head_dim)

    w = cluster_weights[active]
    w = w / w.sum()
    return w @ val_centroids[active]


def evaluate_query(q, K, V, query_pos, head_dim, precomputed):
    """Evaluate all correction variants for one query."""
    gt_output, gt_logits, gt_weights, _ = compute_ground_truth_attention(
        q, K, V, query_pos, head_dim
    )
    valid_keys = K[:query_pos + 1]
    valid_values = V[:query_pos + 1]
    nv = query_pos + 1

    results = {}

    for C in CLUSTERS:
        pc = precomputed[C]
        labels = pc['labels'][:nv]
        centroids = pc['centroids']
        key_vars = pc['key_variances']
        gmm_resp = pc['gmm_resp'][:nv]

        # 1. Standard k-means
        out_std = kmeans_attention(
            q, valid_keys, valid_values, head_dim, labels, centroids, C)

        # 2. Diagonal correction
        out_diag = kmeans_corrected_attention(
            q, valid_keys, valid_values, head_dim, labels, centroids,
            key_vars, C, use_exact_sigma=False)

        # 3. Exact correction (oracle sigma from true logits)
        out_exact_corr = kmeans_corrected_attention(
            q, valid_keys, valid_values, head_dim, labels, centroids,
            key_vars, C, use_exact_sigma=True, gt_logits=gt_logits)

        # 4. Exact weights (oracle upper bound)
        out_exact_wt = kmeans_exact_weights_attention(
            q, valid_keys, valid_values, head_dim, labels, gt_weights, C)

        # 5. Standard GMM
        out_gmm, _ = gmm_attention(
            q, valid_keys, valid_values, gt_logits, head_dim, gmm_resp)

        errors = {
            'KMeans Standard':         float(relative_l2_error(out_std, gt_output)),
            'KMeans Diag Correction':  float(relative_l2_error(out_diag, gt_output)),
            'KMeans Exact Correction': float(relative_l2_error(out_exact_corr, gt_output)),
            'KMeans Exact Weights':    float(relative_l2_error(out_exact_wt, gt_output)),
            'GMM Standard':            float(relative_l2_error(out_gmm, gt_output)),
        }

        # Also record the sigma^2 statistics for analysis
        sigma_diag = []
        sigma_exact = []
        for c in range(C):
            mask = labels == c
            if mask.sum() > 1:
                s_d = float(np.sum(q**2 * key_vars[c]) / head_dim)
                s_e = float(np.var(gt_logits[mask]))
                sigma_diag.append(s_d)
                sigma_exact.append(s_e)

        results[C] = {
            'errors': errors,
            'mean_sigma_diag': float(np.mean(sigma_diag)) if sigma_diag else 0,
            'mean_sigma_exact': float(np.mean(sigma_exact)) if sigma_exact else 0,
        }

    return results


def plot_results(all_errors, all_gap_closed, output_dir):
    """Two-panel plot: error comparison + gap closed by correction."""
    fig, axes = plt.subplots(2, len(LAYERS_TO_TEST),
                             figsize=(7 * len(LAYERS_TO_TEST), 10))
    if len(LAYERS_TO_TEST) == 1:
        axes = axes.reshape(-1, 1)

    bar_width = 0.14
    x_base = np.arange(len(CLUSTERS))

    for col, layer_name in enumerate(LAYERS_TO_TEST):
        ax = axes[0, col]
        for i, method in enumerate(METHODS):
            means = []
            for C in CLUSTERS:
                errs = all_errors[layer_name][C][method]
                means.append(np.mean(errs) if errs else 0)
            style = METHOD_STYLES[method]
            ax.bar(x_base + i * bar_width, means, bar_width,
                   label=method, color=style['color'], alpha=0.85, edgecolor='white')

        ax.set_xlabel('Number of Clusters', fontsize=12)
        ax.set_ylabel('Mean Relative L2 Error', fontsize=12)
        ax.set_title(f'Jensen Correction — {LAYER_TITLES[layer_name]}', fontsize=13)
        ax.set_xticks(x_base + bar_width * (len(METHODS) - 1) / 2)
        ax.set_xticklabels([str(c) for c in CLUSTERS])
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        # Gap closed subplot
        ax2 = axes[1, col]
        for method_key, label, color in [
            ('diag', 'Diagonal Correction', '#ff7f0e'),
            ('exact_corr', 'Exact Correction', '#2ca02c'),
        ]:
            gaps = [all_gap_closed[layer_name].get(C, {}).get(method_key, 0) for C in CLUSTERS]
            ax2.bar(x_base + (0 if method_key == 'diag' else 1) * 0.3,
                    [g * 100 for g in gaps], 0.25, label=label, color=color, alpha=0.85)

        ax2.set_xlabel('Number of Clusters', fontsize=12)
        ax2.set_ylabel('Gap Closed (%)', fontsize=12)
        ax2.set_title(f'% of Standard→Exact Gap Closed — {LAYER_TITLES[layer_name]}', fontsize=13)
        ax2.set_xticks(x_base + 0.15)
        ax2.set_xticklabels([str(c) for c in CLUSTERS])
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100% (Exact Weights)')

    fig.tight_layout()
    save_figure(fig, output_dir / 'jensen_correction.png')


def main():
    setup_style()
    np.random.seed(SEED)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 6: Jensen Bias Correction")
    print("=" * 70)
    print(f"  Methods:  {METHODS}")
    print(f"  Clusters: {CLUSTERS}")
    print(f"  Examples: {NUM_EXAMPLES}")
    print(f"  Queries:  {NUM_QUERIES_PER_EXAMPLE} per example")
    print(f"  Layers:   {LAYERS_TO_TEST}")
    print(f"  Output:   {output_dir}")
    print()

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        sys.exit(1)

    all_errors = {
        layer: {C: {m: [] for m in METHODS} for C in CLUSTERS}
        for layer in LAYERS_TO_TEST
    }
    all_sigmas = {layer: {C: {'diag': [], 'exact': []} for C in CLUSTERS}
                  for layer in LAYERS_TO_TEST}

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

                # Precompute k-means + GMM for each cluster count
                precomputed = {}
                for C in CLUSTERS:
                    labels, centroids, key_vars = fit_kmeans_with_stats(K, C, seed=SEED)
                    gmm_resp = fit_gmm(K, C, seed=SEED)
                    precomputed[C] = {
                        'labels': labels,
                        'centroids': centroids,
                        'key_variances': key_vars,
                        'gmm_resp': gmm_resp,
                    }

                # Query positions
                min_pos = max(max(CLUSTERS) + 1, seq_len // 4)
                max_pos = seq_len - 1
                n_queries = min(NUM_QUERIES_PER_EXAMPLE, max_pos - min_pos + 1)
                if n_queries <= 0:
                    continue
                query_positions = np.random.choice(
                    range(min_pos, max_pos + 1), size=n_queries, replace=False)

                for qpos in query_positions:
                    qr = evaluate_query(Q[qpos], K, V, qpos, HEAD_DIM, precomputed)
                    for C, res in qr.items():
                        for method, err in res['errors'].items():
                            all_errors[layer_name][C][method].append(err)
                        all_sigmas[layer_name][C]['diag'].append(res['mean_sigma_diag'])
                        all_sigmas[layer_name][C]['exact'].append(res['mean_sigma_exact'])

    elapsed = time.time() - t0
    print(f"\nComputation done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Compute gap closed
    all_gap_closed = {}
    for layer_name in LAYERS_TO_TEST:
        all_gap_closed[layer_name] = {}
        for C in CLUSTERS:
            std_err = np.mean(all_errors[layer_name][C]['KMeans Standard'])
            exact_err = np.mean(all_errors[layer_name][C]['KMeans Exact Weights'])
            diag_err = np.mean(all_errors[layer_name][C]['KMeans Diag Correction'])
            exact_corr_err = np.mean(all_errors[layer_name][C]['KMeans Exact Correction'])

            gap = std_err - exact_err
            if gap > 1e-8:
                all_gap_closed[layer_name][C] = {
                    'diag': (std_err - diag_err) / gap,
                    'exact_corr': (std_err - exact_corr_err) / gap,
                }
            else:
                all_gap_closed[layer_name][C] = {'diag': 0, 'exact_corr': 0}

    # Plot
    print("\nGenerating plots...")
    plot_results(all_errors, all_gap_closed, output_dir)

    # Save JSON
    json_results = {
        'metadata': {
            'experiment': 'Jensen Bias Correction',
            'clusters': CLUSTERS,
            'num_examples': NUM_EXAMPLES,
            'num_queries_per_example': NUM_QUERIES_PER_EXAMPLE,
            'layers': LAYERS_TO_TEST,
            'seed': SEED,
            'elapsed_seconds': elapsed,
        },
        'results': {},
        'gap_closed': {},
        'sigma_statistics': {},
    }

    for layer_name in LAYERS_TO_TEST:
        layer_out = {}
        for C in CLUSTERS:
            cluster_out = {}
            for method in METHODS:
                errs = all_errors[layer_name][C][method]
                if errs:
                    cluster_out[method] = {
                        'mean': float(np.mean(errs)),
                        'median': float(np.median(errs)),
                        'std': float(np.std(errs)),
                        'n': len(errs),
                    }
            layer_out[str(C)] = cluster_out
        json_results['results'][layer_name] = layer_out
        json_results['gap_closed'][layer_name] = {
            str(C): v for C, v in all_gap_closed[layer_name].items()
        }
        json_results['sigma_statistics'][layer_name] = {
            str(C): {
                'mean_sigma_diag': float(np.mean(all_sigmas[layer_name][C]['diag'])),
                'mean_sigma_exact': float(np.mean(all_sigmas[layer_name][C]['exact'])),
                'correlation': float(np.corrcoef(
                    all_sigmas[layer_name][C]['diag'],
                    all_sigmas[layer_name][C]['exact']
                )[0, 1]) if len(all_sigmas[layer_name][C]['diag']) > 2 else 0,
            }
            for C in CLUSTERS
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
        for C in CLUSTERS:
            print(f"\n  C = {C}")
            header = f"    {'Method':<28s} {'Mean':>8s} {'Median':>8s}  {'Gap%':>6s}"
            print(header)
            print("    " + "-" * 56)
            std_mean = np.mean(all_errors[layer_name][C]['KMeans Standard'])
            exact_mean = np.mean(all_errors[layer_name][C]['KMeans Exact Weights'])
            gap = std_mean - exact_mean
            for method in METHODS:
                errs = all_errors[layer_name][C][method]
                if errs:
                    m_mean = np.mean(errs)
                    pct = ((std_mean - m_mean) / gap * 100) if gap > 1e-8 else 0
                    print(f"    {method:<28s} {m_mean:>8.4f} {np.median(errs):>8.4f}  {pct:>5.1f}%")

        # Sigma accuracy
        print(f"\n  Sigma^2 Estimation Accuracy:")
        for C in CLUSTERS:
            diag = np.mean(all_sigmas[layer_name][C]['diag'])
            exact = np.mean(all_sigmas[layer_name][C]['exact'])
            corr_val = np.corrcoef(
                all_sigmas[layer_name][C]['diag'],
                all_sigmas[layer_name][C]['exact']
            )[0, 1] if len(all_sigmas[layer_name][C]['diag']) > 2 else 0
            print(f"    C={C:>3d}: diag={diag:.4f}, exact={exact:.4f}, "
                  f"ratio={diag/exact:.3f}, corr={corr_val:.3f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
