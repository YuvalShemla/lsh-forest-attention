#!/usr/bin/env python3
"""
Experiment 4: K-Means vs GMM Partition for Sparse Attention

Tests whether the non-monotonicity (error increasing at large C) observed with
GMM soft clustering is a fitting artifact by comparing with k-means hard assignment.

K-means uses one-hot responsibility matrices fed into the same gmm_attention function,
so the only difference is hard vs soft cluster assignment.

Compares across C = [10, 20, 50, 100, 200, 500] clusters, both layers.

Key question: Does k-means error monotonically decrease as C increases,
or does it also show the U-shape seen with GMM?

Results saved to: claude_experiments/04_kmeans_vs_gmm/results/
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture

from algorithms.base import compute_ground_truth_attention, relative_l2_error
from algorithms.gmm_attention import gmm_attention
from visualization.plot_utils import setup_style, save_figure

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

NUM_EXAMPLES = 10
NUM_QUERIES_PER_EXAMPLE = 50
CLUSTER_COUNTS = [10, 20, 50, 100, 200, 500]
LAYERS_TO_TEST = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data',
                         'attention_vectors_long_bench_llama_8b.jsonl')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# Subsample keys for GMM fitting to speed up (predict_proba on all keys after)
GMM_FIT_SUBSAMPLE = 2000

# ============================================================================
# END HYPERPARAMETERS
# ============================================================================

METHODS = ['GMM', 'K-Means']

METHOD_COLORS = {
    'GMM': '#1f77b4',
    'K-Means': '#ff7f0e',
}

LAYER_TITLES = {
    'first_layer': 'First Layer (Layer 0)',
    'last_layer': 'Last Layer (Layer 31)',
}


def fit_gmm_fast(keys, n_clusters, seed=42, subsample=GMM_FIT_SUBSAMPLE):
    """Fit GMM on (possibly subsampled) keys, return responsibilities for ALL keys.

    For large key sets, fits on a random subsample then uses predict_proba
    on the full set. This is much faster than fitting on all keys.
    """
    n_keys = len(keys)
    if n_clusters >= n_keys:
        return np.eye(n_keys, n_clusters)

    if n_clusters == 1:
        return np.ones((n_keys, 1))

    rng = np.random.RandomState(seed)

    # Subsample for fitting if needed
    if n_keys > subsample and subsample > 0:
        fit_idx = rng.choice(n_keys, size=subsample, replace=False)
        fit_keys = keys[fit_idx]
    else:
        fit_keys = keys

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='diag',
        max_iter=50,
        n_init=1,
        random_state=seed,
    )
    gmm.fit(fit_keys)
    # Predict on ALL keys
    resp = gmm.predict_proba(keys)
    return resp


def fit_kmeans(keys, n_clusters, seed=42):
    """Fit k-means and return one-hot responsibility matrix.

    Uses MiniBatchKMeans for speed.
    """
    n_keys = len(keys)
    if n_clusters >= n_keys:
        return np.eye(n_keys, n_clusters)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed,
        n_init=3, max_iter=100, batch_size=min(1024, n_keys))
    labels = kmeans.fit_predict(keys)

    # One-hot encoding
    resp = np.zeros((n_keys, n_clusters))
    resp[np.arange(n_keys), labels] = 1.0
    return resp


def evaluate_query(q, K, V, query_pos, head_dim, resp_gmm_dict, resp_km_dict):
    """Evaluate GMM and K-Means for one query across all cluster counts."""
    gt_output, gt_logits, gt_weights, _ = compute_ground_truth_attention(
        q, K, V, query_pos, head_dim
    )
    valid_keys = K[:query_pos + 1]
    valid_values = V[:query_pos + 1]
    nv = len(valid_keys)

    results = {}
    for C in CLUSTER_COUNTS:
        if C >= nv:
            continue

        errors = {}

        # GMM
        if C in resp_gmm_dict and resp_gmm_dict[C] is not None:
            resp_gmm = resp_gmm_dict[C][:nv]
            out_gmm, _ = gmm_attention(
                q, valid_keys, valid_values, gt_logits, head_dim, resp_gmm)
            errors['GMM'] = float(relative_l2_error(out_gmm, gt_output))
        else:
            errors['GMM'] = None

        # K-Means
        if C in resp_km_dict and resp_km_dict[C] is not None:
            resp_km = resp_km_dict[C][:nv]
            out_km, _ = gmm_attention(
                q, valid_keys, valid_values, gt_logits, head_dim, resp_km)
            errors['K-Means'] = float(relative_l2_error(out_km, gt_output))
        else:
            errors['K-Means'] = None

        results[C] = errors

    return results


def plot_comparison(all_errors, output_dir):
    """Line plot: mean error vs C for GMM and K-Means, one subplot per layer."""
    fig, axes = plt.subplots(1, len(LAYERS_TO_TEST), figsize=(7 * len(LAYERS_TO_TEST), 6))
    if len(LAYERS_TO_TEST) == 1:
        axes = [axes]

    for ax, layer_name in zip(axes, LAYERS_TO_TEST):
        for method in METHODS:
            means = []
            stds = []
            valid_Cs = []
            for C in CLUSTER_COUNTS:
                errs = all_errors[layer_name][C][method]
                if errs:
                    means.append(np.mean(errs))
                    stds.append(np.std(errs))
                    valid_Cs.append(C)

            means = np.array(means)
            stds = np.array(stds)
            valid_Cs = np.array(valid_Cs)

            ax.plot(valid_Cs, means, 'o-', color=METHOD_COLORS[method],
                    label=method, linewidth=2, markersize=6)
            ax.fill_between(valid_Cs, means - stds, means + stds,
                            alpha=0.15, color=METHOD_COLORS[method])

        ax.set_xlabel('Number of Clusters (C)', fontsize=12)
        ax.set_ylabel('Mean Relative L2 Error', fontsize=12)
        ax.set_title(f'K-Means vs GMM -- {LAYER_TITLES[layer_name]}', fontsize=13)
        ax.set_xscale('log')
        ax.set_xticks(CLUSTER_COUNTS)
        ax.set_xticklabels([str(c) for c in CLUSTER_COUNTS])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, output_dir / 'kmeans_vs_gmm_comparison.png')


def plot_per_example(per_example_errors, output_dir):
    """Plot per-example error curves to see consistency."""
    for layer_name in LAYERS_TO_TEST:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, method in zip(axes, METHODS):
            for ex_idx, example_data in enumerate(per_example_errors[layer_name]):
                means = []
                valid_Cs = []
                for C in CLUSTER_COUNTS:
                    errs = example_data.get(C, {}).get(method, [])
                    if errs:
                        means.append(np.mean(errs))
                        valid_Cs.append(C)
                if valid_Cs:
                    ax.plot(valid_Cs, means, 'o-', alpha=0.4, linewidth=1, markersize=3)

            ax.set_xlabel('Number of Clusters (C)', fontsize=12)
            ax.set_ylabel('Mean Relative L2 Error', fontsize=12)
            ax.set_title(f'{method} -- {LAYER_TITLES[layer_name]}', fontsize=13)
            ax.set_xscale('log')
            ax.set_xticks(CLUSTER_COUNTS)
            ax.set_xticklabels([str(c) for c in CLUSTER_COUNTS])
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        save_figure(fig, output_dir / f'per_example_{layer_name}.png')


def plot_ratio(all_errors, output_dir):
    """Plot the ratio of K-Means error / GMM error at each C."""
    fig, axes = plt.subplots(1, len(LAYERS_TO_TEST), figsize=(7 * len(LAYERS_TO_TEST), 6))
    if len(LAYERS_TO_TEST) == 1:
        axes = [axes]

    for ax, layer_name in zip(axes, LAYERS_TO_TEST):
        ratios_mean = []
        ratios_median = []
        valid_Cs = []
        for C in CLUSTER_COUNTS:
            gmm_errs = all_errors[layer_name][C]['GMM']
            km_errs = all_errors[layer_name][C]['K-Means']
            if gmm_errs and km_errs and len(gmm_errs) == len(km_errs):
                paired_ratios = [km / (gmm + 1e-10) for km, gmm in zip(km_errs, gmm_errs)]
                ratios_mean.append(np.mean(paired_ratios))
                ratios_median.append(np.median(paired_ratios))
                valid_Cs.append(C)

        ax.plot(valid_Cs, ratios_mean, 'o-', color='#2ca02c', label='Mean Ratio', linewidth=2)
        ax.plot(valid_Cs, ratios_median, 's--', color='#9467bd', label='Median Ratio', linewidth=2)
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Equal performance')
        ax.set_xlabel('Number of Clusters (C)', fontsize=12)
        ax.set_ylabel('K-Means Error / GMM Error', fontsize=12)
        ax.set_title(f'Error Ratio -- {LAYER_TITLES[layer_name]}', fontsize=13)
        ax.set_xscale('log')
        ax.set_xticks(CLUSTER_COUNTS)
        ax.set_xticklabels([str(c) for c in CLUSTER_COUNTS])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, output_dir / 'error_ratio.png')


def main():
    setup_style()
    np.random.seed(SEED)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(DATA_PATH)
    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        sys.exit(1)

    print("=" * 70)
    print("Experiment 4: K-Means vs GMM Partition")
    print("=" * 70)
    print(f"  Methods:  {METHODS}")
    print(f"  Clusters: {CLUSTER_COUNTS}")
    print(f"  Examples: {NUM_EXAMPLES}")
    print(f"  Queries:  {NUM_QUERIES_PER_EXAMPLE} per example")
    print(f"  Layers:   {LAYERS_TO_TEST}")
    print(f"  GMM fit subsample: {GMM_FIT_SUBSAMPLE}")
    print(f"  Output:   {output_dir}")
    print()

    # Collect errors: {layer: {C: {method: [errors]}}}
    all_errors = {
        layer: {C: {m: [] for m in METHODS} for C in CLUSTER_COUNTS}
        for layer in LAYERS_TO_TEST
    }

    # Per-example tracking
    per_example_errors = {layer: [] for layer in LAYERS_TO_TEST}

    t0 = time.time()
    total_queries = 0
    skipped_fits = {'GMM': 0, 'K-Means': 0}

    with open(data_path, 'r') as f:
        for ex_idx, line in enumerate(f):
            if ex_idx >= NUM_EXAMPLES:
                break

            example = json.loads(line)
            seq_len = example['sequence_length']
            domain = example.get('domain', '?')[:30]
            print(f"\n  [{ex_idx+1:3d}/{NUM_EXAMPLES}] {domain:<30s} (seq_len={seq_len})")

            for layer_name in LAYERS_TO_TEST:
                layer_data = example[layer_name]
                Q = np.array(layer_data['Q'], dtype=np.float32)
                K_mat = np.array(layer_data['K'], dtype=np.float32)
                V_mat = np.array(layer_data['V'], dtype=np.float32)

                # Fit models once per cluster count on ALL keys
                resp_gmm_dict = {}
                resp_km_dict = {}

                for C in CLUSTER_COUNTS:
                    if C >= seq_len:
                        resp_gmm_dict[C] = None
                        resp_km_dict[C] = None
                        print(f"    Skipping C={C} (>= seq_len={seq_len})")
                        continue

                    # Fit GMM (subsampled for speed)
                    t_fit = time.time()
                    try:
                        resp_gmm_dict[C] = fit_gmm_fast(K_mat, C, seed=SEED)
                        gmm_time = time.time() - t_fit
                    except Exception as e:
                        print(f"    GMM fit failed for C={C}: {e}")
                        resp_gmm_dict[C] = None
                        skipped_fits['GMM'] += 1
                        gmm_time = time.time() - t_fit

                    # Fit K-Means (MiniBatchKMeans for speed)
                    t_fit = time.time()
                    try:
                        resp_km_dict[C] = fit_kmeans(K_mat, C, seed=SEED)
                        km_time = time.time() - t_fit
                    except Exception as e:
                        print(f"    K-Means fit failed for C={C}: {e}")
                        resp_km_dict[C] = None
                        skipped_fits['K-Means'] += 1
                        km_time = time.time() - t_fit

                    print(f"    {layer_name} C={C:4d}: GMM={gmm_time:.1f}s, KMeans={km_time:.1f}s")

                # Pick query positions (second half of sequence)
                min_pos = max(max(CLUSTER_COUNTS) + 1, seq_len // 2)
                max_pos = seq_len - 1
                n_queries = min(NUM_QUERIES_PER_EXAMPLE, max_pos - min_pos + 1)
                if n_queries <= 0:
                    print(f"    No valid query positions for {layer_name}")
                    per_example_errors[layer_name].append(
                        {C: {m: [] for m in METHODS} for C in CLUSTER_COUNTS})
                    continue

                query_positions = np.random.choice(
                    range(min_pos, max_pos + 1), size=n_queries, replace=False)

                example_errors = {C: {m: [] for m in METHODS} for C in CLUSTER_COUNTS}

                for qpos in query_positions:
                    qr = evaluate_query(
                        Q[qpos], K_mat, V_mat, qpos, HEAD_DIM,
                        resp_gmm_dict, resp_km_dict)
                    total_queries += 1

                    for C, method_errors in qr.items():
                        for method, err in method_errors.items():
                            if err is not None:
                                all_errors[layer_name][C][method].append(err)
                                example_errors[C][method].append(err)

                per_example_errors[layer_name].append(example_errors)

    elapsed = time.time() - t0
    print(f"\nComputation done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Total queries evaluated: {total_queries}")
    if any(v > 0 for v in skipped_fits.values()):
        print(f"Skipped fits: {skipped_fits}")

    # ========================================================================
    # Plots
    # ========================================================================
    print("\nGenerating plots...")
    plot_comparison(all_errors, output_dir)
    plot_per_example(per_example_errors, output_dir)
    plot_ratio(all_errors, output_dir)

    # ========================================================================
    # Save JSON
    # ========================================================================
    json_results = {
        'metadata': {
            'experiment': 'K-Means vs GMM Partition',
            'cluster_counts': CLUSTER_COUNTS,
            'num_examples': NUM_EXAMPLES,
            'num_queries_per_example': NUM_QUERIES_PER_EXAMPLE,
            'layers': LAYERS_TO_TEST,
            'seed': SEED,
            'gmm_fit_subsample': GMM_FIT_SUBSAMPLE,
            'elapsed_seconds': elapsed,
            'skipped_fits': skipped_fits,
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
                        'mean': float(np.mean(errs)),
                        'median': float(np.median(errs)),
                        'std': float(np.std(errs)),
                        'p25': float(np.percentile(errs, 25)),
                        'p75': float(np.percentile(errs, 75)),
                        'min': float(np.min(errs)),
                        'max': float(np.max(errs)),
                        'n': len(errs),
                    }
            layer_out[str(C)] = cluster_out
        json_results['results'][layer_name] = layer_out

    # Monotonicity analysis
    monotonicity = {}
    for layer_name in LAYERS_TO_TEST:
        layer_mono = {}
        for method in METHODS:
            means = []
            valid_Cs = []
            for C in CLUSTER_COUNTS:
                errs = all_errors[layer_name][C][method]
                if errs:
                    means.append(float(np.mean(errs)))
                    valid_Cs.append(C)
            if len(means) >= 2:
                diffs = [means[i+1] - means[i] for i in range(len(means)-1)]
                is_monotone_decreasing = all(d <= 0 for d in diffs)
                best_idx = np.argmin(means)
                layer_mono[method] = {
                    'means_by_C': dict(zip([str(c) for c in valid_Cs], means)),
                    'is_monotone_decreasing': is_monotone_decreasing,
                    'best_C': valid_Cs[best_idx],
                    'best_mean_error': means[best_idx],
                    'diffs': dict(zip(
                        [f'{valid_Cs[i]}->{valid_Cs[i+1]}' for i in range(len(valid_Cs)-1)],
                        diffs
                    )),
                }
        monotonicity[layer_name] = layer_mono
    json_results['monotonicity_analysis'] = monotonicity

    json_path = output_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved: {json_path}")

    # ========================================================================
    # Summary table
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Mean Relative L2 Error")
    print("=" * 70)

    for layer_name in LAYERS_TO_TEST:
        print(f"\n{LAYER_TITLES[layer_name]}")
        print(f"{'C':>6s}  {'GMM Mean':>12s}  {'KM Mean':>12s}  {'GMM Median':>12s}  {'KM Median':>12s}  {'Winner':>10s}")
        print("  " + "-" * 76)

        for C in CLUSTER_COUNTS:
            gmm_errs = all_errors[layer_name][C]['GMM']
            km_errs = all_errors[layer_name][C]['K-Means']

            if gmm_errs and km_errs:
                gmm_mean = np.mean(gmm_errs)
                km_mean = np.mean(km_errs)
                gmm_med = np.median(gmm_errs)
                km_med = np.median(km_errs)
                winner = 'GMM' if gmm_mean < km_mean else 'K-Means'
                print(f"{C:6d}  {gmm_mean:12.6f}  {km_mean:12.6f}  "
                      f"{gmm_med:12.6f}  {km_med:12.6f}  {winner:>10s}")
            elif gmm_errs:
                gmm_mean = np.mean(gmm_errs)
                gmm_med = np.median(gmm_errs)
                print(f"{C:6d}  {gmm_mean:12.6f}  {'N/A':>12s}  "
                      f"{gmm_med:12.6f}  {'N/A':>12s}  {'GMM':>10s}")
            elif km_errs:
                km_mean = np.mean(km_errs)
                km_med = np.median(km_errs)
                print(f"{C:6d}  {'N/A':>12s}  {km_mean:12.6f}  "
                      f"{'N/A':>12s}  {km_med:12.6f}  {'K-Means':>10s}")
            else:
                print(f"{C:6d}  {'N/A':>12s}  {'N/A':>12s}  "
                      f"{'N/A':>12s}  {'N/A':>12s}  {'N/A':>10s}")

    # Monotonicity analysis summary
    print("\n" + "=" * 70)
    print("MONOTONICITY ANALYSIS")
    print("=" * 70)

    for layer_name in LAYERS_TO_TEST:
        print(f"\n{LAYER_TITLES[layer_name]}")
        for method in METHODS:
            mono = monotonicity.get(layer_name, {}).get(method, {})
            if mono:
                is_mono = mono['is_monotone_decreasing']
                best_C = mono['best_C']
                best_err = mono['best_mean_error']
                print(f"  {method:>10s}: monotone_decreasing={is_mono}, "
                      f"best_C={best_C}, best_error={best_err:.6f}")
                if not is_mono:
                    for k, v in mono['diffs'].items():
                        if v > 0:
                            print(f"             Error INCREASES at {k}: +{v:.6f}")

    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()
