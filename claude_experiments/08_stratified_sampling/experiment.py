#!/usr/bin/env python3
"""
Experiment 8: Stratified Sampling (Strategy C — Within-Cluster IS)

Tests Strategy C from the paper: partition keys into clusters, then sample
within each cluster and use importance weighting for value representatives.

Theory guarantees:
  V_stratified = (1/B) * sum_c W_c * Var_{w|c}(V)
  V_oracle     = (1/B) * Var_w(V)
  V_oracle - V_stratified = (1/B) * sum_c W_c ||mu_c - o*||^2  >= 0

So stratified sampling is ALWAYS at least as good as oracle sampling
(eliminates between-cluster variance), with improvement equal to the
between-cluster variance captured by the partition.

The practical version uses centroid-softmax for cluster weights (Jensen bias)
and uniform within-cluster sampling.

Two allocation strategies tested:
  1. Proportional: b_c ∝ cluster size (more samples from bigger clusters)
  2. Uniform: b_c = B/C (equal allocation)

Compared against:
  - Oracle sampling at budget B
  - K-Means Standard (Strategy A) at C clusters
  - Exact stratified (with true W_c — oracle upper bound)

Run from: claude_experiments/08_stratified_sampling/
Results saved to: claude_experiments/08_stratified_sampling/results/
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
from algorithms.oracle import oracle_sampling
from visualization.plot_utils import setup_style, save_figure

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

NUM_EXAMPLES = 10
NUM_QUERIES_PER_EXAMPLE = 50
TOTAL_BUDGETS = [25, 50, 100, 200, 500]
CLUSTER_COUNTS = [10, 50]
NUM_TRIALS = 5
LAYERS_TO_TEST = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'attention_vectors_long_bench_llama_8b.jsonl')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')

# ============================================================================
# END HYPERPARAMETERS
# ============================================================================

LAYER_TITLES = {
    'first_layer': 'First Layer (Layer 0)',
    'last_layer': 'Last Layer (Layer 31)',
}

METHODS = ['Oracle Sampling', 'Stratified Proportional', 'Stratified Uniform',
           'Stratified Exact-Wt', 'KMeans Standard']


def precompute_cluster_info(keys, values, labels, n_clusters, head_dim):
    """Precompute per-cluster index arrays, centroids — reusable across queries."""
    cluster_indices = []
    key_centroids = np.zeros((n_clusters, head_dim))
    val_centroids = np.zeros((n_clusters, head_dim))
    counts = np.zeros(n_clusters, dtype=int)

    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        cluster_indices.append(idx)
        cnt = len(idx)
        counts[c] = cnt
        if cnt > 0:
            key_centroids[c] = keys[idx].mean(axis=0)
            val_centroids[c] = values[idx].mean(axis=0)

    return cluster_indices, key_centroids, val_centroids, counts


def allocate_budget(counts, n_clusters, budget, allocation, active_mask):
    """Compute per-cluster sample allocation."""
    active = np.where(active_mask)[0]
    n_active = len(active)
    if n_active == 0:
        return np.zeros(n_clusters, dtype=int)

    b = np.zeros(n_clusters, dtype=int)

    if allocation == 'uniform':
        base = budget // n_active
        b[active] = base
        remainder = budget - b.sum()
        for i in range(remainder):
            b[active[i % n_active]] += 1
    else:  # proportional
        total_size = counts[active].sum()
        if total_size == 0:
            b[active] = max(1, budget // n_active)
        else:
            for c in active:
                b[c] = max(1, int(round(budget * counts[c] / total_size)))
            # Adjust
            diff = budget - b.sum()
            sorted_active = active[np.argsort(-counts[active])]
            idx = 0
            while diff > 0 and idx < 10 * n_active:
                b[sorted_active[idx % n_active]] += 1
                diff -= 1
                idx += 1
            idx = 0
            while diff < 0 and idx < 10 * n_active:
                c = sorted_active[idx % n_active]
                if b[c] > 1:
                    b[c] -= 1
                    diff += 1
                idx += 1

    return b


def stratified_sampling_fast(query, keys, values, head_dim, cluster_indices,
                             key_centroids, counts, n_clusters, budget,
                             allocation, gt_weights=None,
                             use_exact_cluster_weights=False):
    """
    Optimized stratified sampling using precomputed cluster info.
    """
    sqrt_d = np.sqrt(head_dim)
    active_mask = counts > 0
    active = np.where(active_mask)[0]

    if len(active) == 0:
        return np.zeros(head_dim)

    # Cluster weights
    if use_exact_cluster_weights and gt_weights is not None:
        cluster_weights = np.zeros(n_clusters)
        for c in active:
            cluster_weights[c] = gt_weights[cluster_indices[c]].sum()
        total_w = cluster_weights.sum()
        if total_w > 0:
            cluster_weights /= total_w
    else:
        scores = (key_centroids[active] @ query) / sqrt_d
        w_active = softmax(scores)
        cluster_weights = np.zeros(n_clusters)
        cluster_weights[active] = w_active

    # Budget allocation
    b = allocate_budget(counts, n_clusters, budget, allocation, active_mask)

    # Within-cluster sampling
    output = np.zeros(head_dim)
    for c in active:
        if b[c] == 0:
            continue

        idx = cluster_indices[c]
        n_c = len(idx)

        if b[c] >= n_c:
            sampled = idx
        else:
            sampled = np.random.choice(idx, size=b[c], replace=False)

        sampled_logits = (keys[sampled] @ query) / sqrt_d
        local_weights = softmax(sampled_logits)
        v_hat_c = local_weights @ values[sampled]
        output += cluster_weights[c] * v_hat_c

    return output


def kmeans_standard(query, key_centroids, val_centroids, counts, head_dim):
    """Standard k-means segmentation (Strategy A)."""
    active = counts > 0
    if not active.any():
        return np.zeros(head_dim)
    scores = (key_centroids[active] @ query) / np.sqrt(head_dim)
    weights = softmax(scores)
    return weights @ val_centroids[active]


def evaluate_query(q, K, V, query_pos, head_dim, precomputed):
    """Evaluate all stratified variants for one query."""
    gt_output, gt_logits, gt_weights, _ = compute_ground_truth_attention(
        q, K, V, query_pos, head_dim
    )
    valid_keys = K[:query_pos + 1]
    valid_values = V[:query_pos + 1]
    nv = query_pos + 1

    results = {}

    for C in CLUSTER_COUNTS:
        labels_full = precomputed[C]['labels']
        labels = labels_full[:nv]

        # Recompute cluster info for causal subset
        ci, kc, vc, cnt = precompute_cluster_info(
            valid_keys, valid_values, labels, C, head_dim)

        # KMeans Standard (deterministic — compute once)
        out_std = kmeans_standard(q, kc, vc, cnt, head_dim)
        err_std = float(relative_l2_error(out_std, gt_output))

        for B in TOTAL_BUDGETS:
            if B >= nv:
                continue

            key = (C, B)
            trial_errors = {m: [] for m in METHODS}

            for trial in range(NUM_TRIALS):
                # Stratified Proportional
                out = stratified_sampling_fast(
                    q, valid_keys, valid_values, head_dim, ci, kc, cnt, C, B,
                    'proportional')
                trial_errors['Stratified Proportional'].append(
                    float(relative_l2_error(out, gt_output)))

                # Stratified Uniform
                out = stratified_sampling_fast(
                    q, valid_keys, valid_values, head_dim, ci, kc, cnt, C, B,
                    'uniform')
                trial_errors['Stratified Uniform'].append(
                    float(relative_l2_error(out, gt_output)))

                # Stratified with exact cluster weights
                out = stratified_sampling_fast(
                    q, valid_keys, valid_values, head_dim, ci, kc, cnt, C, B,
                    'proportional', gt_weights=gt_weights,
                    use_exact_cluster_weights=True)
                trial_errors['Stratified Exact-Wt'].append(
                    float(relative_l2_error(out, gt_output)))

                # Oracle sampling
                out_oracle, _ = oracle_sampling(
                    q, valid_keys, valid_values, gt_logits, gt_weights, B)
                trial_errors['Oracle Sampling'].append(
                    float(relative_l2_error(out_oracle, gt_output)))

            results[key] = {
                method: float(np.mean(errs)) for method, errs in trial_errors.items()
            }
            results[key]['KMeans Standard'] = err_std

    return results


def plot_results(all_errors, output_dir):
    """Error vs budget curves for each layer × cluster count."""
    fig, axes = plt.subplots(len(CLUSTER_COUNTS), len(LAYERS_TO_TEST),
                             figsize=(7 * len(LAYERS_TO_TEST), 5 * len(CLUSTER_COUNTS)))
    if len(CLUSTER_COUNTS) == 1 and len(LAYERS_TO_TEST) == 1:
        axes = np.array([[axes]])
    elif len(CLUSTER_COUNTS) == 1:
        axes = axes.reshape(1, -1)
    elif len(LAYERS_TO_TEST) == 1:
        axes = axes.reshape(-1, 1)

    method_styles = {
        'Oracle Sampling':       ('#8c564b', '--', 'o'),
        'Stratified Proportional': ('#ff7f0e', '-', 's'),
        'Stratified Uniform':    ('#2ca02c', '-', '^'),
        'Stratified Exact-Wt':   ('#e377c2', ':', 'v'),
        'KMeans Standard':       ('#1f77b4', '--', 'x'),
    }

    for row, C in enumerate(CLUSTER_COUNTS):
        for col, layer_name in enumerate(LAYERS_TO_TEST):
            ax = axes[row, col]
            for method, (color, ls, marker) in method_styles.items():
                means = []
                valid_budgets = []
                for B in TOTAL_BUDGETS:
                    key = (C, B)
                    if key in all_errors[layer_name] and method in all_errors[layer_name][key]:
                        errs = all_errors[layer_name][key][method]
                        if errs:
                            means.append(np.mean(errs))
                            valid_budgets.append(B)
                if means:
                    ax.plot(valid_budgets, means, marker=marker, linestyle=ls,
                            color=color, label=method, linewidth=1.5, markersize=5)

            ax.set_xlabel('Total Budget B', fontsize=11)
            ax.set_ylabel('Mean Relative L2 Error', fontsize=11)
            ax.set_title(f'C={C} — {LAYER_TITLES[layer_name]}', fontsize=12)
            ax.legend(fontsize=6, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')

    fig.tight_layout()
    save_figure(fig, output_dir / 'stratified_sampling.png')


def main():
    setup_style()
    np.random.seed(SEED)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 8: Stratified Sampling (Strategy C)")
    print("=" * 70)
    print(f"  Budgets:   {TOTAL_BUDGETS}")
    print(f"  Clusters:  {CLUSTER_COUNTS}")
    print(f"  Trials:    {NUM_TRIALS} per query")
    print(f"  Examples:  {NUM_EXAMPLES}")
    print(f"  Queries:   {NUM_QUERIES_PER_EXAMPLE} per example")
    print(f"  Layers:    {LAYERS_TO_TEST}")
    print(f"  Output:    {output_dir}")
    print()

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        sys.exit(1)

    all_errors = {layer: {} for layer in LAYERS_TO_TEST}

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

                # Fit k-means for each cluster count
                precomputed = {}
                for C in CLUSTER_COUNTS:
                    km = KMeans(n_clusters=C, n_init=3, max_iter=100, random_state=SEED)
                    km.fit(K)
                    precomputed[C] = {'labels': km.labels_}

                # Query positions
                min_pos = max(max(CLUSTER_COUNTS) + 1, seq_len // 4)
                max_pos = seq_len - 1
                n_queries = min(NUM_QUERIES_PER_EXAMPLE, max_pos - min_pos + 1)
                if n_queries <= 0:
                    continue
                query_positions = np.random.choice(
                    range(min_pos, max_pos + 1), size=n_queries, replace=False)

                for qpos in query_positions:
                    qr = evaluate_query(Q[qpos], K, V, qpos, HEAD_DIM, precomputed)
                    for key, method_errors in qr.items():
                        if key not in all_errors[layer_name]:
                            all_errors[layer_name][key] = {m: [] for m in METHODS}
                        for method, err in method_errors.items():
                            all_errors[layer_name][key][method].append(err)

    elapsed = time.time() - t0
    print(f"\nComputation done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Plot
    print("\nGenerating plots...")
    plot_results(all_errors, output_dir)

    # Save JSON
    json_results = {
        'metadata': {
            'experiment': 'Stratified Sampling (Strategy C)',
            'budgets': TOTAL_BUDGETS,
            'clusters': CLUSTER_COUNTS,
            'num_trials': NUM_TRIALS,
            'num_examples': NUM_EXAMPLES,
            'num_queries_per_example': NUM_QUERIES_PER_EXAMPLE,
            'layers': LAYERS_TO_TEST,
            'seed': SEED,
            'elapsed_seconds': elapsed,
        },
        'results': {},
    }

    for layer_name in LAYERS_TO_TEST:
        layer_out = {}
        for key, method_errs in all_errors[layer_name].items():
            C, B = key
            str_key = f"C{C}_B{B}"
            entry = {}
            for method in METHODS:
                errs = method_errs.get(method, [])
                if errs:
                    entry[method] = {
                        'mean': float(np.mean(errs)),
                        'median': float(np.median(errs)),
                        'std': float(np.std(errs)),
                        'n': len(errs),
                    }
            layer_out[str_key] = entry
        json_results['results'][layer_name] = layer_out

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
        for C in CLUSTER_COUNTS:
            print(f"\n  C = {C}")
            header = (f"    {'B':>5s}  {'Oracle':>8s}  {'Strat-P':>8s}  {'Strat-U':>8s}  "
                      f"{'Exact-W':>8s}  {'KM-Std':>8s}")
            print(header)
            print("    " + "-" * 52)
            for B in TOTAL_BUDGETS:
                key = (C, B)
                if key not in all_errors[layer_name]:
                    continue
                vals = []
                for m in METHODS:
                    errs = all_errors[layer_name][key].get(m, [])
                    vals.append(np.mean(errs) if errs else float('nan'))
                print(f"    {B:>5d}  {vals[0]:>8.4f}  {vals[1]:>8.4f}  {vals[2]:>8.4f}  "
                      f"{vals[3]:>8.4f}  {vals[4]:>8.4f}")

            # Improvement of stratified over oracle
            print(f"\n    Improvement of Stratified-Proportional over Oracle:")
            for B in TOTAL_BUDGETS:
                key = (C, B)
                if key not in all_errors[layer_name]:
                    continue
                oracle_errs = all_errors[layer_name][key].get('Oracle Sampling', [])
                strat_errs = all_errors[layer_name][key].get('Stratified Proportional', [])
                if oracle_errs and strat_errs:
                    o_mean = np.mean(oracle_errs)
                    s_mean = np.mean(strat_errs)
                    pct = (o_mean - s_mean) / o_mean * 100 if o_mean > 0 else 0
                    print(f"      B={B:>3d}: oracle={o_mean:.4f}, strat={s_mean:.4f}, "
                          f"improvement={pct:+.1f}%")

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
