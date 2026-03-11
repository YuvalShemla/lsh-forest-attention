#!/usr/bin/env python3
"""
Experiment 7: Random Fourier Features (FAVOR+) for Cluster Weights

Tests Strategy B from the paper: replace centroid-softmax cluster weights with
RFF-approximated unnormalized cluster masses.

FAVOR+ positive random features approximate the softmax kernel:
    exp(q^T k / sqrt(d)) ≈ phi(q_hat)^T phi(k_hat)
where:
    q_hat = q / d^{1/4},  k_hat = k / d^{1/4}
    phi(x) = (1/sqrt(m)) * exp(-||x||^2/2) * [exp(w_1^T x), ..., exp(w_m^T x)]
    w_j ~ N(0, I_d)

Per-cluster precomputation:
    Phi_c = sum_{i in S_c} phi(k_hat_i)  ∈ R^m        (for weights)
    Psi_c = sum_{i in S_c} phi(k_hat_i) v_i^T  ∈ R^{m x d}  (for values)

Per-query:
    W_c ≈ phi(q_hat)^T Phi_c                           (unnorm cluster mass)
    v_c ≈ phi(q_hat)^T Psi_c / phi(q_hat)^T Phi_c     (attn-weighted value)

The theory predicts RFF beats centroid-softmax when m > 4/sigma_c^4.
For sigma_c^2 ≈ 0.321 (last layer), this gives m* ≈ 377.

Six variants compared:
  1. KMeans Standard       — centroid-softmax weights + unweighted value means
  2. RFF Weights Only      — RFF cluster weights + unweighted value means
  3. RFF Both              — RFF cluster weights + RFF value representatives
  4. KMeans Exact Weights  — true W_c (upper bound for weight quality)
  5. GMM Standard          — soft clustering baseline
  6. Oracle Sampling       — sampling baseline at B = C

Run from: claude_experiments/07_rff_cluster_weights/
Results saved to: claude_experiments/07_rff_cluster_weights/results/
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
from algorithms.oracle import oracle_sampling
from visualization.plot_utils import setup_style, save_figure

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

NUM_EXAMPLES = 10
NUM_QUERIES_PER_EXAMPLE = 50
RFF_FEATURES = [16, 64, 256, 1024, 4096]
C_FIXED = 50                          # Fixed cluster count for the RFF sweep
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


# ─── FAVOR+ Implementation ───────────────────────────────────────────────────

def favor_plus_features(X, omega, scale_d):
    """
    Compute FAVOR+ positive random features for softmax kernel.

    For x_hat = x / d^{1/4}:
        phi(x_hat) = (1/sqrt(m)) * exp(-||x_hat||^2/2) * exp(omega @ x_hat)

    Args:
        X: [N, d] input vectors (raw, NOT prescaled)
        omega: [m, d] random projections ~ N(0, I)
        scale_d: d^{1/4} for the attention scaling

    Returns:
        features: [N, m] FAVOR+ feature vectors
    """
    X_scaled = X / scale_d                         # [N, d]
    norms_sq = np.sum(X_scaled**2, axis=1, keepdims=True)  # [N, 1]
    projections = X_scaled @ omega.T               # [N, m]

    m = omega.shape[0]
    # phi(x) = (1/sqrt(m)) * exp(proj - ||x||^2/2)
    # This avoids overflow by computing in log-space relative to max
    log_features = projections - norms_sq / 2.0
    # Stabilize per-row
    log_features_max = log_features.max(axis=1, keepdims=True)
    features = np.exp(log_features - log_features_max) / np.sqrt(m)

    # We need to track the log-scale factor for correct normalization
    # Actually, since we normalize (divide weights/values), the per-row
    # factor cancels. But across different keys it does NOT cancel.
    # So we must keep absolute features.
    features = np.exp(log_features) / np.sqrt(m)

    return features


def favor_plus_features_stable(X, omega, scale_d):
    """
    Numerically stable FAVOR+ features.

    Uses the log-sum-exp trick per sample to avoid overflow.
    Returns features and a per-sample log-scale factor.

    phi_i = (1/sqrt(m)) * exp(omega @ x_i/s - ||x_i/s||^2/2)

    We compute: log_phi_ij = omega_j @ x_i/s - ||x_i/s||^2/2 - log(sqrt(m))
    Then: phi_i = exp(log_phi_i - max_j(log_phi_ij)) * exp(max_j(log_phi_ij))
    """
    m = omega.shape[0]
    X_scaled = X / scale_d
    norms_sq = np.sum(X_scaled**2, axis=1, keepdims=True)
    projections = X_scaled @ omega.T

    log_phi = projections - norms_sq / 2.0 - 0.5 * np.log(m)
    # Per-row stabilization
    log_phi_max = log_phi.max(axis=1, keepdims=True)
    phi = np.exp(log_phi - log_phi_max)

    return phi, log_phi_max.ravel()


def precompute_rff_clusters(keys, values, labels, n_clusters, omega, scale_d):
    """
    Precompute per-cluster RFF aggregates.

    Returns:
        Phi_c: [C, m] — sum of key features per cluster
        Psi_c: [C, m, d] — sum of (key_feature * value^T) per cluster
        val_centroids: [C, d] — unweighted value means (for hybrid)
        key_centroids: [C, d] — unweighted key means (for standard)
        log_scales: per-cluster log normalization info
    """
    m = omega.shape[0]
    d = keys.shape[1]

    # Compute features for all keys
    # For stability, we compute in log space and aggregate carefully
    X_scaled = keys / scale_d
    norms_sq = np.sum(X_scaled**2, axis=1, keepdims=True)
    projections = X_scaled @ omega.T  # [N, m]
    log_phi = projections - norms_sq / 2.0  # [N, m] (without 1/sqrt(m))

    Phi_c = np.zeros((n_clusters, m))
    Psi_c = np.zeros((n_clusters, m, d))
    val_centroids = np.zeros((n_clusters, d))
    key_centroids = np.zeros((n_clusters, d))
    counts = np.zeros(n_clusters)

    for c in range(n_clusters):
        mask = labels == c
        cnt = mask.sum()
        if cnt == 0:
            continue
        counts[c] = cnt

        key_centroids[c] = keys[mask].mean(axis=0)
        val_centroids[c] = values[mask].mean(axis=0)

        # Stable aggregation of features
        log_phi_c = log_phi[mask]  # [cnt, m]
        # Max over all entries for stability
        global_max = log_phi_c.max()
        phi_c = np.exp(log_phi_c - global_max) / np.sqrt(m)

        Phi_c[c] = phi_c.sum(axis=0) * np.exp(global_max)

        # Psi_c = sum_i phi(k_i) v_i^T  →  [m, d]
        vals_c = values[mask]
        weighted_phi = phi_c * np.exp(global_max)  # [cnt, m]
        Psi_c[c] = weighted_phi.T @ vals_c          # [m, d]

    return Phi_c, Psi_c, val_centroids, key_centroids, counts


def rff_query_features(query, omega, scale_d):
    """Compute FAVOR+ features for a single query vector."""
    m = omega.shape[0]
    q_scaled = query / scale_d
    norm_sq = np.sum(q_scaled**2)
    projections = omega @ q_scaled  # [m]
    log_phi = projections - norm_sq / 2.0
    # Stabilize
    log_max = log_phi.max()
    phi_q = np.exp(log_phi - log_max) / np.sqrt(m)
    return phi_q, log_max


def rff_weights_attention(query, Phi_c, val_centroids, counts, omega, scale_d):
    """
    RFF cluster weights + standard value centroids.
    W_c ≈ phi(q)^T Phi_c, then softmax-normalize.
    """
    phi_q, log_q_max = rff_query_features(query, omega, scale_d)

    active = counts > 0
    if not active.any():
        return np.zeros(val_centroids.shape[1])

    # Unnormalized cluster masses (the log_q_max cancels in normalization)
    unnorm = phi_q @ Phi_c[active].T  # [n_active]

    # Handle all-zero case
    if unnorm.max() <= 0:
        return val_centroids[active].mean(axis=0)

    weights = unnorm / unnorm.sum()
    return weights @ val_centroids[active]


def rff_both_attention(query, Phi_c, Psi_c, counts, omega, scale_d):
    """
    RFF cluster weights AND RFF value representatives.
    W_c ≈ phi(q)^T Phi_c
    v_c ≈ phi(q)^T Psi_c / phi(q)^T Phi_c  (attention-weighted mean!)
    """
    phi_q, log_q_max = rff_query_features(query, omega, scale_d)
    d = Psi_c.shape[2]

    active = counts > 0
    if not active.any():
        return np.zeros(d)

    # Unnormalized cluster masses
    unnorm = phi_q @ Phi_c[active].T  # [n_active]

    if unnorm.max() <= 0:
        return np.zeros(d)

    # RFF value representatives: phi(q)^T Psi_c / phi(q)^T Phi_c
    n_active = active.sum()
    rff_values = np.zeros((n_active, d))
    for i, c_idx in enumerate(np.where(active)[0]):
        num = phi_q @ Psi_c[c_idx]  # [d]
        denom = unnorm[i]
        if denom > 1e-12:
            rff_values[i] = num / denom
        else:
            rff_values[i] = 0

    weights = unnorm / unnorm.sum()
    return weights @ rff_values


def kmeans_standard_attention(query, key_centroids, val_centroids, counts, head_dim):
    """Standard k-means: softmax over centroid logits."""
    active = counts > 0
    if not active.any():
        return np.zeros(head_dim)
    scores = (key_centroids[active] @ query) / np.sqrt(head_dim)
    weights = softmax(scores)
    return weights @ val_centroids[active]


def kmeans_exact_weights_attention(query, keys, values, labels, true_weights,
                                   n_clusters, head_dim):
    """K-means with true attention weights (oracle upper bound)."""
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
    """Evaluate all RFF variants for one query."""
    gt_output, gt_logits, gt_weights, _ = compute_ground_truth_attention(
        q, K, V, query_pos, head_dim
    )
    valid_keys = K[:query_pos + 1]
    valid_values = V[:query_pos + 1]
    nv = query_pos + 1

    scale_d = head_dim ** 0.25
    labels = precomputed['labels'][:nv]
    gmm_resp = precomputed['gmm_resp'][:nv]

    results = {}

    for m_rff in RFF_FEATURES:
        omega = precomputed['omegas'][m_rff]

        # Precompute cluster-level RFF aggregates for valid keys
        Phi_c, Psi_c, val_cents, key_cents, counts = precompute_rff_clusters(
            valid_keys, valid_values, labels, C_FIXED, omega, scale_d)

        # 1. Standard k-means
        out_std = kmeans_standard_attention(q, key_cents, val_cents, counts, head_dim)

        # 2. RFF weights only
        out_rff_w = rff_weights_attention(q, Phi_c, val_cents, counts, omega, scale_d)

        # 3. RFF both (weights + values)
        out_rff_b = rff_both_attention(q, Phi_c, Psi_c, counts, omega, scale_d)

        results[m_rff] = {
            'KMeans Standard':      float(relative_l2_error(out_std, gt_output)),
            'RFF Weights Only':     float(relative_l2_error(out_rff_w, gt_output)),
            'RFF Both':             float(relative_l2_error(out_rff_b, gt_output)),
        }

    # Methods that don't depend on m
    # Exact weights (compute once using full valid keys)
    Phi_c0, Psi_c0, val_cents0, key_cents0, counts0 = precompute_rff_clusters(
        valid_keys, valid_values, labels, C_FIXED,
        precomputed['omegas'][RFF_FEATURES[0]], scale_d)
    out_exact = kmeans_exact_weights_attention(
        q, valid_keys, valid_values, labels, gt_weights, C_FIXED, head_dim)
    out_gmm, _ = gmm_attention(q, valid_keys, valid_values, gt_logits, head_dim, gmm_resp)
    out_oracle, _ = oracle_sampling(q, valid_keys, valid_values, gt_logits, gt_weights, C_FIXED)

    for m_rff in RFF_FEATURES:
        results[m_rff]['KMeans Exact Weights'] = float(relative_l2_error(out_exact, gt_output))
        results[m_rff]['GMM Standard'] = float(relative_l2_error(out_gmm, gt_output))
        results[m_rff]['Oracle Sampling'] = float(relative_l2_error(out_oracle, gt_output))

    return results


def plot_rff_sweep(all_errors, output_dir):
    """Error vs m_rff, with baselines as horizontal lines."""
    fig, axes = plt.subplots(1, len(LAYERS_TO_TEST),
                             figsize=(7 * len(LAYERS_TO_TEST), 5))
    if len(LAYERS_TO_TEST) == 1:
        axes = [axes]

    for ax, layer_name in zip(axes, LAYERS_TO_TEST):
        # RFF methods vs m
        for method, color, marker in [
            ('RFF Weights Only', '#ff7f0e', 'o'),
            ('RFF Both', '#2ca02c', 's'),
        ]:
            means = []
            stds = []
            for m_rff in RFF_FEATURES:
                errs = all_errors[layer_name][m_rff][method]
                means.append(np.mean(errs))
                stds.append(np.std(errs) / np.sqrt(len(errs)))
            ax.errorbar(RFF_FEATURES, means, yerr=stds, marker=marker,
                        label=method, color=color, linewidth=2, capsize=3)

        # Baselines as horizontal lines
        for method, color, ls in [
            ('KMeans Standard', '#1f77b4', '--'),
            ('KMeans Exact Weights', '#9467bd', ':'),
            ('GMM Standard', '#d62728', '-.'),
            ('Oracle Sampling', '#8c564b', '--'),
        ]:
            errs = all_errors[layer_name][RFF_FEATURES[0]][method]
            ax.axhline(np.mean(errs), color=color, linestyle=ls, label=method, alpha=0.7)

        ax.set_xscale('log')
        ax.set_xlabel('Number of Random Features (m)', fontsize=12)
        ax.set_ylabel('Mean Relative L2 Error', fontsize=12)
        ax.set_title(f'RFF Cluster Weights — {LAYER_TITLES[layer_name]}', fontsize=13)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, output_dir / 'rff_sweep.png')


def main():
    setup_style()
    np.random.seed(SEED)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 7: Random Fourier Features (FAVOR+) for Cluster Weights")
    print("=" * 70)
    print(f"  RFF features: {RFF_FEATURES}")
    print(f"  Clusters:     C = {C_FIXED}")
    print(f"  Examples:     {NUM_EXAMPLES}")
    print(f"  Queries:      {NUM_QUERIES_PER_EXAMPLE} per example")
    print(f"  Layers:       {LAYERS_TO_TEST}")
    print(f"  Output:       {output_dir}")
    print()

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        sys.exit(1)

    ALL_METHODS = ['KMeans Standard', 'RFF Weights Only', 'RFF Both',
                   'KMeans Exact Weights', 'GMM Standard', 'Oracle Sampling']

    all_errors = {
        layer: {m: {method: [] for method in ALL_METHODS} for m in RFF_FEATURES}
        for layer in LAYERS_TO_TEST
    }

    # Pre-generate random projections for each m (shared across examples)
    omegas = {}
    rng = np.random.RandomState(SEED)
    for m_rff in RFF_FEATURES:
        omegas[m_rff] = rng.randn(m_rff, HEAD_DIM).astype(np.float32)

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

                # Fit k-means and GMM once
                km = KMeans(n_clusters=C_FIXED, n_init=3, max_iter=100, random_state=SEED)
                km.fit(K)
                labels = km.labels_
                gmm_resp = fit_gmm(K, C_FIXED, seed=SEED)

                precomputed = {
                    'labels': labels,
                    'gmm_resp': gmm_resp,
                    'omegas': omegas,
                }

                # Query positions
                min_pos = max(C_FIXED + 1, seq_len // 4)
                max_pos = seq_len - 1
                n_queries = min(NUM_QUERIES_PER_EXAMPLE, max_pos - min_pos + 1)
                if n_queries <= 0:
                    continue
                query_positions = np.random.choice(
                    range(min_pos, max_pos + 1), size=n_queries, replace=False)

                for qpos in query_positions:
                    qr = evaluate_query(Q[qpos], K, V, qpos, HEAD_DIM, precomputed)
                    for m_rff, method_errors in qr.items():
                        for method, err in method_errors.items():
                            all_errors[layer_name][m_rff][method].append(err)

    elapsed = time.time() - t0
    print(f"\nComputation done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Plot
    print("\nGenerating plots...")
    plot_rff_sweep(all_errors, output_dir)

    # Save JSON
    json_results = {
        'metadata': {
            'experiment': 'RFF Cluster Weights (FAVOR+)',
            'rff_features': RFF_FEATURES,
            'clusters': C_FIXED,
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
        for m_rff in RFF_FEATURES:
            m_out = {}
            for method in ALL_METHODS:
                errs = all_errors[layer_name][m_rff][method]
                if errs:
                    m_out[method] = {
                        'mean': float(np.mean(errs)),
                        'median': float(np.median(errs)),
                        'std': float(np.std(errs)),
                        'n': len(errs),
                    }
            layer_out[str(m_rff)] = m_out
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
        header = f"  {'m':>6s}  {'Standard':>10s}  {'RFF-Wt':>10s}  {'RFF-Both':>10s}  {'Exact-Wt':>10s}  {'GMM':>10s}  {'Oracle':>10s}"
        print(header)
        print("  " + "-" * 76)
        for m_rff in RFF_FEATURES:
            vals = []
            for method in ALL_METHODS:
                errs = all_errors[layer_name][m_rff][method]
                vals.append(np.mean(errs) if errs else 0)
            print(f"  {m_rff:>6d}  {vals[0]:>10.4f}  {vals[1]:>10.4f}  {vals[2]:>10.4f}  "
                  f"{vals[3]:>10.4f}  {vals[4]:>10.4f}  {vals[5]:>10.4f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
