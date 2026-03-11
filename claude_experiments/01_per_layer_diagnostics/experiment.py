#!/usr/bin/env python3
"""
Experiment 1: Per-Layer Diagnostics

Measures diagnostic statistics per layer (first_layer vs last_layer) to understand
why segmentation (GMM clustering) wins at last layer but loses at first layer.

Metrics per layer per query:
  1. Attention entropy H(w)
  2. Within-cluster logit variance (GMM clusters)
  3. Key-value cosine correlation
  4. Value matrix effective rank (via SVD)
  5. Max attention weight (peakedness)
  6. GMM attention relative L2 error (C=50)

Run from: claude_experiments/01_per_layer_diagnostics/
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src/ to path so we can import algorithms
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from algorithms import (
    compute_ground_truth_attention,
    relative_l2_error,
    fit_gmm,
    gmm_attention,
    softmax,
)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

NUM_EXAMPLES = 10
NUM_QUERIES_PER_EXAMPLE = 50
GMM_CLUSTERS = 50
HEAD_DIM = 128
SEED = 42
LAYERS = ['first_layer', 'last_layer']

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'data', 'attention_vectors_long_bench_llama_8b.jsonl'
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================


def attention_entropy(weights):
    """Compute Shannon entropy H(w) = -sum(w * log(w)), handling zeros."""
    w = weights[weights > 0]
    return -np.sum(w * np.log(w))


def max_attention_weight(weights):
    """Max attention weight — measures peakedness."""
    return float(np.max(weights))


def within_cluster_logit_variance(logits, resp):
    """
    Compute average within-cluster logit variance, weighted by cluster mass.

    For each cluster c:
      - cluster mass m_c = sum of responsibilities for cluster c
      - weighted mean logit mu_c = (resp[:, c] * logits).sum() / m_c
      - weighted variance sigma_c^2 = (resp[:, c] * (logits - mu_c)^2).sum() / m_c

    Returns: sum_c(m_c * sigma_c^2) / sum_c(m_c)   (mass-weighted average)
    """
    n_clusters = resp.shape[1]
    cluster_mass = resp.sum(axis=0)  # [C]
    total_mass = cluster_mass.sum()

    if total_mass < 1e-12:
        return 0.0

    weighted_var_sum = 0.0
    for c in range(n_clusters):
        mc = cluster_mass[c]
        if mc < 1e-12:
            continue
        # Weighted mean logit for cluster c
        mu_c = np.dot(resp[:, c], logits) / mc
        # Weighted variance
        sigma_c_sq = np.dot(resp[:, c], (logits - mu_c) ** 2) / mc
        weighted_var_sum += mc * sigma_c_sq

    return float(weighted_var_sum / total_mass)


def key_value_cosine_correlation(keys, values):
    """
    Average cosine similarity between k_i and v_i.

    Returns: mean(cos(k_i, v_i)) across all positions.
    """
    k_norms = np.linalg.norm(keys, axis=1, keepdims=True)
    v_norms = np.linalg.norm(values, axis=1, keepdims=True)

    # Avoid division by zero
    k_norms = np.maximum(k_norms, 1e-8)
    v_norms = np.maximum(v_norms, 1e-8)

    cos_sims = np.sum((keys / k_norms) * (values / v_norms), axis=1)
    return float(np.mean(cos_sims))


def value_effective_rank(values):
    """
    Effective rank of the value matrix via SVD.

    effective_rank = (sum of singular values)^2 / (sum of singular values^2)

    This measures how many dimensions are "active" in V.
    """
    _, s, _ = np.linalg.svd(values, full_matrices=False)
    s_sum = np.sum(s)
    s_sq_sum = np.sum(s ** 2)

    if s_sq_sum < 1e-12:
        return 0.0

    return float(s_sum ** 2 / s_sq_sum)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================


def run_experiment():
    """Run per-layer diagnostics experiment."""

    print("=" * 70)
    print("EXPERIMENT 1: PER-LAYER DIAGNOSTICS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  NUM_EXAMPLES = {NUM_EXAMPLES}")
    print(f"  NUM_QUERIES_PER_EXAMPLE = {NUM_QUERIES_PER_EXAMPLE}")
    print(f"  GMM_CLUSTERS = {GMM_CLUSTERS}")
    print(f"  HEAD_DIM = {HEAD_DIM}")
    print(f"  SEED = {SEED}")
    print(f"  Layers: {LAYERS}")
    print(f"  Data: {DATA_PATH}")

    np.random.seed(SEED)

    # Storage for all per-query results
    results = {layer: [] for layer in LAYERS}

    # Load and process examples line-by-line
    print(f"\nProcessing {NUM_EXAMPLES} examples...")
    total_start = time.time()

    with open(DATA_PATH, 'r') as f:
        for ex_idx, line in enumerate(f):
            if ex_idx >= NUM_EXAMPLES:
                break

            example = json.loads(line)
            seq_len = example['sequence_length']
            domain = example.get('domain', 'unknown')

            print(f"\n  Example {ex_idx + 1}/{NUM_EXAMPLES}: "
                  f"seq_len={seq_len}, domain={domain[:40]}")

            for layer_name in LAYERS:
                layer_start = time.time()

                layer = example[layer_name]
                Q = np.array(layer['Q'], dtype=np.float32)
                K = np.array(layer['K'], dtype=np.float32)
                V = np.array(layer['V'], dtype=np.float32)

                # Sample query positions from latter half of sequence
                available = range(seq_len // 2, seq_len)
                n_queries = min(NUM_QUERIES_PER_EXAMPLE, len(available))
                query_positions = np.random.choice(
                    available, size=n_queries, replace=False
                )
                query_positions.sort()

                # Fit GMM on full key set once per layer per example
                # (Use all keys since we'll slice per query position)
                gmm_resp_full = fit_gmm(K, n_clusters=GMM_CLUSTERS, seed=SEED)

                for qi, query_pos in enumerate(query_positions):
                    q = Q[query_pos]
                    valid_keys = K[:query_pos + 1]
                    valid_values = V[:query_pos + 1]
                    n_valid = query_pos + 1

                    # Ground truth attention
                    gt_output, gt_logits, gt_weights, _ = \
                        compute_ground_truth_attention(q, K, V, query_pos, HEAD_DIM)

                    # GMM responsibilities for valid keys only
                    resp = gmm_resp_full[:n_valid]

                    # --- Metric 1: Attention entropy ---
                    entropy = attention_entropy(gt_weights)

                    # --- Metric 2: Within-cluster logit variance ---
                    wc_var = within_cluster_logit_variance(gt_logits, resp)

                    # --- Metric 3: Key-value cosine correlation ---
                    kv_corr = key_value_cosine_correlation(valid_keys, valid_values)

                    # --- Metric 4: Value effective rank ---
                    v_eff_rank = value_effective_rank(valid_values)

                    # --- Metric 5: Max attention weight ---
                    max_w = max_attention_weight(gt_weights)

                    # --- Metric 6: GMM error ---
                    output_gmm, n_active = gmm_attention(
                        q, valid_keys, valid_values, gt_logits, HEAD_DIM, resp
                    )
                    gmm_error = relative_l2_error(output_gmm, gt_output)

                    results[layer_name].append({
                        'example_idx': ex_idx,
                        'query_pos': int(query_pos),
                        'n_valid': int(n_valid),
                        'entropy': float(entropy),
                        'within_cluster_logit_var': float(wc_var),
                        'kv_cosine_correlation': float(kv_corr),
                        'value_effective_rank': float(v_eff_rank),
                        'max_attention_weight': float(max_w),
                        'gmm_error': float(gmm_error),
                        'gmm_n_active_clusters': int(n_active),
                    })

                layer_time = time.time() - layer_start
                print(f"    {layer_name}: {n_queries} queries in {layer_time:.1f}s")

    total_time = time.time() - total_start
    print(f"\nTotal processing time: {total_time:.1f}s ({total_time / 60:.1f} min)")

    # ========================================================================
    # Aggregate statistics
    # ========================================================================

    metrics = [
        'entropy', 'within_cluster_logit_var', 'kv_cosine_correlation',
        'value_effective_rank', 'max_attention_weight', 'gmm_error',
    ]

    aggregated = {}
    for layer_name in LAYERS:
        layer_results = results[layer_name]
        n = len(layer_results)
        layer_agg = {'n_queries': n}

        for metric in metrics:
            vals = [r[metric] for r in layer_results]
            layer_agg[metric] = {
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'p25': float(np.percentile(vals, 25)),
                'p75': float(np.percentile(vals, 75)),
            }

        aggregated[layer_name] = layer_agg

    # ========================================================================
    # Save results
    # ========================================================================

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full per-query results
    full_output = {
        'metadata': {
            'experiment': 'per_layer_diagnostics',
            'num_examples': NUM_EXAMPLES,
            'num_queries_per_example': NUM_QUERIES_PER_EXAMPLE,
            'gmm_clusters': GMM_CLUSTERS,
            'head_dim': HEAD_DIM,
            'seed': SEED,
            'layers': LAYERS,
            'total_time_seconds': total_time,
            'timestamp': datetime.now().isoformat(),
        },
        'per_query_results': {layer: results[layer] for layer in LAYERS},
    }

    full_path = output_dir / 'full_results.json'
    with open(full_path, 'w') as f:
        json.dump(full_output, f, indent=2)
    print(f"\nFull results saved: {full_path}")

    # Aggregated results
    agg_output = {
        'metadata': full_output['metadata'],
        'aggregated': aggregated,
    }

    agg_path = output_dir / 'aggregated.json'
    with open(agg_path, 'w') as f:
        json.dump(agg_output, f, indent=2)
    print(f"Aggregated results saved: {agg_path}")

    # ========================================================================
    # Print summary table
    # ========================================================================

    print(f"\n{'=' * 90}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 90}")

    metric_labels = {
        'entropy': 'Attention Entropy H(w)',
        'within_cluster_logit_var': 'Within-Cluster Logit Var',
        'kv_cosine_correlation': 'Key-Value Cosine Corr',
        'value_effective_rank': 'Value Effective Rank',
        'max_attention_weight': 'Max Attention Weight',
        'gmm_error': 'GMM Error (rel L2)',
    }

    # Header
    print(f"\n{'Metric':<30s} | {'first_layer':>20s} | {'last_layer':>20s} | {'Ratio (L/F)':>12s}")
    print("-" * 90)

    for metric in metrics:
        first_mean = aggregated['first_layer'][metric]['mean']
        last_mean = aggregated['last_layer'][metric]['mean']
        first_std = aggregated['first_layer'][metric]['std']
        last_std = aggregated['last_layer'][metric]['std']

        if abs(first_mean) > 1e-12:
            ratio = last_mean / first_mean
            ratio_str = f"{ratio:.3f}"
        else:
            ratio_str = "N/A"

        label = metric_labels.get(metric, metric)
        print(f"{label:<30s} | {first_mean:>9.4f} +/- {first_std:<7.4f} | "
              f"{last_mean:>9.4f} +/- {last_std:<7.4f} | {ratio_str:>12s}")

    print("-" * 90)

    n_first = aggregated['first_layer']['n_queries']
    n_last = aggregated['last_layer']['n_queries']
    print(f"\nTotal queries: first_layer={n_first}, last_layer={n_last}")

    # Interpretation
    print(f"\n{'=' * 90}")
    print("INTERPRETATION")
    print(f"{'=' * 90}")

    e_first = aggregated['first_layer']['entropy']['mean']
    e_last = aggregated['last_layer']['entropy']['mean']
    print(f"\nEntropy: first={e_first:.4f}, last={e_last:.4f}")
    if e_last < e_first:
        print("  -> Last layer has LOWER entropy (more concentrated attention)")
    else:
        print("  -> Last layer has HIGHER entropy (more diffuse attention)")

    wc_first = aggregated['first_layer']['within_cluster_logit_var']['mean']
    wc_last = aggregated['last_layer']['within_cluster_logit_var']['mean']
    print(f"\nWithin-cluster logit variance: first={wc_first:.4f}, last={wc_last:.4f}")
    if wc_last < wc_first:
        print("  -> Last layer clusters are TIGHTER (keys within same cluster have similar logits)")
    else:
        print("  -> Last layer clusters are LOOSER (more logit variance within clusters)")

    kv_first = aggregated['first_layer']['kv_cosine_correlation']['mean']
    kv_last = aggregated['last_layer']['kv_cosine_correlation']['mean']
    print(f"\nKey-value cosine correlation: first={kv_first:.4f}, last={kv_last:.4f}")
    if abs(kv_last) > abs(kv_first):
        print("  -> Last layer has STRONGER k-v correlation")
    else:
        print("  -> Last layer has WEAKER k-v correlation")

    gmm_first = aggregated['first_layer']['gmm_error']['mean']
    gmm_last = aggregated['last_layer']['gmm_error']['mean']
    print(f"\nGMM error: first={gmm_first:.4f}, last={gmm_last:.4f}")
    if gmm_last < gmm_first:
        print("  -> GMM is MORE effective at last layer (lower error)")
    else:
        print("  -> GMM is LESS effective at last layer (higher error)")

    print(f"\n{'=' * 90}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 90}")
    print(f"\nOutput files in {OUTPUT_DIR}:")
    print(f"  - full_results.json (per-query)")
    print(f"  - aggregated.json (statistics)")

    return aggregated


if __name__ == '__main__':
    run_experiment()
