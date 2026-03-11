#!/usr/bin/env python3
"""
Experiment 10: Attention Sink & Query Correlation Analysis

Prompted by Alex Andoni's observations on the Llama-3-8B attention data:
  1. Queries are highly correlated (cosine sim ~0.95 at last layer)
  2. Key 0 (first token) is an "attention sink" — gets disproportionate weight
  3. Attention entropy tracks the "uniform over ~50% keys" line

This experiment measures:
  A. Query-query cosine similarity distribution
  B. ||q - mean(Q)|| / ||mean(Q)|| ratio (query spread relative to mean)
  C. Key 0 properties: norm, alignment with mean query, attention weight
  D. Attention weight on position 0 vs all other positions
  E. Effect of excluding position 0 on GMM/oracle/sampling error
  F. Attention entropy distribution and the effective support size

Run from: claude_experiments/10_attention_sink_analysis/
Results saved to: claude_experiments/10_attention_sink_analysis/results/
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
from algorithms.uniform import uniform_sampling
from visualization.plot_utils import setup_style, save_figure

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

NUM_EXAMPLES = 10
NUM_QUERIES_PER_EXAMPLE = 100
BUDGET = 100
C_CLUSTERS = 50
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


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_attention_no_sink(q, K, V, query_pos, head_dim, skip_positions=None):
    """
    Compute full attention excluding specified positions (e.g., position 0).
    Returns same tuple as compute_ground_truth_attention but without sink.
    """
    if skip_positions is None:
        skip_positions = {0}

    valid_keys = K[:query_pos + 1]
    valid_values = V[:query_pos + 1]

    # Create mask excluding skip positions
    keep_mask = np.ones(query_pos + 1, dtype=bool)
    for pos in skip_positions:
        if pos <= query_pos:
            keep_mask[pos] = False

    filtered_keys = valid_keys[keep_mask]
    filtered_values = valid_values[keep_mask]

    if len(filtered_keys) == 0:
        return np.zeros(head_dim), np.array([]), np.array([]), 0.0

    logits = (q @ filtered_keys.T) / np.sqrt(head_dim)
    unnorm = np.exp(logits - np.max(logits))
    normalizer = np.sum(unnorm)
    weights = unnorm / normalizer
    output = weights @ filtered_values

    return output, logits, weights, normalizer


def analyze_example(example, layer_name, rng):
    """Analyze one example for all metrics."""
    layer_data = example[layer_name]
    Q = np.array(layer_data['Q'], dtype=np.float32)
    K = np.array(layer_data['K'], dtype=np.float32)
    V = np.array(layer_data['V'], dtype=np.float32)
    seq_len = example['sequence_length']

    results = {
        # Query correlation
        'query_query_cosine_sims': [],
        'q_minus_mean_norms': [],
        'mean_q_norm': 0,

        # Key 0 properties
        'key0_norm': float(np.linalg.norm(K[0])),
        'key0_cosine_with_mean_q': 0,
        'key0_attention_weights': [],

        # Key norm stats
        'all_key_norms': [],

        # Entropy
        'entropies': [],
        'effective_support_fractions': [],

        # Sink effect on algorithms
        'errors_with_sink': {'gmm': [], 'oracle': [], 'uniform': []},
        'errors_no_sink': {'gmm': [], 'oracle': [], 'uniform': []},

        # Attention weight at position 0
        'sink_weight_fracs': [],
    }

    # --- Part A: Query-query similarity ---
    # Sample queries from the second half
    min_pos = max(C_CLUSTERS + 1, seq_len // 2)
    max_pos = seq_len - 1
    n_queries = min(NUM_QUERIES_PER_EXAMPLE, max_pos - min_pos + 1)
    if n_queries <= 0:
        return None

    query_positions = rng.choice(range(min_pos, max_pos + 1), size=n_queries, replace=False)
    query_vectors = Q[query_positions]  # [n_queries, d]

    # Mean query
    mean_q = query_vectors.mean(axis=0)
    mean_q_norm = float(np.linalg.norm(mean_q))
    results['mean_q_norm'] = mean_q_norm

    # ||q - mean(Q)|| for each query
    deviations = query_vectors - mean_q[np.newaxis, :]
    dev_norms = np.linalg.norm(deviations, axis=1)
    results['q_minus_mean_norms'] = dev_norms.tolist()

    # Pairwise cosine similarity (subsample for speed)
    n_pairs = min(200, n_queries * (n_queries - 1) // 2)
    norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized = query_vectors / norms

    # Random pairs
    for _ in range(n_pairs):
        i, j = rng.choice(n_queries, size=2, replace=False)
        sim = float(np.dot(normalized[i], normalized[j]))
        results['query_query_cosine_sims'].append(sim)

    # Key 0 cosine with mean query
    results['key0_cosine_with_mean_q'] = cosine_sim(K[0], mean_q)

    # All key norms
    results['all_key_norms'] = np.linalg.norm(K[:seq_len], axis=1).tolist()

    # --- Part B: Per-query analysis ---
    # Fit GMM once for with-sink / no-sink comparison
    gmm_resp_full = fit_gmm(K[:seq_len], C_CLUSTERS, seed=SEED)
    # For no-sink: fit GMM on keys excluding position 0
    gmm_resp_nosink = fit_gmm(K[1:seq_len], C_CLUSTERS, seed=SEED)

    for qpos in query_positions:
        q = Q[qpos]
        nv = qpos + 1

        # Full attention (with sink)
        gt_output, gt_logits, gt_weights, _ = compute_ground_truth_attention(
            q, K, V, qpos, HEAD_DIM)

        # Attention weight on position 0
        w0 = float(gt_weights[0]) if len(gt_weights) > 0 else 0
        results['sink_weight_fracs'].append(w0)

        # Key 0 attention weight
        results['key0_attention_weights'].append(w0)

        # Entropy
        w_pos = gt_weights[gt_weights > 1e-20]
        entropy = float(-np.sum(w_pos * np.log(w_pos)))
        results['entropies'].append(entropy)

        # Effective support: exp(H) / nv
        eff_support = np.exp(entropy)
        results['effective_support_fractions'].append(float(eff_support / nv))

        # --- Part C: Algorithm comparison with/without sink ---
        valid_keys = K[:nv]
        valid_values = V[:nv]

        # WITH sink
        resp_w = gmm_resp_full[:nv]
        out_gmm_w, _ = gmm_attention(q, valid_keys, valid_values, gt_logits, HEAD_DIM, resp_w)
        out_oracle_w, _ = oracle_sampling(q, valid_keys, valid_values, gt_logits, gt_weights, BUDGET)
        out_uniform_w, _ = uniform_sampling(q, valid_keys, valid_values, gt_logits, BUDGET)

        results['errors_with_sink']['gmm'].append(
            float(relative_l2_error(out_gmm_w, gt_output)))
        results['errors_with_sink']['oracle'].append(
            float(relative_l2_error(out_oracle_w, gt_output)))
        results['errors_with_sink']['uniform'].append(
            float(relative_l2_error(out_uniform_w, gt_output)))

        # WITHOUT sink (exclude position 0)
        gt_output_ns, gt_logits_ns, gt_weights_ns, _ = compute_attention_no_sink(
            q, K, V, qpos, HEAD_DIM, skip_positions={0})

        if len(gt_logits_ns) > 0:
            valid_keys_ns = K[1:nv]
            valid_values_ns = V[1:nv]
            resp_ns = gmm_resp_nosink[:nv - 1]

            out_gmm_ns, _ = gmm_attention(
                q, valid_keys_ns, valid_values_ns, gt_logits_ns, HEAD_DIM, resp_ns)
            out_oracle_ns, _ = oracle_sampling(
                q, valid_keys_ns, valid_values_ns, gt_logits_ns, gt_weights_ns,
                min(BUDGET, len(gt_weights_ns)))
            out_uniform_ns, _ = uniform_sampling(
                q, valid_keys_ns, valid_values_ns, gt_logits_ns,
                min(BUDGET, len(gt_logits_ns)))

            results['errors_no_sink']['gmm'].append(
                float(relative_l2_error(out_gmm_ns, gt_output_ns)))
            results['errors_no_sink']['oracle'].append(
                float(relative_l2_error(out_oracle_ns, gt_output_ns)))
            results['errors_no_sink']['uniform'].append(
                float(relative_l2_error(out_uniform_ns, gt_output_ns)))

    return results


def plot_results(all_results, output_dir):
    """Generate comprehensive visualization."""
    fig = plt.figure(figsize=(20, 16))

    for col, layer_name in enumerate(LAYERS_TO_TEST):
        res = all_results[layer_name]

        # 1. Query-query cosine similarity histogram
        ax1 = fig.add_subplot(3, len(LAYERS_TO_TEST), col + 1)
        sims = res['query_query_cosine_sims']
        ax1.hist(sims, bins=50, color='#9467bd', alpha=0.7, edgecolor='white')
        ax1.axvline(np.mean(sims), color='red', linestyle='--',
                    label=f'Mean: {np.mean(sims):.3f}')
        ax1.set_xlabel('Cosine Similarity')
        ax1.set_ylabel('Count')
        ax1.set_title(f'Query-Query Similarity\n{LAYER_TITLES[layer_name]}')
        ax1.legend()

        # 2. ||q - mean(Q)|| distribution
        ax2 = fig.add_subplot(3, len(LAYERS_TO_TEST), col + 1 + len(LAYERS_TO_TEST))
        dev_norms = res['q_minus_mean_norms']
        ax2.hist(dev_norms, bins=50, color='#ff7f0e', alpha=0.7, edgecolor='white')
        ax2.axvline(np.median(dev_norms), color='red', linestyle='--',
                    label=f'Median: {np.median(dev_norms):.2f}')
        ax2.axvline(res['mean_q_norm'], color='blue', linestyle=':',
                    label=f'||mean(Q)||: {res["mean_q_norm"]:.2f}')
        ax2.set_xlabel('||q - mean(Q)||')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Query Deviation from Mean\n'
                      f'Ratio: {np.median(dev_norms)/res["mean_q_norm"]:.3f}')
        ax2.legend(fontsize=8)

        # 3. Sink effect on errors
        ax3 = fig.add_subplot(3, len(LAYERS_TO_TEST), col + 1 + 2 * len(LAYERS_TO_TEST))
        methods = ['gmm', 'oracle', 'uniform']
        labels = ['GMM', 'Oracle', 'Uniform']
        x = np.arange(len(methods))
        width = 0.35

        with_means = [np.mean(res['errors_with_sink'][m]) for m in methods]
        no_means = [np.mean(res['errors_no_sink'][m]) for m in methods]

        ax3.bar(x - width/2, with_means, width, label='With Sink', color='#1f77b4', alpha=0.8)
        ax3.bar(x + width/2, no_means, width, label='No Sink', color='#2ca02c', alpha=0.8)
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Mean Relative L2 Error')
        ax3.set_title(f'Effect of Excluding Position 0\n{LAYER_TITLES[layer_name]}')
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    save_figure(fig, output_dir / 'attention_sink_analysis.png')

    # Second figure: entropy and sink weight
    fig2, axes2 = plt.subplots(2, len(LAYERS_TO_TEST),
                               figsize=(14, 10))
    if len(LAYERS_TO_TEST) == 1:
        axes2 = axes2.reshape(-1, 1)

    for col, layer_name in enumerate(LAYERS_TO_TEST):
        res = all_results[layer_name]

        # Effective support fraction histogram
        ax = axes2[0, col]
        fracs = res['effective_support_fractions']
        ax.hist(fracs, bins=50, color='#2ca02c', alpha=0.7, edgecolor='white')
        ax.axvline(np.median(fracs), color='red', linestyle='--',
                   label=f'Median: {np.median(fracs):.3f}')
        ax.axvline(0.5, color='blue', linestyle=':', alpha=0.7, label='50% line')
        ax.axvline(0.1, color='orange', linestyle=':', alpha=0.7, label='10% line')
        ax.set_xlabel('Effective Support Fraction (exp(H)/N)')
        ax.set_ylabel('Count')
        ax.set_title(f'Attention Support Size\n{LAYER_TITLES[layer_name]}')
        ax.legend(fontsize=8)

        # Sink weight distribution
        ax = axes2[1, col]
        sink_w = res['sink_weight_fracs']
        ax.hist(sink_w, bins=50, color='#d62728', alpha=0.7, edgecolor='white')
        ax.axvline(np.median(sink_w), color='red', linestyle='--',
                   label=f'Median: {np.median(sink_w):.4f}')
        # Expected uniform weight for reference
        median_nv = 6000  # approximate
        ax.axvline(1.0 / median_nv, color='blue', linestyle=':',
                   label=f'Uniform: 1/{median_nv}={1.0/median_nv:.5f}')
        ax.set_xlabel('Attention Weight on Position 0')
        ax.set_ylabel('Count')
        ax.set_title(f'Attention Sink Weight\n{LAYER_TITLES[layer_name]}')
        ax.legend(fontsize=8)

    fig2.tight_layout()
    save_figure(fig2, output_dir / 'entropy_and_sink.png')


def main():
    setup_style()
    rng = np.random.RandomState(SEED)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 10: Attention Sink & Query Correlation Analysis")
    print("=" * 70)
    print(f"  Examples: {NUM_EXAMPLES}")
    print(f"  Queries:  {NUM_QUERIES_PER_EXAMPLE} per example")
    print(f"  Budget:   {BUDGET}")
    print(f"  Clusters: {C_CLUSTERS}")
    print(f"  Layers:   {LAYERS_TO_TEST}")
    print(f"  Output:   {output_dir}")
    print()

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        sys.exit(1)

    # Aggregate results across examples
    all_results = {layer: {
        'query_query_cosine_sims': [],
        'q_minus_mean_norms': [],
        'mean_q_norms': [],
        'key0_norms': [],
        'key0_cosine_with_mean_q': [],
        'key0_attention_weights': [],
        'entropies': [],
        'effective_support_fractions': [],
        'sink_weight_fracs': [],
        'errors_with_sink': {'gmm': [], 'oracle': [], 'uniform': []},
        'errors_no_sink': {'gmm': [], 'oracle': [], 'uniform': []},
        'all_key_norms': [],
    } for layer in LAYERS_TO_TEST}

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
                res = analyze_example(example, layer_name, rng)
                if res is None:
                    continue

                ar = all_results[layer_name]
                ar['query_query_cosine_sims'].extend(res['query_query_cosine_sims'])
                ar['q_minus_mean_norms'].extend(res['q_minus_mean_norms'])
                ar['mean_q_norms'].append(res['mean_q_norm'])
                ar['key0_norms'].append(res['key0_norm'])
                ar['key0_cosine_with_mean_q'].append(res['key0_cosine_with_mean_q'])
                ar['key0_attention_weights'].extend(res['key0_attention_weights'])
                ar['entropies'].extend(res['entropies'])
                ar['effective_support_fractions'].extend(res['effective_support_fractions'])
                ar['sink_weight_fracs'].extend(res['sink_weight_fracs'])
                for m in ['gmm', 'oracle', 'uniform']:
                    ar['errors_with_sink'][m].extend(res['errors_with_sink'][m])
                    ar['errors_no_sink'][m].extend(res['errors_no_sink'][m])
                # Only keep a summary of key norms per example (too large otherwise)
                knorms = np.array(res['all_key_norms'])
                ar['all_key_norms'].append({
                    'key0_norm': float(knorms[0]) if len(knorms) > 0 else 0,
                    'median_norm': float(np.median(knorms)),
                    'mean_norm': float(np.mean(knorms)),
                    'key0_rank': int(np.sum(knorms >= knorms[0])) if len(knorms) > 0 else 0,
                })

    elapsed = time.time() - t0
    print(f"\nComputation done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Add aggregated mean_q_norm for plotting
    for layer_name in LAYERS_TO_TEST:
        ar = all_results[layer_name]
        ar['mean_q_norm'] = float(np.mean(ar['mean_q_norms']))

    # Plot
    print("\nGenerating plots...")
    plot_results(all_results, output_dir)

    # Save JSON (convert lists for JSON serialization)
    json_results = {
        'metadata': {
            'experiment': 'Attention Sink & Query Correlation Analysis',
            'num_examples': NUM_EXAMPLES,
            'num_queries_per_example': NUM_QUERIES_PER_EXAMPLE,
            'budget': BUDGET,
            'clusters': C_CLUSTERS,
            'layers': LAYERS_TO_TEST,
            'seed': SEED,
            'elapsed_seconds': elapsed,
        },
        'summary': {},
    }

    for layer_name in LAYERS_TO_TEST:
        ar = all_results[layer_name]

        qq_sims = ar['query_query_cosine_sims']
        dev_norms = ar['q_minus_mean_norms']
        entropies = ar['entropies']
        eff_fracs = ar['effective_support_fractions']
        sink_w = ar['sink_weight_fracs']

        json_results['summary'][layer_name] = {
            'query_query_cosine': {
                'mean': float(np.mean(qq_sims)),
                'median': float(np.median(qq_sims)),
                'std': float(np.std(qq_sims)),
                'p5': float(np.percentile(qq_sims, 5)),
                'p95': float(np.percentile(qq_sims, 95)),
                'n': len(qq_sims),
            },
            'query_deviation': {
                'mean_dev_norm': float(np.mean(dev_norms)),
                'median_dev_norm': float(np.median(dev_norms)),
                'mean_q_norm': float(np.mean(ar['mean_q_norms'])),
                'ratio_median': float(np.median(dev_norms) / np.mean(ar['mean_q_norms'])),
            },
            'key0_properties': {
                'mean_norm': float(np.mean(ar['key0_norms'])),
                'mean_cosine_with_mean_q': float(np.mean(ar['key0_cosine_with_mean_q'])),
                'median_attention_weight': float(np.median(ar['key0_attention_weights'])),
                'mean_attention_weight': float(np.mean(ar['key0_attention_weights'])),
                'max_attention_weight': float(np.max(ar['key0_attention_weights'])),
                'key_norm_stats': ar['all_key_norms'],
            },
            'entropy': {
                'mean': float(np.mean(entropies)),
                'median': float(np.median(entropies)),
                'std': float(np.std(entropies)),
                'min': float(np.min(entropies)),
                'max': float(np.max(entropies)),
            },
            'effective_support_fraction': {
                'mean': float(np.mean(eff_fracs)),
                'median': float(np.median(eff_fracs)),
                'p10': float(np.percentile(eff_fracs, 10)),
                'p90': float(np.percentile(eff_fracs, 90)),
            },
            'sink_weight': {
                'median': float(np.median(sink_w)),
                'mean': float(np.mean(sink_w)),
                'max': float(np.max(sink_w)),
                'ratio_to_uniform': float(np.median(sink_w) * 6000),  # ~6000 valid keys
            },
            'sink_exclusion_effect': {
                method: {
                    'with_sink': float(np.mean(ar['errors_with_sink'][method])),
                    'no_sink': float(np.mean(ar['errors_no_sink'][method])),
                    'change_pct': float(
                        (np.mean(ar['errors_no_sink'][method]) -
                         np.mean(ar['errors_with_sink'][method])) /
                        np.mean(ar['errors_with_sink'][method]) * 100
                    ) if ar['errors_with_sink'][method] else 0,
                }
                for method in ['gmm', 'oracle', 'uniform']
            },
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
        s = json_results['summary'][layer_name]
        print(f"\n{LAYER_TITLES[layer_name]}")
        print(f"  Query-Query Cosine Similarity:")
        print(f"    Mean: {s['query_query_cosine']['mean']:.4f}, "
              f"Median: {s['query_query_cosine']['median']:.4f}, "
              f"[p5, p95]: [{s['query_query_cosine']['p5']:.4f}, "
              f"{s['query_query_cosine']['p95']:.4f}]")

        print(f"  Query Deviation from Mean:")
        print(f"    Median ||q-mean||: {s['query_deviation']['median_dev_norm']:.2f}, "
              f"||mean(Q)||: {s['query_deviation']['mean_q_norm']:.2f}, "
              f"Ratio: {s['query_deviation']['ratio_median']:.4f}")

        print(f"  Key 0 (Attention Sink):")
        print(f"    Norm: {s['key0_properties']['mean_norm']:.2f}, "
              f"Cosine w/ mean(Q): {s['key0_properties']['mean_cosine_with_mean_q']:.4f}")
        print(f"    Median attn weight: {s['key0_properties']['median_attention_weight']:.5f}, "
              f"Max: {s['key0_properties']['max_attention_weight']:.5f}")
        print(f"    Weight ratio to uniform: {s['sink_weight']['ratio_to_uniform']:.1f}x")

        print(f"  Attention Entropy:")
        print(f"    Mean: {s['entropy']['mean']:.3f} nats, "
              f"Range: [{s['entropy']['min']:.3f}, {s['entropy']['max']:.3f}]")
        print(f"    Effective support: median={s['effective_support_fraction']['median']:.3f} "
              f"(={s['effective_support_fraction']['median']*100:.1f}% of context)")
        print(f"    [p10, p90]: [{s['effective_support_fraction']['p10']:.3f}, "
              f"{s['effective_support_fraction']['p90']:.3f}]")

        print(f"  Sink Exclusion Effect on Errors:")
        for method in ['gmm', 'oracle', 'uniform']:
            se = s['sink_exclusion_effect'][method]
            print(f"    {method:>8s}: with={se['with_sink']:.4f}, "
                  f"without={se['no_sink']:.4f}, "
                  f"change={se['change_pct']:+.1f}%")

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
