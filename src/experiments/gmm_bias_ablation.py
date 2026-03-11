#!/usr/bin/env python3
"""
GMM Bias Source Ablation Experiment.

Determines which bias source dominates in GMM attention:
  - Source 1: Weight distortion (softmax over centroid logits != true W_c)
  - Source 2: Value averaging (responsibility-weighted != attention-weighted)

Four variants compared:
  1. Standard GMM         — both sources present
  2. GMM Exact Weights    — eliminates Source 1 only
  3. GMM Exact Values     — eliminates Source 2 only
  4. GMM Exact Both       — eliminates both (partition-only error)
  5. Oracle Sampling      — sampling baseline at matching budget

Results saved to: results/gmm_bias_ablation/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

from algorithms.base import compute_ground_truth_attention, relative_l2_error
from algorithms.gmm_attention import fit_gmm, gmm_attention
from algorithms.gmm_ablation import gmm_exact_weights, gmm_exact_values, gmm_exact_both
from algorithms.oracle import oracle_sampling
from visualization.plot_utils import setup_style, save_figure

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

NUM_EXAMPLES = 50
NUM_QUERIES_PER_EXAMPLE = 100
CLUSTERS = [10, 50, 100]
LAYERS_TO_TEST = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
DATA_PATH = '../../data/attention_vectors_long_bench_llama_8b.jsonl'
OUTPUT_DIR = '../../results/gmm_bias_ablation'

# ============================================================================
# END HYPERPARAMETERS
# ============================================================================

METHODS = ['Standard GMM', 'Exact Weights', 'Exact Values', 'Exact Both', 'Oracle Sampling']

METHOD_STYLES = {
    'Standard GMM':   {'color': '#1f77b4', 'hatch': ''},
    'Exact Weights':  {'color': '#ff7f0e', 'hatch': ''},
    'Exact Values':   {'color': '#2ca02c', 'hatch': ''},
    'Exact Both':     {'color': '#9467bd', 'hatch': ''},
    'Oracle Sampling': {'color': '#d62728', 'hatch': '//'},
}

LAYER_TITLES = {
    'first_layer': 'First Layer (Layer 0)',
    'last_layer': 'Last Layer (Layer 31)',
}


def evaluate_query(q, K, V, query_pos, head_dim, resp_dict):
    """Evaluate all ablation variants for one query across all cluster counts."""
    gt_output, gt_logits, gt_weights, _ = compute_ground_truth_attention(
        q, K, V, query_pos, head_dim
    )
    valid_keys = K[:query_pos + 1]
    valid_values = V[:query_pos + 1]
    nv = len(valid_keys)

    results = {}
    for C, full_resp in resp_dict.items():
        resp = full_resp[:nv]

        out_std, n_active = gmm_attention(
            q, valid_keys, valid_values, gt_logits, head_dim, resp)
        out_ew, _ = gmm_exact_weights(
            q, valid_keys, valid_values, gt_logits, head_dim, resp, gt_weights)
        out_ev, _ = gmm_exact_values(
            q, valid_keys, valid_values, gt_logits, head_dim, resp, gt_weights)
        out_eb, _ = gmm_exact_both(
            q, valid_keys, valid_values, gt_logits, head_dim, resp, gt_weights)
        out_oracle, _ = oracle_sampling(
            q, valid_keys, valid_values, gt_logits, gt_weights, C)

        results[C] = {
            'Standard GMM':    float(relative_l2_error(out_std, gt_output)),
            'Exact Weights':   float(relative_l2_error(out_ew, gt_output)),
            'Exact Values':    float(relative_l2_error(out_ev, gt_output)),
            'Exact Both':      float(relative_l2_error(out_eb, gt_output)),
            'Oracle Sampling': float(relative_l2_error(out_oracle, gt_output)),
        }

    return results


def plot_ablation(all_errors, output_dir):
    """Bar chart: methods grouped by cluster count, one subplot per layer."""
    fig, axes = plt.subplots(1, len(LAYERS_TO_TEST), figsize=(7 * len(LAYERS_TO_TEST), 6))
    if len(LAYERS_TO_TEST) == 1:
        axes = [axes]

    bar_width = 0.15
    x_base = np.arange(len(CLUSTERS))

    for ax, layer_name in zip(axes, LAYERS_TO_TEST):
        for i, method in enumerate(METHODS):
            means = []
            stds = []
            for C in CLUSTERS:
                errs = all_errors[layer_name][C][method]
                means.append(np.mean(errs) if errs else 0)
                stds.append(np.std(errs) if errs else 0)

            style = METHOD_STYLES[method]
            ax.bar(x_base + i * bar_width, means, bar_width,
                   yerr=stds, capsize=3,
                   label=method, color=style['color'],
                   hatch=style['hatch'], alpha=0.85, edgecolor='white')

        ax.set_xlabel('Number of Clusters', fontsize=12)
        ax.set_ylabel('Mean Relative L2 Error', fontsize=12)
        ax.set_title(f'GMM Bias Ablation — {LAYER_TITLES[layer_name]}', fontsize=13)
        ax.set_xticks(x_base + bar_width * (len(METHODS) - 1) / 2)
        ax.set_xticklabels([str(c) for c in CLUSTERS])
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    save_figure(fig, output_dir / 'bias_ablation.png')


def main():
    setup_style()
    np.random.seed(SEED)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(DATA_PATH)
    if not data_path.exists():
        data_path = Path(__file__).parent / DATA_PATH

    print("=" * 70)
    print("GMM Bias Source Ablation")
    print("=" * 70)
    print(f"  Methods:  {METHODS}")
    print(f"  Clusters: {CLUSTERS}")
    print(f"  Examples: {NUM_EXAMPLES}")
    print(f"  Queries:  {NUM_QUERIES_PER_EXAMPLE} per example")
    print(f"  Layers:   {LAYERS_TO_TEST}")
    print(f"  Output:   {output_dir}")
    print()

    # Collect errors: {layer: {C: {method: [errors]}}}
    all_errors = {
        layer: {C: {m: [] for m in METHODS} for C in CLUSTERS}
        for layer in LAYERS_TO_TEST
    }

    t0 = time.time()

    with open(data_path, 'r') as f:
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

                # Fit GMM once per cluster count (on all keys)
                resp_dict = {}
                for C in CLUSTERS:
                    resp_dict[C] = fit_gmm(K, C, seed=SEED)

                # Pick query positions
                min_pos = max(max(CLUSTERS) + 1, seq_len // 4)
                max_pos = seq_len - 1
                n_queries = min(NUM_QUERIES_PER_EXAMPLE, max_pos - min_pos + 1)
                if n_queries <= 0:
                    continue
                query_positions = np.random.choice(
                    range(min_pos, max_pos + 1), size=n_queries, replace=False)

                for qpos in query_positions:
                    qr = evaluate_query(Q[qpos], K, V, qpos, HEAD_DIM, resp_dict)
                    for C, method_errors in qr.items():
                        for method, err in method_errors.items():
                            all_errors[layer_name][C][method].append(err)

    elapsed = time.time() - t0
    print(f"\nComputation done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Plot
    print("\nGenerating plots...")
    plot_ablation(all_errors, output_dir)

    # Save JSON
    json_results = {
        'metadata': {
            'clusters': CLUSTERS,
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

    json_path = output_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved: {json_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for layer_name in LAYERS_TO_TEST:
        print(f"\n{LAYER_TITLES[layer_name]}")
        for C in CLUSTERS:
            print(f"\n  C = {C}")
            header = f"    {'Method':<20s} {'Mean':>10s} {'Median':>10s} {'Std':>10s}"
            print(header)
            print("    " + "-" * 52)
            for method in METHODS:
                errs = all_errors[layer_name][C][method]
                if errs:
                    print(f"    {method:<20s} {np.mean(errs):>10.4f} "
                          f"{np.median(errs):>10.4f} {np.std(errs):>10.4f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
