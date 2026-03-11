#!/usr/bin/env python3
"""
Oracle Variants Comparison: Budget Sweep

Compares four methods across a budget sweep:
  1. Oracle (weights only)     — sample ~ w_i, average values
  2. Oracle Value-Weighted     — sample ~ w_i*||v_i||, IS-corrected
  3. Uniform Sampling          — sample uniformly, subset softmax
  4. TopK Attention            — top-B by logit, subset softmax

Results saved to: results/oracle_variants_comparison/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time

from algorithms.base import compute_ground_truth_attention, relative_l2_error
from algorithms.oracle import oracle_sampling
from algorithms.oracle_value_weighted import oracle_value_weighted
from algorithms.uniform import uniform_sampling
from algorithms.topk import topk_attention
from visualization.plot_utils import setup_style, save_figure

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

BUDGETS = [10, 20, 40, 60, 80, 100, 150, 200, 300, 500]
NUM_EXAMPLES = 50
NUM_QUERIES_PER_EXAMPLE = 100
LAYERS_TO_TEST = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
DATA_PATH = '../../data/attention_vectors_long_bench_llama_8b.jsonl'
OUTPUT_DIR = '../../results/oracle_variants_comparison'

# ============================================================================
# END HYPERPARAMETERS
# ============================================================================

METHODS = {
    'Oracle':    {'color': '#2ca02c', 'marker': '^', 'linestyle': '-'},
    'OracleVW':  {'color': '#006400', 'marker': 'v', 'linestyle': '-'},
    'Uniform':   {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-'},
    'TopK':      {'color': '#d62728', 'marker': 'o', 'linestyle': '-'},
}

LAYER_TITLES = {
    'first_layer': 'First Layer (Layer 0)',
    'last_layer': 'Last Layer (Layer 31)',
}


def evaluate_query(q, K, V, query_pos, head_dim):
    """Evaluate all 4 methods at every budget for one query."""
    gt_output, gt_logits, gt_weights, _ = compute_ground_truth_attention(
        q, K, V, query_pos, head_dim
    )
    valid_keys = K[:query_pos + 1]
    valid_values = V[:query_pos + 1]
    num_valid = len(valid_keys)

    results = {}
    for budget in BUDGETS:
        if budget > num_valid:
            continue

        out_oracle, _ = oracle_sampling(
            q, valid_keys, valid_values, gt_logits, gt_weights, budget)
        out_vw, _ = oracle_value_weighted(
            q, valid_keys, valid_values, gt_logits, gt_weights, budget)
        out_uniform, _ = uniform_sampling(
            q, valid_keys, valid_values, gt_logits, budget)
        out_topk, _ = topk_attention(
            q, valid_keys, valid_values, gt_logits, budget)

        results[budget] = {
            'Oracle':   float(relative_l2_error(out_oracle, gt_output)),
            'OracleVW': float(relative_l2_error(out_vw, gt_output)),
            'Uniform':  float(relative_l2_error(out_uniform, gt_output)),
            'TopK':     float(relative_l2_error(out_topk, gt_output)),
        }

    return results


def plot_error_vs_budget(all_errors, layer_name, output_dir):
    """Error vs budget curve with shaded std bands."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, style in METHODS.items():
        means, stds, budgets_valid = [], [], []
        for b in BUDGETS:
            errs = all_errors[layer_name].get(b, {}).get(method)
            if errs is not None and len(errs) > 0:
                means.append(np.mean(errs))
                stds.append(np.std(errs))
                budgets_valid.append(b)

        if not budgets_valid:
            continue

        m = np.array(means)
        s = np.array(stds)
        bv = np.array(budgets_valid)

        ax.plot(bv, m, marker=style['marker'], linestyle=style['linestyle'],
                color=style['color'], linewidth=2.5, markersize=6,
                label=method, alpha=0.95, zorder=3)
        ax.fill_between(bv, np.maximum(m - s, 1e-6), m + s,
                        color=style['color'], alpha=0.15, linewidth=0)

    ax.set_xlabel('Budget (number of keys sampled)', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title(f'Oracle Variants Comparison — {LAYER_TITLES[layer_name]}', fontsize=13)
    ax.set_yscale('log')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    save_figure(fig, output_dir / f'error_vs_budget_{layer_name}.png')


def plot_improvement_ratio(all_errors, layer_name, output_dir):
    """Plot ratio of Oracle error / OracleVW error to show relative improvement."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ratios_mean, ratios_std, budgets_valid = [], [], []
    for b in BUDGETS:
        oracle_errs = all_errors[layer_name].get(b, {}).get('Oracle')
        vw_errs = all_errors[layer_name].get(b, {}).get('OracleVW')
        if oracle_errs is not None and vw_errs is not None and len(oracle_errs) > 0:
            # Per-query ratio
            o = np.array(oracle_errs)
            v = np.array(vw_errs)
            r = o / np.maximum(v, 1e-10)
            ratios_mean.append(np.mean(r))
            ratios_std.append(np.std(r))
            budgets_valid.append(b)

    if not budgets_valid:
        plt.close()
        return

    m = np.array(ratios_mean)
    s = np.array(ratios_std)
    bv = np.array(budgets_valid)

    ax.plot(bv, m, 'o-', color='#7570b3', linewidth=2.5, markersize=7)
    ax.fill_between(bv, m - s, m + s, color='#7570b3', alpha=0.15)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
               label='No difference (ratio=1)')
    ax.set_xlabel('Budget', fontsize=12)
    ax.set_ylabel('Error ratio: Oracle / Oracle-VW', fontsize=12)
    ax.set_title(
        f'Value-Weighting Improvement — {LAYER_TITLES[layer_name]}\n'
        f'(>1 means VW is better)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, output_dir / f'improvement_ratio_{layer_name}.png')


def main():
    setup_style()
    np.random.seed(SEED)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(DATA_PATH)
    if not data_path.exists():
        data_path = Path(__file__).parent / DATA_PATH

    print("=" * 70)
    print("Oracle Variants Comparison — Budget Sweep")
    print("=" * 70)
    print(f"  Methods:  {list(METHODS.keys())}")
    print(f"  Budgets:  {BUDGETS}")
    print(f"  Examples: {NUM_EXAMPLES}")
    print(f"  Queries:  {NUM_QUERIES_PER_EXAMPLE} per example")
    print(f"  Layers:   {LAYERS_TO_TEST}")
    print(f"  Output:   {output_dir}")
    print()

    # Collect errors: {layer: {budget: {method: [errors]}}}
    all_errors = {layer: {b: {m: [] for m in METHODS} for b in BUDGETS}
                  for layer in LAYERS_TO_TEST}

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

                # Pick query positions from the latter portion
                min_pos = max(BUDGETS[-1] + 1, seq_len // 4)
                max_pos = seq_len - 1
                n_queries = min(NUM_QUERIES_PER_EXAMPLE, max_pos - min_pos + 1)
                query_positions = np.random.choice(
                    range(min_pos, max_pos + 1), size=n_queries, replace=False)

                for qpos in query_positions:
                    qr = evaluate_query(Q[qpos], K, V, qpos, HEAD_DIM)
                    for budget, method_errors in qr.items():
                        for method, err in method_errors.items():
                            all_errors[layer_name][budget][method].append(err)

    elapsed = time.time() - t0
    print(f"\nComputation done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Plots
    print("\nGenerating plots...")
    for layer_name in LAYERS_TO_TEST:
        plot_error_vs_budget(all_errors, layer_name, output_dir)
        plot_improvement_ratio(all_errors, layer_name, output_dir)

    # Save JSON
    json_results = {
        'metadata': {
            'budgets': BUDGETS,
            'num_examples': NUM_EXAMPLES,
            'num_queries_per_example': NUM_QUERIES_PER_EXAMPLE,
            'layers': LAYERS_TO_TEST,
            'seed': SEED,
            'elapsed_seconds': elapsed,
        },
        'results': {},
    }

    for layer_name in LAYERS_TO_TEST:
        layer_data = {}
        for b in BUDGETS:
            budget_data = {}
            for method in METHODS:
                errs = all_errors[layer_name][b][method]
                if errs:
                    budget_data[method] = {
                        'mean': float(np.mean(errs)),
                        'median': float(np.median(errs)),
                        'std': float(np.std(errs)),
                        'n': len(errs),
                    }
            layer_data[str(b)] = budget_data
        json_results['results'][layer_name] = layer_data

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
        header = f"{'Budget':>8}"
        for m in METHODS:
            header += f"  {m:>12}"
        print(header)
        print("-" * len(header))

        for b in BUDGETS:
            row = f"{b:>8}"
            for m in METHODS:
                errs = all_errors[layer_name][b][m]
                if errs:
                    row += f"  {np.mean(errs):>12.4f}"
                else:
                    row += f"  {'—':>12}"
            print(row)

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
