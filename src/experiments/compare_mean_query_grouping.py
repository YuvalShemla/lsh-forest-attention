#!/usr/bin/env python3
"""
Mean-Query Fixed Grouping: Full Experiment

Runs ALL grouping methods (fixed + per-query) and baselines (TopK, Uniform, Oracle)
across all examples and 30 queries per example.

Usage:
  python compare_mean_query_grouping.py              # run compute + plot
  python compare_mean_query_grouping.py --plot-only   # regenerate plots from JSON
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
from pathlib import Path

from algorithms.base import softmax
from algorithms.oracle import oracle_sampling
from algorithms.sorted_keys_grouping import (
    grouped_attention, GROUPING_METHODS,
    group_equal_splits, group_kmeans_1d, group_log_spaced,
    group_quantile_weight, group_variance_minimizing,
)
from visualization.plot_utils import setup_style, save_figure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
DATA_PATH = '../../data/attention_vectors_long_bench_llama_8b.jsonl'
OUTPUT_DIR = Path('../../results/mean_query_grouping')
NUM_EXAMPLES = 503  # all examples
NUM_TEST_QUERIES = 30
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
BUDGETS = [2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024]

# Grouping methods (excluding overlap and threshold)
GROUPING_METHODS_USED = {
    k: v for k, v in GROUPING_METHODS.items()
    if k not in ('overlap', 'threshold')
}

# Colors per grouping method
METHOD_COLORS = {
    'equal':      '#1f77b4',
    'kmeans':     '#9467bd',
    'log_spaced': '#8c564b',
    'quantile':   '#17becf',
    'variance':   '#bcbd22',
}


# ============================================================================
# HELPERS
# ============================================================================

def rel_l2(approx, truth):
    return np.linalg.norm(approx - truth) / (np.linalg.norm(truth) + 1e-8)


def format_eta(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


class ProgressTracker:
    def __init__(self, total, prefix=""):
        self.total = total
        self.done = 0
        self.t0 = time.time()
        self.prefix = prefix

    def step(self, info=""):
        self.done += 1
        el = time.time() - self.t0
        rem = el / self.done * (self.total - self.done)
        print(f"\r  {self.prefix}[{100*self.done/self.total:5.1f}%] "
              f"{self.done}/{self.total}  "
              f"elapsed {format_eta(el)}  ETA {format_eta(rem)}  {info}",
              end="", flush=True)

    def finish(self):
        print(f"\r  {self.prefix}Done in {format_eta(time.time()-self.t0)}" + " " * 40)


def compute_topk(logits, values, k):
    k = min(k, len(logits))
    idx = np.argpartition(logits, -k)[-k:]
    return softmax(logits[idx]) @ values[idx]


def compute_uniform(logits, values, k, rng):
    k = min(k, len(logits))
    idx = rng.choice(len(logits), size=k, replace=False)
    return softmax(logits[idx]) @ values[idx]


def get_group_labels(method, n_keys, num_groups, sorted_weights):
    """Get group labels for sorted keys."""
    num_groups = max(1, min(num_groups, n_keys))
    if method == 'equal':
        return group_equal_splits(n_keys, num_groups)
    elif method == 'kmeans':
        return group_kmeans_1d(sorted_weights, num_groups)
    elif method == 'log_spaced':
        return group_log_spaced(n_keys, num_groups)
    elif method == 'quantile':
        return group_quantile_weight(sorted_weights, num_groups)
    elif method == 'variance':
        return group_variance_minimizing(sorted_weights, num_groups)
    raise ValueError(f"Unknown method: {method}")


def fixed_grouping_attention(query, keys, values, sorted_indices, labels,
                             head_dim, size_weighted=False):
    """
    Attend to G group representatives (mean_key, mean_value) instead of N keys.
    """
    unique_labels = np.unique(labels)
    G = len(unique_labels)

    mean_keys = np.zeros((G, head_dim), dtype=np.float64)
    mean_values = np.zeros((G, head_dim), dtype=np.float64)
    group_sizes = np.zeros(G, dtype=np.float64)

    for i, g in enumerate(unique_labels):
        mask = labels == g
        idxs = sorted_indices[mask]
        mean_keys[i] = keys[idxs].mean(axis=0)
        mean_values[i] = values[idxs].mean(axis=0)
        group_sizes[i] = mask.sum()

    group_logits = (query @ mean_keys.T) / np.sqrt(head_dim)
    if size_weighted:
        group_logits = group_logits + np.log(group_sizes + 1e-10)
    group_weights = softmax(group_logits)

    return group_weights @ mean_values, G


# ============================================================================
# COMPUTE
# ============================================================================

def get_method_names():
    return (
        ['TopK', 'Uniform', 'Oracle']
        + [f'fixed_{m}' for m in GROUPING_METHODS_USED]
        + [f'fixed_sw_{m}' for m in GROUPING_METHODS_USED]
        + [f'perquery_{m}' for m in GROUPING_METHODS_USED]
    )


def analyze_layer(examples, layer_name, rng):
    print(f"\n{'='*60}")
    print(f"  {layer_name}")
    print(f"{'='*60}")

    method_names = get_method_names()
    errors = {m: {b: [] for b in BUDGETS} for m in method_names}

    total_queries = NUM_TEST_QUERIES * len(examples)
    progress = ProgressTracker(total_queries, f"{layer_name}: ")

    for ex_idx, example in enumerate(examples):
        Q = np.array(example[layer_name]['Q'], dtype=np.float32)
        K_mat = np.array(example[layer_name]['K'], dtype=np.float32)
        V = np.array(example[layer_name]['V'], dtype=np.float32)
        seq_len = Q.shape[0]

        # ---- Fixed grouping setup (once per example) ----
        mean_q = Q.mean(axis=0)
        n_all = seq_len
        mean_logits = (mean_q @ K_mat[:n_all].T) / np.sqrt(HEAD_DIM)
        mean_weights = softmax(mean_logits)
        sorted_indices = np.argsort(mean_logits)[::-1]
        sorted_mean_weights = mean_weights[sorted_indices]

        # Pre-compute group labels for each method and budget
        precomputed = {}
        for mk in GROUPING_METHODS_USED:
            precomputed[mk] = {}
            for budget in BUDGETS:
                precomputed[mk][budget] = get_group_labels(
                    mk, n_all, budget, sorted_mean_weights
                )

        # ---- Test queries ----
        test_positions = list(range(seq_len - NUM_TEST_QUERIES, seq_len))

        for qi, qpos in enumerate(test_positions):
            q = Q[qpos]
            keys = K_mat[:qpos + 1]
            vals = V[:qpos + 1]
            n_keys = qpos + 1

            logits = (q @ keys.T) / np.sqrt(HEAD_DIM)
            full_w = softmax(logits)
            full_out = full_w @ vals

            causal_mask = sorted_indices < n_keys
            valid_si = sorted_indices[causal_mask]

            for budget in BUDGETS:
                b = min(budget, n_keys)

                # Baselines
                errors['TopK'][budget].append(
                    rel_l2(compute_topk(logits, vals, b), full_out)
                )
                errors['Uniform'][budget].append(
                    rel_l2(compute_uniform(logits, vals, b, rng), full_out)
                )
                out_oracle, _ = oracle_sampling(
                    q, keys, vals, logits, full_w, b
                )
                errors['Oracle'][budget].append(
                    rel_l2(out_oracle, full_out)
                )

                for mk in GROUPING_METHODS_USED:
                    valid_labels = precomputed[mk][budget][causal_mask]

                    # Fixed: unweighted
                    fo, _ = fixed_grouping_attention(
                        q, keys, vals, valid_si, valid_labels, HEAD_DIM,
                        size_weighted=False
                    )
                    errors[f'fixed_{mk}'][budget].append(
                        rel_l2(fo, full_out)
                    )

                    # Fixed: size-weighted
                    fo_sw, _ = fixed_grouping_attention(
                        q, keys, vals, valid_si, valid_labels, HEAD_DIM,
                        size_weighted=True
                    )
                    errors[f'fixed_sw_{mk}'][budget].append(
                        rel_l2(fo_sw, full_out)
                    )

                    # Per-query grouping
                    _, go = grouped_attention(
                        logits, vals, full_w, b, method=mk
                    )
                    errors[f'perquery_{mk}'][budget].append(
                        rel_l2(go, full_out)
                    )

            progress.step(
                f"ex {ex_idx+1}/{len(examples)} q {qi+1}/{NUM_TEST_QUERIES}"
            )

    progress.finish()

    results = {'budgets': BUDGETS}
    for m in method_names:
        results[f'{m}_mean'] = [
            float(np.mean(errors[m][b])) for b in BUDGETS
        ]
        results[f'{m}_std'] = [
            float(np.std(errors[m][b])) for b in BUDGETS
        ]
    return results


def run_compute(data_path, output_dir):
    """Run full computation and save JSON."""
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    print(f"Scanning: {data_path}")
    with open(data_path, 'r') as f:
        total = sum(1 for _ in f)
    print(f"Found {total} examples, using all {total}")

    # Load all examples
    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples\n")

    all_results = {
        'config': {
            'num_examples': len(examples),
            'num_test_queries': NUM_TEST_QUERIES,
            'budgets': BUDGETS,
            'seed': SEED,
            'head_dim': HEAD_DIM,
            'grouping_methods': list(GROUPING_METHODS_USED.keys()),
            'method_names': get_method_names(),
        }
    }
    for layer in LAYERS:
        all_results[layer] = analyze_layer(examples, layer, rng)

    with open(output_dir / 'full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_dir / 'full_results.json'}")
    return all_results


# ============================================================================
# PLOTTING (can run standalone from JSON)
# ============================================================================

def _plot_fixed_vs_baselines(ax, data, layer_name):
    """Fixed grouping (size-weighted) + baselines."""
    x = np.array(data['budgets'])

    for name, color, marker in [('TopK', '#d62728', 'o'),
                                 ('Uniform', '#ff7f0e', 's'),
                                 ('Oracle', '#2ca02c', '^')]:
        means = np.array(data[f'{name}_mean'])
        stds = np.array(data[f'{name}_std'])
        ax.plot(x, means, marker=marker, color=color, lw=2.5,
                label=name, zorder=4, markersize=5)
        ax.fill_between(x, means, means + stds, color=color, alpha=0.12)

    markers = ['o', 'D', 'X', 'h', 'd']
    for i, (mk, display) in enumerate(GROUPING_METHODS_USED.items()):
        c = METHOD_COLORS[mk]
        m_sw = np.array(data[f'fixed_sw_{mk}_mean'])
        s_sw = np.array(data[f'fixed_sw_{mk}_std'])
        ax.plot(x, m_sw, ls='-', color=c, lw=2.2,
                marker=markers[i % len(markers)], markersize=5,
                label=f'{display} (fixed)', zorder=3)
        ax.fill_between(x, m_sw, m_sw + s_sw, color=c, alpha=0.1)

    layer_title = 'First Layer (Layer 0)' if 'first' in layer_name else 'Last Layer (Layer 31)'
    ax.set_title(layer_title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Budget (num groups)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative L2 Error', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, ls='--', which='both')


def _plot_perquery_vs_baselines(ax, data, layer_name):
    """Per-query grouping + baselines."""
    x = np.array(data['budgets'])

    for name, color, marker in [('TopK', '#d62728', 'o'),
                                 ('Uniform', '#ff7f0e', 's'),
                                 ('Oracle', '#2ca02c', '^')]:
        means = np.array(data[f'{name}_mean'])
        stds = np.array(data[f'{name}_std'])
        ax.plot(x, means, marker=marker, color=color, lw=2.5,
                label=name, zorder=4, markersize=5)
        ax.fill_between(x, means, means + stds, color=color, alpha=0.12)

    markers = ['o', 'D', 'X', 'h', 'd']
    for i, (mk, display) in enumerate(GROUPING_METHODS_USED.items()):
        c = METHOD_COLORS[mk]
        m_pq = np.array(data[f'perquery_{mk}_mean'])
        s_pq = np.array(data[f'perquery_{mk}_std'])
        ax.plot(x, m_pq, ls='-', color=c, lw=2.2,
                marker=markers[i % len(markers)], markersize=5,
                label=f'{display} (per-query)', zorder=3)
        ax.fill_between(x, m_pq, m_pq + s_pq, color=c, alpha=0.1)

    layer_title = 'First Layer (Layer 0)' if 'first' in layer_name else 'Last Layer (Layer 31)'
    ax.set_title(layer_title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Budget (num groups)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative L2 Error', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, ls='--', which='both')


def _plot_fixed_vs_perquery(ax, data, layer_name):
    """Side-by-side: fixed (solid) vs per-query (dashed) per method + baselines."""
    x = np.array(data['budgets'])

    for name, color, marker in [('TopK', '#d62728', 'o'),
                                 ('Uniform', '#ff7f0e', 's'),
                                 ('Oracle', '#2ca02c', '^')]:
        means = np.array(data[f'{name}_mean'])
        stds = np.array(data[f'{name}_std'])
        ax.plot(x, means, marker=marker, color=color, lw=2.5,
                label=name, zorder=4, markersize=5)
        ax.fill_between(x, means, means + stds, color=color, alpha=0.12)

    markers_f = ['o', 'D', 'X', 'h', 'd']
    for i, (mk, display) in enumerate(GROUPING_METHODS_USED.items()):
        c = METHOD_COLORS[mk]

        # Fixed size-weighted (solid)
        m_sw = np.array(data[f'fixed_sw_{mk}_mean'])
        s_sw = np.array(data[f'fixed_sw_{mk}_std'])
        ax.plot(x, m_sw, ls='-', color=c, lw=2.2,
                marker=markers_f[i % len(markers_f)], markersize=5,
                label=f'{display} (fixed)')
        ax.fill_between(x, m_sw, m_sw + s_sw, color=c, alpha=0.08)

        # Per-query (dashed)
        m_pq = np.array(data[f'perquery_{mk}_mean'])
        ax.plot(x, m_pq, ls='--', color=c, lw=1.5, alpha=0.6,
                marker='x', markersize=4,
                label=f'{display} (per-query)')

    layer_title = 'First Layer (Layer 0)' if 'first' in layer_name else 'Last Layer (Layer 31)'
    ax.set_title(layer_title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Budget (num groups)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative L2 Error', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, ls='--', which='both')


def make_figures(all_results, output_dir):
    """Generate all figures from results dict."""
    cfg = all_results.get('config', {})
    n_ex = cfg.get('num_examples', '?')
    n_q = cfg.get('num_test_queries', '?')
    total_q = n_ex * n_q if isinstance(n_ex, int) and isinstance(n_q, int) else '?'

    subtitle = (f'{n_ex} examples, {n_q} queries each ({total_q} total)'
                f'  |  Llama-3-8B  |  Shaded = +1 std')

    # Figure 1: Fixed grouping vs baselines
    fig1, axes1 = plt.subplots(1, 2, figsize=(22, 8.5))
    for ax, layer in zip(axes1, LAYERS):
        _plot_fixed_vs_baselines(ax, all_results[layer], layer)
    handles, labels = axes1[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper center',
                bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=10,
                framealpha=0.95, columnspacing=1.2, handletextpad=0.5)
    fig1.suptitle(f'Fixed Grouping (Mean Query) vs Baselines\n{subtitle}',
                  fontsize=14, fontweight='bold', y=1.06)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig1, output_dir / 'fixed_vs_baselines.png', dpi=200)

    # Figure 2: Per-query grouping vs baselines
    fig2, axes2 = plt.subplots(1, 2, figsize=(22, 8.5))
    for ax, layer in zip(axes2, LAYERS):
        _plot_perquery_vs_baselines(ax, all_results[layer], layer)
    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='upper center',
                bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=10,
                framealpha=0.95, columnspacing=1.2, handletextpad=0.5)
    fig2.suptitle(f'Per-Query Grouping vs Baselines\n{subtitle}',
                  fontsize=14, fontweight='bold', y=1.06)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig2, output_dir / 'perquery_vs_baselines.png', dpi=200)

    # Figure 3: Fixed vs per-query (all on one plot)
    fig3, axes3 = plt.subplots(1, 2, figsize=(22, 8.5))
    for ax, layer in zip(axes3, LAYERS):
        _plot_fixed_vs_perquery(ax, all_results[layer], layer)
    handles, labels = axes3[0].get_legend_handles_labels()
    fig3.legend(handles, labels, loc='upper center',
                bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=9,
                framealpha=0.95, columnspacing=1.0, handletextpad=0.4)
    fig3.suptitle(f'Fixed vs Per-Query Grouping\n{subtitle}',
                  fontsize=14, fontweight='bold', y=1.06)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig3, output_dir / 'fixed_vs_perquery.png', dpi=200)


# ============================================================================
# MAIN
# ============================================================================

def main():
    plot_only = '--plot-only' in sys.argv

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(os.path.join(script_dir, str(OUTPUT_DIR)))
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    if plot_only:
        print("Plot-only mode: loading from JSON...")
        with open(output_dir / 'full_results.json') as f:
            all_results = json.load(f)
        print("Generating figures...")
        make_figures(all_results, output_dir)
        print("Done!")
        return

    print("=" * 60)
    print("MEAN-QUERY FIXED GROUPING — FULL RUN")
    print("=" * 60)
    print(f"Config: {NUM_EXAMPLES} examples, {NUM_TEST_QUERIES} queries/example")
    print(f"Budgets: {BUDGETS}")
    print(f"Methods: TopK, Uniform, Oracle + "
          f"{list(GROUPING_METHODS_USED.values())} (fixed + fixed_sw + per-query)")
    print()

    t0 = time.time()
    data_path = os.path.join(script_dir, DATA_PATH)
    all_results = run_compute(data_path, output_dir)

    print("\nGenerating figures...")
    make_figures(all_results, output_dir)

    print(f"\nTotal time: {format_eta(time.time() - t0)}")
    print(f"Results in: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
