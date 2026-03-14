#!/usr/bin/env python3
"""
Grouping Methods Comparison: TopK vs Uniform vs Oracle vs Sorted-Keys Grouping

Compares TopK, Uniform, Oracle, Oracle-VW, and 5 sorted-keys grouping strategies
across fixed absolute budget sizes. Produces publication-quality figures
with mean error +/- 1 std shaded bands across multiple queries.

Output:
  - results/grouping_comparison/final_results_combined.png  (4-panel)
  - results/grouping_comparison/full_results.json
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
from pathlib import Path

from algorithms.base import softmax
from algorithms.sorted_keys_grouping import grouped_attention, GROUPING_METHODS
from algorithms.oracle import oracle_sampling
from algorithms.oracle_value_weighted import oracle_value_weighted
from visualization.plot_utils import setup_style, save_figure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ============================================================================
# CONFIG
# ============================================================================
DATA_PATH = '../../data/attention_vectors_long_bench_llama_8b.jsonl'
OUTPUT_DIR = Path('../../results/grouping_comparison')
NUM_EXAMPLES = 10
NUM_QUERIES = 50
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
BUDGETS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128]

# Grouping methods to include (excluding 'overlap' and 'threshold')
GROUPING_METHODS_USED = {k: v for k, v in GROUPING_METHODS.items() if k not in ('overlap', 'threshold')}


# ============================================================================
# BASELINES
# ============================================================================

def compute_topk(logits, values, k):
    n = len(logits)
    k = min(k, n)
    idx = np.argpartition(logits, -k)[-k:]
    w = softmax(logits[idx])
    return w @ values[idx]


def compute_uniform(logits, values, k, rng):
    n = len(logits)
    k = min(k, n)
    idx = rng.choice(n, size=k, replace=False)
    w = softmax(logits[idx])
    return w @ values[idx]


# ============================================================================
# METRICS
# ============================================================================

def rel_l2(approx, truth):
    return np.linalg.norm(approx - truth) / (np.linalg.norm(truth) + 1e-8)


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def format_eta(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


class ProgressTracker:
    def __init__(self, total_steps, prefix=""):
        self.total = total_steps
        self.done = 0
        self.start_time = time.time()
        self.prefix = prefix

    def step(self, info=""):
        self.done += 1
        elapsed = time.time() - self.start_time
        per_step = elapsed / self.done
        remaining = per_step * (self.total - self.done)
        pct = 100 * self.done / self.total
        print(f"\r  {self.prefix}[{pct:5.1f}%] "
              f"{self.done}/{self.total}  "
              f"elapsed {format_eta(elapsed)}  "
              f"ETA {format_eta(remaining)}"
              f"  {info}",
              end="", flush=True)

    def finish(self):
        elapsed = time.time() - self.start_time
        print(f"\r  {self.prefix}Done: {self.total} steps in {format_eta(elapsed)}"
              + " " * 30)


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_layer(examples, layer_name, rng):
    print(f"\n{'='*60}")
    print(f"  {layer_name}")
    print(f"{'='*60}")

    method_names = (['TopK', 'Uniform', 'Oracle', 'OracleVW']
                    + [f'group_{m}' for m in GROUPING_METHODS_USED])
    output_errors = {m: {b: [] for b in BUDGETS} for m in method_names}

    total_queries = NUM_QUERIES * len(examples)
    progress = ProgressTracker(total_queries, prefix=f"{layer_name}: ")

    for ex_idx, example in enumerate(examples):
        Q = np.array(example[layer_name]['Q'], dtype=np.float32)
        K_mat = np.array(example[layer_name]['K'], dtype=np.float32)
        V = np.array(example[layer_name]['V'], dtype=np.float32)
        seq_len = Q.shape[0]

        query_positions = list(range(seq_len - NUM_QUERIES, seq_len))

        for qi, qpos in enumerate(query_positions):
            q = Q[qpos]
            keys = K_mat[:qpos + 1]
            vals = V[:qpos + 1]
            n_keys = len(keys)

            logits = (q @ keys.T) / np.sqrt(HEAD_DIM)
            full_weights = softmax(logits)
            full_output = full_weights @ vals

            for budget in BUDGETS:
                b = min(budget, n_keys)

                to = compute_topk(logits, vals, b)
                output_errors['TopK'][budget].append(rel_l2(to, full_output))

                uo = compute_uniform(logits, vals, b, rng)
                output_errors['Uniform'][budget].append(rel_l2(uo, full_output))

                out_oracle, _ = oracle_sampling(q, keys, vals, logits, full_weights, b)
                output_errors['Oracle'][budget].append(rel_l2(out_oracle, full_output))

                out_vw, _ = oracle_value_weighted(q, keys, vals, logits, full_weights, b)
                output_errors['OracleVW'][budget].append(rel_l2(out_vw, full_output))

                for method_key in GROUPING_METHODS_USED:
                    _, go = grouped_attention(
                        logits, vals, full_weights, b, method=method_key
                    )
                    output_errors[f'group_{method_key}'][budget].append(
                        rel_l2(go, full_output)
                    )

            progress.step(f"ex {ex_idx+1}/{len(examples)} q {qi+1}/{NUM_QUERIES}")

    progress.finish()

    results = {'budgets': BUDGETS}
    for m in method_names:
        results[f'{m}_output_mean'] = [float(np.mean(output_errors[m][b])) for b in BUDGETS]
        results[f'{m}_output_std'] = [float(np.std(output_errors[m][b])) for b in BUDGETS]

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

CURVE_STYLES = {
    'TopK':              {'color': '#d62728', 'marker': 'o', 'ls': '-',  'lw': 2.8},
    'Uniform':           {'color': '#ff7f0e', 'marker': 's', 'ls': '--', 'lw': 2.5},
    'Oracle':            {'color': '#2ca02c', 'marker': '^', 'ls': '--', 'lw': 2.5},
    'OracleVW':          {'color': '#006400', 'marker': 'v', 'ls': '--', 'lw': 2.5},
    'group_equal':       {'color': '#1f77b4', 'marker': '>', 'ls': '-',  'lw': 2.0},
    'group_kmeans':      {'color': '#9467bd', 'marker': 'D', 'ls': '-',  'lw': 2.0},
    'group_log_spaced':  {'color': '#8c564b', 'marker': 'X', 'ls': '-',  'lw': 2.0},
    'group_quantile':    {'color': '#17becf', 'marker': 'h', 'ls': '-',  'lw': 2.0},
    'group_variance':    {'color': '#bcbd22', 'marker': 'd', 'ls': '-',  'lw': 2.0},
}

DISPLAY_NAMES = {
    'TopK': 'Top-K (subset softmax)',
    'Uniform': 'Uniform (subset softmax)',
    'Oracle': 'Oracle (sample ~ w)',
    'OracleVW': 'Oracle VW (sample ~ w||v||)',
}
for k, v in GROUPING_METHODS_USED.items():
    DISPLAY_NAMES[f'group_{k}'] = v

METHOD_ORDER = (['TopK', 'Uniform', 'Oracle', 'OracleVW']
                + [f'group_{m}' for m in GROUPING_METHODS_USED])


def _plot_panel(ax, data, layer_name, log_scale):
    """Plot one panel of the combined figure."""
    x = np.array(data['budgets'])
    layer_title = 'First Layer (Layer 0)' if 'first' in layer_name else 'Last Layer (Layer 31)'
    scale_label = 'Log' if log_scale else 'Linear'

    for method in METHOD_ORDER:
        mean_key = f'{method}_output_mean'
        std_key = f'{method}_output_std'
        if mean_key not in data:
            continue

        means = np.array(data[mean_key])
        stds = np.array(data[std_key])
        style = CURVE_STYLES[method]
        label = DISPLAY_NAMES[method]

        ax.plot(x, means,
                marker=style['marker'], linestyle=style['ls'],
                linewidth=style['lw'], markersize=4,
                color=style['color'], label=label, alpha=0.9, zorder=3)

        if log_scale:
            ax.fill_between(x, means, means + stds,
                             color=style['color'], alpha=0.15, zorder=1)
        else:
            ax.fill_between(x, np.maximum(means - stds, 0), means + stds,
                             color=style['color'], alpha=0.12, zorder=1)

    ax.set_xlabel('Budget (number of groups / keys)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative L2 Error', fontsize=11, fontweight='bold')
    ax.set_title(f'{layer_title} ({scale_label} Scale)',
                 fontsize=12, fontweight='bold', pad=8)

    if log_scale:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda y, _: f'{y:.3f}' if y < 0.01 else (f'{y:.2f}' if y < 1 else f'{y:.1f}')
        ))
    else:
        all_means = []
        for method in METHOD_ORDER:
            key = f'{method}_output_mean'
            if key in data:
                all_means.extend(data[key])
        y_max = max(all_means) * 1.15
        ax.set_ylim([0, y_max])

    ax.set_xlim([0, max(data['budgets']) + 3])
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')


def make_combined_figure(data_first, data_last):
    """Side-by-side log scale: first layer (left), last layer (right)."""
    fig, (ax_first, ax_last) = plt.subplots(1, 2, figsize=(22, 8.5))

    _plot_panel(ax_first, data_first, 'first_layer', log_scale=True)
    _plot_panel(ax_last,  data_last,  'last_layer',  log_scale=True)

    # Shared legend between the two panels
    handles, labels = ax_first.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.52, 1.0),
               ncol=5, fontsize=10, framealpha=0.95,
               edgecolor='#cccccc', borderpad=0.5,
               columnspacing=1.2, handletextpad=0.4)

    total_queries = NUM_EXAMPLES * NUM_QUERIES
    fig.suptitle(f'Sorted-Keys Grouping vs Baselines: Output Error (Log Scale)'
                 f'\n{NUM_EXAMPLES} examples, {NUM_QUERIES} queries each ({total_queries} total)  |  '
                 f'Llama-3-8B  |  Shaded = +1 std',
                 fontsize=14, fontweight='bold', y=1.08)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def make_single_figure(data, layer_name, log_scale=False):
    """Single-panel figure (kept for individual saves)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    _plot_panel(ax, data, layer_name, log_scale)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=9,
              edgecolor='#cccccc', ncol=2, borderpad=0.8)
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("GROUPING METHODS COMPARISON")
    print("=" * 60)
    print(f"Config: {NUM_EXAMPLES} examples, {NUM_QUERIES} queries/example")
    print(f"Layers: {LAYERS}")
    print(f"Budgets: {BUDGETS}")
    print(f"Methods: TopK, Uniform, Oracle, OracleVW + {list(GROUPING_METHODS_USED.values())}")
    print()

    t0 = time.time()
    setup_style()
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, DATA_PATH)
    output_dir = Path(os.path.join(script_dir, str(OUTPUT_DIR)))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning: {data_path}")
    with open(data_path, 'r') as f:
        total = sum(1 for _ in f)
    print(f"Found {total} examples")

    selected = sorted(rng.choice(total, NUM_EXAMPLES, replace=False).tolist())
    print(f"Selected indices: {selected}")

    selected_set = set(selected)
    examples = []
    with open(data_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx in selected_set:
                examples.append(json.loads(line))
            if len(examples) >= NUM_EXAMPLES:
                break
    print(f"Loaded {len(examples)} examples\n")

    all_results = {}
    for layer in LAYERS:
        all_results[layer] = analyze_layer(examples, layer, rng)

    results_path = output_dir / 'full_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results: {results_path}")

    # Generate figures
    print("\nGenerating figures...")

    # Combined 4-panel
    fig_combined = make_combined_figure(
        all_results['first_layer'], all_results['last_layer']
    )
    save_figure(fig_combined,
                output_dir / 'final_results_combined.png', dpi=200)

    # Individual panels
    for layer in LAYERS:
        layer_short = 'first' if 'first' in layer else 'last'
        fig_lin = make_single_figure(all_results[layer], layer, log_scale=False)
        save_figure(fig_lin, output_dir / f'final_results_{layer_short}_layer_linear.png', dpi=200)
        fig_log = make_single_figure(all_results[layer], layer, log_scale=True)
        save_figure(fig_log, output_dir / f'final_results_{layer_short}_layer_log.png', dpi=200)

    total_time = time.time() - t0
    print(f"\nTotal time: {format_eta(total_time)}")
    print("Done!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
