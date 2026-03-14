#!/usr/bin/env python3
"""
Local Query-Group Fixed Grouping — Full Comparison

Compares global fixed, local fixed (4/8/16/32 query groups), and per-query grouping.
Processes examples in batches of 50 to avoid memory issues.

Usage:
  python compare_local_grouping.py              # run compute + plot
  python compare_local_grouping.py --plot-only  # regenerate plots from JSON
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
OUTPUT_DIR = Path('../../results/local_query_grouping')
NUM_EXAMPLES = 100
NUM_TEST_QUERIES = 30
BATCH_SIZE = 50
Q_GROUPS_LIST = [4, 8, 16, 32]
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
BUDGETS = [2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024]

GROUPING_METHODS_USED = {
    k: v for k, v in GROUPING_METHODS.items()
    if k not in ('overlap', 'threshold')
}

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
        print(f"\r  {self.prefix}Done in {format_eta(time.time()-self.t0)}"
              + " " * 40)


def compute_topk(logits, values, k):
    k = min(k, len(logits))
    idx = np.argpartition(logits, -k)[-k:]
    return softmax(logits[idx]) @ values[idx]


def compute_uniform(logits, values, k, rng):
    k = min(k, len(logits))
    idx = rng.choice(len(logits), size=k, replace=False)
    return softmax(logits[idx]) @ values[idx]


def get_group_labels(method, n_keys, num_groups, sorted_weights):
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
                             head_dim, size_weighted=True):
    """Attend to G group representatives (mean_key, mean_value)."""
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
# PRE-COMPUTE STRUCTURES
# ============================================================================

def build_fixed_grouping(mean_q, K_mat, seq_len):
    """Build sorted indices + labels for a single mean query."""
    logits = (mean_q @ K_mat[:seq_len].T) / np.sqrt(HEAD_DIM)
    weights = softmax(logits)
    si = np.argsort(logits)[::-1]
    sw = weights[si]

    labels = {}
    for mk in GROUPING_METHODS_USED:
        labels[mk] = {}
        for budget in BUDGETS:
            labels[mk][budget] = get_group_labels(mk, seq_len, budget, sw)

    return si, labels


def build_local_groups(Q, K_mat, seq_len, n_qgroups):
    """
    Split queries into n_qgroups contiguous groups.
    Returns mean queries and pre-computed groupings per local group.
    """
    boundaries = np.round(np.linspace(0, seq_len, n_qgroups + 1)).astype(int)

    qgroup_means = np.zeros((n_qgroups, HEAD_DIM), dtype=np.float64)
    qgroup_si = []      # sorted_indices per group
    qgroup_labels = []   # {method: {budget: labels}} per group

    for g in range(n_qgroups):
        start, end = boundaries[g], boundaries[g + 1]
        if end <= start:
            qgroup_means[g] = qgroup_means[max(0, g - 1)]
            qgroup_si.append(qgroup_si[-1] if qgroup_si else None)
            qgroup_labels.append(qgroup_labels[-1] if qgroup_labels else None)
            continue

        mean_q_g = Q[start:end].mean(axis=0)
        qgroup_means[g] = mean_q_g

        si_g, labels_g = build_fixed_grouping(mean_q_g, K_mat, seq_len)
        qgroup_si.append(si_g)
        qgroup_labels.append(labels_g)

    return qgroup_means, qgroup_si, qgroup_labels


def find_closest_qgroup(query, qgroup_means):
    """Cosine similarity to pick closest query group."""
    q_norm = query / (np.linalg.norm(query) + 1e-10)
    m_norms = qgroup_means / (
        np.linalg.norm(qgroup_means, axis=1, keepdims=True) + 1e-10
    )
    return int(np.argmax(m_norms @ q_norm))


# ============================================================================
# METHOD NAMES
# ============================================================================

def get_method_names():
    names = ['TopK', 'Uniform', 'Oracle']
    # Global fixed
    for mk in GROUPING_METHODS_USED:
        names.append(f'global_fixed_{mk}')
    # Local fixed for each Q_GROUPS value
    for nqg in Q_GROUPS_LIST:
        for mk in GROUPING_METHODS_USED:
            names.append(f'local{nqg}_{mk}')
    # Per-query
    for mk in GROUPING_METHODS_USED:
        names.append(f'perquery_{mk}')
    return names


# ============================================================================
# PROCESS ONE BATCH OF EXAMPLES
# ============================================================================

def process_batch(examples, layer_name, rng, errors, progress, ex_offset):
    """Process a batch of examples, accumulating into errors dict."""

    for ex_idx, example in enumerate(examples):
        Q = np.array(example[layer_name]['Q'], dtype=np.float32)
        K_mat = np.array(example[layer_name]['K'], dtype=np.float32)
        V = np.array(example[layer_name]['V'], dtype=np.float32)
        seq_len = Q.shape[0]

        # ---- Global fixed ----
        global_mean_q = Q.mean(axis=0)
        global_si, global_labels = build_fixed_grouping(
            global_mean_q, K_mat, seq_len
        )

        # ---- Local fixed for each Q_GROUPS value ----
        local_data = {}
        for nqg in Q_GROUPS_LIST:
            means, si_list, labels_list = build_local_groups(
                Q, K_mat, seq_len, nqg
            )
            local_data[nqg] = (means, si_list, labels_list)

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

            # Causal mask for global
            g_causal = global_si < n_keys
            g_valid_si = global_si[g_causal]

            # Pre-compute causal masks + closest group for each local config
            local_causal = {}
            for nqg in Q_GROUPS_LIST:
                means, si_list, labels_list = local_data[nqg]
                closest = find_closest_qgroup(q, means)
                l_si = si_list[closest]
                l_causal = l_si < n_keys
                local_causal[nqg] = (closest, l_si, l_causal, labels_list)

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
                    # Global fixed
                    gl = global_labels[mk][budget][g_causal]
                    fo_g, _ = fixed_grouping_attention(
                        q, keys, vals, g_valid_si, gl, HEAD_DIM
                    )
                    errors[f'global_fixed_{mk}'][budget].append(
                        rel_l2(fo_g, full_out)
                    )

                    # Local fixed for each Q_GROUPS
                    for nqg in Q_GROUPS_LIST:
                        closest, l_si, l_causal_mask, labels_list = \
                            local_causal[nqg]
                        ll = labels_list[closest][mk][budget][l_causal_mask]
                        l_valid_si = l_si[l_causal_mask]
                        fo_l, _ = fixed_grouping_attention(
                            q, keys, vals, l_valid_si, ll, HEAD_DIM
                        )
                        errors[f'local{nqg}_{mk}'][budget].append(
                            rel_l2(fo_l, full_out)
                        )

                    # Per-query
                    _, go = grouped_attention(
                        logits, vals, full_w, b, method=mk
                    )
                    errors[f'perquery_{mk}'][budget].append(
                        rel_l2(go, full_out)
                    )

            global_ex = ex_offset + ex_idx + 1
            progress.step(
                f"ex {global_ex}/{NUM_EXAMPLES} q {qi+1}/{NUM_TEST_QUERIES}"
            )

        # Free memory after each example
        del Q, K_mat, V


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_layer(data_path, selected_indices, layer_name, rng):
    print(f"\n{'='*60}")
    print(f"  {layer_name}")
    print(f"{'='*60}")

    method_names = get_method_names()
    errors = {m: {b: [] for b in BUDGETS} for m in method_names}

    total_queries = NUM_TEST_QUERIES * NUM_EXAMPLES
    progress = ProgressTracker(total_queries, f"{layer_name}: ")

    # Process in batches
    selected_set = set(selected_indices)
    batch = []
    batch_start_ex = 0

    with open(data_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx not in selected_set:
                continue
            batch.append(json.loads(line))

            if len(batch) >= BATCH_SIZE:
                process_batch(
                    batch, layer_name, rng, errors, progress, batch_start_ex
                )
                batch_start_ex += len(batch)
                batch = []

    # Process remaining
    if batch:
        process_batch(batch, layer_name, rng, errors, progress, batch_start_ex)

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


# ============================================================================
# PLOTTING
# ============================================================================

def _plot_best_method_comparison(ax, data, layer_name, grouping_method='equal'):
    """
    For one grouping method: plot global, local(4/8/16/32), per-query + baselines.
    """
    x = np.array(data['budgets'])
    mk = grouping_method
    display = GROUPING_METHODS_USED[mk]

    # Baselines
    for name, color, marker in [('TopK', '#d62728', 'o'),
                                 ('Uniform', '#ff7f0e', 's'),
                                 ('Oracle', '#2ca02c', '^')]:
        means = np.array(data[f'{name}_mean'])
        stds = np.array(data[f'{name}_std'])
        ax.plot(x, means, marker=marker, color=color, lw=2.5,
                label=name, zorder=4, markersize=5)
        ax.fill_between(x, means, means + stds, color=color, alpha=0.12)

    # Global
    m_g = np.array(data[f'global_fixed_{mk}_mean'])
    s_g = np.array(data[f'global_fixed_{mk}_std'])
    ax.plot(x, m_g, ls=':', color='gray', lw=2, marker='s', markersize=4,
            label='Global (1 group)', zorder=3)
    ax.fill_between(x, m_g, m_g + s_g, color='gray', alpha=0.08)

    # Local variants
    local_colors = {4: '#e377c2', 8: '#8c564b', 16: '#1f77b4', 32: '#17becf'}
    local_markers = {4: 'v', 8: 'D', 16: 'o', 32: '^'}
    for nqg in Q_GROUPS_LIST:
        c = local_colors[nqg]
        m_l = np.array(data[f'local{nqg}_{mk}_mean'])
        s_l = np.array(data[f'local{nqg}_{mk}_std'])
        ax.plot(x, m_l, ls='-', color=c, lw=2.2,
                marker=local_markers[nqg], markersize=5,
                label=f'Local ({nqg} groups)', zorder=3)
        ax.fill_between(x, m_l, m_l + s_l, color=c, alpha=0.08)

    # Per-query
    m_pq = np.array(data[f'perquery_{mk}_mean'])
    s_pq = np.array(data[f'perquery_{mk}_std'])
    ax.plot(x, m_pq, ls='--', color='#2ca02c', lw=2, marker='x', markersize=5,
            label='Per-Query', zorder=3, alpha=0.7)
    ax.fill_between(x, m_pq, m_pq + s_pq, color='#2ca02c', alpha=0.06)

    layer_title = ('First Layer (Layer 0)' if 'first' in layer_name
                   else 'Last Layer (Layer 31)')
    ax.set_title(f'{layer_title} — {display}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Budget (num groups)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative L2 Error', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, ls='--', which='both')


def _plot_all_methods_one_variant(ax, data, layer_name, variant_prefix,
                                  variant_label):
    """Plot all grouping methods for one variant + baselines."""
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
        key = f'{variant_prefix}_{mk}_mean'
        key_s = f'{variant_prefix}_{mk}_std'
        if key not in data:
            continue
        means = np.array(data[key])
        stds = np.array(data[key_s])
        ax.plot(x, means, ls='-', color=c, lw=2.2,
                marker=markers[i % len(markers)], markersize=5,
                label=display, zorder=3)
        ax.fill_between(x, means, means + stds, color=c, alpha=0.1)

    layer_title = ('First Layer (Layer 0)' if 'first' in layer_name
                   else 'Last Layer (Layer 31)')
    ax.set_title(f'{layer_title} — {variant_label}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Budget (num groups)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative L2 Error', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, ls='--', which='both')


def make_figures(all_results, output_dir):
    cfg = all_results.get('config', {})
    n_ex = cfg.get('num_examples', NUM_EXAMPLES)
    n_q = cfg.get('num_test_queries', NUM_TEST_QUERIES)
    subtitle = (f'{n_ex} examples, {n_q} queries each  |  '
                f'Llama-3-8B  |  Shaded = +1 std')

    # Figure 1: Per grouping method — global vs local(4/8/16/32) vs per-query
    for mk, display in GROUPING_METHODS_USED.items():
        fig, axes = plt.subplots(1, 2, figsize=(22, 8.5))
        for ax, layer in zip(axes, LAYERS):
            _plot_best_method_comparison(ax, all_results[layer], layer, mk)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   ncol=5, fontsize=9.5, framealpha=0.95,
                   columnspacing=1.0, handletextpad=0.4)
        fig.suptitle(
            f'Global vs Local vs Per-Query — {display}\n{subtitle}',
            fontsize=14, fontweight='bold', y=1.06)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        save_figure(fig, output_dir / f'locality_{mk}.png', dpi=200)
        plt.close(fig)

    # Figure 2: All grouping methods for each variant
    variants = [('global_fixed', 'Global Fixed (1 mean query)')]
    for nqg in Q_GROUPS_LIST:
        variants.append((f'local{nqg}', f'Local Fixed ({nqg} query groups)'))
    variants.append(('perquery', 'Per-Query'))

    for prefix, label in variants:
        fig, axes = plt.subplots(1, 2, figsize=(22, 8.5))
        for ax, layer in zip(axes, LAYERS):
            _plot_all_methods_one_variant(
                ax, all_results[layer], layer, prefix, label
            )
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   ncol=4, fontsize=10, framealpha=0.95)
        fig.suptitle(f'{label} vs Baselines\n{subtitle}',
                     fontsize=14, fontweight='bold', y=1.06)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        safe_name = prefix.replace('/', '_')
        save_figure(fig, output_dir / f'methods_{safe_name}.png', dpi=200)
        plt.close(fig)


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
    print("LOCAL QUERY-GROUP FIXED GROUPING — FULL COMPARISON")
    print("=" * 60)
    print(f"Config: {NUM_EXAMPLES} examples, {NUM_TEST_QUERIES} queries/example")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Local query groups: {Q_GROUPS_LIST}")
    print(f"Budgets: {BUDGETS}")
    print(f"Grouping methods: {list(GROUPING_METHODS_USED.values())}")
    print()

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    data_path = os.path.join(script_dir, DATA_PATH)

    print(f"Scanning: {data_path}")
    with open(data_path, 'r') as f:
        total = sum(1 for _ in f)
    print(f"Found {total} examples")

    n_select = min(NUM_EXAMPLES, total)
    selected = sorted(rng.choice(total, n_select, replace=False).tolist())
    print(f"Selected {n_select} examples: {selected[:10]}{'...' if n_select > 10 else ''}")

    all_results = {
        'config': {
            'num_examples': n_select,
            'num_test_queries': NUM_TEST_QUERIES,
            'q_groups_list': Q_GROUPS_LIST,
            'budgets': BUDGETS,
            'seed': SEED,
            'batch_size': BATCH_SIZE,
            'method_names': get_method_names(),
        }
    }

    for layer in LAYERS:
        all_results[layer] = analyze_layer(data_path, selected, layer, rng)

    with open(output_dir / 'full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_dir / 'full_results.json'}")

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
