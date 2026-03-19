#!/usr/bin/env python3
"""
Clustering Baselines by Query Position Experiment

Same as compare_clustering_baselines.py but evaluates three query regions:
  - First 30 queries (early positions — short causal context)
  - Middle 30 queries (center of the sequence)
  - Last 30 queries (late positions — full context)

Produces per-region error curves and a combined comparison plot.

Usage:
  python compare_clustering_by_position.py              # run compute + plot
  python compare_clustering_by_position.py --plot-only  # regenerate plots from JSON
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

from algorithms.base import softmax
from algorithms.oracle import oracle_sampling
from algorithms.sorted_keys_grouping import grouped_attention, group_quantile_weight
from visualization.plot_utils import setup_style, save_figure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
DATA_PATH = '../../data/attention_vectors_long_bench_llama_8b.jsonl'
OUTPUT_DIR = Path('../../results/clustering_by_position')
NUM_EXAMPLES = 10
NUM_TEST_QUERIES = 30
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
BUDGETS = [2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024]
Q_CLUSTERS_LIST = [2, 4, 8, 16, 32]
LOCAL_Q_GROUPS = [8, 32]

REGIONS = ['first', 'middle', 'last']
REGION_DISPLAY = {
    'first':  'First 30 Queries (Early Positions)',
    'middle': 'Middle 30 Queries (Center Positions)',
    'last':   'Last 30 Queries (Late Positions)',
}

METHOD_NAMES = [
    'Oracle', 'Uniform', 'Fixed Quantile', 'Per-Query Quantile',
    'KMeans Keys', 'Query KMeans',
] + [f'Local KMeans Q={nq}' for nq in Q_CLUSTERS_LIST] + [
    f'Local Query Sort {nq}' for nq in LOCAL_Q_GROUPS
]

METHOD_COLORS = {
    'Oracle':             '#2ca02c',
    'Uniform':            '#7fbf7f',
    'KMeans Keys':        'darkorange',
    'Query KMeans':       '#e8a020',
    'Fixed Quantile':     'lightskyblue',
    'Per-Query Quantile': 'navy',
    'Local KMeans Q=8':   'mediumslateblue',
    'Local KMeans Q=32':  'steelblue',
    'Local Query Sort 8':  'teal',
    'Local Query Sort 32': 'darkslategray',
}
for nq in Q_CLUSTERS_LIST:
    if f'Local KMeans Q={nq}' not in METHOD_COLORS:
        METHOD_COLORS[f'Local KMeans Q={nq}'] = '#cccccc'

METHOD_MARKERS = {
    'Oracle':             '^',
    'Uniform':            's',
    'Fixed Quantile':     'o',
    'Per-Query Quantile': 'D',
    'KMeans Keys':        'X',
    'Query KMeans':       'P',
}
_LOCAL_MARKERS = ['v', 'D', 'o', '^', 'h']
for i, nq in enumerate(Q_CLUSTERS_LIST):
    METHOD_MARKERS[f'Local KMeans Q={nq}'] = _LOCAL_MARKERS[i]
METHOD_MARKERS['Local Query Sort 8'] = 'd'
METHOD_MARKERS['Local Query Sort 32'] = '>'

METHOD_LINESTYLES = {
    'Oracle':             '-',
    'Uniform':            '-',
    'Fixed Quantile':     '-',
    'Per-Query Quantile': '--',
    'KMeans Keys':        '-',
    'Query KMeans':       '-',
}
for nq in Q_CLUSTERS_LIST:
    METHOD_LINESTYLES[f'Local KMeans Q={nq}'] = '-'
for nq in LOCAL_Q_GROUPS:
    METHOD_LINESTYLES[f'Local Query Sort {nq}'] = '-'

PLOT_METHODS = [
    'Oracle', 'Uniform', 'KMeans Keys', 'Local KMeans Q=32',
]

PLOT_DISPLAY_NAMES = {
    'Local KMeans Q=8':    'Sorting KMeans 8',
    'Local KMeans Q=32':   'Sorting KMeans 32',
    'Local Query Sort 8':  'Local Query Sort 8',
    'Local Query Sort 32': 'Local Query Sort 32',
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


# ============================================================================
# QUERY POSITION SELECTION
# ============================================================================

def select_query_positions(seq_len, num_queries):
    """Return dict of {region: list_of_positions} for first/middle/last."""
    positions = {}

    # First: positions starting from 30 (need at least some causal context)
    # We skip position 0 (only 1 key) and start at a minimum offset so
    # budgets up to 1024 don't exceed n_keys too badly.
    # Use positions [min_offset .. min_offset + num_queries)
    min_offset = max(30, BUDGETS[-1])  # ensure at least 1024 keys available
    first_start = min(min_offset, seq_len - num_queries)
    first_start = max(0, first_start)
    positions['first'] = list(range(first_start, first_start + num_queries))

    # Middle: centered in the sequence
    mid_center = seq_len // 2
    mid_start = mid_center - num_queries // 2
    mid_start = max(0, min(mid_start, seq_len - num_queries))
    positions['middle'] = list(range(mid_start, mid_start + num_queries))

    # Last: last num_queries positions
    last_start = max(0, seq_len - num_queries)
    positions['last'] = list(range(last_start, last_start + num_queries))

    return positions


# ============================================================================
# FIXED QUANTILE GROUPING
# ============================================================================

def build_fixed_quantile_grouping(mean_q, K_mat, n_keys):
    logits = (mean_q @ K_mat[:n_keys].T) / np.sqrt(HEAD_DIM)
    weights = softmax(logits)
    sorted_indices = np.argsort(logits)[::-1]
    sorted_weights = weights[sorted_indices]

    labels = {}
    for budget in BUDGETS:
        b = min(budget, n_keys)
        labels[budget] = group_quantile_weight(sorted_weights, b)

    return sorted_indices, labels


def fixed_grouping_attention(query, keys, values, sorted_indices, labels, head_dim):
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
    group_logits = group_logits + np.log(group_sizes + 1e-10)
    group_weights = softmax(group_logits)

    return group_weights @ mean_values, G


# ============================================================================
# K-MEANS KEY CLUSTERING
# ============================================================================

def build_kmeans_key_groups(keys, n_keys):
    groups = {}
    for budget in BUDGETS:
        b = min(budget, n_keys)
        if b >= n_keys:
            groups[budget] = np.arange(n_keys)
        else:
            km = KMeans(n_clusters=b, n_init=3, max_iter=100, random_state=SEED)
            km.fit(keys[:n_keys])
            groups[budget] = km.labels_
    return groups


def kmeans_key_attention(query, keys, values, labels, head_dim):
    unique_labels = np.unique(labels)
    G = len(unique_labels)

    mean_keys = np.zeros((G, head_dim), dtype=np.float64)
    mean_values = np.zeros((G, head_dim), dtype=np.float64)
    group_sizes = np.zeros(G, dtype=np.float64)

    for i, g in enumerate(unique_labels):
        mask = labels == g
        mean_keys[i] = keys[mask].mean(axis=0)
        mean_values[i] = values[mask].mean(axis=0)
        group_sizes[i] = mask.sum()

    group_logits = (query @ mean_keys.T) / np.sqrt(head_dim)
    group_logits = group_logits + np.log(group_sizes + 1e-10)
    group_weights = softmax(group_logits)

    return group_weights @ mean_values, G


# ============================================================================
# LOCAL QUERY SORTING (contiguous position-based splits)
# ============================================================================

def build_local_query_groups(Q, K_mat, seq_len, n_qgroups):
    boundaries = np.round(np.linspace(0, seq_len, n_qgroups + 1)).astype(int)

    qgroup_means = np.zeros((n_qgroups, HEAD_DIM), dtype=np.float64)
    qgroup_groupings = []

    for g in range(n_qgroups):
        start, end = boundaries[g], boundaries[g + 1]
        if end <= start:
            qgroup_means[g] = qgroup_means[max(0, g - 1)]
            qgroup_groupings.append(qgroup_groupings[-1] if qgroup_groupings else None)
            continue

        mean_q_g = Q[start:end].mean(axis=0)
        qgroup_means[g] = mean_q_g
        si_g, labels_g = build_fixed_quantile_grouping(mean_q_g, K_mat, seq_len)
        qgroup_groupings.append((si_g, labels_g))

    return qgroup_means, qgroup_groupings


def find_closest_qgroup(query, qgroup_means):
    q_norm = query / (np.linalg.norm(query) + 1e-10)
    m_norms = qgroup_means / (
        np.linalg.norm(qgroup_means, axis=1, keepdims=True) + 1e-10
    )
    return int(np.argmax(m_norms @ q_norm))


# ============================================================================
# QUERY K-MEANS CLUSTERING
# ============================================================================

def build_query_kmeans(Q, n_clusters):
    km = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=SEED)
    km.fit(Q)
    return km.labels_, km.cluster_centers_


def query_kmeans_attention(cluster_mean_q, keys, values, head_dim):
    logits = (cluster_mean_q @ keys.T) / np.sqrt(head_dim)
    w = softmax(logits)
    return w @ values


# ============================================================================
# MAIN COMPUTATION
# ============================================================================

def analyze_layer(data_path, selected_indices, layer_name, rng):
    print(f"\n{'='*60}")
    print(f"  {layer_name}")
    print(f"{'='*60}")

    # errors[region][method][budget] = list of floats
    errors = {
        region: {m: {b: [] for b in BUDGETS} for m in METHOD_NAMES}
        for region in REGIONS
    }

    total_queries = NUM_TEST_QUERIES * len(REGIONS) * NUM_EXAMPLES
    progress = ProgressTracker(total_queries, f"{layer_name}: ")

    selected_set = set(selected_indices)
    ex_count = 0

    with open(data_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx not in selected_set:
                continue

            example = json.loads(line)
            Q = np.array(example[layer_name]['Q'], dtype=np.float32)
            K_mat = np.array(example[layer_name]['K'], dtype=np.float32)
            V = np.array(example[layer_name]['V'], dtype=np.float32)
            seq_len = Q.shape[0]

            # ---- Select query positions for each region ----
            region_positions = select_query_positions(seq_len, NUM_TEST_QUERIES)

            # ---- Pre-compute: Fixed Quantile (global mean query) ----
            global_mean_q = Q.mean(axis=0)
            fixed_si, fixed_labels = build_fixed_quantile_grouping(
                global_mean_q, K_mat, seq_len
            )

            # ---- Pre-compute: KMeans on keys ----
            kmeans_key_labels = build_kmeans_key_groups(K_mat, seq_len)

            # ---- Pre-compute: Query KMeans ----
            query_kmeans_data = {}
            for budget in BUDGETS:
                b = min(budget, seq_len)
                if b >= seq_len:
                    query_kmeans_data[budget] = None
                else:
                    q_labels, q_centers = build_query_kmeans(Q[:seq_len], b)
                    query_kmeans_data[budget] = (q_labels, q_centers)

            # ---- Pre-compute: Local KMeans Query grouping ----
            local_kmeans_data = {}
            for nq in Q_CLUSTERS_LIST:
                nq_actual = min(nq, seq_len)
                q_labels, q_centers = build_query_kmeans(Q[:seq_len], nq_actual)
                cluster_groupings = []
                for c in range(nq_actual):
                    si_c, labels_c = build_fixed_quantile_grouping(
                        q_centers[c], K_mat, seq_len
                    )
                    cluster_groupings.append((si_c, labels_c))
                local_kmeans_data[nq] = (q_labels, cluster_groupings)

            # ---- Pre-compute: Local Query Sorting ----
            local_sort_data = {}
            for nq in LOCAL_Q_GROUPS:
                means_lq, groupings_lq = build_local_query_groups(
                    Q, K_mat, seq_len, nq
                )
                local_sort_data[nq] = (means_lq, groupings_lq)

            ex_count += 1

            # ---- Evaluate each region ----
            for region in REGIONS:
                test_positions = region_positions[region]

                for qi, qpos in enumerate(test_positions):
                    q = Q[qpos]
                    keys = K_mat[:qpos + 1]
                    vals = V[:qpos + 1]
                    n_keys = qpos + 1

                    logits = (q @ keys.T) / np.sqrt(HEAD_DIM)
                    full_w = softmax(logits)
                    full_out = full_w @ vals

                    causal_mask = fixed_si < n_keys
                    valid_si = fixed_si[causal_mask]

                    for budget in BUDGETS:
                        b = min(budget, n_keys)

                        # --- Oracle ---
                        out_oracle, _ = oracle_sampling(
                            q, keys, vals, logits, full_w, b
                        )
                        errors[region]['Oracle'][budget].append(
                            rel_l2(out_oracle, full_out)
                        )

                        # --- Uniform ---
                        u_idx = rng.choice(n_keys, size=b, replace=False)
                        out_uniform = softmax(logits[u_idx]) @ vals[u_idx]
                        errors[region]['Uniform'][budget].append(
                            rel_l2(out_uniform, full_out)
                        )

                        # --- Fixed Quantile ---
                        fl = fixed_labels[budget][causal_mask]
                        out_fq, _ = fixed_grouping_attention(
                            q, keys, vals, valid_si, fl, HEAD_DIM
                        )
                        errors[region]['Fixed Quantile'][budget].append(
                            rel_l2(out_fq, full_out)
                        )

                        # --- Per-Query Quantile ---
                        _, out_pq = grouped_attention(
                            logits, vals, full_w, b, method='quantile'
                        )
                        errors[region]['Per-Query Quantile'][budget].append(
                            rel_l2(out_pq, full_out)
                        )

                        # --- KMeans Key Clustering ---
                        km_labels = kmeans_key_labels[budget]
                        km_causal = km_labels[:n_keys]
                        out_km, _ = kmeans_key_attention(
                            q, keys, vals, km_causal, HEAD_DIM
                        )
                        errors[region]['KMeans Keys'][budget].append(
                            rel_l2(out_km, full_out)
                        )

                        # --- Query KMeans ---
                        qkm = query_kmeans_data[budget]
                        if qkm is None:
                            out_qkm = full_out
                        else:
                            q_labels, q_centers = qkm
                            cluster_id = q_labels[qpos]
                            mean_q_cluster = q_centers[cluster_id]
                            out_qkm = query_kmeans_attention(
                                mean_q_cluster, keys, vals, HEAD_DIM
                            )
                        errors[region]['Query KMeans'][budget].append(
                            rel_l2(out_qkm, full_out)
                        )

                        # --- Local KMeans Query ---
                        for nq in Q_CLUSTERS_LIST:
                            q_labels_lk, cluster_groupings = local_kmeans_data[nq]
                            cluster_id_lk = q_labels_lk[qpos]
                            si_lk, labels_lk = cluster_groupings[cluster_id_lk]
                            cm_lk = si_lk < n_keys
                            vsi_lk = si_lk[cm_lk]
                            fl_lk = labels_lk[budget][cm_lk]
                            out_lk, _ = fixed_grouping_attention(
                                q, keys, vals, vsi_lk, fl_lk, HEAD_DIM
                            )
                            errors[region][f'Local KMeans Q={nq}'][budget].append(
                                rel_l2(out_lk, full_out)
                            )

                        # --- Local Query Sorting ---
                        for nq in LOCAL_Q_GROUPS:
                            means_lq, groupings_lq = local_sort_data[nq]
                            closest = find_closest_qgroup(q, means_lq)
                            si_lq, labels_lq = groupings_lq[closest]
                            cm_lq = si_lq < n_keys
                            vsi_lq = si_lq[cm_lq]
                            fl_lq = labels_lq[budget][cm_lq]
                            out_lq, _ = fixed_grouping_attention(
                                q, keys, vals, vsi_lq, fl_lq, HEAD_DIM
                            )
                            errors[region][f'Local Query Sort {nq}'][budget].append(
                                rel_l2(out_lq, full_out)
                            )

                    progress.step(
                        f"ex {ex_count}/{NUM_EXAMPLES} {region} q {qi+1}/{NUM_TEST_QUERIES}"
                    )

            del Q, K_mat, V

    progress.finish()

    # Aggregate per region
    results = {'budgets': BUDGETS}
    for region in REGIONS:
        results[region] = {}
        for m in METHOD_NAMES:
            results[region][f'{m}_mean'] = [
                float(np.mean(errors[region][m][b])) for b in BUDGETS
            ]
            results[region][f'{m}_std'] = [
                float(np.std(errors[region][m][b])) for b in BUDGETS
            ]
    return results


# ============================================================================
# PLOTTING
# ============================================================================

def _plot_comparison(ax, data, region, layer_name):
    """Plot selected methods on one axis for a single region."""
    x = np.array(data['budgets'])
    region_data = data[region]

    for method in PLOT_METHODS:
        if f'{method}_mean' not in region_data:
            continue
        means = np.array(region_data[f'{method}_mean'])
        stds = np.array(region_data[f'{method}_std'])
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        ls = METHOD_LINESTYLES[method]
        label = PLOT_DISPLAY_NAMES.get(method, method)

        ax.plot(x, means, marker=marker, color=color, lw=2.5, ls=ls,
                label=label, zorder=4, markersize=6)
        hi = means + stds
        ax.fill_between(x, means, hi, color=color, alpha=0.12)

    ax.set_title(REGION_DISPLAY[region], fontsize=12, fontweight='bold')
    ax.set_xlabel('Budget (num groups / clusters)', fontsize=10)
    ax.set_ylabel('Relative L2 Error', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(left=0, right=512)
    ax.grid(True, alpha=0.3, ls='--', which='both')


def _plot_region_overlay(ax, data, method, layer_name):
    """Plot one method across all three regions on one axis."""
    x = np.array(data['budgets'])
    region_colors = {'first': '#e41a1c', 'middle': '#377eb8', 'last': '#4daf4a'}
    region_markers = {'first': 'o', 'middle': 's', 'last': '^'}

    for region in REGIONS:
        region_data = data[region]
        if f'{method}_mean' not in region_data:
            continue
        means = np.array(region_data[f'{method}_mean'])
        stds = np.array(region_data[f'{method}_std'])
        color = region_colors[region]
        marker = region_markers[region]

        ax.plot(x, means, marker=marker, color=color, lw=2.5,
                label=f'{region.capitalize()} queries', zorder=4, markersize=6)
        hi = means + stds
        ax.fill_between(x, means, hi, color=color, alpha=0.12)

    display = PLOT_DISPLAY_NAMES.get(method, method)
    ax.set_title(display, fontsize=12, fontweight='bold')
    ax.set_xlabel('Budget', fontsize=10)
    ax.set_ylabel('Relative L2 Error', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(left=0, right=512)
    ax.grid(True, alpha=0.3, ls='--', which='both')


def make_figures(all_results, output_dir):
    cfg = all_results.get('config', {})
    n_ex = cfg.get('num_examples', NUM_EXAMPLES)
    n_q = cfg.get('num_test_queries', NUM_TEST_QUERIES)
    subtitle = (f'{n_ex} examples, {n_q} queries each  |  '
                f'Llama-3-8B  |  Shaded = +1 std')

    for layer in LAYERS:
        layer_data = all_results[layer]
        layer_short = 'first_layer' if 'first' in layer else 'last_layer'
        layer_title = ('First Layer (Layer 0)' if 'first' in layer
                       else 'Last Layer (Layer 31)')

        # --- Figure 1: Three subplots, one per region ---
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
        for i, region in enumerate(REGIONS):
            _plot_comparison(axes[i], layer_data, region, layer)
            if i > 0:
                axes[i].set_ylabel('')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', fontsize=10,
                   framealpha=0.95, ncol=5, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(
            f'Clustering Baselines by Query Position — {layer_title}\n{subtitle}',
            fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.06, 1, 0.94])
        save_figure(fig, output_dir / f'by_region_{layer_short}.png', dpi=200)
        plt.close(fig)

        # --- Figure 2: Per-method overlay (first/middle/last on same axes) ---
        n_methods = len(PLOT_METHODS)
        ncols = 3
        nrows = (n_methods + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows),
                                 sharey=True)
        axes_flat = axes.flatten()

        for i, method in enumerate(PLOT_METHODS):
            _plot_region_overlay(axes_flat[i], layer_data, method, layer)
            if i == 0:
                axes_flat[i].legend(fontsize=10, framealpha=0.95)

        # Hide unused axes
        for j in range(n_methods, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f'Per-Method Position Comparison — {layer_title}\n{subtitle}',
            fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, output_dir / f'per_method_{layer_short}.png', dpi=200)
        plt.close(fig)

        # --- Figure 3: Summary — key methods only, overlay regions ---
        key_methods = ['Oracle', 'Uniform', 'KMeans Keys', 'Local KMeans Q=32']
        fig, axes = plt.subplots(1, len(key_methods), figsize=(6 * len(key_methods), 6),
                                 sharey=True)
        for i, method in enumerate(key_methods):
            _plot_region_overlay(axes[i], layer_data, method, layer)
            if i == 0:
                axes[i].legend(fontsize=10, framealpha=0.95)
            if i > 0:
                axes[i].set_ylabel('')

        fig.suptitle(
            f'Position Effect on Key Methods — {layer_title}\n{subtitle}',
            fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        save_figure(fig, output_dir / f'summary_{layer_short}.png', dpi=200)
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
    print("CLUSTERING BASELINES BY QUERY POSITION")
    print("=" * 60)
    print(f"Config: {NUM_EXAMPLES} examples, {NUM_TEST_QUERIES} queries/region")
    print(f"Regions: {REGIONS} (3x {NUM_TEST_QUERIES} = {3*NUM_TEST_QUERIES} queries/example)")
    print(f"Budgets: {BUDGETS}")
    print(f"Methods: {len(METHOD_NAMES)}")
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
    print(f"Selected {n_select} examples: {selected}")

    all_results = {
        'config': {
            'num_examples': n_select,
            'num_test_queries': NUM_TEST_QUERIES,
            'budgets': BUDGETS,
            'seed': SEED,
            'head_dim': HEAD_DIM,
            'method_names': METHOD_NAMES,
            'regions': REGIONS,
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
