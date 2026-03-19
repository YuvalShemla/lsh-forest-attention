#!/usr/bin/env python3
"""
Incremental Frozen-Assignment Clustering Experiment

Cluster assignments are precomputed ONCE on all keys. At inference time,
keys are revealed causally (one at a time). Each revealed key updates its
cluster's running mean key, mean value, and count. Queries attend only
over the current state of non-empty cluster representatives.

This is O(N*G*d) — linear in sequence length for fixed budget G.

Three query regions (first/middle/last) to measure how the incremental
build-up affects accuracy at different positions.

Baselines (Oracle, Uniform, Per-Query Quantile) use causal keys directly.

Usage:
  python compare_clustering_incremental.py              # run compute + plot
  python compare_clustering_incremental.py --plot-only  # regenerate plots from JSON
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
OUTPUT_DIR = Path('../../results/clustering_incremental')
NUM_EXAMPLES = 10
NUM_TEST_QUERIES = 30
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
BUDGETS = [2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024]
Q_CLUSTERS_LIST = [8, 32]
LOCAL_Q_GROUPS = [8, 32]

REGIONS = ['first', 'middle', 'last']
REGION_DISPLAY = {
    'first':  'First 30 Queries (Early Positions)',
    'middle': 'Middle 30 Queries (Center Positions)',
    'last':   'Last 30 Queries (Late Positions)',
}

METHOD_NAMES = [
    'Oracle', 'Uniform', 'Per-Query Quantile',
    'Fixed Quantile (incr)', 'KMeans Keys (incr)',
] + [f'Local KMeans Q={nq} (incr)' for nq in Q_CLUSTERS_LIST] + [
    f'Local Query Sort {nq} (incr)' for nq in LOCAL_Q_GROUPS
]

METHOD_COLORS = {
    'Oracle':                         '#2ca02c',
    'Uniform':                        '#7fbf7f',
    'Per-Query Quantile':             'navy',
    'Fixed Quantile (incr)':          'lightskyblue',
    'KMeans Keys (incr)':             'darkorange',
    'Local KMeans Q=8 (incr)':        'mediumslateblue',
    'Local KMeans Q=32 (incr)':       'steelblue',
    'Local Query Sort 8 (incr)':      'teal',
    'Local Query Sort 32 (incr)':     'darkslategray',
}

METHOD_MARKERS = {
    'Oracle':                         '^',
    'Uniform':                        's',
    'Per-Query Quantile':             'D',
    'Fixed Quantile (incr)':          'o',
    'KMeans Keys (incr)':             'X',
    'Local KMeans Q=8 (incr)':        'v',
    'Local KMeans Q=32 (incr)':       'h',
    'Local Query Sort 8 (incr)':      'd',
    'Local Query Sort 32 (incr)':     '>',
}

METHOD_LINESTYLES = {
    'Oracle':                         '-',
    'Uniform':                        '-',
    'Per-Query Quantile':             '--',
    'Fixed Quantile (incr)':          '-',
    'KMeans Keys (incr)':             '-',
    'Local KMeans Q=8 (incr)':        '-',
    'Local KMeans Q=32 (incr)':       '-',
    'Local Query Sort 8 (incr)':      '-',
    'Local Query Sort 32 (incr)':     '-',
}

PLOT_METHODS = [
    'Oracle', 'Uniform', 'KMeans Keys (incr)', 'Local KMeans Q=32 (incr)',
]

PLOT_DISPLAY_NAMES = {
    'Local KMeans Q=8 (incr)':    'Sorting KMeans 8 (incr)',
    'Local KMeans Q=32 (incr)':   'Sorting KMeans 32 (incr)',
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
    positions = {}

    min_offset = max(30, BUDGETS[-1])
    first_start = min(min_offset, seq_len - num_queries)
    first_start = max(0, first_start)
    positions['first'] = list(range(first_start, first_start + num_queries))

    mid_center = seq_len // 2
    mid_start = mid_center - num_queries // 2
    mid_start = max(0, min(mid_start, seq_len - num_queries))
    positions['middle'] = list(range(mid_start, mid_start + num_queries))

    last_start = max(0, seq_len - num_queries)
    positions['last'] = list(range(last_start, last_start + num_queries))

    return positions


# ============================================================================
# INCREMENTAL CLUSTER STATE
# ============================================================================

class IncrementalClusters:
    """
    Maintains running sums for G clusters. Keys are added one at a time.
    At any point, non-empty clusters expose (mean_key, mean_value, size).
    """

    def __init__(self, G, head_dim):
        self.G = G
        self.d = head_dim
        self.key_sums = np.zeros((G, head_dim), dtype=np.float64)
        self.val_sums = np.zeros((G, head_dim), dtype=np.float64)
        self.counts = np.zeros(G, dtype=np.float64)

    def add_key(self, cluster_id, key_vec, val_vec):
        """Add one key-value pair to its preassigned cluster. O(d)."""
        self.key_sums[cluster_id] += key_vec
        self.val_sums[cluster_id] += val_vec
        self.counts[cluster_id] += 1

    def get_representatives(self):
        """
        Return (mean_keys, mean_values, sizes) for non-empty clusters only.
        """
        active = self.counts > 0
        n_active = active.sum()
        if n_active == 0:
            return None, None, None

        counts_active = self.counts[active]
        mk = self.key_sums[active] / counts_active[:, None]
        mv = self.val_sums[active] / counts_active[:, None]
        return mk, mv, counts_active

    def snapshot(self):
        """Return a copy of current state (for rewinding if needed)."""
        return (
            self.key_sums.copy(),
            self.val_sums.copy(),
            self.counts.copy(),
        )

    def restore(self, snapshot):
        self.key_sums, self.val_sums, self.counts = snapshot


def frozen_group_attention(query, mean_keys, mean_values, group_sizes):
    """Attend to group representatives with size-weighted logits."""
    group_logits = (query @ mean_keys.T) / np.sqrt(HEAD_DIM)
    group_logits = group_logits + np.log(group_sizes + 1e-10)
    group_weights = softmax(group_logits)
    return group_weights @ mean_values


# ============================================================================
# PRECOMPUTE CLUSTER ASSIGNMENTS (labels only, no representatives)
# ============================================================================

def assign_fixed_quantile(mean_q, K_mat, seq_len):
    """
    Sort keys by logits w.r.t. mean query, assign quantile group labels.
    Returns {budget: labels[seq_len]} where labels[i] = group id for key i.
    """
    logits = (mean_q @ K_mat[:seq_len].T) / np.sqrt(HEAD_DIM)
    weights = softmax(logits)
    sorted_indices = np.argsort(logits)[::-1]
    sorted_weights = weights[sorted_indices]

    assignments = {}
    for budget in BUDGETS:
        b = min(budget, seq_len)
        sorted_labels = group_quantile_weight(sorted_weights, b)
        # Map back to original key order
        labels = np.zeros(seq_len, dtype=np.int32)
        labels[sorted_indices] = sorted_labels
        assignments[budget] = labels

    return assignments


def assign_kmeans_keys(K_mat, seq_len):
    """
    KMeans on all keys, return {budget: labels[seq_len]}.
    """
    assignments = {}
    for budget in BUDGETS:
        b = min(budget, seq_len)
        if b >= seq_len:
            assignments[budget] = np.arange(seq_len, dtype=np.int32)
        else:
            km = KMeans(n_clusters=b, n_init=3, max_iter=100, random_state=SEED)
            km.fit(K_mat[:seq_len])
            assignments[budget] = km.labels_.astype(np.int32)
    return assignments


def assign_local_kmeans(Q, K_mat, seq_len, nq):
    """
    Cluster queries into nq groups, for each cluster sort keys by the cluster's
    mean query and assign quantile labels.
    Returns (q_labels, {budget: list of labels[seq_len] per query cluster}).
    """
    nq_actual = min(nq, seq_len)
    km = KMeans(n_clusters=nq_actual, n_init=3, max_iter=100, random_state=SEED)
    km.fit(Q[:seq_len])
    q_labels = km.labels_
    q_centers = km.cluster_centers_

    cluster_assignments = []  # list of {budget: labels}
    for c in range(nq_actual):
        ca = assign_fixed_quantile(q_centers[c], K_mat, seq_len)
        cluster_assignments.append(ca)

    return q_labels, cluster_assignments


def assign_local_sort(Q, K_mat, seq_len, n_qgroups):
    """
    Split queries into n_qgroups contiguous blocks, for each block sort keys
    by the block's mean query and assign quantile labels.
    Returns (qgroup_means, list of {budget: labels}).
    """
    boundaries = np.round(np.linspace(0, seq_len, n_qgroups + 1)).astype(int)

    qgroup_means = np.zeros((n_qgroups, HEAD_DIM), dtype=np.float64)
    qgroup_assignments = []

    for g in range(n_qgroups):
        start, end = boundaries[g], boundaries[g + 1]
        if end <= start:
            qgroup_means[g] = qgroup_means[max(0, g - 1)]
            qgroup_assignments.append(
                qgroup_assignments[-1] if qgroup_assignments else {}
            )
            continue

        mean_q_g = Q[start:end].mean(axis=0)
        qgroup_means[g] = mean_q_g
        ga = assign_fixed_quantile(mean_q_g, K_mat, seq_len)
        qgroup_assignments.append(ga)

    return qgroup_means, qgroup_assignments


def find_closest_qgroup(query, qgroup_means):
    q_norm = query / (np.linalg.norm(query) + 1e-10)
    m_norms = qgroup_means / (
        np.linalg.norm(qgroup_means, axis=1, keepdims=True) + 1e-10
    )
    return int(np.argmax(m_norms @ q_norm))


# ============================================================================
# INCREMENTAL EVALUATION
# ============================================================================

def run_incremental_for_budget(
    Q, K_mat, V, seq_len, labels, budget, test_positions_by_region
):
    """
    Incrementally reveal keys 0..seq_len-1, maintaining cluster running means.
    At each test position, compute frozen_group_attention.

    Returns {region: list of outputs} for test queries.
    """
    G = int(labels.max()) + 1
    state = IncrementalClusters(G, HEAD_DIM)

    # Collect all test positions across regions, sorted
    all_test = set()
    for region in REGIONS:
        all_test.update(test_positions_by_region[region])
    max_test_pos = max(all_test) if all_test else 0

    # Results: {region: list of (qpos, output)}
    outputs = {r: [] for r in REGIONS}
    test_set_by_region = {
        r: set(test_positions_by_region[r]) for r in REGIONS
    }

    # Walk through keys causally
    for pos in range(max_test_pos + 1):
        # Reveal key at position pos
        state.add_key(labels[pos], K_mat[pos], V[pos])

        # Check if this position is a test query
        for region in REGIONS:
            if pos in test_set_by_region[region]:
                mk, mv, gs = state.get_representatives()
                if mk is not None:
                    out = frozen_group_attention(Q[pos], mk, mv, gs)
                else:
                    out = np.zeros(HEAD_DIM, dtype=np.float64)
                outputs[region].append((pos, out))

    return outputs


# ============================================================================
# MAIN COMPUTATION
# ============================================================================

def analyze_layer(data_path, selected_indices, layer_name, rng):
    print(f"\n{'='*60}")
    print(f"  {layer_name}")
    print(f"{'='*60}")

    errors = {
        region: {m: {b: [] for b in BUDGETS} for m in METHOD_NAMES}
        for region in REGIONS
    }

    selected_set = set(selected_indices)
    ex_count = 0
    total_examples = len(selected_indices)

    with open(data_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx not in selected_set:
                continue

            example = json.loads(line)
            Q = np.array(example[layer_name]['Q'], dtype=np.float32)
            K_mat = np.array(example[layer_name]['K'], dtype=np.float32)
            V = np.array(example[layer_name]['V'], dtype=np.float32)
            seq_len = Q.shape[0]

            region_positions = select_query_positions(seq_len, NUM_TEST_QUERIES)
            ex_count += 1

            print(f"\n  [{ex_count}/{total_examples}] Example {idx}: "
                  f"seq_len={seq_len}")
            print(f"    Positions: first={region_positions['first'][0]}-"
                  f"{region_positions['first'][-1]}, "
                  f"middle={region_positions['middle'][0]}-"
                  f"{region_positions['middle'][-1]}, "
                  f"last={region_positions['last'][0]}-"
                  f"{region_positions['last'][-1]}")

            # ---- Precompute cluster assignments ----
            print(f"    Precomputing assignments...")
            t_pre = time.time()

            global_mean_q = Q.mean(axis=0)
            fixed_q_assignments = assign_fixed_quantile(
                global_mean_q, K_mat, seq_len
            )
            kmeans_assignments = assign_kmeans_keys(K_mat, seq_len)

            local_kmeans_assignments = {}
            for nq in Q_CLUSTERS_LIST:
                local_kmeans_assignments[nq] = assign_local_kmeans(
                    Q, K_mat, seq_len, nq
                )

            local_sort_assignments = {}
            for nq in LOCAL_Q_GROUPS:
                local_sort_assignments[nq] = assign_local_sort(
                    Q, K_mat, seq_len, nq
                )

            print(f"    Assignments done in {time.time()-t_pre:.1f}s")

            # ---- Compute ground truth + baselines for all test queries ----
            print(f"    Computing ground truth + baselines...")
            t_gt = time.time()

            gt_data = {}  # {qpos: (full_out, logits, full_w, keys, vals)}
            for region in REGIONS:
                for qpos in region_positions[region]:
                    if qpos in gt_data:
                        continue
                    q = Q[qpos]
                    keys = K_mat[:qpos + 1]
                    vals = V[:qpos + 1]
                    logits = (q @ keys.T) / np.sqrt(HEAD_DIM)
                    full_w = softmax(logits)
                    full_out = full_w @ vals
                    gt_data[qpos] = (full_out, logits, full_w, keys, vals)

            # Baselines: Oracle, Uniform, Per-Query Quantile
            for region in REGIONS:
                for qpos in region_positions[region]:
                    full_out, logits, full_w, keys, vals = gt_data[qpos]
                    q = Q[qpos]
                    n_keys = qpos + 1

                    for budget in BUDGETS:
                        b = min(budget, n_keys)

                        out_oracle, _ = oracle_sampling(
                            q, keys, vals, logits, full_w, b
                        )
                        errors[region]['Oracle'][budget].append(
                            rel_l2(out_oracle, full_out)
                        )

                        u_idx = rng.choice(n_keys, size=b, replace=False)
                        out_uniform = softmax(logits[u_idx]) @ vals[u_idx]
                        errors[region]['Uniform'][budget].append(
                            rel_l2(out_uniform, full_out)
                        )

                        _, out_pq = grouped_attention(
                            logits, vals, full_w, b, method='quantile'
                        )
                        errors[region]['Per-Query Quantile'][budget].append(
                            rel_l2(out_pq, full_out)
                        )

            print(f"    Ground truth + baselines done in {time.time()-t_gt:.1f}s")

            # ---- Incremental methods: one walk per (method, budget) ----
            print(f"    Running incremental methods...")
            t_incr = time.time()

            for budget in BUDGETS:
                b = min(budget, seq_len)

                # --- Fixed Quantile (incr) ---
                labels_fq = fixed_q_assignments[budget]
                incr_out = run_incremental_for_budget(
                    Q, K_mat, V, seq_len, labels_fq, budget,
                    region_positions
                )
                for region in REGIONS:
                    for qpos, out in incr_out[region]:
                        full_out = gt_data[qpos][0]
                        errors[region]['Fixed Quantile (incr)'][budget].append(
                            rel_l2(out, full_out)
                        )

                # --- KMeans Keys (incr) ---
                labels_km = kmeans_assignments[budget]
                incr_out = run_incremental_for_budget(
                    Q, K_mat, V, seq_len, labels_km, budget,
                    region_positions
                )
                for region in REGIONS:
                    for qpos, out in incr_out[region]:
                        full_out = gt_data[qpos][0]
                        errors[region]['KMeans Keys (incr)'][budget].append(
                            rel_l2(out, full_out)
                        )

                # --- Local KMeans (incr) ---
                for nq in Q_CLUSTERS_LIST:
                    q_labels_lk, cluster_assgn_list = local_kmeans_assignments[nq]

                    # For each query, pick its cluster's labels
                    # We need to run incremental per cluster assignment,
                    # but queries from different clusters use different labels.
                    # Approach: for each query cluster, run incremental walk
                    # with that cluster's labels.
                    # Since test queries may span multiple clusters, collect
                    # per-cluster test positions.
                    cluster_test_positions = {}
                    for region in REGIONS:
                        for qpos in region_positions[region]:
                            cid = q_labels_lk[qpos]
                            if cid not in cluster_test_positions:
                                cluster_test_positions[cid] = {
                                    r: [] for r in REGIONS
                                }
                            cluster_test_positions[cid][region].append(qpos)

                    for cid, ctp in cluster_test_positions.items():
                        c_labels = cluster_assgn_list[cid][budget]
                        incr_out = run_incremental_for_budget(
                            Q, K_mat, V, seq_len, c_labels, budget, ctp
                        )
                        method_name = f'Local KMeans Q={nq} (incr)'
                        for region in REGIONS:
                            for qpos, out in incr_out[region]:
                                full_out = gt_data[qpos][0]
                                errors[region][method_name][budget].append(
                                    rel_l2(out, full_out)
                                )

                # --- Local Query Sort (incr) ---
                for nq in LOCAL_Q_GROUPS:
                    qg_means, qg_assgn_list = local_sort_assignments[nq]

                    # For each query, find closest qgroup
                    qgroup_test_positions = {}
                    for region in REGIONS:
                        for qpos in region_positions[region]:
                            closest = find_closest_qgroup(Q[qpos], qg_means)
                            if closest not in qgroup_test_positions:
                                qgroup_test_positions[closest] = {
                                    r: [] for r in REGIONS
                                }
                            qgroup_test_positions[closest][region].append(qpos)

                    for gid, gtp in qgroup_test_positions.items():
                        g_labels = qg_assgn_list[gid][budget]
                        incr_out = run_incremental_for_budget(
                            Q, K_mat, V, seq_len, g_labels, budget, gtp
                        )
                        method_name = f'Local Query Sort {nq} (incr)'
                        for region in REGIONS:
                            for qpos, out in incr_out[region]:
                                full_out = gt_data[qpos][0]
                                errors[region][method_name][budget].append(
                                    rel_l2(out, full_out)
                                )

                print(f"\r    Budget {budget}: done", end="", flush=True)

            print(f"\n    Incremental methods done in "
                  f"{time.time()-t_incr:.1f}s")

            del Q, K_mat, V, gt_data

    # Aggregate
    results = {'budgets': BUDGETS}
    for region in REGIONS:
        results[region] = {}
        for m in METHOD_NAMES:
            results[region][f'{m}_mean'] = [
                float(np.mean(errors[region][m][b]))
                if errors[region][m][b] else 0.0
                for b in BUDGETS
            ]
            results[region][f'{m}_std'] = [
                float(np.std(errors[region][m][b]))
                if errors[region][m][b] else 0.0
                for b in BUDGETS
            ]
    return results


# ============================================================================
# PLOTTING
# ============================================================================

def _plot_comparison(ax, data, region, layer_name):
    x = np.array(data['budgets'])
    region_data = data[region]

    for method in PLOT_METHODS:
        if f'{method}_mean' not in region_data:
            continue
        means = np.array(region_data[f'{method}_mean'])
        stds = np.array(region_data[f'{method}_std'])
        color = METHOD_COLORS.get(method, '#999')
        marker = METHOD_MARKERS.get(method, 'o')
        ls = METHOD_LINESTYLES.get(method, '-')
        label = PLOT_DISPLAY_NAMES.get(method, method)

        ax.plot(x, means, marker=marker, color=color, lw=2.5, ls=ls,
                label=label, zorder=4, markersize=6)
        hi = means + stds
        ax.fill_between(x, means, hi, color=color, alpha=0.12)

    ax.set_title(REGION_DISPLAY[region], fontsize=12, fontweight='bold')
    ax.set_xlabel('Budget (num groups)', fontsize=10)
    ax.set_ylabel('Relative L2 Error', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(left=0, right=512)
    ax.grid(True, alpha=0.3, ls='--', which='both')


def _plot_region_overlay(ax, data, method, layer_name):
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
                f'Llama-3-8B  |  Incremental clusters  |  Shaded = +1 std')

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
            f'Incremental Clustering by Query Position — {layer_title}\n'
            f'{subtitle}',
            fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.06, 1, 0.94])
        save_figure(fig, output_dir / f'by_region_{layer_short}.png', dpi=200)
        plt.close(fig)

        # --- Figure 2: Per-method overlay ---
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

        for j in range(n_methods, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f'Per-Method Position Comparison (Incremental) — {layer_title}\n'
            f'{subtitle}',
            fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, output_dir / f'per_method_{layer_short}.png', dpi=200)
        plt.close(fig)

        # --- Figure 3: Summary ---
        key_methods = ['Oracle', 'Uniform', 'KMeans Keys (incr)',
                       'Local KMeans Q=32 (incr)']
        fig, axes = plt.subplots(1, len(key_methods),
                                 figsize=(6 * len(key_methods), 6), sharey=True)
        for i, method in enumerate(key_methods):
            _plot_region_overlay(axes[i], layer_data, method, layer)
            if i == 0:
                axes[i].legend(fontsize=10, framealpha=0.95)
            if i > 0:
                axes[i].set_ylabel('')

        fig.suptitle(
            f'Position Effect on Key Methods (Incremental) — {layer_title}\n'
            f'{subtitle}',
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
    print("INCREMENTAL FROZEN-ASSIGNMENT CLUSTERING")
    print("=" * 60)
    print(f"Config: {NUM_EXAMPLES} examples, {NUM_TEST_QUERIES} queries/region")
    print(f"Regions: {REGIONS}")
    print(f"Budgets: {BUDGETS}")
    print(f"Methods: {len(METHOD_NAMES)}")
    print(f"Key: assignments precomputed on all keys, representatives built "
          f"incrementally (causal)")
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
            'incremental': True,
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
