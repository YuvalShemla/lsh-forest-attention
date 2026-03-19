#!/usr/bin/env python3
"""
LSH vs KMeans Clustering — Centered vs Raw

Compares data-independent LSH clustering against data-dependent KMeans for
grouping keys in approximate attention. Tests both Cross-Polytope and
SimHash (random hyperplane) LSH.

Key insight from MagicPIG: centering keys (subtracting mean key vector) before
hashing is critical — without it, keys cluster in a narrow cone and LSH
collisions become near-uniform, destroying discriminative power.

Each LSH variant produces a fixed number of buckets determined by its hash
parameters. We evaluate at the natural bucket count and plot as fixed scatter
points alongside budget-sweep curves.

Methods (9 total):
  Curves (budget sweep):
  - Oracle                           — privileged baseline
  - Uniform                          — causal random sampling
  - KMeans Keys (incr)               — sklearn KMeans, frozen labels

  Fixed points (natural bucket count, causal incremental):
  - CP-LSH centered (incr)           — Cross-Polytope k=1, centered (~256 buckets)
  - CP-LSH raw (incr)                — Cross-Polytope k=1, raw
  - SimHash k=8 centered (incr)      — 8 hyperplanes, centered (up to 256 buckets)
  - SimHash k=8 raw (incr)           — 8 hyperplanes, raw
  - SimHash k=6 centered (incr)      — 6 hyperplanes, centered (up to 64 buckets)
  - SimHash k=6 raw (incr)           — 6 hyperplanes, raw

All LSH variants run N_ITERS iterations with different random seeds.
Per-query errors are averaged across iterations.

Usage:
  python compare_lsh_vs_kmeans_clustering.py              # run compute + plot
  python compare_lsh_vs_kmeans_clustering.py --plot-only  # regenerate plots from JSON
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
from algorithms.lsh_index import CrossPolytopeIndex, SimHashIndex
from visualization.plot_utils import setup_style, save_figure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
DATA_PATH = '../../data/attention_vectors_long_bench_llama_8b.jsonl'
OUTPUT_DIR = Path('../../results/lsh_vs_kmeans_clustering')
NUM_EXAMPLES = 10
NUM_TEST_QUERIES = 30
N_ITERS = 30
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
BUDGETS = [2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024]

REGIONS = ['first', 'middle', 'last']
REGION_DISPLAY = {
    'first':  'First 30 Queries (Early Positions)',
    'middle': 'Middle 30 Queries (Center Positions)',
    'last':   'Last 30 Queries (Late Positions)',
}

# Curve methods (budget sweep)
CURVE_METHODS = ['Oracle', 'Uniform', 'KMeans Keys (incr)', 'GMM Keys (incr)']

# Point methods (fixed bucket count from LSH)
# (display_name, lsh_type, k, center_keys)
#   lsh_type: 'cp' for CrossPolytope, 'simhash' for random hyperplane
LSH_VARIANTS = [
    ('CP-LSH centered',        'cp',      1, True),
    ('CP-LSH raw',             'cp',      1, False),
    ('SimHash k=8 centered',   'simhash', 8, True),
    ('SimHash k=8 raw',        'simhash', 8, False),
    ('SimHash k=6 centered',   'simhash', 6, True),
    ('SimHash k=6 raw',        'simhash', 6, False),
]
PLOT_POINT_METHODS = [name for name, _, _, _ in LSH_VARIANTS if 'k=6' not in name]
POINT_METHODS = [name for name, _, _, _ in LSH_VARIANTS]

ALL_METHODS = CURVE_METHODS + POINT_METHODS

CURVE_COLORS = {
    'Oracle':              '#2ca02c',
    'Uniform':             '#7fbf7f',
    'KMeans Keys (incr)':  'darkorange',
    'GMM Keys (incr)':     '#e377c2',
}

CURVE_MARKERS = {
    'Oracle':              '^',
    'Uniform':             's',
    'KMeans Keys (incr)':  'X',
    'GMM Keys (incr)':     'P',
}

POINT_COLORS = {
    'CP-LSH centered':        '#d62728',
    'CP-LSH raw':             '#ff9896',
    'SimHash k=8 centered':   '#9467bd',
    'SimHash k=8 raw':        '#c5b0d5',
    'SimHash k=6 centered':   '#1f77b4',
    'SimHash k=6 raw':        '#aec7e8',
}

POINT_MARKERS = {
    'CP-LSH centered':        'X',
    'CP-LSH raw':             'X',
    'SimHash k=8 centered':   'P',
    'SimHash k=8 raw':        'P',
    'SimHash k=6 centered':   'D',
    'SimHash k=6 raw':        'D',
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
    """Running sums for G clusters. Keys added one at a time (causal)."""

    def __init__(self, G, head_dim):
        self.G = G
        self.key_sums = np.zeros((G, head_dim), dtype=np.float64)
        self.val_sums = np.zeros((G, head_dim), dtype=np.float64)
        self.counts = np.zeros(G, dtype=np.float64)

    def add_key(self, cluster_id, key_vec, val_vec):
        self.key_sums[cluster_id] += key_vec
        self.val_sums[cluster_id] += val_vec
        self.counts[cluster_id] += 1

    def get_representatives(self):
        active = self.counts > 0
        if not active.any():
            return None, None, None
        counts_active = self.counts[active]
        mk = self.key_sums[active] / counts_active[:, None]
        mv = self.val_sums[active] / counts_active[:, None]
        return mk, mv, counts_active


def frozen_group_attention(query, mean_keys, mean_values, group_sizes):
    """Attend to group representatives with size-weighted logits."""
    group_logits = (query @ mean_keys.T) / np.sqrt(HEAD_DIM)
    group_logits = group_logits + np.log(group_sizes + 1e-10)
    group_weights = softmax(group_logits)
    return group_weights @ mean_values


# ============================================================================
# CLUSTER ASSIGNMENT METHODS
# ============================================================================

def assign_kmeans_keys(K_mat, seq_len):
    """KMeans on all keys, return {budget: labels[seq_len]}."""
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


def assign_gmm_keys(K_mat, seq_len):
    """GMM hard assignments on all keys, return {budget: labels[seq_len]}."""
    from sklearn.mixture import GaussianMixture

    assignments = {}
    for budget in BUDGETS:
        b = min(budget, seq_len)
        if b >= seq_len:
            assignments[budget] = np.arange(seq_len, dtype=np.int32)
        else:
            gmm = GaussianMixture(
                n_components=b, covariance_type='diag',
                max_iter=100, n_init=1, random_state=SEED
            )
            gmm.fit(K_mat[:seq_len])
            assignments[budget] = gmm.predict(K_mat[:seq_len]).astype(np.int32)
    return assignments


def assign_lsh_labels(keys, seq_len, lsh_type, k, center_keys, seed):
    """
    Build an LSH index and extract bucket labels.

    lsh_type='cp': CrossPolytope with k_cp=k (buckets = 2*d per CP round)
    lsh_type='simhash': SimHash with k hyperplanes (buckets up to 2^k)

    Returns (labels[seq_len], G_eff) where labels are consecutive [0, G_eff-1].
    """
    if lsh_type == 'cp':
        idx = CrossPolytopeIndex(
            num_tables=1, max_cp=k, head_dim=HEAD_DIM,
            center_keys=center_keys, seed=seed
        )
        idx.build_index(keys[:seq_len])
        raw_labels = idx.key_codes[:, 0, 0]  # [seq_len]
    elif lsh_type == 'simhash':
        idx = SimHashIndex(
            num_tables=1, max_depth=k, head_dim=HEAD_DIM,
            center_keys=center_keys, seed=seed
        )
        idx.build_index(keys[:seq_len])
        # Combine k binary bits into a single integer label
        codes = idx.key_codes[:, 0, :]  # [seq_len, k], values 0 or 1
        raw_labels = np.zeros(seq_len, dtype=np.int64)
        for bit in range(k):
            raw_labels = raw_labels * 2 + codes[:, bit].astype(np.int64)
    else:
        raise ValueError(f"Unknown lsh_type: {lsh_type}")

    # Remap to consecutive [0, G_eff-1]
    _, labels = np.unique(raw_labels, return_inverse=True)
    G_eff = len(np.unique(labels))
    return labels.astype(np.int32), G_eff


# ============================================================================
# INCREMENTAL EVALUATION
# ============================================================================

def run_incremental_multi_budget(
    Q, K_mat, V, seq_len, labels_by_budget, test_positions_by_region
):
    """
    Single causal walk maintaining one IncrementalClusters per budget.
    Returns {budget: {region: list of (qpos, output)}}.
    """
    all_test = set()
    for region in REGIONS:
        all_test.update(test_positions_by_region[region])
    if not all_test:
        return {b: {r: [] for r in REGIONS} for b in labels_by_budget}
    max_test_pos = max(all_test)

    test_set_by_region = {r: set(test_positions_by_region[r]) for r in REGIONS}

    states = {}
    for budget, labels in labels_by_budget.items():
        G = int(labels.max()) + 1
        states[budget] = (IncrementalClusters(G, HEAD_DIM), labels)

    outputs = {b: {r: [] for r in REGIONS} for b in labels_by_budget}

    for pos in range(max_test_pos + 1):
        for budget, (state, labels) in states.items():
            state.add_key(labels[pos], K_mat[pos], V[pos])

        for region in REGIONS:
            if pos in test_set_by_region[region]:
                for budget, (state, labels) in states.items():
                    mk, mv, gs = state.get_representatives()
                    if mk is not None:
                        out = frozen_group_attention(Q[pos], mk, mv, gs)
                    else:
                        out = np.zeros(HEAD_DIM, dtype=np.float64)
                    outputs[budget][region].append((pos, out))

    return outputs


def run_incremental_single(Q, K_mat, V, seq_len, labels, test_positions_by_region):
    """
    Single causal walk for one set of labels.
    Returns {region: list of (qpos, output)}.
    """
    G = int(labels.max()) + 1
    state = IncrementalClusters(G, HEAD_DIM)

    all_test = set()
    for region in REGIONS:
        all_test.update(test_positions_by_region[region])
    if not all_test:
        return {r: [] for r in REGIONS}
    max_test_pos = max(all_test)

    test_set_by_region = {r: set(test_positions_by_region[r]) for r in REGIONS}
    outputs = {r: [] for r in REGIONS}

    for pos in range(max_test_pos + 1):
        state.add_key(labels[pos], K_mat[pos], V[pos])

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

def analyze_layer(data_path, selected_indices, layer_name, master_rng):
    print(f"\n{'='*60}")
    print(f"  {layer_name}")
    print(f"{'='*60}")

    # Curve methods: errors[region][method][budget] = list of per-query errors
    curve_errors = {
        region: {m: {b: [] for b in BUDGETS} for m in CURVE_METHODS}
        for region in REGIONS
    }

    # Point methods: errors[region][method] = list of per-query errors
    # Also track G_eff per method across examples
    point_errors = {
        region: {m: [] for m in POINT_METHODS}
        for region in REGIONS
    }
    g_eff_values = {m: [] for m in POINT_METHODS}

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
            all_test_positions = set()
            for r in REGIONS:
                all_test_positions.update(region_positions[r])

            ex_count += 1
            print(f"\n  [{ex_count}/{total_examples}] Example {idx}: "
                  f"seq_len={seq_len}")

            # ---- Ground truth ----
            print(f"    Computing ground truth...")
            t0 = time.time()
            gt_data = {}
            for qpos in all_test_positions:
                q = Q[qpos]
                keys = K_mat[:qpos + 1]
                vals = V[:qpos + 1]
                logits = (q @ keys.T) / np.sqrt(HEAD_DIM)
                full_w = softmax(logits)
                full_out = full_w @ vals
                gt_data[qpos] = (full_out, logits, full_w, keys, vals)
            print(f"    Ground truth done in {time.time()-t0:.1f}s")

            # ---- Baselines (Oracle, Uniform) ----
            print(f"    Baselines...")
            t0 = time.time()
            baseline_rng = np.random.default_rng(master_rng.integers(2**32))
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
                        curve_errors[region]['Oracle'][budget].append(
                            rel_l2(out_oracle, full_out)
                        )

                        u_idx = baseline_rng.choice(n_keys, size=b, replace=False)
                        out_uniform = softmax(logits[u_idx]) @ vals[u_idx]
                        curve_errors[region]['Uniform'][budget].append(
                            rel_l2(out_uniform, full_out)
                        )
            print(f"    Baselines done in {time.time()-t0:.1f}s")

            # ---- KMeans Keys (incr) ----
            print(f"    KMeans Keys assignments...")
            t0 = time.time()
            kmeans_assignments = assign_kmeans_keys(K_mat, seq_len)
            print(f"    KMeans done in {time.time()-t0:.1f}s")

            print(f"    KMeans incremental eval...")
            t0 = time.time()
            km_outputs = run_incremental_multi_budget(
                Q, K_mat, V, seq_len, kmeans_assignments, region_positions
            )
            for budget in BUDGETS:
                for region in REGIONS:
                    for qpos, out in km_outputs[budget][region]:
                        full_out = gt_data[qpos][0]
                        curve_errors[region]['KMeans Keys (incr)'][budget].append(
                            rel_l2(out, full_out)
                        )
            print(f"    KMeans eval done in {time.time()-t0:.1f}s")

            # ---- GMM Keys (incr) ----
            print(f"    GMM Keys assignments...")
            t0 = time.time()
            gmm_assignments = assign_gmm_keys(K_mat, seq_len)
            print(f"    GMM done in {time.time()-t0:.1f}s")

            print(f"    GMM incremental eval...")
            t0 = time.time()
            gmm_outputs = run_incremental_multi_budget(
                Q, K_mat, V, seq_len, gmm_assignments, region_positions
            )
            for budget in BUDGETS:
                for region in REGIONS:
                    for qpos, out in gmm_outputs[budget][region]:
                        full_out = gt_data[qpos][0]
                        curve_errors[region]['GMM Keys (incr)'][budget].append(
                            rel_l2(out, full_out)
                        )
            print(f"    GMM eval done in {time.time()-t0:.1f}s")

            # ---- LSH variants: N_ITERS iterations, natural bucket count ----
            iter_errors = {
                m: {
                    r: [[] for _ in range(NUM_TEST_QUERIES)]
                    for r in REGIONS
                }
                for m in POINT_METHODS
            }
            iter_g_effs = {m: [] for m in POINT_METHODS}

            print(f"    Running {N_ITERS} iterations of LSH variants "
                  f"({len(LSH_VARIANTS)} variants)...")
            t0 = time.time()

            for it in range(N_ITERS):
                iter_seed = int(master_rng.integers(2**32))
                iter_rng = np.random.default_rng(iter_seed)

                for method_name, lsh_type, k, center_keys in LSH_VARIANTS:
                    lsh_seed = int(iter_rng.integers(2**32))
                    labels, G_eff = assign_lsh_labels(
                        K_mat, seq_len, lsh_type, k, center_keys, lsh_seed
                    )
                    iter_g_effs[method_name].append(G_eff)

                    # Single incremental walk at natural bucket count
                    variant_outputs = run_incremental_single(
                        Q, K_mat, V, seq_len, labels, region_positions
                    )

                    for region in REGIONS:
                        for qi, (qpos, out) in enumerate(
                            variant_outputs[region]
                        ):
                            full_out = gt_data[qpos][0]
                            err = rel_l2(out, full_out)
                            iter_errors[method_name][region][qi].append(err)

                if (it + 1) % 5 == 0 or it == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / (it + 1) * (N_ITERS - it - 1)
                    print(f"\r    Iter {it+1}/{N_ITERS}  "
                          f"elapsed {format_eta(elapsed)}  "
                          f"ETA {format_eta(eta)}",
                          end="", flush=True)

            print(f"\n    LSH variants done in {format_eta(time.time()-t0)}")

            # Average across iterations, append to point_errors
            for method_name in POINT_METHODS:
                avg_g = float(np.mean(iter_g_effs[method_name]))
                g_eff_values[method_name].append(avg_g)
                print(f"    {method_name}: avg G_eff = {avg_g:.1f}")

                for region in REGIONS:
                    for qi in range(NUM_TEST_QUERIES):
                        vals_list = iter_errors[method_name][region][qi]
                        if vals_list:
                            avg_err = float(np.mean(vals_list))
                            point_errors[region][method_name].append(avg_err)

            del Q, K_mat, V, gt_data

    # Aggregate curves
    results = {'budgets': BUDGETS}
    for region in REGIONS:
        results[region] = {}
        for m in CURVE_METHODS:
            results[region][f'{m}_mean'] = [
                float(np.mean(curve_errors[region][m][b]))
                if curve_errors[region][m][b] else 0.0
                for b in BUDGETS
            ]
            results[region][f'{m}_std'] = [
                float(np.std(curve_errors[region][m][b]))
                if curve_errors[region][m][b] else 0.0
                for b in BUDGETS
            ]

    # Aggregate points
    results['cp_points'] = {}
    for m in POINT_METHODS:
        avg_g_eff = float(np.mean(g_eff_values[m]))
        results['cp_points'][m] = {
            'g_eff': avg_g_eff,
            'g_eff_all': [float(v) for v in g_eff_values[m]],
        }
        for region in REGIONS:
            errs = point_errors[region][m]
            results['cp_points'][m][f'{region}_mean'] = float(np.mean(errs)) if errs else 0.0
            results['cp_points'][m][f'{region}_std'] = float(np.std(errs)) if errs else 0.0

    return results


# ============================================================================
# PLOTTING
# ============================================================================

def _plot_comparison(ax, data, region):
    """Plot curves + CP-LSH fixed points for one region."""
    x = np.array(data['budgets'])
    region_data = data[region]

    # Draw curves
    for method in CURVE_METHODS:
        if f'{method}_mean' not in region_data:
            continue
        means = np.array(region_data[f'{method}_mean'])
        stds = np.array(region_data[f'{method}_std'])
        color = CURVE_COLORS[method]
        marker = CURVE_MARKERS[method]

        ax.plot(x, means, marker=marker, color=color, lw=2.5,
                label=method, zorder=4, markersize=6)
        hi = means + stds
        ax.fill_between(x, means, hi, color=color, alpha=0.12)

    # Draw LSH fixed points
    cp_points = data.get('cp_points', {})
    for method_name in PLOT_POINT_METHODS:
        if method_name not in cp_points:
            continue
        pt = cp_points[method_name]
        g_eff = pt['g_eff']
        mean_err = pt.get(f'{region}_mean', 0)
        std_err = pt.get(f'{region}_std', 0)
        color = POINT_COLORS[method_name]
        marker = POINT_MARKERS[method_name]

        ax.scatter([g_eff], [mean_err], marker=marker, color=color,
                   s=200, zorder=6, edgecolors='black', linewidths=1.0,
                   label=f'{method_name} (G={g_eff:.0f})')
        ax.errorbar([g_eff], [mean_err], yerr=[std_err],
                    fmt='none', color=color, capsize=5, capthick=2,
                    zorder=5)

    ax.set_title(REGION_DISPLAY[region], fontsize=12, fontweight='bold')
    ax.set_xlabel('Budget (num groups)', fontsize=10)
    ax.set_ylabel('Relative L2 Error', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(left=0, right=512)
    ax.grid(True, alpha=0.3, ls='--', which='both')


def _plot_region_overlay(ax, data, method):
    """Plot regions overlaid for a curve method."""
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

    ax.set_title(method, fontsize=12, fontweight='bold')
    ax.set_xlabel('Budget', fontsize=10)
    ax.set_ylabel('Relative L2 Error', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(left=0, right=512)
    ax.grid(True, alpha=0.3, ls='--', which='both')


def _plot_region_overlay_point(ax, data, method_name):
    """Plot regions overlaid for a CP-LSH point method."""
    region_colors = {'first': '#e41a1c', 'middle': '#377eb8', 'last': '#4daf4a'}
    region_markers = {'first': 'o', 'middle': 's', 'last': '^'}

    cp_points = data.get('cp_points', {})
    if method_name not in cp_points:
        return
    pt = cp_points[method_name]
    g_eff = pt['g_eff']

    for region in REGIONS:
        mean_err = pt.get(f'{region}_mean', 0)
        std_err = pt.get(f'{region}_std', 0)
        color = region_colors[region]
        marker = region_markers[region]

        ax.scatter([g_eff], [mean_err], marker=marker, color=color,
                   s=200, zorder=6, edgecolors='black', linewidths=1.0,
                   label=f'{region.capitalize()} queries')
        ax.errorbar([g_eff], [mean_err], yerr=[std_err],
                    fmt='none', color=color, capsize=5, capthick=2,
                    zorder=5)

    ax.set_title(f'{method_name} (G={g_eff:.0f})', fontsize=12,
                 fontweight='bold')
    ax.set_xlabel('Budget', fontsize=10)
    ax.set_ylabel('Relative L2 Error', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(left=0, right=512)
    ax.grid(True, alpha=0.3, ls='--', which='both')


def make_figures(all_results, output_dir):
    cfg = all_results.get('config', {})
    n_ex = cfg.get('num_examples', NUM_EXAMPLES)
    n_q = cfg.get('num_test_queries', NUM_TEST_QUERIES)
    n_it = cfg.get('n_iters', N_ITERS)
    subtitle = (f'{n_ex} examples, {n_q} queries each, {n_it} iters  |  '
                f'Llama-3-8B  |  Incremental  |  Shaded = +1 std')

    for layer in LAYERS:
        layer_data = all_results[layer]
        layer_short = 'first_layer' if 'first' in layer else 'last_layer'
        layer_title = ('First Layer (Layer 0)' if 'first' in layer
                       else 'Last Layer (Layer 31)')

        # --- Figure 1: by_region — 3 panels, curves + CP scatter ---
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
        for i, region in enumerate(REGIONS):
            _plot_comparison(axes[i], layer_data, region)
            if i > 0:
                axes[i].set_ylabel('')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', fontsize=8,
                   framealpha=0.95, ncol=3, bbox_to_anchor=(0.5, -0.06))
        fig.suptitle(
            f'LSH vs KMeans Clustering — {layer_title}\n{subtitle}',
            fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.10, 1, 0.94])
        save_figure(fig, output_dir / f'by_region_{layer_short}.png', dpi=200)
        plt.close(fig)

        # --- Figure 2: per_method — subplots (curves + points) ---
        all_plot = CURVE_METHODS + PLOT_POINT_METHODS
        n_methods = len(all_plot)
        ncols = 3  # 9 methods -> 3x3
        nrows = (n_methods + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(7 * ncols, 6 * nrows),
                                 sharey=True)
        axes_flat = axes.flatten()

        for i, method in enumerate(all_plot):
            if method in CURVE_METHODS:
                _plot_region_overlay(axes_flat[i], layer_data, method)
            else:
                _plot_region_overlay_point(axes_flat[i], layer_data, method)
            if i == 0:
                axes_flat[i].legend(fontsize=10, framealpha=0.95)

        for j in range(n_methods, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f'Per-Method Position Comparison — {layer_title}\n{subtitle}',
            fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, output_dir / f'per_method_{layer_short}.png', dpi=200)
        plt.close(fig)

        # --- Figure 3: summary — Oracle, KMeans, both CP points ---
        fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)
        for i, region in enumerate(REGIONS):
            ax = axes[i]
            x = np.array(layer_data['budgets'])
            region_data = layer_data[region]

            # Oracle, KMeans, and GMM curves
            for method in ['Oracle', 'KMeans Keys (incr)', 'GMM Keys (incr)']:
                means = np.array(region_data[f'{method}_mean'])
                stds = np.array(region_data[f'{method}_std'])
                color = CURVE_COLORS[method]
                marker = CURVE_MARKERS[method]
                ax.plot(x, means, marker=marker, color=color, lw=2.5,
                        label=method, zorder=4, markersize=6)
                ax.fill_between(x, means, means + stds, color=color, alpha=0.12)

            # LSH points
            cp_points = layer_data.get('cp_points', {})
            for method_name in PLOT_POINT_METHODS:
                if method_name not in cp_points:
                    continue
                pt = cp_points[method_name]
                g_eff = pt['g_eff']
                mean_err = pt.get(f'{region}_mean', 0)
                std_err = pt.get(f'{region}_std', 0)
                color = POINT_COLORS[method_name]
                marker = POINT_MARKERS[method_name]

                ax.scatter([g_eff], [mean_err], marker=marker, color=color,
                           s=200, zorder=6, edgecolors='black', linewidths=1.0,
                           label=f'{method_name} (G={g_eff:.0f})')
                ax.errorbar([g_eff], [mean_err], yerr=[std_err],
                            fmt='none', color=color, capsize=5, capthick=2,
                            zorder=5)

            ax.set_title(REGION_DISPLAY[region], fontsize=12, fontweight='bold')
            ax.set_xlabel('Budget (num groups)', fontsize=10)
            if i == 0:
                ax.set_ylabel('Relative L2 Error', fontsize=10)
            ax.set_yscale('log')
            ax.set_xlim(left=0, right=512)
            ax.grid(True, alpha=0.3, ls='--', which='both')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', fontsize=10,
                   framealpha=0.95, ncol=4, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(
            f'LSH Clustering Head-to-Head — {layer_title}\n{subtitle}',
            fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.06, 1, 0.94])
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
    print("CP-LSH vs KMEANS CLUSTERING (CENTERED vs RAW)")
    print("=" * 60)
    print(f"Config: {NUM_EXAMPLES} examples, {NUM_TEST_QUERIES} queries/region")
    print(f"LSH N_ITERS={N_ITERS} (natural bucket count)")
    print(f"Regions: {REGIONS}")
    print(f"Budgets (curves): {BUDGETS}")
    print(f"Curve methods: {CURVE_METHODS}")
    print(f"Point methods: {POINT_METHODS}")
    print()

    t0 = time.time()
    master_rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    data_path = os.path.join(script_dir, DATA_PATH)

    print(f"Scanning: {data_path}")
    with open(data_path, 'r') as f:
        total = sum(1 for _ in f)
    print(f"Found {total} examples")

    n_select = min(NUM_EXAMPLES, total)
    selected = sorted(
        master_rng.choice(total, n_select, replace=False).tolist()
    )
    print(f"Selected {n_select} examples: {selected}")

    all_results = {
        'config': {
            'num_examples': n_select,
            'num_test_queries': NUM_TEST_QUERIES,
            'n_iters': N_ITERS,
            'budgets': BUDGETS,
            'seed': SEED,
            'head_dim': HEAD_DIM,
            'curve_methods': CURVE_METHODS,
            'point_methods': POINT_METHODS,
            'regions': REGIONS,
        }
    }

    for layer in LAYERS:
        all_results[layer] = analyze_layer(
            data_path, selected, layer, master_rng
        )

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
