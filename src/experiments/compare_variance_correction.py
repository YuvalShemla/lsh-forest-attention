#!/usr/bin/env python3
"""
Variance Correction for Clustering Methods (Full Covariance)

For every clustering method (KMeans, LSH), compares mean-only vs
mean+variance-corrected attention. The variance correction adds a second-order
term using the full covariance matrix: logit += q^T Σ q / (2*d).

LSH iterations are averaged at the logit level (not error level) for stability.

Methods:
  Curves (budget sweep):
  - Oracle, Uniform                     — baselines
  - KMeans (mean)  / KMeans (+var)      — paired comparison

  Fixed points (natural bucket count):
  - CP-LSH centered (mean) / (+var)
  - CP-LSH raw (mean)      / (+var)
  - SimHash k=8 centered (mean) / (+var)
  - SimHash k=8 raw (mean)      / (+var)

Usage:
  python compare_variance_correction.py              # run compute + plot
  python compare_variance_correction.py --plot-only  # replot from JSON
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
OUTPUT_DIR = Path('../../results/variance_correction')
NUM_EXAMPLES = 50
NUM_TEST_QUERIES = 50
N_ITERS = 30
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
BUDGETS = [2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512]

REGIONS = ['first', 'middle', 'last']
REGION_DISPLAY = {
    'first':  f'First {NUM_TEST_QUERIES} Queries (Early Positions)',
    'middle': f'Middle {NUM_TEST_QUERIES} Queries (Center Positions)',
    'last':   f'Last {NUM_TEST_QUERIES} Queries (Late Positions)',
}

CURVE_METHODS = [
    'Oracle', 'Uniform',
    'KMeans (mean)', 'KMeans (+var)',
]

LSH_VARIANTS = [
    ('CP-LSH centered',      'cp',      1, True),
    ('CP-LSH raw',           'cp',      1, False),
    ('SimHash k=8 centered', 'simhash', 8, True),
    ('SimHash k=8 raw',      'simhash', 8, False),
]
# Each LSH variant produces both mean and +var points
POINT_METHODS_MEAN = [name + ' (mean)' for name, _, _, _ in LSH_VARIANTS]
POINT_METHODS_VAR = [name + ' (+var)' for name, _, _, _ in LSH_VARIANTS]
POINT_METHODS = POINT_METHODS_MEAN + POINT_METHODS_VAR

CURVE_COLORS = {
    'Oracle':          '#2ca02c',
    'Uniform':         '#7fbf7f',
    'KMeans (mean)':   'darkorange',
    'KMeans (+var)':   '#d62728',
}
CURVE_MARKERS = {
    'Oracle':          '^',
    'Uniform':         's',
    'KMeans (mean)':   'o',
    'KMeans (+var)':   'X',
}
CURVE_LINESTYLES = {
    'Oracle': '-', 'Uniform': '-',
    'KMeans (mean)': '-', 'KMeans (+var)': '--',
}

# Point colors: mean = lighter, +var = darker
POINT_COLORS = {}
POINT_MARKERS = {}
_base_colors = {
    'CP-LSH centered': ('#ff7f0e', '#d62728'),
    'CP-LSH raw': ('#ffbb78', '#ff9896'),
    'SimHash k=8 centered': ('#1f77b4', '#9467bd'),
    'SimHash k=8 raw': ('#aec7e8', '#c5b0d5'),
}
for base, (c_mean, c_var) in _base_colors.items():
    POINT_COLORS[base + ' (mean)'] = c_mean
    POINT_COLORS[base + ' (+var)'] = c_var
    POINT_MARKERS[base + ' (mean)'] = 'o'
    POINT_MARKERS[base + ' (+var)'] = 'X'


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
# INCREMENTAL CLUSTER STATE (tracks mean + full covariance)
# ============================================================================

class IncrementalVarClusters:
    """G clusters tracking sum, outer-product sum (full cov), and value sums."""

    def __init__(self, G, head_dim):
        self.G = G
        self.d = head_dim
        self.key_sums = np.zeros((G, head_dim), dtype=np.float64)
        self.key_outer_sums = np.zeros((G, head_dim, head_dim), dtype=np.float64)
        self.val_sums = np.zeros((G, head_dim), dtype=np.float64)
        self.counts = np.zeros(G, dtype=np.float64)

    def add_key(self, cluster_id, key_vec, val_vec):
        self.key_sums[cluster_id] += key_vec
        self.key_outer_sums[cluster_id] += np.outer(key_vec, key_vec)
        self.val_sums[cluster_id] += val_vec
        self.counts[cluster_id] += 1

    def query_mean(self, query):
        active = self.counts > 0
        if not active.any():
            return np.zeros(self.d)
        c = self.counts[active]
        mk = self.key_sums[active] / c[:, None]
        mv = self.val_sums[active] / c[:, None]
        logits = (query @ mk.T) / np.sqrt(self.d) + np.log(c + 1e-10)
        return softmax(logits) @ mv

    def query_var(self, query):
        active = self.counts > 0
        if not active.any():
            return np.zeros(self.d)
        c = self.counts[active]
        mk = self.key_sums[active] / c[:, None]
        mv = self.val_sums[active] / c[:, None]
        # Full covariance: Σ = E[kk^T] - μμ^T
        cov = self.key_outer_sums[active] / c[:, None, None] - mk[:, :, None] * mk[:, None, :]
        cov = np.maximum(cov, 0)  # element-wise clamp for numerical stability
        # Quadratic form: q^T Σ q / (2d)
        correction = np.einsum('i,gij,j->g', query, cov, query) / (2 * self.d)
        logits = (query @ mk.T) / np.sqrt(self.d)
        logits += np.log(c + 1e-10)
        logits += correction
        return softmax(logits) @ mv

    def query_per_key_logits(self, query, labels, n_keys):
        """Return per-key logits (without log(c)) for both mean and +var."""
        active = self.counts > 0
        if not active.any():
            return np.zeros(n_keys), np.zeros(n_keys)
        c = self.counts[active]
        mk = self.key_sums[active] / c[:, None]

        # Mean logits per group (without log(c) — N-way softmax handles sizes)
        group_logits_mean = (query @ mk.T) / np.sqrt(self.d)

        # +Var logits per group
        cov = self.key_outer_sums[active] / c[:, None, None] - mk[:, :, None] * mk[:, None, :]
        cov = np.maximum(cov, 0)
        correction = np.einsum('i,gij,j->g', query, cov, query) / (2 * self.d)
        group_logits_var = group_logits_mean + correction

        # Map active group indices: original group id -> active index
        active_indices = np.where(active)[0]
        group_to_active = np.full(self.G, -1, dtype=np.int32)
        group_to_active[active_indices] = np.arange(len(active_indices))

        # Broadcast to per-key logits
        mean_logits = np.full(n_keys, -1e10)
        var_logits = np.full(n_keys, -1e10)
        for i in range(n_keys):
            a_idx = group_to_active[labels[i]]
            if a_idx >= 0:
                mean_logits[i] = group_logits_mean[a_idx]
                var_logits[i] = group_logits_var[a_idx]

        return mean_logits, var_logits


# ============================================================================
# CLUSTER ASSIGNMENTS
# ============================================================================

def assign_kmeans(K_mat, seq_len):
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

def assign_lsh_labels(keys, seq_len, lsh_type, k, center_keys, seed):
    if lsh_type == 'cp':
        idx = CrossPolytopeIndex(num_tables=1, max_cp=k, head_dim=HEAD_DIM,
                                 center_keys=center_keys, seed=seed)
        idx.build_index(keys[:seq_len])
        raw_labels = idx.key_codes[:, 0, 0]
    elif lsh_type == 'simhash':
        idx = SimHashIndex(num_tables=1, max_depth=k, head_dim=HEAD_DIM,
                           center_keys=center_keys, seed=seed)
        idx.build_index(keys[:seq_len])
        codes = idx.key_codes[:, 0, :]
        raw_labels = np.zeros(seq_len, dtype=np.int64)
        for bit in range(k):
            raw_labels = raw_labels * 2 + codes[:, bit].astype(np.int64)
    else:
        raise ValueError(f"Unknown lsh_type: {lsh_type}")
    _, labels = np.unique(raw_labels, return_inverse=True)
    return labels.astype(np.int32), len(np.unique(labels))


# ============================================================================
# INCREMENTAL EVALUATION
# ============================================================================

def run_clustering_multi_budget(Q, K_mat, V, seq_len, labels_by_budget,
                                test_positions_by_region):
    """Walk once, produce both mean and +var outputs for all budgets."""
    all_test = set()
    for region in REGIONS:
        all_test.update(test_positions_by_region[region])
    if not all_test:
        empty = {b: {r: [] for r in REGIONS} for b in labels_by_budget}
        return empty, empty
    max_test_pos = max(all_test)
    test_set = {r: set(test_positions_by_region[r]) for r in REGIONS}

    states = {}
    for budget, labels in labels_by_budget.items():
        G = int(labels.max()) + 1
        states[budget] = (IncrementalVarClusters(G, HEAD_DIM), labels)

    out_mean = {b: {r: [] for r in REGIONS} for b in labels_by_budget}
    out_var = {b: {r: [] for r in REGIONS} for b in labels_by_budget}

    for pos in range(max_test_pos + 1):
        for budget, (state, labels) in states.items():
            state.add_key(labels[pos], K_mat[pos], V[pos])
        for region in REGIONS:
            if pos in test_set[region]:
                for budget, (state, labels) in states.items():
                    out_mean[budget][region].append(
                        (pos, state.query_mean(Q[pos])))
                    out_var[budget][region].append(
                        (pos, state.query_var(Q[pos])))

    return out_mean, out_var


def run_lsh_single_logits(Q, K_mat, V, seq_len, labels, test_positions_by_region):
    """Single label set — return per-key logits (mean and +var) for each test query."""
    G = int(labels.max()) + 1
    state = IncrementalVarClusters(G, HEAD_DIM)

    all_test = set()
    for region in REGIONS:
        all_test.update(test_positions_by_region[region])
    if not all_test:
        empty = {r: [] for r in REGIONS}
        return empty, empty
    max_test_pos = max(all_test)
    test_set = {r: set(test_positions_by_region[r]) for r in REGIONS}

    out_mean = {r: [] for r in REGIONS}
    out_var = {r: [] for r in REGIONS}

    for pos in range(max_test_pos + 1):
        state.add_key(labels[pos], K_mat[pos], V[pos])
        for region in REGIONS:
            if pos in test_set[region]:
                n_keys = pos + 1
                mean_logits, var_logits = state.query_per_key_logits(
                    Q[pos], labels[:n_keys], n_keys)
                out_mean[region].append((pos, mean_logits))
                out_var[region].append((pos, var_logits))

    return out_mean, out_var


# ============================================================================
# MAIN COMPUTATION
# ============================================================================

def analyze_layer(data_path, selected_indices, layer_name, master_rng):
    print(f"\n{'='*60}")
    print(f"  {layer_name}")
    print(f"{'='*60}")

    curve_errors = {
        region: {m: {b: [] for b in BUDGETS} for m in CURVE_METHODS}
        for region in REGIONS
    }
    point_errors = {region: {m: [] for m in POINT_METHODS} for region in REGIONS}
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
            all_test = set()
            for r in REGIONS:
                all_test.update(region_positions[r])

            ex_count += 1
            print(f"\n  [{ex_count}/{total_examples}] Example {idx}: seq_len={seq_len}")

            # Ground truth
            t0 = time.time()
            gt_data = {}
            for qpos in all_test:
                q = Q[qpos]
                keys = K_mat[:qpos+1]; vals = V[:qpos+1]
                logits = (q @ keys.T) / np.sqrt(HEAD_DIM)
                full_w = softmax(logits)
                gt_data[qpos] = (full_w @ vals, logits, full_w, keys, vals)
            print(f"    Ground truth: {time.time()-t0:.1f}s")

            # Baselines
            t0 = time.time()
            brng = np.random.default_rng(master_rng.integers(2**32))
            for region in REGIONS:
                for qpos in region_positions[region]:
                    fo, logits, fw, keys, vals = gt_data[qpos]
                    n = qpos + 1
                    for budget in BUDGETS:
                        b = min(budget, n)
                        oo, _ = oracle_sampling(Q[qpos], keys, vals, logits, fw, b)
                        curve_errors[region]['Oracle'][budget].append(rel_l2(oo, fo))
                        ui = brng.choice(n, size=b, replace=False)
                        ou = softmax(logits[ui]) @ vals[ui]
                        curve_errors[region]['Uniform'][budget].append(rel_l2(ou, fo))
            print(f"    Baselines: {time.time()-t0:.1f}s")

            # KMeans
            t0 = time.time()
            km_assign = assign_kmeans(K_mat, seq_len)
            print(f"    KMeans assign: {time.time()-t0:.1f}s")
            t0 = time.time()
            km_mean, km_var = run_clustering_multi_budget(
                Q, K_mat, V, seq_len, km_assign, region_positions)
            for budget in BUDGETS:
                for region in REGIONS:
                    for qpos, out in km_mean[budget][region]:
                        curve_errors[region]['KMeans (mean)'][budget].append(
                            rel_l2(out, gt_data[qpos][0]))
                    for qpos, out in km_var[budget][region]:
                        curve_errors[region]['KMeans (+var)'][budget].append(
                            rel_l2(out, gt_data[qpos][0]))
            print(f"    KMeans eval: {time.time()-t0:.1f}s")

            # LSH variants — average logits across iterations, then compute output
            # Accumulators: {region: {variant_base: {qi: (mean_logits_acc, var_logits_acc, count)}}}
            lsh_logit_accum = {}
            for region in REGIONS:
                lsh_logit_accum[region] = {}
                for base_name, _, _, _ in LSH_VARIANTS:
                    lsh_logit_accum[region][base_name] = {}

            iter_g_effs = {m: [] for m in POINT_METHODS}

            print(f"    Running {N_ITERS} LSH iterations...")
            t0 = time.time()
            for it in range(N_ITERS):
                iter_seed = int(master_rng.integers(2**32))
                iter_rng = np.random.default_rng(iter_seed)
                for base_name, lsh_type, k, center in LSH_VARIANTS:
                    lsh_seed = int(iter_rng.integers(2**32))
                    labels, G_eff = assign_lsh_labels(
                        K_mat, seq_len, lsh_type, k, center, lsh_seed)
                    mean_name = base_name + ' (mean)'
                    var_name = base_name + ' (+var)'
                    iter_g_effs[mean_name].append(G_eff)
                    iter_g_effs[var_name].append(G_eff)

                    o_mean, o_var = run_lsh_single_logits(
                        Q, K_mat, V, seq_len, labels, region_positions)
                    for region in REGIONS:
                        for qi, (qpos, mean_logits) in enumerate(o_mean[region]):
                            if qi not in lsh_logit_accum[region][base_name]:
                                n_keys = qpos + 1
                                lsh_logit_accum[region][base_name][qi] = {
                                    'qpos': qpos,
                                    'mean_acc': np.zeros(n_keys, dtype=np.float64),
                                    'var_acc': np.zeros(n_keys, dtype=np.float64),
                                    'count': 0,
                                }
                            entry = lsh_logit_accum[region][base_name][qi]
                            entry['mean_acc'] += mean_logits
                            _, var_logits = o_var[region][qi]
                            entry['var_acc'] += var_logits
                            entry['count'] += 1

                if (it+1) % 5 == 0 or it == 0:
                    el = time.time() - t0
                    eta = el/(it+1) * (N_ITERS-it-1)
                    print(f"\r    Iter {it+1}/{N_ITERS}  "
                          f"elapsed {format_eta(el)}  ETA {format_eta(eta)}",
                          end="", flush=True)

            print(f"\n    LSH done in {format_eta(time.time()-t0)}")

            # Compute outputs from averaged logits
            for base_name, _, _, _ in LSH_VARIANTS:
                mean_name = base_name + ' (mean)'
                var_name = base_name + ' (+var)'
                avg_g = float(np.mean(iter_g_effs[mean_name]))
                g_eff_values[mean_name].append(avg_g)
                g_eff_values[var_name].append(avg_g)

                for region in REGIONS:
                    for qi, entry in lsh_logit_accum[region][base_name].items():
                        qpos = entry['qpos']
                        n_keys = qpos + 1
                        avg_mean_logits = entry['mean_acc'] / entry['count']
                        avg_var_logits = entry['var_acc'] / entry['count']
                        vals = V[:n_keys]
                        fo = gt_data[qpos][0]

                        out_mean = softmax(avg_mean_logits) @ vals
                        out_var = softmax(avg_var_logits) @ vals

                        point_errors[region][mean_name].append(rel_l2(out_mean, fo))
                        point_errors[region][var_name].append(rel_l2(out_var, fo))

            # Print G_eff for base variants only
            for base_name, _, _, _ in LSH_VARIANTS:
                mn = base_name + ' (mean)'
                print(f"    {base_name}: avg G_eff = {np.mean(iter_g_effs[mn]):.1f}")

            del Q, K_mat, V, gt_data

    # Aggregate
    results = {'budgets': BUDGETS}
    for region in REGIONS:
        results[region] = {}
        for m in CURVE_METHODS:
            results[region][f'{m}_mean'] = [
                float(np.mean(curve_errors[region][m][b]))
                if curve_errors[region][m][b] else 0.0 for b in BUDGETS]
            results[region][f'{m}_std'] = [
                float(np.std(curve_errors[region][m][b]))
                if curve_errors[region][m][b] else 0.0 for b in BUDGETS]

    results['lsh_points'] = {}
    for m in POINT_METHODS:
        avg_g = float(np.mean(g_eff_values[m]))
        results['lsh_points'][m] = {
            'g_eff': avg_g,
            'g_eff_all': [float(v) for v in g_eff_values[m]],
        }
        for region in REGIONS:
            errs = point_errors[region][m]
            results['lsh_points'][m][f'{region}_mean'] = float(np.mean(errs)) if errs else 0.0
            results['lsh_points'][m][f'{region}_std'] = float(np.std(errs)) if errs else 0.0

    return results


# ============================================================================
# PLOTTING
# ============================================================================

def _plot_comparison(ax, data, region):
    x = np.array(data['budgets'])
    rd = data[region]
    for method in CURVE_METHODS:
        if f'{method}_mean' not in rd:
            continue
        means = np.array(rd[f'{method}_mean'])
        stds = np.array(rd[f'{method}_std'])
        color = CURVE_COLORS[method]
        marker = CURVE_MARKERS[method]
        ls = CURVE_LINESTYLES.get(method, '-')
        ax.plot(x, means, marker=marker, color=color, lw=2.5, ls=ls,
                label=method, zorder=4, markersize=6)
        ax.fill_between(x, means, means+stds, color=color, alpha=0.10)

    pts = data.get('lsh_points', {})
    # Jitter offset for side-by-side mean vs +var
    jitter_offset = 8  # in data units

    for base_name, _, _, _ in LSH_VARIANTS:
        mean_name = base_name + ' (mean)'
        var_name = base_name + ' (+var)'
        if mean_name not in pts or var_name not in pts:
            continue
        pt_mean = pts[mean_name]
        pt_var = pts[var_name]
        g = pt_mean['g_eff']
        me_mean = pt_mean.get(f'{region}_mean', 0)
        se_mean = pt_mean.get(f'{region}_std', 0)
        me_var = pt_var.get(f'{region}_mean', 0)
        se_var = pt_var.get(f'{region}_std', 0)

        g_left = g - jitter_offset
        g_right = g + jitter_offset

        # Connecting line between mean and +var
        ax.plot([g_left, g_right], [me_mean, me_var],
                color='gray', lw=0.8, ls='-', alpha=0.5, zorder=3)

        # Mean point (left)
        ax.scatter([g_left], [me_mean], marker=POINT_MARKERS[mean_name],
                   color=POINT_COLORS[mean_name],
                   s=180, zorder=6, edgecolors='black', linewidths=0.8,
                   label=f'{mean_name} (G={g:.0f})')
        ax.errorbar([g_left], [me_mean], yerr=[se_mean], fmt='none',
                    color=POINT_COLORS[mean_name],
                    capsize=4, capthick=1.5, zorder=5)

        # +Var point (right)
        ax.scatter([g_right], [me_var], marker=POINT_MARKERS[var_name],
                   color=POINT_COLORS[var_name],
                   s=180, zorder=6, edgecolors='black', linewidths=0.8,
                   label=f'{var_name} (G={g:.0f})')
        ax.errorbar([g_right], [me_var], yerr=[se_var], fmt='none',
                    color=POINT_COLORS[var_name],
                    capsize=4, capthick=1.5, zorder=5)

    ax.set_title(REGION_DISPLAY[region], fontsize=12, fontweight='bold')
    ax.set_xlabel('Budget (num groups)', fontsize=10)
    ax.set_ylabel('Relative L2 Error', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(left=0, right=550)
    ax.grid(True, alpha=0.3, ls='--', which='both')


def make_figures(all_results, output_dir):
    cfg = all_results.get('config', {})
    n_ex = cfg.get('num_examples', NUM_EXAMPLES)
    n_q = cfg.get('num_test_queries', NUM_TEST_QUERIES)
    n_it = cfg.get('n_iters', N_ITERS)
    subtitle = (f'{n_ex} examples, {n_q} queries each, {n_it} iters  |  '
                f'Llama-3-8B  |  Solid=mean, Dashed=+var  |  Shaded = +1 std')

    for layer in LAYERS:
        ld = all_results[layer]
        ls = 'first_layer' if 'first' in layer else 'last_layer'
        lt = 'First Layer (Layer 0)' if 'first' in layer else 'Last Layer (Layer 31)'

        # Figure 1: by_region
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
        for i, region in enumerate(REGIONS):
            _plot_comparison(axes[i], ld, region)
            if i > 0:
                axes[i].set_ylabel('')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', fontsize=7,
                   framealpha=0.95, ncol=4, bbox_to_anchor=(0.5, -0.08))
        fig.suptitle(f'Variance Correction (Full Cov) — {lt}\n{subtitle}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.12, 1, 0.94])
        save_figure(fig, output_dir / f'by_region_{ls}.png', dpi=200)
        plt.close(fig)

        # Figure 2: summary — KMeans mean vs +var head-to-head
        fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)
        for i, region in enumerate(REGIONS):
            ax = axes[i]
            x = np.array(ld['budgets'])
            rd = ld[region]
            for method in ['Oracle', 'KMeans (mean)', 'KMeans (+var)']:
                if f'{method}_mean' not in rd:
                    continue
                means = np.array(rd[f'{method}_mean'])
                stds = np.array(rd[f'{method}_std'])
                color = CURVE_COLORS[method]
                marker = CURVE_MARKERS[method]
                ls_style = CURVE_LINESTYLES.get(method, '-')
                ax.plot(x, means, marker=marker, color=color, lw=2.5,
                        ls=ls_style, label=method, zorder=4, markersize=6)
                ax.fill_between(x, means, means+stds, color=color, alpha=0.10)
            ax.set_title(REGION_DISPLAY[region], fontsize=12, fontweight='bold')
            ax.set_xlabel('Budget', fontsize=10)
            if i == 0:
                ax.set_ylabel('Relative L2 Error', fontsize=10)
            ax.set_yscale('log')
            ax.set_xlim(left=0, right=550)
            ax.grid(True, alpha=0.3, ls='--', which='both')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', fontsize=10,
                   framealpha=0.95, ncol=3, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f'Mean vs +Var Correction (Full Cov) — {lt}\n{subtitle}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.06, 1, 0.94])
        save_figure(fig, output_dir / f'summary_{ls}.png', dpi=200)
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
        print("Plot-only mode...")
        with open(output_dir / 'full_results.json') as f:
            all_results = json.load(f)
        make_figures(all_results, output_dir)
        print("Done!")
        return

    print("=" * 60)
    print("VARIANCE CORRECTION (FULL COVARIANCE)")
    print("=" * 60)
    print(f"Config: {NUM_EXAMPLES} examples, {NUM_TEST_QUERIES} queries/region")
    print(f"LSH N_ITERS={N_ITERS}")
    print(f"Budgets: {BUDGETS}")
    print()

    t0 = time.time()
    master_rng = np.random.default_rng(SEED)
    np.random.seed(SEED)
    data_path = os.path.join(script_dir, DATA_PATH)

    with open(data_path, 'r') as f:
        total = sum(1 for _ in f)
    n_select = min(NUM_EXAMPLES, total)
    selected = sorted(master_rng.choice(total, n_select, replace=False).tolist())
    print(f"Selected {n_select} examples: {selected}")

    all_results = {
        'config': {
            'num_examples': n_select, 'num_test_queries': NUM_TEST_QUERIES,
            'n_iters': N_ITERS, 'budgets': BUDGETS, 'seed': SEED,
            'head_dim': HEAD_DIM, 'regions': REGIONS,
            'curve_methods': CURVE_METHODS, 'point_methods': POINT_METHODS,
        }
    }

    for layer in LAYERS:
        all_results[layer] = analyze_layer(data_path, selected, layer, master_rng)

    with open(output_dir / 'full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_dir / 'full_results.json'}")

    print("\nGenerating figures...")
    make_figures(all_results, output_dir)
    print(f"\nTotal time: {format_eta(time.time() - t0)}")
    print("Done!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
