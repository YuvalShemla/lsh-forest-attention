#!/usr/bin/env python3
"""
RoPE Combined Experiment: Baselines + Sorted-Keys Grouping + Local Grouping + KMeans Grouping

Methods:
  Sampling baselines    — TopK, Uniform, Oracle, OracleVW
  Sorted-keys grouping  — equal, kmeans, log_spaced, quantile, variance  (per-query sort)
  Fixed Quantile        — sort keys by global mean-query once, quantile group
  Per-Query Quantile    — sort keys by each test query, quantile group
  KMeans Keys           — cluster keys by key-vector embedding, attend via cluster reps
  KMeans Values         — cluster keys by value-vector embedding, attend via cluster reps
  KMeans Queries        — cluster queries; test query uses its cluster mean-q for full attn
  Local+Proximity N     — split queries into N contiguous windows, sort keys by window mean-q;
                          route test query to closest window mean (cosine similarity)
  Local+Fixed N         — same precomputation; each query uses the window it falls into
                          by position (no proximity lookup)

Split sizes for local methods: LOCAL_SPLITS = [32, 64, 128, 256]

Usage:
  python compare_rope_combined.py              # compute + plot
  python compare_rope_combined.py --plot-only  # regenerate plots from existing JSON
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

from algorithms.base import softmax
from algorithms.sorted_keys_grouping import grouped_attention, GROUPING_METHODS, group_quantile_weight
from algorithms.oracle import oracle_sampling
from algorithms.oracle_value_weighted import oracle_value_weighted
from visualization.plot_utils import setup_style, save_figure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG — edit for pilot vs full runs
# ============================================================================
DATA_PATH = '../../data/attention_vectors_llama_8b_with_rope.jsonl'
OUTPUT_DIR = Path('../../results/rope_combined_comparison_v2')

NUM_EXAMPLES = 50       # first N examples from the file
NUM_QUERIES  = 20       # last N query positions per example (causal)

LAYERS   = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED     = 42

BUDGETS = [2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096]

# Sorted-keys grouping methods (drop overlap / threshold)
GROUPING_METHODS_USED = {k: v for k, v in GROUPING_METHODS.items()
                          if k not in ('overlap', 'threshold')}

# Window counts for local grouping variants
LOCAL_SPLITS = [8, 16, 32, 64, 128, 256, 512, 1024]


# ============================================================================
# METHOD REGISTRY
# ============================================================================

SAMPLING_METHODS      = ['TopK', 'Uniform', 'Oracle', 'OracleVW']
GROUPING_METHOD_NAMES = [f'group_{k}' for k in GROUPING_METHODS_USED]
FIXED_METHODS         = ['Fixed Quantile', 'Per-Query Quantile']
KMEANS_METHODS        = ['KMeans Keys', 'KMeans Values', 'KMeans Queries']
LOCAL_PROXIMITY       = [f'Local+Proximity N={n}' for n in LOCAL_SPLITS]
LOCAL_FIXED           = [f'Local+Fixed N={n}'     for n in LOCAL_SPLITS]

ALL_METHODS = (SAMPLING_METHODS + GROUPING_METHOD_NAMES
               + FIXED_METHODS + KMEANS_METHODS
               + LOCAL_PROXIMITY + LOCAL_FIXED)

DISPLAY_NAMES = {
    'TopK':               'Top-K (subset softmax)',
    'Uniform':            'Uniform (subset softmax)',
    'Oracle':             'Oracle (sample ~ w)',
    'OracleVW':           'Oracle VW (sample ~ w·‖v‖)',
    'Fixed Quantile':     'Fixed Quantile (global mean-q)',
    'Per-Query Quantile': 'Per-Query Quantile',
    'KMeans Keys':        'KMeans Keys (cluster by k)',
    'KMeans Values':      'KMeans Values (cluster by v)',
    'KMeans Queries':     'KMeans Queries (cluster by q)',
}
for k, v in GROUPING_METHODS_USED.items():
    DISPLAY_NAMES[f'group_{k}'] = f'Sorted-{v}'
for n in LOCAL_SPLITS:
    DISPLAY_NAMES[f'Local+Proximity N={n}'] = f'Local+Proximity N={n}'
    DISPLAY_NAMES[f'Local+Fixed N={n}']     = f'Local+Fixed N={n}'

# ---- colour / marker palette ----
_PROX_COLORS  = ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#084594', '#042f6b']   # blues
_FIXED_COLORS = ['#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704', '#4a1a01']   # oranges

CURVE_STYLES = {
    'TopK':               {'color': '#d62728', 'marker': 'o',  'ls': '-',  'lw': 2.8},
    'Uniform':            {'color': '#ff7f0e', 'marker': 's',  'ls': '--', 'lw': 2.2},
    'Oracle':             {'color': '#2ca02c', 'marker': '^',  'ls': '--', 'lw': 2.5},
    'OracleVW':           {'color': '#006400', 'marker': 'v',  'ls': '--', 'lw': 2.2},
    # Sorted-keys grouping
    'group_equal':        {'color': '#1f77b4', 'marker': '>',  'ls': '-',  'lw': 1.8},
    'group_kmeans':       {'color': '#9467bd', 'marker': 'D',  'ls': '-',  'lw': 1.8},
    'group_log_spaced':   {'color': '#8c564b', 'marker': 'X',  'ls': '-',  'lw': 1.8},
    'group_quantile':     {'color': '#17becf', 'marker': 'h',  'ls': '-',  'lw': 1.8},
    'group_variance':     {'color': '#bcbd22', 'marker': 'd',  'ls': '-',  'lw': 1.8},
    # Fixed / per-query
    'Fixed Quantile':     {'color': 'lightskyblue', 'marker': 'o',  'ls': '-',  'lw': 2.0},
    'Per-Query Quantile': {'color': 'navy',          'marker': 'D',  'ls': '--', 'lw': 2.0},
    # KMeans grouping
    'KMeans Keys':        {'color': 'darkorange',   'marker': 'X',  'ls': '-',  'lw': 2.0},
    'KMeans Values':      {'color': '#e377c2',       'marker': 'P',  'ls': '-',  'lw': 2.0},
    'KMeans Queries':     {'color': '#7f7f7f',       'marker': '*',  'ls': '-',  'lw': 2.0},
}
_LOC_MARKERS_PROX  = ['<', 'P', 'h', 'd', '8', 'p', 'H', '*']
_LOC_MARKERS_FIXED = ['>', 'X', 'v', '^', '8', 'p', 'H', '*']
for i, n in enumerate(LOCAL_SPLITS):
    CURVE_STYLES[f'Local+Proximity N={n}'] = {
        'color': _PROX_COLORS[i],  'marker': _LOC_MARKERS_PROX[i],  'ls': '-', 'lw': 1.8}
    CURVE_STYLES[f'Local+Fixed N={n}'] = {
        'color': _FIXED_COLORS[i], 'marker': _LOC_MARKERS_FIXED[i], 'ls': '-', 'lw': 1.8}


# ============================================================================
# HELPERS
# ============================================================================

def rel_l2(approx, truth):
    return np.linalg.norm(approx - truth) / (np.linalg.norm(truth) + 1e-8)


def format_eta(seconds):
    if seconds < 60:     return f"{seconds:.0f}s"
    elif seconds < 3600: return f"{seconds/60:.1f}min"
    else:                return f"{seconds/3600:.1f}h"


class ProgressTracker:
    def __init__(self, total, prefix=""):
        self.total  = total
        self.done   = 0
        self.t0     = time.time()
        self.prefix = prefix

    def step(self, info=""):
        self.done += 1
        el  = time.time() - self.t0
        rem = el / self.done * (self.total - self.done) if self.done < self.total else 0
        print(f"\r  {self.prefix}[{100*self.done/self.total:5.1f}%] "
              f"{self.done}/{self.total}  elapsed {format_eta(el)}  "
              f"ETA {format_eta(rem)}  {info}", end="", flush=True)

    def finish(self):
        print(f"\r  {self.prefix}Done in {format_eta(time.time()-self.t0)}" + " " * 60)


# ============================================================================
# PRE-COMPUTATION: GROUPING / SORTING HELPERS
# ============================================================================

def build_fixed_quantile_grouping(mean_q, K_mat, n_keys):
    """Sort keys by mean-query logits; return (sorted_indices, {budget: labels})."""
    logits         = (mean_q @ K_mat[:n_keys].T) / np.sqrt(HEAD_DIM)
    weights        = softmax(logits)
    sorted_indices = np.argsort(logits)[::-1]
    sorted_weights = weights[sorted_indices]
    labels = {b: group_quantile_weight(sorted_weights, min(b, n_keys)) for b in BUDGETS}
    return sorted_indices, labels


def fixed_grouping_attention(query, keys, values, sorted_indices, labels, head_dim):
    """Attend via group representatives (mean key, mean value, size-weighted logit)."""
    unique_labels = np.unique(labels)
    G = len(unique_labels)
    mean_keys   = np.zeros((G, head_dim), dtype=np.float64)
    mean_values = np.zeros((G, head_dim), dtype=np.float64)
    group_sizes = np.zeros(G,             dtype=np.float64)
    for i, g in enumerate(unique_labels):
        mask           = labels == g
        idxs           = sorted_indices[mask]
        mean_keys[i]   = keys[idxs].mean(axis=0)
        mean_values[i] = values[idxs].mean(axis=0)
        group_sizes[i] = mask.sum()
    gl = (query @ mean_keys.T) / np.sqrt(head_dim) + np.log(group_sizes + 1e-10)
    return softmax(gl) @ mean_values, G


# ============================================================================
# PRE-COMPUTATION: KMEANS GROUPING
# ============================================================================

def _kmeans_cluster_attention(query, keys, values, labels, head_dim):
    """
    Given cluster labels for each key, attend via cluster representatives:
      mean key, mean value, size-weighted logit correction.
    """
    unique_labels = np.unique(labels)
    G = len(unique_labels)
    mean_keys   = np.zeros((G, head_dim), dtype=np.float64)
    mean_values = np.zeros((G, head_dim), dtype=np.float64)
    group_sizes = np.zeros(G,             dtype=np.float64)
    for i, g in enumerate(unique_labels):
        mask           = labels == g
        mean_keys[i]   = keys[mask].mean(axis=0)
        mean_values[i] = values[mask].mean(axis=0)
        group_sizes[i] = mask.sum()
    gl = (query @ mean_keys.T) / np.sqrt(head_dim) + np.log(group_sizes + 1e-10)
    return softmax(gl) @ mean_values, G


def build_kmeans_key_groups(keys, n_keys):
    """Cluster keys by their key vectors for each budget. Returns {budget: labels}."""
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


def build_kmeans_value_groups(values, n_keys):
    """Cluster keys by their value vectors for each budget. Returns {budget: labels}."""
    groups = {}
    for budget in BUDGETS:
        b = min(budget, n_keys)
        if b >= n_keys:
            groups[budget] = np.arange(n_keys)
        else:
            km = KMeans(n_clusters=b, n_init=3, max_iter=100, random_state=SEED)
            km.fit(values[:n_keys])
            groups[budget] = km.labels_
    return groups


def build_kmeans_query_clusters(Q, seq_len):
    """
    Cluster queries by their query vectors for each budget.
    Returns {budget: (labels, centers)}.
    """
    data = {}
    for budget in BUDGETS:
        b = min(budget, seq_len)
        if b >= seq_len:
            data[budget] = None   # budget >= seq_len: no compression, use exact query
        else:
            km = KMeans(n_clusters=b, n_init=3, max_iter=100, random_state=SEED)
            km.fit(Q[:seq_len])
            data[budget] = (km.labels_, km.cluster_centers_)
    return data


# ============================================================================
# PRE-COMPUTATION: LOCAL WINDOW STRUCTURES
# ============================================================================

def build_local_windows(Q, K_mat, seq_len, n_splits):
    """
    Divide the sequence into n_splits contiguous windows.
    For each window compute mean(Q[window]) and sort keys by it.

    Returns:
      boundaries   : int array length n_splits+1
      window_means : float32 array (n_splits, HEAD_DIM)
      window_data  : list of (sorted_indices, labels_dict)
    """
    n_splits     = min(n_splits, seq_len)
    boundaries   = np.round(np.linspace(0, seq_len, n_splits + 1)).astype(int)
    window_means = np.zeros((n_splits, HEAD_DIM), dtype=np.float32)
    window_data  = []

    for g in range(n_splits):
        start, end = boundaries[g], boundaries[g + 1]
        if end <= start:
            prev = window_means[max(0, g - 1)]
            window_means[g] = prev
            window_data.append(window_data[-1] if window_data else None)
            continue
        mq = Q[start:end].mean(axis=0)
        window_means[g] = mq
        si, lab = build_fixed_quantile_grouping(mq, K_mat, seq_len)
        window_data.append((si, lab))

    return boundaries, window_means, window_data


def proximity_window(query, window_means):
    """Index of window whose mean is closest (cosine) to query."""
    q_n = query        / (np.linalg.norm(query)                                 + 1e-10)
    m_n = window_means / (np.linalg.norm(window_means, axis=1, keepdims=True)   + 1e-10)
    return int(np.argmax(m_n @ q_n))


def fixed_window(qpos, boundaries):
    """Index of the window that position qpos falls into."""
    g = int(np.searchsorted(boundaries[1:], qpos, side='right'))
    return min(g, len(boundaries) - 2)


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_layer(examples, layer_name, rng):
    print(f"\n{'='*60}")
    print(f"  {layer_name}")
    print(f"{'='*60}")

    errors = {m: {b: [] for b in BUDGETS} for m in ALL_METHODS}
    total_queries = NUM_QUERIES * len(examples)
    progress = ProgressTracker(total_queries, f"{layer_name}: ")

    for ex_idx, example in enumerate(examples):
        Q      = np.array(example[layer_name]['Q'], dtype=np.float32)
        K_mat  = np.array(example[layer_name]['K'], dtype=np.float32)
        V      = np.array(example[layer_name]['V'], dtype=np.float32)
        seq_len = Q.shape[0]

        # ---- Fixed Quantile (one global sort) ----
        global_mean_q = Q.mean(axis=0)
        fixed_si, fixed_labels = build_fixed_quantile_grouping(global_mean_q, K_mat, seq_len)

        # ---- KMeans groupings ----
        print(f"\n    [ex {ex_idx+1}] KMeans Keys...", end="", flush=True)
        km_key_labels = build_kmeans_key_groups(K_mat, seq_len)
        print(" Values...", end="", flush=True)
        km_val_labels = build_kmeans_value_groups(V, seq_len)
        print(" Queries...", end="", flush=True)
        km_query_data = build_kmeans_query_clusters(Q, seq_len)
        print(" done", flush=True)

        # ---- Local window structures ----
        print(f"    [ex {ex_idx+1}] Local windows...", end="", flush=True)
        local_data = {}
        for n in LOCAL_SPLITS:
            local_data[n] = build_local_windows(Q, K_mat, seq_len, n)
        print(" done", flush=True)

        # ---- Test queries — last NUM_QUERIES positions ----
        query_positions = list(range(seq_len - NUM_QUERIES, seq_len))

        for qi, qpos in enumerate(query_positions):
            q      = Q[qpos]
            keys   = K_mat[:qpos + 1]
            vals   = V[:qpos + 1]
            n_keys = qpos + 1

            logits   = (q @ keys.T) / np.sqrt(HEAD_DIM)
            full_w   = softmax(logits)
            full_out = full_w @ vals

            # causal mask for global fixed-sort (built over full seq_len)
            cm_fixed  = fixed_si < n_keys
            vsi_fixed = fixed_si[cm_fixed]

            for budget in BUDGETS:
                b = min(budget, n_keys)

                # ---- Sampling baselines ----
                idx_topk = np.argpartition(logits, -b)[-b:]
                errors['TopK'][budget].append(
                    rel_l2(softmax(logits[idx_topk]) @ vals[idx_topk], full_out))

                u_idx = rng.choice(n_keys, size=b, replace=False)
                errors['Uniform'][budget].append(
                    rel_l2(softmax(logits[u_idx]) @ vals[u_idx], full_out))

                out_oracle, _ = oracle_sampling(q, keys, vals, logits, full_w, b)
                errors['Oracle'][budget].append(rel_l2(out_oracle, full_out))

                out_vw, _ = oracle_value_weighted(q, keys, vals, logits, full_w, b)
                errors['OracleVW'][budget].append(rel_l2(out_vw, full_out))

                # ---- Sorted-keys grouping (per-query sort) ----
                for mk in GROUPING_METHODS_USED:
                    _, go = grouped_attention(logits, vals, full_w, b, method=mk)
                    errors[f'group_{mk}'][budget].append(rel_l2(go, full_out))

                # ---- Fixed Quantile (global mean-q) ----
                fl_fixed = fixed_labels[budget][cm_fixed]
                out_fq, _ = fixed_grouping_attention(
                    q, keys, vals, vsi_fixed, fl_fixed, HEAD_DIM)
                errors['Fixed Quantile'][budget].append(rel_l2(out_fq, full_out))

                # ---- Per-Query Quantile ----
                _, out_pq = grouped_attention(logits, vals, full_w, b, method='quantile')
                errors['Per-Query Quantile'][budget].append(rel_l2(out_pq, full_out))

                # ---- KMeans Keys ----
                km_k_lbl = km_key_labels[budget][:n_keys]
                out_kmk, _ = _kmeans_cluster_attention(q, keys, vals, km_k_lbl, HEAD_DIM)
                errors['KMeans Keys'][budget].append(rel_l2(out_kmk, full_out))

                # ---- KMeans Values ----
                km_v_lbl = km_val_labels[budget][:n_keys]
                out_kmv, _ = _kmeans_cluster_attention(q, keys, vals, km_v_lbl, HEAD_DIM)
                errors['KMeans Values'][budget].append(rel_l2(out_kmv, full_out))

                # ---- KMeans Queries ----
                # Use cluster mean-query to compute full attention over causal keys
                qkm = km_query_data[budget]
                if qkm is None:
                    # budget >= seq_len: no compression needed
                    out_kmq = full_out
                else:
                    q_labels, q_centers = qkm
                    mean_q_cluster = q_centers[q_labels[qpos]]
                    cluster_logits = (mean_q_cluster @ keys.T) / np.sqrt(HEAD_DIM)
                    out_kmq = softmax(cluster_logits) @ vals
                errors['KMeans Queries'][budget].append(rel_l2(out_kmq, full_out))

                # ---- Local+Proximity and Local+Fixed ----
                for n in LOCAL_SPLITS:
                    boundaries, window_means, window_data = local_data[n]

                    # Proximity: route to window whose mean-q is cosine-closest
                    g_prox = proximity_window(q, window_means)
                    si_p, lab_p = window_data[g_prox]
                    cm_p = si_p < n_keys
                    out_p, _ = fixed_grouping_attention(
                        q, keys, vals, si_p[cm_p], lab_p[budget][cm_p], HEAD_DIM)
                    errors[f'Local+Proximity N={n}'][budget].append(rel_l2(out_p, full_out))

                    # Fixed: use the window that qpos positionally falls into
                    g_fix = fixed_window(qpos, boundaries)
                    si_f, lab_f = window_data[g_fix]
                    cm_f = si_f < n_keys
                    out_f, _ = fixed_grouping_attention(
                        q, keys, vals, si_f[cm_f], lab_f[budget][cm_f], HEAD_DIM)
                    errors[f'Local+Fixed N={n}'][budget].append(rel_l2(out_f, full_out))

            progress.step(f"ex {ex_idx+1}/{len(examples)} q {qi+1}/{NUM_QUERIES}")

        del Q, K_mat, V

    progress.finish()

    results = {'budgets': BUDGETS}
    for m in ALL_METHODS:
        results[f'{m}_mean'] = [float(np.mean(errors[m][b])) for b in BUDGETS]
        results[f'{m}_std']  = [float(np.std(errors[m][b]))  for b in BUDGETS]
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def _plot_panel(ax, data, layer_name, method_group, title_suffix=""):
    x = np.array(data['budgets'])
    layer_title = ('First Layer (Layer 0)' if 'first' in layer_name
                   else 'Last Layer (Layer 31)')

    for method in method_group:
        mk = f'{method}_mean'
        if mk not in data:
            continue
        means = np.array(data[mk])
        stds  = np.array(data[f'{method}_std'])
        style = CURVE_STYLES.get(method, {'color': 'gray', 'marker': 'o', 'ls': '-', 'lw': 1.5})
        label = DISPLAY_NAMES.get(method, method)

        ax.plot(x, means,
                marker=style['marker'], linestyle=style['ls'],
                linewidth=style['lw'], markersize=5,
                color=style['color'], label=label, alpha=0.9, zorder=3)
        # Upper-only shading
        ax.fill_between(x, means, means + stds,
                        color=style['color'], alpha=0.12, zorder=1)

    ax.set_xlabel('Budget (num groups / keys)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Relative L2 Error',          fontsize=10, fontweight='bold')
    ax.set_title(f'{layer_title}{title_suffix}', fontsize=11, fontweight='bold', pad=6)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')
    ax.legend(fontsize=8, framealpha=0.95, edgecolor='#cccccc',
              ncol=2, borderpad=0.5, columnspacing=0.8, handletextpad=0.4)


def _two_panel(all_results, output_dir, methods, suptitle, fname, subtitle):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, layer in zip(axes, LAYERS):
        _plot_panel(ax, all_results[layer], layer, methods)
    fig.suptitle(f'{suptitle}\n{subtitle}', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir / fname, dpi=200)
    plt.close(fig)


def _single_panel(all_results, layer, output_dir, methods, title_suffix, fname, subtitle):
    fig, ax = plt.subplots(figsize=(11, 7))
    _plot_panel(ax, all_results[layer], layer, methods, title_suffix)
    fig.suptitle(subtitle, fontsize=9, y=1.01)
    plt.tight_layout()
    save_figure(fig, output_dir / fname, dpi=200)
    plt.close(fig)


def make_figures(all_results, output_dir, num_examples, num_queries):
    subtitle = (f'{num_examples} examples, {num_queries} queries each  |  '
                f'Llama-3-8B + RoPE  |  Shaded = mean + 1 std')

    # ------------------------------------------------------------------
    # Fig 1 — Sampling baselines  (TopK, Uniform, Oracle, OracleVW)
    # ------------------------------------------------------------------
    _two_panel(all_results, output_dir,
               SAMPLING_METHODS,
               'Sampling Baselines',
               'fig1_sampling_baselines.png', subtitle)

    # ------------------------------------------------------------------
    # Fig 2 — Key grouping strategies vs TopK & Oracle
    # Sorted-keys methods + Fixed Quantile + Per-Query Quantile
    # ------------------------------------------------------------------
    _two_panel(all_results, output_dir,
               ['TopK', 'Oracle'] + GROUPING_METHOD_NAMES + FIXED_METHODS,
               'Key Grouping Strategies vs TopK & Oracle',
               'fig2_key_grouping.png', subtitle)

    # ------------------------------------------------------------------
    # Fig 3 — KMeans grouping variants vs TopK & Oracle
    # Keys clustered by key-vector / value-vector / query-cluster mean
    # ------------------------------------------------------------------
    _two_panel(all_results, output_dir,
               ['TopK', 'Oracle'] + KMEANS_METHODS,
               'KMeans Grouping (Keys / Values / Queries) vs TopK & Oracle',
               'fig3_kmeans_grouping.png', subtitle)

    # ------------------------------------------------------------------
    # Fig 4 — Local+Proximity: all split sizes vs TopK & Oracle
    # ------------------------------------------------------------------
    _two_panel(all_results, output_dir,
               ['TopK', 'Oracle', 'Fixed Quantile'] + LOCAL_PROXIMITY,
               'Local+Proximity: Window-Mean Routing vs TopK & Oracle',
               'fig4_local_proximity.png', subtitle)

    # ------------------------------------------------------------------
    # Fig 5 — Local+Fixed: all split sizes vs TopK & Oracle
    # ------------------------------------------------------------------
    _two_panel(all_results, output_dir,
               ['TopK', 'Oracle', 'Fixed Quantile'] + LOCAL_FIXED,
               'Local+Fixed: Positional Assignment vs TopK & Oracle',
               'fig5_local_fixed.png', subtitle)

    # ------------------------------------------------------------------
    # Fig 6 — Local+Proximity vs Local+Fixed head-to-head
    # ------------------------------------------------------------------
    interleaved = ['TopK', 'Oracle']
    for n in LOCAL_SPLITS:
        interleaved += [f'Local+Proximity N={n}', f'Local+Fixed N={n}']
    _two_panel(all_results, output_dir,
               interleaved,
               'Local+Proximity vs Local+Fixed (head-to-head)',
               'fig6_proximity_vs_fixed.png', subtitle)

    # ------------------------------------------------------------------
    # Fig 7 — Global → Local progression
    # Fixed Quantile (global) → Local variants → Per-Query Quantile
    # ------------------------------------------------------------------
    mid_n = LOCAL_SPLITS[len(LOCAL_SPLITS) // 2]
    _two_panel(all_results, output_dir,
               ['TopK', 'Oracle',
                'Fixed Quantile',
                f'Local+Proximity N={mid_n}',
                f'Local+Fixed N={mid_n}',
                'Per-Query Quantile'],
               f'Global → Local (N={mid_n}) → Per-Query: Granularity Sweep',
               'fig7_global_to_local.png', subtitle)

    # ------------------------------------------------------------------
    # Fig 8 — KMeans vs Local: comparing grouping paradigms
    # ------------------------------------------------------------------
    _two_panel(all_results, output_dir,
               ['TopK', 'Oracle'] + KMEANS_METHODS
               + [f'Local+Proximity N={mid_n}', f'Local+Fixed N={mid_n}']
               + ['Fixed Quantile'],
               'KMeans Grouping vs Local Grouping vs TopK & Oracle',
               'fig8_kmeans_vs_local.png', subtitle)

    # ------------------------------------------------------------------
    # Fig 9 — All methods overview
    # ------------------------------------------------------------------
    _two_panel(all_results, output_dir,
               ALL_METHODS,
               'All Methods Overview',
               'fig9_all_methods.png', subtitle)

    # ------------------------------------------------------------------
    # Per-layer single-panel versions (useful for paper, one layer at a time)
    # ------------------------------------------------------------------
    for layer in LAYERS:
        ls = 'first' if 'first' in layer else 'last'

        _single_panel(all_results, layer, output_dir,
                      ['TopK', 'Oracle'] + GROUPING_METHOD_NAMES + FIXED_METHODS,
                      ' — Key Grouping vs TopK & Oracle',
                      f'fig_{ls}_key_grouping.png', subtitle)

        _single_panel(all_results, layer, output_dir,
                      ['TopK', 'Oracle'] + KMEANS_METHODS,
                      ' — KMeans Grouping vs TopK & Oracle',
                      f'fig_{ls}_kmeans_grouping.png', subtitle)

        _single_panel(all_results, layer, output_dir,
                      ['TopK', 'Oracle', 'Fixed Quantile'] + LOCAL_PROXIMITY,
                      ' — Local+Proximity vs TopK & Oracle',
                      f'fig_{ls}_local_proximity.png', subtitle)

        _single_panel(all_results, layer, output_dir,
                      ['TopK', 'Oracle', 'Fixed Quantile'] + LOCAL_FIXED,
                      ' — Local+Fixed vs TopK & Oracle',
                      f'fig_{ls}_local_fixed.png', subtitle)

        _single_panel(all_results, layer, output_dir,
                      interleaved,
                      ' — Local+Proximity vs Local+Fixed',
                      f'fig_{ls}_proximity_vs_fixed.png', subtitle)


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
        print("Plot-only mode — loading JSON...")
        with open(output_dir / 'full_results.json') as f:
            all_results = json.load(f)
        cfg = all_results.get('config', {})
        make_figures(all_results, output_dir,
                     cfg.get('num_examples', NUM_EXAMPLES),
                     cfg.get('num_queries',  NUM_QUERIES))
        print("Done!")
        return

    print("=" * 60)
    print("RoPE COMBINED EXPERIMENT")
    print("=" * 60)
    print(f"Data:          {DATA_PATH}")
    print(f"Config:        {NUM_EXAMPLES} examples × last {NUM_QUERIES} query positions")
    print(f"Budgets:       {BUDGETS}")
    print(f"Local splits:  {LOCAL_SPLITS}")
    print(f"Total methods: {len(ALL_METHODS)}")
    print(f"  Sampling:    {SAMPLING_METHODS}")
    print(f"  Grouping:    {GROUPING_METHOD_NAMES}")
    print(f"  Fixed:       {FIXED_METHODS}")
    print(f"  KMeans:      {KMEANS_METHODS}")
    print(f"  Local prox:  {LOCAL_PROXIMITY}")
    print(f"  Local fixed: {LOCAL_FIXED}")
    print()

    t0  = time.time()
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    data_path = os.path.join(script_dir, DATA_PATH)
    print(f"Loading first {NUM_EXAMPLES} examples from:\n  {data_path}")
    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            if len(examples) >= NUM_EXAMPLES:
                break
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples\n")

    if not examples:
        print("ERROR: no examples loaded — check DATA_PATH")
        sys.exit(1)

    all_results = {
        'config': {
            'num_examples': len(examples),
            'num_queries':  NUM_QUERIES,
            'budgets':      BUDGETS,
            'local_splits': LOCAL_SPLITS,
            'seed':         SEED,
            'head_dim':     HEAD_DIM,
            'data_file':    DATA_PATH,
            'all_methods':  ALL_METHODS,
        }
    }

    for layer in LAYERS:
        all_results[layer] = analyze_layer(examples, layer, rng)

    results_path = output_dir / 'full_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {results_path}")

    print("\nGenerating figures...")
    make_figures(all_results, output_dir, len(examples), NUM_QUERIES)

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
