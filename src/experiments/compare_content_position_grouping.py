#!/usr/bin/env python3
"""
Content Cluster + Position Bins Experiment

Tests a factored approach to key grouping:
  1. Cluster pre-RoPE keys by KMeans (pure content similarity)
  2. Split each content cluster into position bins
  3. Reconstruct RoPE-aware representative keys at each bin's mean position

Hypothesis: factoring content and position produces better group representatives
than clustering post-RoPE keys where the two are entangled.

Baselines: TopK, Uniform, Oracle, KMeans on post-RoPE keys

Usage:
  python compare_content_position_grouping.py              # compute + plot
  python compare_content_position_grouping.py --plot-only  # regenerate plots from JSON
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
from visualization.plot_utils import setup_style, save_figure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ============================================================================
# CONFIG
# ============================================================================
DATA_PATH = '../../data/attention_vectors_long_bench_llama_8b.jsonl'
OUTPUT_DIR = Path('../../results/content_position_grouping')

NUM_EXAMPLES = 10
NUM_QUERIES  = 10
LAYERS       = ['first_layer', 'last_layer']
HEAD_DIM     = 128
SEED         = 42

# Content+Position grid
C_VALUES = [4, 8, 16, 32, 64]
P_VALUES = [1, 2, 4, 8, 16]

# Baseline budget sweep
BUDGETS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Llama-3-8B RoPE config
ROPE_THETA = 500000.0


# ============================================================================
# RoPE FUNCTIONS (inlined from data_extraction/apply_rope_to_vectors.py)
# ============================================================================

def compute_rope_cache(seq_len, head_dim=HEAD_DIM, rope_theta=ROPE_THETA):
    """Compute cos/sin caches for RoPE (matches HF Transformers LlamaRotaryEmbedding)."""
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    position_ids = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(position_ids, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x):
    """Matches HF Transformers rotate_half()."""
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE to Q and K matrices."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rope_single(k_raw, pos_int, cos_cache, sin_cache):
    """Apply RoPE to a single vector at a given position."""
    pos_int = min(pos_int, len(cos_cache) - 1)
    return k_raw * cos_cache[pos_int] + rotate_half(k_raw) * sin_cache[pos_int]


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
# CONTENT + POSITION GROUPING
# ============================================================================

def build_content_position_groups(K_raw, V, cos_cache, sin_cache, C, P, seed=SEED):
    """
    Build content+position groups:
      1. KMeans on K_raw -> C content clusters
      2. Split each cluster into P equal-width position bins
      3. For each non-empty (c, p) group: compute RoPE-aware representative key

    Returns list of dicts, each with:
      rep_k: RoPE-aware representative key [head_dim]
      mean_v: mean value vector [head_dim]
      group_size: number of keys in group
      max_pos: maximum position in group (for causal masking)
    """
    seq_len = K_raw.shape[0]

    # Step 1: KMeans on pre-RoPE keys
    n_clusters = min(C, seq_len)
    km = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=seed)
    labels = km.fit_predict(K_raw)

    # Step 2: Position bin edges
    bin_edges = np.linspace(0, seq_len, P + 1)

    # Step 3: Build groups
    groups = []
    positions = np.arange(seq_len)

    for c in range(n_clusters):
        cluster_mask = labels == c
        cluster_positions = positions[cluster_mask]

        if len(cluster_positions) == 0:
            continue

        # Assign positions to bins
        bin_assignments = np.digitize(cluster_positions, bin_edges[1:])  # 0..P-1
        bin_assignments = np.clip(bin_assignments, 0, P - 1)

        for p in range(P):
            bin_mask = bin_assignments == p
            if not np.any(bin_mask):
                continue

            indices = cluster_positions[bin_mask]
            mean_k_raw = K_raw[indices].mean(axis=0)
            mean_pos = int(round(indices.mean()))
            rep_k = apply_rope_single(mean_k_raw, mean_pos, cos_cache, sin_cache)
            mean_v = V[indices].mean(axis=0)

            groups.append({
                'rep_k': rep_k,
                'mean_v': mean_v,
                'group_size': len(indices),
                'max_pos': int(indices.max()),
                'content_cluster': c,
                'position_bin': p,
            })

    return groups


def content_position_attention(q_rope, groups, head_dim, qpos):
    """
    Attend using content+position groups with causal masking.

    Returns (output, n_groups_used).
    """
    # Filter by causal mask
    causal_groups = [g for g in groups if g['max_pos'] <= qpos]

    if len(causal_groups) == 0:
        return np.zeros(head_dim, dtype=np.float64), 0

    n = len(causal_groups)
    rep_keys = np.array([g['rep_k'] for g in causal_groups], dtype=np.float64)
    mean_vals = np.array([g['mean_v'] for g in causal_groups], dtype=np.float64)
    sizes = np.array([g['group_size'] for g in causal_groups], dtype=np.float64)

    logits = (q_rope @ rep_keys.T) / np.sqrt(head_dim) + np.log(sizes + 1e-10)
    output = softmax(logits) @ mean_vals

    return output, n


# ============================================================================
# KMEANS ON POST-ROPE KEYS (BASELINE)
# ============================================================================

def _kmeans_cluster_attention(query, keys, values, labels, head_dim):
    """Attend via cluster representatives: mean key, mean value, size-weighted logit."""
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


def build_kmeans_rope_groups(K_rope, budgets, seed=SEED):
    """KMeans on post-RoPE keys for each budget. Returns {budget: labels}."""
    seq_len = K_rope.shape[0]
    groups = {}
    for budget in budgets:
        b = min(budget, seq_len)
        if b >= seq_len:
            groups[budget] = np.arange(seq_len)
        else:
            km = KMeans(n_clusters=b, n_init=3, max_iter=100, random_state=seed)
            km.fit(K_rope)
            groups[budget] = km.labels_
    return groups


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_layer(examples_raw, layer_name, rng):
    print(f"\n{'='*60}")
    print(f"  {layer_name}")
    print(f"{'='*60}")

    # Results storage
    # Baselines: {method: {budget: [errors]}}
    baseline_errors = {
        'TopK':       {b: [] for b in BUDGETS},
        'Uniform':    {b: [] for b in BUDGETS},
        'Oracle':     {b: [] for b in BUDGETS},
        'KMeans-RoPE': {b: [] for b in BUDGETS},
    }
    # Content+Position: {(C,P): [errors]} and {(C,P): [budgets]}
    cp_errors = {}
    cp_budgets = {}
    for C in C_VALUES:
        for P in P_VALUES:
            cp_errors[(C, P)] = []
            cp_budgets[(C, P)] = []

    # Per-group error breakdown for diagnostic (C=16, P=4)
    diag_cluster_errors = {}  # {cluster_id: [errors]}
    diag_bin_errors = {}      # {bin_id: [errors]}

    total_queries = NUM_QUERIES * len(examples_raw)
    progress = ProgressTracker(total_queries, f"{layer_name}: ")

    for ex_idx, example in enumerate(examples_raw):
        seq_len = example['sequence_length']
        Q_raw = np.array(example[layer_name]['Q'], dtype=np.float32)
        K_raw = np.array(example[layer_name]['K'], dtype=np.float32)
        V     = np.array(example[layer_name]['V'], dtype=np.float32)

        # Apply RoPE
        cos_cache, sin_cache = compute_rope_cache(seq_len)
        Q_rope, K_rope = apply_rotary_pos_emb(Q_raw, K_raw, cos_cache, sin_cache)

        # Build content+position groups for all (C, P) configs
        print(f"\n    [ex {ex_idx+1}] Building content+position groups...", end="", flush=True)
        cp_groups = {}
        for C in C_VALUES:
            for P in P_VALUES:
                cp_groups[(C, P)] = build_content_position_groups(
                    K_raw, V, cos_cache, sin_cache, C, P, seed=SEED)
        print(" done", flush=True)

        # Build KMeans-RoPE groups
        print(f"    [ex {ex_idx+1}] Building KMeans-RoPE groups...", end="", flush=True)
        km_rope_labels = build_kmeans_rope_groups(K_rope, BUDGETS, seed=SEED)
        print(" done", flush=True)

        # Test queries — last NUM_QUERIES positions
        query_positions = list(range(seq_len - NUM_QUERIES, seq_len))

        for qi, qpos in enumerate(query_positions):
            q_rope = Q_rope[qpos]
            keys   = K_rope[:qpos + 1]
            vals   = V[:qpos + 1]
            n_keys = qpos + 1

            logits   = (q_rope @ keys.T) / np.sqrt(HEAD_DIM)
            full_w   = softmax(logits)
            full_out = full_w @ vals

            # ---- Baselines ----
            for budget in BUDGETS:
                b = min(budget, n_keys)

                # TopK
                idx_topk = np.argpartition(logits, -b)[-b:]
                baseline_errors['TopK'][budget].append(
                    rel_l2(softmax(logits[idx_topk]) @ vals[idx_topk], full_out))

                # Uniform
                u_idx = rng.choice(n_keys, size=b, replace=False)
                baseline_errors['Uniform'][budget].append(
                    rel_l2(softmax(logits[u_idx]) @ vals[u_idx], full_out))

                # Oracle
                out_oracle, _ = oracle_sampling(q_rope, keys, vals, logits, full_w, b)
                baseline_errors['Oracle'][budget].append(rel_l2(out_oracle, full_out))

                # KMeans-RoPE
                km_lbl = km_rope_labels[budget][:n_keys]
                out_km, _ = _kmeans_cluster_attention(q_rope, keys, vals, km_lbl, HEAD_DIM)
                baseline_errors['KMeans-RoPE'][budget].append(rel_l2(out_km, full_out))

            # ---- Content+Position configs ----
            for C in C_VALUES:
                for P in P_VALUES:
                    groups = cp_groups[(C, P)]
                    out_cp, n_used = content_position_attention(q_rope, groups, HEAD_DIM, qpos)
                    cp_errors[(C, P)].append(rel_l2(out_cp, full_out))
                    cp_budgets[(C, P)].append(n_used)

                    # Diagnostic: per-group breakdown for C=16, P=4
                    if C == 16 and P == 4:
                        causal_groups = [g for g in groups if g['max_pos'] <= qpos]
                        if len(causal_groups) > 0:
                            # Compute per-group contribution to error
                            rep_keys = np.array([g['rep_k'] for g in causal_groups], dtype=np.float64)
                            mean_vals_arr = np.array([g['mean_v'] for g in causal_groups], dtype=np.float64)
                            sizes = np.array([g['group_size'] for g in causal_groups], dtype=np.float64)
                            group_logits = (q_rope @ rep_keys.T) / np.sqrt(HEAD_DIM) + np.log(sizes + 1e-10)
                            group_weights = softmax(group_logits)

                            for gi, g in enumerate(causal_groups):
                                cc = g['content_cluster']
                                pb = g['position_bin']
                                w = group_weights[gi]
                                # Weighted error contribution: w * ||mean_v - full_out||
                                contrib = w * np.linalg.norm(mean_vals_arr[gi] - full_out)
                                diag_cluster_errors.setdefault(cc, []).append(contrib)
                                diag_bin_errors.setdefault(pb, []).append(contrib)

            progress.step(f"ex {ex_idx+1}/{len(examples_raw)} q {qi+1}/{NUM_QUERIES}")

        del Q_raw, K_raw, V, Q_rope, K_rope

    progress.finish()

    # Aggregate results
    results = {
        'budgets': BUDGETS,
        'C_values': C_VALUES,
        'P_values': P_VALUES,
    }

    # Baselines
    for method in baseline_errors:
        results[f'{method}_mean'] = [float(np.mean(baseline_errors[method][b])) for b in BUDGETS]
        results[f'{method}_std']  = [float(np.std(baseline_errors[method][b]))  for b in BUDGETS]

    # Content+Position
    cp_results = {}
    for C in C_VALUES:
        for P in P_VALUES:
            key = f"C{C}_P{P}"
            errs = cp_errors[(C, P)]
            buds = cp_budgets[(C, P)]
            cp_results[key] = {
                'C': C, 'P': P,
                'mean_error': float(np.mean(errs)),
                'std_error':  float(np.std(errs)),
                'mean_budget': float(np.mean(buds)),
                'std_budget':  float(np.std(buds)),
                'errors': [float(e) for e in errs],
                'budgets_actual': [int(b) for b in buds],
            }
    results['content_position'] = cp_results

    # Diagnostic
    results['diagnostic'] = {
        'cluster_errors': {str(k): float(np.mean(v)) for k, v in sorted(diag_cluster_errors.items())},
        'bin_errors':     {str(k): float(np.mean(v)) for k, v in sorted(diag_bin_errors.items())},
    }

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

BASELINE_STYLES = {
    'TopK':        {'color': '#d62728', 'marker': 'o',  'ls': '-',  'lw': 2.5},
    'Uniform':     {'color': '#ff7f0e', 'marker': 's',  'ls': '--', 'lw': 2.0},
    'Oracle':      {'color': '#2ca02c', 'marker': '^',  'ls': '--', 'lw': 2.5},
    'KMeans-RoPE': {'color': 'darkorange', 'marker': 'X', 'ls': '-', 'lw': 2.0},
}

# P-value color map: gray for P=1, deeper blue for higher P
P_COLORS = {1: '#999999', 2: '#9ecae1', 4: '#4292c6', 8: '#2171b5', 16: '#084594'}
C_MARKERS = {4: 'o', 8: 's', 16: 'D', 32: '^', 64: 'v'}


def _plot_baselines(ax, data, methods=None):
    """Plot baseline curves on an axis."""
    if methods is None:
        methods = list(BASELINE_STYLES.keys())
    x = np.array(data['budgets'])
    for method in methods:
        mk = f'{method}_mean'
        if mk not in data:
            continue
        means = np.array(data[mk])
        stds  = np.array(data[f'{method}_std'])
        style = BASELINE_STYLES[method]
        ax.plot(x, means, marker=style['marker'], linestyle=style['ls'],
                linewidth=style['lw'], markersize=5, color=style['color'],
                label=method, alpha=0.9, zorder=3)
        ax.fill_between(x, means, means + stds,
                        color=style['color'], alpha=0.12, zorder=1)


def _style_axis(ax, title, xlabel='Budget (num groups)', ylabel='Relative L2 Error'):
    ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')


def layer_title(layer_name):
    return 'First Layer (Layer 0)' if 'first' in layer_name else 'Last Layer (Layer 31)'


def make_fig1(all_results, output_dir, subtitle):
    """Fig 1 — Error vs Budget: baselines as lines, C+P configs as scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, layer in zip(axes, LAYERS):
        data = all_results[layer]
        _plot_baselines(ax, data)

        # Scatter content+position configs
        cp = data['content_position']
        for key, info in cp.items():
            C, P = info['C'], info['P']
            ax.scatter(info['mean_budget'], info['mean_error'],
                       color=P_COLORS.get(P, 'gray'),
                       marker=C_MARKERS.get(C, 'o'),
                       s=80, alpha=0.85, zorder=5, edgecolors='black', linewidths=0.5)

        # Legend entries for C and P
        for P, col in sorted(P_COLORS.items()):
            ax.scatter([], [], color=col, s=50, label=f'P={P}', edgecolors='black', linewidths=0.5)
        for C, mk in sorted(C_MARKERS.items()):
            ax.scatter([], [], color='gray', marker=mk, s=50, label=f'C={C}')

        _style_axis(ax, layer_title(layer))
        ax.legend(fontsize=7, framealpha=0.95, ncol=3, loc='upper right')

    fig.suptitle(f'Content+Position Grouping vs Baselines\n{subtitle}',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir / 'fig1_error_vs_budget.png', dpi=200)
    plt.close(fig)


def make_fig2(all_results, output_dir, subtitle):
    """Fig 2 — Heatmap: (C, P) grid, mean error."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, layer in zip(axes, LAYERS):
        cp = all_results[layer]['content_position']
        grid = np.full((len(P_VALUES), len(C_VALUES)), np.nan)

        for ci, C in enumerate(C_VALUES):
            for pi, P in enumerate(P_VALUES):
                key = f"C{C}_P{P}"
                if key in cp:
                    grid[pi, ci] = cp[key]['mean_error']

        im = ax.imshow(grid, cmap='RdYlGn_r', aspect='auto',
                       norm=mcolors.LogNorm(vmin=np.nanmin(grid), vmax=np.nanmax(grid)))

        # Annotate
        for ci, C in enumerate(C_VALUES):
            for pi, P in enumerate(P_VALUES):
                key = f"C{C}_P{P}"
                if key in cp:
                    err = cp[key]['mean_error']
                    bud = cp[key]['mean_budget']
                    ax.text(ci, pi, f'{err:.3f}\n(b={bud:.0f})',
                            ha='center', va='center', fontsize=7, fontweight='bold')

        ax.set_xticks(range(len(C_VALUES)))
        ax.set_xticklabels([str(c) for c in C_VALUES])
        ax.set_yticks(range(len(P_VALUES)))
        ax.set_yticklabels([str(p) for p in P_VALUES])
        ax.set_xlabel('Content Clusters (C)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Position Bins (P)', fontsize=10, fontweight='bold')
        ax.set_title(layer_title(layer), fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, label='Mean Rel. L2 Error', shrink=0.8)

    fig.suptitle(f'Content+Position Heatmap: (C, P) Grid\n{subtitle}',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir / 'fig2_heatmap.png', dpi=200)
    plt.close(fig)


def make_fig3(all_results, output_dir, subtitle):
    """Fig 3 — Value of position splitting: for each C, plot error at varying P."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    c_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(C_VALUES)))

    for ax, layer in zip(axes, LAYERS):
        cp = all_results[layer]['content_position']

        for ci, C in enumerate(C_VALUES):
            p_vals = []
            errs = []
            for P in P_VALUES:
                key = f"C{C}_P{P}"
                if key in cp:
                    p_vals.append(P)
                    errs.append(cp[key]['mean_error'])
            if p_vals:
                ax.plot(p_vals, errs, marker='o', linewidth=2, markersize=6,
                        color=c_colors[ci], label=f'C={C}')

        # Reference lines: TopK and Oracle at selected budgets
        data = all_results[layer]
        for ref, style in [('TopK', '--'), ('Oracle', ':')]:
            means = data[f'{ref}_mean']
            budgets = data['budgets']
            # Show at budget=32 and budget=128 as horizontal reference
            for bi, b in enumerate(budgets):
                if b in [32, 128]:
                    ax.axhline(means[bi], color=BASELINE_STYLES[ref]['color'],
                               linestyle=style, alpha=0.5, linewidth=1)
                    ax.text(P_VALUES[-1] * 0.95, means[bi],
                            f'{ref}@{b}', fontsize=7, ha='right', va='bottom',
                            color=BASELINE_STYLES[ref]['color'], alpha=0.7)

        ax.set_xlabel('Position Bins (P)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Mean Rel. L2 Error', fontsize=10, fontweight='bold')
        ax.set_title(layer_title(layer), fontsize=11, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xscale('log', base=2)
        ax.set_xticks(P_VALUES)
        ax.set_xticklabels([str(p) for p in P_VALUES])
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')
        ax.legend(fontsize=8, framealpha=0.95)

    fig.suptitle(f'Value of Position Splitting: Error vs P for each C\n{subtitle}',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir / 'fig3_position_splitting_value.png', dpi=200)
    plt.close(fig)


def make_fig4(all_results, output_dir, subtitle):
    """Fig 4 — Content+Position vs KMeans-RoPE head-to-head scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, layer in zip(axes, LAYERS):
        data = all_results[layer]
        cp = data['content_position']
        km_means = dict(zip(data['budgets'], data['KMeans-RoPE_mean']))

        for key, info in cp.items():
            C, P = info['C'], info['P']
            cp_error = info['mean_error']
            cp_budget = info['mean_budget']

            # Find closest KMeans-RoPE budget
            closest_b = min(BUDGETS, key=lambda b: abs(b - cp_budget))
            km_error = km_means[closest_b]

            ax.scatter(km_error, cp_error,
                       color=P_COLORS.get(P, 'gray'),
                       marker=C_MARKERS.get(C, 'o'),
                       s=80, alpha=0.85, zorder=5, edgecolors='black', linewidths=0.5)

        # Diagonal line
        all_vals = []
        for info in cp.values():
            all_vals.append(info['mean_error'])
            closest_b = min(BUDGETS, key=lambda b: abs(b - info['mean_budget']))
            all_vals.append(km_means[closest_b])
        if all_vals:
            lo, hi = min(all_vals) * 0.8, max(all_vals) * 1.2
            ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, linewidth=1, label='y=x')

        # Legend
        for P, col in sorted(P_COLORS.items()):
            ax.scatter([], [], color=col, s=50, label=f'P={P}', edgecolors='black', linewidths=0.5)
        for C, mk in sorted(C_MARKERS.items()):
            ax.scatter([], [], color='gray', marker=mk, s=50, label=f'C={C}')

        ax.set_xlabel('KMeans-RoPE Error', fontsize=10, fontweight='bold')
        ax.set_ylabel('Content+Position Error', fontsize=10, fontweight='bold')
        ax.set_title(layer_title(layer), fontsize=11, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')
        ax.legend(fontsize=7, framealpha=0.95, ncol=2)

    fig.suptitle(f'Content+Position vs KMeans-RoPE (below diagonal = C+P wins)\n{subtitle}',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir / 'fig4_head_to_head.png', dpi=200)
    plt.close(fig)


def make_fig5(all_results, output_dir, subtitle):
    """Fig 5 — Per-group error breakdown for C=16, P=4 diagnostic."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for col, layer in enumerate(LAYERS):
        diag = all_results[layer].get('diagnostic', {})
        cluster_errs = diag.get('cluster_errors', {})
        bin_errs = diag.get('bin_errors', {})

        # Top row: by content cluster
        ax = axes[0, col]
        if cluster_errs:
            clusters = sorted(cluster_errs.keys(), key=int)
            vals = [cluster_errs[c] for c in clusters]
            ax.bar(range(len(clusters)), vals, color='steelblue', alpha=0.8)
            ax.set_xlabel('Content Cluster', fontsize=10)
            ax.set_ylabel('Mean Weighted Error', fontsize=10)
            ax.set_title(f'{layer_title(layer)} — By Content Cluster', fontsize=10, fontweight='bold')
            ax.set_xticks(range(0, len(clusters), max(1, len(clusters) // 10)))
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{layer_title(layer)} — By Content Cluster', fontsize=10)

        # Bottom row: by position bin
        ax = axes[1, col]
        if bin_errs:
            bins = sorted(bin_errs.keys(), key=int)
            vals = [bin_errs[b] for b in bins]
            ax.bar(range(len(bins)), vals, color='coral', alpha=0.8)
            ax.set_xlabel('Position Bin', fontsize=10)
            ax.set_ylabel('Mean Weighted Error', fontsize=10)
            ax.set_title(f'{layer_title(layer)} — By Position Bin', fontsize=10, fontweight='bold')
            ax.set_xticks(range(len(bins)))
            ax.set_xticklabels(bins)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{layer_title(layer)} — By Position Bin', fontsize=10)

    fig.suptitle(f'Per-Group Error Breakdown (C=16, P=4)\n{subtitle}',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir / 'fig5_diagnostic_breakdown.png', dpi=200)
    plt.close(fig)


def make_figures(all_results, output_dir, num_examples, num_queries):
    subtitle = (f'{num_examples} examples, {num_queries} queries each  |  '
                f'Llama-3-8B pre-RoPE keys + RoPE reconstruction  |  seed={SEED}')

    make_fig1(all_results, output_dir, subtitle)
    make_fig2(all_results, output_dir, subtitle)
    make_fig3(all_results, output_dir, subtitle)
    make_fig4(all_results, output_dir, subtitle)
    make_fig5(all_results, output_dir, subtitle)


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
    print("CONTENT + POSITION GROUPING EXPERIMENT")
    print("=" * 60)
    print(f"Data:       {DATA_PATH} (pre-RoPE keys, RoPE applied on-the-fly)")
    print(f"Config:     {NUM_EXAMPLES} examples x last {NUM_QUERIES} query positions")
    print(f"C values:   {C_VALUES}")
    print(f"P values:   {P_VALUES}")
    print(f"Baselines:  TopK, Uniform, Oracle, KMeans-RoPE  (budgets {BUDGETS})")
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
            'C_values':     C_VALUES,
            'P_values':     P_VALUES,
            'seed':         SEED,
            'head_dim':     HEAD_DIM,
            'data_file':    DATA_PATH,
            'rope_theta':   ROPE_THETA,
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
