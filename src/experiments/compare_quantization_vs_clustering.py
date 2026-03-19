#!/usr/bin/env python3
"""
Quantization vs Incremental KMeans Clustering

Compares the attention approximation error from KV-cache integer quantization
against incremental KMeans grouping at various budgets.

Background: the source vectors are extracted from Llama-3-8B which uses bfloat16
internally, so float32 and float16 quantization give ~0 error. The meaningful
precision steps are int8, int4, int2 (symmetric per-vector quantization), which
give ~2%, ~36%, ~180% mean relative error respectively.

Quantization baselines (horizontal lines — full attention, quantized K and V):
  - int8  — 8-bit symmetric per-vector quantization  (~2%  error)
  - int4  — 4-bit symmetric per-vector quantization  (~36% error)
  - int2  — 2-bit symmetric per-vector quantization  (~180% error)

Clustering curves (budget sweep, incremental causal grouping):
  - Oracle        — privileged upper bound
  - Uniform       — causal random sampling
  - KMeans (incr) — KMeans on float32 keys, frozen-group incremental attention

Ground truth: full attention computed in float64 (original JSON precision).

The key question: at what budget does KMeans clustering achieve lower error
than int8/int4 quantization? Crossing points are highlighted in the plots.

Usage:
  python compare_quantization_vs_clustering.py              # run compute + plot
  python compare_quantization_vs_clustering.py --plot-only  # regenerate plots from JSON
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

# ============================================================================
# CONFIG
# ============================================================================
DATA_PATH = '../../data/attention_vectors_long_bench_llama_8b.jsonl'
OUTPUT_DIR = Path('../../results/quantization_vs_clustering')
NUM_EXAMPLES = 10
NUM_TEST_QUERIES = 30
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
BUDGETS = [2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512]

REGIONS = ['first', 'middle', 'last']
REGION_DISPLAY = {
    'first':  'First 30 Queries (Early Positions)',
    'middle': 'Middle 30 Queries (Center Positions)',
    'last':   'Last 30 Queries (Late Positions)',
}

CURVE_METHODS = ['Oracle', 'Uniform', 'KMeans Keys (incr)']

# Quantization methods: (display_name, method_key)
# method_key is either an int (bits for symmetric INT quant) or a string for special methods
QUANT_VARIANTS = [
    ('int8',   8),
    ('nvfp4',  'nvfp4'),  # NVIDIA FP4 E2M1, block-16 FP8 E4M3 scales
    ('int4',   4),
    ('int2',   2),
]
QUANT_METHODS = [name for name, _ in QUANT_VARIANTS]
QUANT_PLOT_METHODS = ['int8', 'nvfp4', 'int4']  # int2 excluded (ternary, not true 2-bit)

CURVE_COLORS = {
    'Oracle':              '#2ca02c',
    'Uniform':             '#7fbf7f',
    'KMeans Keys (incr)':  'darkorange',
}

CURVE_MARKERS = {
    'Oracle':              '^',
    'Uniform':             's',
    'KMeans Keys (incr)':  'X',
}

QUANT_COLORS = {
    'int8':   '#08519c',  # dark blue
    'nvfp4':  '#2171b5',  # medium blue
    'int4':   '#6baed6',  # light blue
    'int2':   '#d62728',
}

QUANT_LINESTYLES = {
    'int8':   '--',
    'nvfp4':  (0, (5, 2)),  # densely dashed
    'int4':   '-.',
    'int2':   ':',
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
# QUANTIZATION
# ============================================================================

def quantize_symmetric(x, bits):
    """
    Symmetric per-vector integer quantization.
    x: [N, D] float64 array
    Returns: [N, D] float64 dequantized array
    """
    max_val = 2 ** (bits - 1) - 1  # 127 for int8, 7 for int4, 1 for int2
    scale = np.max(np.abs(x), axis=1, keepdims=True) / max_val
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.round(x / scale), -max_val, max_val)
    return q * scale  # stays float64


# ---------------------------------------------------------------------------
# NVFP4 (E2M1) — NVIDIA Blackwell two-level block quantization
# ---------------------------------------------------------------------------
# FP4 E2M1 representable magnitudes: 1 sign bit, 2 exponent bits, 1 mantissa bit
# Positive values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
_FP4_POS    = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
_FP4_MIDS   = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],  dtype=np.float64)
_FP4_MAX    = 6.0
_FP8_E4M3_MAX = 240.0  # E4M3, bias=7: max normal = 2^7 × 1.875 = 240


def _nearest_fp4(abs_x):
    """Map non-negative values to nearest FP4 E2M1 magnitude (vectorized)."""
    idx = np.searchsorted(_FP4_MIDS, abs_x)   # returns index in [0, 7]
    return _FP4_POS[idx]


def _quantize_fp8_e4m3(x):
    """
    Round array to nearest FP8 E4M3 representable value.
    E4M3: 1 sign, 4 exponent, 3 mantissa bits, bias=7.
    Rounds mantissa to 3 bits; valid normal range: 2^-6 to 240.
    """
    x = np.clip(x, -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    out = np.zeros_like(x)
    nz = x != 0
    if not nz.any():
        return out
    v      = x[nz]
    sign   = np.sign(v)
    abs_v  = np.abs(v)
    exp    = np.floor(np.log2(abs_v))
    exp    = np.clip(exp, -6.0, 7.0)          # valid unbiased E4M3 exponent range
    scale  = np.exp2(exp)
    mant   = abs_v / scale                     # in [1, 2)
    mant_q = np.round(mant * 8.0) / 8.0       # round to 3 mantissa bits
    # handle rare rounding-up overflow
    over         = mant_q >= 2.0
    exp[over]    = np.clip(exp[over] + 1, -6.0, 7.0)
    mant_q[over] = 1.0
    out[nz] = sign * mant_q * np.exp2(exp)
    return out


def quantize_nvfp4(x, block_size=16):
    """
    NVFP4 two-level block quantization (NVIDIA Blackwell).
      1. Divide each vector into blocks of `block_size` (16) elements.
      2. Per-block FP8 E4M3 scale = max(|block|) / FP4_MAX (6.0).
      3. Normalize block by FP8 scale, then round to nearest FP4 E2M1 value.
      4. Dequantize: fp4_val × fp8_scale.
    x: [N, D] float64 — D must be divisible by block_size (128 / 16 = 8 blocks).
    Returns: [N, D] float64 dequantized array.
    """
    N, D = x.shape
    assert D % block_size == 0, f"D={D} must be divisible by block_size={block_size}"
    n_blocks = D // block_size
    xb = x.reshape(N, n_blocks, block_size)                      # [N, B, 16]

    # Per-block scale in FP32, then quantized to FP8 E4M3
    block_max  = np.max(np.abs(xb), axis=2)                      # [N, B]
    scale_f32  = np.where(block_max == 0, 1.0, block_max / _FP4_MAX)
    scale_fp8  = _quantize_fp8_e4m3(scale_f32)                   # [N, B]
    scale_fp8  = np.where(scale_fp8 == 0, 1.0, scale_fp8)

    # Normalize and round to nearest FP4 value
    x_norm = xb / scale_fp8[:, :, np.newaxis]                    # [N, B, 16]
    sign   = np.where(x_norm >= 0, 1.0, -1.0)
    q      = sign * _nearest_fp4(np.abs(x_norm))                 # [N, B, 16]

    # Dequantize
    return (q * scale_fp8[:, :, np.newaxis]).reshape(N, D)


def quantize_kv(K, V, method):
    """Dispatch to the appropriate quantization scheme."""
    if method == 'nvfp4':
        return quantize_nvfp4(K), quantize_nvfp4(V)
    else:
        return quantize_symmetric(K, method), quantize_symmetric(V, method)


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
# CLUSTER ASSIGNMENT
# ============================================================================

def assign_kmeans_keys(K_mat, seq_len):
    """KMeans on float32 keys, return {budget: labels[seq_len]}."""
    K_f32 = K_mat.astype(np.float32)
    assignments = {}
    for budget in BUDGETS:
        b = min(budget, seq_len)
        if b >= seq_len:
            assignments[budget] = np.arange(seq_len, dtype=np.int32)
        else:
            km = KMeans(n_clusters=b, n_init=3, max_iter=100, random_state=SEED)
            km.fit(K_f32[:seq_len])
            assignments[budget] = km.labels_.astype(np.int32)
    return assignments


# ============================================================================
# INCREMENTAL EVALUATION
# ============================================================================

def run_incremental_multi_budget(Q, K_mat, V, seq_len, labels_by_budget,
                                  test_positions_by_region):
    """Single causal walk maintaining one IncrementalClusters per budget."""
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
                for budget, (state, _labels) in states.items():
                    mk, mv, gs = state.get_representatives()
                    if mk is not None:
                        out = frozen_group_attention(Q[pos], mk, mv, gs)
                    else:
                        out = np.zeros(HEAD_DIM, dtype=np.float64)
                    outputs[budget][region].append((pos, out))

    return outputs


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
    quant_errors = {
        region: {m: [] for m in QUANT_METHODS}
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
            # Load as float64 — this is the original bfloat16 model precision
            Q     = np.array(example[layer_name]['Q'], dtype=np.float64)
            K_mat = np.array(example[layer_name]['K'], dtype=np.float64)
            V     = np.array(example[layer_name]['V'], dtype=np.float64)
            seq_len = Q.shape[0]

            region_positions = select_query_positions(seq_len, NUM_TEST_QUERIES)
            all_test_positions = set()
            for r in REGIONS:
                all_test_positions.update(region_positions[r])

            ex_count += 1
            print(f"\n  [{ex_count}/{total_examples}] Example {idx}: seq_len={seq_len}")

            # ---- Ground truth (float64) ----
            print(f"    Computing ground truth (float64)...")
            t0 = time.time()
            gt_data = {}
            for qpos in all_test_positions:
                q    = Q[qpos]
                keys = K_mat[:qpos + 1]
                vals = V[:qpos + 1]
                logits  = (q @ keys.T) / np.sqrt(HEAD_DIM)
                full_w  = softmax(logits)
                full_out = full_w @ vals
                gt_data[qpos] = (full_out, logits, full_w)
            print(f"    GT done in {time.time()-t0:.1f}s")

            # ---- Quantization errors (full attention with quantized K, V) ----
            print(f"    Quantization errors ({', '.join(QUANT_METHODS)})...")
            t0 = time.time()
            for quant_name, bits in QUANT_VARIANTS:
                K_q, V_q = quantize_kv(K_mat, V, bits)
                for region in REGIONS:
                    for qpos in region_positions[region]:
                        q       = Q[qpos]
                        keys_q  = K_q[:qpos + 1]
                        vals_q  = V_q[:qpos + 1]
                        logits_q = (q @ keys_q.T) / np.sqrt(HEAD_DIM)
                        out_q    = softmax(logits_q) @ vals_q
                        full_out = gt_data[qpos][0]
                        quant_errors[region][quant_name].append(
                            rel_l2(out_q, full_out)
                        )
            print(f"    Quantization done in {time.time()-t0:.1f}s")

            # ---- Oracle + Uniform baselines ----
            print(f"    Oracle + Uniform...")
            t0 = time.time()
            baseline_rng = np.random.default_rng(master_rng.integers(2**32))
            for region in REGIONS:
                for qpos in region_positions[region]:
                    full_out, logits, full_w = gt_data[qpos]
                    q    = Q[qpos]
                    keys = K_mat[:qpos + 1]
                    vals = V[:qpos + 1]
                    n_keys = qpos + 1
                    for budget in BUDGETS:
                        b = min(budget, n_keys)

                        out_oracle, _ = oracle_sampling(q, keys, vals, logits, full_w, b)
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
            # Fit KMeans on float32 keys (realistic scenario), but accumulate
            # incremental state in float64 for consistency with ground truth.
            print(f"    KMeans assignments (float32)...")
            t0 = time.time()
            kmeans_assignments = assign_kmeans_keys(K_mat, seq_len)
            print(f"    KMeans done in {time.time()-t0:.1f}s")

            print(f"    KMeans incremental eval...")
            t0 = time.time()
            K_f32 = K_mat.astype(np.float32).astype(np.float64)
            V_f32 = V.astype(np.float32).astype(np.float64)
            km_outputs = run_incremental_multi_budget(
                Q, K_f32, V_f32, seq_len, kmeans_assignments, region_positions
            )
            for budget in BUDGETS:
                for region in REGIONS:
                    for qpos, out in km_outputs[budget][region]:
                        full_out = gt_data[qpos][0]
                        curve_errors[region]['KMeans Keys (incr)'][budget].append(
                            rel_l2(out, full_out)
                        )
            print(f"    KMeans eval done in {time.time()-t0:.1f}s")

            del Q, K_mat, V, gt_data, K_f32, V_f32

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

    # Aggregate quantization
    results['quant'] = {}
    for quant_name, _bits in QUANT_VARIANTS:
        results['quant'][quant_name] = {}
        for region in REGIONS:
            errs = quant_errors[region][quant_name]
            results['quant'][quant_name][f'{region}_mean'] = (
                float(np.mean(errs)) if errs else 0.0
            )
            results['quant'][quant_name][f'{region}_std'] = (
                float(np.std(errs)) if errs else 0.0
            )

    return results


# ============================================================================
# PLOTTING
# ============================================================================

def _plot_panel(ax, data, region):
    """One region panel: clustering curves + quantization horizontal lines."""
    x = np.array(data['budgets'])
    region_data = data[region]
    quant = data.get('quant', {})

    # Clustering curves
    for method in CURVE_METHODS:
        means = np.array(region_data[f'{method}_mean'])
        stds  = np.array(region_data[f'{method}_std'])
        color  = CURVE_COLORS[method]
        marker = CURVE_MARKERS[method]
        ax.plot(x, means, marker=marker, color=color, lw=2.5,
                label=method, zorder=4, markersize=6)
        ax.fill_between(x, means, means + stds, color=color, alpha=0.12)

    # Quantization horizontal lines
    for quant_name in QUANT_PLOT_METHODS:
        if quant_name not in quant:
            continue
        mean_err = quant[quant_name].get(f'{region}_mean', 0.0)
        std_err  = quant[quant_name].get(f'{region}_std', 0.0)
        color = QUANT_COLORS[quant_name]
        ls    = QUANT_LINESTYLES[quant_name]
        ax.axhline(mean_err, color=color, lw=2.5, ls=ls, zorder=3,
                   label=f'{quant_name} quant  ({mean_err:.2e})')
        ax.axhspan(max(mean_err - std_err, 1e-10), mean_err + std_err,
                   color=color, alpha=0.07)

    ax.set_title(REGION_DISPLAY[region], fontsize=12, fontweight='bold')
    ax.set_xlabel('Budget (num groups)', fontsize=10)
    ax.set_ylabel('Relative L2 Error', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(left=0, right=512)
    ax.grid(True, alpha=0.3, ls='--', which='both')


def _plot_method_panel(ax, data, method):
    """Per-method panel: all three regions + quantization lines."""
    x = np.array(data['budgets'])
    quant = data.get('quant', {})

    region_colors  = {'first': '#e41a1c', 'middle': '#377eb8', 'last': '#4daf4a'}
    region_markers = {'first': 'o', 'middle': 's', 'last': '^'}

    for region in REGIONS:
        region_data = data[region]
        means = np.array(region_data[f'{method}_mean'])
        stds  = np.array(region_data[f'{method}_std'])
        color  = region_colors[region]
        marker = region_markers[region]
        ax.plot(x, means, marker=marker, color=color, lw=2.5,
                label=f'{region.capitalize()} queries', zorder=4, markersize=6)
        ax.fill_between(x, means, means + stds, color=color, alpha=0.12)

    # Quantization lines (use 'last' region as representative; they're near-identical)
    for quant_name in QUANT_PLOT_METHODS:
        if quant_name not in quant:
            continue
        # Average mean across regions for the per-method panel
        mean_err = float(np.mean([
            quant[quant_name].get(f'{r}_mean', 0.0) for r in REGIONS
        ]))
        color = QUANT_COLORS[quant_name]
        ls    = QUANT_LINESTYLES[quant_name]
        ax.axhline(mean_err, color=color, lw=2.0, ls=ls, zorder=3,
                   label=f'{quant_name} quant ({mean_err:.2e})', alpha=0.85)

    ax.set_title(method, fontsize=12, fontweight='bold')
    ax.set_xlabel('Budget', fontsize=10)
    ax.set_ylabel('Relative L2 Error', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(left=0, right=512)
    ax.grid(True, alpha=0.3, ls='--', which='both')


def make_figures(all_results, output_dir):
    cfg = all_results.get('config', {})
    n_ex = cfg.get('num_examples', NUM_EXAMPLES)
    n_q  = cfg.get('num_test_queries', NUM_TEST_QUERIES)
    subtitle = (
        f'{n_ex} examples, {n_q} queries each  |  '
        f'Llama-3-8B  |  GT=float64  |  '
        f'Quant=symmetric per-vector  |  Shaded = +1 std'
    )

    for layer in LAYERS:
        layer_data  = all_results[layer]
        layer_short = 'first_layer' if 'first' in layer else 'last_layer'
        layer_title = ('First Layer (Layer 0)' if 'first' in layer
                       else 'Last Layer (Layer 31)')

        # ---- Figure 1: by_region (3 panels) ----
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
        for i, region in enumerate(REGIONS):
            _plot_panel(axes[i], layer_data, region)
            if i > 0:
                axes[i].set_ylabel('')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', fontsize=9,
                   framealpha=0.95, ncol=3, bbox_to_anchor=(0.5, -0.06))
        fig.suptitle(
            f'KV Quantization vs KMeans Clustering — {layer_title}\n{subtitle}',
            fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.10, 1, 0.94])
        save_figure(fig, output_dir / f'by_region_{layer_short}.png', dpi=200)
        plt.close(fig)

        # ---- Figure 2: per_method (3 panels — one per clustering method) ----
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
        for i, method in enumerate(CURVE_METHODS):
            _plot_method_panel(axes[i], layer_data, method)
            axes[i].legend(fontsize=9, framealpha=0.95)
            if i > 0:
                axes[i].set_ylabel('')

        fig.suptitle(
            f'Per-Method Region Comparison — {layer_title}\n{subtitle}',
            fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        save_figure(fig, output_dir / f'per_method_{layer_short}.png', dpi=200)
        plt.close(fig)

        # ---- Print quantization summary ----
        print(f"\n  Quantization error summary ({layer_title}):")
        print(f"  {'Method':6s}  {'first':>10s}  {'middle':>10s}  {'last':>10s}")
        quant = layer_data.get('quant', {})
        for quant_name in QUANT_METHODS:
            if quant_name in quant:
                q = quant[quant_name]
                print(f"  {quant_name:6s}  "
                      f"{q.get('first_mean', 0):.4e}  "
                      f"{q.get('middle_mean', 0):.4e}  "
                      f"{q.get('last_mean', 0):.4e}")


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
    print("QUANTIZATION vs KMeans CLUSTERING")
    print("=" * 60)
    print(f"Config: {NUM_EXAMPLES} examples, {NUM_TEST_QUERIES} queries/region")
    print(f"Quant methods: {QUANT_METHODS}  (symmetric per-vector)")
    print(f"Clustering curves: {CURVE_METHODS}")
    print(f"Budgets: {BUDGETS}")
    print(f"Ground truth: float64 (source is Llama-3-8B bfloat16)")
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
            'budgets': BUDGETS,
            'seed': SEED,
            'head_dim': HEAD_DIM,
            'curve_methods': CURVE_METHODS,
            'quant_methods': QUANT_METHODS,
            'quant_scheme': 'symmetric per-vector',
            'regions': REGIONS,
            'ground_truth': 'float64',
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
