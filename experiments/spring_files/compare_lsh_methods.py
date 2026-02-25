#!/usr/bin/env python3
"""
LSH Method Comparison: SimHash SNIS vs Cross-Polytope SNIS
with multi-hit thresholds (MagicPIG-style min_hits=1,2,3).

Baselines (budget-controlled):
  1. Top-K (biased)
  2. Uniform Sampling (biased)
  3. Oracle Sampling (unbiased, privileged)

LSH-SNIS (variable budget, sweep K, L, min_hits):
  4. SimHash SNIS with min_hits in {1, 2, 3}
  5. Cross-Polytope SNIS with min_hits in {1, 2, 3}

Optimization: per-table match vectors computed once per hash depth,
then cumulative-summed for instant L slicing and min_hits thresholding.

Output: timestamped subfolder with JSON + plots.
"""

import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy.special import comb as scipy_comb

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = '../../data/attention_vectors_updated_long.jsonl'
BASE_OUTPUT_DIR = Path('../../results/spring_comparison')
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
NUM_EXAMPLES = 50
NUM_QUERIES = 100

# Budget percentages for baselines (no 100 — drops to 0 trivially)
K_PERCENTAGES = [3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 95]

# Min-hits variants (MagicPIG uses 2)
MIN_HITS_VALUES = [1, 2, 3]

# SimHash parameter grid — lower K and higher L to compensate for stricter min_hits
SIMHASH_K_VALUES = [2, 3, 4, 5, 6, 7, 8, 10]
SIMHASH_L_VALUES = [5, 10, 15, 25, 40, 50, 75]

# Cross-Polytope parameter grid
# k=1 is the useful range; k=2 only with high L and low min_hits
CP_K_VALUES = [1, 2]
CP_L_VALUES_K1 = [5, 10, 25, 50, 75, 100, 150, 200]   # k=1: sweep broadly
CP_L_VALUES_K2 = [50, 100, 150, 200]                    # k=2: only high L

# Derived maxima for shared index
MAX_SH_K = max(SIMHASH_K_VALUES)
MAX_SH_L = max(SIMHASH_L_VALUES)
MAX_CP_K = max(CP_K_VALUES)
MAX_CP_L = max(max(CP_L_VALUES_K1), max(CP_L_VALUES_K2))

# ============================================================================
# SETUP
# ============================================================================
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{TIMESTAMP}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def snis_estimator(logits, values, inclusion_probs):
    if len(logits) == 0:
        return np.zeros(HEAD_DIM)
    weighted_logits = logits - np.log(np.clip(inclusion_probs, 1e-12, None))
    weights = softmax(weighted_logits)
    return weights @ values


def inclusion_prob(p_table, L, min_hits):
    """
    P(key collides in >= min_hits out of L tables).
    Uses exact binomial CDF complement.
    """
    if min_hits == 1:
        return 1.0 - np.power(1.0 - p_table, L)
    elif min_hits == 2:
        p0 = np.power(1.0 - p_table, L)
        p1 = L * p_table * np.power(1.0 - p_table, L - 1)
        return 1.0 - p0 - p1
    elif min_hits == 3:
        p0 = np.power(1.0 - p_table, L)
        p1 = L * p_table * np.power(1.0 - p_table, L - 1)
        p2 = (L * (L - 1) / 2.0) * np.power(p_table, 2) * np.power(1.0 - p_table, L - 2)
        return 1.0 - p0 - p1 - p2
    else:
        raise ValueError(f"min_hits={min_hits} not supported")


# ============================================================================
# BASELINE METHODS
# ============================================================================

def compute_topk(logits, values, k):
    n = len(logits)
    k = min(k, n)
    idx = np.argpartition(logits, -k)[-k:]
    w = softmax(logits[idx])
    return w @ values[idx]


def compute_uniform(logits, values, budget, rng):
    n = len(logits)
    budget = min(budget, n)
    idx = rng.choice(n, size=budget, replace=False)
    w = softmax(logits[idx])
    return w @ values[idx]


def compute_oracle(logits, values, weights, budget, rng):
    n = len(logits)
    budget = min(budget, n)
    idx = rng.choice(n, size=budget, p=weights, replace=True)
    return np.mean(values[idx], axis=0)


# ============================================================================
# SIMHASH LSH (shared max-size structure)
# ============================================================================

class SimHashIndex:
    def __init__(self, num_tables, max_depth, head_dim, center_keys=True, seed=None):
        self.num_tables = num_tables
        self.max_depth = max_depth
        self.head_dim = head_dim
        self.center_keys = center_keys
        rng = np.random.RandomState(seed)
        hp = rng.randn(num_tables, max_depth, head_dim).astype(np.float32)
        self.hyperplanes = hp / np.linalg.norm(hp, axis=2, keepdims=True)
        self.key_mean = None
        self.key_codes = None

    def build_index(self, keys):
        if self.center_keys:
            self.key_mean = np.mean(keys, axis=0)
            c = keys - self.key_mean
        else:
            self.key_mean = np.zeros(self.head_dim, dtype=np.float32)
            c = keys
        self.key_codes = (np.einsum('nd,ltd->nlt', c, self.hyperplanes) > 0).astype(np.int8)

    def batch_hash_queries(self, Q):
        """Hash all queries at once. Returns [num_queries, L, max_depth]."""
        c = Q - self.key_mean if self.center_keys else Q
        return (np.einsum('qd,ltd->qlt', c, self.hyperplanes) > 0).astype(np.int8)

    @staticmethod
    def collision_prob(thetas, depth_k):
        p_bit = 1.0 - thetas / np.pi
        return np.power(np.clip(p_bit, 0, 1), depth_k)


# ============================================================================
# CROSS-POLYTOPE LSH (shared max-size structure)
# ============================================================================

class CrossPolytopeIndex:
    def __init__(self, num_tables, max_cp, head_dim, center_keys=True, seed=None):
        self.num_tables = num_tables
        self.max_cp = max_cp
        self.head_dim = head_dim
        self.center_keys = center_keys
        rng = np.random.RandomState(seed)
        self.rotations = rng.randn(num_tables, max_cp, head_dim, head_dim).astype(np.float32)
        self.key_mean = None
        self.key_codes = None

    def _hash_batch(self, vectors):
        N = vectors.shape[0]
        codes = np.zeros((N, self.num_tables, self.max_cp), dtype=np.int32)
        for l in range(self.num_tables):
            for cp in range(self.max_cp):
                rot = vectors @ self.rotations[l, cp].T
                norms = np.linalg.norm(rot, axis=1, keepdims=True)
                rot = rot / (norms + 1e-10)
                max_j = np.argmax(np.abs(rot), axis=1)
                signs = rot[np.arange(N), max_j]
                codes[:, l, cp] = 2 * max_j + (signs < 0).astype(np.int32)
        return codes

    def build_index(self, keys):
        if self.center_keys:
            self.key_mean = np.mean(keys, axis=0)
            c = keys - self.key_mean
        else:
            self.key_mean = np.zeros(self.head_dim, dtype=np.float32)
            c = keys
        self.key_codes = self._hash_batch(c)

    def batch_hash_queries(self, Q):
        c = Q - self.key_mean if self.center_keys else Q
        return self._hash_batch(c)

    def collision_prob_single(self, thetas):
        d = self.head_dim
        tau_sq = 2.0 * (1.0 - np.cos(thetas))
        denom = np.clip(4.0 - tau_sq, 1e-10, None)
        return np.power(float(d), -(tau_sq / denom))

    def collision_prob(self, thetas, k_cp):
        return np.power(self.collision_prob_single(thetas), k_cp)


# ============================================================================
# FAST LSH-SNIS: compute per-table matches once, reuse for all (L, min_hits)
# ============================================================================

def do_snis(query, keys, values, logits, retrieved_idx, collision_prob_fn,
            depth_k, L_use, mh):
    """Run SNIS on a retrieved set. Returns (output, n_retrieved)."""
    n = len(retrieved_idx)
    if n == 0:
        return np.zeros(HEAD_DIM), 0
    r_keys = keys[retrieved_idx]
    r_values = values[retrieved_idx]
    r_logits = logits[retrieved_idx]
    q_norm = np.linalg.norm(query)
    k_norms = np.linalg.norm(r_keys, axis=1)
    cos_sims = np.clip(
        (r_keys @ query) / (q_norm * k_norms + 1e-8), -1.0 + 1e-8, 1.0 - 1e-8)
    thetas = np.arccos(cos_sims)
    p_table = collision_prob_fn(thetas, depth_k)
    inc = np.clip(inclusion_prob(p_table, L_use, mh), 1e-12, 1.0)
    return snis_estimator(r_logits, r_values, inc), n


# ============================================================================
# PER-EXAMPLE ANALYSIS (optimized inner loop)
# ============================================================================

def build_config_list():
    """Build the list of all LSH configs to evaluate."""
    configs = []
    for K in SIMHASH_K_VALUES:
        for L in SIMHASH_L_VALUES:
            for mh in MIN_HITS_VALUES:
                if mh > L:
                    continue
                configs.append({
                    'type': 'simhash', 'K': K, 'L': L, 'min_hits': mh,
                    'key': f"simhash_K{K}_L{L}_h{mh}"
                })
    for k_cp in CP_K_VALUES:
        L_vals = CP_L_VALUES_K1 if k_cp == 1 else CP_L_VALUES_K2
        for L in L_vals:
            for mh in MIN_HITS_VALUES:
                if mh > L:
                    continue
                configs.append({
                    'type': 'crosspolytope', 'k_cp': k_cp, 'L': L, 'min_hits': mh,
                    'key': f"cp_k{k_cp}_L{L}_h{mh}"
                })
    return configs

ALL_CONFIGS = build_config_list()


def analyze_example(example, layer_name, sh_index, cp_index, rng):
    Q = np.array(example[layer_name]['Q'], dtype=np.float32)
    K_mat = np.array(example[layer_name]['K'], dtype=np.float32)
    V = np.array(example[layer_name]['V'], dtype=np.float32)
    seq_len = Q.shape[0]
    query_positions = list(range(max(0, seq_len - NUM_QUERIES), seq_len))

    # Build indexes once
    sh_index.build_index(K_mat)
    cp_index.build_index(K_mat)

    # Pre-compute all logits and batch-hash all queries
    all_logits = (Q @ K_mat.T) / np.sqrt(HEAD_DIM)  # [seq, seq]
    sh_qhashes = sh_index.batch_hash_queries(Q)       # [seq, L_sh, K_sh]
    cp_qhashes = cp_index.batch_hash_queries(Q)       # [seq, L_cp, k_cp]

    # Init result containers
    results = {
        'baselines': {m: {str(k): [] for k in K_PERCENTAGES}
                      for m in ['topk', 'uniform', 'oracle']},
        'lsh': {c['key']: {'budgets': [], 'errors': []} for c in ALL_CONFIGS}
    }

    for qpos in query_positions:
        nv = qpos + 1
        logits = all_logits[qpos, :nv]
        valid_keys = K_mat[:nv]
        valid_values = V[:nv]
        full_weights = softmax(logits)
        full_output = full_weights @ valid_values
        out_norm = np.linalg.norm(full_output) + 1e-8

        # --- Baselines ---
        for k_pct in K_PERCENTAGES:
            k_abs = max(1, min(int(np.ceil(nv * k_pct / 100)), nv))
            topk_out = compute_topk(logits, valid_values, k_abs)
            results['baselines']['topk'][str(k_pct)].append(
                float(np.linalg.norm(topk_out - full_output) / out_norm))
            uni_out = compute_uniform(logits, valid_values, k_abs, rng)
            results['baselines']['uniform'][str(k_pct)].append(
                float(np.linalg.norm(uni_out - full_output) / out_norm))
            oracle_out = compute_oracle(logits, valid_values, full_weights, k_abs, rng)
            results['baselines']['oracle'][str(k_pct)].append(
                float(np.linalg.norm(oracle_out - full_output) / out_norm))

        # --- SimHash: compute per-table matches once per K, reuse ---
        q_sh = sh_qhashes[qpos]  # [L_max, K_max]
        for K_sh in SIMHASH_K_VALUES:
            # Per-table match for this K across all L_max tables
            kp = sh_index.key_codes[:nv, :, :K_sh]       # [nv, L_max, K_sh]
            qp = q_sh[:, :K_sh]                            # [L_max, K_sh]
            per_table = np.all(kp == qp[np.newaxis, :, :], axis=2)  # [nv, L_max]
            # Cumulative sum for fast L slicing
            cum = np.cumsum(per_table, axis=1)  # [nv, L_max]

            for L_sh in SIMHASH_L_VALUES:
                match_counts = cum[:, L_sh - 1]  # [nv]
                for mh in MIN_HITS_VALUES:
                    if mh > L_sh:
                        continue
                    key = f"simhash_K{K_sh}_L{L_sh}_h{mh}"
                    r_idx = np.where(match_counts >= mh)[0]
                    out, nr = do_snis(Q[qpos], valid_keys, valid_values, logits,
                                      r_idx, SimHashIndex.collision_prob, K_sh, L_sh, mh)
                    err = float(np.linalg.norm(out - full_output) / out_norm) if nr > 0 else float('nan')
                    results['lsh'][key]['errors'].append(err)
                    results['lsh'][key]['budgets'].append(nr)

        # --- CrossPolytope: same strategy ---
        q_cp = cp_qhashes[qpos]  # [L_max, k_max]
        for k_cp in CP_K_VALUES:
            kp = cp_index.key_codes[:nv, :, :k_cp]
            qp = q_cp[:, :k_cp]
            per_table = np.all(kp == qp[np.newaxis, :, :], axis=2)  # [nv, L_max]
            cum = np.cumsum(per_table, axis=1)

            L_vals = CP_L_VALUES_K1 if k_cp == 1 else CP_L_VALUES_K2
            for L_cp in L_vals:
                match_counts = cum[:, L_cp - 1]
                for mh in MIN_HITS_VALUES:
                    if mh > L_cp:
                        continue
                    key = f"cp_k{k_cp}_L{L_cp}_h{mh}"
                    r_idx = np.where(match_counts >= mh)[0]
                    out, nr = do_snis(Q[qpos], valid_keys, valid_values, logits,
                                      r_idx, lambda th, kk: cp_index.collision_prob(th, kk),
                                      k_cp, L_cp, mh)
                    err = float(np.linalg.norm(out - full_output) / out_norm) if nr > 0 else float('nan')
                    results['lsh'][key]['errors'].append(err)
                    results['lsh'][key]['budgets'].append(nr)

    return results


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_results(all_results):
    agg = {'baselines': {}, 'lsh': {}}

    for method in ['topk', 'uniform', 'oracle']:
        agg['baselines'][method] = {}
        for k_pct in K_PERCENTAGES:
            k_str = str(k_pct)
            errs = []
            for r in all_results:
                errs.extend(r['baselines'][method][k_str])
            arr = np.array(errs)
            agg['baselines'][method][k_str] = {
                'mean': float(np.nanmean(arr)),
                'median': float(np.nanmedian(arr)),
                'std': float(np.nanstd(arr)),
                'n': len(arr)
            }

    for c in ALL_CONFIGS:
        key = c['key']
        errs, buds = [], []
        for r in all_results:
            errs.extend(r['lsh'][key]['errors'])
            buds.extend(r['lsh'][key]['budgets'])
        ea, ba = np.array(errs), np.array(buds)
        valid = ~np.isnan(ea)
        agg['lsh'][key] = {
            'mean_error': float(np.nanmean(ea)),
            'median_error': float(np.nanmedian(ea)),
            'std_error': float(np.nanstd(ea)),
            'mean_budget': float(np.mean(ba)),
            'median_budget': float(np.median(ba)),
            'budget_std': float(np.std(ba)),
            'empty_fraction': float(1.0 - np.mean(valid)),
            'n': int(np.sum(valid)),
        }
        agg['lsh'][key].update({k: v for k, v in c.items() if k != 'key'})

    return agg


# ============================================================================
# PLOTTING
# ============================================================================

MH_MARKERS = {1: 'o', 2: 's', 3: '^'}
MH_LABELS = {1: 'h>=1', 2: 'h>=2 (MagicPIG)', 3: 'h>=3'}


def plot_baselines(agg_first, agg_last):
    """Baseline curves: TopK, Uniform, Oracle vs budget %."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    x = np.array(K_PERCENTAGES, dtype=float)

    for ax, agg, title in [
        (axes[0], agg_first, 'First Layer (Layer 0)'),
        (axes[1], agg_last, 'Last Layer (Layer 31)')
    ]:
        for method, label, color, marker, ls in [
            ('topk', 'Top-K', '#8b5cf6', 'o', '-'),
            ('uniform', 'Uniform Sampling', '#f97316', '^', '-.'),
            ('oracle', 'Oracle Sampling', '#16a34a', 'D', '--'),
        ]:
            means = np.array([agg['baselines'][method][str(k)]['mean'] for k in K_PERCENTAGES])
            stds = np.array([agg['baselines'][method][str(k)]['std'] for k in K_PERCENTAGES])
            ax.plot(x, means, marker=marker, linewidth=2.2, markersize=5,
                    color=color, label=label, linestyle=ls, alpha=0.9)
            ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.08)

        ax.set_xlabel('Budget (% of keys)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Relative L2 Error', fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.set_xlim([0, 100])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9)

    fig.suptitle('Baseline Methods: TopK vs Uniform vs Oracle', fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def plot_scatter_by_minhits(agg, layer_label, mh_val):
    """Scatter: budget vs error for a specific min_hits, both SimHash and CP."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # SimHash
    sh_cmap = plt.cm.Blues(np.linspace(0.3, 0.95, len(SIMHASH_K_VALUES)))
    for i, K in enumerate(SIMHASH_K_VALUES):
        bx, by, labs = [], [], []
        for L in SIMHASH_L_VALUES:
            key = f"simhash_K{K}_L{L}_h{mh_val}"
            if key not in agg['lsh']:
                continue
            d = agg['lsh'][key]
            if d['n'] > 0 and d['mean_budget'] > 0:
                bx.append(d['mean_budget'])
                by.append(d['mean_error'])
                labs.append(f"L={L}")
        if bx:
            ax.scatter(bx, by, color=sh_cmap[i], marker='o', s=55, alpha=0.85,
                       label=f'SimHash K={K}', zorder=5)
            for xi, yi, lab in zip(bx, by, labs):
                ax.annotate(lab, (xi, yi), fontsize=5.5, alpha=0.5,
                            xytext=(3, 3), textcoords='offset points')

    # CrossPoly
    cp_colors = {1: '#e11d48', 2: '#f97316'}
    for k_cp in CP_K_VALUES:
        L_vals = CP_L_VALUES_K1 if k_cp == 1 else CP_L_VALUES_K2
        bx, by, labs = [], [], []
        for L in L_vals:
            key = f"cp_k{k_cp}_L{L}_h{mh_val}"
            if key not in agg['lsh']:
                continue
            d = agg['lsh'][key]
            if d['n'] > 0 and d['mean_budget'] > 0:
                bx.append(d['mean_budget'])
                by.append(d['mean_error'])
                labs.append(f"L={L}")
        if bx:
            ax.scatter(bx, by, color=cp_colors[k_cp], marker='*', s=90, alpha=0.85,
                       label=f'CrossPoly k={k_cp}', zorder=5)
            for xi, yi, lab in zip(bx, by, labs):
                ax.annotate(lab, (xi, yi), fontsize=5.5, alpha=0.5,
                            xytext=(3, 3), textcoords='offset points')

    # Baseline reference lines
    for method, label, color in [
        ('topk', 'Top-K @10%', '#8b5cf6'),
        ('uniform', 'Uniform @10%', '#f97316'),
        ('oracle', 'Oracle @10%', '#16a34a'),
    ]:
        val = agg['baselines'][method]['10']['mean']
        ax.axhline(y=val, color=color, linestyle='--', alpha=0.4,
                   label=f'{label} = {val:.3f}')

    ax.set_xlabel('Mean Keys Retrieved', fontweight='bold', fontsize=12)
    ax.set_ylabel('Mean Relative L2 Error', fontweight='bold', fontsize=12)
    ax.set_title(f'Budget vs Error — {MH_LABELS[mh_val]} ({layer_label})',
                 fontweight='bold', fontsize=13)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    return fig


def plot_minhits_comparison(agg, layer_label):
    """Compare min_hits=1 vs 2 vs 3 for the same (K, L) configs."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: SimHash
    ax = axes[0]
    ax.set_title(f'SimHash: Effect of min_hits ({layer_label})', fontweight='bold')
    for mh, marker, alpha in [(1, 'o', 0.5), (2, 's', 0.8), (3, '^', 1.0)]:
        bx, by = [], []
        for K in SIMHASH_K_VALUES:
            for L in SIMHASH_L_VALUES:
                key = f"simhash_K{K}_L{L}_h{mh}"
                if key not in agg['lsh']:
                    continue
                d = agg['lsh'][key]
                if d['n'] > 0 and d['mean_budget'] > 0:
                    bx.append(d['mean_budget'])
                    by.append(d['mean_error'])
        if bx:
            ax.scatter(bx, by, marker=marker, s=40, alpha=alpha,
                       label=MH_LABELS[mh])
    ax.set_xlabel('Mean Keys Retrieved')
    ax.set_ylabel('Mean Relative L2 Error')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

    # Right: CrossPoly k=1
    ax = axes[1]
    ax.set_title(f'CrossPoly k=1: Effect of min_hits ({layer_label})', fontweight='bold')
    for mh, marker, alpha in [(1, 'o', 0.5), (2, 's', 0.8), (3, '^', 1.0)]:
        bx, by = [], []
        for L in CP_L_VALUES_K1:
            key = f"cp_k1_L{L}_h{mh}"
            if key not in agg['lsh']:
                continue
            d = agg['lsh'][key]
            if d['n'] > 0 and d['mean_budget'] > 0:
                bx.append(d['mean_budget'])
                by.append(d['mean_error'])
        if bx:
            ax.scatter(bx, by, marker=marker, s=60, alpha=alpha,
                       label=MH_LABELS[mh])
    ax.set_xlabel('Mean Keys Retrieved')
    ax.set_ylabel('Mean Relative L2 Error')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    n_configs = len(ALL_CONFIGS)
    print("=" * 70)
    print("LSH METHOD COMPARISON: SimHash vs Cross-Polytope + min_hits sweep")
    print("=" * 70)
    print(f"Config: {NUM_EXAMPLES} examples, {NUM_QUERIES} queries/example")
    print(f"SimHash: K={SIMHASH_K_VALUES}, L={SIMHASH_L_VALUES}")
    print(f"CrossPoly k=1: L={CP_L_VALUES_K1}")
    print(f"CrossPoly k=2: L={CP_L_VALUES_K2}")
    print(f"Min hits: {MIN_HITS_VALUES}")
    print(f"Total LSH configs: {n_configs}")
    print(f"Shared index: SimHash [{MAX_SH_L}T, {MAX_SH_K}K], CP [{MAX_CP_L}T, {MAX_CP_K}k]")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Count and select
    print(f"Counting examples in {DATA_PATH}...")
    with open(DATA_PATH, 'r') as f:
        total = sum(1 for _ in f)
    print(f"Found {total} examples")
    num_to_load = min(NUM_EXAMPLES, total)
    selected = sorted(rng.choice(total, num_to_load, replace=False).tolist())
    print(f"Selected {num_to_load} random examples")

    # Build shared structures
    print("Building shared LSH structures...")
    sh_idx = SimHashIndex(MAX_SH_L, MAX_SH_K, HEAD_DIM, center_keys=True, seed=SEED)
    cp_idx = CrossPolytopeIndex(MAX_CP_L, MAX_CP_K, HEAD_DIM, center_keys=True, seed=SEED + 999)
    print(f"  SimHash: {MAX_SH_L}T x {MAX_SH_K}K")
    print(f"  CrossPoly: {MAX_CP_L}T x {MAX_CP_K}k")

    # Process
    print("\nProcessing examples...")
    sel_set = set(selected)
    per_layer = {l: [] for l in LAYERS}

    loaded = 0
    with open(DATA_PATH, 'r') as f:
        for idx, line in enumerate(f):
            if idx not in sel_set:
                continue
            example = json.loads(line)
            loaded += 1

            for layer in LAYERS:
                t_ex = time.time()
                res = analyze_example(example, layer, sh_idx, cp_idx, rng)
                per_layer[layer].append(res)
                dt = time.time() - t_ex
                print(f"  [{loaded:3d}/{num_to_load}] {layer}: "
                      f"{example.get('domain', '?')[:30]:<30s} ({dt:.1f}s)")

            if loaded >= num_to_load:
                break

    # Aggregate
    print("\nAggregating results...")
    aggregated = {}
    for layer in LAYERS:
        aggregated[layer] = aggregate_results(per_layer[layer])

    # Save JSON
    output_json = {
        'metadata': {
            'timestamp': TIMESTAMP,
            'num_examples': num_to_load,
            'num_queries_per_example': NUM_QUERIES,
            'seed': SEED,
            'head_dim': HEAD_DIM,
            'min_hits_values': MIN_HITS_VALUES,
            'k_percentages': K_PERCENTAGES,
            'simhash_K_values': SIMHASH_K_VALUES,
            'simhash_L_values': SIMHASH_L_VALUES,
            'cp_k_values': CP_K_VALUES,
            'cp_L_values_k1': CP_L_VALUES_K1,
            'cp_L_values_k2': CP_L_VALUES_K2,
            'layers': LAYERS,
            'total_configs': n_configs,
            'total_time_seconds': time.time() - t0
        },
        'aggregated': aggregated,
        'configs': [c for c in ALL_CONFIGS]
    }
    json_path = OUTPUT_DIR / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"Saved: {json_path}")

    # Plots
    print("Generating plots...")

    fig = plot_baselines(aggregated['first_layer'], aggregated['last_layer'])
    fig.savefig(OUTPUT_DIR / 'baselines.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    for layer in LAYERS:
        ll = 'First Layer' if 'first' in layer else 'Last Layer'
        for mh in MIN_HITS_VALUES:
            fig = plot_scatter_by_minhits(aggregated[layer], ll, mh)
            fig.savefig(OUTPUT_DIR / f'scatter_{layer}_h{mh}.png',
                        dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()

        fig = plot_minhits_comparison(aggregated[layer], ll)
        fig.savefig(OUTPUT_DIR / f'minhits_comparison_{layer}.png',
                    dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

    elapsed = time.time() - t0
    print(f"\nDone! Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Results: {OUTPUT_DIR}")

    # Summary
    print("\n" + "=" * 95)
    print("SUMMARY — Last Layer, min_hits=2 (MagicPIG-style)")
    print("=" * 95)
    agg = aggregated['last_layer']

    print(f"\n{'Method':<35} {'Budget':>8} {'Error':>8} {'Med Err':>8} {'Empty%':>7}")
    print("-" * 70)
    for m in ['topk', 'uniform', 'oracle']:
        d = agg['baselines'][m]['10']
        print(f"{m.capitalize()+' @10%':<35} {'~600':>8} {d['mean']:>8.4f} {d['median']:>8.4f} {'0.0%':>7}")

    print()
    for K in SIMHASH_K_VALUES:
        for L in SIMHASH_L_VALUES:
            key = f"simhash_K{K}_L{L}_h2"
            if key not in agg['lsh']:
                continue
            d = agg['lsh'][key]
            print(f"{'SimHash '+key:<35} {d['mean_budget']:>8.0f} "
                  f"{d['mean_error']:>8.4f} {d['median_error']:>8.4f} "
                  f"{d['empty_fraction']*100:>6.1f}%")

    print()
    for k_cp in CP_K_VALUES:
        L_vals = CP_L_VALUES_K1 if k_cp == 1 else CP_L_VALUES_K2
        for L in L_vals:
            key = f"cp_k{k_cp}_L{L}_h2"
            if key not in agg['lsh']:
                continue
            d = agg['lsh'][key]
            print(f"{'CrossPoly '+key:<35} {d['mean_budget']:>8.0f} "
                  f"{d['mean_error']:>8.4f} {d['median_error']:>8.4f} "
                  f"{d['empty_fraction']*100:>6.1f}%")


if __name__ == "__main__":
    main()
