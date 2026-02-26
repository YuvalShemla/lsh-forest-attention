#!/usr/bin/env python3
"""
Hierarchical LSH Attention Evaluation

Evaluates the tree-aggregation approach across a grid of (K, L) configurations,
with budget-controlled baselines (TopK, Uniform, Oracle).

Optimization: builds a shared SimHashIndex at max depth/tables once per example,
batch-hashes all queries, and computes full LCP at max depth once per query.
Per-config results are obtained by slicing: np.minimum(full_lcp[:, :L], K).

Output: timestamped subfolder with JSON + plots.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from algorithms.base import softmax as stable_softmax
from algorithms.hierarchical_lsh import hierarchical_lsh_attention

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = '../../../data/attention_vectors_long_bench_llama_8b.jsonl'
BASE_OUTPUT_DIR = Path('../../../results/hierarchical_grid_sweep')
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
NUM_EXAMPLES = 10
NUM_QUERIES = 100

# Hierarchical LSH parameter grid
K_VALUES = [1, 2, 4, 8, 12, 16, 20]
L_VALUES = [1, 5, 10, 20, 50, 100]

# Shared index: must cover max of both
MAX_K = max(K_VALUES)
MAX_L = max(L_VALUES)

# Baseline budgets (absolute key counts)
BASELINE_BUDGETS = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]

# ============================================================================
# SETUP
# ============================================================================
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{TIMESTAMP}'


# ============================================================================
# LOCAL SIMHASH INDEX (for shared max-size structure)
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


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


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
# PER-EXAMPLE ANALYSIS
# ============================================================================

def analyze_example(example, layer_name, sh_index, rng):
    """Analyze one example: compute baselines and all hierarchical configs per query."""
    Q = np.array(example[layer_name]['Q'], dtype=np.float32)
    K_mat = np.array(example[layer_name]['K'], dtype=np.float32)
    V = np.array(example[layer_name]['V'], dtype=np.float32)
    seq_len = Q.shape[0]
    query_positions = list(range(max(0, seq_len - NUM_QUERIES), seq_len))

    # Build index once
    sh_index.build_index(K_mat)

    # Pre-compute all logits and batch-hash all queries
    all_logits = (Q @ K_mat.T) / np.sqrt(HEAD_DIM)
    all_qhashes = sh_index.batch_hash_queries(Q)  # [seq, MAX_L, MAX_K]

    # Init result containers
    results = {
        'baselines': {m: {str(b): [] for b in BASELINE_BUDGETS}
                      for m in ['topk', 'uniform', 'oracle']},
        'hierarchical': {f"K{K}_L{L}": {'budgets': [], 'errors': []}
                         for K in K_VALUES for L in L_VALUES}
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
        for budget in BASELINE_BUDGETS:
            if budget > nv:
                # Skip budgets larger than available keys
                for m in ['topk', 'uniform', 'oracle']:
                    results['baselines'][m][str(budget)].append(float('nan'))
                continue
            topk_out = compute_topk(logits, valid_values, budget)
            results['baselines']['topk'][str(budget)].append(
                float(np.linalg.norm(topk_out - full_output) / out_norm))
            uni_out = compute_uniform(logits, valid_values, budget, rng)
            results['baselines']['uniform'][str(budget)].append(
                float(np.linalg.norm(uni_out - full_output) / out_norm))
            oracle_out = compute_oracle(logits, valid_values, full_weights, budget, rng)
            results['baselines']['oracle'][str(budget)].append(
                float(np.linalg.norm(oracle_out - full_output) / out_norm))

        # --- Hierarchical LSH: compute full LCP once at max depth ---
        q_hash = all_qhashes[qpos]  # [MAX_L, MAX_K]
        key_codes = sh_index.key_codes[:nv]  # [nv, MAX_L, MAX_K]

        # Full LCP at max depth: vectorized
        matches = (key_codes[:, :MAX_L, :MAX_K] == q_hash[:MAX_L, :MAX_K])
        cum_match = np.cumprod(matches, axis=2)
        full_lcp = np.sum(cum_match, axis=2).astype(np.int32)  # [nv, MAX_L]

        for K_depth in K_VALUES:
            # Clamp LCP to current depth
            clamped_lcp = np.minimum(full_lcp, K_depth)  # [nv, MAX_L]

            for L_trees in L_VALUES:
                key = f"K{K_depth}_L{L_trees}"
                lcp_slice = clamped_lcp[:, :L_trees]  # [nv, L_trees]

                # Run hierarchical attention per tree, average
                tree_outputs = []
                total_groups = 0
                sqrt_d = np.sqrt(HEAD_DIM)

                for l in range(L_trees):
                    lcp_l = lcp_slice[:, l]

                    group_avg_keys = []
                    group_avg_values = []
                    group_counts = []

                    for d in range(K_depth + 1):
                        if d < K_depth:
                            mask = (lcp_l == d)
                        else:
                            mask = (lcp_l >= K_depth)

                        count = np.sum(mask)
                        if count == 0:
                            continue

                        avg_key = np.mean(valid_keys[mask], axis=0)
                        avg_value = np.mean(valid_values[mask], axis=0)
                        group_avg_keys.append(avg_key)
                        group_avg_values.append(avg_value)
                        group_counts.append(count)

                    n_groups = len(group_counts)
                    if n_groups == 0:
                        continue

                    total_groups += n_groups

                    avg_keys_arr = np.array(group_avg_keys)
                    avg_vals_arr = np.array(group_avg_values)
                    counts_arr = np.array(group_counts, dtype=np.float64)

                    scores = (avg_keys_arr @ Q[qpos]) / sqrt_d + np.log(counts_arr)
                    weights = stable_softmax(scores)
                    tree_out = weights @ avg_vals_arr
                    tree_outputs.append(tree_out)

                if len(tree_outputs) == 0:
                    results['hierarchical'][key]['errors'].append(float('nan'))
                    results['hierarchical'][key]['budgets'].append(0)
                else:
                    h_output = np.mean(tree_outputs, axis=0)
                    err = float(np.linalg.norm(h_output - full_output) / out_norm)
                    results['hierarchical'][key]['errors'].append(err)
                    results['hierarchical'][key]['budgets'].append(total_groups)

    return results


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_results(all_results):
    """Aggregate per-example results into statistics."""
    agg = {'baselines': {}, 'hierarchical': {}}

    for method in ['topk', 'uniform', 'oracle']:
        agg['baselines'][method] = {}
        for budget in BASELINE_BUDGETS:
            b_str = str(budget)
            errs = []
            for r in all_results:
                errs.extend(r['baselines'][method][b_str])
            arr = np.array(errs)
            agg['baselines'][method][b_str] = {
                'mean': float(np.nanmean(arr)),
                'median': float(np.nanmedian(arr)),
                'std': float(np.nanstd(arr)),
                'n': int(np.sum(~np.isnan(arr))),
                'budget': budget,
            }

    for K_depth in K_VALUES:
        for L_trees in L_VALUES:
            key = f"K{K_depth}_L{L_trees}"
            errs, buds = [], []
            for r in all_results:
                errs.extend(r['hierarchical'][key]['errors'])
                buds.extend(r['hierarchical'][key]['budgets'])
            ea, ba = np.array(errs), np.array(buds)
            valid = ~np.isnan(ea)
            agg['hierarchical'][key] = {
                'K': K_depth,
                'L': L_trees,
                'mean_error': float(np.nanmean(ea)),
                'median_error': float(np.nanmedian(ea)),
                'std_error': float(np.nanstd(ea)),
                'mean_budget': float(np.mean(ba)),
                'median_budget': float(np.median(ba)),
                'budget_std': float(np.std(ba)),
                'empty_fraction': float(1.0 - np.mean(valid)),
                'n': int(np.sum(valid)),
            }

    return agg


# ============================================================================
# PLOTTING
# ============================================================================

K_COLORS = {
    1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c', 8: '#d62728',
    12: '#9467bd', 16: '#8c564b', 20: '#e377c2'
}


def plot_scatter(agg, layer_label, output_dir):
    """Scatter plot: effective_budget vs error, color by K, baselines overlaid."""
    fig, ax = plt.subplots(figsize=(14, 9))

    # Plot hierarchical configs
    for K_depth in K_VALUES:
        bx, by, labs = [], [], []
        for L_trees in L_VALUES:
            key = f"K{K_depth}_L{L_trees}"
            d = agg['hierarchical'][key]
            if d['n'] > 0 and d['mean_budget'] > 0:
                bx.append(d['mean_budget'])
                by.append(d['mean_error'])
                labs.append(f"L={L_trees}")
        if bx:
            color = K_COLORS.get(K_depth, '#333333')
            ax.scatter(bx, by, color=color, marker='o', s=70, alpha=0.85,
                       label=f'Hierarchical K={K_depth}', zorder=5)
            for xi, yi, lab in zip(bx, by, labs):
                ax.annotate(lab, (xi, yi), fontsize=6, alpha=0.5,
                            xytext=(4, 4), textcoords='offset points')

    # Baseline curves
    for method, label, color, marker, ls in [
        ('topk', 'Top-K', '#8b5cf6', 'v', '-'),
        ('uniform', 'Uniform', '#f97316', '^', '-.'),
        ('oracle', 'Oracle', '#16a34a', 'D', '--'),
    ]:
        bx, by = [], []
        for budget in BASELINE_BUDGETS:
            d = agg['baselines'][method][str(budget)]
            if d['n'] > 0:
                bx.append(budget)
                by.append(d['mean'])
        if bx:
            ax.plot(bx, by, marker=marker, linewidth=2, markersize=6,
                    color=color, label=label, linestyle=ls, alpha=0.7, zorder=3)

    ax.set_xlabel('Effective Budget (keys/groups used)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Mean Relative L2 Error', fontweight='bold', fontsize=12)
    ax.set_title(f'Hierarchical LSH Attention: Budget vs Error ({layer_label})',
                 fontweight='bold', fontsize=14)
    ax.set_xscale('log')
    ax.legend(fontsize=8, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fname = f'scatter_{layer_label.lower().replace(" ", "_")}.png'
    fig.savefig(output_dir / fname, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_error_heatmap(agg, layer_label, output_dir):
    """Heatmap: K x L grid showing mean error."""
    error_matrix = np.full((len(K_VALUES), len(L_VALUES)), np.nan)
    for i, K_depth in enumerate(K_VALUES):
        for j, L_trees in enumerate(L_VALUES):
            key = f"K{K_depth}_L{L_trees}"
            d = agg['hierarchical'][key]
            if d['n'] > 0:
                error_matrix[i, j] = d['mean_error']

    fig, ax = plt.subplots(figsize=(10, 7))
    im = sns.heatmap(error_matrix, annot=True, fmt='.4f', cmap='RdYlGn_r',
                     xticklabels=[str(l) for l in L_VALUES],
                     yticklabels=[str(k) for k in K_VALUES],
                     ax=ax, linewidths=0.5, cbar_kws={'label': 'Mean Rel. L2 Error'})
    ax.set_xlabel('L (number of trees)', fontweight='bold', fontsize=12)
    ax.set_ylabel('K (tree depth)', fontweight='bold', fontsize=12)
    ax.set_title(f'Mean Error: K x L Grid ({layer_label})',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    fname = f'heatmap_error_{layer_label.lower().replace(" ", "_")}.png'
    fig.savefig(output_dir / fname, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_budget_heatmap(agg, layer_label, output_dir):
    """Heatmap: K x L grid showing mean effective budget."""
    budget_matrix = np.full((len(K_VALUES), len(L_VALUES)), np.nan)
    for i, K_depth in enumerate(K_VALUES):
        for j, L_trees in enumerate(L_VALUES):
            key = f"K{K_depth}_L{L_trees}"
            d = agg['hierarchical'][key]
            if d['n'] > 0:
                budget_matrix[i, j] = d['mean_budget']

    fig, ax = plt.subplots(figsize=(10, 7))
    im = sns.heatmap(budget_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                     xticklabels=[str(l) for l in L_VALUES],
                     yticklabels=[str(k) for k in K_VALUES],
                     ax=ax, linewidths=0.5,
                     cbar_kws={'label': 'Mean Effective Budget'})
    ax.set_xlabel('L (number of trees)', fontweight='bold', fontsize=12)
    ax.set_ylabel('K (tree depth)', fontweight='bold', fontsize=12)
    ax.set_title(f'Mean Effective Budget: K x L Grid ({layer_label})',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    fname = f'heatmap_budget_{layer_label.lower().replace(" ", "_")}.png'
    fig.savefig(output_dir / fname, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_configs = len(K_VALUES) * len(L_VALUES)
    print("=" * 70)
    print("HIERARCHICAL LSH ATTENTION EVALUATION")
    print("=" * 70)
    print(f"Config: {NUM_EXAMPLES} examples, {NUM_QUERIES} queries/example")
    print(f"K (depth): {K_VALUES}")
    print(f"L (trees): {L_VALUES}")
    print(f"Total hierarchical configs: {n_configs}")
    print(f"Baseline budgets: {BASELINE_BUDGETS}")
    print(f"Shared SimHash index: {MAX_L}T x {MAX_K}K")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Count and select examples
    print(f"Counting examples in {DATA_PATH}...")
    with open(DATA_PATH, 'r') as f:
        total = sum(1 for _ in f)
    print(f"Found {total} examples")
    num_to_load = min(NUM_EXAMPLES, total)
    selected = sorted(rng.choice(total, num_to_load, replace=False).tolist())
    print(f"Selected {num_to_load} random examples")

    # Build shared SimHash index
    print("Building shared SimHash index...")
    sh_idx = SimHashIndex(MAX_L, MAX_K, HEAD_DIM, center_keys=True, seed=SEED)
    print(f"  SimHash: {MAX_L}T x {MAX_K}K")

    # Process examples
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
                res = analyze_example(example, layer, sh_idx, rng)
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
            'K_values': K_VALUES,
            'L_values': L_VALUES,
            'baseline_budgets': BASELINE_BUDGETS,
            'layers': LAYERS,
            'total_configs': n_configs,
            'total_time_seconds': time.time() - t0,
        },
        'aggregated': aggregated,
    }
    json_path = OUTPUT_DIR / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"Saved: {json_path}")

    # Plots
    print("Generating plots...")
    for layer in LAYERS:
        ll = 'First Layer' if 'first' in layer else 'Last Layer'
        plot_scatter(aggregated[layer], ll, OUTPUT_DIR)
        plot_error_heatmap(aggregated[layer], ll, OUTPUT_DIR)
        plot_budget_heatmap(aggregated[layer], ll, OUTPUT_DIR)

    elapsed = time.time() - t0
    print(f"\nDone! Total time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"Results: {OUTPUT_DIR}")

    # Summary
    print("\n" + "=" * 85)
    print("SUMMARY -- Last Layer")
    print("=" * 85)
    agg = aggregated['last_layer']

    print(f"\n{'Method':<25} {'Budget':>10} {'Error':>10} {'Med Err':>10}")
    print("-" * 60)

    # Baselines
    for m in ['topk', 'uniform', 'oracle']:
        for b in BASELINE_BUDGETS:
            b_str = str(b)
            if b_str not in agg['baselines'][m]:
                continue
            d = agg['baselines'][m][b_str]
            if d['n'] > 0:
                print(f"{m.capitalize() + f' @{b}':<25} {b:>10d} "
                      f"{d['mean']:>10.4f} {d['median']:>10.4f}")

    print()
    # Hierarchical
    for K_depth in K_VALUES:
        for L_trees in L_VALUES:
            key = f"K{K_depth}_L{L_trees}"
            d = agg['hierarchical'][key]
            if d['n'] > 0:
                print(f"{'Hier ' + key:<25} {d['mean_budget']:>10.0f} "
                      f"{d['mean_error']:>10.4f} {d['median_error']:>10.4f}")


if __name__ == "__main__":
    main()
