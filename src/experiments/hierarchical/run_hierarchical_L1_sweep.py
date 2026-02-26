#!/usr/bin/env python3
"""
Hierarchical LSH Attention: L=1 Depth Sweep with Multiple Seeds

Runs 50 independent experiments (different LSH hash functions each time)
with L=1 tree and K=[1..16] depths. Precomputes baselines once, then
only re-runs hierarchical part per seed.

Output:
  results/hierarchical_lsh_L1_sweep/
    individual_runs/run_XX.png   (50 per-run plots)
    averaged_results.json
    averaged_scatter.png
    averaged_heatmap.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = '../../data/attention_vectors_updated_long.jsonl'
OUTPUT_DIR = Path('../../results/hierarchical_lsh_L1_sweep')
INDIVIDUAL_DIR = OUTPUT_DIR / 'individual_runs'
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128

NUM_EXAMPLES = 50
NUM_QUERIES = 100
NUM_RUNS = 50
EXAMPLE_SEED = 42  # Fixed seed for example selection (same examples every run)

K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16]
L_FIXED = 1

BASELINE_BUDGETS = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]

# ============================================================================
# SETUP
# ============================================================================
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

K_COLORS = {}
cmap = plt.cm.viridis(np.linspace(0.1, 0.95, len(K_VALUES)))
for i, k in enumerate(K_VALUES):
    K_COLORS[k] = cmap[i]


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


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


def build_simhash_hyperplanes(max_depth, head_dim, seed):
    """Build L=1 SimHash hyperplanes. Returns [1, max_depth, head_dim]."""
    rng = np.random.RandomState(seed)
    hp = rng.randn(1, max_depth, head_dim).astype(np.float32)
    return hp / np.linalg.norm(hp, axis=2, keepdims=True)


def hash_keys(keys, hyperplanes, key_mean):
    """Hash keys with L=1. Returns [nv, 1, max_depth] int8."""
    c = keys - key_mean
    return (np.einsum('nd,ltd->nlt', c, hyperplanes) > 0).astype(np.int8)


def hash_queries(queries, hyperplanes, key_mean):
    """Hash queries with L=1. Returns [nq, 1, max_depth] int8."""
    c = queries - key_mean
    return (np.einsum('qd,ltd->qlt', c, hyperplanes) > 0).astype(np.int8)


def hierarchical_L1(query, valid_keys, valid_values, key_codes_1d,
                    query_hash_1d, K_depth, head_dim, sqrt_d):
    """
    Hierarchical attention for a single tree (L=1).

    key_codes_1d: [nv, max_depth]
    query_hash_1d: [max_depth]
    """
    nv = len(valid_keys)

    # Compute LCP for L=1
    matches = (key_codes_1d[:nv, :K_depth] == query_hash_1d[:K_depth])
    cum_match = np.cumprod(matches, axis=1)
    lcp = np.sum(cum_match, axis=1).astype(np.int32)  # [nv]

    group_avg_keys = []
    group_avg_values = []
    group_counts = []

    for d in range(K_depth + 1):
        if d < K_depth:
            mask = (lcp == d)
        else:
            mask = (lcp >= K_depth)

        count = np.sum(mask)
        if count == 0:
            continue

        group_avg_keys.append(np.mean(valid_keys[mask], axis=0))
        group_avg_values.append(np.mean(valid_values[mask], axis=0))
        group_counts.append(count)

    n_groups = len(group_counts)
    if n_groups == 0:
        return np.zeros(head_dim), 0

    avg_keys_arr = np.array(group_avg_keys)
    avg_vals_arr = np.array(group_avg_values)
    counts_arr = np.array(group_counts, dtype=np.float64)

    scores = (avg_keys_arr @ query) / sqrt_d + np.log(counts_arr)
    weights = stable_softmax(scores)
    output = weights @ avg_vals_arr

    return output, n_groups


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)

    max_K = max(K_VALUES)
    sqrt_d = np.sqrt(HEAD_DIM)

    print("=" * 70)
    print("HIERARCHICAL LSH L=1 SWEEP (50 seeds)")
    print("=" * 70)
    print(f"Config: {NUM_EXAMPLES} examples, {NUM_QUERIES} queries/example, {NUM_RUNS} runs")
    print(f"K (depth): {K_VALUES}")
    print(f"L: {L_FIXED} (fixed)")
    print(f"Baselines: {BASELINE_BUDGETS}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # ------------------------------------------------------------------
    # 1. Load examples (fixed selection)
    # ------------------------------------------------------------------
    print("Loading examples...")
    rng_select = np.random.RandomState(EXAMPLE_SEED)

    with open(DATA_PATH, 'r') as f:
        total = sum(1 for _ in f)
    print(f"  Found {total} examples")

    num_to_load = min(NUM_EXAMPLES, total)
    selected = sorted(rng_select.choice(total, num_to_load, replace=False).tolist())
    sel_set = set(selected)

    examples = []
    with open(DATA_PATH, 'r') as f:
        for idx, line in enumerate(f):
            if idx in sel_set:
                examples.append(json.loads(line))
            if len(examples) >= num_to_load:
                break
    print(f"  Loaded {len(examples)} examples")

    # ------------------------------------------------------------------
    # 2. Precompute per-example data + baselines (done once)
    # ------------------------------------------------------------------
    print("Precomputing ground truth and baselines...")

    # Structure: per_example[layer][ex_idx] = dict with Q, K, V,
    #   query_positions, ground_truths, out_norms, baselines
    precomputed = {layer: [] for layer in LAYERS}
    baseline_rng = np.random.RandomState(EXAMPLE_SEED + 1000)

    for ex_idx, example in enumerate(examples):
        for layer in LAYERS:
            Q = np.array(example[layer]['Q'], dtype=np.float32)
            K_mat = np.array(example[layer]['K'], dtype=np.float32)
            V = np.array(example[layer]['V'], dtype=np.float32)
            seq_len = Q.shape[0]
            query_positions = list(range(max(0, seq_len - NUM_QUERIES), seq_len))

            all_logits = (Q @ K_mat.T) / np.sqrt(HEAD_DIM)

            # Per-query precomputation
            ground_truths = []
            out_norms = []
            query_logits_list = []
            query_weights_list = []

            baseline_errors = {m: {str(b): [] for b in BASELINE_BUDGETS}
                               for m in ['topk', 'uniform', 'oracle']}

            for qpos in query_positions:
                nv = qpos + 1
                logits = all_logits[qpos, :nv]
                valid_values = V[:nv]
                full_w = softmax(logits)
                full_out = full_w @ valid_values
                out_norm = np.linalg.norm(full_out) + 1e-8

                ground_truths.append(full_out)
                out_norms.append(out_norm)
                query_logits_list.append(logits)
                query_weights_list.append(full_w)

                for budget in BASELINE_BUDGETS:
                    if budget > nv:
                        for m in ['topk', 'uniform', 'oracle']:
                            baseline_errors[m][str(budget)].append(float('nan'))
                        continue
                    topk_out = compute_topk(logits, valid_values, budget)
                    baseline_errors['topk'][str(budget)].append(
                        float(np.linalg.norm(topk_out - full_out) / out_norm))
                    uni_out = compute_uniform(logits, valid_values, budget, baseline_rng)
                    baseline_errors['uniform'][str(budget)].append(
                        float(np.linalg.norm(uni_out - full_out) / out_norm))
                    oracle_out = compute_oracle(logits, valid_values, full_w, budget, baseline_rng)
                    baseline_errors['oracle'][str(budget)].append(
                        float(np.linalg.norm(oracle_out - full_out) / out_norm))

            precomputed[layer].append({
                'Q': Q, 'K': K_mat, 'V': V,
                'query_positions': query_positions,
                'ground_truths': ground_truths,
                'out_norms': out_norms,
                'baseline_errors': baseline_errors,
            })

        if (ex_idx + 1) % 10 == 0 or ex_idx == 0:
            print(f"  Precomputed {ex_idx + 1}/{num_to_load} examples")

    # Aggregate baselines once
    print("Aggregating baselines...")
    baseline_agg = {}
    for layer in LAYERS:
        baseline_agg[layer] = {}
        for method in ['topk', 'uniform', 'oracle']:
            baseline_agg[layer][method] = {}
            for budget in BASELINE_BUDGETS:
                b_str = str(budget)
                errs = []
                for ex_data in precomputed[layer]:
                    errs.extend(ex_data['baseline_errors'][method][b_str])
                arr = np.array(errs)
                baseline_agg[layer][method][b_str] = {
                    'mean': float(np.nanmean(arr)),
                    'median': float(np.nanmedian(arr)),
                    'std': float(np.nanstd(arr)),
                    'n': int(np.sum(~np.isnan(arr))),
                    'budget': budget,
                }

    t_precomp = time.time() - t0
    print(f"Precomputation done in {t_precomp:.0f}s ({t_precomp/60:.1f}min)")

    # ------------------------------------------------------------------
    # 3. Run 50 seeds
    # ------------------------------------------------------------------
    print(f"\nRunning {NUM_RUNS} hierarchical experiments...")

    # Collect per-run results for averaging
    # all_run_results[layer][K_str] = list of 50 dicts with 'errors', 'budgets'
    all_run_results = {
        layer: {str(k): [] for k in K_VALUES}
        for layer in LAYERS
    }

    for run_idx in range(NUM_RUNS):
        run_seed = run_idx * 7 + 100  # Deterministic but varied seeds
        t_run = time.time()

        # Per-run results
        run_results = {
            layer: {str(k): {'errors': [], 'budgets': []} for k in K_VALUES}
            for layer in LAYERS
        }

        for layer in LAYERS:
            for ex_idx, ex_data in enumerate(precomputed[layer]):
                Q = ex_data['Q']
                K_mat = ex_data['K']
                V = ex_data['V']
                query_positions = ex_data['query_positions']
                ground_truths = ex_data['ground_truths']
                out_norms = ex_data['out_norms']

                # Build L=1 SimHash index for this run's seed
                key_mean = np.mean(K_mat, axis=0)
                hyperplanes = build_simhash_hyperplanes(max_K, HEAD_DIM, run_seed)

                # Hash all keys: [seq_len, 1, max_K]
                key_codes_full = hash_keys(K_mat, hyperplanes, key_mean)
                # Squeeze L dimension: [seq_len, max_K]
                key_codes_1d = key_codes_full[:, 0, :]

                # Hash all queries
                query_hashes_full = hash_queries(Q, hyperplanes, key_mean)
                query_hashes_1d = query_hashes_full[:, 0, :]  # [seq_len, max_K]

                for q_idx, qpos in enumerate(query_positions):
                    nv = qpos + 1
                    full_out = ground_truths[q_idx]
                    out_norm = out_norms[q_idx]

                    q_hash = query_hashes_1d[qpos]  # [max_K]
                    kc = key_codes_1d[:nv]            # [nv, max_K]
                    valid_keys = K_mat[:nv]
                    valid_values = V[:nv]

                    for K_depth in K_VALUES:
                        h_out, n_groups = hierarchical_L1(
                            Q[qpos], valid_keys, valid_values,
                            kc, q_hash, K_depth, HEAD_DIM, sqrt_d
                        )
                        if n_groups > 0:
                            err = float(np.linalg.norm(h_out - full_out) / out_norm)
                        else:
                            err = float('nan')
                        run_results[layer][str(K_depth)]['errors'].append(err)
                        run_results[layer][str(K_depth)]['budgets'].append(n_groups)

        # Aggregate this run
        run_agg = {}
        for layer in LAYERS:
            run_agg[layer] = {}
            for K_depth in K_VALUES:
                k_str = str(K_depth)
                ea = np.array(run_results[layer][k_str]['errors'])
                ba = np.array(run_results[layer][k_str]['budgets'])
                run_agg[layer][k_str] = {
                    'K': K_depth,
                    'mean_error': float(np.nanmean(ea)),
                    'median_error': float(np.nanmedian(ea)),
                    'std_error': float(np.nanstd(ea)),
                    'mean_budget': float(np.mean(ba)),
                }
                # Store for averaging
                all_run_results[layer][k_str].append({
                    'mean_error': float(np.nanmean(ea)),
                    'median_error': float(np.nanmedian(ea)),
                    'mean_budget': float(np.mean(ba)),
                    'errors': ea.tolist(),
                    'budgets': ba.tolist(),
                })

        # Plot this run
        plot_single_run(run_agg, baseline_agg, run_idx, run_seed)

        dt = time.time() - t_run
        # Print summary line
        ll_err = run_agg['last_layer'][str(K_VALUES[-1])]['mean_error']
        fl_err = run_agg['first_layer'][str(K_VALUES[-1])]['mean_error']
        print(f"  Run {run_idx+1:2d}/{NUM_RUNS} (seed={run_seed:4d}): "
              f"last_layer K={K_VALUES[-1]} err={ll_err:.4f}, "
              f"first_layer K={K_VALUES[-1]} err={fl_err:.4f}  ({dt:.1f}s)")

    # ------------------------------------------------------------------
    # 4. Average across runs
    # ------------------------------------------------------------------
    print("\nAveraging across runs...")
    averaged = {}
    for layer in LAYERS:
        averaged[layer] = {}
        for K_depth in K_VALUES:
            k_str = str(K_depth)
            run_means = [r['mean_error'] for r in all_run_results[layer][k_str]]
            run_medians = [r['median_error'] for r in all_run_results[layer][k_str]]
            run_budgets = [r['mean_budget'] for r in all_run_results[layer][k_str]]

            averaged[layer][k_str] = {
                'K': K_depth,
                'mean_error_across_runs': float(np.mean(run_means)),
                'std_error_across_runs': float(np.std(run_means)),
                'median_error_across_runs': float(np.mean(run_medians)),
                'mean_budget_across_runs': float(np.mean(run_budgets)),
                'std_budget_across_runs': float(np.std(run_budgets)),
                'per_run_mean_errors': run_means,
                'per_run_mean_budgets': run_budgets,
            }

    # Save JSON
    output_json = {
        'metadata': {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'num_examples': num_to_load,
            'num_queries_per_example': NUM_QUERIES,
            'num_runs': NUM_RUNS,
            'example_seed': EXAMPLE_SEED,
            'K_values': K_VALUES,
            'L_fixed': L_FIXED,
            'baseline_budgets': BASELINE_BUDGETS,
            'layers': LAYERS,
            'total_time_seconds': time.time() - t0,
        },
        'baselines': baseline_agg,
        'averaged_hierarchical': averaged,
    }
    json_path = OUTPUT_DIR / 'averaged_results.json'
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"Saved: {json_path}")

    # Averaged plots
    plot_averaged(averaged, baseline_agg)

    elapsed = time.time() - t0
    print(f"\nDone! Total time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"Results: {OUTPUT_DIR}")

    # Summary
    print("\n" + "=" * 80)
    print("AVERAGED RESULTS ACROSS 50 RUNS")
    print("=" * 80)
    for layer in LAYERS:
        ll = 'First Layer' if 'first' in layer else 'Last Layer'
        print(f"\n--- {ll} ---")
        print(f"{'K':<6} {'Budget':>8} {'MeanErr':>10} {'StdErr':>10}")
        print("-" * 38)
        for K_depth in K_VALUES:
            d = averaged[layer][str(K_depth)]
            print(f"{K_depth:<6} {d['mean_budget_across_runs']:>8.1f} "
                  f"{d['mean_error_across_runs']:>10.4f} "
                  f"{d['std_error_across_runs']:>10.4f}")

        print(f"\nBaselines:")
        for m in ['topk', 'uniform', 'oracle']:
            for b in [10, 50, 100, 500]:
                d = baseline_agg[layer][m][str(b)]
                if d['n'] > 0:
                    print(f"  {m.capitalize()+f' @{b}':<20} {b:>6d}  {d['mean']:>10.4f}")


# ============================================================================
# PLOTTING
# ============================================================================

def plot_single_run(run_agg, baseline_agg, run_idx, seed):
    """Plot a single run: scatter of K configs vs baselines, both layers."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, layer, title in [
        (axes[0], 'first_layer', 'First Layer'),
        (axes[1], 'last_layer', 'Last Layer'),
    ]:
        # Hierarchical points
        for K_depth in K_VALUES:
            k_str = str(K_depth)
            d = run_agg[layer][k_str]
            if d['mean_budget'] > 0:
                ax.scatter(d['mean_budget'], d['mean_error'],
                           color=K_COLORS[K_depth], s=80, zorder=5,
                           label=f'K={K_depth}')

        # Baseline curves
        for method, label, color, marker, ls in [
            ('topk', 'Top-K', '#8b5cf6', 'v', '-'),
            ('uniform', 'Uniform', '#f97316', '^', '-.'),
            ('oracle', 'Oracle', '#16a34a', 'D', '--'),
        ]:
            bx, by = [], []
            for budget in BASELINE_BUDGETS:
                bd = baseline_agg[layer][method][str(budget)]
                if bd['n'] > 0:
                    bx.append(budget)
                    by.append(bd['mean'])
            if bx:
                ax.plot(bx, by, marker=marker, linewidth=1.5, markersize=5,
                        color=color, label=label, linestyle=ls, alpha=0.6, zorder=3)

        ax.set_xlabel('Effective Budget', fontsize=11)
        ax.set_ylabel('Mean Relative L2 Error', fontsize=11)
        ax.set_title(f'{title} (seed={seed})', fontweight='bold', fontsize=12)
        ax.set_xscale('log')
        ax.legend(fontsize=7, loc='upper right', ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)

    fig.suptitle(f'Hierarchical LSH L=1: Run {run_idx+1}',
                 fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(INDIVIDUAL_DIR / f'run_{run_idx+1:02d}.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_averaged(averaged, baseline_agg):
    """Plot averaged results across all runs."""

    # --- Scatter with error bars ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, layer, title in [
        (axes[0], 'first_layer', 'First Layer'),
        (axes[1], 'last_layer', 'Last Layer'),
    ]:
        # Hierarchical: mean +/- std across runs
        for K_depth in K_VALUES:
            d = averaged[layer][str(K_depth)]
            bud = d['mean_budget_across_runs']
            err = d['mean_error_across_runs']
            err_std = d['std_error_across_runs']
            bud_std = d['std_budget_across_runs']

            ax.errorbar(bud, err, xerr=bud_std, yerr=err_std,
                        fmt='o', color=K_COLORS[K_depth], markersize=8,
                        capsize=3, capthick=1, linewidth=1,
                        label=f'K={K_depth}', zorder=5)

        # Baselines
        for method, label, color, marker, ls in [
            ('topk', 'Top-K', '#8b5cf6', 'v', '-'),
            ('uniform', 'Uniform', '#f97316', '^', '-.'),
            ('oracle', 'Oracle', '#16a34a', 'D', '--'),
        ]:
            bx, by = [], []
            for budget in BASELINE_BUDGETS:
                bd = baseline_agg[layer][method][str(budget)]
                if bd['n'] > 0:
                    bx.append(budget)
                    by.append(bd['mean'])
            if bx:
                ax.plot(bx, by, marker=marker, linewidth=2, markersize=6,
                        color=color, label=label, linestyle=ls, alpha=0.7, zorder=3)

        ax.set_xlabel('Effective Budget', fontweight='bold', fontsize=12)
        ax.set_ylabel('Mean Relative L2 Error', fontweight='bold', fontsize=12)
        ax.set_title(f'{title} (avg over {NUM_RUNS} runs)', fontweight='bold', fontsize=13)
        ax.set_xscale('log')
        ax.legend(fontsize=7, loc='upper right', ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)

    fig.suptitle('Hierarchical LSH L=1: Averaged Across 50 Seeds',
                 fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'averaged_scatter.png',
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # --- Bar chart: error vs K for each layer ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, layer, title in [
        (axes[0], 'first_layer', 'First Layer'),
        (axes[1], 'last_layer', 'Last Layer'),
    ]:
        ks = K_VALUES
        means = [averaged[layer][str(k)]['mean_error_across_runs'] for k in ks]
        stds = [averaged[layer][str(k)]['std_error_across_runs'] for k in ks]
        colors = [K_COLORS[k] for k in ks]

        bars = ax.bar(range(len(ks)), means, yerr=stds, capsize=4,
                      color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel('K (tree depth)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Mean Relative L2 Error', fontweight='bold', fontsize=12)
        ax.set_title(f'{title}: Error vs Depth (L=1, avg {NUM_RUNS} runs)',
                     fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Reference lines
        for method, label, color, ls in [
            ('uniform', 'Uniform @50', '#f97316', '--'),
            ('oracle', 'Oracle @50', '#16a34a', ':'),
        ]:
            val = baseline_agg[layer][method]['50']['mean']
            ax.axhline(y=val, color=color, linestyle=ls, alpha=0.7,
                       linewidth=1.5, label=f'{label} = {val:.3f}')

        ax.legend(fontsize=9, loc='upper right')
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'averaged_bar_chart.png',
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # --- Violin/box plot: distribution of per-run mean errors ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, layer, title in [
        (axes[0], 'first_layer', 'First Layer'),
        (axes[1], 'last_layer', 'Last Layer'),
    ]:
        data_for_box = []
        labels = []
        for K_depth in K_VALUES:
            run_means = averaged[layer][str(K_depth)]['per_run_mean_errors']
            data_for_box.append(run_means)
            labels.append(f'K={K_depth}')

        bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5))
        for patch, K_depth in zip(bp['boxes'], K_VALUES):
            patch.set_facecolor(K_COLORS[K_depth])
            patch.set_alpha(0.7)

        ax.set_xlabel('Configuration', fontweight='bold', fontsize=12)
        ax.set_ylabel('Mean Relative L2 Error (per run)', fontweight='bold', fontsize=12)
        ax.set_title(f'{title}: Distribution Across {NUM_RUNS} Seeds',
                     fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'distribution_boxplot.png',
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == "__main__":
    main()
