#!/usr/bin/env python3
"""
Compare All Algorithms: Full Attention Approximation Benchmark

Runs all 7 algorithms on the same data with a budget sweep:

Budget-controlled methods (evaluated at each budget):
  1. Full Attention (ground truth baseline)
  2. Top-K (biased)
  3. Uniform Sampling (biased)
  4. Oracle Sampling (unbiased, privileged)
  5. Jungle Sampling / prefix_sampling (ours)

Non-budget-controlled LSH-SNIS methods (evaluated once, variable budget):
  6. SimHash SNIS
  7. Cross-Polytope SNIS

Results saved to: results/approximation_evaluation/v2/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time

from algorithms import (
    full_attention,
    topk_attention,
    uniform_sampling,
    oracle_sampling,
    jungle_sampling,
    simhash_snis,
    cross_polytope_snis,
)
from algorithms.lsh_index import (
    LSHStructure,
    SimHashIndex,
    CrossPolytopeIndex,
)
from algorithms.base import (
    compute_ground_truth_attention,
    relative_l2_error,
)
from visualization.plot_utils import (
    setup_style,
    plot_error_vs_budget,
    plot_scatter_with_fit,
    save_figure,
    format_log_yaxis,
    METHOD_STYLES,
)

# ============================================================================
# HYPERPARAMETERS - MODIFY HERE
# ============================================================================

# LSH Configuration
L = 50           # Tables for Jungle Sampling
K_MAX = 30       # Max depth
SEED = 42

# SimHash-SNIS fixed params (one config for this experiment)
SIMHASH_K = 7
SIMHASH_L = 30
SIMHASH_MIN_HITS = 2

# Cross-Polytope fixed params
CP_K = 1
CP_L = 75
CP_MIN_HITS = 2

# Jungle Sampling params
GAMMA = 1.0
TAU = 0.0
MIN_DEPTH = 5

# Budget sweep
BUDGETS = list(range(20, 220, 20))

# Evaluation
NUM_EXAMPLES = 1
NUM_QUERIES_PER_EXAMPLE = 100
LAYERS_TO_TEST = ['first_layer', 'last_layer']
DATA_PATH = '../../data/attention_vectors_updated_long.jsonl'
OUTPUT_DIR = '../../results/approximation_evaluation/v2'
HEAD_DIM = 128

# ============================================================================
# END HYPERPARAMETERS
# ============================================================================


def evaluate_single_query(Q, K, V, query_pos, head_dim,
                          lsh_jungle, sh_index, cp_index):
    """
    Evaluate all 7 methods on one query.

    Args:
        lsh_jungle: LSHStructure for jungle sampling (prefix_sampling).
        sh_index: SimHashIndex for simhash_snis.
        cp_index: CrossPolytopeIndex for cross_polytope_snis.

    Returns:
        Dictionary with results for budget-controlled and SNIS methods.
    """
    q = Q[query_pos]
    valid_keys = K[:query_pos + 1]
    valid_values = V[:query_pos + 1]
    num_valid = len(valid_keys)

    # Ground truth
    gt_output, gt_logits, gt_weights, _ = compute_ground_truth_attention(
        q, K, V, query_pos, head_dim
    )

    results = {
        'budget_controlled': {},   # Full, TopK, Uniform, Oracle, Jungle
        'simhash_snis': None,      # SimHash SNIS (single config)
        'cp_snis': None,           # Cross-Polytope SNIS (single config)
    }

    # ================================================================
    # Budget-Controlled Methods
    # ================================================================

    # Build jungle LSH index
    lsh_jungle.build_index(valid_keys)

    for budget in BUDGETS:
        if budget > num_valid:
            continue

        # Full attention (for reference at this budget -- just ground truth)
        error_full = relative_l2_error(gt_output, gt_output)

        # Top-K
        output_topk, _ = topk_attention(q, valid_keys, valid_values, gt_logits, budget)
        error_topk = relative_l2_error(output_topk, gt_output)

        # Uniform Sampling
        output_uniform, _ = uniform_sampling(q, valid_keys, valid_values, gt_logits, budget)
        error_uniform = relative_l2_error(output_uniform, gt_output)

        # Oracle Sampling
        output_oracle, unique_budget = oracle_sampling(
            q, valid_keys, valid_values, gt_logits, gt_weights, budget
        )
        error_oracle = relative_l2_error(output_oracle, gt_output)

        # Jungle Sampling (prefix_sampling)
        output_jungle, _ = jungle_sampling(
            q, valid_keys, valid_values, gt_logits, head_dim,
            lsh_jungle, budget, GAMMA, TAU, MIN_DEPTH
        )
        error_jungle = relative_l2_error(output_jungle, gt_output)

        results['budget_controlled'][budget] = {
            'FullAttention': float(error_full),
            'TopK': float(error_topk),
            'Uniform': float(error_uniform),
            'Oracle': float(error_oracle),
            'JungleSampling': float(error_jungle),
        }

    # ================================================================
    # SimHash SNIS (non-budget-controlled, single config)
    # ================================================================
    sh_index.build_index(valid_keys)

    # Compute query hash
    q_sh = sh_index.batch_hash_queries(q[np.newaxis, :])[0]  # [L, K]
    kp = sh_index.key_codes[:num_valid, :SIMHASH_L, :SIMHASH_K]
    qp = q_sh[:SIMHASH_L, :SIMHASH_K]
    per_table = np.all(kp == qp[np.newaxis, :, :], axis=2)
    match_counts = np.sum(per_table, axis=1)
    retrieved_idx = np.where(match_counts >= SIMHASH_MIN_HITS)[0]

    if len(retrieved_idx) > 0:
        r_keys = valid_keys[retrieved_idx]
        r_values = valid_values[retrieved_idx]
        r_logits = gt_logits[retrieved_idx]
        q_norm = np.linalg.norm(q)
        k_norms = np.linalg.norm(r_keys, axis=1)
        cos_sims = np.clip(
            (r_keys @ q) / (q_norm * k_norms + 1e-8),
            -1.0 + 1e-8, 1.0 - 1e-8
        )
        thetas = np.arccos(cos_sims)
        p_table = SimHashIndex.collision_prob(thetas, SIMHASH_K)
        inc = _inclusion_prob(p_table, SIMHASH_L, SIMHASH_MIN_HITS)
        output_sh = _snis_estimator(r_logits, r_values, inc)
        error_sh = relative_l2_error(output_sh, gt_output)
        results['simhash_snis'] = {
            'budget': len(retrieved_idx),
            'error': float(error_sh),
            'K': SIMHASH_K, 'L': SIMHASH_L, 'min_hits': SIMHASH_MIN_HITS,
        }

    # ================================================================
    # Cross-Polytope SNIS (non-budget-controlled, single config)
    # ================================================================
    cp_index.build_index(valid_keys)

    q_cp = cp_index.batch_hash_queries(q[np.newaxis, :])[0]  # [L, k_cp]
    kp_cp = cp_index.key_codes[:num_valid, :CP_L, :CP_K]
    qp_cp = q_cp[:CP_L, :CP_K]
    per_table_cp = np.all(kp_cp == qp_cp[np.newaxis, :, :], axis=2)
    match_counts_cp = np.sum(per_table_cp, axis=1)
    retrieved_idx_cp = np.where(match_counts_cp >= CP_MIN_HITS)[0]

    if len(retrieved_idx_cp) > 0:
        r_keys_cp = valid_keys[retrieved_idx_cp]
        r_values_cp = valid_values[retrieved_idx_cp]
        r_logits_cp = gt_logits[retrieved_idx_cp]
        q_norm = np.linalg.norm(q)
        k_norms_cp = np.linalg.norm(r_keys_cp, axis=1)
        cos_sims_cp = np.clip(
            (r_keys_cp @ q) / (q_norm * k_norms_cp + 1e-8),
            -1.0 + 1e-8, 1.0 - 1e-8
        )
        thetas_cp = np.arccos(cos_sims_cp)
        p_table_cp = cp_index.collision_prob(thetas_cp, CP_K)
        inc_cp = _inclusion_prob(p_table_cp, CP_L, CP_MIN_HITS)
        output_cp = _snis_estimator(r_logits_cp, r_values_cp, inc_cp)
        error_cp = relative_l2_error(output_cp, gt_output)
        results['cp_snis'] = {
            'budget': len(retrieved_idx_cp),
            'error': float(error_cp),
            'k_cp': CP_K, 'L': CP_L, 'min_hits': CP_MIN_HITS,
        }

    return results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _softmax(x):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def _snis_estimator(logits, values, inclusion_probs):
    """Self-Normalized Importance Sampling estimator."""
    if len(logits) == 0:
        return np.zeros(HEAD_DIM)
    weighted_logits = logits - np.log(np.clip(inclusion_probs, 1e-12, None))
    weights = _softmax(weighted_logits)
    return weights @ values


def _inclusion_prob(p_table, L, min_hits):
    """P(key collides in >= min_hits out of L tables)."""
    if min_hits == 1:
        return np.clip(1.0 - np.power(1.0 - p_table, L), 1e-12, 1.0)
    elif min_hits == 2:
        p0 = np.power(1.0 - p_table, L)
        p1 = L * p_table * np.power(1.0 - p_table, L - 1)
        return np.clip(1.0 - p0 - p1, 1e-12, 1.0)
    elif min_hits == 3:
        p0 = np.power(1.0 - p_table, L)
        p1 = L * p_table * np.power(1.0 - p_table, L - 1)
        p2 = (L * (L - 1) / 2.0) * np.power(p_table, 2) * np.power(1.0 - p_table, L - 2)
        return np.clip(1.0 - p0 - p1 - p2, 1e-12, 1.0)
    else:
        raise ValueError(f"min_hits={min_hits} not supported")


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_results(all_results):
    """Aggregate results across queries."""
    budget_methods = ['FullAttention', 'TopK', 'Uniform', 'Oracle', 'JungleSampling']
    aggregated = {
        'budget_controlled': {method: {} for method in budget_methods},
        'simhash_snis': {'budgets': [], 'errors': []},
        'cp_snis': {'budgets': [], 'errors': []},
    }

    # Aggregate budget-controlled methods
    for query_results in all_results:
        for budget, method_errors in query_results['budget_controlled'].items():
            for method, error in method_errors.items():
                if budget not in aggregated['budget_controlled'][method]:
                    aggregated['budget_controlled'][method][budget] = []
                aggregated['budget_controlled'][method][budget].append(error)

    # Compute statistics for budget-controlled
    for method in aggregated['budget_controlled']:
        for budget in list(aggregated['budget_controlled'][method].keys()):
            errors = aggregated['budget_controlled'][method][budget]
            aggregated['budget_controlled'][method][budget] = {
                'mean': float(np.mean(errors)),
                'median': float(np.median(errors)),
                'std': float(np.std(errors)),
                'n': len(errors),
            }

    # Aggregate SNIS results
    for query_results in all_results:
        if query_results['simhash_snis'] is not None:
            aggregated['simhash_snis']['budgets'].append(
                query_results['simhash_snis']['budget'])
            aggregated['simhash_snis']['errors'].append(
                query_results['simhash_snis']['error'])
        if query_results['cp_snis'] is not None:
            aggregated['cp_snis']['budgets'].append(
                query_results['cp_snis']['budget'])
            aggregated['cp_snis']['errors'].append(
                query_results['cp_snis']['error'])

    # Compute SNIS statistics
    for snis_key in ['simhash_snis', 'cp_snis']:
        budgets = aggregated[snis_key]['budgets']
        errors = aggregated[snis_key]['errors']
        if len(budgets) > 0:
            aggregated[snis_key] = {
                'mean_budget': float(np.mean(budgets)),
                'std_budget': float(np.std(budgets)),
                'mean_error': float(np.mean(errors)),
                'median_error': float(np.median(errors)),
                'std_error': float(np.std(errors)),
                'n': len(errors),
            }
        else:
            aggregated[snis_key] = {
                'mean_budget': 0, 'std_budget': 0,
                'mean_error': float('nan'), 'median_error': float('nan'),
                'std_error': 0, 'n': 0,
            }

    return aggregated


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(aggregated_dict, output_dir):
    """Generate comparison plots: error-vs-budget for budget-controlled methods,
    scatter points for SNIS methods. Separate mean and median plots per layer."""

    setup_style()

    layer_titles = {
        'first_layer': 'First Layer (Layer 0)',
        'last_layer': 'Last Layer (Layer 31)',
    }

    # Method config for budget-controlled lines
    methods_config = [
        ('TopK',            'o', 'Top-K',             '#d62728'),   # Red
        ('Uniform',         's', 'Uniform Sampling',   '#ff7f0e'),   # Orange
        ('Oracle',          '^', 'Oracle',             '#2ca02c'),   # Green
        ('JungleSampling',  'D', 'Jungle Sampling',    '#9467bd'),   # Purple
    ]

    for layer_name, agg_data in aggregated_dict.items():
        layer_title = layer_titles.get(layer_name,
                                       layer_name.replace('_', ' ').title())
        sample_method = 'Uniform'
        n_queries = 0
        if sample_method in agg_data['budget_controlled']:
            first_budget = next(iter(agg_data['budget_controlled'][sample_method].values()), None)
            if first_budget is not None:
                n_queries = first_budget['n']

        max_budget = max(BUDGETS)
        min_budget = min(BUDGETS)

        for stat_type in ['mean', 'median']:
            fig, ax = plt.subplots(figsize=(14, 8))

            # Plot budget-controlled method curves
            for method_name, marker, label, color in methods_config:
                bc = agg_data['budget_controlled'].get(method_name, {})
                if len(bc) == 0:
                    continue

                budgets = sorted(bc.keys())
                vals = [bc[b][stat_type] for b in budgets]
                stds = [bc[b]['std'] for b in budgets]

                budgets_arr = np.array(budgets)
                vals_arr = np.array(vals)
                stds_arr = np.array(stds)

                upper = vals_arr + stds_arr
                lower = np.maximum(vals_arr - stds_arr, 1e-6)

                ax.fill_between(budgets_arr, lower, upper,
                                color=color, alpha=0.2, linewidth=0,
                                zorder=1, label='_nolegend_')
                ax.plot(budgets_arr, vals_arr, marker=marker, linestyle='-',
                        label=label, color=color, linewidth=2.5,
                        markersize=6, alpha=0.95, zorder=3)

            # Plot SimHash-SNIS scatter
            sh = agg_data.get('simhash_snis', {})
            if sh.get('n', 0) > 0:
                sh_budget = sh['mean_budget']
                sh_error = sh['mean_error'] if stat_type == 'mean' else sh.get('median_error', sh['mean_error'])
                if min_budget <= sh_budget <= max_budget:
                    ax.scatter([sh_budget], [sh_error], marker='x', s=160,
                               linewidths=3, color='#1f77b4', alpha=0.9,
                               label=f'SimHash-SNIS (K={SIMHASH_K}, L={SIMHASH_L})',
                               zorder=5)

            # Plot Cross-Polytope SNIS scatter
            cp = agg_data.get('cp_snis', {})
            if cp.get('n', 0) > 0:
                cp_budget = cp['mean_budget']
                cp_error = cp['mean_error'] if stat_type == 'mean' else cp.get('median_error', cp['mean_error'])
                if min_budget <= cp_budget <= max_budget:
                    ax.scatter([cp_budget], [cp_error], marker='P', s=160,
                               linewidths=2, color='#e11d48', alpha=0.9,
                               label=f'CrossPoly-SNIS (k={CP_K}, L={CP_L})',
                               zorder=5)

            # Formatting
            ax.set_xlabel('Budget (Number of Keys)', fontsize=13,
                          fontweight='bold', family='sans-serif')
            ax.set_ylabel(f'Relative L2 Error ({stat_type.capitalize()})',
                          fontsize=13, fontweight='bold', family='sans-serif')
            ax.set_title(
                f'Attention Approximation Methods - {stat_type.capitalize()}\n'
                f'{layer_title} (averaged over {n_queries} queries)',
                fontsize=14, fontweight='bold', family='sans-serif')
            ax.set_yscale('log')

            format_log_yaxis(ax)

            ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, which='major')
            ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.3, which='minor')
            ax.legend(fontsize=10, loc='best', ncol=2,
                      framealpha=0.95, fancybox=True, shadow=True)

            plt.tight_layout()
            plot_path = output_dir / f'{layer_name}_{stat_type}.png'
            save_figure(fig, plot_path, dpi=150)
            print(f"  Generated: {plot_path}")
            plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the full comparison of all 7 algorithms."""

    print("=" * 70)
    print("ATTENTION APPROXIMATION COMPARISON - ALL 7 ALGORITHMS")
    print("=" * 70)
    print(f"\nBudget-controlled methods:")
    print(f"  1. Full Attention (ground truth)")
    print(f"  2. Top-K (biased baseline)")
    print(f"  3. Uniform Sampling")
    print(f"  4. Oracle Sampling (gold standard)")
    print(f"  5. Jungle Sampling (ours): L={L}, gamma={GAMMA}, tau={TAU}, min_depth={MIN_DEPTH}")
    print(f"\nNon-budget-controlled LSH-SNIS methods:")
    print(f"  6. SimHash-SNIS:  K={SIMHASH_K}, L={SIMHASH_L}, min_hits={SIMHASH_MIN_HITS}")
    print(f"  7. CrossPoly-SNIS: k={CP_K}, L={CP_L}, min_hits={CP_MIN_HITS}")
    print(f"\nConfiguration:")
    print(f"  Budgets: {len(BUDGETS)} points from {BUDGETS[0]} to {BUDGETS[-1]}")
    print(f"  Examples: {NUM_EXAMPLES}")
    print(f"  Queries per example: {NUM_QUERIES_PER_EXAMPLE}")
    print(f"  Layers: {LAYERS_TO_TEST}")

    np.random.seed(SEED)

    # Load data line-by-line
    print(f"\nLoading: {DATA_PATH}")
    load_start = time.time()

    examples = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            examples.append(json.loads(line))

    load_time = time.time() - load_start
    print(f"Loaded {len(examples)} examples in {load_time:.1f}s")

    examples = examples[:NUM_EXAMPLES]
    print(f"Using {len(examples)} examples")

    # Create LSH structures
    head_dim = HEAD_DIM

    # Jungle Sampling LSH forest
    lsh_jungle = LSHStructure(L, K_MAX, head_dim, center_keys=True, seed=SEED)

    # SimHash index (shared max-size, sliced per config)
    sh_index = SimHashIndex(SIMHASH_L, SIMHASH_K, head_dim,
                            center_keys=True, seed=SEED + 100)

    # Cross-Polytope index
    cp_index = CrossPolytopeIndex(CP_L, CP_K, head_dim,
                                  center_keys=True, seed=SEED + 999)

    print(f"Created LSH structures:")
    print(f"  Jungle: L={L}, K_MAX={K_MAX}")
    print(f"  SimHash: L={SIMHASH_L}, K={SIMHASH_K}")
    print(f"  CrossPoly: L={CP_L}, k={CP_K}")

    # Storage
    all_results = {
        'metadata': {
            'methods': {
                'budget_controlled': ['FullAttention', 'TopK', 'Uniform',
                                      'Oracle', 'JungleSampling'],
                'snis': ['SimHash-SNIS', 'CrossPoly-SNIS'],
            },
            'jungle_params': {'L': L, 'K_MAX': K_MAX,
                              'gamma': GAMMA, 'tau': TAU, 'min_depth': MIN_DEPTH},
            'simhash_params': {'K': SIMHASH_K, 'L': SIMHASH_L,
                               'min_hits': SIMHASH_MIN_HITS},
            'cp_params': {'k_cp': CP_K, 'L': CP_L, 'min_hits': CP_MIN_HITS},
            'budgets': BUDGETS,
            'num_examples': len(examples),
            'num_queries_per_example': NUM_QUERIES_PER_EXAMPLE,
            'layers': LAYERS_TO_TEST,
            'head_dim': HEAD_DIM,
            'seed': SEED,
            'timestamp': datetime.now().isoformat(),
        },
        'results_by_layer': {layer: [] for layer in LAYERS_TO_TEST},
    }

    # Evaluate
    total_queries = len(examples) * NUM_QUERIES_PER_EXAMPLE * len(LAYERS_TO_TEST)
    print(f"\nEvaluating {total_queries} queries...")
    print(f"Estimated time: ~{total_queries * 2 / 60:.0f}-{total_queries * 3 / 60:.0f} minutes\n")

    eval_start = time.time()

    for ex_idx, example in enumerate(examples):
        print(f"\nExample {ex_idx + 1}/{len(examples)}: "
              f"{example.get('domain', '?')[:50]}")

        for layer_name in LAYERS_TO_TEST:
            layer_start = time.time()

            Q = np.array(example[layer_name]['Q'], dtype=np.float32)
            K_mat = np.array(example[layer_name]['K'], dtype=np.float32)
            V = np.array(example[layer_name]['V'], dtype=np.float32)
            seq_len = Q.shape[0]

            query_positions = range(seq_len - NUM_QUERIES_PER_EXAMPLE, seq_len)

            for query_pos in tqdm(query_positions,
                                  desc=f"  {layer_name}", leave=False):
                query_results = evaluate_single_query(
                    Q, K_mat, V, query_pos, head_dim,
                    lsh_jungle, sh_index, cp_index
                )
                all_results['results_by_layer'][layer_name].append(query_results)

            layer_time = time.time() - layer_start
            print(f"  {layer_name}: {len(query_positions)} queries ({layer_time:.1f}s)")

    eval_time = time.time() - eval_start
    all_results['metadata']['eval_time_seconds'] = eval_time

    # Save results
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("SAVING RESULTS")
    print(f"{'=' * 70}")
    print(f"Total time: {eval_time:.1f}s ({eval_time / 60:.1f} minutes)")

    # Full results JSON
    full_json = output_dir / 'full_results.json'
    with open(full_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results: {full_json} ({full_json.stat().st_size / (1024 * 1024):.1f} MB)")

    # Aggregate
    print(f"\nAggregating...")
    aggregated = {}
    for layer in LAYERS_TO_TEST:
        aggregated[layer] = aggregate_results(
            all_results['results_by_layer'][layer])

    # JSON-safe aggregated output
    aggregated_json_safe = {}
    for layer in LAYERS_TO_TEST:
        aggregated_json_safe[layer] = {
            'budget_controlled': aggregated[layer]['budget_controlled'],
            'simhash_snis': aggregated[layer]['simhash_snis'],
            'cp_snis': aggregated[layer]['cp_snis'],
        }

    agg_json = output_dir / 'aggregated.json'
    with open(agg_json, 'w') as f:
        json.dump({'aggregated': aggregated_json_safe,
                   'metadata': all_results['metadata']}, f, indent=2)
    print(f"Aggregated: {agg_json}")

    # Plots
    print(f"\nGenerating plots...")
    plot_results(aggregated, output_dir)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY (Budget = {BUDGETS[len(BUDGETS) // 2]})")
    print(f"{'=' * 70}")

    ref_budget = BUDGETS[len(BUDGETS) // 2]
    for layer in LAYERS_TO_TEST:
        print(f"\n{layer.upper()}:")
        print(f"  Budget-controlled (budget={ref_budget}):")
        for method in ['Oracle', 'JungleSampling', 'Uniform', 'TopK']:
            bc = aggregated[layer]['budget_controlled'].get(method, {})
            if ref_budget in bc:
                stats = bc[ref_budget]
                print(f"    {method:18s}: {stats['mean']:.4f} +/- {stats['std']:.4f} (n={stats['n']})")

        sh = aggregated[layer]['simhash_snis']
        if sh['n'] > 0:
            print(f"\n  SimHash-SNIS (K={SIMHASH_K}, L={SIMHASH_L}, h>={SIMHASH_MIN_HITS}):")
            print(f"    budget={sh['mean_budget']:.0f} +/- {sh['std_budget']:.0f}, "
                  f"error={sh['mean_error']:.4f} +/- {sh['std_error']:.4f}")

        cp = aggregated[layer]['cp_snis']
        if cp['n'] > 0:
            print(f"  CrossPoly-SNIS (k={CP_K}, L={CP_L}, h>={CP_MIN_HITS}):")
            print(f"    budget={cp['mean_budget']:.0f} +/- {cp['std_budget']:.0f}, "
                  f"error={cp['mean_error']:.4f} +/- {cp['std_error']:.4f}")

    print(f"\n{'=' * 70}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nFiles in {OUTPUT_DIR}:")
    print(f"  - full_results.json")
    print(f"  - aggregated.json")
    for layer in LAYERS_TO_TEST:
        print(f"  - {layer}_mean.png")
        print(f"  - {layer}_median.png")


if __name__ == "__main__":
    main()
