#!/usr/bin/env python3
"""
Clean Attention Approximation Comparison

Compares 4 methods with easy-to-modify hyperparameters at the top.
All methods run once per query (single sampling, fair comparison).

Results saved to: results/approximation_evaluation/v2/
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time

import utils
import methods

# ============================================================================
# HYPERPARAMETERS - MODIFY HERE
# ============================================================================

# LSH Configuration
L = 50  # Number of hash tables/trees (for prefix_sampling)
K_MAX = 30 
SEED = 42  # Random seed

# LSH-SNIS Sweep (budget determined by K and L combinations)
LSH_K_VALUES = [5, 6, 7, 8, 9, 10]  # Depth values to test
LSH_L_VALUES = [10,15, 20, 25, 30, 40]  # Number of tables to test (more L → higher budgets)
LSH_MIN_HITS = 2  # Minimum table matches (MagicPIG uses 2 for selectivity)

# prefix_sampling Parameters
GAMMA = 1.0  # Bucket size penalty (1.0 = linear: 1/bucket_size)
TAU = 0.0  # Smoothing term (0.0 = no smoothing)
MIN_DEPTH = 5  # Minimum depth threshold (0 = no filtering, only sample from keys with max_depth >= MIN_DEPTH)

# Budget Range (for methods with budget control)
BUDGETS = list(range(20, 220, 20))  # [50, 100, 150, ..., 1000]

# Evaluation Scale
NUM_EXAMPLES = 1  # Number of examples to test (out of 11 available)
NUM_QUERIES_PER_EXAMPLE = 100  # Last N queries per example
LAYERS_TO_TEST = ['first_layer', 'last_layer',]  # all layers is first layer and last layer

# Data Path
DATA_PATH = '../../data/attention_vectors_updated_long.jsonl'

# Output
OUTPUT_DIR = '../../results/approximation_evaluation/v2'

# ============================================================================
# END HYPERPARAMETERS
# ============================================================================


def evaluate_single_query(Q, K, V, query_pos, head_dim, lsh_union, lsh_snis_dict):
    """
    Evaluate all methods on one query.
    
    Args:
        lsh_union: LSH structure for prefix_sampling (L=25)
        lsh_snis_dict: Dictionary of LSH structures for LSH-SNIS {L_value: lsh_structure}
    """
    q = Q[query_pos]
    valid_keys = K[:query_pos + 1]
    valid_values = V[:query_pos + 1]
    num_valid = len(valid_keys)
    
    # Ground truth
    gt_output, gt_logits, gt_weights, _ = utils.compute_ground_truth_attention(
        q, K, V, query_pos, head_dim
    )
    
    results = {
        'budget_controlled': {},  # TopK, Naive, Oracle, prefix_sampling
        'lsh_snis': []  # LSH-SNIS with different (K, L) combinations
    }
    
    # ================================================================
    # Budget-Controlled Methods
    # ================================================================
    
    # Build LSH for prefix_sampling
    lsh_union.build_index(valid_keys)
    
    for budget in BUDGETS:
        if budget > num_valid:
            continue
        
        # Top-K
        output_topk, _ = methods.topk_approximation(q, valid_keys, valid_values, gt_logits, budget)
        error_topk = utils.relative_l2_error(output_topk, gt_output)
        
        # Naive
        output_naive, _ = methods.naive_sampling(q, valid_keys, valid_values, gt_logits, budget)
        error_naive = utils.relative_l2_error(output_naive, gt_output)
        
        # Oracle
        output_oracle, unique_budget = methods.oracle_sampling(
            q, valid_keys, valid_values, gt_logits, gt_weights, budget
        )
        error_oracle = utils.relative_l2_error(output_oracle, gt_output)
        
        # prefix_sampling
        output_prefix, _ = methods.prefix_sampling(
            q, valid_keys, valid_values, gt_logits, head_dim,
            lsh_union, budget, GAMMA, TAU, MIN_DEPTH
        )
        error_prefix = utils.relative_l2_error(output_prefix, gt_output)
        
        results['budget_controlled'][budget] = {
            'TopK': float(error_topk),
            'Naive': float(error_naive),
            'Oracle': float(error_oracle),
            'prefix_sampling': float(error_prefix)
        }
    
    # ================================================================
    # LSH-SNIS (budget determined by both K and L)
    # ================================================================
    
    for L_val in LSH_L_VALUES:
        lsh_for_L = lsh_snis_dict[L_val]
        lsh_for_L.build_index(valid_keys)
        
        for K_val in LSH_K_VALUES:
            # Run LSH-SNIS at this (L, K) combination
            output_lsh, retrieved_budget = methods.lsh_snis(
                q, valid_keys, valid_values, gt_logits, head_dim,
                lsh_for_L, K_val, LSH_MIN_HITS
            )
            
            if retrieved_budget > 0:
                error_lsh = utils.relative_l2_error(output_lsh, gt_output)
                
                results['lsh_snis'].append({
                    'L': L_val,
                    'K': K_val,
                    'budget': retrieved_budget,
                    'error': float(error_lsh)
                })
    
    return results


def aggregate_results(all_results):
    """Aggregate across queries."""
    aggregated = {
        'budget_controlled': {method: {} for method in ['TopK', 'Naive', 'Oracle', 'prefix_sampling']},
        'lsh_snis': {}  # Will be {(L, K): {budget: stats}}
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
                'n': len(errors)
            }
    
    # Aggregate LSH-SNIS results by (L, K) combination
    # For each (L, K): collect all (budget, error) pairs and compute averages
    for query_results in all_results:
        for lsh_result in query_results['lsh_snis']:
            L_val = lsh_result['L']
            K_val = lsh_result['K']
            budget = lsh_result['budget']
            error = lsh_result['error']
            
            key = (L_val, K_val)
            if key not in aggregated['lsh_snis']:
                aggregated['lsh_snis'][key] = {'budgets': [], 'errors': []}
            
            aggregated['lsh_snis'][key]['budgets'].append(budget)
            aggregated['lsh_snis'][key]['errors'].append(error)
    
    # Compute average budget and average error for each (L, K)
    for (L_val, K_val) in aggregated['lsh_snis']:
        budgets = aggregated['lsh_snis'][(L_val, K_val)]['budgets']
        errors = aggregated['lsh_snis'][(L_val, K_val)]['errors']
        
        aggregated['lsh_snis'][(L_val, K_val)] = {
            'L': L_val,
            'K': K_val,
            'mean_budget': float(np.mean(budgets)),
            'std_budget': float(np.std(budgets)),
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'n': len(errors)
        }
    
    return aggregated


def plot_results(aggregated_dict, output_dir):
    """Generate comparison plots - separate plots for mean and median.
    
    Args:
        aggregated_dict: Dictionary mapping layer names to aggregated data
        output_dir: Output directory for plots
    """
    
    from matplotlib.ticker import LogLocator, FuncFormatter
    
    # Set all text to sans-serif font family (applies globally)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Verdana', 'Liberation Sans']
    
    # Map layer names to titles
    layer_titles = {
        'first_layer': 'First Layer (Layer 0)',
        'last_layer': 'Last Layer (Layer 31)'
    }
    
    # Method configuration
    methods_config = [
        ('TopK', 'o', 'Top-K', '#d62728'),  # Red
        ('Naive', 's', 'Uniform Sampling', '#ff7f0e'),  # Orange
        ('Oracle', '^', 'Oracle', '#2ca02c'),  # Green
        ('prefix_sampling', 'D', 'LSH-Forest-SNIS', '#9467bd')  # Purple
    ]
    
    for layer_name, agg_data in aggregated_dict.items():
        layer_title = layer_titles.get(layer_name, layer_name.replace('_', ' ').title())
        n_queries = list(agg_data['budget_controlled']['Naive'].values())[0]['n'] if 'Naive' in agg_data['budget_controlled'] else 0
        
        max_budget = max(BUDGETS)
        min_budget = min(BUDGETS)
        
        # Collect LSH-SNIS data
        lsh_budgets = []
        lsh_errors = []
        for (L_val, K_val) in sorted(agg_data['lsh_snis'].keys()):
            stats = agg_data['lsh_snis'][(L_val, K_val)]
            mean_budget = stats['mean_budget']
            mean_error = stats['mean_error']
            if min_budget <= mean_budget <= max_budget:
                lsh_budgets.append(mean_budget)
                lsh_errors.append(mean_error)
        
        # ========================================================================
        # PLOT 1: MEAN with shaded variance
        # ========================================================================
        fig_mean, ax_mean = plt.subplots(figsize=(14, 8))
        
        for method_name, marker, label, color in methods_config:
            if method_name not in agg_data['budget_controlled'] or len(agg_data['budget_controlled'][method_name]) == 0:
                continue
            
            budgets = sorted(agg_data['budget_controlled'][method_name].keys())
            means = [agg_data['budget_controlled'][method_name][b]['mean'] for b in budgets]
            stds = [agg_data['budget_controlled'][method_name][b]['std'] for b in budgets]
            
            budgets_arr = np.array(budgets)
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            
            # Shaded area: mean ± std (variance range)
            upper_bound = means_arr + stds_arr
            lower_bound = np.maximum(means_arr - stds_arr, 1e-6)  # Prevent negative values on log scale
            
            # Plot shaded variance area
            ax_mean.fill_between(budgets_arr, lower_bound, upper_bound, 
                                color=color, alpha=0.2, linewidth=0, zorder=1, label='_nolegend_')
            
            # Plot mean line
            ax_mean.plot(budgets_arr, means_arr, marker=marker, linestyle='-', 
                        label=label, color=color, linewidth=2.5, 
                        markersize=6, alpha=0.95, zorder=3)
        
        # Plot LSH-SNIS
        if lsh_budgets:
            lsh_color = '#1f77b4'
            ax_mean.scatter(lsh_budgets, lsh_errors, marker='x', s=120, linewidths=2.5,
                           color=lsh_color, alpha=0.8, label='LSH-SNIS', zorder=5)
            
            if len(lsh_budgets) >= 3:
                log_budgets = np.log(lsh_budgets)
                log_errors = np.log(lsh_errors)
                coeffs = np.polyfit(log_budgets, log_errors, 1)
                budget_range = np.linspace(min(lsh_budgets), max(lsh_budgets), 100)
                fitted_errors = np.exp(np.poly1d(coeffs)(np.log(budget_range)))
                ax_mean.plot(budget_range, fitted_errors, '--', color=lsh_color, linewidth=2, alpha=0.6,
                           label=f'LSH-SNIS fit (error ∝ budget^{coeffs[0]:.2f})')
        
        # Formatting for mean plot
        ax_mean.set_xlabel('Budget (Number of Keys)', fontsize=13, fontweight='bold', family='sans-serif')
        ax_mean.set_ylabel('Relative L2 Error (Mean)', fontsize=13, fontweight='bold', family='sans-serif')
        ax_mean.set_title(f'Attention Approximation Methods - Mean\n{layer_title} (averaged over {n_queries} queries)',
                          fontsize=14, fontweight='bold', family='sans-serif')
        ax_mean.set_yscale('log')
        
        # Set y-axis range
        all_errors_mean = []
        for method_name in agg_data['budget_controlled']:
            for budget, stats in agg_data['budget_controlled'][method_name].items():
                all_errors_mean.append(stats['mean'])
                all_errors_mean.append(stats['mean'] + stats['std'])
                all_errors_mean.append(max(stats['mean'] - stats['std'], 1e-6))
        if lsh_errors:
            all_errors_mean.extend(lsh_errors)
        
        if all_errors_mean:
            min_error = min(all_errors_mean)
            max_error = max(all_errors_mean)
            y_min = 10 ** (np.floor(np.log10(min_error)) - 0.1)
            y_max = 10 ** (np.ceil(np.log10(max_error)) + 0.1)
            ax_mean.set_ylim(y_min, y_max)
        
        ax_mean.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
        ax_mean.yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=200))
        ax_mean.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}' if y < 1 else f'{y:.1f}'))
        ax_mean.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: ''))
        
        for label in ax_mean.get_xticklabels():
            label.set_family('sans-serif')
        for label in ax_mean.get_yticklabels():
            label.set_family('sans-serif')
        
        ax_mean.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, which='major')
        ax_mean.grid(True, alpha=0.15, linestyle='--', linewidth=0.3, which='minor')
        ax_mean.legend(fontsize=10, loc='best', ncol=2, framealpha=0.95, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plot_path_mean = output_dir / f'{layer_name}_mean.png'
        plt.savefig(plot_path_mean, dpi=150, bbox_inches='tight')
        print(f"  ✓ Generated: {plot_path_mean}")
        plt.close()
        
        # ========================================================================
        # PLOT 2: MEDIAN with shaded variance
        # ========================================================================
        fig_median, ax_median = plt.subplots(figsize=(14, 8))
        
        for method_name, marker, label, color in methods_config:
            if method_name not in agg_data['budget_controlled'] or len(agg_data['budget_controlled'][method_name]) == 0:
                continue
            
            budgets = sorted(agg_data['budget_controlled'][method_name].keys())
            medians = [agg_data['budget_controlled'][method_name][b]['median'] for b in budgets]
            stds = [agg_data['budget_controlled'][method_name][b]['std'] for b in budgets]
            
            budgets_arr = np.array(budgets)
            medians_arr = np.array(medians)
            stds_arr = np.array(stds)
            
            # Shaded area: median ± std (variance range)
            upper_bound = medians_arr + stds_arr
            lower_bound = np.maximum(medians_arr - stds_arr, 1e-6)
            
            # Plot shaded variance area
            ax_median.fill_between(budgets_arr, lower_bound, upper_bound, 
                                  color=color, alpha=0.2, linewidth=0, zorder=1, label='_nolegend_')
            
            # Plot median line
            ax_median.plot(budgets_arr, medians_arr, marker=marker, linestyle='-', 
                          label=label, color=color, linewidth=2.5, 
                          markersize=6, alpha=0.95, zorder=3)
        
        # Plot LSH-SNIS (using mean_error as approximation for median)
        if lsh_budgets:
            lsh_color = '#1f77b4'
            ax_median.scatter(lsh_budgets, lsh_errors, marker='x', s=120, linewidths=2.5,
                             color=lsh_color, alpha=0.8, label='LSH-SNIS', zorder=5)
            
            if len(lsh_budgets) >= 3:
                log_budgets = np.log(lsh_budgets)
                log_errors = np.log(lsh_errors)
                coeffs = np.polyfit(log_budgets, log_errors, 1)
                budget_range = np.linspace(min(lsh_budgets), max(lsh_budgets), 100)
                fitted_errors = np.exp(np.poly1d(coeffs)(np.log(budget_range)))
                ax_median.plot(budget_range, fitted_errors, '--', color=lsh_color, linewidth=2, alpha=0.6,
                             label=f'LSH-SNIS fit (error ∝ budget^{coeffs[0]:.2f})')
        
        # Formatting for median plot
        ax_median.set_xlabel('Budget (Number of Keys)', fontsize=13, fontweight='bold', family='sans-serif')
        ax_median.set_ylabel('Relative L2 Error (Median)', fontsize=13, fontweight='bold', family='sans-serif')
        ax_median.set_title(f'Attention Approximation Methods - Median\n{layer_title} (averaged over {n_queries} queries)',
                           fontsize=14, fontweight='bold', family='sans-serif')
        ax_median.set_yscale('log')
        
        # Set y-axis range
        all_errors_median = []
        for method_name in agg_data['budget_controlled']:
            for budget, stats in agg_data['budget_controlled'][method_name].items():
                all_errors_median.append(stats['median'])
                all_errors_median.append(stats['median'] + stats['std'])
                all_errors_median.append(max(stats['median'] - stats['std'], 1e-6))
        if lsh_errors:
            all_errors_median.extend(lsh_errors)
        
        if all_errors_median:
            min_error = min(all_errors_median)
            max_error = max(all_errors_median)
            y_min = 10 ** (np.floor(np.log10(min_error)) - 0.1)
            y_max = 10 ** (np.ceil(np.log10(max_error)) + 0.1)
            ax_median.set_ylim(y_min, y_max)
        
        ax_median.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
        ax_median.yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=200))
        ax_median.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}' if y < 1 else f'{y:.1f}'))
        ax_median.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: ''))
        
        for label in ax_median.get_xticklabels():
            label.set_family('sans-serif')
        for label in ax_median.get_yticklabels():
            label.set_family('sans-serif')
        
        ax_median.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, which='major')
        ax_median.grid(True, alpha=0.15, linestyle='--', linewidth=0.3, which='minor')
        ax_median.legend(fontsize=10, loc='best', ncol=2, framealpha=0.95, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plot_path_median = output_dir / f'{layer_name}_median.png'
        plt.savefig(plot_path_median, dpi=150, bbox_inches='tight')
        print(f"  ✓ Generated: {plot_path_median}")
        plt.close()


def main():
    """Run comparison."""
    
    print("="*70)
    print("ATTENTION APPROXIMATION COMPARISON (v2)")
    print("="*70)
    print(f"\nMethods:")
    print(f"  1. Top-K (biased baseline)")
    print(f"  2. Naive Sampling (uniform)")
    print(f"  3. Oracle Sampling (gold standard)")
    print(f"  4. LSH-SNIS (MagicPIG): K={LSH_K_VALUES}, L={LSH_L_VALUES}, min_hits={LSH_MIN_HITS}")
    print(f"  5. prefix_sampling (ours): L={L}, γ={GAMMA}, τ={TAU}, min_depth={MIN_DEPTH}")
    print(f"\nNote: LSH-SNIS budget varies by (K, L) combination (not controlled)")
    print(f"      Total LSH-SNIS combinations: {len(LSH_K_VALUES) * len(LSH_L_VALUES)}")
    print(f"\nConfiguration:")
    print(f"  Budgets: {len(BUDGETS)} points from {BUDGETS[0]} to {BUDGETS[-1]}")
    print(f"  Examples: {NUM_EXAMPLES}")
    print(f"  Queries per example: {NUM_QUERIES_PER_EXAMPLE}")
    print(f"  Single run per method (fair comparison)")
    
    np.random.seed(SEED)
    
    # Load data
    print(f"\nLoading: {DATA_PATH}")
    load_start = time.time()
    
    examples = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    load_time = time.time() - load_start
    print(f"✓ Loaded {len(examples)} examples in {load_time:.1f}s")
    
    # Use first N examples
    examples = examples[:NUM_EXAMPLES]
    print(f"✓ Using {len(examples)} examples")
    
    # Create LSH structures
    head_dim = 128
    
    # LSH for prefix_sampling (L=25, default)
    lsh_union = utils.LSHStructure(L, K_MAX, head_dim, center_keys=True, seed=SEED)
    
    # Multiple LSH structures for LSH-SNIS (different L values)
    lsh_snis_dict = {}
    for L_val in LSH_L_VALUES:
        lsh_snis_dict[L_val] = utils.LSHStructure(
            L_val, max(LSH_K_VALUES), head_dim, center_keys=True, seed=SEED + L_val
        )
    
    print(f"✓ Created LSH structures: prefix_sampling (L={L}) + LSH-SNIS ({len(LSH_L_VALUES)} L values × {len(LSH_K_VALUES)} K values = {len(LSH_L_VALUES)*len(LSH_K_VALUES)} combinations)")
    
    # Storage
    all_results = {
        'metadata': {
            'methods': {
                'budget_controlled': ['TopK', 'Naive', 'Oracle', 'prefix_sampling'],
                'lsh_snis': f'K={LSH_K_VALUES}, L={LSH_L_VALUES}'
            },
            'prefix_sampling': {'L': L, 'gamma': GAMMA, 'tau': TAU, 'min_depth': MIN_DEPTH},
            'lsh_snis_sweep': {'K_values': LSH_K_VALUES, 'L_values': LSH_L_VALUES, 'min_hits': LSH_MIN_HITS},
            'K_max': K_MAX,
            'budgets': BUDGETS,
            'num_examples': len(examples),
            'num_queries_per_example': NUM_QUERIES_PER_EXAMPLE,
            'layers': LAYERS_TO_TEST,
            'single_run': True,
            'timestamp': datetime.now().isoformat()
        },
        'results_by_layer': {layer: [] for layer in LAYERS_TO_TEST}
    }
    
    # Evaluate
    total_queries = len(examples) * NUM_QUERIES_PER_EXAMPLE * len(LAYERS_TO_TEST)
    print(f"\nEvaluating {total_queries} queries...")
    print(f"Estimated time: ~{total_queries * 2 / 60:.0f}-{total_queries * 3 / 60:.0f} minutes\n")
    
    eval_start = time.time()
    
    for ex_idx, example in enumerate(examples):
        print(f"\nExample {ex_idx+1}/{len(examples)}: {example['domain'][:50]}")
        
        for layer_name in LAYERS_TO_TEST:
            layer_start = time.time()
            
            # Load data
            Q = np.array(example[layer_name]['Q'], dtype=np.float32)
            K = np.array(example[layer_name]['K'], dtype=np.float32)
            V = np.array(example[layer_name]['V'], dtype=np.float32)
            seq_len = Q.shape[0]
            
            # Last N queries
            query_positions = range(seq_len - NUM_QUERIES_PER_EXAMPLE, seq_len)
            
            for query_pos in tqdm(query_positions, desc=f"  {layer_name}", leave=False):
                query_results = evaluate_single_query(Q, K, V, query_pos, head_dim, lsh_union, lsh_snis_dict)
                all_results['results_by_layer'][layer_name].append(query_results)
            
            layer_time = time.time() - layer_start
            print(f"  {layer_name}: ✓ {len(query_positions)} queries ({layer_time:.1f}s)")
    
    eval_time = time.time() - eval_start
    all_results['metadata']['eval_time_seconds'] = eval_time
    
    # Save results
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    print(f"Total time: {eval_time:.1f}s ({eval_time/60:.1f} minutes)")
    
    # Full results
    full_json = output_dir / 'full_results.json'
    with open(full_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Full results: {full_json} ({full_json.stat().st_size / (1024*1024):.1f} MB)")
    
    # Aggregate
    print(f"\nAggregating...")
    aggregated = {}
    for layer in LAYERS_TO_TEST:
        aggregated[layer] = aggregate_results(all_results['results_by_layer'][layer])
    
    # Convert tuple keys to strings for JSON serialization
    aggregated_json_safe = {}
    for layer in LAYERS_TO_TEST:
        aggregated_json_safe[layer] = {
            'budget_controlled': aggregated[layer]['budget_controlled'],
            'lsh_snis': {}
        }
        # Convert (L, K) tuples to "L10_K6" strings
        for (L_val, K_val), data in aggregated[layer]['lsh_snis'].items():
            key_str = f"L{L_val}_K{K_val}"
            aggregated_json_safe[layer]['lsh_snis'][key_str] = data
    
    agg_json = output_dir / 'aggregated.json'
    with open(agg_json, 'w') as f:
        json.dump({'aggregated': aggregated_json_safe, 'metadata': all_results['metadata']}, f, indent=2)
    print(f"✓ Aggregated: {agg_json}")
    
    # Plots
    print(f"\nGenerating plots...")
    plot_results(aggregated, output_dir)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY (Budget = 500)")
    print(f"{'='*70}")
    
    for layer in LAYERS_TO_TEST:
        print(f"\n{layer.upper()}:")
        
        # Budget-controlled methods
        for method in ['Oracle', 'prefix_sampling', 'Naive', 'TopK']:
            if method in aggregated[layer]['budget_controlled'] and 500 in aggregated[layer]['budget_controlled'][method]:
                stats = aggregated[layer]['budget_controlled'][method][500]
                print(f"  {method:15s}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['n']})")
        
        # LSH-SNIS (average budget and error for each (L, K) combination)
        print(f"\n  LSH-SNIS (budget varies by K and L, min_hits={LSH_MIN_HITS}):")
        
        # Sort by L then K
        for (L_val, K_val) in sorted(aggregated[layer]['lsh_snis'].keys()):
            stats = aggregated[layer]['lsh_snis'][(L_val, K_val)]
            print(f"    L={L_val:2d}, K={K_val:2d}: budget={stats['mean_budget']:6.1f} ± {stats['std_budget']:5.1f}, "
                  f"error={stats['mean_error']:.4f} ± {stats['std_error']:.4f}")
    
    print(f"\n{'='*70}")
    print("✅ EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nFiles in {OUTPUT_DIR}:")
    print(f"  - full_results.json")
    print(f"  - aggregated.json")
    for layer in LAYERS_TO_TEST:
        print(f"  - {layer}_mean.png")
        print(f"  - {layer}_median.png")


if __name__ == "__main__":
    main()

