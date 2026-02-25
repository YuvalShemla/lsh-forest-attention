#!/usr/bin/env python3
"""
Minimum Depth Comparison

Compares Uniform, Oracle, and prefix_sampling with different min_depth values (0-10).
Shows how minimum depth filtering affects performance.

Results saved to: results/approximation_evaluation/v2/min_depth_sweep/
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
L = 25  # Number of hash tables/trees (for prefix_sampling)
K_MAX = 30 
SEED = 42  # Random seed

# prefix_sampling Parameters
GAMMA = 1.0  # Bucket size penalty (1.0 = linear: 1/bucket_size)
TAU = 0.0  # Smoothing term (0.0 = no smoothing)

# Minimum Depth Sweep
MIN_DEPTH_VALUES = list(range(0, 11, 2))  # [0,, 2, ..., 10]

# Budget Range (for methods with budget control)
BUDGETS = list(range(20, 220, 20))  # [20, 40, 60, ..., 380]

# Evaluation Scale
NUM_EXAMPLES = 10  # Number of examples to test (out of 11 available)
NUM_QUERIES_PER_EXAMPLE = 100  # Last N queries per example
LAYERS_TO_TEST = ['first_layer', 'last_layer']  # Both layers

# Data Path
DATA_PATH = '../data/attention_vectors_updated_long.jsonl'

# Output
OUTPUT_DIR = '../results/approximation_evaluation/v2/min_depth_sweep'

# ============================================================================
# END HYPERPARAMETERS
# ============================================================================


def evaluate_single_query(Q, K, V, query_pos, head_dim, lsh_union):
    """
    Evaluate methods on one query with different min_depth values.
    
    Args:
        lsh_union: LSH structure for prefix_sampling
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
        'budget_controlled': {}  # {budget: {method: error}}
    }
    
    # Build LSH for prefix_sampling
    lsh_union.build_index(valid_keys)
    
    for budget in BUDGETS:
        if budget > num_valid:
            continue
        
        budget_results = {}
        
        # Uniform Sampling
        output_naive, _ = methods.naive_sampling(q, valid_keys, valid_values, gt_logits, budget)
        error_naive = utils.relative_l2_error(output_naive, gt_output)
        budget_results['Uniform'] = float(error_naive)
        
        # Oracle Sampling
        output_oracle, unique_budget = methods.oracle_sampling(
            q, valid_keys, valid_values, gt_logits, gt_weights, budget
        )
        error_oracle = utils.relative_l2_error(output_oracle, gt_output)
        budget_results['Oracle'] = float(error_oracle)
        
        # prefix_sampling with different min_depth values
        for min_depth in MIN_DEPTH_VALUES:
            try:
                output_prefix, actual_budget = methods.prefix_sampling(
                    q, valid_keys, valid_values, gt_logits, head_dim,
                    lsh_union, budget, GAMMA, TAU, min_depth
                )
                
                if actual_budget > 0:  # Only if we got some keys
                    error_prefix = utils.relative_l2_error(output_prefix, gt_output)
                    budget_results[f'prefix_sampling_min{min_depth}'] = float(error_prefix)
                else:
                    # No keys met the min_depth requirement
                    budget_results[f'prefix_sampling_min{min_depth}'] = None
            except Exception as e:
                # Handle case where min_depth filters out all keys
                budget_results[f'prefix_sampling_min{min_depth}'] = None
        
        results['budget_controlled'][budget] = budget_results
    
    return results


def aggregate_results(all_results):
    """Aggregate across queries."""
    # Collect all methods
    all_methods = set()
    for query_results in all_results:
        for budget_results in query_results['budget_controlled'].values():
            all_methods.update(budget_results.keys())
    
    aggregated = {method: {} for method in all_methods}
    
    # Aggregate
    for query_results in all_results:
        for budget, method_errors in query_results['budget_controlled'].items():
            for method, error in method_errors.items():
                if error is None:  # Skip None values (no keys met requirement)
                    continue
                if budget not in aggregated[method]:
                    aggregated[method][budget] = []
                aggregated[method][budget].append(error)
    
    # Compute statistics
    for method in aggregated:
        for budget in list(aggregated[method].keys()):
            errors = aggregated[method][budget]
            if len(errors) > 0:
                aggregated[method][budget] = {
                    'mean': float(np.mean(errors)),
                    'median': float(np.median(errors)),
                    'std': float(np.std(errors)),
                    'n': len(errors)
                }
            else:
                del aggregated[method][budget]
    
    return aggregated


def plot_results(aggregated_dict, output_dir):
    """Generate comparison plots for min_depth sweep."""
    
    from matplotlib.ticker import LogLocator, FuncFormatter
    
    # Set all text to sans-serif font family
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Verdana', 'Liberation Sans']
    
    # Map layer names to titles
    layer_titles = {
        'first_layer': 'First Layer (Layer 0)',
        'last_layer': 'Last Layer (Layer 31)'
    }
    
    for layer_name, agg_data in aggregated_dict.items():
        layer_title = layer_titles.get(layer_name, layer_name.replace('_', ' ').title())
        
        # ========================================================================
        # PLOT 1: MEAN with shaded variance
        # ========================================================================
        fig_mean, ax_mean = plt.subplots(figsize=(14, 8))
        
        # Plot Uniform and Oracle (baselines)
        for method_name in ['Uniform', 'Oracle']:
            if method_name not in agg_data or len(agg_data[method_name]) == 0:
                continue
            
            budgets = sorted(agg_data[method_name].keys())
            means = [agg_data[method_name][b]['mean'] for b in budgets]
            stds = [agg_data[method_name][b]['std'] for b in budgets]
            
            budgets_arr = np.array(budgets)
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            
            # Shaded area
            upper_bound = means_arr + stds_arr
            lower_bound = np.maximum(means_arr - stds_arr, 1e-6)
            
            # Colors
            color = '#2ca02c' if method_name == 'Oracle' else '#ff7f0e'
            marker = '^' if method_name == 'Oracle' else 's'
            linestyle = '-' if method_name == 'Oracle' else '--'
            
            ax_mean.fill_between(budgets_arr, lower_bound, upper_bound, 
                                 color=color, alpha=0.2, linewidth=0, zorder=1, label='_nolegend_')
            ax_mean.plot(budgets_arr, means_arr, marker=marker, linestyle=linestyle,
                        label=method_name, color=color, linewidth=2.5, 
                        markersize=6, alpha=0.95, zorder=3)
        
        # Plot prefix_sampling with different min_depth values
        # Use a colormap for different min_depth values
        import matplotlib.cm as cm
        colors = cm.viridis(np.linspace(0, 1, len(MIN_DEPTH_VALUES)))
        
        for idx, min_depth in enumerate(MIN_DEPTH_VALUES):
            method_name = f'prefix_sampling_min{min_depth}'
            if method_name not in agg_data or len(agg_data[method_name]) == 0:
                continue
            
            budgets = sorted(agg_data[method_name].keys())
            means = [agg_data[method_name][b]['mean'] for b in budgets]
            stds = [agg_data[method_name][b]['std'] for b in budgets]
            
            budgets_arr = np.array(budgets)
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            
            # Shaded area
            upper_bound = means_arr + stds_arr
            lower_bound = np.maximum(means_arr - stds_arr, 1e-6)
            
            color = colors[idx]
            ax_mean.fill_between(budgets_arr, lower_bound, upper_bound, 
                                 color=color, alpha=0.15, linewidth=0, zorder=1, label='_nolegend_')
            ax_mean.plot(budgets_arr, means_arr, marker='D', linestyle='-',
                        label=f'prefix_sampling (min_depth={min_depth})', color=color, 
                        linewidth=2.0, markersize=5, alpha=0.9, zorder=2)
        
        # Formatting
        ax_mean.set_xlabel('Budget (Number of Keys)', fontsize=13, fontweight='bold', family='sans-serif')
        ax_mean.set_ylabel('Relative L2 Error (Mean)', fontsize=13, fontweight='bold', family='sans-serif')
        
        n_queries = list(agg_data['Uniform'].values())[0]['n'] if 'Uniform' in agg_data else 0
        ax_mean.set_title(f'Minimum Depth Comparison - Mean\n{layer_title} (averaged over {n_queries} queries)',
                         fontsize=14, fontweight='bold', family='sans-serif')
        ax_mean.set_yscale('log')
        
        # Set y-axis range
        all_errors = []
        for method_name in agg_data:
            for budget, stats in agg_data[method_name].items():
                all_errors.append(stats['mean'])
                all_errors.append(stats['mean'] + stats['std'])
                all_errors.append(max(stats['mean'] - stats['std'], 1e-6))
        
        if all_errors:
            min_error = min(all_errors)
            max_error = max(all_errors)
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
        ax_mean.legend(fontsize=9, loc='best', ncol=2, framealpha=0.95, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plot_path_mean = output_dir / f'{layer_name}_mean.png'
        plt.savefig(plot_path_mean, dpi=150, bbox_inches='tight')
        print(f"  ✓ Generated: {plot_path_mean}")
        plt.close()
        
        # ========================================================================
        # PLOT 2: MEDIAN with shaded variance
        # ========================================================================
        fig_median, ax_median = plt.subplots(figsize=(14, 8))
        
        # Plot Uniform and Oracle
        for method_name in ['Uniform', 'Oracle']:
            if method_name not in agg_data or len(agg_data[method_name]) == 0:
                continue
            
            budgets = sorted(agg_data[method_name].keys())
            medians = [agg_data[method_name][b]['median'] for b in budgets]
            stds = [agg_data[method_name][b]['std'] for b in budgets]
            
            budgets_arr = np.array(budgets)
            medians_arr = np.array(medians)
            stds_arr = np.array(stds)
            
            upper_bound = medians_arr + stds_arr
            lower_bound = np.maximum(medians_arr - stds_arr, 1e-6)
            
            color = '#2ca02c' if method_name == 'Oracle' else '#ff7f0e'
            marker = '^' if method_name == 'Oracle' else 's'
            linestyle = '-' if method_name == 'Oracle' else '--'
            
            ax_median.fill_between(budgets_arr, lower_bound, upper_bound, 
                                  color=color, alpha=0.2, linewidth=0, zorder=1, label='_nolegend_')
            ax_median.plot(budgets_arr, medians_arr, marker=marker, linestyle=linestyle,
                          label=method_name, color=color, linewidth=2.5, 
                          markersize=6, alpha=0.95, zorder=3)
        
        # Plot prefix_sampling with different min_depth values
        for idx, min_depth in enumerate(MIN_DEPTH_VALUES):
            method_name = f'prefix_sampling_min{min_depth}'
            if method_name not in agg_data or len(agg_data[method_name]) == 0:
                continue
            
            budgets = sorted(agg_data[method_name].keys())
            medians = [agg_data[method_name][b]['median'] for b in budgets]
            stds = [agg_data[method_name][b]['std'] for b in budgets]
            
            budgets_arr = np.array(budgets)
            medians_arr = np.array(medians)
            stds_arr = np.array(stds)
            
            upper_bound = medians_arr + stds_arr
            lower_bound = np.maximum(medians_arr - stds_arr, 1e-6)
            
            color = colors[idx]
            ax_median.fill_between(budgets_arr, lower_bound, upper_bound, 
                                  color=color, alpha=0.15, linewidth=0, zorder=1, label='_nolegend_')
            ax_median.plot(budgets_arr, medians_arr, marker='D', linestyle='-',
                          label=f'prefix_sampling (min_depth={min_depth})', color=color, 
                          linewidth=2.0, markersize=5, alpha=0.9, zorder=2)
        
        # Formatting
        ax_median.set_xlabel('Budget (Number of Keys)', fontsize=13, fontweight='bold', family='sans-serif')
        ax_median.set_ylabel('Relative L2 Error (Median)', fontsize=13, fontweight='bold', family='sans-serif')
        ax_median.set_title(f'Minimum Depth Comparison - Median\n{layer_title} (averaged over {n_queries} queries)',
                           fontsize=14, fontweight='bold', family='sans-serif')
        ax_median.set_yscale('log')
        
        # Set y-axis range
        all_errors = []
        for method_name in agg_data:
            for budget, stats in agg_data[method_name].items():
                all_errors.append(stats['median'])
                all_errors.append(stats['median'] + stats['std'])
                all_errors.append(max(stats['median'] - stats['std'], 1e-6))
        
        if all_errors:
            min_error = min(all_errors)
            max_error = max(all_errors)
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
        ax_median.legend(fontsize=9, loc='best', ncol=2, framealpha=0.95, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plot_path_median = output_dir / f'{layer_name}_median.png'
        plt.savefig(plot_path_median, dpi=150, bbox_inches='tight')
        print(f"  ✓ Generated: {plot_path_median}")
        plt.close()


def main():
    """Run min_depth comparison."""
    
    print("="*70)
    print("MINIMUM DEPTH COMPARISON")
    print("="*70)
    print(f"\nMethods:")
    print(f"  1. Uniform Sampling (baseline)")
    print(f"  2. Oracle Sampling (gold standard)")
    print(f"  3. prefix_sampling with min_depth = {MIN_DEPTH_VALUES}")
    print(f"\nConfiguration:")
    print(f"  L={L}, γ={GAMMA}, τ={TAU}")
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
    
    # Create LSH structure
    head_dim = 128
    lsh_union = utils.LSHStructure(L, K_MAX, head_dim, center_keys=True, seed=SEED)
    
    print(f"✓ Created LSH structure: L={L}, K_MAX={K_MAX}")
    
    # Storage
    all_results = {
        'metadata': {
            'methods': ['Uniform', 'Oracle'] + [f'prefix_sampling_min{d}' for d in MIN_DEPTH_VALUES],
            'prefix_sampling': {'L': L, 'gamma': GAMMA, 'tau': TAU, 'min_depth_values': MIN_DEPTH_VALUES},
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
                query_results = evaluate_single_query(Q, K, V, query_pos, head_dim, lsh_union)
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
    
    agg_json = output_dir / 'aggregated.json'
    with open(agg_json, 'w') as f:
        json.dump({'aggregated': aggregated, 'metadata': all_results['metadata']}, f, indent=2)
    print(f"✓ Aggregated: {agg_json}")
    
    # Plots
    print(f"\nGenerating plots...")
    plot_results(aggregated, output_dir)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY (Budget = 200)")
    print(f"{'='*70}")
    
    for layer in LAYERS_TO_TEST:
        print(f"\n{layer.upper()}:")
        
        # Baselines
        for method in ['Oracle', 'Uniform']:
            if method in aggregated[layer] and 200 in aggregated[layer][method]:
                stats = aggregated[layer][method][200]
                print(f"  {method:20s}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['n']})")
        
        # prefix_sampling with different min_depth
        print(f"\n  prefix_sampling (min_depth sweep):")
        for min_depth in MIN_DEPTH_VALUES:
            method = f'prefix_sampling_min{min_depth}'
            if method in aggregated[layer] and 200 in aggregated[layer][method]:
                stats = aggregated[layer][method][200]
                print(f"    min_depth={min_depth:2d}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['n']})")
            else:
                print(f"    min_depth={min_depth:2d}: No data (all keys filtered out)")
    
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

