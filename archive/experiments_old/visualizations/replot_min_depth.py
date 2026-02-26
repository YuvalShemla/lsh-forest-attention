#!/usr/bin/env python3
"""
Replot min_depth sweep results without shaded areas and add TopK from main results.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from matplotlib.ticker import LogLocator, FuncFormatter

# Paths
MIN_DEPTH_RESULTS = '../../results/approximation_evaluation/v2/min_depth_sweep/full_results.json'
MAIN_RESULTS = '../../results/approximation_evaluation/v2/full_results.json'
OUTPUT_DIR = Path('../../results/approximation_evaluation/v2/min_depth_sweep')

# Minimum Depth Values (only even numbers: 0, 2, 4, 6, 8)
MIN_DEPTH_VALUES = [0, 2, 4, 6, 8]

# Custom colors for min_depth values
MIN_DEPTH_COLORS = {
    0: '#FFD700',  # Yellow
    2: '#87CEEB',  # Light blue (Sky blue)
    4: '#0000FF',  # Blue
    6: '#A52A2A',  # Brown
    8: '#808080'   # Gray
}

# Budget range to plot
BUDGET_MIN = 20
BUDGET_MAX = 220

# Set all text to sans-serif font family
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Verdana', 'Liberation Sans']

# Map layer names to titles
layer_titles = {
    'first_layer': 'First Layer (Layer 0)',
    'last_layer': 'Last Layer (Layer 31)'
}


def load_and_aggregate_min_depth_results():
    """Load and aggregate min_depth sweep results."""
    with open(MIN_DEPTH_RESULTS, 'r') as f:
        data = json.load(f)
    
    aggregated = {}
    for layer_name in data['results_by_layer']:
        all_results = data['results_by_layer'][layer_name]
        
        # Collect all methods
        all_methods = set()
        for query_results in all_results:
            for budget_results in query_results['budget_controlled'].values():
                all_methods.update(budget_results.keys())
        
        layer_agg = {method: {} for method in all_methods}
        
        # Aggregate
        for query_results in all_results:
            for budget, method_errors in query_results['budget_controlled'].items():
                for method, error in method_errors.items():
                    if error is None:
                        continue
                    if budget not in layer_agg[method]:
                        layer_agg[method][budget] = []
                    layer_agg[method][budget].append(error)
        
        # Compute statistics
        for method in layer_agg:
            for budget in list(layer_agg[method].keys()):
                errors = layer_agg[method][budget]
                if len(errors) > 0:
                    layer_agg[method][budget] = {
                        'mean': float(np.mean(errors)),
                        'median': float(np.median(errors)),
                        'std': float(np.std(errors)),
                        'n': len(errors)
                    }
                else:
                    del layer_agg[method][budget]
        
        aggregated[layer_name] = layer_agg
    
    return aggregated


def load_topk_results():
    """Load TopK results from main results file."""
    with open(MAIN_RESULTS, 'r') as f:
        data = json.load(f)
    
    topk_by_layer = {}
    for layer_name in data['results_by_layer']:
        all_results = data['results_by_layer'][layer_name]
        
        # Aggregate TopK
        topk_agg = {}
        for query_results in all_results:
            for budget, method_errors in query_results['budget_controlled'].items():
                if 'TopK' in method_errors:
                    error = method_errors['TopK']
                    if budget not in topk_agg:
                        topk_agg[budget] = []
                    topk_agg[budget].append(error)
        
        # Compute statistics
        for budget in list(topk_agg.keys()):
            errors = topk_agg[budget]
            if len(errors) > 0:
                topk_agg[budget] = {
                    'mean': float(np.mean(errors)),
                    'median': float(np.median(errors)),
                    'std': float(np.std(errors)),
                    'n': len(errors)
                }
            else:
                del topk_agg[budget]
        
        topk_by_layer[layer_name] = topk_agg
    
    return topk_by_layer


def plot_results(min_depth_agg, topk_data):
    """Generate plots without shaded areas and with TopK, filtered to budget range 20-220."""
    
    for layer_name in min_depth_agg:
        layer_title = layer_titles.get(layer_name, layer_name.replace('_', ' ').title())
        agg_data = min_depth_agg[layer_name]
        topk_agg = topk_data[layer_name]
        
        # Get n_queries (try from any method)
        n_queries = 0
        for method_name in agg_data:
            if len(agg_data[method_name]) > 0:
                n_queries = list(agg_data[method_name].values())[0]['n']
                break
        
        # ========================================================================
        # PLOT 1: MEAN
        # ========================================================================
        fig_mean, ax_mean = plt.subplots(figsize=(14, 8))
        
        # Plot TopK (filtered to budget range)
        if len(topk_agg) > 0:
            # Convert string keys to integers and filter
            budgets = sorted([int(b) for b in topk_agg.keys() if BUDGET_MIN <= int(b) <= BUDGET_MAX])
            if len(budgets) > 0:
                means = [topk_agg[str(b)]['mean'] for b in budgets]
                budgets_arr = np.array(budgets)
                means_arr = np.array(means)
                
                ax_mean.plot(budgets_arr, means_arr, marker='o', linestyle='-',
                            label='Top-K', color='#d62728', linewidth=2.5, 
                            markersize=6, alpha=0.95, zorder=4)
        
        # Plot Uniform and Oracle (baselines)
        for method_name in ['Uniform', 'Oracle']:
            if method_name not in agg_data or len(agg_data[method_name]) == 0:
                continue
            
            # Filter budgets to range (convert string keys to int)
            all_budgets = sorted([int(b) for b in agg_data[method_name].keys()])
            budgets = [b for b in all_budgets if BUDGET_MIN <= b <= BUDGET_MAX]
            
            if len(budgets) == 0:
                continue
            
            means = [agg_data[method_name][str(b)]['mean'] for b in budgets]
            budgets_arr = np.array(budgets)
            means_arr = np.array(means)
            
            # Colors
            color = '#2ca02c' if method_name == 'Oracle' else '#ff7f0e'
            marker = '^' if method_name == 'Oracle' else 's'
            linestyle = '-' if method_name == 'Oracle' else '--'
            
            ax_mean.plot(budgets_arr, means_arr, marker=marker, linestyle=linestyle,
                        label=method_name, color=color, linewidth=2.5, 
                        markersize=6, alpha=0.95, zorder=3)
        
        # Plot prefix_sampling with different min_depth values (filtered to budget range)
        for min_depth in MIN_DEPTH_VALUES:
            method_name = f'prefix_sampling_min{min_depth}'
            if method_name not in agg_data or len(agg_data[method_name]) == 0:
                continue
            
            # Filter budgets to range (convert string keys to int)
            all_budgets = sorted([int(b) for b in agg_data[method_name].keys()])
            budgets = [b for b in all_budgets if BUDGET_MIN <= b <= BUDGET_MAX]
            
            if len(budgets) == 0:
                continue
            
            means = [agg_data[method_name][str(b)]['mean'] for b in budgets]
            budgets_arr = np.array(budgets)
            means_arr = np.array(means)
            
            color = MIN_DEPTH_COLORS[min_depth]
            ax_mean.plot(budgets_arr, means_arr, marker='D', linestyle='-',
                        label=f'prefix_sampling (min_depth={min_depth})', color=color, 
                        linewidth=2.0, markersize=5, alpha=0.9, zorder=2)
        
        # Formatting
        ax_mean.set_xlabel('Budget (Number of Keys)', fontsize=13, fontweight='bold', family='sans-serif')
        ax_mean.set_ylabel('Relative L2 Error (Mean)', fontsize=13, fontweight='bold', family='sans-serif')
        ax_mean.set_title(f'Minimum Depth Comparison - Mean\n{layer_title} (averaged over {n_queries} queries)',
                         fontsize=14, fontweight='bold', family='sans-serif')
        ax_mean.set_yscale('log')
        
        # Set x-axis range
        ax_mean.set_xlim(BUDGET_MIN - 10, BUDGET_MAX + 10)
        
        # Set y-axis range (only from data in budget range)
        all_errors = []
        for method_name in agg_data:
            for budget, stats in agg_data[method_name].items():
                budget_int = int(budget)
                if BUDGET_MIN <= budget_int <= BUDGET_MAX:
                    all_errors.append(stats['mean'])
        if len(topk_agg) > 0:
            for budget, stats in topk_agg.items():
                budget_int = int(budget)
                if BUDGET_MIN <= budget_int <= BUDGET_MAX:
                    all_errors.append(stats['mean'])
        
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
        
        # Set x-axis ticks
        ax_mean.set_xticks(range(BUDGET_MIN, BUDGET_MAX + 1, 40))  # Every 40 keys
        
        for label in ax_mean.get_xticklabels():
            label.set_family('sans-serif')
        for label in ax_mean.get_yticklabels():
            label.set_family('sans-serif')
        
        ax_mean.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, which='major')
        ax_mean.grid(True, alpha=0.15, linestyle='--', linewidth=0.3, which='minor')
        ax_mean.legend(fontsize=9, loc='best', ncol=2, framealpha=0.95, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plot_path_mean = OUTPUT_DIR / f'{layer_name}_mean.png'
        plt.savefig(plot_path_mean, dpi=150, bbox_inches='tight')
        print(f"  ✓ Generated: {plot_path_mean}")
        plt.close()
        
        # ========================================================================
        # PLOT 2: MEDIAN
        # ========================================================================
        fig_median, ax_median = plt.subplots(figsize=(14, 8))
        
        # Plot TopK (filtered to budget range)
        if len(topk_agg) > 0:
            # Convert string keys to integers and filter
            budgets = sorted([int(b) for b in topk_agg.keys() if BUDGET_MIN <= int(b) <= BUDGET_MAX])
            if len(budgets) > 0:
                medians = [topk_agg[str(b)]['median'] for b in budgets]
                budgets_arr = np.array(budgets)
                medians_arr = np.array(medians)
                
                ax_median.plot(budgets_arr, medians_arr, marker='o', linestyle='-',
                             label='Top-K', color='#d62728', linewidth=2.5, 
                             markersize=6, alpha=0.95, zorder=4)
        
        # Plot Uniform and Oracle (baselines)
        for method_name in ['Uniform', 'Oracle']:
            if method_name not in agg_data or len(agg_data[method_name]) == 0:
                continue
            
            # Filter budgets to range (convert string keys to int)
            all_budgets = sorted([int(b) for b in agg_data[method_name].keys()])
            budgets = [b for b in all_budgets if BUDGET_MIN <= b <= BUDGET_MAX]
            
            if len(budgets) == 0:
                continue
            
            medians = [agg_data[method_name][str(b)]['median'] for b in budgets]
            budgets_arr = np.array(budgets)
            medians_arr = np.array(medians)
            
            # Colors
            color = '#2ca02c' if method_name == 'Oracle' else '#ff7f0e'
            marker = '^' if method_name == 'Oracle' else 's'
            linestyle = '-' if method_name == 'Oracle' else '--'
            
            ax_median.plot(budgets_arr, medians_arr, marker=marker, linestyle=linestyle,
                          label=method_name, color=color, linewidth=2.5, 
                          markersize=6, alpha=0.95, zorder=3)
        
        # Plot prefix_sampling with different min_depth values (filtered to budget range)
        for min_depth in MIN_DEPTH_VALUES:
            method_name = f'prefix_sampling_min{min_depth}'
            if method_name not in agg_data or len(agg_data[method_name]) == 0:
                continue
            
            # Filter budgets to range (convert string keys to int)
            all_budgets = sorted([int(b) for b in agg_data[method_name].keys()])
            budgets = [b for b in all_budgets if BUDGET_MIN <= b <= BUDGET_MAX]
            
            if len(budgets) == 0:
                continue
            
            medians = [agg_data[method_name][str(b)]['median'] for b in budgets]
            budgets_arr = np.array(budgets)
            medians_arr = np.array(medians)
            
            color = MIN_DEPTH_COLORS[min_depth]
            ax_median.plot(budgets_arr, medians_arr, marker='D', linestyle='-',
                          label=f'prefix_sampling (min_depth={min_depth})', color=color, 
                          linewidth=2.0, markersize=5, alpha=0.9, zorder=2)
        
        # Formatting
        ax_median.set_xlabel('Budget (Number of Keys)', fontsize=13, fontweight='bold', family='sans-serif')
        ax_median.set_ylabel('Relative L2 Error (Median)', fontsize=13, fontweight='bold', family='sans-serif')
        ax_median.set_title(f'Minimum Depth Comparison - Median\n{layer_title} (averaged over {n_queries} queries)',
                           fontsize=14, fontweight='bold', family='sans-serif')
        ax_median.set_yscale('log')
        
        # Set x-axis range
        ax_median.set_xlim(BUDGET_MIN - 10, BUDGET_MAX + 10)
        
        # Set y-axis range (only from data in budget range)
        all_errors = []
        for method_name in agg_data:
            for budget, stats in agg_data[method_name].items():
                budget_int = int(budget)
                if BUDGET_MIN <= budget_int <= BUDGET_MAX:
                    all_errors.append(stats['median'])
        if len(topk_agg) > 0:
            for budget, stats in topk_agg.items():
                budget_int = int(budget)
                if BUDGET_MIN <= budget_int <= BUDGET_MAX:
                    all_errors.append(stats['median'])
        
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
        
        # Set x-axis ticks
        ax_median.set_xticks(range(BUDGET_MIN, BUDGET_MAX + 1, 40))  # Every 40 keys
        
        for label in ax_median.get_xticklabels():
            label.set_family('sans-serif')
        for label in ax_median.get_yticklabels():
            label.set_family('sans-serif')
        
        ax_median.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, which='major')
        ax_median.grid(True, alpha=0.15, linestyle='--', linewidth=0.3, which='minor')
        ax_median.legend(fontsize=9, loc='best', ncol=2, framealpha=0.95, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plot_path_median = OUTPUT_DIR / f'{layer_name}_median.png'
        plt.savefig(plot_path_median, dpi=150, bbox_inches='tight')
        print(f"  ✓ Generated: {plot_path_median}")
        plt.close()


def main():
    """Load data and regenerate plots."""
    print("="*70)
    print("REPLOTTING MIN_DEPTH SWEEP RESULTS")
    print("="*70)
    
    try:
        print("\nLoading min_depth sweep results...")
        min_depth_agg = load_and_aggregate_min_depth_results()
        print(f"✓ Loaded results for layers: {list(min_depth_agg.keys())}")
        
        print("\nLoading TopK results from main results...")
        topk_data = load_topk_results()
        print(f"✓ Loaded TopK results for layers: {list(topk_data.keys())}")
        
        print("\nGenerating plots (without shaded areas, with TopK)...")
        plot_results(min_depth_agg, topk_data)
        
        print(f"\n{'='*70}")
        print("✅ PLOTTING COMPLETE")
        print(f"{'='*70}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

