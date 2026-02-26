#!/usr/bin/env python3
"""
Reprocess existing full_results.json to create aggregated JSON and plots.
Quick fix to generate outputs from the successful evaluation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import LogLocator, FuncFormatter

# Load configuration from original
L = 25
GAMMA = 1.0

def aggregate_results(all_results):
    """Aggregate across queries."""
    aggregated = {
        'budget_controlled': {method: {} for method in ['TopK', 'Naive', 'Oracle', 'prefix_sampling']},
        'lsh_snis': {}
    }
    
    # Aggregate budget-controlled
    for query_results in all_results:
        for budget, method_errors in query_results['budget_controlled'].items():
            for method, error in method_errors.items():
                if budget not in aggregated['budget_controlled'][method]:
                    aggregated['budget_controlled'][method][budget] = []
                aggregated['budget_controlled'][method][budget].append(error)
    
    for method in aggregated['budget_controlled']:
        for budget in list(aggregated['budget_controlled'][method].keys()):
            errors = aggregated['budget_controlled'][method][budget]
            aggregated['budget_controlled'][method][budget] = {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'n': len(errors)
            }
    
    # Aggregate LSH-SNIS
    for query_results in all_results:
        for lsh_result in query_results['lsh_snis']:
            L_val = lsh_result['L']
            K_val = lsh_result['K']
            budget = lsh_result['budget']
            error = lsh_result['error']
            
            key = (L_val, K_val)
            if key not in aggregated['lsh_snis']:
                aggregated['lsh_snis'][key] = {}
            if budget not in aggregated['lsh_snis'][key]:
                aggregated['lsh_snis'][key][budget] = []
            aggregated['lsh_snis'][key][budget].append(error)
    
    for (L_val, K_val) in aggregated['lsh_snis']:
        for budget in list(aggregated['lsh_snis'][(L_val, K_val)].keys()):
            errors = aggregated['lsh_snis'][(L_val, K_val)][budget]
            aggregated['lsh_snis'][(L_val, K_val)][budget] = {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'n': len(errors)
            }
    
    return aggregated


# Load full results
print("Loading full_results.json...")
full_json = Path('../../results/approximation_evaluation/v2/full_results.json')

with open(full_json, 'r') as f:
    all_results = json.load(f)

print(f"✓ Loaded {len(all_results['results_by_layer']['first_layer'])} queries per layer")

# Aggregate
print("Aggregating...")
aggregated = {}
for layer in ['first_layer', 'last_layer']:
    aggregated[layer] = aggregate_results(all_results['results_by_layer'][layer])

# Save with tuple keys converted to strings
aggregated_json_safe = {}
for layer in ['first_layer', 'last_layer']:
    aggregated_json_safe[layer] = {
        'budget_controlled': aggregated[layer]['budget_controlled'],
        'lsh_snis': {}
    }
    for (L_val, K_val), data in aggregated[layer]['lsh_snis'].items():
        key_str = f"L={L_val},K={K_val}"
        aggregated_json_safe[layer]['lsh_snis'][key_str] = data

output_dir = Path('../../results/approximation_evaluation/v2')
agg_json = output_dir / 'aggregated.json'

with open(agg_json, 'w') as f:
    json.dump({'aggregated': aggregated_json_safe, 'metadata': all_results['metadata']}, f, indent=2)

print(f"✓ Saved: {agg_json}")

# Generate plots
print("\nGenerating plots...")

for layer_name in ['first_layer', 'last_layer']:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    agg_data = aggregated[layer_name]
    layer_title = 'First Layer' if 'first' in layer_name else 'Last Layer'
    
    # Budget-controlled methods
    for method, marker, label in [
        ('TopK', 'o-', 'Top-K'),
        ('Naive', 's--', 'Naive Sampling'),
        ('Oracle', '^-', 'Oracle Sampling'),
        ('prefix_sampling', 'd-', 'prefix_sampling')
    ]:
        if method in agg_data['budget_controlled']:
            budgets = sorted(agg_data['budget_controlled'][method].keys())
            means = [agg_data['budget_controlled'][method][b]['mean'] for b in budgets]
            ax.plot(budgets, means, marker, label=label, linewidth=2.5, markersize=5, alpha=0.9)
    
    # LSH-SNIS points
    colors = {'(10, 6)': 'C4', '(10, 7)': 'C5', '(10, 8)': 'C6', '(10, 9)': 'C7',
              '(25, 6)': 'C8', '(25, 7)': 'C9', '(25, 8)': 'purple', '(25, 9)': 'brown'}
    
    for (L_val, K_val), lsh_data in agg_data['lsh_snis'].items():
        budgets_lsh = []
        means_lsh = []
        for budget, stats in lsh_data.items():
            budgets_lsh.append(budget)
            means_lsh.append(stats['mean'])
        
        if budgets_lsh:
            color_key = f'({L_val}, {K_val})'
            color = colors.get(color_key, 'gray')
            ax.plot(budgets_lsh, means_lsh, 'x', label=f'LSH-SNIS (L={L_val}, K={K_val})',
                   markersize=8, markeredgewidth=2, color=color, alpha=0.8)
    
    ax.set_xlabel('Budget (Number of Keys)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Relative L2 Error', fontsize=13, fontweight='bold')
    
    n_queries = list(agg_data['budget_controlled']['Naive'].values())[0]['n']
    ax.set_title(f'Attention Approximation Methods\n{layer_title} (averaged over {n_queries} queries)',
                fontsize=14, fontweight='bold')
    
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1g}'))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: ''))
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best', ncol=2)
    
    plt.tight_layout()
    
    plot_path = output_dir / f'{layer_name}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Generated: {plot_path}")
    plt.close()

print("\n✅ Reprocessing complete!")

