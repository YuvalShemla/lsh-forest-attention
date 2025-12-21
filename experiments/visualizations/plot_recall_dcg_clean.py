#!/usr/bin/env python3
"""
Clean Bar Plots for Recall and DCG Evaluation

Professional scientific plots with:
- Same color bars (gray)
- No error bars
- Oracle on far right
- Clean, clear appearance
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
AGGREGATED_JSON = '../../results/approximation_evaluation/v2/recall_dcg_evaluation/aggregated.json'
OUTPUT_DIR = Path('../../results/approximation_evaluation/v2/recall_dcg_evaluation')

# Set professional style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Layer titles
layer_titles = {
    'first_layer': 'First Layer (Layer 0)',
    'last_layer': 'Last Layer (Layer 31)'
}

# Metrics configuration
metrics_config = [
    ('recall_at_10', 'Recall@10', 'Fraction of Top 10 Keys Sampled'),
    ('recall_at_100', 'Recall@100', 'Fraction of Top 100 Keys Sampled'),
    ('recall_at_100_of_top_10', 'Recall@100 of Top 10', 'Fraction of Top 10 in 100 Sampled'),
    ('dcg_at_100', 'DCG@100', 'Discounted Cumulative Gain'),
    ('avg_rank', 'Average Rank', 'Average Rank of Sampled Keys')
]


def get_method_order(methods):
    """
    Order methods: Uniform (first), prefix_sampling (by min_depth), Oracle (last).
    """
    prefix_methods = []
    
    for method in methods:
        if method.startswith('prefix_sampling_min'):
            try:
                min_depth = int(method.replace('prefix_sampling_min', ''))
                prefix_methods.append((min_depth, method))
            except:
                pass
    
    # Sort prefix methods by min_depth
    prefix_methods.sort(key=lambda x: x[0])
    
    # Final order: Uniform (first), prefix methods (sorted), Oracle (last)
    ordered = []
    if 'Uniform' in methods:
        ordered.append('Uniform')
    ordered.extend([m[1] for m in prefix_methods])
    if 'Oracle' in methods:
        ordered.append('Oracle')
    
    return ordered


def format_method_label(method):
    """Format method name for display."""
    if method == 'Uniform':
        return 'Uniform'
    elif method == 'Oracle':
        return 'Oracle'
    elif method.startswith('prefix_sampling_min'):
        min_depth = method.replace('prefix_sampling_min', '')
        return f'k ≥ {min_depth}'
    else:
        return method


def plot_metric(agg_data, layer_name, metric_key, metric_title, metric_ylabel, output_dir):
    """Create a clean bar plot for one metric."""
    
    # Get all methods
    all_methods = list(agg_data.keys())
    
    # Order methods (Oracle last)
    ordered_methods = get_method_order(all_methods)
    
    # Collect values
    values = []
    labels = []
    
    for method in ordered_methods:
        if method in agg_data:
            method_data = agg_data[method]
            if metric_key in method_data and method_data[metric_key] is not None:
                values.append(method_data[metric_key]['mean'])
                labels.append(format_method_label(method))
    
    if len(values) == 0:
        print(f"    Skipping {metric_key} (no data)")
        return
    
    # Create figure (smaller)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Bar positions
    x_pos = np.arange(len(labels))
    
    # Assign colors: Uniform and Oracle = pink, prefix_sampling = purple
    bar_colors = []
    for label in labels:
        if label == 'Uniform' or label == 'Oracle':
            bar_colors.append('#e91e63')  # Pink
        else:
            bar_colors.append('#9467bd')  # Purple
    
    # Create bars
    bars = ax.bar(x_pos, values, width=0.6, color=bar_colors, edgecolor='black', 
                  linewidth=0.8, alpha=0.85)
    
    # Formatting
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    # No y-axis label (values shown on bars)
    
    # Title: just metric title, with layer info
    if layer_name == 'last_layer':
        subtitle = 'Layer 31, avg over 1100 queries'
    elif layer_name == 'first_layer':
        subtitle = 'Layer 0, avg over 1100 queries'
    else:
        subtitle = layer_name.replace('_', ' ').title()
    
    ax.set_title(f'{metric_title}\n{subtitle}', fontsize=13, fontweight='bold', pad=15)
    
    # X-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    
    # Y-axis (no labels, no grid)
    if metric_key == 'avg_rank':
        # For avg_rank, lower is better - set y-axis appropriately
        y_max = max(values) * 1.15
        ax.set_ylim(0, y_max)
        # Add note that lower is better
        ax.text(0.02, 0.98, 'Lower is better', transform=ax.transAxes,
               fontsize=9, verticalalignment='top', style='italic', alpha=0.7)
    else:
        # For other metrics, higher is better
        ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1.0)
    
    # Remove y-axis tick labels
    ax.set_yticklabels([])
    
    # Add value labels on bars (if space allows)
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        # Format value based on metric
        if metric_key == 'avg_rank':
            label = f'{val:.0f}'
        elif metric_key == 'dcg_at_100':
            label = f'{val:.3f}'
        else:
            label = f'{val:.3f}'
        
        # Only add label if bar is tall enough
        if height > max(values) * 0.05:  # At least 5% of max height
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plot_path = output_dir / f'{layer_name}_{metric_key}_clean.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Generated: {plot_path}")
    plt.close()


def main():
    """Generate clean bar plots."""
    
    print("="*70)
    print("GENERATING CLEAN BAR PLOTS")
    print("="*70)
    
    # Load data
    print(f"\nLoading: {AGGREGATED_JSON}")
    with open(AGGREGATED_JSON, 'r') as f:
        data = json.load(f)
    
    aggregated = data['aggregated']
    
    print(f"✓ Loaded data for layers: {list(aggregated.keys())}")
    
    # Generate plots
    print(f"\nGenerating plots...")
    
    for layer_name, agg_data in aggregated.items():
        print(f"\n{layer_name}:")
        
        for metric_key, metric_title, metric_ylabel in metrics_config:
            # Check if metric exists in any method
            metric_exists = False
            for method in agg_data.keys():
                if metric_key in agg_data[method] and agg_data[method][metric_key] is not None:
                    metric_exists = True
                    break
            
            if not metric_exists:
                print(f"    Skipping {metric_key} (not in data)")
                continue
            
            try:
                plot_metric(agg_data, layer_name, metric_key, metric_title, metric_ylabel, OUTPUT_DIR)
            except Exception as e:
                print(f"    Error plotting {metric_key}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("✅ PLOTTING COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

