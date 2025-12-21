#!/usr/bin/env python3
"""
Concentration Statistics Plot - Standalone

Shows how attention mass accumulates as we include more keys (by percentage).
Computes statistics (mean, p10, p50, p90, p99) over the last N queries.

X-axis: Percentage of keys (0-100%)
Y-axis: Percentage of attention mass captured (0-100%)
Lines: Mean, p10, p50 (median), p90, p99
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting concentration statistics analysis...")

sns.set_style("whitegrid")

def softmax(x):
    """Numerically stable softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compute_concentration_curve(attn_weights, num_percentile_points=100):
    """
    Compute concentration curve: what % of mass is captured by top X% of keys?
    
    Returns:
        percentiles: array of key percentages (e.g., [1, 2, 3, ..., 100])
        mass_captured: array of attention mass percentages at each percentile
    """
    # Sort attention weights in descending order
    sorted_weights = np.sort(attn_weights)[::-1]
    
    # Compute cumulative sum (normalized to percentage)
    cumsum = np.cumsum(sorted_weights)
    total = cumsum[-1]
    cumsum_pct = (cumsum / total) * 100  # Convert to percentage
    
    # Create percentile points for X-axis
    num_keys = len(attn_weights)
    percentiles = np.linspace(0, 100, num_percentile_points + 1)[1:]  # 1%, 2%, ..., 100%
    
    # For each percentile, find how much mass is captured
    mass_at_percentiles = []
    for pct in percentiles:
        # Top X% of keys
        num_keys_at_pct = max(1, int(np.ceil(num_keys * pct / 100)))
        if num_keys_at_pct <= len(cumsum_pct):
            mass_at_percentiles.append(cumsum_pct[num_keys_at_pct - 1])
        else:
            mass_at_percentiles.append(100.0)
    
    return percentiles, np.array(mass_at_percentiles)

def analyze_concentration_statistics(
    filepath='../data/attention_vectors_updated.jsonl',
    layer_name='first_layer',
    num_queries=1000,
    num_percentile_points=100
):
    """
    Analyze concentration statistics over last N queries
    """
    
    print("="*70)
    print("📊 CONCENTRATION STATISTICS ANALYZER")
    print("="*70)
    
    # Load data
    print(f"\n📂 Loading: {filepath}")
    with open(filepath, 'r') as f:
        example = json.loads(f.readline())
    
    print("✅ JSON loaded successfully")
    
    # Extract arrays
    print("📊 Converting to numpy arrays...")
    Q = np.array(example[layer_name]['Q'], dtype=np.float32)
    K = np.array(example[layer_name]['K'], dtype=np.float32)
    
    seq_len, head_dim = Q.shape
    layer_idx = example[layer_name]['layer_idx']
    head_idx = example[layer_name]['head_idx']
    domain = example['domain']
    
    print(f"\n✅ Data loaded!")
    print(f"   Example: {example['example_id']}")
    print(f"   Domain: {domain}")
    print(f"   Layer: {layer_idx}, Head: {head_idx}")
    print(f"   Sequence length: {seq_len}")
    
    # Select last N queries (or all if less than N)
    num_queries = min(num_queries, seq_len - 100)  # Need at least some keys
    query_positions = list(range(seq_len - num_queries, seq_len))
    
    print(f"\n🔍 Analyzing last {len(query_positions)} queries...")
    print(f"   Positions: {query_positions[0]} to {query_positions[-1]}")
    
    # Compute concentration curves for all queries
    all_curves = []
    
    for idx, query_pos in enumerate(query_positions):
        if (idx + 1) % 100 == 0:
            print(f"   Processing query {idx+1}/{len(query_positions)}...")
        
        # Get query and valid keys
        q = Q[query_pos]
        k = K[:query_pos+1]
        
        # Compute attention weights
        scores = q @ k.T / np.sqrt(head_dim)
        attn_weights = softmax(scores)
        
        # Compute concentration curve
        percentiles, mass_captured = compute_concentration_curve(
            attn_weights, num_percentile_points
        )
        all_curves.append(mass_captured)
    
    # Convert to array: [num_queries, num_percentile_points]
    all_curves = np.array(all_curves)
    
    print(f"\n✅ Computed concentration curves for {len(all_curves)} queries")
    
    # Compute statistics across queries
    print("\n📊 Computing statistics...")
    mean_curve = np.mean(all_curves, axis=0)
    p10_curve = np.percentile(all_curves, 10, axis=0)
    p50_curve = np.percentile(all_curves, 50, axis=0)  # Median
    p90_curve = np.percentile(all_curves, 90, axis=0)
    p99_curve = np.percentile(all_curves, 99, axis=0)
    
    # Create plot
    print("\n📈 Creating plot...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot statistics as lines
    ax.plot(percentiles, mean_curve, label='Mean', color='black', 
            linewidth=3, linestyle='-', alpha=0.9)
    ax.plot(percentiles, p50_curve, label='Median (p50)', color='blue', 
            linewidth=2.5, linestyle='--', alpha=0.8)
    ax.plot(percentiles, p10_curve, label='p10', color='red', 
            linewidth=2, linestyle=':', alpha=0.7)
    ax.plot(percentiles, p90_curve, label='p90', color='green', 
            linewidth=2, linestyle=':', alpha=0.7)
    ax.plot(percentiles, p99_curve, label='p99', color='purple', 
            linewidth=2, linestyle='-.', alpha=0.7)
    
    # Fill between p10 and p90 for visual reference
    ax.fill_between(percentiles, p10_curve, p90_curve, alpha=0.2, 
                    color='gray', label='p10-p90 range')
    
    # Diagonal reference line (perfect uniform distribution)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1, 
            label='Uniform (reference)')
    
    # Formatting
    ax.set_xlabel('Percentage of Keys (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage of Attention Mass Captured (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Attention Concentration Statistics\n'
                 f'Layer {layer_idx}, Head {head_idx} | {len(query_positions)} queries (positions {query_positions[0]}-{query_positions[-1]})\n'
                 f'Domain: {domain[:60]}...',
                 fontsize=15, fontweight='bold', pad=20)
    
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    
    # Add reference markers
    for x_val in [10, 25, 50, 75]:
        ax.axvline(x=x_val, color='gray', alpha=0.2, linewidth=0.8, linestyle=':')
    
    for y_val in [25, 50, 75]:
        ax.axhline(y=y_val, color='gray', alpha=0.2, linewidth=0.8, linestyle=':')
    
    # Add text annotations for key points
    # At 10% of keys
    mean_at_10 = mean_curve[int(10 * num_percentile_points / 100) - 1]
    ax.annotate(f'Mean at 10% keys:\n{mean_at_10:.1f}% mass',
                xy=(10, mean_at_10), xytext=(15, mean_at_10 - 10),
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=1))
    
    # At 50% of keys
    mean_at_50 = mean_curve[int(50 * num_percentile_points / 100) - 1]
    ax.annotate(f'Mean at 50% keys:\n{mean_at_50:.1f}% mass',
                xy=(50, mean_at_50), xytext=(55, mean_at_50 + 8),
                fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=1))
    
    plt.tight_layout()
    
    # Save
    output_path = f'../results/concentration_statistics_{layer_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n💾 Saved: {output_path}")
    
    # Print key statistics
    print("\n" + "="*70)
    print("📊 KEY STATISTICS")
    print("="*70)
    
    # At different percentages of keys
    for key_pct in [1, 5, 10, 25, 50, 75, 100]:
        idx = int(key_pct * num_percentile_points / 100) - 1
        if idx < 0:
            idx = 0
        
        print(f"\nTop {key_pct}% of keys:")
        print(f"  Mean mass captured:   {mean_curve[idx]:.2f}%")
        print(f"  Median (p50):         {p50_curve[idx]:.2f}%")
        print(f"  Range (p10-p90):      {p10_curve[idx]:.2f}% - {p90_curve[idx]:.2f}%")
        print(f"  p99:                  {p99_curve[idx]:.2f}%")
    
    # Efficiency metric: how many keys needed to capture X% of mass?
    print("\n" + "="*70)
    print("🎯 EFFICIENCY METRICS")
    print("="*70)
    
    for target_mass in [50, 80, 90, 95]:
        # Find percentage of keys needed (on average)
        idx_mean = np.searchsorted(mean_curve, target_mass)
        if idx_mean < len(percentiles):
            keys_needed_mean = percentiles[idx_mean]
            print(f"\nTo capture {target_mass}% of mass (on average):")
            print(f"  Need ~{keys_needed_mean:.1f}% of keys")
            
            # For median query
            idx_median = np.searchsorted(p50_curve, target_mass)
            if idx_median < len(percentiles):
                keys_needed_median = percentiles[idx_median]
                print(f"  Median query needs ~{keys_needed_median:.1f}% of keys")
    
    print("\n" + "="*70)
    print("✅ Analysis complete!")
    print("="*70)
    
    return {
        'percentiles': percentiles,
        'mean': mean_curve,
        'p10': p10_curve,
        'p50': p50_curve,
        'p90': p90_curve,
        'p99': p99_curve,
        'num_queries_analyzed': len(query_positions)
    }

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze concentration statistics over many queries'
    )
    parser.add_argument('--file', type=str, 
                       default='../data/attention_vectors_updated.jsonl',
                       help='Path to attention vectors JSONL file')
    parser.add_argument('--layer', type=str, default='first_layer',
                       choices=['first_layer', 'last_layer'],
                       help='Which layer to analyze')
    parser.add_argument('--num-queries', type=int, default=1000,
                       help='Number of last queries to analyze (default: 1000)')
    parser.add_argument('--num-points', type=int, default=100,
                       help='Number of percentile points (default: 100)')
    
    args = parser.parse_args()
    
    try:
        analyze_concentration_statistics(
            filepath=args.file,
            layer_name=args.layer,
            num_queries=args.num_queries,
            num_percentile_points=args.num_points
        )
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



