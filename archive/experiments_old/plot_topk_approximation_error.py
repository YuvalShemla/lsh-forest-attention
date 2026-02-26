#!/usr/bin/env python3
"""
Top-K vs Sampling-Based Approximation Error Analysis

Compares three approximation methods for attention computation:

1. TOP-K (biased):
   - Select K highest-logit keys
   - Renormalize with subset softmax
   - Missing mass bias from ignoring low-logit keys

2. UNIFORM SAMPLING (biased with subset softmax):
   - Sample K keys uniformly at random (no replacement)
   - Apply subset softmax on sampled keys
   - Unbiased selection but biased renormalization

3. ORACLE SAMPLING (unbiased, privileged):
   - Sample K keys from TRUE attention distribution (with replacement)
   - Simple average estimator (no renormalization)
   - MagicPIG Definition 3.1 - requires knowing true distribution

Two metrics per method:
- Softmax Weights: L2 error in attention distribution
- Value Aggregation: L2 error in final output (weighted sum of values)

Key Finding:
In diffuse attention regimes, Top-K can have HIGHER output error than uniform sampling
due to missing mass bias, despite having lower weight error.

Output: Side-by-side plots for First Layer (Layer 0) vs Last Layer (Layer 31).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = '../data/attention_vectors_updated_long.jsonl'
OUTPUT_DIR = Path('../results')
NUM_EXAMPLES = 100  # Random sample from 503 examples
NUM_QUERIES = 100   # Last 100 queries per example
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42

# Budget percentages to test
K_PERCENTAGES = [3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 95,98, 100]

# ============================================================================
# PLOTTING SETUP
# ============================================================================
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

np.random.seed(SEED)

# ============================================================================
# ATTENTION COMPUTATION
# ============================================================================

def softmax(x):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax(x):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ============================================================================
# APPROXIMATION METHODS
# ============================================================================

def compute_topk_approximation(logits, values, k):
    """
    Top-K approximation with subset softmax (biased).
    
    Selects K keys with highest logits, applies softmax to subset only,
    then computes weighted sum. This is the standard "Top-K attention" baseline.
    
    Returns:
        approx_weights: [n_keys] - subset softmax weights (zeros for non-top-K)
        approx_output: [head_dim] - weighted sum using subset softmax
        top_k_indices: indices of selected keys
    """
    n_keys = len(logits)
    k = min(k, n_keys)
    
    top_k_indices = np.argpartition(logits, -k)[-k:]
    subset_logits = logits[top_k_indices]
    subset_weights = softmax(subset_logits)
    subset_values = values[top_k_indices]
    approx_output = subset_weights @ subset_values
    
    approx_weights = np.zeros(n_keys)
    approx_weights[top_k_indices] = subset_weights
    
    return approx_weights, approx_output, top_k_indices

def compute_uniform_sampling(logits, values, weights, budget):
    """
    Uniform random sampling with subset softmax (biased).
    
    Matches methods.naive_sampling: samples K keys uniformly (no replacement),
    applies subset softmax, computes weighted sum. This tests whether random
    selection is better than Top-K in diffuse attention regimes.
    
    Returns:
        approx_weights: [n_keys] - subset softmax weights (zeros for non-sampled)
        approx_output: [head_dim] - weighted sum using subset softmax
    """
    n_keys = len(logits)
    budget = min(budget, n_keys)
    
    # Sample uniformly WITHOUT replacement
    selected_indices = np.random.choice(n_keys, size=budget, replace=False)
    
    # Subset softmax (same as Top-K but random selection)
    selected_logits = logits[selected_indices]
    selected_values = values[selected_indices]
    subset_weights = softmax(selected_logits)
    approx_output = subset_weights @ selected_values
    
    # Create full weight vector
    approx_weights = np.zeros(n_keys)
    approx_weights[selected_indices] = subset_weights
    
    return approx_weights, approx_output

def compute_oracle_sampling(logits, values, weights, budget):
    """
    Oracle sampling (unbiased, privileged) - MagicPIG Definition 3.1.
    
    Samples K indices from TRUE attention distribution (with replacement),
    uses simple average estimator. This is the theoretical lower bound - 
    assumes we know the exact attention distribution (privileged information).
    
    Returns:
        approx_weights: [n_keys] - estimated by counting samples (has variance)
        approx_output: [head_dim] - simple average of sampled values (unbiased)
    """
    n_keys = len(logits)
    budget = min(budget, n_keys)
    
    # Sample from true distribution WITH replacement
    sampled_indices = np.random.choice(n_keys, size=budget, p=weights, replace=True)
    
    # Simple average estimator (unbiased!)
    sampled_values = values[sampled_indices]
    approx_output = np.mean(sampled_values, axis=0)
    
    # Estimate weights by counting samples
    approx_weights = np.zeros(n_keys)
    unique, counts = np.unique(sampled_indices, return_counts=True)
    approx_weights[unique] = counts / budget
    
    return approx_weights, approx_output

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_layer(examples, layer_name):
    """
    Analyze approximation errors across examples and queries.
    
    For each query:
    1. Compute full attention (ground truth)
    2. Test each method at different budget percentages
    3. Measure weight error and output error
    
    Returns aggregated statistics (mean and std) across all queries.
    """
    
    print(f"\n{'='*70}")
    print(f"Analyzing {layer_name}")
    print('='*70)
    
    errors = {
        'weight': {k: [] for k in K_PERCENTAGES},
        'output': {k: [] for k in K_PERCENTAGES},
        'uniform_weight': {k: [] for k in K_PERCENTAGES},
        'uniform_output': {k: [] for k in K_PERCENTAGES},
        'oracle_weight': {k: [] for k in K_PERCENTAGES},
        'oracle_output': {k: [] for k in K_PERCENTAGES},
    }
    
    for ex_idx, example in enumerate(examples):
        print(f"  Example {ex_idx+1}/{len(examples)}: {example.get('domain', '?')[:40]}...")
        
        Q = np.array(example[layer_name]['Q'], dtype=np.float32)
        K_mat = np.array(example[layer_name]['K'], dtype=np.float32)
        V = np.array(example[layer_name]['V'], dtype=np.float32)
        seq_len = Q.shape[0]
        
        query_positions = list(range(seq_len - NUM_QUERIES, seq_len))
        
        for query_pos in query_positions:
            q = Q[query_pos]
            valid_keys = K_mat[:query_pos + 1]
            valid_values = V[:query_pos + 1]
            n_keys = len(valid_keys)
            
            # Full attention (ground truth)
            logits = (q @ valid_keys.T) / np.sqrt(HEAD_DIM)
            full_weights = softmax(logits)
            full_output = full_weights @ valid_values
            
            # Test different budgets
            for k_pct in K_PERCENTAGES:
                k_abs = max(1, int(np.ceil(n_keys * k_pct / 100)))
                k_abs = min(k_abs, n_keys)
                
                # Top-K approximation
                topk_weights, topk_output, topk_idx = compute_topk_approximation(logits, valid_values, k_abs)
                errors['weight'][k_pct].append(
                    np.linalg.norm(topk_weights - full_weights) / (np.linalg.norm(full_weights) + 1e-8)
                )
                errors['output'][k_pct].append(
                    np.linalg.norm(topk_output - full_output) / (np.linalg.norm(full_output) + 1e-8)
                )
                
                # Uniform sampling
                uniform_weights, uniform_output = compute_uniform_sampling(logits, valid_values, full_weights, k_abs)
                errors['uniform_weight'][k_pct].append(
                    np.linalg.norm(uniform_weights - full_weights) / (np.linalg.norm(full_weights) + 1e-8)
                )
                errors['uniform_output'][k_pct].append(
                    np.linalg.norm(uniform_output - full_output) / (np.linalg.norm(full_output) + 1e-8)
                )
                
                # Oracle sampling
                oracle_weights, oracle_output = compute_oracle_sampling(logits, valid_values, full_weights, k_abs)
                errors['oracle_weight'][k_pct].append(
                    np.linalg.norm(oracle_weights - full_weights) / (np.linalg.norm(full_weights) + 1e-8)
                )
                errors['oracle_output'][k_pct].append(
                    np.linalg.norm(oracle_output - full_output) / (np.linalg.norm(full_output) + 1e-8)
                )
    
    print(f"  ✓ Analyzed {len(query_positions) * len(examples)} queries")
    
    # Compute statistics
    k_vals = sorted(K_PERCENTAGES)
    result = {'k_percentages': k_vals}
    for err_type in errors:
        result[f'{err_type}_mean'] = [np.mean(errors[err_type][k]) for k in k_vals]
        result[f'{err_type}_std'] = [np.std(errors[err_type][k]) for k in k_vals]
    
    return result

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_plot(data_first, data_last, use_log_scale=False):
    """
    Create publication-quality side-by-side plots.
    
    Shows 6 curves (3 methods × 2 metrics) for each layer:
    - Purple: Top-K (biased)
    - Orange: Uniform (biased with subset softmax)
    - Green: Oracle (unbiased, privileged)
    
    Args:
        use_log_scale: If True, use log y-axis; if False, use linear y-axis
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Curve definitions - grouped by method with color shades
    curves = [
        ('weight', 'Top-K Softmax Weights', '#8b5cf6', 'o', '-', 2.5),      # Purple
        ('output', 'Top-K Value Aggregation', '#c084fc', 's', '-', 2.5),    # Light purple
        ('uniform_weight', 'Uniform Softmax Weights', '#f97316', '^', '-.', 2.3),  # Orange
        ('uniform_output', 'Uniform Value Aggregation', '#fb923c', 'v', '-.', 2.3), # Light orange
        ('oracle_weight', 'Oracle Softmax Weights', '#16a34a', 'D', '--', 2.3),    # Green
        ('oracle_output', 'Oracle Value Aggregation', '#4ade80', 'p', '--', 2.3),  # Light green
    ]
    
    for ax, data, layer_title in [(axes[0], data_first, 'First Layer (Layer 0)'),
                                   (axes[1], data_last, 'Last Layer (Layer 31)')]:
        
        x = np.array(data['k_percentages'])  # Include 100%
        
        for err_key, label, color, marker, linestyle, linewidth in curves:
            means = np.array(data[f'{err_key}_mean'])  # Include 100%
            stds = np.array(data[f'{err_key}_std'])
            
            ax.plot(x, means, marker=marker, linewidth=linewidth, markersize=4.5,
                    color=color, label=label, alpha=0.9, linestyle=linestyle)
            ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.1)
        
        ax.set_xlabel('Budget (% of keys)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Relative L2 Error', fontweight='bold', fontsize=12)
        ax.set_title(layer_title, fontweight='bold', fontsize=13, pad=12)
        ax.set_xlim([0, 105])
        
        # Set y-axis scale based on parameter
        all_means = []
        for err_key, _, _, _, _, _ in curves:
            all_means.extend(data[f'{err_key}_mean'])
        
        if use_log_scale:
            # Log scale
            ax.set_yscale('log')
            y_max = max(all_means) * 1.5
            y_min = min([m for m in all_means if m > 0]) * 0.4
            ax.set_ylim([y_min, y_max])
            
            # Format log scale y-axis
            from matplotlib.ticker import FuncFormatter
            def format_func(value, tick_number):
                if value >= 1:
                    return f'{value:.0f}'
                elif value >= 0.01:
                    return f'{value:.2f}'
                else:
                    return f'{value:.3f}'
            ax.yaxis.set_major_formatter(FuncFormatter(format_func))
        else:
            # Linear scale - show true differences
            y_max = np.ceil(max(all_means) * 1.15 * 10) / 10  # Round up to nearest 0.1
            ax.set_ylim([0, y_max])
            
            # Fixed interval ticks
            from matplotlib.ticker import MultipleLocator
            if y_max <= 2:
                tick_interval = 0.2
            elif y_max <= 5:
                tick_interval = 0.5
            else:
                tick_interval = 1.0
            ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
        
        ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
        ax.legend(loc='upper right', framealpha=0.95, fontsize=7, edgecolor='black', ncol=1)
    
    # Explanatory text above plots
    scale_note = "Linear y-axis shows true scale of errors." if not use_log_scale else "Log y-axis emphasizes relative differences."
    explanation = [
        "Top-K (purple): Select K highest-logit keys, subset softmax (biased). Uniform (orange): Sample K uniformly, subset softmax (biased).",
        "Oracle (green): Sample K from true distribution, simple average (unbiased, privileged). Sampling variance prevents exact zero at 100%.",
        f"Softmax Weights = attention distribution error. Value Aggregation = final output error. {scale_note}",
        "Note: In diffuse attention, Top-K's missing mass bias can exceed uniform sampling's random error (see first layer)."
    ]
    
    title_suffix = " (Linear Scale)" if not use_log_scale else " (Log Scale)"
    fig.text(0.5, 0.985, f"Six Approximation Curves (3 Methods × 2 Metrics){title_suffix}", ha='center', fontsize=13, fontweight='bold')
    for i, line in enumerate(explanation):
        fig.text(0.5, 0.96 - i*0.015, line, ha='center', fontsize=8.5, style='italic')
    
    plt.tight_layout(rect=[0, 0, 1, 0.89])
    return fig

# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main execution:
    1. Load random sample of examples (memory efficient)
    2. Analyze both layers
    3. Generate comparison plots
    """
    print("="*70)
    print("TOP-K VS SAMPLING APPROXIMATION ERROR ANALYSIS")
    print("="*70)
    print(f"Config: {NUM_EXAMPLES} examples, {NUM_QUERIES} queries/example")
    print(f"Total queries: {NUM_EXAMPLES * NUM_QUERIES}")
    print(f"Budgets tested: {K_PERCENTAGES}")
    print()
    
    # Count and select
    print(f"Counting examples in: {DATA_PATH}")
    with open(DATA_PATH, 'r') as f:
        total = sum(1 for _ in f)
    print(f"✓ Found {total} examples")
    
    selected_indices = sorted(np.random.choice(total, NUM_EXAMPLES, replace=False).tolist())
    print(f"✓ Selected {NUM_EXAMPLES} random indices")
    
    # Load only selected
    print(f"Loading selected examples...")
    selected_set = set(selected_indices)
    examples = []
    with open(DATA_PATH, 'r') as f:
        for idx, line in enumerate(f):
            if idx in selected_set:
                examples.append(json.loads(line))
            if len(examples) >= NUM_EXAMPLES:
                break
    print(f"✓ Loaded {len(examples)} examples")
    
    # Analyze
    results = {}
    for layer_name in LAYERS:
        results[layer_name] = analyze_layer(examples, layer_name)
    
    # Generate both versions
    print(f"\nGenerating plots...")
    
    # Linear scale
    print("  Creating linear scale plot...")
    fig_linear = create_plot(results['first_layer'], results['last_layer'], use_log_scale=False)
    output_linear = OUTPUT_DIR / 'topk_approximation_error_linear.png'
    fig_linear.savefig(output_linear, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_linear}")
    
    # Log scale
    print("  Creating log scale plot...")
    fig_log = create_plot(results['first_layer'], results['last_layer'], use_log_scale=True)
    output_log = OUTPUT_DIR / 'topk_approximation_error_log.png'
    fig_log.savefig(output_log, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_log}")
    
    print("\n✓ Both plots generated!")
    print(f"  - Linear scale: {output_linear}")
    print(f"  - Log scale: {output_log}")
    print("\nDone!")

if __name__ == "__main__":
    main()
