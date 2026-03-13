#!/usr/bin/env python3
"""
Top-K vs Sampling-Based Approximation Error Analysis

Compares three methods:
1. Top-K (biased truncation)
2. Uniform Sampling (biased with subset softmax)
3. Oracle Sampling (unbiased, privileged)

Two metrics per method: weight error and output error.
Shows that Top-K can have HIGHER output error than uniform in diffuse regimes.

Output: PNG plots (linear and log scale) and printed statistics.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from algorithms.base import softmax
from visualization.plot_utils import setup_style, save_figure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MultipleLocator

# CONFIG
DATA_PATH = '../../data/attention_vectors_long_bench_llama_8b.jsonl'
OUTPUT_DIR = Path('../../results/exploration')
NUM_EXAMPLES = 1
NUM_QUERIES = 1
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
K_PERCENTAGES = [3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 95, 98]
PROFILE_BINS = 200


def relative_l1_error(approx: np.ndarray, target: np.ndarray) -> float:
    """Relative L1 error, suitable for comparing distributions."""
    return np.linalg.norm(approx - target, ord=1) / (np.linalg.norm(target, ord=1) + 1e-8)


def relative_l2_error(approx: np.ndarray, target: np.ndarray) -> float:
    """Relative L2 error, used for value aggregation vectors."""
    return np.linalg.norm(approx - target) / (np.linalg.norm(target) + 1e-8)


def resample_profile(x: np.ndarray, n_bins: int = PROFILE_BINS) -> np.ndarray:
    """Resample a 1D profile to fixed length using linear interpolation."""
    if len(x) == 0:
        return np.zeros(n_bins, dtype=np.float32)
    if len(x) == 1:
        return np.full(n_bins, x[0], dtype=np.float32)
    src = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    dst = np.linspace(0.0, 1.0, n_bins, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32)


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


def compute_grouped_min_weight_output(logits, values, weights, num_groups):
    """
    Group sorted logits into num_groups contiguous chunks.

    For each group, set w'_j to the minimum true weight in that group for all
    members j in the group, then compute output = sum_j w'_j v_j.
    """
    n_keys = len(logits)
    num_groups = max(1, min(num_groups, n_keys))

    sorted_indices = np.argsort(logits)[::-1]
    groups = np.array_split(sorted_indices, num_groups)

    approx_weights = np.zeros(n_keys, dtype=np.float32)
    for group in groups:
        if len(group) == 0:
            continue
        min_w = np.min(weights[group])
        approx_weights[group] = min_w

    approx_output = approx_weights @ values
    return approx_weights, approx_output


def compute_grouped_mean_weight_output(logits, values, weights, num_groups):
    """
    Group sorted logits into num_groups contiguous chunks.

    For each group, set w'_j to the mean true weight in that group for all
    members j in the group, then compute output = sum_j w'_j v_j.
    """
    n_keys = len(logits)
    num_groups = max(1, min(num_groups, n_keys))

    sorted_indices = np.argsort(logits)[::-1]
    groups = np.array_split(sorted_indices, num_groups)

    approx_weights = np.zeros(n_keys, dtype=np.float32)
    for group in groups:
        if len(group) == 0:
            continue
        mean_w = np.mean(weights[group])
        approx_weights[group] = mean_w

    approx_output = approx_weights @ values
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
        'grouped_min_weight': {k: [] for k in K_PERCENTAGES},
        'grouped_mean_weight': {k: [] for k in K_PERCENTAGES},
        'grouped_min_output': {k: [] for k in K_PERCENTAGES},
        'grouped_mean_output': {k: [] for k in K_PERCENTAGES},
    }
    profile_logits = []
    profile_vnorms = []

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

            # Sorted profile diagnostics for the extra panel
            sorted_idx = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_idx]
            sorted_vnorms = np.linalg.norm(valid_values[sorted_idx], axis=1)
            profile_logits.append(resample_profile(sorted_logits, PROFILE_BINS))
            profile_vnorms.append(resample_profile(sorted_vnorms, PROFILE_BINS))

            # Test different budgets
            for k_pct in K_PERCENTAGES:
                k_abs = max(1, int(np.ceil(n_keys * k_pct / 100)))
                k_abs = min(k_abs, n_keys)

                # Top-K approximation
                topk_weights, topk_output, topk_idx = compute_topk_approximation(logits, valid_values, k_abs)
                errors['weight'][k_pct].append(
                    relative_l1_error(topk_weights, full_weights)
                )
                errors['output'][k_pct].append(
                    relative_l2_error(topk_output, full_output)
                )

                # Uniform sampling
                uniform_weights, uniform_output = compute_uniform_sampling(logits, valid_values, full_weights, k_abs)
                errors['uniform_weight'][k_pct].append(
                    relative_l1_error(uniform_weights, full_weights)
                )
                errors['uniform_output'][k_pct].append(
                    relative_l2_error(uniform_output, full_output)
                )

                # Oracle sampling
                oracle_weights, oracle_output = compute_oracle_sampling(logits, valid_values, full_weights, k_abs)
                errors['oracle_weight'][k_pct].append(
                    relative_l1_error(oracle_weights, full_weights)
                )
                errors['oracle_output'][k_pct].append(
                    relative_l2_error(oracle_output, full_output)
                )

                # Grouped-min weight approximation (deterministic)
                grouped_weights, grouped_output = compute_grouped_min_weight_output(
                    logits, valid_values, full_weights, num_groups=k_abs
                )
                errors['grouped_min_weight'][k_pct].append(
                    relative_l1_error(grouped_weights, full_weights)
                )
                errors['grouped_min_output'][k_pct].append(
                    relative_l2_error(grouped_output, full_output)
                )

                # Grouped-mean weight approximation (deterministic)
                grouped_mean_weights, grouped_mean_output = compute_grouped_mean_weight_output(
                    logits, valid_values, full_weights, num_groups=k_abs
                )
                errors['grouped_mean_weight'][k_pct].append(
                    relative_l1_error(grouped_mean_weights, full_weights)
                )
                errors['grouped_mean_output'][k_pct].append(
                    relative_l2_error(grouped_mean_output, full_output)
                )

    print(f"  Analyzed {len(query_positions) * len(examples)} queries")

    # Compute statistics
    k_vals = sorted(K_PERCENTAGES)
    result = {'k_percentages': k_vals}
    for err_type in errors:
        result[f'{err_type}_mean'] = [np.mean(errors[err_type][k]) for k in k_vals]
        result[f'{err_type}_std'] = [np.std(errors[err_type][k]) for k in k_vals]

    # Profile summary for diagnostics panel
    profile_logits = np.stack(profile_logits, axis=0)
    profile_vnorms = np.stack(profile_vnorms, axis=0)
    result['profile_rank_percent'] = np.linspace(0.0, 100.0, PROFILE_BINS).tolist()
    result['profile_logits_mean'] = profile_logits.mean(axis=0).tolist()
    result['profile_logits_std'] = profile_logits.std(axis=0).tolist()
    result['profile_vnorms_mean'] = profile_vnorms.mean(axis=0).tolist()
    result['profile_vnorms_std'] = profile_vnorms.std(axis=0).tolist()

    return result


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_plot(data_first, data_last, use_log_scale=False):
    """
    Create publication-quality 5-panel figure.

    Layout:
    - Row 1: First layer (weight error panel, value error panel)
    - Row 2: Last layer (weight error panel, value error panel)
    - Row 3: Shared diagnostics (sorted logits + corresponding value norms)

    Error panels show method curves for:
    - Purple: Top-K
    - Orange: Uniform
    - Green: Oracle
    - Blue: Grouped-Min
    - Teal: Grouped-Mean

    Args:
        use_log_scale: If True, use log y-axis; if False, use linear y-axis
    """
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.85])
    axes = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ])
    diag_ax = fig.add_subplot(gs[2, :])

    weight_curves = [
        ('weight', 'Top-K Weights', '#8b5cf6', 'o', '-', 2.5),
        ('uniform_weight', 'Uniform Weights', '#f97316', '^', '-.', 2.3),
        ('oracle_weight', 'Oracle Weights', '#16a34a', 'D', '--', 2.3),
        ('grouped_min_weight', 'Grouped-Min Weights', '#2563eb', 'X', ':', 2.5),
        ('grouped_mean_weight', 'Grouped-Mean Weights', '#0f766e', 'h', '-', 2.3),
    ]
    value_curves = [
        ('output', 'Top-K Value Aggregation', '#c084fc', 's', '-', 2.5),
        ('uniform_output', 'Uniform Value Aggregation', '#fb923c', 'v', '-.', 2.3),
        ('oracle_output', 'Oracle Value Aggregation', '#4ade80', 'p', '--', 2.3),
        ('grouped_min_output', 'Grouped-Min Weights Aggregation', '#2563eb', 'X', ':', 2.5),
        ('grouped_mean_output', 'Grouped-Mean Weights Aggregation', '#0f766e', 'h', '-', 2.3),
    ]

    panel_specs = [
        (axes[0, 0], data_first, 'First Layer (Layer 0) - Weight Error', weight_curves, 'Relative L1 Error'),
        (axes[0, 1], data_first, 'First Layer (Layer 0) - Value Error', value_curves, 'Relative L2 Error'),
        (axes[1, 0], data_last, 'Last Layer (Layer 31) - Weight Error', weight_curves, 'Relative L1 Error'),
        (axes[1, 1], data_last, 'Last Layer (Layer 31) - Value Error', value_curves, 'Relative L2 Error'),
    ]

    for ax, data, panel_title, curves, y_label in panel_specs:

        x = np.array(data['k_percentages'])  # Include 100%

        for err_key, label, color, marker, linestyle, linewidth in curves:
            means = np.array(data[f'{err_key}_mean'])  # Include 100%
            stds = np.array(data[f'{err_key}_std'])

            ax.plot(x, means, marker=marker, linewidth=linewidth, markersize=4.5,
                    color=color, label=label, alpha=0.9, linestyle=linestyle)
            ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.1)

        ax.set_xlabel('Budget (% of keys)', fontweight='bold', fontsize=12)
        ax.set_ylabel(y_label, fontweight='bold', fontsize=12)
        ax.set_title(panel_title, fontweight='bold', fontsize=13, pad=12)
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
            if y_max <= 2:
                tick_interval = 0.2
            elif y_max <= 5:
                tick_interval = 0.5
            else:
                tick_interval = 1.0
            ax.yaxis.set_major_locator(MultipleLocator(tick_interval))

        ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
        ax.legend(loc='upper right', framealpha=0.95, fontsize=8, edgecolor='black', ncol=1)

    # Shared diagnostics panel: sorted logits and corresponding value norms
    rank = np.array(data_first['profile_rank_percent'])
    first_logits_mean = np.array(data_first['profile_logits_mean'])
    first_logits_std = np.array(data_first['profile_logits_std'])
    first_vnorm_mean = np.array(data_first['profile_vnorms_mean'])
    first_vnorm_std = np.array(data_first['profile_vnorms_std'])

    last_logits_mean = np.array(data_last['profile_logits_mean'])
    last_logits_std = np.array(data_last['profile_logits_std'])
    last_vnorm_mean = np.array(data_last['profile_vnorms_mean'])
    last_vnorm_std = np.array(data_last['profile_vnorms_std'])

    diag_ax.plot(rank, first_logits_mean, color='#0ea5e9', linewidth=2.2, label='First Layer Sorted Logits')
    diag_ax.fill_between(rank, first_logits_mean - first_logits_std, first_logits_mean + first_logits_std,
                         color='#0ea5e9', alpha=0.12)
    diag_ax.plot(rank, last_logits_mean, color='#6366f1', linewidth=2.2, label='Last Layer Sorted Logits')
    diag_ax.fill_between(rank, last_logits_mean - last_logits_std, last_logits_mean + last_logits_std,
                         color='#6366f1', alpha=0.12)
    diag_ax.set_xlabel('Rank Percentile (sorted by logit, high to low)', fontweight='bold', fontsize=11)
    diag_ax.set_ylabel('Logit Value', fontweight='bold', fontsize=11, color='#334155')
    diag_ax.tick_params(axis='y', labelcolor='#334155')

    diag_ax2 = diag_ax.twinx()
    diag_ax2.plot(rank, first_vnorm_mean, color='#f59e0b', linewidth=2.2, linestyle='--',
                  label='First Layer ||v||_2')
    diag_ax2.fill_between(rank, first_vnorm_mean - first_vnorm_std, first_vnorm_mean + first_vnorm_std,
                          color='#f59e0b', alpha=0.1)
    diag_ax2.plot(rank, last_vnorm_mean, color='#16a34a', linewidth=2.2, linestyle='--',
                  label='Last Layer ||v||_2')
    diag_ax2.fill_between(rank, last_vnorm_mean - last_vnorm_std, last_vnorm_mean + last_vnorm_std,
                          color='#16a34a', alpha=0.1)
    diag_ax2.set_ylabel('Value Vector Norm ||v||_2', fontweight='bold', fontsize=11, color='#3f3f46')
    diag_ax2.tick_params(axis='y', labelcolor='#3f3f46')

    # Merge legends from both y-axes
    handles1, labels1 = diag_ax.get_legend_handles_labels()
    handles2, labels2 = diag_ax2.get_legend_handles_labels()
    diag_ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right', framealpha=0.95, fontsize=8, edgecolor='black', ncol=2)
    diag_ax.set_title('Sorted Logits with Superimposed Corresponding Value Norms', fontweight='bold', fontsize=13, pad=10)
    diag_ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Explanatory text above plots
    scale_note = "Linear y-axis shows true scale of errors." if not use_log_scale else "Log y-axis emphasizes relative differences."
    explanation = [
        "Top-K (purple): Select K highest-logit keys, subset softmax (biased). Uniform (orange): Sample K uniformly, subset softmax (biased). Oracle (green): sample from true distribution.",
        "Oracle (green): Sample K from true distribution, simple average (unbiased, privileged). Sampling variance prevents exact zero at 100%.",
        "Blue/teal grouped curves: sort logits, split into N*B groups, assign each group its min or mean true weight, then evaluate both weight and output errors.",
        f"Panels are separated by metric: Weights use relative L1 error; Value Aggregation uses relative L2 error. Bottom panel overlays sorted logits and ||v||_2 profiles. {scale_note}",
        "Note: In diffuse attention, Top-K's missing mass bias can exceed uniform sampling's random error (see first layer)."
    ]

    title_suffix = " (Linear Scale)" if not use_log_scale else " (Log Scale)"
    fig.text(0.5, 0.985, f"Top-K vs Sampling: Weight/Value Errors + Logit/Value-Norm Diagnostics{title_suffix}", ha='center', fontsize=13, fontweight='bold')
    for i, line in enumerate(explanation):
        fig.text(0.5, 0.96 - i*0.015, line, ha='center', fontsize=8.5, style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.88])
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

    setup_style()
    np.random.seed(SEED)

    # Resolve data path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, DATA_PATH) if not os.path.isabs(DATA_PATH) else DATA_PATH
    output_dir = Path(os.path.join(script_dir, str(OUTPUT_DIR))) if not OUTPUT_DIR.is_absolute() else OUTPUT_DIR

    # Count and select
    print(f"Counting examples in: {data_path}")
    with open(data_path, 'r') as f:
        total = sum(1 for _ in f)
    print(f"Found {total} examples")

    selected_indices = sorted(np.random.choice(total, NUM_EXAMPLES, replace=False).tolist())
    print(f"Selected {NUM_EXAMPLES} random indices")

    # Load only selected
    print(f"Loading selected examples...")
    selected_set = set(selected_indices)
    examples = []
    with open(data_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx in selected_set:
                examples.append(json.loads(line))
            if len(examples) >= NUM_EXAMPLES:
                break
    print(f"Loaded {len(examples)} examples")

    # Analyze
    results = {}
    for layer_name in LAYERS:
        results[layer_name] = analyze_layer(examples, layer_name)

    # Generate both versions
    print(f"\nGenerating plots...")

    # Linear scale
    print("  Creating linear scale plot...")
    fig_linear = create_plot(results['first_layer'], results['last_layer'], use_log_scale=False)
    output_linear = output_dir / 'topk_approximation_error_linear.png'
    save_figure(fig_linear, output_linear, dpi=220)

    # Log scale
    print("  Creating log scale plot...")
    fig_log = create_plot(results['first_layer'], results['last_layer'], use_log_scale=True)
    output_log = output_dir / 'topk_approximation_error_log.png'
    save_figure(fig_log, output_log, dpi=220)

    print("\nBoth plots generated!")
    print(f"  - Linear scale: {output_linear}")
    print(f"  - Log scale: {output_log}")
    print("\nDone!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
