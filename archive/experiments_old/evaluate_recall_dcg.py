#!/usr/bin/env python3
"""
Recall and DCG Evaluation

Samples 100 unique keys from different methods and measures:
- Recall@10: How many of top 10 closest keys were sampled?
- Recall@100: How many of top 100 closest keys were sampled?
- Recall@100 of top 10: How many of top 10 closest keys in the 100 sampled?
- DCG@100: Discounted Cumulative Gain of sampled keys based on original rank

Averages over last 100 queries of each of 11 examples.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time
from collections import defaultdict

import utils
import methods

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# LSH Configuration
L = 10  # Number of hash tables/trees (for prefix_sampling)
K_MAX = 30 
SEED = 42  # Random seed

# prefix_sampling Parameters
GAMMA = 1.0  # Bucket size penalty
TAU = 0.0  # Smoothing term

# Minimum Depth Values to test
MIN_DEPTH_VALUES = [0, 2, 4, 6, 8, 10, 12, 14]

# Sampling Budget (unique keys)
BUDGET = 100

# Evaluation Scale
NUM_EXAMPLES = 10  # All examples
NUM_QUERIES_PER_EXAMPLE = 100  # Last N queries per example
LAYERS_TO_TEST = ['last_layer',]

# Data Path
DATA_PATH = '../data/attention_vectors_updated_long.jsonl'

# Output
OUTPUT_DIR = '../results/approximation_evaluation/v2/recall_dcg_evaluation'

# ============================================================================
# END HYPERPARAMETERS
# ============================================================================


def compute_true_ranking(query, keys, logits):
    """
    Compute true ranking of keys by softmax weight (attention weight).
    
    This is the fair ranking based on actual attention distribution.
    
    Returns:
        ranked_indices: Indices sorted by softmax weight (highest first)
        ranks: Dictionary mapping key_index -> rank (0 = highest weight)
    """
    num_keys = len(keys)
    # Compute softmax weights (true attention distribution)
    weights = utils.softmax(logits)
    
    # Sort by weight (descending)
    ranked_indices = np.argsort(weights)[::-1]
    
    # Create rank mapping (0 = highest weight)
    ranks = {idx: rank for rank, idx in enumerate(ranked_indices)}
    
    return ranked_indices, ranks


def compute_recall_at_k(sampled_indices, top_k_indices):
    """
    Compute recall@k: fraction of top-k keys that were sampled.
    
    Args:
        sampled_indices: Set of sampled key indices
        top_k_indices: List of top-k key indices (by true ranking)
    
    Returns:
        recall: Fraction of top-k keys that were sampled
    """
    if len(top_k_indices) == 0:
        return 0.0
    
    sampled_set = set(sampled_indices)
    top_k_set = set(top_k_indices)
    
    intersection = len(sampled_set & top_k_set)
    return intersection / len(top_k_set)


def compute_dcg(sampled_indices, ranks, k=None):
    """
    Compute Discounted Cumulative Gain (DCG) of sampled keys.
    
    DCG = sum(rel_i / log2(i+1)) for i in [1, k]
    where rel_i = 1 / (rank_i + 1) (inverse rank as relevance)
    
    Args:
        sampled_indices: List of sampled key indices
        ranks: Dictionary mapping key_index -> rank (0 = highest)
        k: Number of items to consider (None = all sampled)
    
    Returns:
        dcg: DCG score
    """
    if len(sampled_indices) == 0:
        return 0.0
    
    # Get relevance scores (inverse rank)
    relevance_scores = []
    for idx in sampled_indices[:k] if k else sampled_indices:
        rank = ranks.get(idx, len(ranks))  # If not in ranking, assign worst rank
        # Relevance = 1 / (rank + 1), so top key has relevance 1.0
        relevance = 1.0 / (rank + 1)
        relevance_scores.append(relevance)
    
    # Compute DCG
    dcg = 0.0
    for i, rel in enumerate(relevance_scores):
        position = i + 1  # 1-indexed position
        dcg += rel / np.log2(position + 1)
    
    return dcg


def compute_avg_rank(sampled_indices, ranks):
    """
    Compute average rank of sampled keys.
    
    Lower is better (rank 0 = best key).
    
    Args:
        sampled_indices: List of sampled key indices
        ranks: Dictionary mapping key_index -> rank (0 = highest weight)
    
    Returns:
        avg_rank: Average rank of sampled keys
    """
    if len(sampled_indices) == 0:
        return float('inf')
    
    rank_values = []
    for idx in sampled_indices:
        rank = ranks.get(idx, len(ranks))  # If not in ranking, assign worst rank
        rank_values.append(rank)
    
    return float(np.mean(rank_values))


def sample_uniform_with_indices(query, keys, values, logits, budget):
    """Uniform sampling that returns indices."""
    num_keys = len(logits)
    budget = min(budget, num_keys)
    selected_indices = np.random.choice(num_keys, size=budget, replace=False)
    return selected_indices.tolist()


def sample_oracle_with_indices(query, keys, values, logits, true_weights, budget):
    """Oracle sampling that returns indices."""
    num_keys = len(logits)
    budget = min(budget, num_keys)
    # Sample with replacement, then get unique
    sampled_indices = np.random.choice(num_keys, size=budget * 2, p=true_weights, replace=True)
    unique_indices = list(dict.fromkeys(sampled_indices))  # Preserve order, remove duplicates
    return unique_indices[:budget]


def sample_prefix_with_indices(query, keys, values, logits, head_dim, lsh_structure, budget, gamma, tau, min_depth):
    """prefix_sampling that returns indices."""
    num_keys = len(keys)
    query_hash = lsh_structure.hash_query(query)
    key_codes = lsh_structure.hash_codes  # [num_keys, num_tables, max_depth]
    
    # Compute max depth for each key
    max_depths = np.zeros(num_keys, dtype=np.int32)
    for key_idx in range(num_keys):
        key_max_depth = 0
        for table_idx in range(lsh_structure.num_tables):
            depth = 0
            for bit_idx in range(lsh_structure.max_depth):
                if key_codes[key_idx, table_idx, bit_idx] == query_hash[table_idx, bit_idx]:
                    depth += 1
                else:
                    break
            key_max_depth = max(key_max_depth, depth)
        max_depths[key_idx] = key_max_depth
    
    # Filter by minimum depth
    valid_mask = max_depths >= min_depth
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return []
    
    valid_keys = keys[valid_indices]
    valid_logits = logits[valid_indices]
    valid_max_depths = max_depths[valid_indices]
    num_valid = len(valid_indices)
    
    # Compute cumulative bucket sizes
    cumulative_bucket_sizes = np.zeros(num_valid, dtype=np.int32)
    for depth in range(min_depth, lsh_structure.max_depth + 1):
        keys_at_least_depth = np.sum(valid_max_depths >= depth)
        keys_exactly_at_depth = (valid_max_depths == depth)
        cumulative_bucket_sizes[keys_exactly_at_depth] = keys_at_least_depth
    
    # Compute probabilities
    query_norm = np.linalg.norm(query)
    valid_key_norms = np.linalg.norm(valid_keys, axis=1)
    cos_sims = np.clip(
        (valid_keys @ query) / (query_norm * valid_key_norms + 1e-8),
        -1.0 + 1e-8, 1.0 - 1e-8
    )
    thetas = np.arccos(cos_sims)
    p_bits = 1.0 - thetas / np.pi
    
    p_retrieval = np.zeros(num_valid)
    for i in range(num_valid):
        depth_i = valid_max_depths[i]
        p_collision = p_bits[i] ** depth_i
        p_retrieval[i] = 1.0 - (1.0 - p_collision) ** lsh_structure.num_tables
    
    p_retrieval = np.clip(p_retrieval, 1e-8, 1.0)
    intensities = np.power(1.0 / (cumulative_bucket_sizes + tau), gamma)
    u_i_unnormalized = p_retrieval * intensities
    p_distribution = u_i_unnormalized / np.sum(u_i_unnormalized)
    
    # Sample
    budget = min(budget, num_valid)
    sampled_local_indices = np.random.choice(
        num_valid,
        size=budget,
        replace=False,
        p=p_distribution
    )
    
    # Convert to global indices
    sampled_global_indices = valid_indices[sampled_local_indices]
    return sampled_global_indices.tolist()


def evaluate_single_query(Q, K, V, query_pos, head_dim, lsh_union):
    """
    Evaluate recall and DCG metrics for one query.
    """
    q = Q[query_pos]
    valid_keys = K[:query_pos + 1]
    valid_values = V[:query_pos + 1]
    num_valid = len(valid_keys)
    
    if num_valid < BUDGET:
        return None  # Skip if not enough keys
    
    # Ground truth
    gt_output, gt_logits, gt_weights, _ = utils.compute_ground_truth_attention(
        q, K, V, query_pos, head_dim
    )
    
    # Compute true ranking
    ranked_indices, ranks = compute_true_ranking(q, valid_keys, gt_logits)
    top_10_indices = ranked_indices[:10].tolist()
    top_100_indices = ranked_indices[:100].tolist()
    
    # Build LSH for prefix_sampling
    lsh_union.build_index(valid_keys)
    
    results = {}
    
    # ========================================================================
    # Uniform Sampling
    # ========================================================================
    sampled_uniform = sample_uniform_with_indices(q, valid_keys, valid_values, gt_logits, BUDGET)
    if len(sampled_uniform) > 0:
        results['Uniform'] = {
            'recall_at_10': compute_recall_at_k(sampled_uniform, top_10_indices),
            'recall_at_100': compute_recall_at_k(sampled_uniform, top_100_indices),
            'recall_at_100_of_top_10': compute_recall_at_k(sampled_uniform[:100], top_10_indices),
            'dcg_at_100': compute_dcg(sampled_uniform, ranks, k=100),
            'avg_rank': compute_avg_rank(sampled_uniform, ranks)
        }
    else:
        results['Uniform'] = None
    
    # ========================================================================
    # Oracle Sampling
    # ========================================================================
    sampled_oracle = sample_oracle_with_indices(q, valid_keys, valid_values, gt_logits, gt_weights, BUDGET)
    if len(sampled_oracle) > 0:
        results['Oracle'] = {
            'recall_at_10': compute_recall_at_k(sampled_oracle, top_10_indices),
            'recall_at_100': compute_recall_at_k(sampled_oracle, top_100_indices),
            'recall_at_100_of_top_10': compute_recall_at_k(sampled_oracle[:100], top_10_indices),
            'dcg_at_100': compute_dcg(sampled_oracle, ranks, k=100),
            'avg_rank': compute_avg_rank(sampled_oracle, ranks)
        }
    else:
        results['Oracle'] = None
    
    # ========================================================================
    # prefix_sampling with different min_depth values
    # ========================================================================
    for min_depth in MIN_DEPTH_VALUES:
        try:
            sampled_prefix = sample_prefix_with_indices(
                q, valid_keys, valid_values, gt_logits, head_dim,
                lsh_union, BUDGET, GAMMA, TAU, min_depth
            )
            if len(sampled_prefix) > 0:
                results[f'prefix_sampling_min{min_depth}'] = {
                    'recall_at_10': compute_recall_at_k(sampled_prefix, top_10_indices),
                    'recall_at_100': compute_recall_at_k(sampled_prefix, top_100_indices),
                    'recall_at_100_of_top_10': compute_recall_at_k(sampled_prefix[:100], top_10_indices),
                    'dcg_at_100': compute_dcg(sampled_prefix, ranks, k=100),
                    'avg_rank': compute_avg_rank(sampled_prefix, ranks)
                }
            else:
                results[f'prefix_sampling_min{min_depth}'] = None
        except Exception as e:
            results[f'prefix_sampling_min{min_depth}'] = None
    
    return results


def aggregate_results(all_results):
    """Aggregate results across queries."""
    aggregated = {}
    
    for layer_name in all_results['results_by_layer']:
        layer_results = all_results['results_by_layer'][layer_name]
        
        # Collect all methods
        all_methods = set()
        for query_result in layer_results:
            if query_result is not None:
                all_methods.update(query_result.keys())
        
        layer_agg = {}
        for method in all_methods:
            layer_agg[method] = {
                'recall_at_10': [],
                'recall_at_100': [],
                'recall_at_100_of_top_10': [],
                'dcg_at_100': [],
                'avg_rank': []
            }
        
        # Aggregate
        for query_result in layer_results:
            if query_result is None:
                continue
            for method, metrics in query_result.items():
                if metrics is None:
                    continue
                if method in layer_agg:
                    for metric_name, value in metrics.items():
                        layer_agg[method][metric_name].append(value)
        
        # Compute statistics
        for method in layer_agg:
            for metric_name in layer_agg[method]:
                values = layer_agg[method][metric_name]
                if len(values) > 0:
                    layer_agg[method][metric_name] = {
                        'mean': float(np.mean(values)),
                        'median': float(np.median(values)),
                        'std': float(np.std(values)),
                        'n': len(values)
                    }
                else:
                    layer_agg[method][metric_name] = None
        
        aggregated[layer_name] = layer_agg
    
    return aggregated


def plot_results(aggregated_dict, output_dir):
    """Generate informative plots for recall and DCG metrics."""
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Verdana', 'Liberation Sans']
    
    layer_titles = {
        'first_layer': 'First Layer (Layer 0)',
        'last_layer': 'Last Layer (Layer 31)'
    }
    
    # Method colors
    method_colors = {
        'Uniform': '#ff7f0e',  # Orange
        'Oracle': '#2ca02c',   # Green
        'TopK': '#d62728'      # Red (if we add it)
    }
    
    # Min depth colors
    min_depth_colors = {
        0: '#FFD700',   # Yellow
        2: '#87CEEB',   # Light blue
        4: '#0000FF',   # Blue
        6: '#A52A2A',   # Brown
        8: '#808080',   # Gray
        10: '#FF1493',  # Deep pink
        12: '#8B008B',  # Dark magenta
        14: '#2F4F4F'   # Dark slate gray
    }
    
    metrics_to_plot = [
        ('recall_at_10', 'Recall@10', 'Fraction of Top 10 Keys Sampled'),
        ('recall_at_100', 'Recall@100', 'Fraction of Top 100 Keys Sampled'),
        ('recall_at_100_of_top_10', 'Recall@100 of Top 10', 'Fraction of Top 10 in 100 Sampled'),
        ('dcg_at_100', 'DCG@100', 'Discounted Cumulative Gain'),
        ('avg_rank', 'Average Rank', 'Average Rank of Sampled Keys (Lower is Better)')
    ]
    
    for layer_name, agg_data in aggregated_dict.items():
        layer_title = layer_titles.get(layer_name, layer_name.replace('_', ' ').title())
        
        for metric_key, metric_title, metric_ylabel in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Collect methods and their values
            methods_to_plot = []
            values = []
            colors_list = []
            labels_list = []
            
            # Add baseline methods
            for method_name in ['Uniform', 'Oracle']:
                if method_name in agg_data and agg_data[method_name][metric_key] is not None:
                    methods_to_plot.append(method_name)
                    values.append(agg_data[method_name][metric_key]['mean'])
                    colors_list.append(method_colors.get(method_name, '#000000'))
                    labels_list.append(method_name)
            
            # Add prefix_sampling methods
            for min_depth in MIN_DEPTH_VALUES:
                method_name = f'prefix_sampling_min{min_depth}'
                if method_name in agg_data and agg_data[method_name][metric_key] is not None:
                    methods_to_plot.append(method_name)
                    values.append(agg_data[method_name][metric_key]['mean'])
                    colors_list.append(min_depth_colors.get(min_depth, '#000000'))
                    labels_list.append(f'prefix_sampling (k={min_depth})')
            
            if len(methods_to_plot) == 0:
                continue
            
            # Create bar plot
            x_pos = np.arange(len(methods_to_plot))
            bars = ax.bar(x_pos, values, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add error bars (std)
            errors = []
            for method in methods_to_plot:
                if method in agg_data and agg_data[method][metric_key] is not None:
                    errors.append(agg_data[method][metric_key]['std'])
                else:
                    errors.append(0)
            
            ax.errorbar(x_pos, values, yerr=errors, fmt='none', color='black', capsize=5, capthick=2)
            
            # Formatting
            ax.set_xlabel('Method', fontsize=13, fontweight='bold', family='sans-serif')
            ax.set_ylabel(metric_ylabel, fontsize=13, fontweight='bold', family='sans-serif')
            ax.set_title(f'{metric_title}\n{layer_title}', fontsize=14, fontweight='bold', family='sans-serif')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(bottom=0)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            plot_path = output_dir / f'{layer_name}_{metric_key}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Generated: {plot_path}")
            plt.close()


def main():
    """Run recall and DCG evaluation."""
    
    print("="*70)
    print("RECALL AND DCG EVALUATION")
    print("="*70)
    print(f"\nMethods:")
    print(f"  1. Uniform Sampling")
    print(f"  2. Oracle Sampling")
    print(f"  3. prefix_sampling with min_depth = {MIN_DEPTH_VALUES}")
    print(f"\nMetrics:")
    print(f"  - Recall@10: Fraction of top 10 keys sampled")
    print(f"  - Recall@100: Fraction of top 100 keys sampled")
    print(f"  - Recall@100 of top 10: Fraction of top 10 in 100 sampled")
    print(f"  - DCG@100: Discounted Cumulative Gain")
    print(f"  - Average Rank: Average rank of sampled keys (lower is better)")
    print(f"\nConfiguration:")
    print(f"  Budget: {BUDGET} unique keys")
    print(f"  Examples: {NUM_EXAMPLES}")
    print(f"  Queries per example: {NUM_QUERIES_PER_EXAMPLE}")
    
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
            'metrics': ['recall_at_10', 'recall_at_100', 'recall_at_100_of_top_10', 'dcg_at_100', 'avg_rank'],
            'budget': BUDGET,
            'prefix_sampling': {'L': L, 'gamma': GAMMA, 'tau': TAU, 'min_depth_values': MIN_DEPTH_VALUES},
            'num_examples': len(examples),
            'num_queries_per_example': NUM_QUERIES_PER_EXAMPLE,
            'layers': LAYERS_TO_TEST,
            'timestamp': datetime.now().isoformat()
        },
        'results_by_layer': {layer: [] for layer in LAYERS_TO_TEST}
    }
    
    # Evaluate
    total_queries = len(examples) * NUM_QUERIES_PER_EXAMPLE * len(LAYERS_TO_TEST)
    print(f"\nEvaluating {total_queries} queries...")
    print(f"Estimated time: ~{total_queries * 1 / 60:.0f}-{total_queries * 2 / 60:.0f} minutes\n")
    
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
                if query_results is not None:
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
    print(f"✓ Full results: {full_json}")
    
    # Aggregate
    print(f"\nAggregating...")
    aggregated = aggregate_results(all_results)
    
    # Save aggregated results
    agg_json = output_dir / 'aggregated.json'
    with open(agg_json, 'w') as f:
        json.dump({'aggregated': aggregated, 'metadata': all_results['metadata']}, f, indent=2)
    print(f"✓ Aggregated: {agg_json}")
    
    # Plot
    print(f"\nGenerating plots...")
    plot_results(aggregated, output_dir)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for layer in LAYERS_TO_TEST:
        print(f"\n{layer.upper()}:")
        if layer in aggregated:
            for method in ['Uniform', 'Oracle'] + [f'prefix_sampling_min{d}' for d in MIN_DEPTH_VALUES]:
                if method in aggregated[layer]:
                    print(f"\n  {method}:")
                    for metric in ['recall_at_10', 'recall_at_100', 'recall_at_100_of_top_10', 'dcg_at_100', 'avg_rank']:
                        if aggregated[layer][method][metric] is not None:
                            stats = aggregated[layer][method][metric]
                            print(f"    {metric:25s}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['n']})")
    
    print(f"\n{'='*70}")
    print("✅ EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nFiles in {OUTPUT_DIR}:")
    print(f"  - full_results.json")
    print(f"  - aggregated.json")
    for layer in LAYERS_TO_TEST:
        for metric in ['recall_at_10', 'recall_at_100', 'recall_at_100_of_top_10', 'dcg_at_100', 'avg_rank']:
            print(f"  - {layer}_{metric}.png")


if __name__ == "__main__":
    main()

