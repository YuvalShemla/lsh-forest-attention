#!/usr/bin/env python3
"""
Key-Value Norm Relationship Analysis

Investigates:
1. Are key and value norms correlated (position by position)?
2. Do top-K keys (by attention logit) have high-norm values?
3. Per-sequence K-V norm correlation distribution

Output: PNG plot and printed statistics.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from scipy import stats
from algorithms.base import softmax
from visualization.plot_utils import setup_style, save_figure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# CONFIG
DATA_PATH = '../../data/attention_vectors_long_bench_llama_8b.jsonl'
OUTPUT_DIR = Path('../../results/exploration')
NUM_EXAMPLES = 20
NUM_QUERIES = 100
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_layer(examples, layer_name):
    """Analyze key-value norm relationship."""

    print(f"\n{'='*70}")
    print(f"Analyzing {layer_name}")
    print('='*70)

    # Accumulators
    key_norms_all = []
    value_norms_all = []
    key_value_norm_corrs = []  # per-sequence correlation

    # For top-K analysis: do high-weight keys have high-norm values?
    top_key_norms = []
    top_value_norms = []

    for ex_idx, example in enumerate(examples):
        print(f"  Example {ex_idx+1}/{len(examples)}...")

        Q = np.array(example[layer_name]['Q'], dtype=np.float32)
        K = np.array(example[layer_name]['K'], dtype=np.float32)
        V = np.array(example[layer_name]['V'], dtype=np.float32)
        seq_len = Q.shape[0]

        # Per-position norms
        k_norms = np.linalg.norm(K, axis=1)
        v_norms = np.linalg.norm(V, axis=1)

        key_norms_all.extend(k_norms)
        value_norms_all.extend(v_norms)

        # Correlation within this sequence
        if len(k_norms) > 10:
            corr = np.corrcoef(k_norms, v_norms)[0, 1]
            key_value_norm_corrs.append(corr)

        # For queries: are top-K keys associated with high-norm or low-norm values?
        query_positions = list(range(seq_len - NUM_QUERIES, seq_len))

        for query_pos in query_positions:
            q = Q[query_pos]
            valid_keys = K[:query_pos + 1]
            valid_values = V[:query_pos + 1]
            n_keys = len(valid_keys)

            if n_keys < 100:
                continue

            # Compute logits
            logits = (q @ valid_keys.T) / np.sqrt(HEAD_DIM)

            # Get top-10% by attention logit
            k_top = max(10, int(n_keys * 0.1))
            top_indices = np.argsort(logits)[-k_top:]

            # Get norms of top keys and their corresponding values
            top_k_norms = np.linalg.norm(valid_keys[top_indices], axis=1)
            top_v_norms = np.linalg.norm(valid_values[top_indices], axis=1)

            top_key_norms.extend(top_k_norms)
            top_value_norms.extend(top_v_norms)

    print(f"  Analyzed {len(key_norms_all)} positions, {len(top_key_norms)} top-K pairs")

    return {
        'key_norms': np.array(key_norms_all),
        'value_norms': np.array(value_norms_all),
        'kv_norm_corrs': np.array(key_value_norm_corrs),
        'top_key_norms': np.array(top_key_norms),
        'top_value_norms': np.array(top_value_norms),
    }


# ============================================================================
# PLOTTING
# ============================================================================

def create_plots(data_first, data_last):
    """Create comprehensive norm relationship plots."""

    fig = plt.figure(figsize=(18, 12))

    # ============================================================
    # ROW 1: Key-Value Norm Correlation (all positions)
    # ============================================================
    for col, (data, layer_title) in enumerate([(data_first, 'First Layer (Layer 0)'),
                                                 (data_last, 'Last Layer (Layer 31)')]):
        ax = plt.subplot(3, 2, col + 1)

        # Subsample for plotting
        n_sample = min(10000, len(data['key_norms']))
        idx = np.random.choice(len(data['key_norms']), n_sample, replace=False)
        k_norms = data['key_norms'][idx]
        v_norms = data['value_norms'][idx]

        # Hexbin density plot
        hb = ax.hexbin(k_norms, v_norms, gridsize=50, cmap='viridis', mincnt=1, alpha=0.8)

        # Add regression line
        slope, intercept = np.polyfit(k_norms, v_norms, 1)
        x_line = np.array([k_norms.min(), k_norms.max()])
        ax.plot(x_line, slope * x_line + intercept, 'r--', linewidth=2.5, alpha=0.9, label='Linear fit')

        # Correlation
        r = np.corrcoef(data['key_norms'], data['value_norms'])[0, 1]
        ax.text(0.05, 0.95, f'Pearson r = {r:.3f}', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

        ax.set_xlabel('||Key||_2', fontweight='bold', fontsize=11)
        ax.set_ylabel('||Value||_2', fontweight='bold', fontsize=11)
        ax.set_title(f'{layer_title}\nKey-Value Norm Correlation (All Positions)', fontweight='bold', fontsize=12)
        ax.legend(fontsize=9)
        plt.colorbar(hb, ax=ax, label='Density')

    # ============================================================
    # ROW 2: Per-Sequence Correlation Distribution
    # ============================================================
    for col, (data, layer_title) in enumerate([(data_first, 'First Layer'), (data_last, 'Last Layer')]):
        ax = plt.subplot(3, 2, col + 3)

        if len(data['kv_norm_corrs']) > 0:
            ax.hist(data['kv_norm_corrs'], bins=40, color='#f59e0b', alpha=0.75,
                    edgecolor='black', linewidth=0.6)
            mean_corr = np.mean(data['kv_norm_corrs'])
            ax.axvline(mean_corr, color='red', linestyle='--', linewidth=2.5,
                       label=f'Mean: {mean_corr:.3f}')
            ax.set_xlabel('Pearson r (within sequence)', fontweight='bold', fontsize=11)
            ax.set_ylabel('Frequency (# sequences)', fontweight='bold', fontsize=11)
            ax.set_title(f'{layer_title}\nPer-Sequence K-V Norm Correlation', fontweight='bold', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2)

    # ============================================================
    # ROW 3: Top-K Keys - Do they have high-norm values?
    # ============================================================
    for col, (data, layer_title) in enumerate([(data_first, 'First Layer'), (data_last, 'Last Layer')]):
        ax = plt.subplot(3, 2, col + 5)

        # Subsample
        n_sample = min(5000, len(data['top_key_norms']))
        idx = np.random.choice(len(data['top_key_norms']), n_sample, replace=False)
        top_k = data['top_key_norms'][idx]
        top_v = data['top_value_norms'][idx]

        # Hexbin
        hb = ax.hexbin(top_k, top_v, gridsize=40, cmap='plasma', mincnt=1, alpha=0.8)

        # Regression
        slope, intercept = np.polyfit(top_k, top_v, 1)
        x_line = np.array([top_k.min(), top_k.max()])
        ax.plot(x_line, slope * x_line + intercept, 'r--', linewidth=2.5, alpha=0.9)

        r = np.corrcoef(data['top_key_norms'], data['top_value_norms'])[0, 1]
        ax.text(0.05, 0.95, f'Pearson r = {r:.3f}', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

        ax.set_xlabel('||Key||_2 (for top-10% keys by logit)', fontweight='bold', fontsize=11)
        ax.set_ylabel('||Value||_2 (corresponding values)', fontweight='bold', fontsize=11)
        ax.set_title(f'{layer_title}\nTop Keys: Do They Have High-Norm Values?', fontweight='bold', fontsize=12)
        plt.colorbar(hb, ax=ax, label='Density')

    fig.suptitle('Key-Value Norm Relationship: Does Scale Alignment Matter?',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig


# ============================================================================
# PRINT ANALYSIS
# ============================================================================

def print_analysis(data_first, data_last):
    """Print numerical analysis."""

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    for layer_name, data in [('First Layer', data_first), ('Last Layer', data_last)]:
        print(f"\n{layer_name}:")
        print("-" * 70)

        # Overall correlation
        r_all = np.corrcoef(data['key_norms'], data['value_norms'])[0, 1]
        print(f"  Overall K-V norm correlation: {r_all:.3f}")

        # Mean correlation per sequence
        if len(data['kv_norm_corrs']) > 0:
            print(f"  Mean per-sequence correlation: {np.mean(data['kv_norm_corrs']):.3f} +/- {np.std(data['kv_norm_corrs']):.3f}")

        # For top-K keys
        r_top = np.corrcoef(data['top_key_norms'], data['top_value_norms'])[0, 1]
        print(f"  Top-K keys K-V norm correlation: {r_top:.3f}")

        # Scale ratios
        k_mean = np.mean(data['key_norms'])
        v_mean = np.mean(data['value_norms'])
        print(f"  Mean ||K||: {k_mean:.2f}, Mean ||V||: {v_mean:.2f}, Ratio: {k_mean/v_mean:.3f}")

        # Coefficient of variation
        k_cv = np.std(data['key_norms']) / k_mean
        v_cv = np.std(data['value_norms']) / v_mean
        print(f"  Key norm CV: {k_cv:.3f}, Value norm CV: {v_cv:.3f}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
High K-V norm correlation (r -> 1):
  -> Keys and values are "aligned" in scale
  -> High-weight keys tend to have high-norm values
  -> Errors more predictable; top-K approximation captures both attention and magnitude

Low K-V norm correlation (r -> 0):
  -> Keys and values have independent scales
  -> High-weight keys may have low-norm values (or vice versa)
  -> Top-K by logit may miss high-magnitude values
  -> Output error can be larger due to scale mismatch

If norms are uncorrelated or misaligned, consider:
  1. Normalizing K and V to unit norm before attention
  2. Using weighted sampling that accounts for value norms
  3. Separate budgets for "high attention" and "high magnitude" keys
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("KEY-VALUE NORM RELATIONSHIP ANALYSIS")
    print("="*70)
    print(f"Config: {NUM_EXAMPLES} examples, {NUM_QUERIES} queries/example")
    print()

    setup_style()
    np.random.seed(SEED)

    # Resolve data path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, DATA_PATH) if not os.path.isabs(DATA_PATH) else DATA_PATH
    output_dir = Path(os.path.join(script_dir, str(OUTPUT_DIR))) if not OUTPUT_DIR.is_absolute() else OUTPUT_DIR

    # Load examples
    print(f"Counting and selecting examples...")
    with open(data_path, 'r') as f:
        total = sum(1 for _ in f)

    selected_indices = sorted(np.random.choice(total, NUM_EXAMPLES, replace=False).tolist())
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

    # Numerical analysis
    print_analysis(results['first_layer'], results['last_layer'])

    # Create plots
    print(f"\nGenerating plot...")
    fig = create_plots(results['first_layer'], results['last_layer'])

    output_path = output_dir / 'key_value_norm_relationship.png'
    save_figure(fig, output_path, dpi=200)

    print(f"\nDone!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
