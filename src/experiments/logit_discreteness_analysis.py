"""
Logit & Weight Discreteness Analysis.

Investigates whether attention logits/weights take discrete step-like values.

For each example, last 10 queries:
1. Compute logits q·k/√d and attention weights
2. Measure discreteness: number of unique values, spacing histogram, etc.
3. Check if logits or weights are the source
4. Verify float16 hypothesis by rounding vectors to float16 and comparing
5. Compare logit duplication rate vs token ID duplication rate
6. Check if keys with same token ID produce identical key vectors

Saves JSON + diagnostic plots.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from algorithms.base import softmax

# ── Hyperparameters ──
DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/attention_vectors_long_bench_llama_8b.jsonl')
LONGBENCH_PATH = os.path.join(os.path.dirname(__file__), '../../data/longbench_v2_truncated_7k_smart.json')
TOKENIZER_NAME = 'NousResearch/Meta-Llama-3-8B'
NUM_EXAMPLES = 50
NUM_QUERIES = 10
HEAD_DIM = 128
SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../results/logit_discreteness')


def compute_gap_histogram(x):
    """Compute gaps between sorted unique values."""
    unique_sorted = np.sort(np.unique(x))
    if len(unique_sorted) < 2:
        return np.array([])
    return np.diff(unique_sorted)


def check_float16_match(vec):
    """Check if a float32 vector is exactly representable in float16."""
    vec_f16 = vec.astype(np.float16).astype(np.float32)
    return np.allclose(vec, vec_f16, atol=0, rtol=0)


def analyze_discreteness(logits, weights):
    """Analyze discreteness of logits and weights for a single query."""
    n = len(logits)

    logit_unique = len(np.unique(logits))
    logit_unique_ratio = logit_unique / n
    logit_gaps = compute_gap_histogram(logits)
    logit_min_gap = float(np.min(logit_gaps)) if len(logit_gaps) > 0 else 0.0
    logit_median_gap = float(np.median(logit_gaps)) if len(logit_gaps) > 0 else 0.0

    weight_unique = len(np.unique(weights))
    weight_unique_ratio = weight_unique / n

    _, counts = np.unique(logits, return_counts=True)
    max_duplicates = int(np.max(counts))
    mean_duplicates = float(np.mean(counts))
    frac_with_duplicates = float(np.sum(counts > 1)) / len(counts) if len(counts) > 0 else 0.0

    return {
        'n_keys': n,
        'logit_unique': logit_unique,
        'logit_unique_ratio': logit_unique_ratio,
        'logit_min_gap': logit_min_gap,
        'logit_median_gap': logit_median_gap,
        'weight_unique': weight_unique,
        'weight_unique_ratio': weight_unique_ratio,
        'logit_max_duplicates': max_duplicates,
        'logit_mean_duplicates': mean_duplicates,
        'logit_frac_with_duplicates': frac_with_duplicates,
    }


def tokenize_example(tokenizer, example):
    """Tokenize an example the same way as extraction script."""
    prompt = f"Context: {example['context']}\n\nQuestion: {example['question']}\n\nAnswer:"
    tokens = tokenizer(prompt, truncation=True, max_length=8192)
    return tokens['input_ids']


def analyze_token_duplication(token_ids):
    """Analyze token ID duplication in a sequence."""
    n = len(token_ids)
    unique_ids = len(set(token_ids))
    unique_ratio = unique_ids / n

    # Count how many positions share each token ID
    from collections import Counter
    counts = Counter(token_ids)
    count_values = list(counts.values())
    max_dup = max(count_values)
    mean_dup = np.mean(count_values)
    frac_with_dup = sum(1 for c in count_values if c > 1) / len(count_values)

    return {
        'n_tokens': n,
        'unique_token_ids': unique_ids,
        'unique_ratio': unique_ratio,
        'max_duplicates': max_dup,
        'mean_duplicates': float(mean_dup),
        'frac_with_duplicates': frac_with_dup,
    }


def check_key_identity_by_token_id(K, token_ids, layer_label=""):
    """Check if positions with the same token ID have identical key vectors."""
    from collections import defaultdict
    token_to_positions = defaultdict(list)
    for pos, tid in enumerate(token_ids[:K.shape[0]]):
        token_to_positions[tid].append(pos)

    # For tokens that appear multiple times, check if their keys are identical
    n_groups_checked = 0
    n_groups_identical = 0
    n_groups_close = 0  # within float16 tolerance
    max_within_group_dist = []

    for tid, positions in token_to_positions.items():
        if len(positions) < 2:
            continue
        n_groups_checked += 1
        keys_for_tid = K[positions]  # [n_occurrences, head_dim]

        # Check pairwise max distance
        dists = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dists.append(np.linalg.norm(keys_for_tid[i] - keys_for_tid[j]))

        max_dist = max(dists)
        max_within_group_dist.append(max_dist)

        if max_dist == 0.0:
            n_groups_identical += 1
        if max_dist < 0.01:  # roughly float16 tolerance for these magnitudes
            n_groups_close += 1

    return {
        'n_token_groups_with_repeats': n_groups_checked,
        'n_groups_identical_keys': n_groups_identical,
        'n_groups_close_keys': n_groups_close,
        'frac_identical': n_groups_identical / max(n_groups_checked, 1),
        'frac_close': n_groups_close / max(n_groups_checked, 1),
        'max_within_group_dist_mean': float(np.mean(max_within_group_dist)) if max_within_group_dist else 0.0,
        'max_within_group_dist_median': float(np.median(max_within_group_dist)) if max_within_group_dist else 0.0,
        'max_within_group_dist_max': float(np.max(max_within_group_dist)) if max_within_group_dist else 0.0,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load tokenizer
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"  Tokenizer: {TOKENIZER_NAME}, vocab size: {tokenizer.vocab_size}")

    # Load LongBench examples for tokenization
    print("Loading LongBench data for tokenization...")
    with open(LONGBENCH_PATH, 'r') as f:
        longbench_data = json.load(f)
    longbench_examples = longbench_data['examples']
    print(f"  {len(longbench_examples)} examples loaded\n")

    all_results = []
    sample_data = {}

    # Per-example: token dup rate, logit dup rate (for comparison scatter)
    token_dup_rates = []
    logit_dup_rates_first = []
    logit_dup_rates_last = []
    key_identity_results = {'first_layer': [], 'last_layer': []}

    print(f"Processing {NUM_EXAMPLES} examples, {NUM_QUERIES} queries each\n")

    with open(DATA_PATH, 'r') as f:
        for ex_idx, line in enumerate(f):
            if ex_idx >= NUM_EXAMPLES:
                break
            example = json.loads(line)
            domain = example.get('domain', 'unknown')
            seq_len = example.get('sequence_length', 0)
            print(f"  [{ex_idx:3d}] {domain}")

            # Tokenize
            token_ids = tokenize_example(tokenizer, longbench_examples[ex_idx])
            token_stats = analyze_token_duplication(token_ids)
            token_dup_rates.append(token_stats['unique_ratio'])

            if ex_idx == 0:
                print(f"    Tokens: {token_stats['n_tokens']}, unique: {token_stats['unique_token_ids']} "
                      f"({token_stats['unique_ratio']:.3f}), max dup: {token_stats['max_duplicates']}")

            ex_result = {'example_idx': ex_idx, 'domain': domain, 'token_stats': token_stats}

            for layer_name in ['first_layer', 'last_layer']:
                Q = np.array(example[layer_name]['Q'], dtype=np.float32)
                K = np.array(example[layer_name]['K'], dtype=np.float32)
                actual_seq_len = Q.shape[0]

                # Check if vectors are float16-representable (first example only)
                if ex_idx == 0:
                    q_is_f16 = check_float16_match(Q[0])
                    k_is_f16 = check_float16_match(K[0])
                    print(f"    {layer_name}: Q[0] exact float16? {q_is_f16}, K[0] exact float16? {k_is_f16}")

                # Check key identity by token ID
                kid_result = check_key_identity_by_token_id(K, token_ids, layer_name)
                key_identity_results[layer_name].append(kid_result)
                if ex_idx == 0:
                    print(f"    {layer_name}: {kid_result['n_token_groups_with_repeats']} repeated token groups, "
                          f"{kid_result['frac_identical']:.1%} have identical keys, "
                          f"{kid_result['frac_close']:.1%} have close keys")

                start = max(0, actual_seq_len - NUM_QUERIES)
                query_positions = list(range(start, actual_seq_len))

                layer_stats = []
                for qpos in query_positions:
                    q = Q[qpos]
                    valid_keys = K[:qpos + 1]
                    logits = (q @ valid_keys.T) / np.sqrt(HEAD_DIM)
                    weights = softmax(logits)

                    stats = analyze_discreteness(logits, weights)
                    layer_stats.append(stats)

                    if ex_idx == 0 and qpos == actual_seq_len - 1:
                        sample_data[layer_name] = {
                            'logits': logits.copy(),
                            'weights': weights.copy(),
                            'q': q.copy(),
                            'valid_keys': valid_keys.copy(),
                        }

                # Aggregate over queries
                avg_stats = {}
                for key in layer_stats[0]:
                    vals = [s[key] for s in layer_stats]
                    avg_stats[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
                ex_result[layer_name] = avg_stats

                if 'first' in layer_name:
                    logit_dup_rates_first.append(avg_stats['logit_unique_ratio']['mean'])
                else:
                    logit_dup_rates_last.append(avg_stats['logit_unique_ratio']['mean'])

            all_results.append(ex_result)
            del example

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("DISCRETENESS SUMMARY")
    print("=" * 70)
    for layer in ['first_layer', 'last_layer']:
        unique_ratios = [r[layer]['logit_unique_ratio']['mean'] for r in all_results]
        max_dups = [r[layer]['logit_max_duplicates']['mean'] for r in all_results]
        weight_uniq = [r[layer]['weight_unique_ratio']['mean'] for r in all_results]
        frac_dup = [r[layer]['logit_frac_with_duplicates']['mean'] for r in all_results]
        print(f"\n{'FIRST' if 'first' in layer else 'LAST'} LAYER:")
        print(f"  Logit unique ratio:      {np.mean(unique_ratios):.4f} (±{np.std(unique_ratios):.4f})")
        print(f"  Logit max duplicates:    {np.mean(max_dups):.1f} (±{np.std(max_dups):.1f})")
        print(f"  Logit frac w/ duplicates:{np.mean(frac_dup):.4f}")
        print(f"  Weight unique ratio:     {np.mean(weight_uniq):.4f} (±{np.std(weight_uniq):.4f})")

    print(f"\nTOKEN ID UNIQUENESS:")
    print(f"  Unique token ratio:      {np.mean(token_dup_rates):.4f} (±{np.std(token_dup_rates):.4f})")

    print(f"\nKEY IDENTITY BY TOKEN ID:")
    for layer in ['first_layer', 'last_layer']:
        frac_id = [r['frac_identical'] for r in key_identity_results[layer]]
        frac_cl = [r['frac_close'] for r in key_identity_results[layer]]
        max_d = [r['max_within_group_dist_mean'] for r in key_identity_results[layer]]
        lbl = 'FIRST' if 'first' in layer else 'LAST'
        print(f"  {lbl} LAYER: {np.mean(frac_id):.1%} identical, {np.mean(frac_cl):.1%} close, "
              f"mean max within-group dist = {np.mean(max_d):.4f}")

    # ══════════════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════════════

    # ── Plot 1: Logit histogram + gap distribution ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for col, layer_name in enumerate(['first_layer', 'last_layer']):
        logits = sample_data[layer_name]['logits']
        layer_label = 'First Layer' if 'first' in layer_name else 'Last Layer'

        ax = axes[0, col]
        ax.hist(logits, bins=200, alpha=0.7, color='steelblue', edgecolor='none')
        n_unique = len(np.unique(logits))
        ax.set_title(f'Logit Distribution — {layer_label}\n'
                     f'{n_unique} unique values out of {len(logits)}')
        ax.set_xlabel('Logit value (q·k/√d)')
        ax.set_ylabel('Count')

        ax = axes[1, col]
        gaps = compute_gap_histogram(logits)
        if len(gaps) > 0:
            ax.hist(gaps, bins=200, alpha=0.7, color='darkorange', edgecolor='none')
            ax.axvline(np.median(gaps), color='red', linestyle='--', label=f'Median gap = {np.median(gaps):.6f}')
            logit_mag = np.median(np.abs(logits))
            f16_step = float(np.float16(logit_mag + np.finfo(np.float16).eps) - np.float16(logit_mag))
            ax.axvline(f16_step, color='green', linestyle=':', linewidth=2,
                       label=f'float16 step @ mag {logit_mag:.1f} ≈ {f16_step:.5f}')
            ax.set_title(f'Logit Gap Distribution — {layer_label}')
            ax.set_xlabel('Gap between consecutive unique logits')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)

    plt.suptitle('Logit Discreteness Diagnostic (Example #0, last query)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'logit_discreteness_example0.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: logit_discreteness_example0.png")

    # ── Plot 2: Attention weight stepping ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for col, layer_name in enumerate(['first_layer', 'last_layer']):
        weights = sample_data[layer_name]['weights']
        layer_label = 'First Layer' if 'first' in layer_name else 'Last Layer'

        ax = axes[0, col]
        ax.plot(weights, linewidth=0.3, alpha=0.7, color='steelblue')
        ax.set_title(f'Attention Weights — {layer_label}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Attention Weight')

        ax = axes[1, col]
        sorted_w = np.sort(weights)[::-1]
        mid_start = len(sorted_w) // 4
        mid_end = 3 * len(sorted_w) // 4
        ax.plot(range(mid_start, mid_end), sorted_w[mid_start:mid_end],
                linewidth=0.5, alpha=0.8, color='darkorange')
        n_unique = len(np.unique(weights))
        ax.set_title(f'Sorted Weights (middle 50%) — {layer_label}\n'
                     f'{n_unique} unique weight values')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Attention Weight')

    plt.suptitle('Attention Weight Stepping (Example #0, last query)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'weight_stepping_example0.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: weight_stepping_example0.png")

    # ── Plot 3: Float16 verification ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for col, layer_name in enumerate(['first_layer', 'last_layer']):
        q = sample_data[layer_name]['q']
        valid_keys = sample_data[layer_name]['valid_keys']
        layer_label = 'First Layer' if 'first' in layer_name else 'Last Layer'

        logits_f32 = (q @ valid_keys.T) / np.sqrt(HEAD_DIM)
        q_f16 = q.astype(np.float16).astype(np.float32)
        k_f16 = valid_keys.astype(np.float16).astype(np.float32)
        logits_f16_sim = (q_f16 @ k_f16.T) / np.sqrt(HEAD_DIM)
        diff = np.abs(logits_f32 - logits_f16_sim)

        ax = axes[col]
        ax.hist(diff, bins=100, alpha=0.7, color='purple', edgecolor='none')
        ax.axvline(np.median(diff), color='red', linestyle='--',
                   label=f'Median diff = {np.median(diff):.2e}')
        match_pct = 100 * np.mean(diff == 0)
        ax.set_title(f'{layer_label}: f32 vs f16-rounded logits\n{match_pct:.1f}% exact matches')
        ax.set_xlabel('|logit_f32 - logit_f16_sim|')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)

    plt.suptitle('Float16 Origin Verification', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'float16_verification.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: float16_verification.png")

    # ── Plot 4: Token ID uniqueness vs Logit uniqueness — THE KEY COMPARISON ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 4a: Scatter — token unique ratio vs logit unique ratio
    ax = axes[0]
    ax.scatter(token_dup_rates, logit_dup_rates_first, alpha=0.6, s=40, c='steelblue',
               edgecolors='k', linewidth=0.3, label='First Layer')
    ax.scatter(token_dup_rates, logit_dup_rates_last, alpha=0.6, s=40, c='darkorange',
               edgecolors='k', linewidth=0.3, label='Last Layer')
    lims = [0, 1.05]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Token ID Unique Ratio')
    ax.set_ylabel('Logit Unique Ratio')
    ax.set_title('Token ID Uniqueness vs Logit Uniqueness')
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # 4b: Bar comparison — mean unique ratios
    ax = axes[1]
    means = [np.mean(token_dup_rates), np.mean(logit_dup_rates_first), np.mean(logit_dup_rates_last)]
    stds = [np.std(token_dup_rates), np.std(logit_dup_rates_first), np.std(logit_dup_rates_last)]
    bars = ax.bar(['Token IDs', 'Logits\n(First Layer)', 'Logits\n(Last Layer)'],
                  means, yerr=stds, capsize=5,
                  color=['gray', 'steelblue', 'darkorange'], alpha=0.7, edgecolor='k', linewidth=0.5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{m:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Unique Ratio')
    ax.set_title('Average Uniqueness Comparison')
    ax.set_ylim(0, 1.15)

    plt.suptitle(f'Token Duplication vs Logit Duplication ({NUM_EXAMPLES} examples)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'token_vs_logit_uniqueness.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: token_vs_logit_uniqueness.png")

    # ── Plot 5: Key identity check — do same token IDs → same key vectors? ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, layer in zip(axes, ['first_layer', 'last_layer']):
        frac_identical = [r['frac_identical'] for r in key_identity_results[layer]]
        frac_close = [r['frac_close'] for r in key_identity_results[layer]]
        layer_label = 'First Layer' if 'first' in layer else 'Last Layer'

        x = np.arange(len(frac_identical))
        width = 0.35
        ax.bar(x - width / 2, frac_identical, width, label='Identical (dist=0)', color='steelblue', alpha=0.7)
        ax.bar(x + width / 2, frac_close, width, label='Close (dist<0.01)', color='darkorange', alpha=0.7)
        ax.set_xlabel('Example Index')
        ax.set_ylabel('Fraction of Repeated-Token Groups')
        ax.set_title(f'{layer_label}: Same Token ID → Same Key Vector?')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    plt.suptitle('Key Vector Identity for Repeated Token IDs', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'key_identity_by_token_id.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: key_identity_by_token_id.png")

    # ── Plot 6: Per-example boxplot ──
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([logit_dup_rates_first, logit_dup_rates_last],
                    tick_labels=['First Layer (0)', 'Last Layer (31)'],
                    patch_artist=True, widths=0.4)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightsalmon')
    for i, data in enumerate([logit_dup_rates_first, logit_dup_rates_last], 1):
        x_jit = np.random.default_rng(42).normal(i, 0.03, len(data))
        ax.scatter(x_jit, data, alpha=0.4, s=15, c='gray', zorder=3)
    ax.axhline(np.mean(token_dup_rates), color='green', linestyle='--', alpha=0.6,
               label=f'Token ID unique ratio = {np.mean(token_dup_rates):.3f}')
    ax.set_ylabel('Unique Logit Values / Total Keys')
    ax.set_title(f'Logit Uniqueness ({NUM_EXAMPLES} examples)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'unique_ratio_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: unique_ratio_boxplot.png")

    # ── Plot 7: Logit duplicate count distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, layer_name in zip(axes, ['first_layer', 'last_layer']):
        logits = sample_data[layer_name]['logits']
        _, counts = np.unique(logits, return_counts=True)
        layer_label = 'First Layer' if 'first' in layer_name else 'Last Layer'
        ax.hist(counts, bins=range(1, min(max(counts) + 2, 50)), alpha=0.7,
                color='steelblue', edgecolor='white', linewidth=0.3)
        ax.set_xlabel('Number of keys sharing same logit value')
        ax.set_ylabel('Count of unique logit values')
        ax.set_title(f'{layer_label} — Logit Duplicate Distribution\nMax duplicates: {max(counts)}')
    plt.suptitle('How many keys share the exact same logit? (Example #0)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'duplicate_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: duplicate_distribution.png")

    # ── Save JSON ──
    json_summary = {
        'config': {
            'num_examples': NUM_EXAMPLES,
            'num_queries': NUM_QUERIES,
            'tokenizer': TOKENIZER_NAME,
        },
        'aggregate': {
            'token_unique_ratio': {'mean': float(np.mean(token_dup_rates)), 'std': float(np.std(token_dup_rates))},
            'first_layer': {
                'logit_unique_ratio': {'mean': float(np.mean(logit_dup_rates_first)),
                                       'std': float(np.std(logit_dup_rates_first))},
                'key_identity_frac_identical': {
                    'mean': float(np.mean([r['frac_identical'] for r in key_identity_results['first_layer']])),
                },
                'key_identity_frac_close': {
                    'mean': float(np.mean([r['frac_close'] for r in key_identity_results['first_layer']])),
                },
            },
            'last_layer': {
                'logit_unique_ratio': {'mean': float(np.mean(logit_dup_rates_last)),
                                       'std': float(np.std(logit_dup_rates_last))},
                'key_identity_frac_identical': {
                    'mean': float(np.mean([r['frac_identical'] for r in key_identity_results['last_layer']])),
                },
                'key_identity_frac_close': {
                    'mean': float(np.mean([r['frac_close'] for r in key_identity_results['last_layer']])),
                },
            },
        },
        'per_example': all_results,
    }
    json_path = os.path.join(OUTPUT_DIR, 'discreteness_stats.json')
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    print(f"Saved: {json_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
