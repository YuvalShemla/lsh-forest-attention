#!/usr/bin/env python3
"""
Experiment 2: Crossover Budget Verification

Tests whether the theoretical crossover formula
    B_cross = Var_w(V) / ||seg_bias||^2
predicts when segmentation (GMM) beats sampling (oracle).

Theory:
  - Oracle sampling has MSE ~ Var_w(V) / B
    where Var_w(V) = sum_i w_i * ||v_i - o*||^2
  - GMM segmentation has fixed bias ||o_seg - o*||^2 (independent of budget)
  - Crossover: segmentation wins when B > B_cross = Var_w(V) / seg_bias^2

For each query, we:
  1. Compute ground truth output o* and weights w
  2. Compute Var_w(V) = sum_i w_i * ||v_i - o*||^2
  3. Compute GMM output o_seg, then seg_bias_sq = ||o_seg - o*||^2
  4. Compute B_cross = Var_w(V) / seg_bias_sq
  5. Run oracle sampling at multiple budgets and find empirical crossover
  6. Compare predicted B_cross with empirical crossover

Results saved to: claude_experiments/02_crossover_budget/results/
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time

from algorithms import (
    compute_ground_truth_attention,
    relative_l2_error,
    oracle_sampling,
    fit_gmm,
    gmm_attention,
)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

NUM_EXAMPLES = 10
NUM_QUERIES_PER_EXAMPLE = 50
ORACLE_TRIALS = 20
BUDGETS = [10, 20, 50, 100, 200, 500]
GMM_CLUSTERS = 50
HEAD_DIM = 128
SEED = 42
LAYERS_TO_TEST = ['first_layer', 'last_layer']

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', 'data',
                         'attention_vectors_long_bench_llama_8b.jsonl')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# ============================================================================
# END HYPERPARAMETERS
# ============================================================================


def compute_crossover_for_query(q, K, V, query_pos, head_dim, resp):
    """
    Compute all crossover-related quantities for a single query.

    Returns a dict with:
      - var_w_V: weighted variance of values around ground truth output
      - seg_bias_sq: squared L2 norm of segmentation bias
      - gmm_error: relative L2 error of GMM output
      - B_cross: predicted crossover budget (or None if seg_bias ~ 0)
      - oracle_errors: dict budget -> mean oracle relative L2 error (over trials)
      - empirical_crossover: smallest budget where oracle error < gmm error (or None)
    """
    # Ground truth
    gt_output, gt_logits, gt_weights, _ = compute_ground_truth_attention(
        q, K, V, query_pos, head_dim
    )

    valid_keys = K[:query_pos + 1]
    valid_values = V[:query_pos + 1]
    nv = len(valid_keys)

    # 1. Var_w(V) = sum_i w_i * ||v_i - o*||^2
    deviations = valid_values - gt_output[np.newaxis, :]  # [nv, d]
    sq_norms = np.sum(deviations ** 2, axis=1)            # [nv]
    var_w_V = float(np.sum(gt_weights * sq_norms))

    # 2. GMM output and segmentation bias
    resp_valid = resp[:nv]
    output_gmm, n_active = gmm_attention(
        q, valid_keys, valid_values, gt_logits, head_dim, resp_valid
    )
    gmm_error = float(relative_l2_error(output_gmm, gt_output))
    seg_bias_sq = float(np.sum((output_gmm - gt_output) ** 2))

    # 3. Predicted crossover budget
    if seg_bias_sq > 1e-16:
        B_cross = var_w_V / seg_bias_sq
    else:
        B_cross = None  # GMM is near-perfect, crossover undefined

    # 4. Oracle sampling at each budget (average over trials for stability)
    oracle_errors = {}
    oracle_sq_errors = {}
    for budget in BUDGETS:
        if budget > nv:
            continue
        errors = []
        sq_errs = []
        for trial in range(ORACLE_TRIALS):
            output_oracle, _ = oracle_sampling(
                q, valid_keys, valid_values, gt_logits, gt_weights, budget
            )
            errors.append(relative_l2_error(output_oracle, gt_output))
            sq_errs.append(float(np.sum((output_oracle - gt_output) ** 2)))
        oracle_errors[budget] = float(np.mean(errors))
        oracle_sq_errors[budget] = float(np.mean(sq_errs))

    # 5. Empirical crossover: smallest budget where oracle error < gmm error
    empirical_crossover = None
    for budget in sorted(oracle_errors.keys()):
        if oracle_errors[budget] < gmm_error:
            empirical_crossover = budget
            break

    # Also find empirical crossover using squared errors (MSE-based)
    empirical_crossover_mse = None
    for budget in sorted(oracle_sq_errors.keys()):
        if oracle_sq_errors[budget] < seg_bias_sq:
            empirical_crossover_mse = budget
            break

    return {
        'var_w_V': var_w_V,
        'seg_bias_sq': seg_bias_sq,
        'gmm_error': gmm_error,
        'B_cross': B_cross,
        'oracle_errors': oracle_errors,
        'oracle_sq_errors': oracle_sq_errors,
        'empirical_crossover': empirical_crossover,
        'empirical_crossover_mse': empirical_crossover_mse,
        'n_active_clusters': n_active,
        'num_valid_keys': nv,
        'gt_output_norm': float(np.linalg.norm(gt_output)),
    }


def plot_crossover_scatter(results_by_layer, output_dir):
    """Plot predicted B_cross vs empirical crossover for each layer."""

    for layer_name, results in results_by_layer.items():
        predicted = []
        empirical = []
        empirical_mse = []

        for r in results:
            if r['B_cross'] is not None and r['empirical_crossover'] is not None:
                predicted.append(r['B_cross'])
                empirical.append(r['empirical_crossover'])
            if r['B_cross'] is not None and r['empirical_crossover_mse'] is not None:
                predicted.append(r['B_cross'])
                empirical_mse.append(r['empirical_crossover_mse'])

        if len(predicted) == 0:
            print(f"  [{layer_name}] No queries with both predicted and empirical crossover")
            continue

        fig, ax = plt.subplots(figsize=(8, 8))
        pred_arr = np.array(predicted[:len(empirical)])
        emp_arr = np.array(empirical)

        ax.scatter(pred_arr, emp_arr, alpha=0.5, s=30, color='#2ca02c', label='Queries')

        # Identity line
        all_vals = np.concatenate([pred_arr, emp_arr])
        lo, hi = max(1, all_vals.min() * 0.5), all_vals.max() * 2
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.5, label='y = x (perfect prediction)')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Predicted B_cross = Var_w(V) / ||seg_bias||^2', fontsize=12)
        ax.set_ylabel('Empirical Crossover Budget', fontsize=12)
        ax.set_title(f'Crossover Budget Prediction\n{layer_name.replace("_", " ").title()}',
                      fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = output_dir / f'crossover_scatter_{layer_name}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


def plot_error_curves(results_by_layer, output_dir):
    """Plot oracle error vs budget curves with GMM horizontal line for a few example queries."""

    for layer_name, results in results_by_layer.items():
        # Pick up to 6 representative queries with valid B_cross
        valid_queries = [r for r in results if r['B_cross'] is not None and len(r['oracle_errors']) > 0]
        if len(valid_queries) == 0:
            continue

        # Sort by B_cross to show diversity
        valid_queries.sort(key=lambda r: r['B_cross'])
        n_show = min(6, len(valid_queries))
        indices = np.linspace(0, len(valid_queries) - 1, n_show, dtype=int)
        selected = [valid_queries[i] for i in indices]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (ax, r) in enumerate(zip(axes, selected)):
            budgets = sorted(r['oracle_errors'].keys())
            oracle_errs = [r['oracle_errors'][b] for b in budgets]

            ax.plot(budgets, oracle_errs, 'o-', color='#2ca02c', linewidth=2,
                    markersize=6, label='Oracle sampling')
            ax.axhline(y=r['gmm_error'], color='#d62728', linewidth=2,
                       linestyle='--', label=f'GMM (error={r["gmm_error"]:.4f})')

            if r['B_cross'] is not None:
                ax.axvline(x=r['B_cross'], color='#1f77b4', linewidth=1.5,
                           linestyle=':', alpha=0.7,
                           label=f'Predicted B_cross={r["B_cross"]:.0f}')

            if r['empirical_crossover'] is not None:
                ax.axvline(x=r['empirical_crossover'], color='#ff7f0e',
                           linewidth=1.5, linestyle='-.',
                           alpha=0.7,
                           label=f'Empirical cross={r["empirical_crossover"]}')

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Budget')
            ax.set_ylabel('Relative L2 Error')
            ax.set_title(f'Query {idx + 1} (B_cross={r["B_cross"]:.0f})')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)

        # Hide extra axes
        for ax in axes[n_show:]:
            ax.set_visible(False)

        fig.suptitle(f'Oracle Error vs Budget with GMM Baseline\n{layer_name.replace("_", " ").title()}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = output_dir / f'error_curves_{layer_name}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


def plot_bcross_distribution(results_by_layer, output_dir):
    """Plot distribution of B_cross values per layer."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (layer_name, results) in zip(axes, results_by_layer.items()):
        bcross_vals = [r['B_cross'] for r in results if r['B_cross'] is not None]
        if len(bcross_vals) == 0:
            ax.text(0.5, 0.5, 'No valid B_cross', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        bcross_arr = np.array(bcross_vals)
        # Log-scale histogram
        log_vals = np.log10(np.clip(bcross_arr, 1e-3, None))
        ax.hist(log_vals, bins=40, color='#1f77b4', alpha=0.7, edgecolor='black')

        # Add budget reference lines
        for b in BUDGETS:
            ax.axvline(x=np.log10(b), color='#d62728', linewidth=0.8,
                       linestyle='--', alpha=0.5)

        ax.set_xlabel('log10(B_cross)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{layer_name.replace("_", " ").title()}\n'
                     f'Median B_cross = {np.median(bcross_arr):.1f}, '
                     f'Mean = {np.mean(bcross_arr):.1f}\n'
                     f'(N = {len(bcross_vals)})',
                     fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Distribution of Predicted Crossover Budget B_cross',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = output_dir / 'bcross_distribution.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_seg_wins_fraction(results_by_layer, output_dir):
    """Plot fraction of queries where segmentation wins at each budget."""

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'first_layer': '#1f77b4', 'last_layer': '#d62728'}
    markers = {'first_layer': 'o', 'last_layer': 's'}

    for layer_name, results in results_by_layer.items():
        fractions = []
        valid_budgets = []

        for budget in BUDGETS:
            seg_wins = 0
            total = 0
            for r in results:
                if budget in r['oracle_errors']:
                    total += 1
                    if r['gmm_error'] < r['oracle_errors'][budget]:
                        seg_wins += 1
            if total > 0:
                fractions.append(seg_wins / total)
                valid_budgets.append(budget)

        ax.plot(valid_budgets, fractions, marker=markers[layer_name],
                linewidth=2.5, markersize=8, color=colors[layer_name],
                label=layer_name.replace('_', ' ').title())

    ax.axhline(y=0.5, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Budget', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraction of Queries Where GMM Wins', fontsize=12, fontweight='bold')
    ax.set_title('GMM vs Oracle: Fraction Where Segmentation Wins by Budget',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / 'seg_wins_fraction.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_var_vs_bias(results_by_layer, output_dir):
    """Scatter plot of Var_w(V) vs seg_bias_sq colored by layer."""

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {'first_layer': '#1f77b4', 'last_layer': '#d62728'}

    for layer_name, results in results_by_layer.items():
        var_vals = [r['var_w_V'] for r in results if r['seg_bias_sq'] > 1e-16]
        bias_vals = [r['seg_bias_sq'] for r in results if r['seg_bias_sq'] > 1e-16]

        ax.scatter(var_vals, bias_vals, alpha=0.4, s=20,
                   color=colors[layer_name],
                   label=layer_name.replace('_', ' ').title())

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Var_w(V) = sum_i w_i ||v_i - o*||^2', fontsize=12)
    ax.set_ylabel('seg_bias_sq = ||o_seg - o*||^2', fontsize=12)
    ax.set_title('Weighted Value Variance vs Segmentation Bias',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / 'var_vs_bias.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def compute_statistics(results_by_layer):
    """Compute summary statistics for reporting."""

    stats = {}
    for layer_name, results in results_by_layer.items():
        bcross_vals = [r['B_cross'] for r in results if r['B_cross'] is not None]
        var_vals = [r['var_w_V'] for r in results]
        bias_vals = [r['seg_bias_sq'] for r in results]
        gmm_errors = [r['gmm_error'] for r in results]

        # Correlation between predicted and empirical crossover
        predicted = []
        empirical = []
        for r in results:
            if r['B_cross'] is not None and r['empirical_crossover'] is not None:
                predicted.append(r['B_cross'])
                empirical.append(r['empirical_crossover'])

        if len(predicted) >= 2:
            corr_log = float(np.corrcoef(np.log10(np.clip(predicted, 1e-3, None)),
                                          np.log10(np.clip(empirical, 1, None)))[0, 1])
        else:
            corr_log = None

        # Fraction of queries where seg wins at each budget
        seg_wins_frac = {}
        for budget in BUDGETS:
            seg_wins = 0
            total = 0
            for r in results:
                if budget in r['oracle_errors']:
                    total += 1
                    if r['gmm_error'] < r['oracle_errors'][budget]:
                        seg_wins += 1
            if total > 0:
                seg_wins_frac[budget] = {'fraction': seg_wins / total,
                                         'seg_wins': seg_wins, 'total': total}

        # B_cross statistics
        bcross_stats = {}
        if len(bcross_vals) > 0:
            bcross_arr = np.array(bcross_vals)
            bcross_stats = {
                'mean': float(np.mean(bcross_arr)),
                'median': float(np.median(bcross_arr)),
                'std': float(np.std(bcross_arr)),
                'min': float(np.min(bcross_arr)),
                'max': float(np.max(bcross_arr)),
                'p10': float(np.percentile(bcross_arr, 10)),
                'p25': float(np.percentile(bcross_arr, 25)),
                'p75': float(np.percentile(bcross_arr, 75)),
                'p90': float(np.percentile(bcross_arr, 90)),
                'n_valid': len(bcross_vals),
                'n_total': len(results),
                'frac_valid': len(bcross_vals) / len(results) if len(results) > 0 else 0,
            }

        # Prediction accuracy: fraction of queries where crossover
        # is correctly predicted within a factor
        prediction_accuracy = {}
        if len(predicted) > 0:
            pred_arr = np.array(predicted)
            emp_arr = np.array(empirical)
            ratio = pred_arr / emp_arr
            for factor in [2, 5, 10]:
                within = np.sum((ratio >= 1.0 / factor) & (ratio <= factor))
                prediction_accuracy[f'within_{factor}x'] = {
                    'count': int(within),
                    'total': len(predicted),
                    'fraction': float(within / len(predicted)),
                }
            prediction_accuracy['median_ratio'] = float(np.median(ratio))
            prediction_accuracy['mean_ratio'] = float(np.mean(ratio))
            prediction_accuracy['log_ratio_mean'] = float(np.mean(np.log10(ratio)))
            prediction_accuracy['log_ratio_std'] = float(np.std(np.log10(ratio)))

        stats[layer_name] = {
            'B_cross': bcross_stats,
            'seg_wins_fraction': seg_wins_frac,
            'correlation_log_predicted_empirical': corr_log,
            'n_both_valid': len(predicted),
            'prediction_accuracy': prediction_accuracy,
            'gmm_error': {
                'mean': float(np.mean(gmm_errors)),
                'median': float(np.median(gmm_errors)),
                'std': float(np.std(gmm_errors)),
            },
            'var_w_V': {
                'mean': float(np.mean(var_vals)),
                'median': float(np.median(var_vals)),
            },
            'seg_bias_sq': {
                'mean': float(np.mean(bias_vals)),
                'median': float(np.median(bias_vals)),
            },
        }

    return stats


def main():
    """Run crossover budget verification experiment."""

    print("=" * 70)
    print("EXPERIMENT 2: CROSSOVER BUDGET VERIFICATION")
    print("=" * 70)
    print(f"\nTheory: B_cross = Var_w(V) / ||seg_bias||^2")
    print(f"  - At B < B_cross, oracle sampling has lower error")
    print(f"  - At B > B_cross, GMM segmentation has lower error")
    print(f"\nConfiguration:")
    print(f"  NUM_EXAMPLES = {NUM_EXAMPLES}")
    print(f"  NUM_QUERIES_PER_EXAMPLE = {NUM_QUERIES_PER_EXAMPLE}")
    print(f"  ORACLE_TRIALS = {ORACLE_TRIALS}")
    print(f"  BUDGETS = {BUDGETS}")
    print(f"  GMM_CLUSTERS = {GMM_CLUSTERS}")
    print(f"  LAYERS = {LAYERS_TO_TEST}")
    print(f"  SEED = {SEED}")

    np.random.seed(SEED)

    # Resolve data path
    data_path = os.path.abspath(DATA_PATH)
    print(f"\nData: {data_path}")
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        sys.exit(1)

    # Load data
    print("Loading examples...")
    load_start = time.time()
    examples = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= NUM_EXAMPLES:
                break
            examples.append(json.loads(line))
    load_time = time.time() - load_start
    print(f"Loaded {len(examples)} examples in {load_time:.1f}s")

    # Results storage
    results_by_layer = {layer: [] for layer in LAYERS_TO_TEST}

    total_queries = len(examples) * NUM_QUERIES_PER_EXAMPLE * len(LAYERS_TO_TEST)
    print(f"\nProcessing {total_queries} queries "
          f"({len(examples)} examples x {NUM_QUERIES_PER_EXAMPLE} queries x "
          f"{len(LAYERS_TO_TEST)} layers)")

    eval_start = time.time()

    for ex_idx, example in enumerate(examples):
        seq_len = example['sequence_length']
        domain = example.get('domain', '?')
        print(f"\nExample {ex_idx + 1}/{len(examples)}: "
              f"domain={domain[:40]}, seq_len={seq_len}")

        for layer_name in LAYERS_TO_TEST:
            layer_start = time.time()

            Q = np.array(example[layer_name]['Q'], dtype=np.float32)
            K_mat = np.array(example[layer_name]['K'], dtype=np.float32)
            V = np.array(example[layer_name]['V'], dtype=np.float32)

            # Select query positions from second half of sequence
            np.random.seed(SEED + ex_idx * 1000 + hash(layer_name) % 1000)
            n_available = seq_len // 2
            n_queries = min(NUM_QUERIES_PER_EXAMPLE, n_available)
            query_positions = np.random.choice(
                range(seq_len // 2, seq_len),
                size=n_queries,
                replace=False
            )

            # Fit GMM once per example+layer (on all keys)
            # Use maximum valid keys (full sequence) for GMM fitting
            resp_full = fit_gmm(K_mat, n_clusters=GMM_CLUSTERS, seed=SEED)

            for query_pos in tqdm(query_positions,
                                  desc=f"  {layer_name}", leave=False):
                q = Q[query_pos]

                result = compute_crossover_for_query(
                    q, K_mat, V, query_pos, HEAD_DIM, resp_full
                )
                result['example_idx'] = ex_idx
                result['query_pos'] = int(query_pos)
                result['layer'] = layer_name

                results_by_layer[layer_name].append(result)

            layer_time = time.time() - layer_start
            n_valid_bcross = sum(1 for r in results_by_layer[layer_name][-n_queries:]
                                if r['B_cross'] is not None)
            print(f"  {layer_name}: {n_queries} queries ({layer_time:.1f}s), "
                  f"{n_valid_bcross}/{n_queries} have valid B_cross")

    eval_time = time.time() - eval_start
    print(f"\nTotal evaluation time: {eval_time:.1f}s ({eval_time / 60:.1f} min)")

    # Create output directory
    output_dir = Path(os.path.abspath(OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute statistics
    print(f"\n{'=' * 70}")
    print("COMPUTING STATISTICS")
    print(f"{'=' * 70}")

    stats = compute_statistics(results_by_layer)

    # Print summary
    for layer_name in LAYERS_TO_TEST:
        s = stats[layer_name]
        print(f"\n--- {layer_name.upper()} ---")

        if s['B_cross']:
            bc = s['B_cross']
            print(f"  B_cross distribution (N={bc['n_valid']}/{bc['n_total']}):")
            print(f"    Median: {bc['median']:.1f}")
            print(f"    Mean:   {bc['mean']:.1f} +/- {bc['std']:.1f}")
            print(f"    Range:  [{bc['min']:.1f}, {bc['max']:.1f}]")
            print(f"    IQR:    [{bc['p25']:.1f}, {bc['p75']:.1f}]")

        print(f"  GMM error: mean={s['gmm_error']['mean']:.4f}, "
              f"median={s['gmm_error']['median']:.4f}")

        if s['correlation_log_predicted_empirical'] is not None:
            print(f"  Correlation (log predicted vs log empirical): "
                  f"{s['correlation_log_predicted_empirical']:.3f}")

        if s['prediction_accuracy']:
            pa = s['prediction_accuracy']
            print(f"  Prediction accuracy:")
            for factor in [2, 5, 10]:
                key = f'within_{factor}x'
                if key in pa:
                    print(f"    Within {factor}x: "
                          f"{pa[key]['fraction']:.1%} "
                          f"({pa[key]['count']}/{pa[key]['total']})")
            print(f"    Median pred/emp ratio: {pa.get('median_ratio', 'N/A')}")

        print(f"  Fraction where GMM wins by budget:")
        for budget in BUDGETS:
            if budget in s['seg_wins_fraction']:
                sf = s['seg_wins_fraction'][budget]
                print(f"    B={budget:4d}: {sf['fraction']:.1%} "
                      f"({sf['seg_wins']}/{sf['total']})")

    # Save full results
    print(f"\n{'=' * 70}")
    print("SAVING RESULTS")
    print(f"{'=' * 70}")

    # Convert oracle_errors keys from int to str for JSON
    full_results_json = {
        'metadata': {
            'experiment': 'crossover_budget_verification',
            'num_examples': NUM_EXAMPLES,
            'num_queries_per_example': NUM_QUERIES_PER_EXAMPLE,
            'oracle_trials': ORACLE_TRIALS,
            'budgets': BUDGETS,
            'gmm_clusters': GMM_CLUSTERS,
            'head_dim': HEAD_DIM,
            'seed': SEED,
            'layers': LAYERS_TO_TEST,
            'eval_time_seconds': eval_time,
            'timestamp': datetime.now().isoformat(),
        },
        'results_by_layer': {},
    }

    for layer_name, results in results_by_layer.items():
        layer_results = []
        for r in results:
            r_copy = dict(r)
            # Convert int keys to str for JSON
            r_copy['oracle_errors'] = {str(k): v for k, v in r['oracle_errors'].items()}
            r_copy['oracle_sq_errors'] = {str(k): v for k, v in r['oracle_sq_errors'].items()}
            layer_results.append(r_copy)
        full_results_json['results_by_layer'][layer_name] = layer_results

    full_path = output_dir / 'full_results.json'
    with open(full_path, 'w') as f:
        json.dump(full_results_json, f, indent=2)
    print(f"  Full results: {full_path} ({full_path.stat().st_size / 1024:.1f} KB)")

    # Save aggregated statistics
    agg_path = output_dir / 'aggregated.json'
    with open(agg_path, 'w') as f:
        json.dump({
            'metadata': full_results_json['metadata'],
            'statistics': stats,
        }, f, indent=2)
    print(f"  Aggregated: {agg_path}")

    # Generate plots
    print(f"\nGenerating plots...")
    plot_crossover_scatter(results_by_layer, output_dir)
    plot_error_curves(results_by_layer, output_dir)
    plot_bcross_distribution(results_by_layer, output_dir)
    plot_seg_wins_fraction(results_by_layer, output_dir)
    plot_var_vs_bias(results_by_layer, output_dir)

    print(f"\n{'=' * 70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Files:")
    print(f"  - full_results.json")
    print(f"  - aggregated.json")
    print(f"  - crossover_scatter_{{layer}}.png")
    print(f"  - error_curves_{{layer}}.png")
    print(f"  - bcross_distribution.png")
    print(f"  - seg_wins_fraction.png")
    print(f"  - var_vs_bias.png")


if __name__ == "__main__":
    main()
