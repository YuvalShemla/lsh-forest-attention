"""
Vector Correlation Analysis: Queries, Keys, Values, and Outputs.

For each example, takes the last 10 positions and measures pairwise
cosine similarity and distance from mean for:
1. Queries (Q[pos])
2. Keys (K[pos])  — same last 10 positions for fair comparison
3. Values (V[pos]) — same last 10 positions
4. Attention outputs (o = softmax(qK^T/√d) V with causal mask)

Aggregates over 50 examples, saves JSON + plots.
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
NUM_EXAMPLES = 50
NUM_QUERIES = 10   # last 10 positions per example
HEAD_DIM = 128
SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../results/query_correlation')

VEC_TYPES = ['query', 'key', 'value', 'output']
VEC_COLORS = {'query': 'steelblue', 'key': '#2ca02c', 'value': '#9467bd', 'output': 'darkorange'}


def pairwise_cosine_all(X):
    """All pairwise cosine similarities for X [n, d]. Returns flat array of n*(n-1)/2 values."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normed = X / np.maximum(norms, 1e-12)
    sim_matrix = X_normed @ X_normed.T
    n = X.shape[0]
    idx = np.triu_indices(n, k=1)
    return sim_matrix[idx]


def pairwise_l2_all(X):
    """All pairwise L2 distances for X [n, d]. Returns flat array."""
    n = X.shape[0]
    norms_sq = np.sum(X ** 2, axis=1)
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * (X @ X.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    dist_matrix = np.sqrt(dist_sq)
    idx = np.triu_indices(n, k=1)
    return dist_matrix[idx]


def compute_attention_output(q, K, V, query_pos):
    """Full attention with causal mask."""
    valid_keys = K[:query_pos + 1]
    valid_values = V[:query_pos + 1]
    logits = (q @ valid_keys.T) / np.sqrt(HEAD_DIM)
    weights = softmax(logits)
    return weights @ valid_values


def analyze_vectors(X):
    """Compute similarity stats for a set of vectors [n, d]."""
    mean_vec = X.mean(axis=0)
    mean_vec_norm = float(np.linalg.norm(mean_vec))
    dists_from_mean = np.linalg.norm(X - mean_vec, axis=1)
    norms = np.linalg.norm(X, axis=1)
    cosine_sims = pairwise_cosine_all(X)
    l2_dists = pairwise_l2_all(X)

    return {
        'mean_vec_norm': mean_vec_norm,
        'norms': {'mean': float(np.mean(norms)), 'std': float(np.std(norms))},
        'dist_from_mean': {'mean': float(np.mean(dists_from_mean)), 'std': float(np.std(dists_from_mean)),
                           'values': dists_from_mean.tolist()},
        'cosine_sim': {'mean': float(np.mean(cosine_sims)), 'std': float(np.std(cosine_sims)),
                       'min': float(np.min(cosine_sims)), 'max': float(np.max(cosine_sims)),
                       'values': cosine_sims.tolist()},
        'l2_dist': {'mean': float(np.mean(l2_dists)), 'std': float(np.std(l2_dists)),
                    'values': l2_dists.tolist()},
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    # Aggregated histograms
    agg = {
        layer: {vt: {'dists_from_mean': [], 'cosine_sims': [], 'l2_dists': []}
                for vt in VEC_TYPES}
        for layer in ['first_layer', 'last_layer']
    }

    print(f"Loading {NUM_EXAMPLES} examples, {NUM_QUERIES} positions each")
    print(f"Analyzing: {', '.join(VEC_TYPES)}\n")

    with open(DATA_PATH, 'r') as f:
        for ex_idx, line in enumerate(f):
            if ex_idx >= NUM_EXAMPLES:
                break
            example = json.loads(line)
            domain = example.get('domain', 'unknown')
            seq_len = example.get('sequence_length', 0)
            print(f"  [{ex_idx:3d}] {domain} (seq_len={seq_len})")

            ex_result = {'example_idx': ex_idx, 'domain': domain, 'seq_len': seq_len}

            for layer_name in ['first_layer', 'last_layer']:
                Q = np.array(example[layer_name]['Q'], dtype=np.float64)
                K_mat = np.array(example[layer_name]['K'], dtype=np.float64)
                V_mat = np.array(example[layer_name]['V'], dtype=np.float64)
                actual_seq_len = Q.shape[0]

                # Last NUM_QUERIES positions
                start = max(0, actual_seq_len - NUM_QUERIES)
                positions = list(range(start, actual_seq_len))

                # Extract vectors at the same positions
                queries = Q[start:actual_seq_len]      # [10, 128]
                keys = K_mat[start:actual_seq_len]      # [10, 128]
                values = V_mat[start:actual_seq_len]    # [10, 128]

                # Compute attention outputs
                outputs = np.zeros((len(positions), HEAD_DIM), dtype=np.float64)
                for i, qpos in enumerate(positions):
                    outputs[i] = compute_attention_output(Q[qpos], K_mat, V_mat, qpos)

                # Analyze all four
                vec_data = {'query': queries, 'key': keys, 'value': values, 'output': outputs}
                layer_result = {}
                for vt in VEC_TYPES:
                    stats = analyze_vectors(vec_data[vt])
                    layer_result[vt] = {k: v for k, v in stats.items()}

                    # Collect for aggregated histograms
                    agg[layer_name][vt]['dists_from_mean'].extend(stats['dist_from_mean']['values'])
                    agg[layer_name][vt]['cosine_sims'].extend(stats['cosine_sim']['values'])
                    agg[layer_name][vt]['l2_dists'].extend(stats['l2_dist']['values'])

                ex_result[layer_name] = layer_result

            all_results.append(ex_result)
            del example

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("SUMMARY (averaged over examples)")
    print("=" * 70)
    for layer in ['first_layer', 'last_layer']:
        print(f"\n{'FIRST LAYER' if layer == 'first_layer' else 'LAST LAYER'}:")
        for vt in VEC_TYPES:
            means_cos = [r[layer][vt]['cosine_sim']['mean'] for r in all_results]
            means_dist = [r[layer][vt]['dist_from_mean']['mean'] for r in all_results]
            means_norm = [r[layer][vt]['mean_vec_norm'] for r in all_results]
            print(f"  {vt.upper():8s}  cos_sim={np.mean(means_cos):.4f} (±{np.std(means_cos):.4f})  "
                  f"||x-mean||={np.mean(means_dist):.3f} (±{np.std(means_dist):.3f})  "
                  f"||mean||={np.mean(means_norm):.2f}")

    # ── Save JSON ──
    json_path = os.path.join(OUTPUT_DIR, 'vector_correlation_stats.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ═══════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════

    layer_labels = {'first_layer': 'First Layer (0)', 'last_layer': 'Last Layer (31)'}

    # ── Plot 1: Aggregated cosine similarity histograms (4x2: vec_type x layer) ──
    # Find global x-range across all panels
    all_cos_vals = []
    for layer in ['first_layer', 'last_layer']:
        for vt in VEC_TYPES:
            all_cos_vals.extend(agg[layer][vt]['cosine_sims'])
    cos_xmin, cos_xmax = min(all_cos_vals), max(all_cos_vals)
    cos_bins = np.linspace(cos_xmin, cos_xmax, 61)

    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    # First pass: collect max y to share y-axis
    all_counts = []
    for row, vt in enumerate(VEC_TYPES):
        for col, layer in enumerate(['first_layer', 'last_layer']):
            vals = np.array(agg[layer][vt]['cosine_sims'])
            counts, _ = np.histogram(vals, bins=cos_bins)
            all_counts.append(counts.max())
    cos_ymax = max(all_counts) * 1.15

    for row, vt in enumerate(VEC_TYPES):
        for col, layer in enumerate(['first_layer', 'last_layer']):
            ax = axes[row, col]
            vals = np.array(agg[layer][vt]['cosine_sims'])
            m = np.mean(vals)
            ax.hist(vals, bins=cos_bins, alpha=0.7, color=VEC_COLORS[vt], edgecolor='white', linewidth=0.3)
            ax.axvline(m, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {m:.4f}')
            ax.set_xlabel('Pairwise Cosine Similarity')
            ax.set_ylabel('Count')
            ax.set_title(f'{vt.capitalize()} — {layer_labels[layer]}')
            ax.set_xlim(cos_xmin - 0.02, cos_xmax + 0.02)
            ax.set_ylim(0, cos_ymax)
            ax.legend(fontsize=9)
    plt.suptitle(f'Pairwise Cosine Similarity ({NUM_EXAMPLES} examples, {NUM_QUERIES} positions each)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cosine_similarity_all_types.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: cosine_similarity_all_types.png")

    # ── Plot 2: Aggregated ||x - mean|| histograms (4x2) ──
    # Find global x-range
    all_dist_vals = []
    for layer in ['first_layer', 'last_layer']:
        for vt in VEC_TYPES:
            all_dist_vals.extend(agg[layer][vt]['dists_from_mean'])
    dist_xmin, dist_xmax = min(all_dist_vals), max(all_dist_vals)
    dist_bins = np.linspace(dist_xmin, dist_xmax, 61)

    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    # First pass for y-max
    all_dcounts = []
    for row, vt in enumerate(VEC_TYPES):
        for col, layer in enumerate(['first_layer', 'last_layer']):
            vals = np.array(agg[layer][vt]['dists_from_mean'])
            counts, _ = np.histogram(vals, bins=dist_bins)
            all_dcounts.append(counts.max())
    dist_ymax = max(all_dcounts) * 1.15

    for row, vt in enumerate(VEC_TYPES):
        for col, layer in enumerate(['first_layer', 'last_layer']):
            ax = axes[row, col]
            vals = np.array(agg[layer][vt]['dists_from_mean'])
            m = np.mean(vals)
            ax.hist(vals, bins=dist_bins, alpha=0.7, color=VEC_COLORS[vt], edgecolor='white', linewidth=0.3)
            ax.axvline(m, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {m:.3f}')
            ax.set_xlabel('||x - mean(X)||₂')
            ax.set_ylabel('Count')
            ax.set_title(f'{vt.capitalize()} — {layer_labels[layer]}')
            ax.set_xlim(dist_xmin - 0.1, dist_xmax + 0.1)
            ax.set_ylim(0, dist_ymax)
            ax.legend(fontsize=9)
    plt.suptitle(f'Distance from Centroid ({NUM_EXAMPLES} examples, {NUM_QUERIES} positions each)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dist_from_mean_all_types.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: dist_from_mean_all_types.png")

    # ── Plot 3: Box plots — all 4 types × 2 layers (THE MAIN SUMMARY PLOT) ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Cosine sim
    ax = axes[0]
    data = []
    labels = []
    colors = []
    for vt in VEC_TYPES:
        for layer in ['first_layer', 'last_layer']:
            vals = [r[layer][vt]['cosine_sim']['mean'] for r in all_results]
            data.append(vals)
            short_layer = 'L0' if layer == 'first_layer' else 'L31'
            labels.append(f'{vt.capitalize()}\n{short_layer}')
            colors.append(VEC_COLORS[vt])
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)
    for i, d in enumerate(data, 1):
        x_jit = np.random.default_rng(42).normal(i, 0.04, len(d))
        ax.scatter(x_jit, d, alpha=0.3, s=12, c='gray', zorder=3)
    ax.set_ylabel('Mean Pairwise Cosine Similarity')
    ax.set_title('Cosine Similarity')
    ax.tick_params(axis='x', labelsize=8)

    # Dist from mean
    ax = axes[1]
    data = []
    labels = []
    colors_list = []
    for vt in VEC_TYPES:
        for layer in ['first_layer', 'last_layer']:
            vals = [r[layer][vt]['dist_from_mean']['mean'] for r in all_results]
            data.append(vals)
            short_layer = 'L0' if layer == 'first_layer' else 'L31'
            labels.append(f'{vt.capitalize()}\n{short_layer}')
            colors_list.append(VEC_COLORS[vt])
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, c in zip(bp['boxes'], colors_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)
    for i, d in enumerate(data, 1):
        x_jit = np.random.default_rng(42).normal(i, 0.04, len(d))
        ax.scatter(x_jit, d, alpha=0.3, s=12, c='gray', zorder=3)
    ax.set_ylabel('Mean ||x - mean(X)||₂')
    ax.set_title('Distance from Centroid')
    ax.tick_params(axis='x', labelsize=8)

    plt.suptitle(f'Vector Correlation Summary ({NUM_EXAMPLES} examples, last {NUM_QUERIES} positions)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'boxplots_all_types.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: boxplots_all_types.png")

    # ── Plot 4: Per-example scatter — first vs last layer, all 4 types ──
    # Find global axis range across all 4 subplots
    scatter_all_vals = []
    for vt in VEC_TYPES:
        scatter_all_vals.extend([r['first_layer'][vt]['cosine_sim']['mean'] for r in all_results])
        scatter_all_vals.extend([r['last_layer'][vt]['cosine_sim']['mean'] for r in all_results])
    scatter_lims = [min(scatter_all_vals) - 0.03, max(scatter_all_vals) + 0.03]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, vt in zip(axes.flat, VEC_TYPES):
        first = [r['first_layer'][vt]['cosine_sim']['mean'] for r in all_results]
        last = [r['last_layer'][vt]['cosine_sim']['mean'] for r in all_results]
        ax.scatter(first, last, alpha=0.6, s=40, c=VEC_COLORS[vt], edgecolors='k', linewidth=0.3)
        ax.plot(scatter_lims, scatter_lims, 'k--', alpha=0.3, linewidth=1)
        ax.set_xlabel('First Layer — Mean Cosine Sim')
        ax.set_ylabel('Last Layer — Mean Cosine Sim')
        ax.set_title(f'{vt.capitalize()} Pairwise Cosine Similarity')
        ax.set_xlim(scatter_lims)
        ax.set_ylim(scatter_lims)
        ax.set_aspect('equal')
        above = sum(1 for f, l in zip(first, last) if l > f)
        below = len(first) - above
        ax.text(0.05, 0.95, f'Above diag: {above}\nBelow diag: {below}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.suptitle('Per-Example: First Layer vs Last Layer', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'per_example_scatter_all_types.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: per_example_scatter_all_types.png")

    # ── Plot 5: Sorted per-example cosine sim — all types overlaid per layer ──
    # Shared y-axis range
    sorted_ymin = min(scatter_all_vals) - 0.03
    sorted_ymax = max(scatter_all_vals) + 0.03

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, layer in zip(axes, ['first_layer', 'last_layer']):
        for vt in VEC_TYPES:
            vals = sorted([r[layer][vt]['cosine_sim']['mean'] for r in all_results])
            ax.plot(range(len(vals)), vals, 'o-', markersize=3, label=vt.capitalize(),
                    color=VEC_COLORS[vt])
        ax.set_xlabel('Example (sorted)')
        ax.set_ylabel('Mean Pairwise Cosine Similarity')
        ax.set_title(layer_labels[layer])
        ax.set_ylim(sorted_ymin, sorted_ymax)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.suptitle('Per-Example Cosine Similarity — All Vector Types', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sorted_cosine_all_types.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: sorted_cosine_all_types.png")

    # ── Plot 6: Compact bar chart — mean cosine sim per type × layer ──
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(VEC_TYPES))
    width = 0.35
    for i, (layer, offset, hatch) in enumerate([('first_layer', -width/2, ''), ('last_layer', width/2, '//')]):
        means = [np.mean([r[layer][vt]['cosine_sim']['mean'] for r in all_results]) for vt in VEC_TYPES]
        stds = [np.std([r[layer][vt]['cosine_sim']['mean'] for r in all_results]) for vt in VEC_TYPES]
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4, hatch=hatch,
                      color=[VEC_COLORS[vt] for vt in VEC_TYPES], alpha=0.6 + 0.2*i,
                      edgecolor='k', linewidth=0.5, label=layer_labels[layer])
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([vt.capitalize() for vt in VEC_TYPES])
    ax.set_ylabel('Mean Pairwise Cosine Similarity')
    ax.set_title(f'Vector Correlation Summary ({NUM_EXAMPLES} examples)')
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bar_chart_cosine_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: bar_chart_cosine_summary.png")

    print("\nDone!")


if __name__ == '__main__':
    main()
