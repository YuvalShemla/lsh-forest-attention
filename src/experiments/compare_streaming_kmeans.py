#!/usr/bin/env python3
"""
Streaming KMeans vs Frozen-Assignment KMeans Experiment

Compares two fundamentally different approaches to incremental clustering
for approximate attention:

  1. **KMeans Keys (incr)** [existing] — Run full KMeans on ALL keys (looks ahead),
     freeze labels, then build cluster representatives causally. This cheats:
     it sees future keys to decide assignments.

  2. **Streaming KMeans (incr)** [new] — Truly online. Start with G random initial
     centers (sampled from the first G keys). As each new key arrives:
       a) Assign it to the nearest center (L2 distance to current running means)
       b) Update that cluster's running mean key, mean value, and count
     No look-ahead. O(G*d) per key insertion.

Both methods use the same frozen_group_attention at query time:
  logits = q @ mean_keys.T / sqrt(d) + log(group_sizes)
  output = softmax(logits) @ mean_values

Baselines: Oracle (privileged), Uniform (causal random sampling).

Usage:
  python compare_streaming_kmeans.py              # run compute + plot
  python compare_streaming_kmeans.py --plot-only  # regenerate plots from JSON
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

from algorithms.base import softmax
from algorithms.oracle import oracle_sampling
from visualization.plot_utils import setup_style, save_figure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
DATA_PATH = '../../data/attention_vectors_long_bench_llama_8b.jsonl'
OUTPUT_DIR = Path('../../results/streaming_kmeans')
NUM_EXAMPLES = 10
NUM_TEST_QUERIES = 30
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
BUDGETS = [2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024]

REGIONS = ['first', 'middle', 'last']
REGION_DISPLAY = {
    'first':  'First 30 Queries (Early Positions)',
    'middle': 'Middle 30 Queries (Center Positions)',
    'last':   'Last 30 Queries (Late Positions)',
}

METHOD_NAMES = [
    'Oracle',
    'Uniform',
    'KMeans Keys (incr)',
    'Streaming KMeans (incr)',
]

METHOD_COLORS = {
    'Oracle':                    '#2ca02c',
    'Uniform':                   '#7fbf7f',
    'KMeans Keys (incr)':        'darkorange',
    'Streaming KMeans (incr)':   '#1f77b4',
}

METHOD_MARKERS = {
    'Oracle':                    '^',
    'Uniform':                   's',
    'KMeans Keys (incr)':        'X',
    'Streaming KMeans (incr)':   'o',
}

METHOD_LINESTYLES = {m: '-' for m in METHOD_NAMES}

PLOT_METHODS = METHOD_NAMES


# ============================================================================
# HELPERS
# ============================================================================

def rel_l2(approx, truth):
    return np.linalg.norm(approx - truth) / (np.linalg.norm(truth) + 1e-8)


def format_eta(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def select_query_positions(seq_len, num_queries):
    positions = {}
    min_offset = max(30, BUDGETS[-1])
    first_start = min(min_offset, seq_len - num_queries)
    first_start = max(0, first_start)
    positions['first'] = list(range(first_start, first_start + num_queries))

    mid_center = seq_len // 2
    mid_start = mid_center - num_queries // 2
    mid_start = max(0, min(mid_start, seq_len - num_queries))
    positions['middle'] = list(range(mid_start, mid_start + num_queries))

    last_start = max(0, seq_len - num_queries)
    positions['last'] = list(range(last_start, last_start + num_queries))
    return positions


# ============================================================================
# FROZEN-ASSIGNMENT INCREMENTAL CLUSTERS (existing approach)
# ============================================================================

class IncrementalClusters:
    """
    Precomputed labels, running sums built causally.
    Keys are added one at a time to their pre-assigned cluster.
    """

    def __init__(self, G, head_dim):
        self.G = G
        self.d = head_dim
        self.key_sums = np.zeros((G, head_dim), dtype=np.float64)
        self.val_sums = np.zeros((G, head_dim), dtype=np.float64)
        self.counts = np.zeros(G, dtype=np.float64)

    def add_key(self, cluster_id, key_vec, val_vec):
        self.key_sums[cluster_id] += key_vec
        self.val_sums[cluster_id] += val_vec
        self.counts[cluster_id] += 1

    def get_representatives(self):
        active = self.counts > 0
        if not active.any():
            return None, None, None
        counts_active = self.counts[active]
        mk = self.key_sums[active] / counts_active[:, None]
        mv = self.val_sums[active] / counts_active[:, None]
        return mk, mv, counts_active


# ============================================================================
# STREAMING KMEANS
# ============================================================================

class StreamingKMeans:
    """
    Truly online KMeans. No look-ahead.

    Initialization: the first G keys each get their own cluster.
    After that, each new key is assigned to the nearest cluster center
    (L2 distance to current running mean), and the center is updated.

    At query time, expose the same (mean_keys, mean_values, sizes) interface.
    """

    def __init__(self, G, head_dim):
        self.G = G
        self.d = head_dim
        self.key_sums = np.zeros((G, head_dim), dtype=np.float64)
        self.val_sums = np.zeros((G, head_dim), dtype=np.float64)
        self.counts = np.zeros(G, dtype=np.float64)
        self.n_added = 0

    def add_key(self, key_vec, val_vec):
        """
        Add one key-value pair. Assigns to nearest cluster center online.
        First G keys each seed one cluster.
        """
        if self.n_added < self.G:
            # Seeding phase: each of the first G keys gets its own cluster
            cid = self.n_added
        else:
            # Find nearest center (L2 to current mean)
            # mean_keys[c] = key_sums[c] / counts[c] for active clusters
            active = self.counts > 0
            means = self.key_sums[active] / self.counts[active, None]
            # L2 distance to each active center
            diffs = means - key_vec  # [n_active, d]
            dists = np.sum(diffs * diffs, axis=1)  # [n_active]
            # Map back to global cluster id
            active_indices = np.where(active)[0]
            cid = int(active_indices[np.argmin(dists)])

        self.key_sums[cid] += key_vec
        self.val_sums[cid] += val_vec
        self.counts[cid] += 1
        self.n_added += 1

    def get_representatives(self):
        active = self.counts > 0
        if not active.any():
            return None, None, None
        counts_active = self.counts[active]
        mk = self.key_sums[active] / counts_active[:, None]
        mv = self.val_sums[active] / counts_active[:, None]
        return mk, mv, counts_active


def frozen_group_attention(query, mean_keys, mean_values, group_sizes):
    """Attend to group representatives with size-weighted logits."""
    group_logits = (query @ mean_keys.T) / np.sqrt(HEAD_DIM)
    group_logits = group_logits + np.log(group_sizes + 1e-10)
    group_weights = softmax(group_logits)
    return group_weights @ mean_values


# ============================================================================
# INCREMENTAL WALKS
# ============================================================================

def run_frozen_kmeans_walk(
    Q, K_mat, V, seq_len, labels, budget, test_positions_by_region
):
    """Frozen-assignment incremental walk (existing approach)."""
    G = int(labels.max()) + 1
    state = IncrementalClusters(G, HEAD_DIM)

    all_test = set()
    for region in REGIONS:
        all_test.update(test_positions_by_region[region])
    max_test_pos = max(all_test) if all_test else 0

    outputs = {r: [] for r in REGIONS}
    test_set_by_region = {r: set(test_positions_by_region[r]) for r in REGIONS}

    for pos in range(max_test_pos + 1):
        state.add_key(labels[pos], K_mat[pos], V[pos])

        for region in REGIONS:
            if pos in test_set_by_region[region]:
                mk, mv, gs = state.get_representatives()
                if mk is not None:
                    out = frozen_group_attention(Q[pos], mk, mv, gs)
                else:
                    out = np.zeros(HEAD_DIM, dtype=np.float64)
                outputs[region].append((pos, out))

    return outputs


def run_streaming_kmeans_walk(
    Q, K_mat, V, seq_len, budget, test_positions_by_region
):
    """
    Streaming KMeans walk — truly online assignment.
    No precomputed labels; each key is assigned on arrival.
    """
    G = min(budget, seq_len)
    state = StreamingKMeans(G, HEAD_DIM)

    all_test = set()
    for region in REGIONS:
        all_test.update(test_positions_by_region[region])
    max_test_pos = max(all_test) if all_test else 0

    outputs = {r: [] for r in REGIONS}
    test_set_by_region = {r: set(test_positions_by_region[r]) for r in REGIONS}

    for pos in range(max_test_pos + 1):
        state.add_key(K_mat[pos], V[pos])

        for region in REGIONS:
            if pos in test_set_by_region[region]:
                mk, mv, gs = state.get_representatives()
                if mk is not None:
                    out = frozen_group_attention(Q[pos], mk, mv, gs)
                else:
                    out = np.zeros(HEAD_DIM, dtype=np.float64)
                outputs[region].append((pos, out))

    return outputs


# ============================================================================
# ASSIGNMENT: FULL KMEANS (LOOK-AHEAD)
# ============================================================================

def assign_kmeans_keys(K_mat, seq_len):
    """KMeans on all keys, return {budget: labels[seq_len]}."""
    assignments = {}
    for budget in BUDGETS:
        b = min(budget, seq_len)
        if b >= seq_len:
            assignments[budget] = np.arange(seq_len, dtype=np.int32)
        else:
            km = KMeans(n_clusters=b, n_init=3, max_iter=100, random_state=SEED)
            km.fit(K_mat[:seq_len])
            assignments[budget] = km.labels_.astype(np.int32)
    return assignments


# ============================================================================
# MAIN COMPUTATION
# ============================================================================

def analyze_layer(data_path, selected_indices, layer_name, rng):
    print(f"\n{'='*60}")
    print(f"  {layer_name}")
    print(f"{'='*60}")

    errors = {
        region: {m: {b: [] for b in BUDGETS} for m in METHOD_NAMES}
        for region in REGIONS
    }

    selected_set = set(selected_indices)
    ex_count = 0
    total_examples = len(selected_indices)

    with open(data_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx not in selected_set:
                continue

            example = json.loads(line)
            Q = np.array(example[layer_name]['Q'], dtype=np.float32)
            K_mat = np.array(example[layer_name]['K'], dtype=np.float32)
            V = np.array(example[layer_name]['V'], dtype=np.float32)
            seq_len = Q.shape[0]

            region_positions = select_query_positions(seq_len, NUM_TEST_QUERIES)
            ex_count += 1

            print(f"\n  [{ex_count}/{total_examples}] Example {idx}: "
                  f"seq_len={seq_len}")
            print(f"    Positions: first={region_positions['first'][0]}-"
                  f"{region_positions['first'][-1]}, "
                  f"middle={region_positions['middle'][0]}-"
                  f"{region_positions['middle'][-1]}, "
                  f"last={region_positions['last'][0]}-"
                  f"{region_positions['last'][-1]}")

            # ---- Ground truth for all test queries ----
            gt_data = {}
            all_test_positions = set()
            for r in REGIONS:
                all_test_positions.update(region_positions[r])

            for qpos in all_test_positions:
                q = Q[qpos]
                keys = K_mat[:qpos + 1]
                vals = V[:qpos + 1]
                logits = (q @ keys.T) / np.sqrt(HEAD_DIM)
                full_w = softmax(logits)
                full_out = full_w @ vals
                gt_data[qpos] = (full_out, logits, full_w, keys, vals)

            # ---- Baselines: Oracle + Uniform ----
            print(f"    Baselines...")
            t0 = time.time()
            for region in REGIONS:
                for qpos in region_positions[region]:
                    full_out, logits, full_w, keys, vals = gt_data[qpos]
                    q = Q[qpos]
                    n_keys = qpos + 1
                    for budget in BUDGETS:
                        b = min(budget, n_keys)

                        out_oracle, _ = oracle_sampling(
                            q, keys, vals, logits, full_w, b
                        )
                        errors[region]['Oracle'][budget].append(
                            rel_l2(out_oracle, full_out)
                        )

                        u_idx = rng.choice(n_keys, size=b, replace=False)
                        out_uniform = softmax(logits[u_idx]) @ vals[u_idx]
                        errors[region]['Uniform'][budget].append(
                            rel_l2(out_uniform, full_out)
                        )
            print(f"    Baselines done in {time.time()-t0:.1f}s")

            # ---- KMeans Keys (incr) — full look-ahead ----
            print(f"    KMeans Keys (full look-ahead)...")
            t0 = time.time()
            kmeans_assignments = assign_kmeans_keys(K_mat, seq_len)
            print(f"    KMeans fit in {time.time()-t0:.1f}s")

            t0 = time.time()
            for budget in BUDGETS:
                labels_km = kmeans_assignments[budget]
                incr_out = run_frozen_kmeans_walk(
                    Q, K_mat, V, seq_len, labels_km, budget,
                    region_positions
                )
                for region in REGIONS:
                    for qpos, out in incr_out[region]:
                        full_out = gt_data[qpos][0]
                        errors[region]['KMeans Keys (incr)'][budget].append(
                            rel_l2(out, full_out)
                        )
            print(f"    KMeans eval done in {time.time()-t0:.1f}s")

            # ---- Streaming KMeans (incr) — truly online ----
            print(f"    Streaming KMeans (online)...")
            t0 = time.time()
            for budget in BUDGETS:
                incr_out = run_streaming_kmeans_walk(
                    Q, K_mat, V, seq_len, budget, region_positions
                )
                for region in REGIONS:
                    for qpos, out in incr_out[region]:
                        full_out = gt_data[qpos][0]
                        errors[region]['Streaming KMeans (incr)'][budget].append(
                            rel_l2(out, full_out)
                        )
                print(f"\r    Budget {budget}: done", end="", flush=True)
            print(f"\n    Streaming KMeans done in {time.time()-t0:.1f}s")

            del Q, K_mat, V, gt_data

    # Aggregate
    results = {'budgets': BUDGETS}
    for region in REGIONS:
        results[region] = {}
        for m in METHOD_NAMES:
            results[region][f'{m}_mean'] = [
                float(np.mean(errors[region][m][b]))
                if errors[region][m][b] else 0.0
                for b in BUDGETS
            ]
            results[region][f'{m}_std'] = [
                float(np.std(errors[region][m][b]))
                if errors[region][m][b] else 0.0
                for b in BUDGETS
            ]
    return results


# ============================================================================
# PLOTTING
# ============================================================================

def _plot_comparison(ax, data, region):
    x = np.array(data['budgets'])
    region_data = data[region]

    for method in PLOT_METHODS:
        if f'{method}_mean' not in region_data:
            continue
        means = np.array(region_data[f'{method}_mean'])
        stds = np.array(region_data[f'{method}_std'])
        color = METHOD_COLORS.get(method, '#999')
        marker = METHOD_MARKERS.get(method, 'o')
        ls = METHOD_LINESTYLES.get(method, '-')

        ax.plot(x, means, marker=marker, color=color, lw=2.5, ls=ls,
                label=method, zorder=4, markersize=6)
        hi = means + stds
        ax.fill_between(x, means, hi, color=color, alpha=0.12)

    ax.set_title(REGION_DISPLAY[region], fontsize=12, fontweight='bold')
    ax.set_xlabel('Budget (num groups)', fontsize=10)
    ax.set_ylabel('Relative L2 Error', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(left=0, right=512)
    ax.grid(True, alpha=0.3, ls='--', which='both')


def _plot_region_overlay(ax, data, method):
    x = np.array(data['budgets'])
    region_colors = {'first': '#e41a1c', 'middle': '#377eb8', 'last': '#4daf4a'}
    region_markers = {'first': 'o', 'middle': 's', 'last': '^'}

    for region in REGIONS:
        region_data = data[region]
        if f'{method}_mean' not in region_data:
            continue
        means = np.array(region_data[f'{method}_mean'])
        stds = np.array(region_data[f'{method}_std'])
        color = region_colors[region]
        marker = region_markers[region]

        ax.plot(x, means, marker=marker, color=color, lw=2.5,
                label=f'{region.capitalize()} queries', zorder=4, markersize=6)
        hi = means + stds
        ax.fill_between(x, means, hi, color=color, alpha=0.12)

    ax.set_title(method, fontsize=12, fontweight='bold')
    ax.set_xlabel('Budget', fontsize=10)
    ax.set_ylabel('Relative L2 Error', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(left=0, right=512)
    ax.grid(True, alpha=0.3, ls='--', which='both')


def make_figures(all_results, output_dir):
    cfg = all_results.get('config', {})
    n_ex = cfg.get('num_examples', NUM_EXAMPLES)
    n_q = cfg.get('num_test_queries', NUM_TEST_QUERIES)
    subtitle = (f'{n_ex} examples, {n_q} queries each  |  '
                f'Llama-3-8B  |  Shaded = +1 std')

    for layer in LAYERS:
        layer_data = all_results[layer]
        layer_short = 'first_layer' if 'first' in layer else 'last_layer'
        layer_title = ('First Layer (Layer 0)' if 'first' in layer
                       else 'Last Layer (Layer 31)')

        # --- Figure 1: by_region — 3 panels, all methods ---
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
        for i, region in enumerate(REGIONS):
            _plot_comparison(axes[i], layer_data, region)
            if i > 0:
                axes[i].set_ylabel('')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', fontsize=10,
                   framealpha=0.95, ncol=4, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(
            f'Streaming vs Frozen KMeans — {layer_title}\n{subtitle}',
            fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.06, 1, 0.94])
        save_figure(fig, output_dir / f'by_region_{layer_short}.png', dpi=200)
        plt.close(fig)

        # --- Figure 2: per_method — grid, regions overlaid ---
        n_methods = len(PLOT_METHODS)
        ncols = min(n_methods, 4)
        nrows = (n_methods + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(7 * ncols, 6 * nrows), sharey=True)
        if n_methods == 1:
            axes_flat = [axes]
        elif nrows == 1:
            axes_flat = list(axes)
        else:
            axes_flat = axes.flatten()

        for i, method in enumerate(PLOT_METHODS):
            _plot_region_overlay(axes_flat[i], layer_data, method)
            if i == 0:
                axes_flat[i].legend(fontsize=10, framealpha=0.95)

        for j in range(n_methods, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f'Per-Method Position Comparison — {layer_title}\n{subtitle}',
            fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, output_dir / f'per_method_{layer_short}.png', dpi=200)
        plt.close(fig)

        # --- Figure 3: summary — head-to-head all 4 methods ---
        fig, axes = plt.subplots(1, len(METHOD_NAMES),
                                 figsize=(6 * len(METHOD_NAMES), 6),
                                 sharey=True)
        for i, method in enumerate(METHOD_NAMES):
            _plot_region_overlay(axes[i], layer_data, method)
            if i == 0:
                axes[i].legend(fontsize=10, framealpha=0.95)
            if i > 0:
                axes[i].set_ylabel('')

        fig.suptitle(
            f'Streaming vs Frozen KMeans Head-to-Head — {layer_title}\n'
            f'{subtitle}',
            fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        save_figure(fig, output_dir / f'summary_{layer_short}.png', dpi=200)
        plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    plot_only = '--plot-only' in sys.argv

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(os.path.join(script_dir, str(OUTPUT_DIR)))
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    if plot_only:
        print("Plot-only mode: loading from JSON...")
        with open(output_dir / 'full_results.json') as f:
            all_results = json.load(f)
        print("Generating figures...")
        make_figures(all_results, output_dir)
        print("Done!")
        return

    print("=" * 60)
    print("STREAMING KMEANS vs FROZEN-ASSIGNMENT KMEANS")
    print("=" * 60)
    print(f"Config: {NUM_EXAMPLES} examples, {NUM_TEST_QUERIES} queries/region")
    print(f"Regions: {REGIONS}")
    print(f"Budgets: {BUDGETS}")
    print(f"Methods: {METHOD_NAMES}")
    print()
    print("KMeans Keys (incr): full look-ahead KMeans, frozen labels, causal reps")
    print("Streaming KMeans:   no look-ahead, assign-on-arrival, causal reps")
    print()

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    data_path = os.path.join(script_dir, DATA_PATH)

    print(f"Scanning: {data_path}")
    with open(data_path, 'r') as f:
        total = sum(1 for _ in f)
    print(f"Found {total} examples")

    n_select = min(NUM_EXAMPLES, total)
    selected = sorted(rng.choice(total, n_select, replace=False).tolist())
    print(f"Selected {n_select} examples: {selected}")

    all_results = {
        'config': {
            'num_examples': n_select,
            'num_test_queries': NUM_TEST_QUERIES,
            'budgets': BUDGETS,
            'seed': SEED,
            'head_dim': HEAD_DIM,
            'method_names': METHOD_NAMES,
            'regions': REGIONS,
        }
    }

    for layer in LAYERS:
        all_results[layer] = analyze_layer(data_path, selected, layer, rng)

    with open(output_dir / 'full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_dir / 'full_results.json'}")

    print("\nGenerating figures...")
    make_figures(all_results, output_dir)

    print(f"\nTotal time: {format_eta(time.time() - t0)}")
    print(f"Results in: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
