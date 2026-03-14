# Jungle Attention: Forest-Aware LSH Proposals for Budgeted Sparse Attention

## Project Goal

Find the best approximate attention algorithm for long-context Transformer inference. The core problem: full attention is O(N^2) in sequence length and KV cache movement dominates decoding latency. We explore whether **LSH forests** (tree-structured LSH indices) can make sparse attention more robust and controllable than flat LSH tables or hard TopK truncation.

This repo is the experiment codebase for the paper "Jungle Attention: Forest-Aware LSH Proposals for Budgeted Sparse Attention" (Gulcelik & Shemla, December 2025).

## Key Insight

Attention in real models is often **diffuse** (not sharply concentrated on a few keys), so:
- **TopK truncation** suffers from missing-mass bias — it discards the long tail
- **Fixed-depth LSH** (e.g., MagicPIG) produces variable candidate counts and empty-bucket failures
- **Uniform sampling** is a surprisingly strong baseline in diffuse regimes
- **Jungle Sampling** (our contribution) uses the LSH forest hierarchy to define budgeted, depth-mixture proposals corrected via self-normalized importance sampling (SNIS), yielding consistent improvements over uniform

## Repository Structure

```
forest_attention_experiments/
├── README.md
├── CLAUDE.md
├── requirements.txt
│
├── src/
│   ├── algorithms/                         # One file per algorithm, common interface
│   │   ├── __init__.py                     # Re-exports all algorithms + classes
│   │   ├── base.py                         # softmax, snis_estimator, relative_l2_error, inclusion_prob
│   │   ├── lsh_index.py                    # LSHStructure, SimHashIndex, CrossPolytopeIndex
│   │   ├── full_attention.py               # Exact attention (ground truth)
│   │   ├── topk.py                         # TopK approximation
│   │   ├── uniform.py                      # Uniform random sampling
│   │   ├── oracle.py                       # Oracle sampling (from true distribution)
│   │   ├── oracle_value_weighted.py        # Oracle sampling weighted by ||v|| (privileged)
│   │   ├── simhash_snis.py                 # SimHash fixed-depth LSH + SNIS
│   │   ├── cross_polytope_snis.py          # Cross-Polytope fixed-depth LSH + SNIS
│   │   ├── jungle_sampling.py              # LSH forest prefix_sampling (our method)
│   │   ├── hierarchical_lsh.py             # Hierarchical LSH tree-aggregation
│   │   ├── gmm_attention.py               # GMM soft clustering attention
│   │   ├── gmm_ablation.py                # GMM ablation variants (exact weights/values/both)
│   │   └── sorted_keys_grouping.py        # Sorted-keys grouping methods (equal, kmeans, quantile, etc.)
│   │
│   ├── visualization/
│   │   └── plot_utils.py                   # Style setup, error curves, scatter, fig_to_base64, save
│   │
│   ├── exploration/                        # Data analysis scripts
│   │   ├── attention_concentration.py      # Top-K concentration, entropy, Q-K similarity, norms
│   │   ├── kv_norm_correlation.py          # Key-value norm relationship analysis
│   │   └── topk_vs_sampling_bias.py        # TopK vs Uniform vs Oracle bias analysis
│   │
│   └── experiments/                        # Experiment scripts
│       ├── compare_all_algorithms.py       # All 7 algorithms, budget sweep
│       ├── compare_simhash_vs_cp.py        # SimHash vs CrossPolytope parameter sweep
│       ├── exploration_dashboard.py        # HTML dashboard with exploration plots
│       ├── value_deviation_analysis.py     # MagicPIG Fig 10 replication
│       ├── compare_oracle_variants.py      # Oracle vs Oracle-VW comparison
│       ├── attention_entropy_verification.py # Attention entropy analysis
│       ├── gmm_bias_ablation.py            # GMM ablation: exact weights vs values vs both
│       ├── logit_discreteness_analysis.py  # Logit discreteness / concentration analysis
│       ├── query_correlation_analysis.py   # Inter-query correlation analysis
│       ├── run_lsh_vs_gmm_clustering.py    # LSH vs GMM clustering comparison
│       ├── compare_grouping_methods.py     # Sorted-keys grouping vs baselines (TopK, Uniform, Oracle)
│       ├── compare_mean_query_grouping.py  # Fixed (mean-query) vs per-query grouping
│       ├── compare_local_grouping.py       # Global vs local vs per-query grouping
│       ├── compare_clustering_baselines.py # KMeans keys, Query KMeans, vs sorting-based grouping
│       └── hierarchical/                   # Hierarchical LSH experiments
│           ├── compare_hierarchical_lsh.py # (K,L) grid sweep
│           ├── run_hierarchical_L1_sweep.py# L=1 multi-seed sweep
│           └── run_lsh_vs_gmm_clustering.py# LSH vs GMM clustering (extended, more clusters)
│
├── data/
│   ├── attention_vectors_long_bench_llama_8b.jsonl  # Q,K,V from Llama-3-8B (~40GB, 503 examples)
│   ├── longbench_v2_truncated_7k_smart.json         # LongBench v2 dataset (503 examples, ~8K tokens)
│   └── README_attention_vectors.md                  # Data schema docs
│
├── data_extraction/
│   └── extract_vectors.py              # Batch extraction from Llama-3-8B (GPU required)
│
├── results/                            # Generated outputs (not in git)
│   ├── all_algorithms_comparison/
│   ├── simhash_vs_cross_polytope/
│   ├── exploration_dashboard/
│   ├── hierarchical_grid_sweep/
│   ├── value_deviation_analysis/
│   ├── hierarchical_single_tree/
│   ├── hierarchical_multi_seed/
│   ├── grouping_comparison/
│   ├── mean_query_grouping/
│   ├── local_query_grouping/
│   ├── clustering_baselines/
│   └── exploration/
│
└── archive/                            # Old files preserved for reference
    ├── experiments_old/                # All old experiment scripts
    ├── notebooks/                      # All 5 experiment notebooks
    └── data_extraction_old/            # Old extraction notebook
```

## Algorithm Interface

Each algorithm file exports a main function returning `(output: np.ndarray[head_dim], actual_budget: int)`.

All algorithms import shared utilities from `base.py`. Each file is self-contained — no cross-imports between algorithm files.

| File | Function | Key Extra Params |
|------|----------|-----------------|
| `full_attention.py` | `full_attention(query, keys, values, logits, head_dim)` | None |
| `topk.py` | `topk_attention(query, keys, values, logits, budget)` | `budget` |
| `uniform.py` | `uniform_sampling(query, keys, values, logits, budget)` | `budget` |
| `oracle.py` | `oracle_sampling(query, keys, values, logits, true_weights, budget)` | `true_weights`, `budget` |
| `oracle_value_weighted.py` | `oracle_value_weighted(query, keys, values, logits, true_weights, budget)` | `true_weights`, `budget` |
| `simhash_snis.py` | `simhash_snis(query, keys, values, logits, head_dim, index, depth_k, L_use, min_hits)` | LSH index + params |
| `cross_polytope_snis.py` | `cross_polytope_snis(query, keys, values, logits, head_dim, index, k_cp, L_use, min_hits)` | LSH index + params |
| `jungle_sampling.py` | `jungle_sampling(query, keys, values, logits, head_dim, lsh_structure, budget, min_depth, gamma, tau)` | LSH structure + params |
| `hierarchical_lsh.py` | `hierarchical_lsh_attention(query, keys, values, logits, head_dim, key_codes, query_hash, K, L_use)` | Hash codes + depth/trees |
| `gmm_attention.py` | `gmm_attention(query, keys, values, logits, head_dim, resp)` | Precomputed GMM responsibilities |
| `gmm_ablation.py` | `gmm_exact_weights(...)`, `gmm_exact_values(...)`, `gmm_exact_both(...)` | GMM ablation: exact weights/values/both |
| `sorted_keys_grouping.py` | `grouped_attention(logits, values, weights, num_groups, method)` | `method` (equal/kmeans/quantile/log_spaced/variance/overlap) |

`fit_gmm(keys, n_clusters, seed)` fits a GMM on keys and returns `[N, n_clusters]` responsibilities. Call once per example, then pass sliced responsibilities per query.

`GROUPING_METHODS` dict maps method keys to display names. Available: `equal`, `kmeans`, `threshold`, `overlap`, `log_spaced`, `quantile`, `variance`.

## Import Pattern

From project root (with PYTHONPATH=src):
```python
from algorithms import topk_attention, uniform_sampling, jungle_sampling, gmm_attention, fit_gmm
from algorithms import LSHStructure, SimHashIndex, CrossPolytopeIndex
from algorithms import softmax, relative_l2_error, snis_estimator
from visualization.plot_utils import setup_style, save_figure, fig_to_base64
```

From src/experiments/ or src/exploration/ scripts (auto-resolve via sys.path):
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Goes up to src/, where algorithms/ and visualization/ live
from algorithms import ...
from visualization.plot_utils import ...
```

## Five Experiments

**Exp 1 — `compare_all_algorithms.py`**: Runs all 7 algorithms on the same data with one set of fixed params. Sweeps budget for budget-controlled methods. Uses fixed (K, L, min_hits) for SimHash-SNIS and Cross-Polytope-SNIS. Plots error-vs-budget curves (lines for budget-controlled, scatter for SNIS). Saves JSON + PNG.

**Exp 2 — `compare_simhash_vs_cp.py`**: SimHash K=[2-10], L=[5-75] and CrossPolytope k=[1,2], L=[5-200] with min_hits=[1,2,3]. Baselines (TopK, Uniform, Oracle) at budget percentages. Scatter plots of budget-vs-error. Min-hits comparison plots.

**Exp 3 — `exploration_dashboard.py`**: Batched processing of examples. Computes attention concentration, entropy, Q-K distances, norm analysis, K-V correlations. Produces individual PNG plots + self-contained HTML dashboard.

**Exp 4 — `compare_hierarchical_lsh.py`**: Evaluates hierarchical LSH tree-aggregation across K=[1,2,4,8,12,16,20] depths and L=[1,5,10,20,50,100] trees. Partitions all keys by LCP depth, aggregates via count-weighted softmax over group representatives, averages across trees. Baselines (TopK, Uniform, Oracle) at absolute budgets. Produces scatter plots (budget vs error), error heatmaps (K x L), and budget heatmaps (K x L).

**Exp 5 — `value_deviation_analysis.py`**: MagicPIG Figure 10 replication. For each query, plots pre-softmax attention scores (q·k_i^T / √d) vs value deviation (log‖v_i − o‖) across all positions. Shows that attention scores vary wildly while value deviations are flat — justifying focus on attention weight approximation. Produces per-query dual-axis plots, shared-axis overlays, multi-query summary figures, and aggregated variability statistics (boxplots, ratio histograms). Saves JSON + PNG.

## Data Pipeline

- **Source**: LongBench v2 prompts (503 examples, ~8K token contexts)
- **Model**: Llama-3-8B (32 layers, 8 KV heads, head_dim=128)
- **Extraction**: Forward hooks capture Q, K, V at first layer (layer 0) and last layer (layer 31), head 0
- **Format**: JSONL, one example per line (file is ~40GB — must read line-by-line, NOT full load)
- **Schema**: `{example_id, domain, sequence_length, model_metadata, first_layer: {Q, K, V}, last_layer: {Q, K, V}}`

## Evaluation Protocol

For each query position q_i with budget B:
1. Compute ground truth: full softmax attention with causal masking -> exact output o*
2. Run sparse method: retrieve/sample B keys -> approximate output o_hat
3. Metric: **Relative L2 error** = ||o_hat - o*||_2 / ||o*||_2

Standard config: L=10-50 tables, K_MAX=30 bits, budgets 20-200, seed=42, center_keys=True, 100 queries per example.

## Key Results

1. **Uniform > TopK** at low/moderate budgets in diffuse attention regimes
2. **Jungle Sampling > Uniform** with consistent ~5-15% error reduction across budgets
3. **Fixed-depth LSH-SNIS is unstable** — variable budget, sensitive to (K, L) choice
4. **Min-depth filtering**: moderate values (2-4) help slightly; too-high values increase bias
5. **Recall@100 vs. error**: higher NN recall doesn't always mean lower attention error in diffuse regimes

## Running Experiments

```bash
# Setup
conda create -n forest python=3.10 && conda activate forest
pip install -r requirements.txt

# Extract attention vectors (GPU required, ~1 hour)
python3 data_extraction/extract_vectors.py

# Main method comparison (CPU, ~10 min)
cd src/experiments && python3 compare_all_algorithms.py

# SimHash vs CrossPolytope sweep (CPU, ~1 hour)
cd src/experiments && python3 compare_simhash_vs_cp.py

# Exploration dashboard (CPU, ~30 min)
cd src/experiments && python3 exploration_dashboard.py

# Hierarchical LSH tree-aggregation sweep (CPU, ~30 min)
cd src/experiments && python3 compare_hierarchical_lsh.py

# Value deviation analysis - MagicPIG Fig 10 (CPU, ~5 min)
cd src/experiments && python3 value_deviation_analysis.py

# Exploration scripts
cd src/exploration && python3 attention_concentration.py
cd src/exploration && python3 kv_norm_correlation.py
cd src/exploration && python3 topk_vs_sampling_bias.py
```

Results go to `results/`. JSON files contain per-query errors and aggregated statistics.

## Conventions

- All experiment scripts have hyperparameters at the top of the file
- Results are saved as both `full_results.json` (per-query) and `aggregated.json` (statistics)
- Plots use matplotlib/seaborn, saved as PNG
- Seed is always 42 for reproducibility
- The data file is too large for full memory load — always read line-by-line with JSONL iteration
- Old scripts preserved in `archive/` for reference

## Paper Structure (for reference)

- **Section 3**: Exploration phase — ANNA-style forest bucket averaging, evidence of diffuse attention
- **Section 4.2**: Jungle Backtracking — adaptive depth when buckets are empty (Eq. 9)
- **Section 4.3**: Jungle Sampling — depth-mixture proposal + SNIS estimator (Algorithm 1, Eq. 12-15)
- **Section 5**: Evaluation on Llama-3-8B attention snapshots
- **Appendix C**: Proof that fixed-depth forests = flat tables (Lemma 1)
