# Jungle Attention: Forest-Aware LSH Proposals for Budgeted Sparse Attention

Experiment codebase for "Jungle Attention: Forest-Aware LSH Proposals for Budgeted Sparse Attention" (Gulcelik & Shemla, December 2025).

## Key Insight

Attention in real models is often **diffuse**, so:
- **TopK truncation** suffers from missing-mass bias
- **Fixed-depth LSH** (MagicPIG) produces variable candidate counts and empty-bucket failures
- **Uniform sampling** is a surprisingly strong baseline in diffuse regimes
- **Jungle Sampling** (our contribution) uses the LSH forest hierarchy to define budgeted, depth-mixture proposals corrected via SNIS

## Repository Structure

```
forest_attention_experiments/
├── src/
│   ├── algorithms/                     # One file per algorithm, common interface
│   │   ├── base.py                     # softmax, snis_estimator, relative_l2_error, inclusion_prob
│   │   ├── lsh_index.py                # LSHStructure, SimHashIndex, CrossPolytopeIndex
│   │   ├── full_attention.py           # Exact attention (ground truth)
│   │   ├── topk.py                     # TopK approximation
│   │   ├── uniform.py                  # Uniform random sampling
│   │   ├── oracle.py                   # Oracle sampling (from true distribution)
│   │   ├── oracle_value_weighted.py    # Oracle sampling weighted by ||v||
│   │   ├── simhash_snis.py             # SimHash fixed-depth LSH + SNIS
│   │   ├── cross_polytope_snis.py      # Cross-Polytope fixed-depth LSH + SNIS
│   │   ├── jungle_sampling.py          # LSH forest prefix_sampling (our method)
│   │   ├── hierarchical_lsh.py         # Hierarchical LSH tree-aggregation
│   │   ├── gmm_attention.py           # GMM soft clustering attention
│   │   ├── gmm_ablation.py            # GMM ablation variants (exact weights/values/both)
│   │   └── sorted_keys_grouping.py    # Sorted-keys grouping methods
│   │
│   ├── staging/                        # Experimental staging algorithms
│   │   ├── algorithms/
│   │   │   ├── pca_grouping.py        # Multi-Projection Fixed Grouping (PCA-based)
│   │   │   ├── prototype_query_grouping.py  # Prototype Query Grouping
│   │   │   ├── tree_adaptive_refinement.py  # Tree-Based Adaptive Refinement
│   │   │   ├── covariance_correction.py     # Fixed Grouping + Covariance Correction
│   │   │   ├── multi_representative.py      # Multi-Representative Per Group
│   │   │   └── query_aware_pq.py            # Query-Aware Product Quantization
│   │   └── experiments/
│   │       └── compare_staging_algorithms.py  # Main staging comparison
│   │
│   ├── visualization/
│   │   └── plot_utils.py               # Shared plotting: styles, error curves, scatter, save
│   │
│   ├── exploration/                    # Data analysis scripts
│   │   ├── attention_concentration.py  # Top-K concentration, entropy, Q-K similarity
│   │   ├── kv_norm_correlation.py      # Key-value norm relationship analysis
│   │   └── topk_vs_sampling_bias.py    # TopK vs Uniform vs Oracle bias analysis
│   │
│   └── experiments/                    # Experiment scripts
│       ├── compare_all_algorithms.py   # All 7 algorithms, budget sweep
│       ├── compare_simhash_vs_cp.py    # SimHash vs CrossPolytope parameter sweep
│       ├── exploration_dashboard.py    # HTML dashboard with exploration plots
│       ├── value_deviation_analysis.py # MagicPIG Fig 10 replication
│       ├── compare_oracle_variants.py  # Oracle vs Oracle-VW comparison
│       ├── attention_entropy_verification.py  # Attention entropy analysis
│       ├── gmm_bias_ablation.py        # GMM ablation experiments
│       ├── logit_discreteness_analysis.py     # Logit concentration analysis
│       ├── query_correlation_analysis.py      # Inter-query correlation
│       ├── run_lsh_vs_gmm_clustering.py       # LSH vs GMM clustering
│       ├── compare_grouping_methods.py        # Sorted-keys grouping vs baselines
│       ├── compare_mean_query_grouping.py     # Fixed vs per-query grouping
│       ├── compare_local_grouping.py          # Global vs local vs per-query grouping
│       ├── compare_clustering_baselines.py    # KMeans vs sorting-based grouping
│       └── hierarchical/              # Hierarchical LSH experiments
│           ├── compare_hierarchical_lsh.py    # (K, L) grid sweep
│           ├── run_hierarchical_L1_sweep.py   # L=1 multi-seed sweep
│           └── run_lsh_vs_gmm_clustering.py   # LSH vs GMM (extended)
│
├── data/                               # Attention vectors (not in git, see below)
├── data_extraction/
│   └── extract_vectors.py              # Batch extraction from Llama-3-8B (GPU required)
├── results/                            # Experiment outputs (not in git)
└── archive/                            # Old scripts and notebooks
```

## Quick Start

```bash
# 1. Setup environment
conda create -n forest python=3.10 && conda activate forest
pip install -r requirements.txt

# 2. Generate the attention vectors data file (GPU required, ~1 hour)
#    This runs Llama-3-8B inference on LongBench v2 prompts and extracts
#    Q, K, V vectors from the first and last attention layers.
#    Output: data/attention_vectors_long_bench_llama_8b.jsonl (~40GB)
python3 data_extraction/extract_vectors.py

# 3. Verify algorithms package
PYTHONPATH=src python3 -c "from algorithms import *; print('OK')"

# 4. Run experiments (from their respective directories)
cd src/experiments && python3 compare_all_algorithms.py
cd src/experiments && python3 compare_simhash_vs_cp.py
cd src/experiments && python3 exploration_dashboard.py
cd src/experiments/hierarchical && python3 compare_hierarchical_lsh.py
cd src/experiments/hierarchical && python3 run_hierarchical_v2.py
cd src/experiments/hierarchical && python3 run_hierarchical_L1_sweep.py

# 5. Run exploration scripts
cd src/exploration && python3 attention_concentration.py
cd src/exploration && python3 kv_norm_correlation.py
cd src/exploration && python3 topk_vs_sampling_bias.py
```

## Data

All experiments require the attention vectors file at:

```
data/attention_vectors_long_bench_llama_8b.jsonl
```

This file is generated by running `data_extraction/extract_vectors.py`, which:
1. Loads Llama-3-8B and the LongBench v2 dataset (`data/longbench_v2_truncated_7k_smart.json`)
2. Runs inference on each of the 503 prompts (~8K tokens each)
3. Extracts Q, K, V vectors from **layer 0** (first) and **layer 31** (last), head 0
4. Saves one JSONL line per example (total ~40GB)

**Schema per line:**
```json
{
  "example_id": "...",
  "domain": "Long In-context Learning",
  "sequence_length": 8192,
  "first_layer": {"layer_idx": 0, "Q": [[...]], "K": [[...]], "V": [[...]], "meta": {...}},
  "last_layer":  {"layer_idx": 31, "Q": [[...]], "K": [[...]], "V": [[...]], "meta": {...}}
}
```

Each Q/K/V array has shape `[seq_len, 128]` (head_dim=128). See `data/README_attention_vectors.md` for the full schema.

**Important:** The file is too large to load into memory. Always read line-by-line.

## Algorithms

All algorithms share a common interface: `(output: np.ndarray[head_dim], actual_budget: int)`.

| Algorithm | Budget Control | Description |
|-----------|---------------|-------------|
| Full Attention | N/A | Exact ground truth |
| TopK | Fixed | Select B highest-logit keys, subset softmax |
| Uniform Sampling | Fixed | Sample B keys uniformly, subset softmax |
| Oracle Sampling | Fixed | Sample from true distribution (privileged) |
| Oracle Value-Weighted | Fixed | Sample proportional to w_i * \|\|v_i\|\| (privileged) |
| SimHash-SNIS | Variable | Fixed-depth SimHash + SNIS correction |
| Cross-Polytope SNIS | Variable | Fixed-depth cross-polytope + SNIS |
| Jungle Sampling | Fixed | Depth-mixture proposal from LSH forest + SNIS |
| Hierarchical LSH | Variable | Tree-aggregation via count-weighted group softmax |
| GMM Attention | Fixed (n_clusters) | Soft clustering via GMM — responsibility-weighted representative keys/values |
| GMM Ablation | Fixed (n_clusters) | GMM variants: exact weights, exact values, exact both |
| Sorted-Keys Grouping | Fixed (n_groups) | Sort keys by logit, group, assign mean weight per group |

## Experiment Outputs

Each experiment saves results to a subdirectory under `results/`:

| Script | Output Directory |
|--------|-----------------|
| `compare_all_algorithms.py` | `results/all_algorithms_comparison/` |
| `compare_simhash_vs_cp.py` | `results/simhash_vs_cross_polytope/run_<timestamp>/` |
| `exploration_dashboard.py` | `results/exploration_dashboard/` |
| `compare_hierarchical_lsh.py` | `results/hierarchical_grid_sweep/run_<timestamp>/` |
| `run_hierarchical_L1_sweep.py` | `results/hierarchical_multi_seed/` |
| `compare_grouping_methods.py` | `results/grouping_comparison/` |
| `compare_mean_query_grouping.py` | `results/mean_query_grouping/` |
| `compare_local_grouping.py` | `results/local_query_grouping/` |
| `compare_clustering_baselines.py` | `results/clustering_baselines/` |
| `compare_staging_algorithms.py` | `results/staging_algorithms/` |
| Exploration scripts | `results/exploration/` |

## Key Results

1. **Uniform > TopK** at low/moderate budgets in diffuse regimes
2. **Jungle Sampling > Uniform** with ~5-15% error reduction
3. **Fixed-depth LSH-SNIS is unstable** (variable budget, sensitive to K/L)
4. **Cross-Polytope** shows different budget-error tradeoffs vs SimHash
