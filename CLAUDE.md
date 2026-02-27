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
│   │   ├── simhash_snis.py                 # SimHash fixed-depth LSH + SNIS
│   │   ├── cross_polytope_snis.py          # Cross-Polytope fixed-depth LSH + SNIS
│   │   ├── jungle_sampling.py              # LSH forest prefix_sampling (our method)
│   │   ├── hierarchical_lsh.py             # Hierarchical LSH tree-aggregation
│   │   └── gmm_attention.py               # GMM soft clustering attention
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
│       ├── compare_all_algorithms.py       # Exp 1: All 7 algorithms, budget sweep
│       ├── compare_simhash_vs_cp.py        # Exp 2: SimHash vs CrossPolytope parameter sweep
│       ├── exploration_dashboard.py        # Exp 3: HTML dashboard with exploration plots
│       └── hierarchical/                   # Hierarchical LSH experiments
│           ├── compare_hierarchical_lsh.py # Exp 4: (K,L) grid sweep
│           ├── run_hierarchical_v2.py      # Exp 5: Single-tree depth sweep
│           └── run_hierarchical_L1_sweep.py# Exp 6: L=1 multi-seed sweep
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
│   ├── all_algorithms_comparison/      # Exp 1 output
│   ├── simhash_vs_cross_polytope/      # Exp 2 output
│   ├── exploration_dashboard/          # Exp 3 output
│   ├── hierarchical_grid_sweep/        # Exp 4 output
│   ├── hierarchical_single_tree/       # Exp 5 output
│   ├── hierarchical_multi_seed/        # Exp 6 output
│   └── exploration/                    # Exploration script plots
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
| `simhash_snis.py` | `simhash_snis(query, keys, values, logits, head_dim, index, depth_k, L_use, min_hits)` | LSH index + params |
| `cross_polytope_snis.py` | `cross_polytope_snis(query, keys, values, logits, head_dim, index, k_cp, L_use, min_hits)` | LSH index + params |
| `jungle_sampling.py` | `jungle_sampling(query, keys, values, logits, head_dim, lsh_structure, budget, min_depth, gamma, tau)` | LSH structure + params |
| `hierarchical_lsh.py` | `hierarchical_lsh_attention(query, keys, values, logits, head_dim, key_codes, query_hash, K, L_use)` | Hash codes + depth/trees |
| `gmm_attention.py` | `gmm_attention(query, keys, values, logits, head_dim, resp)` | Precomputed GMM responsibilities |

`fit_gmm(keys, n_clusters, seed)` fits a GMM on keys and returns `[N, n_clusters]` responsibilities. Call once per example, then pass sliced responsibilities per query.

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

## Four Experiments

**Exp 1 — `compare_all_algorithms.py`**: Runs all 7 algorithms on the same data with one set of fixed params. Sweeps budget for budget-controlled methods. Uses fixed (K, L, min_hits) for SimHash-SNIS and Cross-Polytope-SNIS. Plots error-vs-budget curves (lines for budget-controlled, scatter for SNIS). Saves JSON + PNG.

**Exp 2 — `compare_simhash_vs_cp.py`**: SimHash K=[2-10], L=[5-75] and CrossPolytope k=[1,2], L=[5-200] with min_hits=[1,2,3]. Baselines (TopK, Uniform, Oracle) at budget percentages. Scatter plots of budget-vs-error. Min-hits comparison plots.

**Exp 3 — `exploration_dashboard.py`**: Batched processing of examples. Computes attention concentration, entropy, Q-K distances, norm analysis, K-V correlations. Produces individual PNG plots + self-contained HTML dashboard.

**Exp 4 — `compare_hierarchical_lsh.py`**: Evaluates hierarchical LSH tree-aggregation across K=[1,2,4,8,12,16,20] depths and L=[1,5,10,20,50,100] trees. Partitions all keys by LCP depth, aggregates via count-weighted softmax over group representatives, averages across trees. Baselines (TopK, Uniform, Oracle) at absolute budgets. Produces scatter plots (budget vs error), error heatmaps (K x L), and budget heatmaps (K x L).

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
