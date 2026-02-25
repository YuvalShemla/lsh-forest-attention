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
├── data/
│   ├── attention_vectors_updated_long.jsonl   # Q,K,V from Llama-3-8B (~40GB, 503 examples)
│   ├── longbench_v2_truncated_7k_smart.json   # LongBench v2 dataset (503 examples, ~8K tokens)
│   └── README_attention_vectors.md            # Data schema docs
├── data_extraction/
│   ├── generate_vectors_fixed.py              # Batch extraction from Llama-3-8B (GPU required)
│   └── extract_attention_vectors.ipynb        # Interactive extraction notebook
├── experiments/
│   ├── methods.py                 # Core: 5 attention approximation methods
│   ├── utils.py                   # LSHStructure class, softmax, error metrics, SNIS estimator
│   ├── compare.py                 # Main method comparison (all 5 methods, budget sweep)
│   ├── compare_min_depth.py       # Min-depth parameter sweep for prefix_sampling
│   ├── evaluate_recall_dcg.py     # Recall@K, DCG retrieval quality metrics
│   ├── plot_topk_approximation_error.py       # TopK vs Uniform vs Oracle comparison
│   ├── analyze_key_value_norm_relationship.py # Key-value norm correlation analysis
│   ├── explore_attention_data.py              # Attention distribution visualization
│   ├── generate_professional_dashboard.py     # HTML dashboard generation
│   ├── generate_dashboard_batched.py          # Batched dashboard computation
│   ├── generate_dashboard_visualize.py        # Dashboard visualization
│   ├── visualizations/
│   │   ├── plot_recall_dcg_clean.py           # Publication-quality recall/DCG bar plots
│   │   ├── plot_concentration_statistics.py   # Attention concentration analysis
│   │   ├── replot_min_depth.py                # Reprocess min-depth sweep results
│   │   └── reprocess_results.py               # Result reprocessing utility
│   ├── toy_needle_in_haystack_demo.ipynb      # Jungle Backtracking toy demo
│   ├── llama3_full_generation_jungle_vs_magicpig.ipynb  # End-to-end generation test
│   ├── approximation_error_measurement.ipynb  # Error tracking during generation
│   ├── match2_benchmark_evaluation.ipynb      # ANNA Match2 benchmark evaluation
│   └── llama2_generation_test.ipynb           # Llama-2 integration test
├── results/
│   ├── approximation_evaluation/v2/
│   │   ├── full_results.json & aggregated.json          # Main comparison results
│   │   ├── min_depth_sweep/                             # Min-depth parameter sweep
│   │   └── recall_dcg_evaluation/                       # Retrieval quality metrics
│   ├── topk_approximation_error_*.png         # TopK vs sampling plots
│   ├── attention_dashboard.html               # Interactive dashboard
│   └── dashboard_batches/                     # Precomputed dashboard data
└── attention_exploration_results/              # Early exploration visualizations
```

## Methods Implemented (`experiments/methods.py`)

Five attention approximation methods, all targeting a fixed key budget B out of ~6000 total keys:

1. **TopK** (`topk_approximation`): Select B highest-logit keys, compute subset softmax. Biased in diffuse regimes due to missing-mass problem (Eq. 4 in paper).

2. **Uniform Sampling** (`naive_sampling`): Sample B keys uniformly at random, compute subset softmax. Surprisingly strong baseline — avoids systematic bias toward high-logit keys.

3. **Oracle Sampling** (`oracle_sampling`): Sample B keys from the true attention distribution w, average values. Unbiased (Theorem 1) but requires full attention computation — serves as performance ceiling.

4. **LSH-SNIS** (`lsh_snis`): Fixed-depth SimHash retrieval with SNIS correction (MagicPIG-style). Variable candidate count depending on (K, L) and query. Includes multi-hit heuristic.

5. **Jungle Sampling / prefix_sampling** (`prefix_sampling`): **Our main contribution**. Uses LSH forest hierarchy to define a depth-mixture proposal:
   - Sample: tree l ~ Uniform([L]) → depth d ~ rho(d|q) → key i ~ Uniform(bucket)
   - Compute proposal probability pi_i(q) via LCP (longest common prefix) across all trees
   - Apply SNIS correction: weight each sample by exp(logit) / pi_i(q)
   - `min_depth` parameter controls selectivity vs. coverage tradeoff
   - `gamma`, `tau` parameters control depth distribution shape

## Data Pipeline

- **Source**: LongBench v2 prompts (503 examples, ~8K token contexts)
- **Model**: Llama-3-8B (32 layers, 8 KV heads, head_dim=128)
- **Extraction**: Forward hooks capture Q, K, V at first layer (layer 0) and last layer (layer 31), head 0
- **Format**: JSONL, one example per line (file is ~40GB — must read line-by-line, NOT full load)
- **Schema**: `{example_id, domain, sequence_length, model_metadata, first_layer: {Q, K, V}, last_layer: {Q, K, V}}`

## Evaluation Protocol

For each query position q_i with budget B:
1. Compute ground truth: full softmax attention with causal masking → exact output o*
2. Run sparse method: retrieve/sample B keys → approximate output o_hat
3. Metric: **Relative L2 error** = ||o_hat - o*||_2 / ||o*||_2

Standard config: L=10-50 tables, K_MAX=30 bits, budgets 20-200, seed=42, center_keys=True, 100 queries per example.

## Key Results

1. **Uniform > TopK** at low/moderate budgets in diffuse attention regimes
2. **Jungle Sampling > Uniform** with consistent ~5-15% error reduction across budgets
3. **Fixed-depth LSH-SNIS is unstable** — variable budget, sensitive to (K, L) choice
4. **Min-depth filtering**: moderate values (2-4) help slightly; too-high values increase bias
5. **Recall@100 vs. error**: higher NN recall doesn't always mean lower attention error in diffuse regimes

## Paper Structure (for reference)

- **Section 3**: Exploration phase — ANNA-style forest bucket averaging, evidence of diffuse attention
- **Section 4.2**: Jungle Backtracking — adaptive depth when buckets are empty (Eq. 9)
- **Section 4.3**: Jungle Sampling — depth-mixture proposal + SNIS estimator (Algorithm 1, Eq. 12-15)
- **Section 5**: Evaluation on Llama-3-8B attention snapshots
- **Appendix C**: Proof that fixed-depth forests = flat tables (Lemma 1)

## Running Experiments

```bash
# Setup
conda create -n forest python=3.10 && conda activate forest
pip install -r requirements.txt

# Extract attention vectors (GPU required, ~1 hour)
cd data_extraction && python generate_vectors_fixed.py

# Main method comparison (CPU, ~10 min)
cd experiments && python compare.py

# Min-depth parameter sweep
python compare_min_depth.py

# Retrieval quality metrics
python evaluate_recall_dcg.py

# TopK vs sampling analysis (100 examples, slower)
python plot_topk_approximation_error.py
```

Results go to `results/approximation_evaluation/v2/`. JSON files contain per-query errors and aggregated statistics (mean, median, std).

## Conventions

- All experiment scripts have hyperparameters at the top of the file
- Results are saved as both `full_results.json` (per-query) and `aggregated.json` (statistics)
- Plots use matplotlib/seaborn, saved as PNG
- Seed is always 42 for reproducibility
- The data file is too large for full memory load — always read line-by-line with `jsonl` iteration
