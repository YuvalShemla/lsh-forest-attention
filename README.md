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
│   │   ├── simhash_snis.py             # SimHash fixed-depth LSH + SNIS
│   │   ├── cross_polytope_snis.py      # Cross-Polytope fixed-depth LSH + SNIS
│   │   └── jungle_sampling.py          # LSH forest prefix_sampling (our method)
│   │
│   ├── visualization/
│   │   └── plot_utils.py               # Shared plotting: styles, error curves, scatter, save
│   │
│   ├── exploration/                    # Data analysis scripts
│   │   ├── attention_concentration.py  # Top-K concentration, entropy, Q-K similarity
│   │   ├── kv_norm_correlation.py      # Key-value norm relationship analysis
│   │   └── topk_vs_sampling_bias.py    # TopK vs Uniform vs Oracle bias analysis
│   │
│   └── experiments/                    # Three well-defined experiments
│       ├── compare_all_algorithms.py   # Exp 1: All 7 algorithms, budget sweep
│       ├── compare_simhash_vs_cp.py    # Exp 2: SimHash vs CrossPolytope parameter sweep
│       └── exploration_dashboard.py    # Exp 3: HTML dashboard with exploration plots
│
├── data/                               # Attention vectors from Llama-3-8B
├── data_extraction/
│   └── extract_vectors.py              # Batch extraction from Llama-3-8B (GPU required)
├── results/                            # Experiment outputs (JSON + PNG)
└── archive/                            # Old scripts and notebooks (preserved for reference)
```

## Algorithms

All algorithms share a common interface: `(output: np.ndarray[head_dim], actual_budget: int)`.

| Algorithm | Budget Control | Description |
|-----------|---------------|-------------|
| Full Attention | N/A | Exact ground truth |
| TopK | Fixed | Select B highest-logit keys, subset softmax |
| Uniform Sampling | Fixed | Sample B keys uniformly, subset softmax |
| Oracle Sampling | Fixed | Sample from true distribution (privileged) |
| SimHash-SNIS | Variable | Fixed-depth SimHash + SNIS correction |
| Cross-Polytope SNIS | Variable | Fixed-depth cross-polytope + SNIS |
| Jungle Sampling | Fixed | Depth-mixture proposal from LSH forest + SNIS |

## Running Experiments

```bash
# Setup
conda create -n forest python=3.10 && conda activate forest
pip install -r requirements.txt

# Verify algorithms package (from project root)
PYTHONPATH=src python3 -c "from algorithms import *; print('OK')"

# Exp 1: All algorithms comparison (CPU, ~10 min)
cd src/experiments && python3 compare_all_algorithms.py

# Exp 2: SimHash vs CrossPolytope sweep (CPU, ~1 hour)
cd src/experiments && python3 compare_simhash_vs_cp.py

# Exp 3: Exploration dashboard (CPU, ~30 min)
cd src/experiments && python3 exploration_dashboard.py

# Exploration scripts
cd src/exploration && python3 attention_concentration.py
cd src/exploration && python3 kv_norm_correlation.py
cd src/exploration && python3 topk_vs_sampling_bias.py
```

## Data

- **Source**: LongBench v2 prompts (503 examples, ~8K tokens)
- **Model**: Llama-3-8B (32 layers, 8 KV heads, head_dim=128)
- **Format**: JSONL (~40GB, read line-by-line, NOT full load)
- **Schema**: `{example_id, domain, sequence_length, first_layer: {Q, K, V}, last_layer: {Q, K, V}}`

Extract vectors (GPU required):
```bash
python3 data_extraction/extract_vectors.py
```

## Key Results

1. **Uniform > TopK** at low/moderate budgets in diffuse regimes
2. **Jungle Sampling > Uniform** with ~5-15% error reduction
3. **Fixed-depth LSH-SNIS is unstable** (variable budget, sensitive to K/L)
4. **Cross-Polytope** shows different budget-error tradeoffs vs SimHash
