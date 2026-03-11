# Experiment 1: Per-Layer Diagnostics

## Goal

Measure diagnostic statistics per layer (first_layer = layer 0 vs last_layer = layer 31) to understand why GMM soft clustering attention is more effective at last layer but less effective at first layer.

## Hypothesis

Last layer should have:
- **Lower entropy** (more concentrated attention)
- **Tighter clusters** (lower within-cluster logit variance)
- **Stronger key-value correlation** (keys predict values better)
- **Lower value effective rank** (values more structured)
- **Higher max attention weight** (sharper peaks)
- **Lower GMM error** (segmentation works better)

## Metrics

| # | Metric | Formula | Interpretation |
|---|--------|---------|----------------|
| 1 | Attention entropy | H(w) = -sum(w * log(w)) | Lower = more concentrated |
| 2 | Within-cluster logit variance | Weighted avg of var(logits) within GMM clusters | Lower = tighter clusters, better for segmentation |
| 3 | Key-value cosine correlation | mean(cos(k_i, v_i)) across positions | Higher magnitude = keys predict values |
| 4 | Value effective rank | (sum s_i)^2 / (sum s_i^2) via SVD | Lower = values more structured/compressible |
| 5 | Max attention weight | max_i(w_i) | Higher = sharper attention peaks |
| 6 | GMM error | Relative L2 error of GMM attention (C=50) | Lower = segmentation more effective |

## Configuration

- NUM_EXAMPLES = 10 (test), 100 (production)
- NUM_QUERIES_PER_EXAMPLE = 50
- GMM_CLUSTERS = 50
- HEAD_DIM = 128, SEED = 42
- Layers: first_layer (layer 0), last_layer (layer 31)
- Model: Llama-3-8B, head 0
- Data: LongBench v2 attention vectors

## Running

```bash
cd claude_experiments/01_per_layer_diagnostics
python experiment.py
```

## Results (10 examples, 500 queries per layer)

| Metric | First Layer | Last Layer | Ratio (L/F) |
|--------|------------|------------|-------------|
| Attention Entropy H(w) | 8.031 +/- 0.323 | 8.276 +/- 0.273 | 1.03 |
| Within-Cluster Logit Var | 0.074 +/- 0.026 | 0.321 +/- 0.114 | **4.33** |
| Key-Value Cosine Corr | -0.004 +/- 0.009 | 0.009 +/- 0.014 | -1.96 |
| Value Effective Rank | 85.8 +/- 10.3 | 90.2 +/- 8.5 | 1.05 |
| Max Attention Weight | 0.0039 +/- 0.0021 | 0.0021 +/- 0.0014 | 0.54 |
| **GMM Error (rel L2)** | **0.552 +/- 0.232** | **0.120 +/- 0.048** | **0.217** |

## Discussion

**GMM is 4.6x more effective at last layer** (error 0.120 vs 0.552), confirming the paper's claim that segmentation works better for later layers.

**Surprising findings that contradict the initial hypothesis:**

1. **Entropy is HIGHER at last layer** (8.28 vs 8.03): Last layer attention is actually *more diffuse*, not less. This is the opposite of what was hypothesized. It means GMM's advantage is not due to concentrated attention.

2. **Within-cluster logit variance is 4.3x HIGHER at last layer** (0.321 vs 0.074): Clusters are *looser*, not tighter. This is counterintuitive — GMM works better despite having worse clusters by this metric.

3. **Key-value correlation is near-zero in both layers** (-0.004 and 0.009): There's essentially no linear relationship between key and value vectors. This suggests k-v alignment is not the mechanism by which GMM helps.

4. **Max attention weight is LOWER at last layer** (0.002 vs 0.004): Attention is less peaked, reinforcing that last layer is more diffuse.

**Why does GMM still win at last layer?** The answer likely lies in the *absolute scale* of values. From Experiment 2 (crossover), we know Var_w(V) is ~350x larger at last layer (33.4 vs 0.096). This means the sampling variance is much larger, making GMM's fixed bias (even if relatively larger) still preferable over a wide budget range. The relevant comparison is not cluster quality but rather the ratio Var_w(V) / bias², which is much higher at last layer (B_cross median 340 vs 55).

**Implication**: GMM wins at last layer not because it's a *better segmentation* there, but because the alternative (sampling) is *much worse* due to high value variance.
