# Experiment 5: GMM Bias Source Ablation

## Question
Which bias source dominates in GMM attention — weight distortion (Source 1) or value averaging (Source 2)?

## Paper Reference
Section 4 bias decomposition, Section 8 ablation analysis

## Method
GMM attention has two bias sources:
- **Source 1 (Weight Distortion)**: softmax over centroid logits != true cluster weights W_c
- **Source 2 (Value Averaging)**: responsibility-weighted value means != attention-weighted value means

Five variants isolate each source:

| Variant | Weights | Values | What it tests |
|---------|---------|--------|---------------|
| Standard GMM | GMM centroids | GMM resp-weighted | Full GMM error |
| Exact Weights | Oracle W_c | GMM resp-weighted | Source 2 only |
| Exact Values | GMM centroids | Oracle attn-weighted | Source 1 only |
| Exact Both | Oracle W_c | Oracle attn-weighted | Partition error only |
| Oracle Sampling | N/A | Sampled at budget=C | Sampling baseline |

## Configuration
- NUM_EXAMPLES = 10 (test), 100 (production)
- NUM_QUERIES_PER_EXAMPLE = 100
- CLUSTERS = [10, 50, 100]
- HEAD_DIM = 128, SEED = 42

## Running
```bash
cd claude_experiments/05_gmm_bias_ablation
python3 experiment.py
```

## Results (10 examples, 1000 queries per layer)

### First Layer — Mean Relative L2 Error

| Variant | C=10 | C=50 | C=100 |
|---------|------|------|-------|
| Standard GMM | 0.573 | 0.557 | 0.600 |
| **Exact Weights** | **0.288** (-50%) | **0.088** (-84%) | **0.066** (-89%) |
| Exact Values | 0.483 (-16%) | 0.545 (-2%) | 0.592 (-1%) |
| Exact Both | ~0 | ~0 | ~0 |
| Oracle Sampling (B=C) | 1.248 | 0.564 | 0.400 |

### Last Layer — Mean Relative L2 Error

| Variant | C=10 | C=50 | C=100 |
|---------|------|------|-------|
| Standard GMM | 0.158 | 0.129 | 0.132 |
| **Exact Weights** | **0.114** (-28%) | **0.079** (-39%) | **0.065** (-51%) |
| Exact Values | 0.107 (-32%) | 0.098 (-24%) | 0.112 (-15%) |
| Exact Both | ~0 | ~0 | ~0 |
| Oracle Sampling (B=C) | 0.643 | 0.289 | 0.204 |

### Error Reduction from Fixing Each Source

| Source Fixed | First Layer (C=50) | Last Layer (C=50) |
|-------------|-------------------|-------------------|
| Weights only (S1 removed) | **84% reduction** | **39% reduction** |
| Values only (S2 removed) | 2% reduction | 24% reduction |
| Both removed | ~100% reduction | ~100% reduction |

## Discussion

**Weight distortion (Source 1) is the dominant bias source**, especially at the first layer and at higher C.

**First layer analysis:**
- Fixing weights eliminates 50-89% of error depending on C. At C=50, going from standard GMM (0.557) to Exact Weights (0.088) is an 84% reduction.
- Fixing values alone barely helps (2% at C=50) — the responsibility-weighted value averages are already good representatives.
- This means the bottleneck is computing correct cluster weights from centroid logits. The softmax(q^T k_bar / sqrt(d)) over averaged keys gives very poor approximations to the true cluster weights sum_i w_i * r_ic.

**Last layer analysis:**
- Both sources matter, but weights still dominate (39% vs 24% reduction at C=50).
- Exact Values helps more at last layer than first — likely because last layer has higher within-cluster logit variance (from Exp 1: 0.321 vs 0.074), making the attention-weighted value mean differ more from the responsibility-weighted mean.

**Partition quality is near-perfect:** Exact Both error is ~1e-6 in both layers, meaning the GMM partition captures the attention structure almost exactly. All the error comes from *how representatives are computed*, not from the partition itself.

**Oracle Sampling comparison:** At budget B=C, oracle sampling is worse than all GMM variants at C=10 (1.248 vs 0.573 at first layer), competitive at C=50 (0.564 vs 0.557), and better at C=100 (0.400 vs 0.600). This aligns with Experiment 2's crossover analysis.

**Practical implication:** The highest-leverage improvement to GMM attention is fixing the weight computation. A correction term or learned adjustment to the centroid logits could close most of the gap. Value averaging is already near-optimal.
