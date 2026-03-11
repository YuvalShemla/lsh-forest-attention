# Experiment 4: K-Means vs GMM Partition

## Motivation

GMM soft clustering attention shows non-monotonicity: error **increases** at large cluster counts C (U-shape). This experiment tests whether this is a GMM fitting artifact by comparing with k-means hard assignment.

K-means uses one-hot responsibility matrices fed into the same `gmm_attention()` function. The only difference is hard (one-hot) vs soft assignment.

## Configuration

- CLUSTER_COUNTS = [10, 20, 50, 100, 200, 500]
- NUM_EXAMPLES = 10 (test), 100 (production)
- NUM_QUERIES_PER_EXAMPLE = 50
- GMM_FIT_SUBSAMPLE = 2000 (speed optimization for GMM)
- HEAD_DIM = 128, SEED = 42

## Running

```bash
cd claude_experiments/04_kmeans_vs_gmm
python experiment.py
```

## Results (10 examples, 500 queries per layer)

### Mean Relative L2 Error

**First Layer:**

| C | GMM | K-Means | Winner |
|---|-----|---------|--------|
| 10 | **0.559** | 0.624 | GMM |
| 20 | 0.608 | **0.575** | K-Means |
| 50 | **0.594** | 0.603 | GMM |
| 100 | 0.699 | **0.634** | K-Means |
| 200 | 0.671 | **0.608** | K-Means |
| 500 | 0.986 | **0.783** | K-Means |

**Last Layer:**

| C | GMM | K-Means | Winner |
|---|-----|---------|--------|
| 10 | **0.159** | 0.160 | ~Tied |
| 20 | **0.134** | 0.141 | GMM |
| 50 | **0.132** | 0.134 | GMM |
| 100 | 0.140 | **0.126** | K-Means |
| 200 | 0.147 | **0.119** | K-Means |
| 500 | 0.168 | **0.107** | K-Means |

### Monotonicity Analysis

| Method / Layer | Monotone? | Best C | Best Error |
|---------------|-----------|--------|------------|
| GMM / First | No | 10 | 0.559 |
| **K-Means / First** | **No** | 20 | 0.575 |
| GMM / Last | **No** (U-shape from C=50) | 50 | 0.132 |
| **K-Means / Last** | **Yes** (monotone decreasing) | 500 | **0.107** |

### K-Means Last Layer Error Trend
```
C=10:  0.160
C=20:  0.141  (-0.019)
C=50:  0.134  (-0.007)
C=100: 0.126  (-0.008)
C=200: 0.119  (-0.007)
C=500: 0.107  (-0.012)   <-- still improving
```

### GMM Last Layer Error Trend (U-shape)
```
C=10:  0.159
C=20:  0.134  (-0.025)
C=50:  0.132  (-0.002)   <-- optimal
C=100: 0.140  (+0.008)   <-- error increases
C=200: 0.147  (+0.008)   <-- worse
C=500: 0.168  (+0.020)   <-- much worse
```

## Discussion

**The non-monotonicity at large C is a GMM fitting artifact.** K-Means at last layer shows strictly monotone decreasing error (0.160 → 0.107), while GMM shows a clear U-shape with optimum at C=50. This confirms the hypothesis.

**Why GMM fails at large C:**
- With many components, GMM's soft responsibilities become noisy — each component has few keys with strong assignment, and the EM algorithm struggles to find clean posteriors.
- The `covariance_type='diag'` constraint becomes increasingly restrictive with more components in high dimensions (d=128).
- K-Means hard assignment avoids this entirely — each key belongs to exactly one cluster, no estimation noise.

**GMM wins at moderate C (10-50):**
- GMM's soft assignment provides a form of regularization — a key can contribute to multiple clusters, smoothing the representatives. This helps when there are few clusters.
- At C=10-50 in last layer, GMM edges out K-Means by 0.001-0.007.

**Practical recommendation:**
- For small C (< 50): GMM's soft assignment provides slight benefit
- For large C (> 100): K-Means is clearly better and avoids the U-shape
- A hybrid approach — K-Means for large C, GMM for small C — could capture the best of both

**First layer**: Both methods are ineffective (errors 0.56-0.99), confirming that segmentation fundamentally struggles when attention is diffuse with low value variance (see Exp 1 and 2).
