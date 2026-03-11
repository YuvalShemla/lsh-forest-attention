# Experiment 3: Nearest-Value Key Selection (Strategy A')

## Hypothesis

Standard GMM attention uses responsibility-weighted averaged keys as cluster representatives. Averaging keys may distort the query-key dot product (Jensen's inequality). Strategy A' replaces the averaged key with the key of the position whose value is nearest to the cluster's value centroid, preserving a "real" key vector.

## Method

**Strategy A' (Nearest-Value Key)**:
1. Compute the value centroid for each cluster: `v_bar_c = sum_i r_ic * v_i / sum_i r_ic`
2. Among positions with `r_ic > 0.01`, find `i* = argmin_i ||v_i - v_bar_c||`
3. Use `k_{i*}` as the representative key for that cluster
4. Value representative remains the responsibility-weighted average (same as standard GMM)

## Comparisons

| Method | Keys | Values | Oracle? |
|--------|------|--------|---------|
| Standard GMM | Resp-weighted avg | Resp-weighted avg | No |
| Nearest-Value Key (A') | Key of nearest-to-centroid value | Resp-weighted avg | No |
| Exact Both | Oracle weights | Attention-weighted avg | Yes |

## Configuration

- NUM_EXAMPLES = 10 (test), 100 (production)
- NUM_QUERIES_PER_EXAMPLE = 50
- CLUSTER_COUNTS = [10, 50, 100]
- HEAD_DIM = 128, SEED = 42

## Running

```bash
cd claude_experiments/03_nearest_value_key
python experiment.py
```

## Results (10 examples, 500 queries per layer)

### Mean Relative L2 Error

**First Layer:**

| Method | C=10 | C=50 | C=100 |
|--------|------|------|-------|
| Standard GMM | 0.584 | 0.552 | 0.594 |
| **Nearest-Value Key (A')** | **1.052** | **0.734** | **0.808** |
| Exact Both | ~0 | ~0 | ~0 |

**Last Layer:**

| Method | C=10 | C=50 | C=100 |
|--------|------|------|-------|
| Standard GMM | 0.156 | 0.120 | 0.123 |
| **Nearest-Value Key (A')** | **0.232** | **0.155** | **0.141** |
| Exact Both | ~0 | ~0 | ~0 |

### Improvement Statistics (A' vs Standard GMM)

| Layer | C=10 | C=50 | C=100 |
|-------|------|------|-------|
| First | **-120%** (worse) | **-49%** (worse) | **-40%** (worse) |
| Last | **-63%** (worse) | **-37%** (worse) | **-19%** (worse) |
| Frac improved | 8-21% | 13-17% | 10-17% |

## Discussion

**Strategy A' is consistently and significantly worse than standard GMM.** This is a clean negative result.

**Why averaged keys are better:** The responsibility-weighted averaged key `k_bar_c = sum_i r_ic * k_i / sum_i r_ic` is actually a *good* representative for computing `softmax(q^T k_bar / sqrt(d))` because:
1. It minimizes the squared distance to all keys in the cluster (weighted by responsibility), which approximately preserves the dot product `q^T k_bar ≈ E[q^T k_i]`.
2. Using a single key (even the one whose *value* is closest to the centroid) introduces high variance — that key may have a very different dot product with the query than the cluster average.

**The error gap shrinks with more clusters** (from -120% to -19% at C=100 in last layer). With more clusters, each cluster is smaller and the chosen representative key is more similar to the averaged key, reducing the damage.

**Exact Both error ≈ 0** confirms the GMM partition itself is near-perfect — all the error comes from how representatives are chosen, not from the partitioning.

**Conclusion**: Averaged keys are the right choice for GMM attention. The Jensen bias from averaging keys (identified theoretically) is small compared to the variance introduced by using a single key. Future improvements should focus on the weight computation (Source 1 from Experiment 5), not on key selection.
