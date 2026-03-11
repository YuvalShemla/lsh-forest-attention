# Sampling vs. Segmentation: A Mathematical Framework for Attention Approximation

**Date**: March 9, 2026
**Authors**: Gulcelik & Shemla

### Confidence Annotations

Throughout this document, we use the following markers:

> **[SOLID]** — Mathematically rigorous, follows from standard results or verifiable algebra.
>
> **[PLAUSIBLE]** — Directionally correct and well-motivated, but relies on approximations, heuristics, or unverified empirical assumptions. Suitable for a paper with appropriate hedging.
>
> **[NEEDS WORK]** — Contains a gap, overstatement, or potential error that must be resolved before publication. May require additional experiments, tighter proofs, or reformulation.

---

## 1. Problem Setup

We approximate the attention output:

$$
\mathbf{o}^* = \sum_{i=1}^{N} w_i \, \mathbf{v}_i, \qquad w_i = \mathrm{softmax}\!\left(\frac{\mathbf{q}^\top \mathbf{k}_i}{\sqrt{d}}\right)_i
$$

where $\mathbf{q} \in \mathbb{R}^d$ is the query, $\{\mathbf{k}_i\}_{i=1}^N \subset \mathbb{R}^d$ are keys, $\{\mathbf{v}_i\}_{i=1}^N \subset \mathbb{R}^d$ are values, and $w_i \geq 0$, $\sum_i w_i = 1$.

We compare two computational models for approximating $\mathbf{o}^*$ with a budget of $B$ operations:

| Model | Description | Budget Meaning |
|-------|-------------|----------------|
| **Sampling** | Draw $B$ indices, compute IS-corrected weighted average | $B$ = number of samples |
| **Segmentation** | Partition into $C$ groups, precompute representatives, sum over groups | $C$ = number of groups |

---

## 2. Model I: Sampling-Based Approximation

> **[SOLID]** — Sections 2.1–2.3 are textbook importance sampling theory (Owen 2013, Robert & Casella 2004). All formulas are standard and verifiable.

### 2.1 General Importance Sampling

Draw $B$ i.i.d. indices $i_1, \ldots, i_B$ from a proposal distribution $q = (q_1, \ldots, q_N)$ with $q_i > 0$. The IS estimator:

$$
\hat{\mathbf{o}}_{\mathrm{IS}} = \frac{1}{B} \sum_{j=1}^{B} \frac{w_{i_j}}{q_{i_j}} \, \mathbf{v}_{i_j}
$$

**Unbiasedness**: $\mathbb{E}[\hat{\mathbf{o}}_{\mathrm{IS}}] = \mathbf{o}^*$ for any proposal $q$.

**Covariance** (trace = total variance across all $d$ dimensions):

$$
\mathrm{Tr}\!\left(\mathrm{Cov}(\hat{\mathbf{o}}_{\mathrm{IS}})\right) = \frac{1}{B}\left[\sum_{i=1}^{N} \frac{w_i^2}{q_i} \|\mathbf{v}_i\|^2 \;-\; \|\mathbf{o}^*\|^2\right]
$$

### 2.2 Optimal Proposal

Minimizing $\mathrm{Tr}(\mathrm{Cov})$ over $q$ subject to $\sum_i q_i = 1$ (via Cauchy-Schwarz):

$$
\boxed{q_i^* = \frac{w_i \|\mathbf{v}_i\|}{\sum_{j} w_j \|\mathbf{v}_j\|}}
$$

This yields the **minimal achievable variance** for IS:

$$
\boxed{V_{\mathrm{IS}}^* = \frac{1}{B}\left[\left(\sum_{i} w_i \|\mathbf{v}_i\|\right)^2 - \|\mathbf{o}^*\|^2\right]}
$$

**Interpretation via Jensen's inequality**: Since $\|\cdot\|$ is convex,

$$
\|\mathbf{o}^*\| = \left\|\sum_i w_i \mathbf{v}_i\right\| \leq \sum_i w_i \|\mathbf{v}_i\|
$$

so $V_{\mathrm{IS}}^* \geq 0$ always. Equality (zero variance) holds iff all $\mathbf{v}_i$ are collinear.

The gap $\left(\sum_i w_i \|\mathbf{v}_i\|\right)^2 - \|\mathbf{o}^*\|^2$ measures the **angular spread of values** weighted by attention — this is the irreducible cost of IS for attention approximation.

> **[SOLID]** — The optimal proposal and its variance are a direct application of Cauchy-Schwarz to the trace-variance minimization. The "angular spread" interpretation via Jensen is clean and correct. Good finding for the paper: this quantity is computable from ground-truth data, so we can measure it empirically and compare it to the actual segmentation bias.

### 2.3 Oracle Sampling (Simpler Estimator)

Our `oracle_sampling` uses $q_i = w_i$ (sample from the true attention distribution) with the simpler estimator:

$$
\hat{\mathbf{o}}_{\mathrm{oracle}} = \frac{1}{B} \sum_{j=1}^{B} \mathbf{v}_{i_j}, \qquad i_j \sim w
$$

This is unbiased with variance:

$$
V_{\mathrm{oracle}} = \frac{1}{B} \sum_{i} w_i \|\mathbf{v}_i - \mathbf{o}^*\|^2 = \frac{1}{B} \, \mathrm{Var}_w(\mathbf{V})
$$

where $\mathrm{Var}_w(\mathbf{V}) = \sum_i w_i \|\mathbf{v}_i - \mathbf{o}^*\|^2$ is the attention-weighted variance of the value vectors.

**Key quantity**: $\mathrm{Var}_w(\mathbf{V})$ determines oracle sampling quality. When values are clustered tightly around $\mathbf{o}^*$, oracle sampling has low variance. When values are spread out, variance is high.

### 2.4 Self-Normalized IS (SNIS)

> **[PLAUSIBLE]** — The asymptotic variance formula below is a first-order approximation (delta method). It's accurate when $B$ is large enough that the denominator concentrates, but for small $B$ the $O(1/B)$ bias term and higher-order variance corrections matter. For our regime ($B=100$, $N \sim 8000$), this approximation is reasonable but not exact. The statement "bias $= O(1/B)$" is standard but the constant in the $O(\cdot)$ depends on the proposal quality and can be large for poor proposals.

In practice (e.g., LSH-based methods), we don't know $Z = \sum_i e^{z_i}$. SNIS:

$$
\hat{\mathbf{o}}_{\mathrm{SNIS}} = \frac{\sum_{j=1}^{B} \frac{\tilde{w}_{i_j}}{q_{i_j}} \, \mathbf{v}_{i_j}}{\sum_{j=1}^{B} \frac{\tilde{w}_{i_j}}{q_{i_j}}}
$$

where $\tilde{w}_i = e^{z_i}$ are unnormalized. SNIS is **biased** (bias $= O(1/B)$) but **consistent**. Its asymptotic variance:

$$
V_{\mathrm{SNIS}} \approx \frac{1}{B} \sum_{i} w_i \cdot \frac{w_i}{q_i} \cdot \|\mathbf{v}_i - \mathbf{o}^*\|^2
$$

This depends on the **value deviation** $\|\mathbf{v}_i - \mathbf{o}^*\|$ (as analyzed in MagicPIG Fig. 10 and our Experiment 5).

---

## 3. Model II: Segmentation-Based Approximation

> **[SOLID]** — Sections 3.1–3.2 are basic algebra. Section 3.3 accurately describes the code.

### 3.1 General Formulation

Partition the index set $\{1, \ldots, N\}$ into $C$ disjoint groups $S_1, \ldots, S_C$. For each group $c$, precompute:
- **Cluster weight**: $W_c = \sum_{i \in S_c} w_i$
- **Representative value**: $\mathbf{r}_c \in \mathbb{R}^d$ (some function of the values in group $c$)

The segmentation estimator:

$$
\hat{\mathbf{o}}_{\mathrm{seg}} = \sum_{c=1}^{C} W_c \, \mathbf{r}_c
$$

This is **deterministic** (zero variance) given the partition.

### 3.2 Exact vs. Approximate Representatives

**Case A: Exact within-cluster weighted means** — if $\mathbf{r}_c = \boldsymbol{\mu}_c := \frac{1}{W_c} \sum_{i \in S_c} w_i \, \mathbf{v}_i$, then:

$$
\hat{\mathbf{o}}_{\mathrm{seg}} = \sum_c W_c \boldsymbol{\mu}_c = \sum_c \sum_{i \in S_c} w_i \mathbf{v}_i = \mathbf{o}^*
$$

This is **exact** (zero error). The tower property of conditional expectation guarantees this: $\mathbb{E}_w[\mathbf{V}] = \mathbb{E}_w[\mathbb{E}_w[\mathbf{V} \mid G]]$ where $G$ is the cluster indicator.

**Case B: Approximate representatives** (our GMM method) — if $\mathbf{r}_c \neq \boldsymbol{\mu}_c$, the bias is:

$$
\mathbf{b} = \hat{\mathbf{o}}_{\mathrm{seg}} - \mathbf{o}^* = \sum_{c=1}^{C} W_c (\mathbf{r}_c - \boldsymbol{\mu}_c)
$$

### 3.3 Our GMM Approach: What Happens Exactly

In `gmm_attention.py`, we do:

1. Compute responsibility-weighted average keys: $\bar{\mathbf{k}}_c = \frac{\sum_i r_{ic} \, \mathbf{k}_i}{\sum_i r_{ic}}$
2. Compute responsibility-weighted average values: $\bar{\mathbf{v}}_c = \frac{\sum_i r_{ic} \, \mathbf{v}_i}{\sum_i r_{ic}}$
3. Compute **new** softmax weights over centroids: $\hat{w}_c = \mathrm{softmax}\!\left(\frac{\mathbf{q}^\top \bar{\mathbf{k}}_c}{\sqrt{d}}\right)_c$
4. Output: $\hat{\mathbf{o}}_{\mathrm{GMM}} = \sum_c \hat{w}_c \, \bar{\mathbf{v}}_c$

This is **not** Case A above, because the weights $\hat{w}_c$ are recomputed via softmax over centroids — they are NOT equal to $W_c$. The bias has two sources:

**Source 1: Weight distortion** — $\hat{w}_c \neq W_c$ because of softmax nonlinearity (Jensen's inequality)

**Source 2: Value averaging** — $\bar{\mathbf{v}}_c \neq \boldsymbol{\mu}_c$ because GMM responsibilities $r_{ic}$ differ from attention weights $w_i / W_c$

> **[SOLID]** — This two-source decomposition is correct and important. It precisely identifies why our GMM is not a Rao-Blackwellization. Good structural insight for the paper. However, **we have not yet quantified the relative magnitude of Source 1 vs Source 2 empirically**. This would be a valuable experiment: compare GMM-with-exact-$W_c$-weights (eliminating Source 1) against GMM-with-attention-weighted-values (eliminating Source 2) to see which bias source dominates.

### 3.4 The Jensen Bias (Weight Distortion)

> **[PLAUSIBLE with caveats]** — The Jensen inequality direction and the second-order Taylor expansion are standard. However, see specific annotations below for subtleties.

The softmax function applied to averaged logits differs from the sum of softmax applied to individual logits. Define:

- Individual logit: $z_i = \mathbf{q}^\top \mathbf{k}_i / \sqrt{d}$
- Centroid logit: $\bar{z}_c = \mathbf{q}^\top \bar{\mathbf{k}}_c / \sqrt{d} = \frac{\sum_i r_{ic} \, z_i}{\sum_i r_{ic}}$ (a weighted average of logits within cluster $c$)

Since $\exp(\cdot)$ is **convex**, Jensen's inequality gives:

$$
\exp(\bar{z}_c) \leq \frac{\sum_i r_{ic} \, \exp(z_i)}{\sum_i r_{ic}}
$$

Define the within-cluster logit variance: $\sigma_c^2 = \frac{\sum_i r_{ic}(z_i - \bar{z}_c)^2}{\sum_i r_{ic}}$.

By second-order Taylor expansion:

$$
\frac{\sum_i r_{ic} \exp(z_i)}{\sum_i r_{ic}} \approx \exp(\bar{z}_c) \cdot \left(1 + \frac{\sigma_c^2}{2} + O(\sigma_c^4)\right)
$$

The relative underestimation of the centroid-based unnormalized weight is $\approx \sigma_c^2 / 2$ per cluster.

**Crucial cancellation**: Since softmax normalizes, the bias in the final weights depends on the **variation of $\sigma_c^2$ across clusters**, not on $\sigma_c^2$ itself. If all clusters have the same within-cluster logit variance, the bias largely cancels in the normalization.

> **[NEEDS WORK]** — The cancellation claim is qualitatively correct (if every cluster underestimates by the same factor, the ratio is preserved). But the degree of cancellation depends on the actual distribution of $\sigma_c^2$ across clusters AND on the interaction with the value averaging bias. We state this as if the cancellation is strong, but we haven't verified it empirically. **Experiment needed**: for each cluster, measure $\sigma_c^2$ and check how much it varies across clusters. If the variation is large, this cancellation argument is weak.

**Formal bound** (Hoeffding-type): For logit range $\Delta_c = \max_{i \in S_c} z_i - \min_{i \in S_c} z_i$ within cluster $c$:

$$
1 \leq \frac{\sum_i r_{ic} \exp(z_i)}{\exp(\bar{z}_c) \sum_i r_{ic}} \leq \exp\!\left(\frac{\Delta_c^2}{8}\right)
$$

So when $\Delta_c \ll 1$ (keys within a cluster have similar logits), the centroid approximation is accurate.

> **[SOLID]** — The Hoeffding-type bound is a correct application of Hoeffding's lemma (bounded random variable MGF bound at $t=1$). This is rigorous.

### 3.5 The Hessian of Softmax and Curvature

The softmax weights are $\mathbf{w} = \nabla A(\mathbf{z})$ where $A(\mathbf{z}) = \log \sum_i \exp(z_i)$ is the log-partition function. The Hessian:

$$
H = \nabla^2 A = \mathrm{diag}(\mathbf{w}) - \mathbf{w}\mathbf{w}^\top
$$

This is the **covariance matrix** of a categorical distribution with probabilities $w_i$. Key properties:

- **Spectral norm**: $\|H\|_2 \leq \max_i w_i$
- **Trace**: $\mathrm{tr}(H) = 1 - \|\mathbf{w}\|_2^2$ (measures attention diffuseness)
- **Weight perturbation**: $\|\Delta \mathbf{w}\|_2 \leq \max_i w_i \cdot \|\Delta \mathbf{z}\|_2$

**When attention is diffuse** (all $w_i$ small): $\|H\|_2$ is small, so logit perturbations from averaging cause small weight changes. **Segmentation works well in diffuse regimes.**

**When attention is peaked** (one $w_i \approx 1$): $\|H\|_2 \approx 1$ near the peak, but TopK handles this case well anyway.

This explains our empirical finding: GMM attention works especially well at **last layers** where attention tends to be more structured/diffuse.

> **[PLAUSIBLE]** — The Hessian properties are standard (it's the covariance of a categorical). The spectral norm bound $\|H\|_2 \leq \max_i w_i$ is correct. The **qualitative argument** (diffuse → small spectral norm → robust to logit perturbation) is sound.
>
> **[NEEDS WORK]** — The final claim "explains our empirical finding" is an **overstatement**. We're claiming that last layers have diffuse attention without having measured it. The argument also ignores Source 2 (value averaging bias) entirely — even if weights are insensitive to logit perturbation, the value representatives could still be poor. The Hessian bound only addresses weight distortion, not the full GMM error. **Needed**: measure attention entropy per layer and within-cluster logit variance per layer to verify this explanation.

---

## 4. MSE Comparison: When Does Segmentation Beat Sampling?

> **[SOLID]** — Sections 4.1–4.2 are direct consequences of the bias-variance decomposition. The crossover formula is a tautology (it just rearranges the MSE comparison). But it's a useful tautology for organizing the analysis.

### 4.1 MSE Decomposition

For any estimator $\hat{\mathbf{o}}$:

$$
\mathrm{MSE}(\hat{\mathbf{o}}) = \|\mathrm{Bias}(\hat{\mathbf{o}})\|^2 + \mathrm{Tr}(\mathrm{Cov}(\hat{\mathbf{o}}))
$$

| Method | Bias² | Variance |
|--------|-------|----------|
| Oracle Sampling ($q = w$, $B$ samples) | $0$ | $\frac{1}{B} \mathrm{Var}_w(\mathbf{V})$ |
| Optimal IS ($q^*$, $B$ samples) | $0$ | $\frac{1}{B}\left[(\sum_i w_i \|\mathbf{v}_i\|)^2 - \|\mathbf{o}^*\|^2\right]$ |
| Segmentation ($C$ clusters) | $\|\hat{\mathbf{o}}_{\mathrm{seg}} - \mathbf{o}^*\|^2$ | $0$ |

### 4.2 The Crossover Condition

Segmentation has lower MSE than oracle sampling when:

$$
\boxed{\|\hat{\mathbf{o}}_{\mathrm{seg}} - \mathbf{o}^*\|^2 < \frac{1}{B} \, \mathrm{Var}_w(\mathbf{V})}
$$

**Rearranging for the crossover budget**:

$$
B_{\mathrm{cross}} = \frac{\mathrm{Var}_w(\mathbf{V})}{\|\hat{\mathbf{o}}_{\mathrm{seg}} - \mathbf{o}^*\|^2}
$$

- For $B < B_{\mathrm{cross}}$: **segmentation wins** (fixed bias smaller than large sampling variance)
- For $B > B_{\mathrm{cross}}$: **sampling wins** (vanishing variance beats fixed bias)

### 4.3 Empirical Verification

From our Llama-3-8B experiments (50 examples, 100 queries each):

**Last layer (layer 31) — where GMM beats oracle:**

| Method | Budget/Clusters | Mean Error |
|--------|----------------|------------|
| Oracle sampling | B=100 | 0.2121 |
| GMM C=2 | C=2 | 0.2396 |
| GMM C=10 | C=10 | 0.1740 |
| GMM C=20 | C=20 | 0.1600 |
| GMM C=50 | C=50 | 0.1424 |
| **GMM C=100** | **C=100** | **0.1376** |
| GMM C=200 | C=200 | 0.1367 |
| GMM C=500 | C=500 | 0.1504 |
| GMM C=1000 | C=1000 | 0.1654 |
| Uniform sampling | B=100 | 0.3151 |
| TopK | B=100 | 0.6280 |
| Mean(V) baseline | N/A | 0.2818 |

GMM with $C=100$ achieves **35% lower error** than oracle sampling at $B=100$. This means the segmentation bias at $C=100$ is much smaller than $\sqrt{\mathrm{Var}_w(\mathbf{V})/100}$.

> **[SOLID]** — The data is real and the comparison is fair (same examples, same queries). The 35% gap is robust (averaged over 50 examples × 100 queries × 20 seeds).
>
> **[NEEDS WORK]** — The comparison between $C=100$ clusters and $B=100$ samples conflates different notions of "budget." The GMM precomputes representatives from ALL $N$ keys (an $O(N \cdot C \cdot d)$ operation), while oracle sampling only touches $B$ keys. The fair comparison should account for total FLOPs, not just the number of representatives/samples. If we count FLOPs, GMM at $C=100$ costs much more than oracle sampling at $B=100$. The comparison is meaningful for the **inference-time** use case (precompute once, serve many queries), but we should state this clearly.

**First layer (layer 0) — where oracle beats GMM:**

| Method | Budget/Clusters | Mean Error |
|--------|----------------|------------|
| Oracle sampling | B=100 | 0.3547 |
| GMM C=20 | C=20 | 0.5508 |
| GMM C=100 | C=100 | 0.6044 |
| Uniform sampling | B=100 | 0.4356 |

Here oracle wins, suggesting that at layer 0, the segmentation bias is larger than $\sqrt{\mathrm{Var}_w(\mathbf{V})/100}$ — i.e., attention structure at layer 0 is harder to capture with key-space clustering.

### 4.4 Why the Layer Difference?

> **[NEEDS WORK]** — This subsection is the weakest part of the analysis. The explanation below is a plausible narrative, but **none of the claims are verified by data**. We assert things about layer-wise attention structure, key clustering quality, and value correlations without measuring any of them. Before publishing, we need experiments that directly measure: (1) attention entropy per layer, (2) within-cluster logit variance $\sigma_c^2$ per layer, (3) key-value correlation per layer, (4) within-cluster value diameter per layer. Without these, the explanation is speculation.

The crossover condition depends on:

1. **$\mathrm{Var}_w(\mathbf{V})$** (attention-weighted value variance): If values weighted by attention are spread out, sampling has high variance.

2. **$\|\hat{\mathbf{o}}_{\mathrm{seg}} - \mathbf{o}^*\|$** (segmentation bias): Depends on within-cluster logit homogeneity and value homogeneity.

At **last layers**: attention is more structured (learned patterns), keys form tighter clusters in directions relevant to the query, and value deviations within clusters are small → **low segmentation bias, relatively high sampling variance** → segmentation wins.

At **first layers**: attention is more diffuse/random, key-value correlations are weaker, within-cluster structure is poor → **high segmentation bias** → sampling wins.

> **[PLAUSIBLE but unverified]** — An alternative hypothesis: the layer difference could be driven purely by the **effective rank of the value matrix** rather than attention structure. If last-layer values are lower rank (more compressible), clustering captures them better regardless of attention patterns. Another alternative: GMM fitting quality may differ by layer (last-layer keys may cluster more naturally than first-layer keys). These alternatives are not considered.

---

## 5. The Law of Total Variance: Connecting the Two Models

> **[SOLID]** — Sections 5.1–5.2 are textbook probability (law of total variance / Eve's law). Section 5.3's Rao-Blackwell theorem statement is standard, and the caveats about our GMM not being a true RB are correctly identified — good honest analysis.

### 5.1 Variance Decomposition

The attention-weighted value variance decomposes via the law of total variance:

$$
\underbrace{\mathrm{Var}_w(\mathbf{V})}_{\text{total}} = \underbrace{\sum_{c=1}^{C} W_c \, \mathrm{Var}_{w|c}(\mathbf{V})}_{\text{within-cluster}} + \underbrace{\sum_{c=1}^{C} W_c \|\boldsymbol{\mu}_c - \mathbf{o}^*\|^2}_{\text{between-cluster}}
$$

where $\mathrm{Var}_{w|c}(\mathbf{V}) = \sum_{i \in S_c} \frac{w_i}{W_c} \|\mathbf{v}_i - \boldsymbol{\mu}_c\|^2$ is the within-cluster weighted value variance.

### 5.2 What Each Component Means

**Between-cluster variance**: $\sum_c W_c \|\boldsymbol{\mu}_c - \mathbf{o}^*\|^2$
- This is the variance **captured** by the segmentation
- If we use exact within-cluster means as representatives, this term contributes zero bias
- The segmentation estimator handles this perfectly

**Within-cluster variance**: $\sum_c W_c \, \mathrm{Var}_{w|c}(\mathbf{V})$
- This is the variance **not captured** by the segmentation
- If we had to sample within each cluster, this would be the residual variance
- This is zero only when values within each cluster are identical

### 5.3 Rao-Blackwellization Interpretation

The partition-based approach can be viewed through the Rao-Blackwell theorem:

**Theorem (Rao-Blackwell)**: Let $T$ be an unbiased estimator of $\theta$ and $S$ a sufficient statistic. Then $T^* = \mathbb{E}[T \mid S]$ satisfies $\mathrm{Var}(T^*) \leq \mathrm{Var}(T)$, with the reduction equal to $\mathbb{E}[\mathrm{Var}(T \mid S)]$.

If we view the cluster assignment as conditioning, then:
- **Oracle sampling** = draw one sample $i \sim w$, return $\mathbf{v}_i$
- **Rao-Blackwellization** = condition on cluster assignment, return $\boldsymbol{\mu}_{G(i)}$ (the cluster mean)

The variance reduction from Rao-Blackwellization is exactly the **within-cluster variance** $\sum_c W_c \, \mathrm{Var}_{w|c}(\mathbf{V})$.

**However**, our GMM approach is **not** a Rao-Blackwellization because:
1. We recompute weights via softmax over centroids (introducing Jensen bias)
2. We use GMM responsibilities, not attention weights, for averaging

A true Rao-Blackwellization would require exact within-cluster attention sums (cost = $N$), defeating the purpose.

### 5.4 The Optimal Hybrid

> **[NEEDS WORK]** — This subsection describes an idealized estimator that **assumes we know the exact cluster weights $W_c = \sum_{i \in S_c} w_i$ and can sample from $w_i / W_c$ within each cluster**. But computing $W_c$ requires evaluating all $N$ logits (which is the full $O(N)$ computation we're trying to avoid). The hybrid is theoretically interesting as a benchmark, but it's **not practically achievable** without already solving the original problem. We should clearly label this as a "privileged oracle" analysis, not a practical algorithm. A practical version would replace exact $W_c$ with the centroid-softmax approximation, reintroducing bias.

The theoretically optimal approach combines both models:

1. **Between-cluster**: Use segmentation (deterministic, handles between-cluster variance perfectly)
2. **Within-cluster**: Use sampling (handles residual within-cluster variance)

**Two-level estimator**:

$$
\hat{\mathbf{o}}_{\mathrm{hybrid}} = \sum_{c=1}^{C} W_c \left(\frac{1}{B_c} \sum_{j=1}^{B_c} \mathbf{v}_{i_j^{(c)}}\right), \qquad i_j^{(c)} \sim w_i / W_c \text{ within cluster } c
$$

This combines:
- Zero between-cluster variance (from exact $W_c$ weights)
- Variance $= \frac{1}{B} \sum_c W_c \, \mathrm{Var}_{w|c}(\mathbf{V})$ (within-cluster only)

The total variance is:

$$
V_{\mathrm{hybrid}} = \frac{1}{B} \sum_c W_c \, \mathrm{Var}_{w|c}(\mathbf{V}) = \frac{1}{B} \left[\mathrm{Var}_w(\mathbf{V}) - \sum_c W_c \|\boldsymbol{\mu}_c - \mathbf{o}^*\|^2\right]
$$

This is always $\leq V_{\mathrm{oracle}} = \frac{1}{B} \mathrm{Var}_w(\mathbf{V})$. The improvement equals $\frac{1}{B} \sum_c W_c \|\boldsymbol{\mu}_c - \mathbf{o}^*\|^2$ (the between-cluster variance), which is exactly Cochran's stratified sampling gain.

---

## 6. Formal Conditions for Segmentation Superiority

### 6.1 Condition 1: Low Within-Cluster Logit Variance

If the within-cluster logit variance $\sigma_c^2 = \mathrm{Var}_{r_c}(z_i)$ is small for all clusters, the Jensen bias from softmax over centroids is small:

$$
\|\hat{\mathbf{o}}_{\mathrm{seg}} - \mathbf{o}^*\| \lesssim \max_c \sigma_c^2 \cdot \max_i \|\mathbf{v}_i\|
$$

**When this holds**: Keys within each cluster have similar dot products with the query. This is automatically satisfied when clustering in the **query-relevant subspace** (the direction of $\mathbf{q}$).

> **[NEEDS WORK]** — The $\lesssim$ hides important details. This bound is **not rigorously derived** — it's a heuristic combination of the second-order Taylor expansion for the Jensen gap with the max value norm. The actual error from weight distortion passes through the softmax normalization and interacts with the value representatives in a way that doesn't reduce to this simple product. A rigorous bound would need to carefully track how the per-cluster multiplicative Jensen factors $(1 + \sigma_c^2/2)$ propagate through the softmax normalization and then multiply with the value representatives. Deriving this properly is doable but needs work.
>
> Also: "clustering in the query-relevant subspace" is a **circular recommendation** — it requires knowing the query at clustering time, but we cluster the keys once and serve multiple queries.

### 6.2 Condition 2: Low Within-Cluster Value Variance

If values within each cluster are homogeneous:

$$
\max_{i, j \in S_c} \|\mathbf{v}_i - \mathbf{v}_j\| \leq \epsilon_c
$$

then the choice of representative is irrelevant up to $\epsilon_c$:

$$
\|\hat{\mathbf{o}}_{\mathrm{seg}} - \mathbf{o}^*\| \leq \max_c \epsilon_c
$$

**When this holds**: Values are correlated with keys (nearby keys → similar values). This is stronger at later layers where the model has learned structured key-value relationships.

> **[PLAUSIBLE]** — The bound $\leq \max_c \epsilon_c$ is correct algebra **for the idealized segmentation with exact weights $W_c$** (i.e., Case A from Section 3.2 where we know the true per-cluster attention mass). It does NOT apply to our GMM estimator which uses centroid-softmax weights $\hat{w}_c$. Our GMM has BOTH weight distortion AND value averaging, and this bound only addresses the latter in isolation. The claim about "later layers having stronger key-value correlations" is plausible but unverified — **experiment needed**.

### 6.3 Condition 3: High Angular Spread of Values (Hurts Sampling)

Oracle sampling variance is:

$$
V_{\mathrm{oracle}} = \frac{1}{B} \sum_i w_i \|\mathbf{v}_i - \mathbf{o}^*\|^2
$$

This is large when the **value vectors point in many different directions**. Quantitatively, if we decompose $\mathbf{v}_i = \alpha_i \hat{\mathbf{o}} + \boldsymbol{\epsilon}_i$ where $\hat{\mathbf{o}} = \mathbf{o}^*/\|\mathbf{o}^*\|$:

$$
V_{\mathrm{oracle}} = \frac{1}{B}\left[\sum_i w_i (\alpha_i - \|\mathbf{o}^*\|)^2 + \sum_i w_i \|\boldsymbol{\epsilon}_i\|^2\right]
$$

The second term (orthogonal component) is the **angular spread** — it's irreducible even by optimal IS over scalar proposals.

**Key insight**: When values have large orthogonal components relative to $\mathbf{o}^*$, sampling suffers but segmentation is unaffected (it averages out the noise deterministically).

> **[NEEDS WORK]** — The decomposition into parallel/orthogonal components is correct algebra. But the "key insight" is **overstated**. Segmentation doesn't "average out noise" — it produces a fixed biased estimate, and that bias vector can have arbitrary direction including large orthogonal components. The correct statement is weaker: the segmentation error depends on **different** quantities (within-cluster structure) than the sampling variance (angular spread of values). They are decoupled, not that one is immune to what hurts the other.

### 6.4 Condition 4: Low Budget Regime

The sampling variance scales as $1/B$ while the segmentation bias is independent of $B$. Therefore:

$$
\text{Segmentation wins when } B < \frac{\mathrm{Var}_w(\mathbf{V})}{\|\mathrm{Bias}\|^2}
$$

At **very low budgets**, segmentation almost always wins because $1/B$ is large.

### 6.5 Summary: Edge Cases

| Regime | Segmentation | Sampling | Winner |
|--------|-------------|----------|--------|
| Keys tightly clustered, values homogeneous within clusters | Very low bias | Moderate variance | **Segmentation** |
| Keys spread uniformly, no cluster structure | High bias | Moderate variance | **Sampling** |
| All values identical ($\mathbf{v}_i = \mathbf{v}$) | Zero bias | Zero variance | **Tie** (both exact) |
| One dominant key ($w_1 \approx 1$) | Low bias (if cluster captures it) | Low variance | **Tie** |
| Diffuse attention + diverse values | Moderate bias (cancellation in softmax) | **High** variance | **Segmentation** |
| Sharp attention + diverse values | Low bias (TopK-like) | Low variance | **Tie** |
| Low budget ($B \ll N$) | Fixed bias | $\propto 1/B$ (large) | **Segmentation** |
| High budget ($B \to N$) | Fixed bias | $\to 0$ | **Sampling** |

> **[PLAUSIBLE]** — The extreme cases (all values identical, high budget, low budget) are rigorously correct. The "diffuse attention + diverse values → Segmentation" row is the **central claim of interest** but is conditional on good clustering quality. It should say "Segmentation **if** clustering captures the key-value structure" rather than presenting it as unconditional. The table is useful for building intuition but should not be presented as proven results.

---

## 7. The Non-Monotonicity of GMM Error in $C$

### 7.1 Observation

In our experiments, GMM error does not monotonically decrease with $C$:

| $C$ | Last Layer Error | First Layer Error |
|-----|-----------------|-------------------|
| 1 | 0.2818 | 1.1754 |
| 2 | 0.2396 | 0.8774 |
| 10 | 0.1740 | 0.6034 |
| 20 | 0.1600 | 0.5508 |
| 50 | 0.1424 | 0.5753 |
| 100 | 0.1376 | 0.6044 |
| **200** | **0.1367** | 0.7555 |
| 500 | 0.1504 | 1.0073 |
| 1000 | 0.1654 | 1.2463 |

Error **increases** for large $C$. Why?

### 7.2 Analysis: Two Competing Effects

> **[NEEDS WORK]** — The two-effect framework is a reasonable narrative, but the "formal analysis" below has gaps. See annotations.

As $C$ increases:

**Effect 1 (helps): Better resolution** — more clusters capture finer-grained structure. The within-cluster logit variance $\sigma_c^2$ decreases, reducing Jensen bias.

**Effect 2 (hurts): Overfitting of centroids** — with many small clusters, each centroid key $\bar{\mathbf{k}}_c$ is based on fewer points. The softmax over centroids becomes more extreme (sharper distribution over more points). Small clusters may have centroid keys that are outliers in the query-relevant direction, attracting disproportionate softmax mass.

**Formal analysis**: Define the "effective temperature" of the centroid softmax. With $C$ centroids, the logit range is typically $\propto \sqrt{\log C}$ (by extreme value theory). As $C$ grows, the softmax over centroids becomes sharper, amplifying any centroid positioning errors.

> **[NEEDS WORK]** — The extreme value theory argument ($\sqrt{\log C}$ logit range) assumes the centroid logits behave like i.i.d. random variables. But centroids are **not** i.i.d. — as $C \to N$, each centroid approaches an individual key, and the logit range approaches the max logit range of the original keys (which is fixed). So the $\sqrt{\log C}$ scaling is wrong in the limit. The real issue is more likely **GMM fitting quality**: with $N \approx 8000$ and $C = 1000$, we have $\sim 8$ points per component on average, and EM with diagonal covariance in $d=128$ may not converge properly. This is a practical artifact, not a fundamental "centroid noise" effect. **Experiment needed**: check GMM log-likelihood and convergence at large $C$, and compare with k-means centroids to isolate GMM-specific issues.

The MSE of the GMM estimator decomposes as:

$$
\mathrm{MSE}_{\mathrm{GMM}}(C) = \underbrace{B_{\mathrm{Jensen}}^2(C)}_{\downarrow \text{ in } C} + \underbrace{B_{\mathrm{centroid}}^2(C)}_{\uparrow \text{ in large } C}
$$

The Jensen bias decreases with $C$ (finer resolution → lower within-cluster variance), but the centroid quality bias increases with $C$ (small clusters → noisy centroids). The optimum is at moderate $C$.

> **[PLAUSIBLE]** — The decomposition is schematic (not a formal identity — the two bias sources interact). The qualitative story is plausible but the monotonicity claims (Jensen bias $\downarrow$, centroid noise $\uparrow$) are not proven. Also note: as $C \to N$, both biases should $\to 0$ (each cluster is one point, recovering exact attention). The data shows error INCREASING from $C=200$ to $C=1000$, which contradicts this limit. This strongly suggests the issue is **GMM fitting failure at large $C$**, not a fundamental tradeoff.

### 7.3 Connection to Vector Quantization Theory

Zador's theorem states that the optimal quantization distortion of $N$ points in $\mathbb{R}^d$ using $C$ centroids scales as $C^{-2/d}$. For $d = 128$, this is $C^{-1/64}$ — an extremely shallow improvement. This means:

- Increasing $C$ from 100 to 200 improves key-space quantization by only $(200/100)^{-1/64} \approx 0.989$ (1.1% improvement)
- But the cost of noisier centroids grows faster

This explains why the optimal $C$ is moderate (100–200) rather than very large.

> **[NEEDS WORK]** — **Misapplication of Zador's theorem.** Zador's theorem is about the **asymptotic** quantization distortion for a continuous source in $\mathbb{R}^d$, assuming points fill the ambient space. It does NOT directly apply here because: (1) we have a finite set of $N$ points, not a continuous distribution; (2) the keys almost certainly lie on a low-dimensional manifold, so the effective dimension $d_{\mathrm{eff}} \ll 128$; (3) the relevant distortion for attention is not the key-space Euclidean error but rather the logit error $|q^\top(k_i - \bar{k}_c)|$, which depends on a 1D projection. Using $d=128$ dramatically overstates the curse of dimensionality. The $C^{-1/64}$ figure is misleading — the actual improvement from more clusters could be much steeper if $d_{\mathrm{eff}} \approx 20$, giving $C^{-1/10}$. **We should either measure $d_{\mathrm{eff}}$ (via PCA on keys) or remove this claim.**

---

## 8. Comparison with the Curse of Dimensionality in Numerical Integration

### 8.1 Classical Results

For approximating an integral $\int f(\mathbf{x}) \, d\mathbf{x}$ over $\mathbb{R}^d$:

**Composite rules** (partition into cells, evaluate at centroids): Error $= O(C^{-s/d})$ for $s$-smooth functions, where $C$ = number of cells.

**Monte Carlo**: Error $= O(B^{-1/2})$ regardless of $d$.

Setting $C = B$: composite rules win when $s/d > 1/2$, i.e., $d < 2s$. For $d = 128$ and $s = 2$ (generous smoothness), the crossover is at $d = 4$ — far below 128.

### 8.2 Why GMM Still Wins Despite Curse of Dimensionality

The classical analysis assumes points are scattered throughout the full $d$-dimensional space. In attention:

1. **Keys lie on a low-dimensional manifold**: Real LLM key vectors cluster in a subspace of dimension $\ll 128$. The **effective dimension** for clustering may be $d_{\mathrm{eff}} \approx 10\text{–}30$.

2. **The attention weight function has low intrinsic complexity**: $w_i = \mathrm{softmax}(\mathbf{q}^\top \mathbf{k}_i / \sqrt{d})$ depends only on the 1D projection $\mathbf{q}^\top \mathbf{k}_i$. The effective dimension for the weight function is **1**, not 128.

3. **Values are correlated with keys**: In practice, $\mathbf{v}_i$ and $\mathbf{k}_i$ are produced by the same token, creating statistical dependencies that clustering can exploit.

The GMM approach works because it clusters in the full key space but the relevant variation is low-dimensional.

> **[PLAUSIBLE — good insights, all unverified]**
>
> Point 1: The claim that keys are low-dimensional is widely believed but we haven't measured it for our Llama-3-8B data. A simple PCA analysis of the key vectors would confirm or refute this. If $d_{\mathrm{eff}}$ turns out to be high (say >50), this argument weakens significantly.
>
> Point 2: **This is the strongest insight in the section** and is mathematically correct — the attention weight is a function of a single scalar $\mathbf{q}^\top \mathbf{k}_i$. However, the GMM clusters in full key space, not in the 1D projection. The question is whether key-space clusters happen to align with the query-relevant direction. Since the GMM is fitted query-independently, it may not cluster along the direction that matters for any particular query. This is a genuine tension that deserves analysis.
>
> Point 3: Key-value correlation from same token embedding is real but after separate linear projections ($W_K$, $W_V$), the correlation depends on the learned weight matrices and could be weak. **Needs measurement.**

---

## 9. Mean Field and Renormalization Group Connections

> **[NEEDS WORK]** — This entire section presents **analogies**, not formal connections. The analogies are suggestive and may add color to a paper, but they should be clearly marked as such. None of them produce quantitative predictions or bounds.

### 9.1 Mean Field Interpretation

> **[NEEDS WORK]** — The mean field analogy is **loose**. In mean field theory, there are genuine spin-spin interactions, and the mean field replaces the effect of neighboring spins with an average. In standard attention, the query-key interactions are **independent** — there are no key-key interactions. The coupling comes only through the softmax normalization (a global operation on the logits). This is structurally different from the Ising model. The analogy is at the level of "replacing many things with their average," which is too generic to be informative. Consider either making the connection precise (e.g., via a specific free energy functional) or demoting this to a brief remark.

The attention output $\mathbf{o}^* = \sum_i w_i \mathbf{v}_i$ involves all-to-all interaction (query interacts with every key). The GMM approach replaces this with a **mean field approximation**:

$$
\hat{\mathbf{o}}_{\mathrm{MF}} = \sum_{c=1}^{C} \hat{w}_c \, \bar{\mathbf{v}}_c
$$

Instead of tracking individual key-value interactions, we replace within-cluster distributions with their means. This is exactly the mean field move in statistical physics: replace the effect of individual spins with an average field.

### 9.2 Variational Perspective

The attention distribution $p(i) = w_i$ is a Gibbs distribution with energy $E_i = -z_i = -\mathbf{q}^\top \mathbf{k}_i / \sqrt{d}$. The GMM approximation can be seen as finding a **coarse-grained** Gibbs distribution $p_{\mathrm{coarse}}(c) = \hat{w}_c$ that minimizes:

$$
\text{KL}(p_{\mathrm{coarse}} \| p_{\mathrm{lifted}}) + \text{approximation terms}
$$

The KL divergence between the fine-grained and coarse-grained distributions:

$$
\mathrm{KL}(p \| p_{\mathrm{coarse,lift}}) = \sum_c W_c \sum_{i \in S_c} \frac{w_i}{W_c} \log\!\left(\frac{w_i/W_c}{1/|S_c|}\right) = \sum_c W_c \, H_c
$$

where $H_c$ is the within-cluster relative entropy. This vanishes when weights are uniform within each cluster.

> **[NEEDS WORK]** — The KL formula is correct algebra, but the claim that GMM "minimizes" this KL is **wrong**. The GMM minimizes the log-likelihood of the keys under the mixture model (a different objective entirely). There is no reason to believe that the GMM clustering minimizes the KL between attention distributions. The "approximation terms" in the first equation are doing all the work and are undefined. This subsection should either establish a formal variational principle that the GMM actually optimizes, or be rewritten as "the KL between fine and coarse attention distributions is a useful diagnostic" without the minimization claim.

### 9.3 Block Spin Renormalization

The clustering approach is analogous to **block spin renormalization** (Kadanoff, 1966):

- **Decimation**: Replace a block of "spins" (key-value pairs) with a single effective spin (cluster representative)
- **Renormalization**: The effective coupling constants (centroid logits) are derived from the original couplings

The approximation is exact when within-block degrees of freedom are "frozen" (within-cluster value homogeneity). The error of the block spin approximation is controlled by the within-block fluctuations — exactly our within-cluster variance.

> **[PLAUSIBLE]** — The block spin analogy is better than the mean field one because it directly maps to the "replace a group with its representative" operation. But block spin RG preserves universality and critical exponents, which have no clear counterpart in the attention setting. Keep this as a brief analogy if it helps the reader, but don't oversell it as a formal connection.

---

## 10. Towards Optimal Algorithm Selection

### 10.1 Practical Decision Rule

> **[NEEDS WORK]** — **This decision rule is circular.** Step 1 requires computing attention entropy, which requires computing all $N$ attention weights (the full $O(N)$ computation). Step 2 requires computing $\mathrm{Var}_{\mathrm{between}} / \mathrm{Var}_{\mathrm{total}}$, which also requires exact attention weights. If we had these, we wouldn't need an approximation. A practical decision rule must use **cheap proxies** (e.g., key norm statistics, query-key angle distribution from a subsample). The rule below is useful as a theoretical framework for understanding when each method works, but should not be presented as a practical algorithm.

Given a query $\mathbf{q}$, keys $\mathbf{K}$, values $\mathbf{V}$, and budget $B$:

**Step 1**: Estimate the attention entropy $H(w) = -\sum_i w_i \log w_i$:
- **Low entropy** ($H < \log B$): Attention is concentrated → TopK is best
- **High entropy** ($H \approx \log N$): Attention is diffuse → continue to Step 2

**Step 2**: Estimate the key-value structural correlation:
- Compute the ratio $\rho = \mathrm{Var}_{\mathrm{between}} / \mathrm{Var}_{\mathrm{total}}$ (proportion of variance explained by clustering)
- **High $\rho$** (good cluster structure): Segmentation wins
- **Low $\rho$** (poor cluster structure): Sampling wins

**Step 3**: If segmentation, choose optimal $C$:
- $C^* \approx \arg\min_{C} \left[\text{Jensen bias}(C) + \text{centroid noise}(C)\right]$
- In practice, $C^* \approx 50\text{–}200$ for our Llama-3-8B data

### 10.2 Adaptive Hybrid: The Best of Both Worlds

The optimal approach is a **two-level estimator**:

1. Cluster keys into $C$ groups (precomputation)
2. For the top-$K$ clusters by weight $W_c$: compute **exact** within-cluster attention
3. For remaining clusters: use centroid approximation
4. Sum up

This gives:
- Zero bias for the high-weight clusters (where accuracy matters most)
- Low-bias centroid approximation for low-weight clusters (where errors have little impact)
- Total cost: $K \cdot \bar{n}_c + C$ where $\bar{n}_c$ is average cluster size for top-$K$ clusters

---

## 11. Key Theoretical Results (Summary)

### Theorem 1: Sampling Variance Lower Bound

> **[SOLID]** — Standard IS optimization via Cauchy-Schwarz. Publishable as stated.

For any proposal distribution $q$, the IS estimator variance satisfies:

$$
\mathrm{Tr}(\mathrm{Cov}(\hat{\mathbf{o}}_{\mathrm{IS}})) \geq \frac{1}{B} \left[\left(\sum_i w_i \|\mathbf{v}_i\|\right)^2 - \|\mathbf{o}^*\|^2\right]
$$

with equality achieved by $q_i^* \propto w_i \|\mathbf{v}_i\|$. The lower bound is positive unless all values are collinear.

### Theorem 2: Segmentation Bias Bound

> **[NEEDS WORK]** — **This is not rigorously proven.** The additive decomposition of Jensen bias and value averaging bias is heuristic. The two bias sources interact through the softmax normalization in a way that doesn't cleanly separate into an additive bound. Specifically: (a) the Jensen bias bound $\Delta_c^2/8$ applies to the unnormalized exponentials, but the softmax normalization creates coupling between clusters; (b) the value averaging bound $\epsilon_c$ assumes exact weights $W_c$, but our estimator uses distorted weights $\hat{w}_c$, so the two errors multiply rather than add. **To make this rigorous**, one would need to write $\hat{o}_{\mathrm{seg}} - o^* = \sum_c (\hat{w}_c \bar{v}_c - W_c \mu_c) = \sum_c (\hat{w}_c - W_c)\bar{v}_c + \sum_c W_c(\bar{v}_c - \mu_c) + \sum_c (\hat{w}_c - W_c)(\bar{v}_c - \mu_c)$ and bound each term. The cross term is missing from our analysis. Do NOT present as a theorem without a proper proof.

For a partition with within-cluster logit range $\Delta_c$ and within-cluster value diameter $\epsilon_c$:

$$
\|\hat{\mathbf{o}}_{\mathrm{seg}} - \mathbf{o}^*\| \leq \underbrace{\max_c \frac{\Delta_c^2}{8} \cdot \max_i \|\mathbf{v}_i\|}_{\text{Jensen bias}} + \underbrace{\max_c \epsilon_c}_{\text{value averaging bias}}
$$

### Theorem 3: Crossover Budget

> **[SOLID in structure, inherits looseness from Theorem 2]** — The crossover formula itself is just rearranging $\text{Bias}^2 < \text{Variance}/B$, which is trivially correct. But if the bias bound from Theorem 2 is loose, the crossover budget estimate will be conservative (predicting segmentation wins in a wider range than it actually does).

The segmentation estimator has lower MSE than the optimal IS estimator when:

$$
B < B_{\mathrm{cross}} = \frac{(\sum_i w_i \|\mathbf{v}_i\|)^2 - \|\mathbf{o}^*\|^2}{\|\hat{\mathbf{o}}_{\mathrm{seg}} - \mathbf{o}^*\|^2}
$$

### Theorem 4: Variance Decomposition (Cochran-Rao-Blackwell)

> **[SOLID]** — This is the law of total variance (Eve's law), a standard probability result. Publishable as stated.

The oracle sampling variance decomposes as:

$$
\mathrm{Var}_w(\mathbf{V}) = \underbrace{\sum_c W_c \, \mathrm{Var}_{w|c}(\mathbf{V})}_{\text{residual (within-cluster)}} + \underbrace{\sum_c W_c \|\boldsymbol{\mu}_c - \mathbf{o}^*\|^2}_{\text{captured by segmentation}}
$$

A stratified sampling estimator eliminates the between-cluster term, achieving:

$$
V_{\mathrm{stratified}} = \frac{1}{B} \sum_c W_c \, \mathrm{Var}_{w|c}(\mathbf{V}) \leq V_{\mathrm{oracle}}
$$

### Theorem 5: Softmax Curvature Bound

> **[NEEDS WORK]** — **The bound as stated is incorrect.** The quantity $(1 - \|\mathbf{w}\|_2^2)^{1/2}$ is $\sqrt{\mathrm{tr}(H)}$, which bounds the **Frobenius norm** $\|H\|_F$, not the **spectral norm** $\|H\|_2$. For the spectral norm, the correct bound is $\|\Delta \mathbf{w}\|_2 \leq \|H\|_2 \cdot \|\Delta \mathbf{z}\|_2 \leq \max_i w_i \cdot \|\Delta \mathbf{z}\|_2$. The document itself notes the correction in the last line ("the effective sensitivity is $\max_i w_i$"), contradicting the theorem statement. **Fix**: replace the theorem with the spectral norm bound $\max_i w_i$, and note that this is $\leq 1/4$ in the maximally diffuse binary case (two equal weights), and $\leq 1/N$ for uniform attention.

The sensitivity of attention weights to logit perturbation is bounded by:

$$
\|\Delta \mathbf{w}\|_2 \leq (1 - \|\mathbf{w}\|_2^2)^{1/2} \cdot \|\Delta \mathbf{z}\|_2
$$

In diffuse regimes where $\|\mathbf{w}\|_2^2 \approx 1/N$, the bound is nearly $\|\Delta \mathbf{z}\|_2$, but this is a loose bound; the effective sensitivity is $\max_i w_i \cdot \|\Delta \mathbf{z}\|_2$.

---

## 12. Open Questions and Future Directions

1. **Can we compute the crossover budget $B_{\mathrm{cross}}$ cheaply at inference time?** This would allow adaptive selection between sampling and segmentation per query. *[Promising but hard — requires cheap proxies for attention entropy and clustering quality.]*

2. **What is the optimal clustering criterion for attention?** Clustering in key space minimizes logit variance, but clustering in value space minimizes value averaging bias. The optimal criterion should minimize a combination, potentially weighted by attention. *[This is a well-posed optimization problem that could yield a clean theoretical result.]*

3. **Is there a connection to the James-Stein phenomenon?** In $d \geq 3$ dimensions, biased shrinkage estimators dominate unbiased MLE for the normal mean. With $d = 128$, the potential for bias to help is large. *[Intriguing analogy but unclear if it leads anywhere concrete for attention — the James-Stein result is about estimating a fixed parameter, not a weighted sum.]*

4. **Can the GMM responsibilities be precomputed offline and amortized across queries?** The current approach fits GMM per example but uses it for all queries. Making it query-dependent could improve quality. *[We already do this — GMM is fitted once per example. The real question is whether query-dependent soft assignments would help, at the cost of per-query GMM inference.]*

5. **Hybrid architecture**: Train the Transformer with a differentiable two-level attention: segmentation for coarse structure + sampling for fine-grained refinement. *[Ambitious but speculative — no evidence yet that this is feasible or beneficial.]*

---

## 13. Summary: What We Can Confidently Claim vs. What Needs More Work

### Strong results (paper-ready)

- **Theorem 1** (optimal IS variance) and **Theorem 4** (variance decomposition): textbook results, correctly applied to the attention setting. The angular-spread interpretation and the within/between cluster decomposition are clean contributions.
- **The crossover framework** (Section 4.2): bias² vs variance/B is the right way to think about sampling vs. segmentation. The crossover budget formula is a useful organizing principle.
- **The two-source bias decomposition** (Section 3.3): weight distortion + value averaging. This correctly identifies the structure of the GMM approximation error and is the right starting point for rigorous bounds.
- **The empirical data**: GMM beats oracle at last layer by 35%, loses at first layer. This is robust (large sample, many seeds). The phenomenon is real and interesting.

### Plausible claims (include with hedging)

- **Diffuse attention helps segmentation** (Section 3.5): The Hessian argument is directionally correct. Present as "we expect" or "this suggests," not "this proves."
- **Low within-cluster logit variance → small bias** (Section 3.4): The second-order Taylor analysis and Hoeffding bound are correct locally, but the propagation through softmax normalization is hand-wavy. Present the individual bounds rigorously and the combined picture as motivation.
- **The edge case table** (Section 6.5): Useful for intuition. Present as "expected behavior" not "proven."
- **Block spin / stratified sampling analogies** (Sections 5, 9.3): Helpful for readers with statistics/physics background. Clearly label as analogies.

### Claims that need fixing before publication

- **Theorem 2** (segmentation bias bound): Not proven. The additive decomposition ignores the cross-term. Needs a proper derivation or should be downgraded to a conjecture.
- **Theorem 5** (softmax curvature): The stated bound is wrong (Frobenius vs spectral norm confusion). Fix to use $\max_i w_i$.
- **Section 7.3** (Zador's theorem): Misapplied at $d=128$. Either measure effective dimensionality or remove.
- **Section 9.2** (variational perspective): Claims GMM minimizes a KL it doesn't actually minimize. Rewrite.
- **Section 10.1** (practical decision rule): Circular — requires the answer to compute the decision criterion.

### Experiments needed to strengthen the analysis

1. **Per-layer diagnostics**: attention entropy, within-cluster logit variance $\sigma_c^2$, key-value correlation, value matrix effective rank — for both layers. This would turn Section 4.4 from speculation into evidence.
2. **Bias source ablation**: compare (a) GMM with exact $W_c$ weights (eliminates Source 1), (b) GMM with attention-weighted value averages (eliminates Source 2), to quantify which source dominates.
3. **PCA on key vectors**: measure effective dimensionality to resolve the Zador's theorem question and the "keys lie on a low-dimensional manifold" claim.
4. **GMM convergence at large $C$**: check log-likelihood, compare with k-means, to determine whether the non-monotonicity at $C > 200$ is a fundamental tradeoff or a GMM fitting artifact.
5. **$B_{\mathrm{cross}}$ empirical measurement**: for each query, compute both the oracle sampling variance (from multiple runs) and the GMM bias (from the single deterministic output), and verify the crossover formula.

---

## References

1. Cochran, W.G. (1977). *Sampling Techniques*, 3rd ed. Wiley. — Stratified sampling theory.
2. Owen, A.B. (2013). *Monte Carlo theory, methods and examples*. — IS, SNIS, Rao-Blackwellization.
3. Robert, C.P. & Casella, G. (2004). *Monte Carlo Statistical Methods*, 2nd ed. Springer. — Optimal IS proposals.
4. Gersho, A. & Gray, R.M. (1992). *Vector Quantization and Signal Compression*. — Zador's theorem, quantization theory.
5. Wainwright, M.J. & Jordan, M.I. (2008). *Graphical Models, Exponential Families, and Variational Inference*. — Exponential families, log-partition.
6. Chen, T. et al. (2024). "MagicPIG: LSH Sampling for Efficient LLM Generation." arXiv:2410.16179. — SNIS for attention, value deviation analysis.
7. Vyas, A. et al. (2020). "Fast Transformers with Clustered Attention." NeurIPS. — Clustered query attention.
8. Kitaev, N. et al. (2020). "Reformer: The Efficient Transformer." ICLR. — LSH-based sparse attention.
9. Choromanski, K. et al. (2021). "Rethinking Attention with Performers." ICLR. — Random feature attention, kernel bounds.
10. Wang, S. et al. (2020). "Linformer: Self-Attention with Linear Complexity." arXiv:2006.04768. — Low-rank attention.
11. Xiong, Y. et al. (2022). "Nystromformer." AAAI. — Landmark-based attention.
12. Kadanoff, L.P. (1966). "Scaling Laws for Ising Models near $T_c$." Physics. — Block spin renormalization.
13. Zador, P.L. (1982). "Asymptotic Quantization Error of Continuous Signals." IEEE Trans. IT. — Quantization theory.
14. Caflisch, R.E. (1998). "Monte Carlo and Quasi-Monte Carlo Methods." Acta Numerica. — MC vs deterministic integration.

---

*Generated March 9, 2026.*
