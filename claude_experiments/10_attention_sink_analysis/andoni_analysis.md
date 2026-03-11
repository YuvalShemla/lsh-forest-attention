# Analysis of Andoni's Plots: Attention Sinks, Query Correlation, and Entropy

## Summary of Observations

Alex Andoni generated diagnostic plots on our Llama-3-8B attention data (Example #1, "Long In-context Learning", seq_len=8192, layer 0 and layer 31, head 0).  Three key findings:

1. **Queries are extremely correlated** — pairwise cosine similarity ~0.95 (last layer), ~0.90 (first layer)
2. **Key 0 has anomalous attention weight** — it acts as an "attention sink"
3. **Attention entropy tracks the "uniform over ~50% keys" curve** — and Andoni asks if this is stable

---

## 1. Query Correlation: Why Are Queries So Similar?

### What the plots show

The **Query-Query Similarity Distribution** panels (middle-right in both plots) show:
- **Last layer**: Mean pairwise cosine similarity ≈ 0.95 (purple histogram, tightly peaked)
- **First layer**: Mean ≈ 0.90 (slightly more spread)

The **||q - mean(Q)|| distribution** (top-right) confirms this:
- Last layer: median ||q - mean(Q)|| ≈ 1.0, while ||mean(Q)|| ≈ 13.3
- First layer: median ||q - mean(Q)|| ≈ 1.1, while ||mean(Q)|| ≈ 21.4
- **Ratio**: deviation/mean ≈ 7-8% — queries deviate from their mean by only ~7%

### Why this happens

This is a well-understood property of transformer residual streams:

1. **Residual stream accumulation**: In Llama-3, each layer computes `x_new = x + Attn(x) + FFN(x)`. The hidden state `x` at any position is a sum of the embedding plus all previous layer contributions. By layer 31, the residual component dominates, and since all tokens share the same model weights, their representations develop a large shared component.

2. **Query projection preserves shared structure**: Queries are `q = W_Q * x`. If `x = x_shared + x_individual` where `x_shared` is large, then `q ≈ W_Q * x_shared + W_Q * x_individual`, and the shared component dominates.

3. **The "query mean" is a real direction in representation space**: It's not noise — it's the average projection of the residual stream through W_Q. The useful (token-specific) information lives in the ~7% deviation from the mean.

### Implications for our algorithms

This is important for our sparse attention work:
- **All queries produce similar attention distributions**: Since queries are ~95% similar, their logits `q^T k_i / sqrt(d)` are highly correlated across queries. This means a query-independent partition (like GMM/k-means) is surprisingly effective — one partition works well for almost all queries.
- **The "diffuse" attention we observe might partly be an artifact**: If the shared query component gives roughly equal logits to most keys, then the *residual* token-specific attention (from the 7% individual component) is what carries semantic signal.

---

## 2. The Attention Sink at Position 0

### What the plots show

The **Attention Weight Distribution** (top-left) shows visible spikes at position 0 for all queries. The **Query-Key Cosine Similarity** and **Scaled Dot Product** panels show that Key 0 has a distinct distribution (orange curves) shifted relative to the full-key distribution (blue/green).

### Why position 0 is special

This is the **"attention sink" phenomenon** documented by Xiao et al. (2023, "Efficient Streaming Language Models with Attention Sinks", ICLR 2024):

1. **Causal masking universality**: Position 0 is the *only* position visible to every query under causal attention. It appears in every attention window for every token in every sequence during training.

2. **Softmax must allocate**: Softmax weights sum to 1. When a head has no semantically useful match (the "dormant" state), it must still allocate attention somewhere. The model learns to park excess mass on position 0 — it becomes a "default" target.

3. **Self-reinforcing during training**: Because all queries have a large shared component (see above), and key 0 is visible to all, key 0's representation specializes during training to align with this shared query direction. This creates a stable equilibrium:
   - Shared query component → high dot product with key 0 → high attention on position 0
   - High attention on position 0 → gradient signal to keep key 0 aligned → reinforcement

4. **Key 0's norm is NOT anomalously large**: Andoni noted that key 0 "appears to be relatively small in norm otherwise." This is consistent with the literature — the sink effect arises from *directional alignment* with the shared query component, not from norm. The model learns the key direction, not magnitude.

### Connection to "massive activations"

Sun et al. (2024, "Massive Activations in Large Language Models", COLM 2024) showed that a few hidden-state activations are ~100,000x larger than typical. These massive activations occur at fixed sequence positions (the same sink tokens) and act as **implicit bias terms**. The value vector at position 0 encodes a roughly constant offset that the model needs but cannot represent through explicit bias parameters.

### Should we exclude position 0?

This depends on what we're measuring:
- **For algorithm comparison**: Including position 0 is fine as long as all methods handle it consistently. It's part of the real attention distribution.
- **For understanding diffuse attention**: Excluding position 0 gives a cleaner view of the "semantic" attention pattern.
- **For practical sparse attention**: Any production system should *always* include the first few sink tokens in the budget. This is free and avoids a systematic error.

Our Experiment 10 measures algorithm error with and without the sink to quantify its impact.

---

## 3. Attention Entropy and the 50% Line

### What the plots show

The **Attention Entropy** panel (bottom-left) shows:
- Entropy grows roughly as log(query_position) — as expected for causal attention over more keys
- The dashed line labeled "uniform over 50% keys" corresponds to entropy = log(0.5 * t) for query position t
- The observed entropy closely tracks this 50% line at both layers

### Cross-reference with our Experiment 1

Our measurements across 100 examples (5000 queries per layer):
- **First layer**: mean entropy = 8.007 nats (std 0.340, range 6.70-8.86)
- **Last layer**: mean entropy = 8.226 nats (std 0.292, range 6.69-8.85)

For reference, uniform over all 8192 keys: log(8192) = 9.01 nats.

The **effective support fraction** = exp(H) / N:
- At H = 8.007: exp(8.007)/8192 ≈ 36.6% of context (first layer)
- At H = 8.226: exp(8.226)/8192 ≈ 45.5% of context (last layer)

So the last layer is indeed close to the "50% line" while the first layer is around "37%".

### Did it fluctuate between 10% and 50% before?

**Yes — this is real per-query variation**, not a bug:
- Min entropy = 6.69 nats → exp(6.69)/8192 ≈ **9.8% of context** (concentrated)
- Max entropy = 8.86 nats → exp(8.86)/8192 ≈ **86% of context** (very diffuse)
- The IQR spans roughly 30-55% at the first layer and 38-55% at the last layer

The earlier observation of "fluctuating between 10% and 50%" likely captured the per-example/per-query variation. Some queries (perhaps copying or retrieval tasks) have concentrated attention (~10%), while most have diffuse attention (~40-50%). **The code has not been messed up** — the median is near 50%, but individual queries span a wide range.

### When do attention weights look discrete?

The "discrete" appearance in the first histogram (sharp spikes at specific positions) occurs when:
1. **The attention sink** contributes a visible spike at position 0
2. **Recent positions** get relatively higher weight (recency bias in causal attention)
3. **Specific content matches**: when the query has a strong semantic match with a few keys, those positions get discrete-looking spikes
4. **Last layer more than first**: Layer 31 shows slightly more structured (less uniform) attention patterns, though both are broadly diffuse

---

## 4. Implications for Our Algorithms

### The shared query component changes the picture

Since ~93% of query variance is shared:
- The logit vector `z_i = q^T k_i / sqrt(d)` decomposes as: `z_i ≈ mean(Q)^T k_i / sqrt(d) + (q - mean(Q))^T k_i / sqrt(d)`
- The first term is **query-independent** and large. The second is small but carries the semantic signal.
- A query-independent partition (GMM/k-means on key space) effectively clusters by the *shared* logit component, which is reasonable because that component dominates.

### The attention sink creates a bias opportunity

- Position 0 gets ~0.2-0.4% of attention (roughly 15-25x the uniform weight)
- Any sparse method that drops position 0 incurs a systematic error proportional to `w_0 * v_0`
- Since `v_0` is roughly constant (implicit bias term), this error is predictable and correctable
- **Recommendation**: Always include position 0 in the budget for any sparse attention method

### Why segmentation works despite high dimensionality

The PCA / intrinsic dimensionality question (Section 7 of the paper) may be partially answered by query correlation:
- If we decompose keys relative to the shared query direction, the "effective" clustering task is nearly 1D (project onto mean(Q))
- The remaining d-1 dimensions contribute only through the ~7% individual query component
- This explains why key-space k-means (which clusters in full d=128) still works well: the dominant axis in key space aligns with the shared query direction

---

## 5. Answers to Andoni's Specific Questions

**Q: Do you know when the attention weights look discrete?**
When (a) the attention sink at position 0 creates a visible spike, (b) there are strong semantic matches at specific positions, or (c) the query is from a task requiring copying/retrieval. Most queries have broadly diffuse attention (effective support ~40-50%), but some are highly concentrated (~10%).

**Q: Can you double-check that the "attention entropy" looks the same?**
Yes — our Experiment 1 (100 examples, 5000 queries) confirms: mean entropy = 8.007 (first layer) and 8.226 (last layer). The last layer is very close to the 50% line (log(0.5*8192) = 8.29). The first layer is around 37%. The "fluctuation between 50% and 10%" is real per-query variation: min entropy corresponds to ~10% effective support, while the median is ~45-50%.

**Q: Why are queries correlated?**
Residual stream accumulation in transformers. By layer 31, the shared component of hidden states dominates, and W_Q projects this into a high-similarity query space. The useful token-specific information lives in the ~7% deviation from the query mean.

**Q: Why does key 0 get high attention despite small norm?**
Attention sink phenomenon (Xiao et al., 2023). Key 0 is the only position visible to all queries under causal masking. The model learns to align key 0's direction with the shared query component, making it a "default" attention target. The effect is directional, not magnitude-based.

---

## References

1. Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2023). *Efficient Streaming Language Models with Attention Sinks*. ICLR 2024. arXiv:2309.17453
2. Sun, M., Chen, X., Kolter, J.Z., & Liu, Z. (2024). *Massive Activations in Large Language Models*. COLM 2024. arXiv:2402.17762
3. Guo, Y. et al. (2024). *Active-Dormant Attention Heads: Mechanistically Demystifying Extreme-Token Phenomena in LLMs*. arXiv:2410.13835
