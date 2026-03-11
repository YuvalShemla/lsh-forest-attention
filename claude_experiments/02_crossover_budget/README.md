# Experiment 2: Crossover Budget Verification

## Goal

Test whether the theoretical crossover formula **B_cross = Var_w(V) / ||seg_bias||^2** predicts when GMM segmentation beats oracle sampling.

## Theory

- **Oracle sampling** has MSE approximately `Var_w(V) / B`, where `Var_w(V) = sum_i w_i * ||v_i - o*||^2`
- **GMM segmentation** has a fixed bias `||o_seg - o*||^2` independent of budget
- **Crossover**: segmentation wins when `B > B_cross = Var_w(V) / seg_bias^2`

## Configuration

| Parameter | Value |
|-----------|-------|
| NUM_EXAMPLES | 10 (test), 100 (production) |
| NUM_QUERIES_PER_EXAMPLE | 50 |
| ORACLE_TRIALS | 20 |
| BUDGETS | [10, 20, 50, 100, 200, 500] |
| GMM_CLUSTERS | 50 |
| HEAD_DIM | 128, SEED = 42 |

## Running

```bash
cd claude_experiments/02_crossover_budget
python experiment.py
```

## Results (10 examples, 500 queries per layer)

### B_cross Distribution

| Statistic | First Layer | Last Layer |
|-----------|------------|------------|
| Median B_cross | **55** | **340** |
| Mean B_cross | 60 +/- 32 | 452 +/- 351 |
| Range | [15, 190] | [35, 1842] |
| IQR | [35, 74] | [196, 612] |

### Prediction Accuracy

| Metric | First Layer | Last Layer |
|--------|------------|------------|
| Log correlation (pred vs emp) | **0.91** | **0.91** |
| Within 2x | 92.6% | 86.1% |
| Within 5x | 100% | 100% |
| Median pred/emp ratio | 0.67 | 0.70 |

### Fraction Where GMM Wins by Budget

| Budget | First Layer | Last Layer |
|--------|------------|------------|
| B=10 | **100%** | **100%** |
| B=20 | 94.2% | **100%** |
| B=50 | 57.4% | **99.4%** |
| B=100 | 10.6% | 92.2% |
| B=200 | 0% | 73.0% |
| B=500 | 0% | 32.6% |

### GMM Error

| Layer | Mean | Median |
|-------|------|--------|
| First | 0.543 | 0.469 |
| Last | **0.119** | **0.110** |

## Discussion

**The B_cross formula is highly predictive.** Log-space correlation of 0.91 between predicted and empirical crossover in both layers. 92-100% of predictions are within 5x of the true crossover.

**Key insight: the layers differ by ~6x in crossover budget.** First layer B_cross median is 55 — meaning oracle sampling beats GMM beyond about 50 keys. Last layer B_cross median is 340 — GMM wins up to budget ~340, which covers most practical budgets.

**Why the difference?** The driving factor is Var_w(V):
- First layer: Var_w(V) mean = 0.096 (low value variance)
- Last layer: Var_w(V) mean = 33.4 (**350x larger**)

This means sampling at last layer has enormously higher variance, so even with a larger absolute GMM bias (0.135 vs 0.002), the ratio B_cross = Var/bias² is much higher.

**Systematic underprediction**: The median pred/emp ratio is ~0.7 (B_cross predicts 30% too low). This is expected because: (1) the formula assumes MSE scaling as 1/B which is approximate, and (2) oracle sampling's simple-average estimator is slightly biased at small B due to replacement effects.

**Practical implication**: At typical inference budgets (50-200 keys), GMM wins almost always at the last layer but rarely at the first layer. This supports a layer-adaptive strategy.
