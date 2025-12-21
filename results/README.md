# 📊 Results

This folder contains generated visualizations and experimental results.

## Attention Data Exploration

### Aggregate Visualizations (Overall Patterns)

- **`attention_data_exploration_first_layer.png`** - Comprehensive analysis of Layer 0 (first layer)
- **`attention_data_exploration_last_layer.png`** - Comprehensive analysis of Layer 31 (last layer)

Each visualization contains 5 plots:
1. Attention weight distribution for last 5 queries (combined)
2. Top-K concentration across positions
3. Query-Key similarity distribution
4. Attention entropy across sequence
5. Key and value vector norms

### Individual Query Dashboards (Per-Query Analysis)

**10 detailed dashboards** (5 queries × 2 layers):

**First Layer (Layer 0):**
- `query_dashboard_first_layer_pos6129.png` through `pos6133.png`

**Last Layer (Layer 31):**
- `query_dashboard_last_layer_pos6129.png` through `pos6133.png`

Each individual dashboard shows **for a single query**:
1. Attention weight distribution (that query → all its keys)
2. Top-K concentration bar chart (how much mass top-K captures)
3. Q-K similarity distribution (histogram)
4. Attention vs Similarity scatter plot (does similarity predict attention?)
5. Statistics summary panel

These dashboards enable **deep-dive per-query analysis** to understand:
- How concentrated is this specific query's attention?
- Which keys does this query attend to most?
- Is attention correlated with Q-K similarity for this query?

### How to Generate

From the `experiments/` directory:

```bash
cd experiments

# Generate aggregate visualizations (overall patterns)
python3 explore_attention_data.py --layer first_layer
python3 explore_attention_data.py --layer last_layer

# Generate individual query dashboards (per-query detail)
python3 explore_individual_queries.py --layer first_layer --num-queries 5
python3 explore_individual_queries.py --layer last_layer --num-queries 5
```

### Key Findings

See documentation for detailed analysis:
- **Aggregate patterns**: `docs/EXPLORATION_FINDINGS.md`
- **Individual query dashboards**: `docs/INDIVIDUAL_QUERY_DASHBOARDS.md`

## Future Results

This folder will also contain:
- Jungle vs MagicPIG comparison plots
- Parameter sweep results
- Performance benchmarks
- Error distribution analyses

