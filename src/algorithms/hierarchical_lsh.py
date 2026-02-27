"""
Hierarchical LSH Attention - Tree-Aggregation approach.

Partitions ALL keys into groups using the LSH tree hierarchy (by LCP depth),
then approximates each group's contribution using its average key and average value.
Covers all keys (no missing mass) using only O(K*L) representative computations.
"""

import numpy as np
from typing import Tuple
from algorithms.base import softmax


def hierarchical_lsh_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    logits: np.ndarray,
    head_dim: int,
    key_codes: np.ndarray,
    query_hash: np.ndarray,
    K: int,
    L: int = None
) -> Tuple[np.ndarray, int]:
    """
    Hierarchical LSH attention via tree-aggregation.

    Groups keys by their LCP (longest common prefix) with the query hash,
    computes average key/value per group, applies count-weighted softmax
    over group representatives. Handles both single tree (L=1) and multiple
    trees (L>1) with output averaging.

    Args:
        query: [head_dim] query vector
        keys: [num_keys, head_dim] key vectors
        values: [num_keys, head_dim] value vectors
        logits: [num_keys] precomputed q·k/√d scores (unused, kept for interface)
        head_dim: dimension
        key_codes: [num_keys, num_tables, max_depth] binary hash codes
        query_hash: [num_tables, max_depth] query hash codes
        K: tree depth (number of hash bits to use per tree)
        L: number of trees to use (default: all available)

    Returns:
        output: [head_dim] approximate attention output
        effective_budget: total non-empty groups across all trees
    """
    nv = len(keys)
    if nv == 0:
        return np.zeros(head_dim), 0

    num_tables = key_codes.shape[1]
    if L is None:
        L = num_tables
    L = min(L, num_tables)

    # Compute LCP: vectorized across all keys and trees
    # matches: [nv, L, K] - does bit d match?
    matches = (key_codes[:nv, :L, :K] == query_hash[:L, :K])
    # cum_match: cumulative product along depth axis - breaks at first mismatch
    cum_match = np.cumprod(matches, axis=2)
    # lcp: [nv, L] - longest common prefix length per tree
    lcp = np.sum(cum_match, axis=2).astype(np.int32)

    tree_outputs = []
    total_groups = 0
    sqrt_d = np.sqrt(head_dim)

    for l in range(L):
        lcp_l = lcp[:, l]  # [nv] - LCP for this tree
        clamped = np.minimum(lcp_l, K)

        # Collect groups: d=0..K-1 are "sibling" groups, d=K is leaf bucket
        group_avg_keys = []
        group_avg_values = []
        group_counts = []

        for d in range(K + 1):
            if d < K:
                mask = (clamped == d)  # Keys with LCP exactly d
            else:
                mask = (clamped >= K)  # Keys with LCP >= K (leaf)

            count = np.sum(mask)
            if count == 0:
                continue

            avg_key = np.mean(keys[mask], axis=0)
            avg_value = np.mean(values[mask], axis=0)
            group_avg_keys.append(avg_key)
            group_avg_values.append(avg_value)
            group_counts.append(count)

        n_groups = len(group_counts)
        if n_groups == 0:
            continue

        total_groups += n_groups

        # Count-weighted softmax: softmax(q·avg_k/√d + log(count))
        avg_keys_arr = np.array(group_avg_keys)
        avg_vals_arr = np.array(group_avg_values)
        counts_arr = np.array(group_counts, dtype=np.float64)

        scores = (avg_keys_arr @ query) / sqrt_d + np.log(counts_arr)
        weights = softmax(scores)
        tree_out = weights @ avg_vals_arr

        tree_outputs.append(tree_out)

    if len(tree_outputs) == 0:
        return np.zeros(head_dim), 0

    # Average outputs across trees (or just return if L=1)
    output = np.mean(tree_outputs, axis=0) if len(tree_outputs) > 1 else tree_outputs[0]
    return output, total_groups
