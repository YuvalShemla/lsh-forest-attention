"""
Jungle Sampling - LSH forest prefix_sampling with SNIS correction.

Our main contribution: uses LSH forest hierarchy to define a
depth-mixture proposal corrected via self-normalized importance sampling.
"""

import numpy as np
from typing import Tuple
from algorithms.base import snis_estimator


def jungle_sampling(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    logits: np.ndarray,
    head_dim: int,
    lsh_structure,  # LSHStructure
    budget: int,
    min_depth: int = 0,
    gamma: float = 1.0,
    tau: float = 0.0
) -> Tuple[np.ndarray, int]:
    """
    Simplified prefix sampling with nested buckets and angle-based correction.

    For each tree, build nested prefix buckets where bucket at depth d contains
    all keys with LCP >= d (prefix match of at least d bits).

    Sampling:
    1. Sample tree uniformly from L trees
    2. Sample depth uniformly from non-empty depths in that tree
    3. Sample key uniformly from that bucket

    Correction probability:
        pi_i = sum over (tree, depth) where key appears:
               (1/L) × (1/|valid_depths|) × (1/|bucket_size|) × p_angle(depth)

    where p_angle(depth) is the LSH collision probability at that depth.

    Args:
        gamma: (unused in this simplified version)
        tau: (unused in this simplified version)
        min_depth: Minimum depth threshold
    """
    num_keys = len(keys)
    if num_keys == 0 or budget <= 0:
        return np.zeros(head_dim), 0

    query_hash = lsh_structure.hash_query(query)          # [L, Kmax]
    key_codes  = lsh_structure.hash_codes                 # [N, L, Kmax]
    L = lsh_structure.num_tables
    Kmax = lsh_structure.max_depth

    # ----------------------------------------------------------
    # 1) Compute LCP (Longest Common Prefix) for each key in each tree
    # ----------------------------------------------------------
    lcp = np.zeros((L, num_keys), dtype=np.int32)

    for l in range(L):
        for i in range(num_keys):
            # Count how many bits match from the start
            match_count = 0
            for bit in range(Kmax):
                if key_codes[i, l, bit] == query_hash[l, bit]:
                    match_count += 1
                else:
                    break
            lcp[l, i] = match_count

    # ----------------------------------------------------------
    # 2) Build nested prefix buckets for each tree
    #    bucket[(tree, depth)] = list of keys with LCP >= depth
    # ----------------------------------------------------------
    buckets = {}  # (tree, depth) -> list of key indices
    tree_valid_depths = {}  # tree -> list of non-empty depths

    for l in range(L):
        valid_depths = []
        for d in range(min_depth, Kmax + 1):
            # Find keys with LCP >= d in this tree
            keys_at_depth = np.where(lcp[l] >= d)[0]
            if len(keys_at_depth) > 0:
                buckets[(l, d)] = keys_at_depth
                valid_depths.append(d)

        if len(valid_depths) > 0:
            tree_valid_depths[l] = valid_depths

    if len(tree_valid_depths) == 0:
        return np.zeros(head_dim), 0

    # ----------------------------------------------------------
    # 3) Precompute angle-based probabilities for correction
    #    p_angle[i, d] = collision probability at depth d
    # ----------------------------------------------------------
    query_norm = np.linalg.norm(query)
    key_norms = np.linalg.norm(keys, axis=1)
    cos_sims = np.clip(
        (keys @ query) / (query_norm * key_norms + 1e-8),
        -1.0 + 1e-8, 1.0 - 1e-8
    )
    thetas = np.arccos(cos_sims)  # [N]
    p_bits = 1.0 - thetas / np.pi  # Probability of single bit match

    # p_collision[i, d] = p_bits[i]^d (probability of d-bit match)
    p_collision = np.zeros((num_keys, Kmax + 1), dtype=np.float64)
    for d in range(Kmax + 1):
        p_collision[:, d] = np.power(p_bits, d)

    # ----------------------------------------------------------
    # 4) Sample budget times
    # ----------------------------------------------------------
    sampled_indices = []
    sampled_pi = []

    for _ in range(budget):
        # Sample tree uniformly from trees with valid depths
        trees_list = list(tree_valid_depths.keys())
        l = int(np.random.choice(trees_list))

        # Sample depth uniformly from valid depths in that tree
        valid_depths = tree_valid_depths[l]
        d = int(np.random.choice(valid_depths))

        # Sample key uniformly from bucket
        bucket_keys = buckets[(l, d)]
        i = int(np.random.choice(bucket_keys))

        sampled_indices.append(i)

        # ----------------------------------------------------------
        # 5) Compute proposal probability pi_i
        #    pi_i = sum over all (tree, depth) where key i appears:
        #           (1/L_valid) × (1/|valid_depths_in_tree|) × (1/|bucket_size|) × p_collision[i, d]
        # ----------------------------------------------------------
        pi = 0.0
        num_valid_trees = len(tree_valid_depths)

        for tree_idx in tree_valid_depths.keys():
            valid_depths_in_tree = tree_valid_depths[tree_idx]
            num_valid_depths = len(valid_depths_in_tree)

            for depth in valid_depths_in_tree:
                bucket = buckets[(tree_idx, depth)]
                # Check if key i is in this bucket (it should be if lcp[tree, i] >= depth)
                if lcp[tree_idx, i] >= depth:
                    bucket_size = len(bucket)
                    # Proposal probability for this (tree, depth, key) combination
                    p_tree = 1.0 / num_valid_trees
                    p_depth = 1.0 / num_valid_depths
                    p_key = 1.0 / bucket_size
                    p_angle = p_collision[i, depth]

                    pi += p_tree * p_depth * p_key * p_angle

        pi = max(pi, 1e-12)  # Avoid zero
        sampled_pi.append(pi)

    if len(sampled_indices) == 0:
        return np.zeros(head_dim), 0

    # ----------------------------------------------------------
    # 6) SNIS correction
    # ----------------------------------------------------------
    sampled_indices = np.array(sampled_indices, dtype=np.int32)
    sampled_pi = np.array(sampled_pi, dtype=np.float64)

    selected_logits = logits[sampled_indices]
    selected_values = values[sampled_indices]

    output = snis_estimator(selected_logits, selected_values, sampled_pi, head_dim)
    return output, len(sampled_indices)
