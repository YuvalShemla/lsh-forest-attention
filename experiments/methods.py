"""
All attention approximation methods in one file.

Methods:
1. Top-K: Select top keys by logit
2. Naive Sampling: Uniform random sampling
3. Oracle Sampling: Sample from true distribution (gold standard)
4. LSH-SNIS: Fixed-depth LSH retrieval (MagicPIG-style)
5. prefix_sampling: Our LSH+SNIS method with cumulative prefix tree buckets
"""

import numpy as np
from typing import Tuple, Dict, Any
import utils


def topk_approximation(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    logits: np.ndarray,
    budget: int
) -> Tuple[np.ndarray, int]:
    """
    Top-K approximation (biased).
    
    Selects top-K keys by logit, renormalizes within subset.
    """
    num_keys = len(logits)
    budget = min(budget, num_keys)
    
    # Select top-K
    top_indices = np.argpartition(logits, -budget)[-budget:]
    top_indices = top_indices[np.argsort(logits[top_indices])[::-1]]
    
    # Subset softmax
    selected_logits = logits[top_indices]
    selected_values = values[top_indices]
    weights = utils.softmax(selected_logits)
    output = weights @ selected_values
    
    return output, budget


def naive_sampling(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    logits: np.ndarray,
    budget: int
) -> Tuple[np.ndarray, int]:
    """
    Naive uniform sampling (biased).
    
    Samples K keys uniformly, renormalizes within subset.
    """
    num_keys = len(logits)
    budget = min(budget, num_keys)
    
    # Uniform sampling
    selected_indices = np.random.choice(num_keys, size=budget, replace=False)
    
    # Subset softmax
    selected_logits = logits[selected_indices]
    selected_values = values[selected_indices]
    weights = utils.softmax(selected_logits)
    output = weights @ selected_values
    
    return output, budget


def oracle_sampling(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    logits: np.ndarray,
    true_weights: np.ndarray,
    budget: int
) -> Tuple[np.ndarray, int]:
    """
    Oracle sampling (unbiased, privileged).
    
    Samples from TRUE attention distribution.
    Simple average estimator.
    """
    num_keys = len(logits)
    budget = min(budget, num_keys)
    
    # Sample from true distribution
    sampled_indices = np.random.choice(num_keys, size=budget, p=true_weights, replace=True)
    
    # Simple average (unbiased!)
    sampled_values = values[sampled_indices]
    output = np.mean(sampled_values, axis=0)
    
    # Count unique keys
    unique_budget = len(np.unique(sampled_indices))
    
    return output, unique_budget


def lsh_snis(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    logits: np.ndarray,
    head_dim: int,
    lsh_structure: utils.LSHStructure,
    depth_k: int,
    min_hits: int = 1
) -> Tuple[np.ndarray, int]:
    """
    LSH-SNIS (MagicPIG-style).
    
    Retrieves ALL keys matching at depth K in at least min_hits tables.
    Applies SNIS correction based on angle.
    Budget NOT controlled - determined by K, L, and collisions.
    
    Args:
        depth_k: Hash depth K
        min_hits: Minimum number of tables that must match (typically 1)
    
    Returns:
        output: Approximated attention output
        budget: Number of keys retrieved (varies by K)
    """
    query_hash = lsh_structure.hash_query(query)
    key_codes = lsh_structure.hash_codes  # [num_keys, L, K]
    num_keys = len(keys)
    
    # Retrieve keys matching at depth K
    # For each key, check if it matches in at least min_hits tables
    key_prefixes = key_codes[:, :, :depth_k]  # [num_keys, L, depth_k]
    query_prefix = query_hash[:, :depth_k]  # [L, depth_k]
    
    # Check matches: [num_keys, L]
    matches = np.all(key_prefixes == query_prefix[np.newaxis, :, :], axis=2)
    
    # Count matches per key
    match_counts = np.sum(matches, axis=1)  # [num_keys]
    
    # Retrieve keys with >= min_hits matches
    retrieved_indices = np.where(match_counts >= min_hits)[0]
    retrieved_budget = len(retrieved_indices)
    
    if retrieved_budget == 0:
        return np.zeros(head_dim), 0
    
    # Get retrieved data
    retrieved_keys = keys[retrieved_indices]
    retrieved_values = values[retrieved_indices]
    retrieved_logits = logits[retrieved_indices]
    
    # Compute inclusion probabilities (angle-based)
    query_norm = np.linalg.norm(query)
    key_norms = np.linalg.norm(retrieved_keys, axis=1)
    cos_sims = np.clip(
        (retrieved_keys @ query) / (query_norm * key_norms + 1e-8),
        -1.0 + 1e-8, 1.0 - 1e-8
    )
    thetas = np.arccos(cos_sims)
    p_bits = 1.0 - thetas / np.pi
    
    # Collision probability at depth K
    p_per_table = np.power(p_bits, depth_k)
    
    # Inclusion probability (at least min_hits of L tables)
    L = lsh_structure.num_tables
    
    if min_hits == 1:
        # P(at least 1 table hits)
        inclusion_probs = 1.0 - np.power(1.0 - p_per_table, L)
    elif min_hits == 2:
        # P(at least 2 tables hit) = 1 - P(0 hits) - P(exactly 1 hit)
        # This is the MagicPIG formula!
        p_zero_hits = np.power(1.0 - p_per_table, L)
        p_one_hit = L * p_per_table * np.power(1.0 - p_per_table, L - 1)
        inclusion_probs = 1.0 - p_zero_hits - p_one_hit
    else:
        # For min_hits > 2, use binomial CDF (requires scipy)
        # For now, approximate
        inclusion_probs = 1.0 - np.power(1.0 - p_per_table, L)
    
    inclusion_probs = np.clip(inclusion_probs, 1e-8, 1.0)
    
    # SNIS estimator
    output = utils.snis_estimator(retrieved_logits, retrieved_values, inclusion_probs, head_dim)
    
    return output, retrieved_budget


def prefix_sampling(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    logits: np.ndarray,
    head_dim: int,
    lsh_structure: utils.LSHStructure,
    budget: int,
    gamma: float = 1.0,
    tau: float = 0.0,
    min_depth: int = 0
) -> Tuple[np.ndarray, int]:
    """
    Prefix sampling with cumulative prefix tree buckets (our method).
    
    Inspired by LSH Forest (Bawa et al., 2005), this method uses prefix tree structure
    for efficient querying. Keys are organized in a prefix tree where:
    - Internal nodes at depth k contain all keys matching k-bit prefix
    - Leaf nodes contain keys with full hash codes
    - Query descends tree to find matching keys at various depths
    
    IMPLEMENTATION NOTE:
    Current implementation explicitly computes max depth for each key (O(N) at query time).
    This can be optimized using proper prefix tree data structure:
    - Preprocessing: Build L prefix trees, store keys at nodes by prefix
    - Query time: Descend each tree following query hash (O(L*K) traversal)
    - Collect keys from nodes at depth k for all k (union operation)
    - Never need to iterate over all N keys!
    
    With prefix tree: Query time becomes O(L*K + |retrieved|) - truly sublinear!
    
    Algorithm:
    1. For each key, find its depth in the prefix tree structure
    2. Filter keys: only consider keys with max_depth >= min_depth
    3. Create cumulative buckets: B_k = {keys at depth >= k in prefix tree}
    4. Compute two-stage probability:
       - p_retrieval: LSH collision probability at key's depth
       - intensity: Based on cumulative bucket size (selectivity)
    5. Sample K keys from normalized distribution (over filtered set)
    6. SNIS correction with unnormalized probabilities
    
    Args:
        gamma: Bucket size penalty (1.0 = linear: 1/bucket_size)
        tau: Smoothing term (0.0 = no smoothing, typically not needed)
        min_depth: Minimum depth threshold (only sample from keys with max_depth >= min_depth)
                  Default 0 means no filtering (all keys eligible)
    
    Returns:
        output: Approximated attention output [head_dim]
        budget: Number of keys actually sampled
    """
    num_keys = len(keys)
    query_hash = lsh_structure.hash_query(query)
    key_codes = lsh_structure.hash_codes  # [num_keys, L, K]
    
    # ==============================================================
    # Step 1: Compute max depth for each key
    # ==============================================================
    max_depths = np.zeros(num_keys, dtype=np.int32)
    
    for key_idx in range(num_keys):
        key_max_depth = 0
        for table_idx in range(lsh_structure.num_tables):
            depth = 0
            for bit_idx in range(lsh_structure.max_depth):
                if key_codes[key_idx, table_idx, bit_idx] == query_hash[table_idx, bit_idx]:
                    depth += 1
                else:
                    break
            key_max_depth = max(key_max_depth, depth)
        max_depths[key_idx] = key_max_depth
    
    # ==============================================================
    # Step 2: Filter by minimum depth
    # ==============================================================
    # Only consider keys with max_depth >= min_depth
    valid_mask = max_depths >= min_depth
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        # No keys meet the minimum depth requirement
        return np.zeros(head_dim), 0
    
    # Filter all arrays to only valid keys
    valid_keys = keys[valid_indices]
    valid_values = values[valid_indices]
    valid_logits = logits[valid_indices]
    valid_max_depths = max_depths[valid_indices]
    num_valid = len(valid_indices)
    
    # ==============================================================
    # Step 3: Create CUMULATIVE bucket sizes (monotonic) for filtered set
    # ==============================================================
    cumulative_bucket_sizes = np.zeros(num_valid, dtype=np.int32)
    
    # Compute cumulative sizes only for valid keys
    for depth in range(min_depth, lsh_structure.max_depth + 1):
        # Count valid keys at THIS depth or deeper
        keys_at_least_depth = np.sum(valid_max_depths >= depth)
        
        # Assign this cumulative size to keys at EXACTLY this depth
        keys_exactly_at_depth = (valid_max_depths == depth)
        cumulative_bucket_sizes[keys_exactly_at_depth] = keys_at_least_depth
    
    # ==============================================================
    # Step 4: Compute unnormalized probabilities (for filtered set)
    # ==============================================================
    
    # Stage 1: LSH retrieval probability
    query_norm = np.linalg.norm(query)
    valid_key_norms = np.linalg.norm(valid_keys, axis=1)
    cos_sims = np.clip(
        (valid_keys @ query) / (query_norm * valid_key_norms + 1e-8),
        -1.0 + 1e-8, 1.0 - 1e-8
    )
    thetas = np.arccos(cos_sims)
    p_bits = 1.0 - thetas / np.pi
    
    p_retrieval = np.zeros(num_valid)
    for i in range(num_valid):
        depth_i = valid_max_depths[i]
        p_collision = p_bits[i] ** depth_i
        p_retrieval[i] = 1.0 - (1.0 - p_collision) ** lsh_structure.num_tables
    
    p_retrieval = np.clip(p_retrieval, 1e-8, 1.0)
    
    # Stage 2: Bucket-based intensity
    intensities = np.power(1.0 / (cumulative_bucket_sizes + tau), gamma)
    
    # Combined (unnormalized)
    u_i_unnormalized = p_retrieval * intensities
    
    # ==============================================================
    # Step 5: Sample from normalized distribution (over filtered set)
    # ==============================================================
    
    # Normalize over filtered set only
    p_distribution = u_i_unnormalized / np.sum(u_i_unnormalized)
    
    # Sample K keys from filtered set
    budget = min(budget, num_valid)
    sampled_local_indices = np.random.choice(
        num_valid,
        size=budget,
        replace=False,
        p=p_distribution
    )
    
    # Convert local indices (within filtered set) to global indices
    sampled_global_indices = valid_indices[sampled_local_indices]
    
    # ==============================================================
    # Step 6: SNIS correction (use UNNORMALIZED u_i from filtered set)
    # ==============================================================
    
    selected_logits = valid_logits[sampled_local_indices]
    selected_values = valid_values[sampled_local_indices]
    selected_u_i = u_i_unnormalized[sampled_local_indices]  # Unnormalized!
    
    output = utils.snis_estimator(selected_logits, selected_values, selected_u_i, head_dim)
    
    return output, budget

