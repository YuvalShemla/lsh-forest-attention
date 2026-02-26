"""SimHash fixed-depth LSH with SNIS correction (MagicPIG-style)."""

import numpy as np
from typing import Tuple
from algorithms.base import snis_estimator, inclusion_prob


def simhash_snis(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    logits: np.ndarray,
    head_dim: int,
    index,  # SimHashIndex or LSHStructure
    depth_k: int,
    L_use: int = None,
    min_hits: int = 1
) -> Tuple[np.ndarray, int]:
    """
    SimHash LSH-SNIS (MagicPIG-style).

    Retrieves ALL keys matching at depth K in at least min_hits tables.
    Applies SNIS correction based on angle.
    Budget NOT controlled.

    Args:
        index: LSH index with .hash_query() and .hash_codes attributes
        depth_k: Hash depth K
        L_use: Number of tables to use (defaults to all)
        min_hits: Minimum number of table matches
    """
    query_hash = index.hash_query(query) if hasattr(index, 'hash_query') else index.batch_hash_queries(query[np.newaxis])[0]
    key_codes = index.hash_codes if hasattr(index, 'hash_codes') else index.key_codes
    num_keys = len(keys)
    L_total = key_codes.shape[1]
    L_use = L_use or L_total

    # Retrieve keys matching at depth_k
    key_prefixes = key_codes[:num_keys, :L_use, :depth_k]
    query_prefix = query_hash[:L_use, :depth_k]

    matches = np.all(key_prefixes == query_prefix[np.newaxis, :, :], axis=2)
    match_counts = np.sum(matches, axis=1)

    retrieved_indices = np.where(match_counts >= min_hits)[0]
    retrieved_budget = len(retrieved_indices)

    if retrieved_budget == 0:
        return np.zeros(head_dim), 0

    # Get retrieved data
    retrieved_keys = keys[retrieved_indices]
    retrieved_values = values[retrieved_indices]
    retrieved_logits = logits[retrieved_indices]

    # Compute inclusion probabilities (angle-based, SimHash)
    query_norm = np.linalg.norm(query)
    key_norms = np.linalg.norm(retrieved_keys, axis=1)
    cos_sims = np.clip(
        (retrieved_keys @ query) / (query_norm * key_norms + 1e-8),
        -1.0 + 1e-8, 1.0 - 1e-8
    )
    thetas = np.arccos(cos_sims)
    p_bits = 1.0 - thetas / np.pi
    p_per_table = np.power(p_bits, depth_k)

    inclusion_probs = np.clip(inclusion_prob(p_per_table, L_use, min_hits), 1e-8, 1.0)

    output = snis_estimator(retrieved_logits, retrieved_values, inclusion_probs, head_dim)
    return output, retrieved_budget
