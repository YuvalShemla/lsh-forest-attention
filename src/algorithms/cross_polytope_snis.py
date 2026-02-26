"""Cross-Polytope LSH with SNIS correction."""

import numpy as np
from typing import Tuple
from algorithms.base import snis_estimator, inclusion_prob


def cross_polytope_snis(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    logits: np.ndarray,
    head_dim: int,
    index,  # CrossPolytopeIndex
    k_cp: int,
    L_use: int = None,
    min_hits: int = 1
) -> Tuple[np.ndarray, int]:
    """
    Cross-Polytope LSH-SNIS.
    Same structure as SimHash but uses cross-polytope hash and different collision probability.
    """
    query_hash = index.batch_hash_queries(query[np.newaxis])[0]
    key_codes = index.key_codes
    num_keys = len(keys)
    L_total = key_codes.shape[1]
    L_use = L_use or L_total

    key_prefixes = key_codes[:num_keys, :L_use, :k_cp]
    query_prefix = query_hash[:L_use, :k_cp]

    matches = np.all(key_prefixes == query_prefix[np.newaxis, :, :], axis=2)
    match_counts = np.sum(matches, axis=1)

    retrieved_indices = np.where(match_counts >= min_hits)[0]
    retrieved_budget = len(retrieved_indices)

    if retrieved_budget == 0:
        return np.zeros(head_dim), 0

    retrieved_keys = keys[retrieved_indices]
    retrieved_values = values[retrieved_indices]
    retrieved_logits = logits[retrieved_indices]

    # Cross-polytope collision probability
    query_norm = np.linalg.norm(query)
    key_norms = np.linalg.norm(retrieved_keys, axis=1)
    cos_sims = np.clip(
        (retrieved_keys @ query) / (query_norm * key_norms + 1e-8),
        -1.0 + 1e-8, 1.0 - 1e-8
    )
    thetas = np.arccos(cos_sims)

    p_per_table = index.collision_prob(thetas, k_cp)
    inclusion_probs = np.clip(inclusion_prob(p_per_table, L_use, min_hits), 1e-8, 1.0)

    output = snis_estimator(retrieved_logits, retrieved_values, inclusion_probs, head_dim)
    return output, retrieved_budget
