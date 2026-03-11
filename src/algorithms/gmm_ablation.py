"""
GMM Bias Source Ablation Variants.

Three variants that isolate the two bias sources in GMM attention:
  - Source 1 (weight distortion): softmax over centroid logits != true W_c
  - Source 2 (value averaging): responsibility-weighted values != attention-weighted means

Each variant replaces one or both sources with oracle (exact) counterparts,
requiring the true attention weights as input.
"""

import numpy as np
from typing import Tuple
from algorithms.base import softmax


def gmm_exact_weights(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    logits: np.ndarray,
    head_dim: int,
    resp: np.ndarray,
    true_weights: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    GMM with exact cluster weights (eliminates Source 1: weight distortion).

    Cluster weights are computed from true attention weights instead of
    softmax over centroid logits. Value representatives remain
    responsibility-weighted averages (same as standard GMM).
    """
    nv = len(keys)
    if nv == 0:
        return np.zeros(head_dim), 0

    effective_counts = resp.sum(axis=0)
    active_mask = effective_counts > 1e-8
    if not active_mask.any():
        return np.zeros(head_dim), 0

    active_resp = resp[:, active_mask]
    active_counts = effective_counts[active_mask]

    # Value representatives: responsibility-weighted (same as standard GMM)
    avg_values = (active_resp.T @ values) / active_counts[:, np.newaxis]

    # Weights: exact cluster masses from true attention weights
    cluster_weights = active_resp.T @ true_weights  # [A]
    cluster_weights = cluster_weights / cluster_weights.sum()

    output = cluster_weights @ avg_values
    return output, int(active_mask.sum())


def gmm_exact_values(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    logits: np.ndarray,
    head_dim: int,
    resp: np.ndarray,
    true_weights: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    GMM with attention-weighted value representatives (eliminates Source 2: value averaging).

    Cluster weights are computed via softmax over centroid logits (same as
    standard GMM). Value representatives use true attention weights instead
    of GMM responsibilities.
    """
    nv = len(keys)
    if nv == 0:
        return np.zeros(head_dim), 0

    effective_counts = resp.sum(axis=0)
    active_mask = effective_counts > 1e-8
    if not active_mask.any():
        return np.zeros(head_dim), 0

    active_resp = resp[:, active_mask]
    active_counts = effective_counts[active_mask]

    # Key representatives: responsibility-weighted (same as standard GMM)
    avg_keys = (active_resp.T @ keys) / active_counts[:, np.newaxis]

    # Value representatives: attention-weighted within-cluster means
    # mu_c = sum_i r_{ic} * w_i * v_i / sum_i r_{ic} * w_i
    w_resp = active_resp * true_weights[:, np.newaxis]  # [nv, A]
    w_resp_sums = w_resp.sum(axis=0)  # [A]
    safe_sums = np.maximum(w_resp_sums, 1e-12)
    avg_values = (w_resp.T @ values) / safe_sums[:, np.newaxis]

    # Weights: softmax over centroid logits (same as standard GMM)
    sqrt_d = np.sqrt(head_dim)
    scores = (avg_keys @ query) / sqrt_d
    weights = softmax(scores)

    output = weights @ avg_values
    return output, int(active_mask.sum())


def gmm_exact_both(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    logits: np.ndarray,
    head_dim: int,
    resp: np.ndarray,
    true_weights: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    GMM with exact weights AND exact value representatives (eliminates both sources).

    Shows the irreducible error from the partition itself: even with perfect
    weights and representatives, the soft partition still loses information.
    """
    nv = len(keys)
    if nv == 0:
        return np.zeros(head_dim), 0

    effective_counts = resp.sum(axis=0)
    active_mask = effective_counts > 1e-8
    if not active_mask.any():
        return np.zeros(head_dim), 0

    active_resp = resp[:, active_mask]

    # Weights: exact cluster masses
    cluster_weights = active_resp.T @ true_weights
    cluster_weights = cluster_weights / cluster_weights.sum()

    # Value representatives: attention-weighted within-cluster means
    w_resp = active_resp * true_weights[:, np.newaxis]
    w_resp_sums = w_resp.sum(axis=0)
    safe_sums = np.maximum(w_resp_sums, 1e-12)
    avg_values = (w_resp.T @ values) / safe_sums[:, np.newaxis]

    output = cluster_weights @ avg_values
    return output, int(active_mask.sum())
