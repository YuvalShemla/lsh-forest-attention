"""
Shared utilities for attention approximation algorithms.

Contains: softmax, snis_estimator, relative_l2_error, inclusion_prob,
          compute_ground_truth_attention
"""

import numpy as np
from typing import Tuple


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def compute_ground_truth_attention(
    q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    query_pos: int,
    head_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute full attention (ground truth).

    Returns:
        output: [head_dim]
        logits: [num_valid]
        weights: [num_valid]
        normalizer: float
    """
    # Causality: only attend to positions 0...query_pos
    valid_keys = K[:query_pos + 1]
    valid_values = V[:query_pos + 1]

    # Scaled dot products
    logits = (q @ valid_keys.T) / np.sqrt(head_dim)

    # Softmax
    unnorm_weights = np.exp(logits - np.max(logits))
    normalizer = np.sum(unnorm_weights)
    weights = unnorm_weights / normalizer

    # Output
    output = weights @ valid_values

    return output, logits, weights, normalizer


def relative_l2_error(approx: np.ndarray, truth: np.ndarray, epsilon: float = 1e-6) -> float:
    """Compute relative L2 error."""
    numerator = np.linalg.norm(approx - truth)
    denominator = np.linalg.norm(truth) + epsilon
    return numerator / denominator


def snis_estimator(logits: np.ndarray, values: np.ndarray,
                   inclusion_probs: np.ndarray, head_dim: int) -> np.ndarray:
    """
    Self-Normalized Importance Sampling estimator.

    Args:
        logits: [num_sampled]
        values: [num_sampled, head_dim]
        inclusion_probs: [num_sampled] - UNNORMALIZED probabilities
        head_dim: dimension

    Returns:
        output: [head_dim]
    """
    if len(logits) == 0:
        return np.zeros(head_dim)

    # Importance-weighted logits
    weighted_logits = logits - np.log(inclusion_probs)

    # Softmax
    weights = softmax(weighted_logits)

    # Output
    output = weights @ values

    return output


def inclusion_prob(p_table: np.ndarray, L: int, min_hits: int) -> np.ndarray:
    """
    P(key collides in >= min_hits out of L tables).
    Uses exact binomial CDF complement.
    """
    if min_hits == 1:
        return 1.0 - np.power(1.0 - p_table, L)
    elif min_hits == 2:
        p0 = np.power(1.0 - p_table, L)
        p1 = L * p_table * np.power(1.0 - p_table, L - 1)
        return 1.0 - p0 - p1
    elif min_hits == 3:
        p0 = np.power(1.0 - p_table, L)
        p1 = L * p_table * np.power(1.0 - p_table, L - 1)
        p2 = (L * (L - 1) / 2.0) * np.power(p_table, 2) * np.power(1.0 - p_table, L - 2)
        return 1.0 - p0 - p1 - p2
    else:
        raise ValueError(f"min_hits={min_hits} not supported")
