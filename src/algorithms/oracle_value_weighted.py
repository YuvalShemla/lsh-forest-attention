"""
Value-weighted oracle sampling.

Samples proportional to w_i * ||v_i|| instead of just w_i, then applies
importance sampling correction to remain unbiased.

Motivation: The optimal IS proposal for estimating o = sum(w_i * v_i) is
q_i* ∝ w_i * ||v_i||, which minimizes estimator variance by upweighting
values that contribute more in both attention weight and magnitude.
"""

import numpy as np
from typing import Tuple


def oracle_value_weighted(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    logits: np.ndarray,
    true_weights: np.ndarray,
    budget: int
) -> Tuple[np.ndarray, int]:
    """
    Value-weighted oracle sampling (unbiased, privileged).

    Proposal: q_i ∝ w_i * ||v_i||
    Estimator: ô = (1/B) * sum( (w_i / q_i) * v_i )  for sampled i ~ q
             = (Z/B) * sum( v_i / ||v_i|| )
    where Z = sum(w_i * ||v_i||).

    Args:
        query: [head_dim] query vector (unused, kept for interface consistency)
        keys: [N, head_dim] key vectors (unused, kept for interface consistency)
        values: [N, head_dim] value vectors
        logits: [N] pre-softmax scores (unused)
        true_weights: [N] normalized attention weights (softmax output)
        budget: number of samples to draw

    Returns:
        output: [head_dim] estimated attention output
        unique_budget: number of unique keys sampled
    """
    num_keys = len(true_weights)
    budget = min(budget, num_keys)

    # Value norms
    value_norms = np.linalg.norm(values, axis=1)  # [N]
    value_norms = np.maximum(value_norms, 1e-10)   # avoid division by zero

    # Proposal distribution: q_i ∝ w_i * ||v_i||
    proposal = true_weights * value_norms
    proposal_sum = proposal.sum()
    if proposal_sum < 1e-12:
        # Fallback to uniform if degenerate
        proposal = np.ones(num_keys) / num_keys
    else:
        proposal = proposal / proposal_sum

    # Sample from proposal
    sampled_indices = np.random.choice(num_keys, size=budget, p=proposal, replace=True)

    # IS correction: w_i / q_i = Z / ||v_i|| where Z = sum(w_i * ||v_i||)
    Z = np.sum(true_weights * value_norms)
    sampled_values = values[sampled_indices]           # [B, head_dim]
    sampled_norms = value_norms[sampled_indices]       # [B]

    # ô = (Z / B) * sum( v_i / ||v_i|| )
    normalized_values = sampled_values / sampled_norms[:, np.newaxis]
    output = (Z / budget) * np.sum(normalized_values, axis=0)

    unique_budget = len(np.unique(sampled_indices))

    return output, unique_budget
