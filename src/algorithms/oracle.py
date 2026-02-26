"""Oracle sampling (from true attention distribution)."""

import numpy as np
from typing import Tuple


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
    Samples from TRUE attention distribution. Simple average estimator.
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
