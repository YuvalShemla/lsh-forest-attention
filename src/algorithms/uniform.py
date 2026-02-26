"""Uniform random sampling attention approximation."""

import numpy as np
from typing import Tuple
from algorithms.base import softmax


def uniform_sampling(
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
    weights = softmax(selected_logits)
    output = weights @ selected_values

    return output, budget
