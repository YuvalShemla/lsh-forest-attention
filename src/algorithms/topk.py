"""Top-K attention approximation (biased truncation)."""

import numpy as np
from typing import Tuple
from algorithms.base import softmax


def topk_attention(
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
    weights = softmax(selected_logits)
    output = weights @ selected_values

    return output, budget
