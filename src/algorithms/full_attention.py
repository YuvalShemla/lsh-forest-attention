"""Full (exact) attention computation - ground truth reference."""

import numpy as np
from typing import Tuple
from algorithms.base import softmax


def full_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    logits: np.ndarray,
    head_dim: int
) -> Tuple[np.ndarray, int]:
    """
    Full exact attention (ground truth).

    Returns:
        output: [head_dim] - exact attention output
        actual_budget: int - number of keys used (= len(keys))
    """
    weights = softmax(logits)
    output = weights @ values
    return output, len(keys)
