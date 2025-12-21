"""
Utility functions for attention approximation.

Contains:
- Softmax
- Ground truth attention computation
- Relative L2 error
- LSH structure
- SNIS estimator
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


class LSHStructure:
    """LSH using SimHash (random hyperplane hashing)."""
    
    def __init__(self, num_tables: int, max_depth: int, head_dim: int, 
                 center_keys: bool = True, seed: int = None):
        self.num_tables = num_tables
        self.max_depth = max_depth
        self.head_dim = head_dim
        self.center_keys = center_keys
        
        # Random hyperplanes: [num_tables, max_depth, head_dim]
        if seed is not None:
            np.random.seed(seed)
        self.hyperplanes = np.random.randn(num_tables, max_depth, head_dim)
        norms = np.linalg.norm(self.hyperplanes, axis=2, keepdims=True)
        self.hyperplanes = self.hyperplanes / norms
        
        self.key_mean = None
        self.hash_codes = None
    
    def build_index(self, keys: np.ndarray) -> None:
        """Build hash index for keys."""
        if self.center_keys:
            self.key_mean = np.mean(keys, axis=0)
            centered_keys = keys - self.key_mean
        else:
            self.key_mean = np.zeros(self.head_dim)
            centered_keys = keys
        
        # Project and binarize: [num_keys, num_tables, max_depth]
        projections = np.einsum('nd,ltd->nlt', centered_keys, self.hyperplanes)
        self.hash_codes = (projections > 0).astype(np.int8)
    
    def hash_query(self, query: np.ndarray) -> np.ndarray:
        """Hash query. Returns: [num_tables, max_depth]"""
        if self.center_keys:
            centered_query = query - self.key_mean
        else:
            centered_query = query
        
        projections = np.tensordot(self.hyperplanes, centered_query, axes=(2, 0))
        return (projections > 0).astype(np.int8)


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

