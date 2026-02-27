"""
GMM Soft Clustering Attention.

Fits a Gaussian Mixture Model on key vectors, creating soft clusters.
Each cluster's representative key and value are responsibility-weighted
averages. Attention over representatives uses softmax on query-to-representative
scores, analogous to hierarchical LSH tree-aggregation but with learned soft clustering.
"""

import numpy as np
from typing import Tuple
from algorithms.base import softmax


def fit_gmm(keys: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    """
    Fit GMM on key vectors and return responsibilities.

    Args:
        keys: [N, head_dim] key vectors
        n_clusters: number of Gaussian components
        seed: random seed for reproducibility

    Returns:
        resp: [N, n_clusters] posterior responsibilities P(cluster | key)
    """
    from sklearn.mixture import GaussianMixture

    n_keys = len(keys)
    if n_clusters >= n_keys:
        resp = np.eye(n_keys, n_clusters)
        return resp

    if n_clusters == 1:
        return np.ones((n_keys, 1))

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='diag',
        max_iter=100,
        n_init=1,
        random_state=seed,
    )
    gmm.fit(keys)
    resp = gmm.predict_proba(keys)
    return resp


def gmm_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    logits: np.ndarray,
    head_dim: int,
    resp: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    GMM-based soft clustering attention.

    Groups keys into soft clusters using precomputed GMM responsibilities.
    Computes responsibility-weighted average key and value per cluster,
    then applies softmax attention over representatives.

    Args:
        query: [head_dim] query vector
        keys: [nv, head_dim] valid key vectors (causal)
        values: [nv, head_dim] valid value vectors (causal)
        logits: unused (kept for interface compatibility)
        head_dim: dimension
        resp: [nv, n_clusters] precomputed responsibilities for valid keys

    Returns:
        output: [head_dim] approximate attention output
        n_active: number of active clusters used
    """
    nv = len(keys)
    n_clusters = resp.shape[1]

    if nv == 0:
        return np.zeros(head_dim), 0

    # Effective counts per cluster: sum of responsibilities
    effective_counts = resp.sum(axis=0)  # [C]
    active_mask = effective_counts > 1e-8

    if not active_mask.any():
        return np.zeros(head_dim), 0

    # Responsibility-weighted average keys and values for active clusters
    active_resp = resp[:, active_mask]          # [nv, A]
    active_counts = effective_counts[active_mask]  # [A]

    avg_keys = (active_resp.T @ keys) / active_counts[:, np.newaxis]    # [A, d]
    avg_values = (active_resp.T @ values) / active_counts[:, np.newaxis]  # [A, d]

    n_active = int(active_mask.sum())

    # Softmax over representative keys (no count weighting — GMM responsibilities
    # already distribute mass across all keys)
    sqrt_d = np.sqrt(head_dim)
    scores = (avg_keys @ query) / sqrt_d
    weights = softmax(scores)
    output = weights @ avg_values

    return output, n_active
