"""
LSH index structures for hashing keys and queries.

Contains: LSHStructure (SimHash-based forest), SimHashIndex, CrossPolytopeIndex
"""

import numpy as np


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


class SimHashIndex:
    """SimHash index optimized for parameter sweeps. Supports batch query hashing."""

    def __init__(self, num_tables, max_depth, head_dim, center_keys=True, seed=None):
        self.num_tables = num_tables
        self.max_depth = max_depth
        self.head_dim = head_dim
        self.center_keys = center_keys
        rng = np.random.RandomState(seed)
        hp = rng.randn(num_tables, max_depth, head_dim).astype(np.float32)
        self.hyperplanes = hp / np.linalg.norm(hp, axis=2, keepdims=True)
        self.key_mean = None
        self.key_codes = None

    def build_index(self, keys):
        if self.center_keys:
            self.key_mean = np.mean(keys, axis=0)
            c = keys - self.key_mean
        else:
            self.key_mean = np.zeros(self.head_dim, dtype=np.float32)
            c = keys
        self.key_codes = (np.einsum('nd,ltd->nlt', c, self.hyperplanes) > 0).astype(np.int8)

    def batch_hash_queries(self, Q):
        """Hash all queries at once. Returns [num_queries, L, max_depth]."""
        c = Q - self.key_mean if self.center_keys else Q
        return (np.einsum('qd,ltd->qlt', c, self.hyperplanes) > 0).astype(np.int8)

    @staticmethod
    def collision_prob(thetas, depth_k):
        p_bit = 1.0 - thetas / np.pi
        return np.power(np.clip(p_bit, 0, 1), depth_k)


class CrossPolytopeIndex:
    """Cross-Polytope LSH index. Supports batch query hashing."""

    def __init__(self, num_tables, max_cp, head_dim, center_keys=True, seed=None):
        self.num_tables = num_tables
        self.max_cp = max_cp
        self.head_dim = head_dim
        self.center_keys = center_keys
        rng = np.random.RandomState(seed)
        self.rotations = rng.randn(num_tables, max_cp, head_dim, head_dim).astype(np.float32)
        self.key_mean = None
        self.key_codes = None

    def _hash_batch(self, vectors):
        N = vectors.shape[0]
        codes = np.zeros((N, self.num_tables, self.max_cp), dtype=np.int32)
        for l in range(self.num_tables):
            for cp in range(self.max_cp):
                rot = vectors @ self.rotations[l, cp].T
                norms = np.linalg.norm(rot, axis=1, keepdims=True)
                rot = rot / (norms + 1e-10)
                max_j = np.argmax(np.abs(rot), axis=1)
                signs = rot[np.arange(N), max_j]
                codes[:, l, cp] = 2 * max_j + (signs < 0).astype(np.int32)
        return codes

    def build_index(self, keys):
        if self.center_keys:
            self.key_mean = np.mean(keys, axis=0)
            c = keys - self.key_mean
        else:
            self.key_mean = np.zeros(self.head_dim, dtype=np.float32)
            c = keys
        self.key_codes = self._hash_batch(c)

    def batch_hash_queries(self, Q):
        c = Q - self.key_mean if self.center_keys else Q
        return self._hash_batch(c)

    def collision_prob_single(self, thetas):
        d = self.head_dim
        tau_sq = 2.0 * (1.0 - np.cos(thetas))
        denom = np.clip(4.0 - tau_sq, 1e-10, None)
        return np.power(float(d), -(tau_sq / denom))

    def collision_prob(self, thetas, k_cp):
        return np.power(self.collision_prob_single(thetas), k_cp)
