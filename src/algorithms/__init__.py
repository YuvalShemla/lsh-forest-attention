"""
Algorithms for sparse attention approximation.

Re-exports all algorithms and shared classes for easy importing:
    from algorithms import topk_attention, uniform_sampling, ...
    from algorithms import LSHStructure, SimHashIndex, CrossPolytopeIndex
    from algorithms import softmax, snis_estimator, relative_l2_error, ...
"""

# Base utilities
from algorithms.base import (
    softmax,
    compute_ground_truth_attention,
    relative_l2_error,
    snis_estimator,
    inclusion_prob,
)

# LSH index structures
from algorithms.lsh_index import (
    LSHStructure,
    SimHashIndex,
    CrossPolytopeIndex,
)

# Algorithms
from algorithms.full_attention import full_attention
from algorithms.topk import topk_attention
from algorithms.uniform import uniform_sampling
from algorithms.oracle import oracle_sampling
from algorithms.simhash_snis import simhash_snis
from algorithms.cross_polytope_snis import cross_polytope_snis
from algorithms.jungle_sampling import jungle_sampling
from algorithms.hierarchical_lsh import hierarchical_lsh_attention

__all__ = [
    # Base
    'softmax', 'compute_ground_truth_attention', 'relative_l2_error',
    'snis_estimator', 'inclusion_prob',
    # Index structures
    'LSHStructure', 'SimHashIndex', 'CrossPolytopeIndex',
    # Algorithms
    'full_attention', 'topk_attention', 'uniform_sampling',
    'oracle_sampling', 'simhash_snis', 'cross_polytope_snis',
    'jungle_sampling', 'hierarchical_lsh_attention',
]
