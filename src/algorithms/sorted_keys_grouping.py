"""
Sorted-Keys Grouping Methods for Approximate Attention.

All methods take sorted logits/weights and return group assignments.
The attention approximation assigns each key the mean weight of its group,
then computes output = approx_weights @ values.

Interface: each grouping function returns group_labels: np.ndarray[int] of length n_keys,
where keys with the same label belong to the same group.
Keys are assumed pre-sorted by descending logit.
"""

import numpy as np
from typing import Tuple
from algorithms.base import softmax


# ============================================================================
# GROUPING METHODS (operate on sorted indices)
# ============================================================================

def group_equal_splits(n_keys: int, num_groups: int) -> np.ndarray:
    """Equal-size contiguous splits of sorted keys."""
    return np.repeat(np.arange(num_groups), np.diff(np.round(
        np.linspace(0, n_keys, num_groups + 1)).astype(int)))


def group_kmeans_1d(sorted_weights: np.ndarray, num_groups: int) -> np.ndarray:
    """
    Optimal 1D k-means for sorted data.

    Since the input is already sorted, optimal cluster boundaries must lie
    at consecutive gaps. We pick the (num_groups - 1) largest gaps as split
    points, which is equivalent to Fisher's natural breaks / Jenks and is
    optimal for sorted 1D data (no point crosses a cluster boundary).
    """
    n = len(sorted_weights)
    num_groups = min(num_groups, n)
    if num_groups >= n:
        return np.arange(n)
    if num_groups == 1:
        return np.zeros(n, dtype=int)

    # For sorted 1D data, optimal k-means splits at largest gaps
    gaps = np.abs(np.diff(sorted_weights))
    n_splits = num_groups - 1
    split_positions = np.sort(np.argpartition(gaps, -n_splits)[-n_splits:])

    labels = np.zeros(n, dtype=int)
    group_id = 0
    prev = 0
    for sp in split_positions:
        labels[prev:sp + 1] = group_id
        group_id += 1
        prev = sp + 1
    labels[prev:] = group_id

    return labels


def group_threshold_merging(sorted_weights: np.ndarray, num_groups: int) -> np.ndarray:
    """
    Threshold-based merging: start with each key as its own group,
    merge adjacent groups with smallest weight gap until num_groups remain.
    Equivalent to single-linkage agglomerative clustering on sorted 1D data.
    """
    n = len(sorted_weights)
    num_groups = min(num_groups, n)
    if num_groups >= n:
        return np.arange(n)

    # Gaps between consecutive sorted weights (descending, so gaps are non-negative)
    gaps = np.abs(np.diff(sorted_weights))

    # Keep the (num_groups - 1) largest gaps as split points
    n_splits = num_groups - 1
    if n_splits == 0:
        return np.zeros(n, dtype=int)

    split_positions = np.sort(np.argpartition(gaps, -n_splits)[-n_splits:])

    labels = np.zeros(n, dtype=int)
    group_id = 0
    prev = 0
    for sp in split_positions:
        labels[prev:sp + 1] = group_id
        group_id += 1
        prev = sp + 1
    labels[prev:] = group_id

    return labels


def group_equal_splits_overlap(n_keys: int, num_groups: int,
                               overlap_frac: float = 0.2) -> np.ndarray:
    """
    Equal splits with overlap. Returns a membership matrix [n_keys, num_groups]
    where each key can belong to multiple groups with fractional membership.

    Keys in the overlap zone get 0.5 membership in each neighboring group.
    """
    boundaries = np.round(np.linspace(0, n_keys, num_groups + 1)).astype(int)
    membership = np.zeros((n_keys, num_groups), dtype=np.float64)

    for g in range(num_groups):
        start, end = boundaries[g], boundaries[g + 1]
        group_size = end - start
        overlap_size = max(1, int(group_size * overlap_frac))

        # Core membership
        membership[start:end, g] = 1.0

        # Left overlap: blend with previous group
        if g > 0:
            ol_start = max(start - overlap_size, boundaries[g - 1])
            membership[ol_start:start, g] = 0.5
            membership[ol_start:start, g - 1] = np.where(
                membership[ol_start:start, g - 1] == 1.0,
                0.5,
                membership[ol_start:start, g - 1]
            )

        # Right overlap: blend with next group
        if g < num_groups - 1:
            ol_end = min(end + overlap_size, boundaries[g + 2] if g + 2 <= num_groups else n_keys)
            membership[end:ol_end, g] = 0.5
            membership[end:ol_end, g + 1] = np.where(
                membership[end:ol_end, g + 1] == 0.0,
                0.5,
                membership[end:ol_end, g + 1]
            )

    # Normalize rows to sum to 1
    row_sums = membership.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    membership /= row_sums

    return membership


def group_log_spaced(n_keys: int, num_groups: int) -> np.ndarray:
    """
    Increasingly large groups: more (smaller) groups at the head (high logits),
    fewer (larger) groups in the tail. Uses log-spaced boundaries.
    """
    num_groups = min(num_groups, n_keys)
    # Log-spaced boundaries: dense near 0 (head), sparse near n_keys (tail)
    boundaries = np.unique(np.round(
        np.logspace(0, np.log10(n_keys), num_groups + 1) - 1
    ).astype(int))
    boundaries[0] = 0
    boundaries[-1] = n_keys
    boundaries = np.sort(np.unique(boundaries))

    labels = np.zeros(n_keys, dtype=int)
    for g in range(len(boundaries) - 1):
        labels[boundaries[g]:boundaries[g + 1]] = g

    return labels


def group_quantile_weight(sorted_weights: np.ndarray, num_groups: int) -> np.ndarray:
    """
    Quantile-based: split so each group captures ~equal total weight mass.
    High-weight keys get more groups (finer resolution where it matters).
    """
    n = len(sorted_weights)
    num_groups = min(num_groups, n)
    if num_groups >= n:
        return np.arange(n)

    cumsum = np.cumsum(sorted_weights)
    total = cumsum[-1]
    if total < 1e-12:
        return group_equal_splits(n, num_groups)

    # Target cumulative weight boundaries
    targets = np.linspace(0, total, num_groups + 1)[1:-1]
    split_indices = np.searchsorted(cumsum, targets)
    split_indices = np.unique(np.clip(split_indices, 1, n - 1))

    labels = np.zeros(n, dtype=int)
    prev = 0
    for g, sp in enumerate(split_indices):
        labels[prev:sp] = g
        prev = sp
    labels[prev:] = len(split_indices)

    return labels


def group_variance_minimizing(sorted_weights: np.ndarray, num_groups: int) -> np.ndarray:
    """
    Greedy variance-minimizing: iteratively split the group with highest
    within-group weight variance at its point of maximum gap.
    Uses a heap for O(n log n) total instead of O(n * num_groups).
    """
    import heapq

    n = len(sorted_weights)
    num_groups = min(num_groups, n)
    if num_groups >= n:
        return np.arange(n)

    def segment_var(s, e):
        """Variance of sorted_weights[s:e]."""
        if e - s <= 1:
            return 0.0
        return float(np.var(sorted_weights[s:e]))

    def best_split(s, e):
        """Find the largest-gap split point in [s, e)."""
        gaps = np.abs(np.diff(sorted_weights[s:e]))
        return s + np.argmax(gaps) + 1

    # Max-heap: (-variance, start, end)
    active = {(0, n)}
    heap = [(-segment_var(0, n), 0, n)]

    while len(active) < num_groups and heap:
        neg_var, s, e = heapq.heappop(heap)
        if e - s <= 1:
            continue
        if (s, e) not in active:
            continue

        sp = best_split(s, e)
        active.discard((s, e))
        active.add((s, sp))
        active.add((sp, e))
        heapq.heappush(heap, (-segment_var(s, sp), s, sp))
        heapq.heappush(heap, (-segment_var(sp, e), sp, e))

    labels = np.zeros(n, dtype=int)
    for g, (s, e) in enumerate(sorted(active)):
        labels[s:e] = g

    return labels


# ============================================================================
# ATTENTION APPROXIMATION USING GROUPING
# ============================================================================

def grouped_attention(
    logits: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    num_groups: int,
    method: str = 'equal',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate attention by grouping sorted keys and assigning group-mean weights.

    Args:
        logits: [n_keys] pre-softmax scores
        values: [n_keys, head_dim] value vectors
        weights: [n_keys] true softmax weights (used to compute group means)
        num_groups: number of groups (the "budget" analog)
        method: grouping method name

    Returns:
        approx_weights: [n_keys] approximate weights
        approx_output: [head_dim] approximate attention output
    """
    n_keys = len(logits)
    num_groups = max(1, min(num_groups, n_keys))

    # Sort by descending logit
    sorted_indices = np.argsort(logits)[::-1]
    sorted_weights = weights[sorted_indices]

    # Special case: overlap method returns a membership matrix
    if method == 'overlap':
        return _grouped_attention_overlap(
            sorted_indices, sorted_weights, values, weights, num_groups
        )

    # Get group labels for sorted keys
    if method == 'equal':
        labels = group_equal_splits(n_keys, num_groups)
    elif method == 'kmeans':
        labels = group_kmeans_1d(sorted_weights, num_groups)
    elif method == 'threshold':
        labels = group_threshold_merging(sorted_weights, num_groups)
    elif method == 'log_spaced':
        labels = group_log_spaced(n_keys, num_groups)
    elif method == 'quantile':
        labels = group_quantile_weight(sorted_weights, num_groups)
    elif method == 'variance':
        labels = group_variance_minimizing(sorted_weights, num_groups)
    else:
        raise ValueError(f"Unknown grouping method: {method}")

    # Assign each key the mean weight of its group
    approx_weights = np.zeros(n_keys, dtype=np.float64)
    unique_labels = np.unique(labels)
    for g in unique_labels:
        mask = labels == g
        group_indices = sorted_indices[mask]
        mean_w = np.mean(weights[group_indices])
        approx_weights[group_indices] = mean_w

    approx_output = approx_weights @ values
    return approx_weights, approx_output


def _grouped_attention_overlap(
    sorted_indices: np.ndarray,
    sorted_weights: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    num_groups: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Overlap grouping: keys contribute to multiple groups via soft membership."""
    n_keys = len(sorted_indices)
    membership = group_equal_splits_overlap(n_keys, num_groups)  # [n_keys, G]

    # Compute group-mean weights: for each group, weighted average of member weights
    # membership[i, g] = how much key i belongs to group g
    group_weight_sums = membership.T @ sorted_weights      # [G]
    group_membership_sums = membership.sum(axis=0)          # [G]
    group_membership_sums[group_membership_sums == 0] = 1.0
    group_means = group_weight_sums / group_membership_sums  # [G]

    # Each key's approximate weight = weighted average of its groups' means
    approx_sorted_weights = membership @ group_means  # [n_keys]

    # Map back to original indices
    approx_weights = np.zeros(n_keys, dtype=np.float64)
    approx_weights[sorted_indices] = approx_sorted_weights

    approx_output = approx_weights @ values
    return approx_weights, approx_output


# All methods with display names
GROUPING_METHODS = {
    'equal': 'Equal Splits',
    'kmeans': '1D K-Means',
    'threshold': 'Threshold Merging',
    'overlap': 'Equal Splits + Overlap',
    'log_spaced': 'Log-Spaced (Head-Dense)',
    'quantile': 'Quantile (Equal Mass)',
    'variance': 'Variance Minimizing',
}
