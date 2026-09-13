"""Rank products with distinct reference observations.

This module evaluates a population marginal-midrank covariance target. The
expectation identity requires iid rows within independent clusters and nuisance
functions fitted on independent data. It does not validate arbitrary panel
dependence or the earlier within-period standardized-rank target.
"""
from __future__ import annotations

import numpy as np


def _twice_peer_ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    below = np.cumsum(counts) - counts
    return (2 * below + counts - 1)[inverse].astype(np.int64)


def _joint_comparison_counts(x, y):
    """Sum four times the product of the two mid-comparisons, excluding self.

    A Fenwick tree counts preceding x groups at each y rank. Equal x/y groups
    are handled explicitly, giving exact integer counts in O(n log n) time.
    """
    n = len(x)
    _, yr = np.unique(y, return_inverse=True)
    yr = yr + 1
    tree = np.zeros(int(yr.max()) + 1, dtype=np.int64)
    out = np.empty(n, dtype=np.int64)

    def prefix(pos):
        total = 0
        while pos:
            total += int(tree[pos])
            pos -= pos & -pos
        return total

    def insert(pos):
        while pos < len(tree):
            tree[pos] += 1
            pos += pos & -pos

    order = np.argsort(x, kind="stable")
    start = 0
    while start < n:
        end = start + 1
        while end < n and x[order[end]] == x[order[start]]:
            end += 1
        indices = order[start:end]
        levels, inverse, counts = np.unique(yr[indices], return_inverse=True, return_counts=True)
        below_same_x = np.cumsum(counts) - counts
        for position, index in enumerate(indices):
            ycode = int(yr[index])
            lower_both = prefix(ycode - 1)
            lower_x_equal_y = prefix(ycode) - lower_both
            group = inverse[position]
            out[index] = (4 * lower_both + 2 * lower_x_equal_y
                          + 2 * int(below_same_x[group]) + int(counts[group]) - 1)
        for index in indices:
            insert(int(yr[index]))
        start = end
    return out


def peer_rank_products(x, y, mean_x, mean_y):
    """Return naive and distinct-reference residual products for one cluster.

    mean_x/mean_y are fitted conditional means of population marginal mid-CDF
    transforms, not conditional CDFs evaluated at x/y. Their arrays must not be
    fitted with this cluster's outcomes. This API cannot verify that provenance.
    Ties receive half credit; all computations refer to input binary64 values.
    """
    arrays = [np.asarray(v, dtype=float) for v in (x, y, mean_x, mean_y)]
    x, y, mean_x, mean_y = arrays
    if any(v.ndim != 1 for v in arrays) or any(v.shape != x.shape for v in arrays):
        raise ValueError("all inputs must be aligned one-dimensional arrays")
    if len(x) < 3:
        raise ValueError("distinct references require at least three observations")
    if not all(np.isfinite(v).all() for v in arrays):
        raise ValueError("all inputs must be finite")
    if np.any((mean_x < 0) | (mean_x > 1) | (mean_y < 0) | (mean_y > 1)):
        raise ValueError("conditional marginal-midrank means must lie in [0,1]")
    # This explicit limit also prevents int64 overflow in products below.
    if len(x) > 1_000_000_000:
        raise ValueError("cluster too large for exact integer comparison products")
    k = len(x) - 1
    rx = _twice_peer_ranks(x)
    ry = _twice_peer_ranks(y)
    diagonal = _joint_comparison_counts(x, y)
    ux, uy = rx / (2.0 * k), ry / (2.0 * k)
    cross_product = (rx * ry - diagonal) / (4.0 * k * (k - 1))
    naive = (ux - mean_x) * (uy - mean_y)
    correction = cross_product - ux * uy
    corrected = naive + correction
    if np.max(np.abs(corrected)) > 1 + 1e-12:
        raise ArithmeticError("distinct-reference bounded-product identity failed")
    return {"forecast_midrank": ux, "outcome_midrank": uy,
            "joint_comparison_count_times_four": diagonal,
            "distinct_reference_product": cross_product,
            "naive_residual_product": naive,
            "corrected_residual_product": corrected,
            "reference_correction": correction}
