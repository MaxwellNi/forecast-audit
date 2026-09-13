"""A split-sample bound for a weighted product of conditional means.

The category masses must be known, not estimated on the supplied observations.
Each stream contains independent range-one observations within each category;
the two streams are independent conditional on their focal category assignments.
The reference category labels remain unconditioned, preserving the marginal
reference law.
For the rank audit, a stream observation is a focal/reference comparison minus
an independently trained fitted conditional mean. The bound requires the declared sampling and metadata conditions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def aggregate_upper_bound(
    masses: Sequence[float],
    mean_a: Sequence[float],
    mean_b: Sequence[float],
    count_a: Sequence[int],
    count_b: Sequence[int],
    delta: float,
    *,
    absent_upper: Sequence[float] | None = None,
) -> dict[str, float | int | str]:
    """Return a (1-delta) upper bound on sum_c p_c a_c b_c.

    ``mean_a`` and ``mean_b`` are empirical residual-comparison means from
    independent halves; their counts may be random. Conditioning on the focal category counts
    leaves the within-category observations independent. Categories absent in
    either half use a deterministic product upper bound (default one).

    No inference is made that these inputs represent valid sampling. In
    particular, a plug-in estimate of ``masses`` is not authorized by the bound.
    """
    length = len(masses)
    if not all(len(v) == length for v in (mean_a, mean_b, count_a, count_b)):
        raise ValueError("All category arrays must have the same length.")
    if not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one.")
    if length == 0 or any(p < 0 or not math.isfinite(p) for p in masses):
        raise ValueError("Known category masses must be finite and nonnegative.")
    if not math.isclose(sum(masses), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Known category masses must sum to one.")
    if any(n < 0 or int(n) != n for n in list(count_a) + list(count_b)):
        raise ValueError("Category counts must be nonnegative integers.")
    if absent_upper is None:
        absent_upper = [1.0] * length
    if len(absent_upper) != length or any(not math.isfinite(q) for q in absent_upper):
        raise ValueError("A finite deterministic upper limit is required per category.")
    selected = [c for c in range(length) if masses[c] > 0 and count_a[c] and count_b[c]]
    omitted = [c for c in range(length) if masses[c] > 0 and c not in selected]
    if any(not math.isfinite(v[c]) or abs(v[c]) > 1 + 1e-12 for c in selected for v in (mean_a, mean_b)):
        raise ValueError("Observed residual-comparison means must be in [-1, 1].")
    x = math.log(3.0 / delta)
    center = sum(masses[c] * mean_a[c] * mean_b[c] for c in selected)
    radius_a = math.sqrt(x / 2 * sum(masses[c] ** 2 * mean_b[c] ** 2 / count_a[c] for c in selected))
    radius_b = math.sqrt(x / 2 * sum(masses[c] ** 2 * mean_a[c] ** 2 / count_b[c] for c in selected))
    products = [masses[c] / (4 * math.sqrt(count_a[c] * count_b[c])) for c in selected]
    radius_product = math.sqrt(2 * x * sum(d * d for d in products)) + x * max(products, default=0)
    absent_allowance = sum(masses[c] * absent_upper[c] for c in omitted)
    return {
        "center": center,
        "linear_a_radius": radius_a,
        "linear_b_radius": radius_b,
        "product_radius": radius_product,
        "absent_allowance": absent_allowance,
        "upper": center + radius_a + radius_b + radius_product + absent_allowance,
        "delta": delta,
        "categories_used": len(selected),
        "categories_absent": len(omitted),
        "information_contract": "category probabilities known exactly; independent streams",
    }
