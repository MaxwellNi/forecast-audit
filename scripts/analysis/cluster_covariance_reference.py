"""Reference operations for a separately specified panel audit.

This module does not change the historical estimators or reproduce their paper
numbers. It provides target-isolated nuisance fitting, observation-centred
cluster standard errors, and explicitly forward-looking return labels. These
operations alone do not establish test validity: nuisance approximation,
cluster dependence, moments, and any selection rule still require justification.

Only NumPy and the Python standard library are required.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _labels(values: Sequence, n: int, name: str) -> tuple[np.ndarray, int]:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != n:
        raise ValueError(f"{name} must have one label for each observation")
    if array.dtype.kind in "fc" and not np.isfinite(array).all():
        raise ValueError(f"{name} contains missing or non-finite labels")
    if array.dtype.kind in "mM" and np.isnat(array).any():
        raise ValueError(f"{name} contains missing labels")
    if array.dtype.kind == "O" and any(
        value is None
        or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
        for value in array
    ):
        raise ValueError(f"{name} contains missing labels")
    try:
        unique, inverse = np.unique(array, return_inverse=True)
    except TypeError as exc:
        raise ValueError(f"{name} must use mutually comparable labels") from exc
    return inverse, len(unique)


def crossfit_additive_nuisance(
    values: Sequence[float],
    groups: Sequence[Sequence],
    folds: Sequence,
    *,
    cluster_ids: Sequence | None = None,
    max_sweeps: int = 1000,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Predict an additive group nuisance without using its evaluation targets.

    ``groups`` is a sequence of grouping vectors, for example
    ``[entity_ids, baseline_bin_ids]``. For each evaluation fold, fit an intercept
    and all group effects using only the complementary training observations.
    Alternating group-mean updates use training residuals throughout; the
    evaluation rows are used only for prediction. An empty group sequence fits
    training intercepts alone. Return nuisance predictions in input row order;
    residuals are ``np.asarray(values) - predictions``.

    Group effects are centred over the training observations. An unseen group
    level receives effect zero, so its prediction uses the training intercept
    plus any other known group effects. Bin labels must be supplied explicitly;
    any target-dependent construction of those labels must itself respect the
    folds. All target values must be finite.

    ``folds`` must contain at least two nonempty folds. Providing ``cluster_ids``
    verifies that every cluster belongs entirely to one fold. This checks the
    partition, not independence between clusters. Leaving it unspecified permits
    observation folds without asserting that they support cluster inference.
    Nonconvergence raises ``RuntimeError`` rather than returning an unchecked fit.
    """
    y = _vector(values, "values")
    n = len(y)
    if n < 2 or not np.isfinite(y).all():
        raise ValueError("values must contain at least two finite observations")
    if (
        isinstance(max_sweeps, (bool, np.bool_))
        or not isinstance(max_sweeps, (int, np.integer))
        or max_sweeps < 1
    ):
        raise ValueError("max_sweeps must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")

    fold_codes, n_folds = _labels(folds, n, "folds")
    if n_folds < 2:
        raise ValueError("at least two nonempty folds are required")
    encoded = [_labels(group, n, f"groups[{j}]") for j, group in enumerate(groups)]
    if cluster_ids is not None:
        clusters, _ = _labels(cluster_ids, n, "cluster_ids")
        order = np.argsort(clusters, kind="stable")
        same_cluster = np.diff(clusters[order]) == 0
        different_fold = np.diff(fold_codes[order]) != 0
        if np.any(same_cluster & different_fold):
            raise ValueError("each cluster must be assigned wholly to one fold")

    predictions = np.empty(n, dtype=float)
    for fold in range(n_folds):
        training = fold_codes != fold
        evaluation = ~training
        target = y[training]
        intercept = float(target.mean())
        residual = target - intercept
        coefficients = [np.zeros(k, dtype=float) for _, k in encoded]
        training_codes = [codes[training] for codes, _ in encoded]
        counts = [
            np.bincount(codes, minlength=k)
            for codes, (_, k) in zip(training_codes, encoded)
        ]
        threshold = tolerance * (1.0 + float(np.max(np.abs(target))))

        for _ in range(max_sweeps):
            previous = residual.copy()
            for block, codes in enumerate(training_codes):
                partial_residual = residual + coefficients[block][codes]
                count = counts[block]
                observed = count > 0
                updated = np.zeros_like(coefficients[block])
                sums = np.bincount(codes, weights=partial_residual, minlength=len(count))
                updated[observed] = sums[observed] / count[observed]
                residual = partial_residual - updated[codes]

                # Centring preserves training fitted values and defines the
                # intercept used when an evaluation category was unseen.
                centre = float(np.dot(count, updated) / len(target))
                updated[observed] -= centre
                intercept += centre
                coefficients[block] = updated

            if np.max(np.abs(residual - previous)) <= threshold:
                break
        else:
            raise RuntimeError(f"additive fit did not converge for fold {fold}")

        fitted = np.full(int(evaluation.sum()), intercept)
        for (codes, _), coefficient in zip(encoded, coefficients):
            fitted += coefficient[codes[evaluation]]
        if not np.isfinite(fitted).all():
            raise ValueError("nuisance predictions are not finite")
        predictions[evaluation] = fitted
    return predictions


def cluster_standard_error(products: Sequence[float], cluster_ids: Sequence) -> float:
    """Return the cluster standard error of the observation-weighted mean.

    With ``S_g = sum_{i in g} products[i]`` and ``n_g`` the cluster size,
    the variance estimate is ``sum_g (S_g - n_g * products.mean())**2 / n**2``.
    There is no finite-cluster degrees-of-freedom correction. At least two
    nonempty clusters and finite products are required. A degenerate estimate
    returns zero; ``cluster_t_statistic`` explicitly refuses to divide by it.
    """
    s = _vector(products, "products")
    if len(s) < 2 or not np.isfinite(s).all():
        raise ValueError("products must contain at least two finite observations")
    cluster, n_clusters = _labels(cluster_ids, len(s), "cluster_ids")
    if n_clusters < 2:
        raise ValueError("at least two nonempty clusters are required")
    if np.all(s == s[0]):
        return 0.0
    # Summing centred observations is algebraically S_g - n_g * mean(s),
    # and avoids cancellation between two large uncentred cluster totals.
    centred = s - s.mean()
    sums = np.bincount(cluster, weights=centred, minlength=n_clusters)
    variance = float(np.dot(sums, sums) / len(s) ** 2)
    if not np.isfinite(variance):
        raise ValueError("cluster variance is not finite")
    return float(np.sqrt(variance))


def cluster_t_statistic(products: Sequence[float], cluster_ids: Sequence) -> float:
    """Studentise the observation mean, with an explicit zero-scale guard.

    The returned statistic has no automatic normal or Student-t guarantee.
    Inference also needs a justified sampling model and negligible bias.
    """
    s = _vector(products, "products")
    standard_error = cluster_standard_error(s, cluster_ids)
    roundoff_scale = 8.0 * np.finfo(float).eps * float(np.max(np.abs(s)))
    if standard_error <= roundoff_scale:
        raise ValueError("cluster standard error is zero at numerical precision; inference is undefined")
    statistic = float(s.mean() / standard_error)
    if not np.isfinite(statistic):
        raise ValueError("cluster statistic is not finite")
    return statistic


def future_compound_return(monthly_returns: Sequence[float], horizon: int) -> np.ndarray:
    """Compute target[t] = product(1 + r[t+1:t+horizon+1]) - 1.

    Input must be one complete chronological monthly grid for one entity.
    Reindex absent calendar months to NaN before calling; this function cannot
    infer a missing calendar month from a compressed array. The current month's
    return is excluded. A missing future month, or an incomplete end window,
    makes that target NaN. NaN at the current month does not by itself invalidate
    the subsequent fully observed window. Inputs are simple returns, in decimal
    units; finite returns below -1 and infinities are rejected.
    """
    returns = _vector(monthly_returns, "monthly_returns")
    if (
        isinstance(horizon, (bool, np.bool_))
        or not isinstance(horizon, (int, np.integer))
        or horizon < 1
    ):
        raise ValueError("horizon must be a positive integer")
    if np.isinf(returns).any() or np.any(returns < -1):
        raise ValueError("returns must be NaN or finite simple returns at least -1")
    targets = np.full(len(returns), np.nan)
    for t in range(max(0, len(returns) - horizon)):
        window = returns[t + 1 : t + horizon + 1]
        if np.isnan(window).any():
            continue
        with np.errstate(over="raise", invalid="raise"):
            targets[t] = np.prod(1.0 + window) - 1.0
    return targets
