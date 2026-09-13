"""Check how temporal fitting and peer sampling change expected audit scores.

Run from the repository root:
    python scripts/analysis/panel_design_checks.py

All examples are synthetic. Only NumPy and the Python standard library are
used. The output checks score expectations; it does not estimate p-values,
prove a limiting distribution, or certify a panel application's calibration.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def _fold_labels(folds, n):
    folds = np.asarray(folds)
    if n < 2 or folds.shape != (n,):
        raise ValueError("folds must contain one label per row, with at least two rows")
    if not np.issubdtype(folds.dtype, np.integer):
        raise ValueError("fold labels must be integers")
    if np.any(folds[1:] < folds[:-1]):
        raise ValueError("fold labels must follow chronological row order")
    if len(np.unique(folds)) < 2:
        raise ValueError("at least two nonempty folds are needed")
    return folds


def _design(weights, folds):
    """Validate deterministic causal weights and chronological fold labels."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weights must be a square matrix")
    folds = _fold_labels(folds, weights.shape[0])
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite")
    if np.any(np.triu(weights) != 0):
        raise ValueError("forecasts may use strictly earlier rows only")
    return weights, folds


def _sets(folds, label):
    current = np.flatnonzero(folds == label)
    earlier = np.flatnonzero(folds < label)
    later = np.flatnonzero(folds > label)
    return current, earlier, later


def feedback_expectation(weights, folds, variance=1.0):
    """Exact expected sum of products after fitting both means on other folds.

    Model: y_r = alpha + epsilon_r and x_r = sum_j weights[r,j] y_j + eta_r.
    Innovations have a common positive variance, are mutually uncorrelated,
    and are uncorrelated with alpha and every eta_r. All variables are square
    integrable. Observation times, folds, and weights are fixed.

    The returned terms isolate the earlier-outcome route, later-forecast
    route, and the product of the two nuisance fitting errors.
    """
    weights, folds = _design(weights, folds)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("variance must be finite and strictly positive")
    terms = []
    for label in np.unique(folds):
        current, earlier, later = _sets(folds, label)
        outside = np.concatenate((earlier, later))
        m = len(outside)
        first = -variance * weights[np.ix_(current, earlier)].sum() / m
        second = -variance * weights[np.ix_(later, current)].sum() / m
        product = variance * len(current) * weights[np.ix_(outside, outside)].sum() / m**2
        terms.append({
            "fold": int(label),
            "earlier_outcome_term": float(first),
            "later_forecast_term": float(second),
            "fitting_error_product": float(product),
            "expected_numerator": float(first + second + product),
        })
    return {
        "expected_numerator": float(sum(t["expected_numerator"] for t in terms)),
        "fold_terms": terms,
    }


def time_directed_residuals(forecast, outcome, folds, evaluation_fold):
    """Use earlier forecast means and later outcome means for one entity.

    A nonempty training set is required on both sides. This function does
    not infer chronology from dates, fit baseline controls, transform ranks,
    or assess the innovation assumptions needed for unbiasedness.
    """
    forecast = np.asarray(forecast, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    folds = np.asarray(folds)
    if forecast.ndim != 1 or forecast.shape != outcome.shape:
        raise ValueError("forecast and outcome must be equally sized vectors")
    folds = _fold_labels(folds, len(forecast))
    if not np.all(np.isfinite(forecast)) or not np.all(np.isfinite(outcome)):
        raise ValueError("forecasts and outcomes must be finite")
    current, earlier, later = _sets(folds, evaluation_fold)
    if not len(current):
        raise ValueError("the evaluation fold must be nonempty")
    if not len(earlier) or not len(later):
        raise ValueError("both earlier and later training rows are required")
    return (
        current,
        forecast[current] - forecast[earlier].mean(),
        outcome[current] - outcome[later].mean(),
    )


def dense_score_expectation(weights, folds, covariance, time_directed=False):
    """Independent dense covariance calculation for a linear forecast.

    Calculates the same raw-score expectation by linear residual operators.
    In the time-directed case only folds having both training sides are used.
    General innovation covariance is allowed here to expose departures from
    the uncorrelated-innovation premise of feedback_expectation.
    """
    weights, folds = _design(weights, folds)
    covariance = np.asarray(covariance, dtype=float)
    n = len(folds)
    if covariance.shape != (n, n) or not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be a finite matrix matching the rows")
    if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    if np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError("covariance must be positive semidefinite")
    eye = np.eye(n)
    expected, evaluated = 0.0, 0
    for label in np.unique(folds):
        current, earlier, later = _sets(folds, label)
        if time_directed:
            if not len(earlier) or not len(later):
                continue
            forecast_training, outcome_training = earlier, later
        else:
            forecast_training = outcome_training = np.concatenate((earlier, later))
        left = weights[current] - weights[forecast_training].mean(axis=0)
        right = eye[current] - eye[outcome_training].mean(axis=0)
        expected += np.einsum("ij,jk,ik->", left, covariance, right)
        evaluated += len(current)
    if not evaluated:
        raise ValueError("time-directed evaluation needs nonempty earlier and later folds")
    return float(expected)


def rank_product_scores(forecast, outcome, forecast_means, outcome_means):
    """Shared and distinct-reference scores, using leave-self-out midranks."""
    arrays = [np.asarray(x, dtype=float)
              for x in (forecast, outcome, forecast_means, outcome_means)]
    forecast, outcome, forecast_means, outcome_means = arrays
    n = len(forecast) if forecast.ndim == 1 else 0
    if n < 3 or any(x.shape != (n,) or not np.all(np.isfinite(x)) for x in arrays):
        raise ValueError("four finite vectors of equal length at least three are required")
    a = (forecast[:, None] > forecast[None, :]).astype(float)
    a += 0.5 * (forecast[:, None] == forecast[None, :])
    b = (outcome[:, None] > outcome[None, :]).astype(float)
    b += 0.5 * (outcome[:, None] == outcome[None, :])
    np.fill_diagonal(a, 0)
    np.fill_diagonal(b, 0)
    u, v = a.sum(axis=1) / (n-1), b.sum(axis=1) / (n-1)
    shared = (u-forecast_means) * (v-outcome_means)
    distinct = shared + (u*v - (a*b).sum(axis=1)/(n-1)) / (n-2)
    return shared, distinct


def fixed_bernoulli_check(opposite=False):
    """Exhaustively enumerate four independent heterogeneous fixed entities.

    The two channel vectors are independent conditional on these fixed
    probabilities. This example includes ties and uses exact entity means.
    """
    px = np.array([0.1, 0.3, 0.7, 0.9])
    py = px[::-1] if opposite else px
    n, k = len(px), len(px)-1
    # For binary variables, the expected midrank comparison is
    # P(X_j < X_i) + 0.5 P(X_j = X_i).
    p = px[:, None]*(1-px[None, :])
    p += 0.5*(px[:, None]*px[None, :] + (1-px[:, None])*(1-px[None, :]))
    q = py[:, None]*(1-py[None, :])
    q += 0.5*(py[:, None]*py[None, :] + (1-py[:, None])*(1-py[None, :]))
    np.fill_diagonal(p, 0)
    np.fill_diagonal(q, 0)
    f, g = p.sum(axis=1)/k, q.sum(axis=1)/k
    predicted = (f*g-(p*q).sum(axis=1)/k)/(k-1)
    shared_mean, distinct_mean, mass = np.zeros(n), np.zeros(n), 0.0
    states = np.array(list(itertools.product((0.0, 1.0), repeat=n)))
    for x, y in itertools.product(states, repeat=2):
        probability = np.prod(np.where(x, px, 1-px))*np.prod(np.where(y, py, 1-py))
        shared, distinct = rank_product_scores(x, y, f, g)
        shared_mean += probability*shared
        distinct_mean += probability*distinct
        mass += probability
    return {
        "direction": "opposite" if opposite else "same",
        "enumerated_probability": float(mass),
        "shared_expected_score": float(shared_mean.mean()),
        "distinct_expected_score": float(distinct_mean.mean()),
        "predicted_distinct_score": float(predicted.mean()),
        "maximum_identity_error": float(max(abs(shared_mean).max(),
                                             abs(distinct_mean-predicted).max())),
    }


def resampled_bernoulli_check():
    """Exhaustively check three i.i.d. peers under a conditional null.

    Z is Bernoulli(1/2). Given Z, the two channels are independent Bernoulli
    variables with success probability 0.2 or 0.8. The conditional population
    midrank means are known. The conditional residual covariance is zero.
    """
    states = []
    for z, x, y in itertools.product((0, 1), repeat=3):
        probability = 0.2 if z == 0 else 0.8
        mass = 0.5*(probability if x else 1-probability)*(probability if y else 1-probability)
        states.append((z, x, y, mass, 0.25+0.5*probability))
    gamma = 0.0
    for first, second in itertools.product(states, repeat=2):
        a = float(second[1] < first[1]) + 0.5*float(second[1] == first[1])
        b = float(second[2] < first[2]) + 0.5*float(second[2] == first[2])
        gamma += first[3]*second[3]*(a*b-(0.25+0.5*first[1])*(0.25+0.5*first[2]))
    shared_mean, distinct_mean, mass = 0.0, 0.0, 0.0
    for rows in itertools.product(states, repeat=3):
        probability = np.prod([r[3] for r in rows])
        shared, distinct = rank_product_scores(
            [r[1] for r in rows], [r[2] for r in rows],
            [r[4] for r in rows], [r[4] for r in rows])
        shared_mean += probability*shared.mean()
        distinct_mean += probability*distinct.mean()
        mass += probability
    return {
        "entities_per_group": 3,
        "enumerated_probability": float(mass),
        "reuse_term": float(gamma),
        "shared_expected_score": float(shared_mean),
        "distinct_expected_score": float(distinct_mean),
        "predicted_shared_score": float(gamma/2),
        "maximum_identity_error": float(max(abs(shared_mean-gamma/2), abs(distinct_mean))),
    }


def run_checks():
    """Produce deterministic demonstrations and independent identity checks."""
    rng = np.random.default_rng(1729)
    errors, temporal_errors = [], []
    for _ in range(40):
        n = int(rng.integers(6, 24))
        folds = np.repeat(np.arange(3), [2, n-4, 2])
        weights = np.tril(rng.normal(size=(n, n)), -1)
        formula = feedback_expectation(weights, folds)["expected_numerator"]
        errors.append(abs(formula-dense_score_expectation(weights, folds, np.eye(n))))
        temporal_errors.append(abs(dense_score_expectation(weights, folds, np.eye(n), True)))
    balanced = []
    for rows_per_fold in (1, 2, 3, 5):
        folds = np.repeat(np.arange(5), rows_per_fold)
        weights = np.eye(len(folds), k=-1)
        observed = feedback_expectation(weights, folds)["expected_numerator"]
        predicted = 5*(rows_per_fold-2)/(4*rows_per_fold)
        balanced.append({
            "folds": 5, "rows_per_fold": rows_per_fold, "lookback": 1,
            "expected_numerator": observed, "closed_form_numerator": predicted,
        })
    folds = np.repeat(np.arange(3), 2)
    weights = np.eye(6, k=-1)
    covariance = 0.8**abs(np.subtract.outer(np.arange(6), np.arange(6)))
    ar_expected = dense_score_expectation(weights, folds, covariance, True)
    fixed = [fixed_bernoulli_check(False), fixed_bernoulli_check(True)]
    resampled = resampled_bernoulli_check()
    maximum_error = max(
        *errors, *temporal_errors,
        *(abs(x["expected_numerator"]-x["closed_form_numerator"]) for x in balanced),
        *(x["maximum_identity_error"] for x in fixed),
        resampled["maximum_identity_error"])
    if maximum_error > 1e-11:
        raise ArithmeticError("an expectation identity failed its numerical check")
    return {
        "scope": "Synthetic score-expectation checks; no p-value or calibration guarantee.",
        "dependencies": ["Python standard library", "NumPy"],
        "data_dependencies": "None; every example is generated or exhaustively enumerated here.",
        "random_causal_designs": {
            "cases": len(errors),
            "maximum_formula_vs_dense_error": float(max(errors)),
            "maximum_time_directed_mean_under_identity_covariance": float(max(temporal_errors)),
        },
        "balanced_lag_one_examples": balanced,
        "serial_dependence_counterexample": {
            "rows": 6, "folds": 3, "rows_per_fold": 2, "innovation_ar_coefficient": 0.8,
            "expected_time_directed_numerator": ar_expected,
            "interpretation": "Serial covariance violates the uncorrelated-innovation premise.",
        },
        "fixed_peers": fixed,
        "independently_resampled_peers": resampled,
        "maximum_identity_error": float(maximum_error),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=(
        Path(__file__).resolve().parents[2] / "results/design_identity_checks/design_checks.json"))
    args = parser.parse_args()
    result = run_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
