"""Conditional finite-sample bounds for honestly held-out cluster products.

This is separate from the normal-approximation panel audit. Its guarantee
requires independent evaluation clusters conditional on the training/design
information, certified score bounds, and an externally justified one-sided
bias envelope. The code cannot verify those statistical assumptions. Ordinary
cross-fitted cluster scores do not automatically meet this independence rule.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np


@dataclass(frozen=True)
class BoundedCertificate:
    weighted_mean: float
    lower_bound: float
    concentration_radius: float
    bias_upper: float
    envelope_failure_probability: float
    alpha: float
    p_one_sided: float
    reject: bool
    effective_clusters: float
    positive_if_bias_below: float
    inference_status: str = "conditional_on_independence_bounds_and_bias_envelope"

    def to_dict(self):
        return asdict(self)


def _vector(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def _exact_sum_products(first, second=None):
    """Sum finite binary floats/products exactly, using dyadic integers."""
    numerator = 0
    exponent = 0
    for index, value in enumerate(first):
        term, denominator = float(value).as_integer_ratio()
        power = denominator.bit_length()-1
        if second is not None:
            other, other_denominator = float(second[index]).as_integer_ratio()
            term *= other
            power += other_denominator.bit_length()-1
        if power > exponent:
            numerator <<= power-exponent
            exponent = power
        numerator += term << (exponent-power)
    return Fraction(numerator, 1 << exponent)


def _float_down(value):
    """Convert an exact rational to a finite float no larger than it."""
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("certificate is outside the finite floating-point range") from error
    if not math.isfinite(result):
        raise ValueError("certificate is outside the finite floating-point range")
    if Fraction.from_float(result) > value:
        result = math.nextafter(result, -math.inf)
    if not math.isfinite(result):
        raise ValueError("certificate is outside the finite floating-point range")
    return result


def _float_up_decimal(value):
    """Convert a Decimal upper bound to a finite float still above it."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("certificate is outside the finite floating-point range")
    if Decimal.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        raise ValueError("certificate is outside the finite floating-point range")
    return result


def _float_up_fraction(value):
    return -_float_down(-value)


@lru_cache(maxsize=256)
def _radius_up(variance_range, tail_budget):
    if not variance_range:
        return 0.0
    # Decimal ln/sqrt are correctly rounded to nearest. Moving by one Decimal
    # ULP in the conservative direction gives a rigorous enclosure. Algebraic
    # operations below explicitly use directed rounding as well.
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_FLOOR
        probability_lower = Decimal(tail_budget.numerator)/Decimal(tail_budget.denominator)
        log_lower = probability_lower.ln().next_minus()
        context.rounding = ROUND_CEILING
        logarithm_upper = -log_lower
        variance_upper = Decimal(variance_range.numerator)/Decimal(variance_range.denominator)
        square_upper = variance_upper*logarithm_upper/2
        radius_upper = square_upper.sqrt().next_plus()
        return _float_up_decimal(radius_upper)


def _pvalue_up(gap, variance_range, delta):
    if gap <= 0:
        return 1.0
    if not variance_range:
        return float(delta)
    exponent = 2*gap*gap/variance_range
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_FLOOR
        exponent_lower = Decimal(exponent.numerator)/Decimal(exponent.denominator)
        context.rounding = ROUND_CEILING
        # Negation is exact for this already-representable Decimal. exp is
        # correctly rounded; next_plus encloses its exact value from above.
        tail_upper = (-exponent_lower).exp().next_plus()
        delta_upper = Decimal(delta.numerator)/Decimal(delta.denominator)
        total_upper = delta_upper+tail_upper
        return min(1.0, _float_up_decimal(total_upper))


def bounded_mean_certificate(scores, lower, upper, *, bias_upper,
                             cluster_weights=None, alpha=0.05,
                             envelope_failure_probability=0.0):
    """Lower-bound theta when E[weighted score | H] - theta <= bias_upper.

    Conditional on H, scores must be independent; nonnegative cluster weights,
    score bounds, and bias_upper must be fixed. H may contain separate training
    and calibration data. A probabilistic bias envelope may fail with at most
    envelope_failure_probability, which is paid from alpha by a union bound.
    Bounds must hold almost surely, not merely contain the observed scores.

    Weights are normalized to sum to one. For a row-weighted target use the
    predetermined cluster sizes. With random sizes this requires conditioning
    on the sizes and justifying independence and the bias envelope there.
    Equal weights instead define an average-cluster target.
    """
    score = _vector(scores, "scores")
    low = np.broadcast_to(np.asarray(lower, float), score.shape)
    high = np.broadcast_to(np.asarray(upper, float), score.shape)
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high):
        raise ValueError("finite ordered bounds are required")
    if np.any(score < low) or np.any(score > high):
        raise ValueError("an observed score is outside the supplied bounds")
    if not np.isfinite(bias_upper):
        raise ValueError("bias_upper must be finite and externally justified")
    delta = float(envelope_failure_probability)
    if not 0 <= delta < alpha < 1:
        raise ValueError("require 0 <= envelope failure probability < alpha < 1")
    mass = np.ones(len(score)) if cluster_weights is None else _vector(cluster_weights, "cluster_weights")
    if mass.shape != score.shape or np.any(mass < 0) or np.max(mass) <= 0:
        raise ValueError("weights must be nonnegative, aligned, and not all zero")
    # Interpret every supplied IEEE float as an exact real number. Floating
    # normalization/dot products are unsafe when the score range is near zero:
    # even five copies of .1 can otherwise produce a certain false rejection.
    equal_mass = bool(np.all(mass == mass[0]))
    if equal_mass:
        total_mass = Fraction(len(score))
        sum_squared_mass = total_mass
        mean_exact = _exact_sum_products(score)/total_mass
        exact_mass = None
    else:
        total_mass = _exact_sum_products(mass)
        sum_squared_mass = _exact_sum_products(mass, mass)
        mean_exact = _exact_sum_products(mass, score)/total_mass
        exact_mass = [Fraction.from_float(float(value)) for value in mass]
    if np.all(low == low[0]) and np.all(high == high[0]):
        exact_width = Fraction.from_float(float(high[0]))-Fraction.from_float(float(low[0]))
        variance_range = exact_width**2*sum_squared_mass/total_mass**2
    else:
        widths = [Fraction.from_float(float(hi))-Fraction.from_float(float(lo)) for lo, hi in zip(low, high)]
        if equal_mass:
            variance_range = sum((width**2 for width in widths), Fraction())/total_mass**2
        else:
            variance_range = sum(((weight*width)**2 for weight, width in zip(exact_mass, widths)), Fraction())/total_mass**2
    delta_exact = Fraction.from_float(delta)
    budget_exact = Fraction.from_float(float(alpha))-delta_exact
    gap = mean_exact-Fraction.from_float(float(bias_upper))
    radius = _radius_up(variance_range, budget_exact)
    lower_bound = _float_down(gap-Fraction.from_float(radius))
    sensitivity = _float_down(mean_exact-Fraction.from_float(radius))
    pvalue = _pvalue_up(gap, variance_range, delta_exact)
    mean = float(mean_exact)
    return BoundedCertificate(
        weighted_mean=mean, lower_bound=float(lower_bound),
        concentration_radius=radius, bias_upper=float(bias_upper),
        envelope_failure_probability=delta, alpha=float(alpha),
        p_one_sided=pvalue, reject=bool(lower_bound > 0),
        effective_clusters=float(total_mass**2/sum_squared_mass),
        positive_if_bias_below=sensitivity,
    )


def _clusters(cluster_ids, n):
    labels = np.asarray(cluster_ids)
    if labels.ndim != 1 or len(labels) != n:
        raise ValueError("cluster IDs must align with rows")
    if any(value is None or value != value for value in labels):
        raise ValueError("cluster IDs must be nonmissing")
    unique, inverse = np.unique(labels, return_inverse=True)
    return unique, inverse


def project_cluster_rms(fits, cluster_ids, radius=1.0):
    """Project each fitted vector to RMS <= radius using fits and IDs only.

    Columns represent resolutions. The map must be evaluated on predictions
    made using independent training data and evaluation controls only. Because
    projection uses the whole cluster, the oracle control sigma-field must
    also include those whole-cluster controls when using the bias identity.
    """
    array = np.asarray(fits, float)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[:, None]
    if array.ndim != 2 or not len(array) or not array.shape[1] or not np.isfinite(array).all():
        raise ValueError("fits must be a nonempty finite vector or matrix")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    _, inverse = _clusters(cluster_ids, len(array))
    result = array.copy()
    for group in np.unique(inverse):
        index = inverse == group
        for column in range(array.shape[1]):
            values = array[index, column]
            maximum = np.max(np.abs(values))
            if maximum == 0:
                continue
            scaled = values / maximum
            scaled_rms = np.sqrt(np.mean(scaled ** 2))
            if maximum > radius / scaled_rms:
                result[index, column] = scaled * (radius / scaled_rms)
            # Verify the *exact* squared norm of returned binary floats. A
            # nearest-rounded projection can otherwise land just outside the
            # ball. nextafter toward zero also works for subnormal radii.
            limit = Fraction.from_float(float(radius))**2*len(values)
            while _exact_sum_products(result[index, column], result[index, column]) > limit:
                result[index, column] = np.nextafter(result[index, column], 0.)
    return result[:, 0] if was_vector else result


def rank_cluster_products(forecast_rank, outcome_rank, forecast_fits,
                          outcome_fits, cluster_ids, resolution_weights):
    """Return bounded cluster means of extrapolated rank-residual products.

    Each observed rank vector must have RMS <= 1 within a cluster, as does
    the sample-SD rank convention, including ties and zero/singleton groups.
    Each supplied fit is projected to the unit RMS ball. The resulting score
    lies in [-4 ||w||_1, 4 ||w||_1]; no bound on individual tied ranks is used.
    The function verifies algebraic contracts, not honesty or independence.
    """
    x, y = _vector(forecast_rank, "forecast_rank"), _vector(outcome_rank, "outcome_rank")
    if x.shape != y.shape:
        raise ValueError("rank vectors must align")
    unique, inverse = _clusters(cluster_ids, len(x))
    weights = _vector(resolution_weights, "resolution_weights")
    exact_weights = [Fraction.from_float(float(value)) for value in weights]
    exact_total_weight = sum(exact_weights, Fraction())
    if abs(exact_total_weight-1) > Fraction.from_float(1e-12):
        raise ValueError("resolution weights must sum to one before exact normalization")
    exact_weights = [value/exact_total_weight for value in exact_weights]
    fx, fy = np.asarray(forecast_fits, float), np.asarray(outcome_fits, float)
    if fx.ndim == 1:
        fx = fx[:, None]
    if fy.ndim == 1:
        fy = fy[:, None]
    if fx.shape != (len(x), len(weights)) or fy.shape != fx.shape:
        raise ValueError("fit matrices must have one row per observation and one column per resolution")
    for group in range(len(unique)):
        index = inverse == group
        count = int(index.sum())
        if _exact_sum_products(x[index], x[index]) > count or _exact_sum_products(y[index], y[index]) > count:
            raise ValueError("observed vectors violate the within-cluster RMS <= 1 contract")
    fx = project_cluster_rms(fx, cluster_ids)
    fy = project_cluster_rms(fy, cluster_ids)
    count = np.bincount(inverse)
    scores = []
    for group in range(len(unique)):
        index = inverse == group
        a, b = x[index], y[index]
        observed_product = _exact_sum_products(a, b)
        combined = Fraction()
        for column, weight in enumerate(exact_weights):
            c, d = fx[index, column], fy[index, column]
            product = (observed_product-_exact_sum_products(a, d)
                       -_exact_sum_products(c, b)+_exact_sum_products(c, d))/int(count[group])
            combined += weight*product
        # Downward score rounding only decreases the mean, preserving every
        # valid one-sided upper bias envelope for the ideal product score.
        scores.append(_float_down(combined))
    scores = np.asarray(scores)
    l1 = sum((abs(weight) for weight in exact_weights), Fraction())
    bound = _float_up_fraction(4*l1)
    # The bound is mathematical; numerical overflow or contract failures stop.
    if not np.isfinite(scores).all() or np.any(np.abs(scores) > bound):
        raise ValueError("numerical score failed its deterministic bound")
    return {"cluster_ids": unique, "cluster_sizes": count,
            "cluster_scores": scores, "score_lower": -bound,
            "score_upper": bound, "resolution_weight_l1": float(l1),
            "cluster_score_rounding": "downward from exact rational products and normalized resolution weights",
            "projected_forecast_fits": fx, "projected_outcome_fits": fy}
