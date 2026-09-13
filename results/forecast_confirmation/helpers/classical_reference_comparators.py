"""Classical same-target comparators, with explicit finite-sample constants.

See REPORT.md for assumptions and derivations. No producer imports. Fits and
kernel range must be fixed independently of these iid evaluation observations.
A supplied bias allowance must cover the fitted target's upward bias with
failure probability at most delta. These APIs cannot verify that design.
"""
import math
import numpy as np
from scipy.special import logsumexp


def midcomparison(focal, reference):
    return (focal > reference) + .5 * (focal == reference)


def reference_u_with_deletions(v, w, f, g, block_size=128):
    """Order-three reference U and every delete-one U in O(N^2) time.

    f,g are the values of independently fixed functions at the evaluation rows.
    Pair comparisons are blocked, requiring O(N * block_size) memory. All
    delete-one statistics keep the same fits; they do not refit nuisances.
    """
    v, w, f, g = [np.asarray(x, float) for x in (v, w, f, g)]
    total = len(v)
    if total < 4 or any(x.shape != (total,) for x in (v, w, f, g)):
        raise ValueError('Matching vectors with at least four observations required')
    if not all(np.isfinite(x).all() for x in (v, w, f, g)):
        raise ValueError('Finite data required')
    if np.any((f < 0) | (f > 1) | (g < 0) | (g > 1)):
        raise ValueError('Fits must lie in [0,1]')
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError('Positive integer block size required')
    row_products, base_terms = np.empty(total), np.empty(total)
    column_changes = np.zeros(total)
    for start in range(0, total, block_size):
        stop = min(total, start + block_size)
        a = midcomparison(v[start:stop, None], v[None, :])
        b = midcomparison(w[start:stop, None], w[None, :])
        local = np.arange(stop - start)
        a[local, np.arange(start, stop)] = 0.
        b[local, np.arange(start, stop)] = 0.
        av, bw = a.sum(axis=1), b.sum(axis=1)
        joint = np.einsum('ij,ij->i', a, b)
        ff, gg = f[start:stop], g[start:stop]
        cross = ff * bw + gg * av
        row_products[start:stop] = ((av * bw - joint) / ((total - 1) * (total - 2))
                                    - cross / (total - 1) + ff * gg)
        # Numerator after reducing sample size, before removing each row/ref.
        base_terms[start:stop] = (av * bw - joint - (total - 3) * cross
                                  + (total - 2) * (total - 3) * ff * gg)
        column_changes += np.einsum('ij,i->j', b, -av + (total - 3) * ff)
        column_changes += np.einsum('ij,i->j', a, -bw + (total - 3) * gg)
        column_changes += 2 * np.einsum('ij,ij->j', a, b)
    deleted = ((base_terms.sum() - base_terms + column_changes)
               / ((total - 1) * (total - 2) * (total - 3)))
    mean = float(row_products.mean())
    np.testing.assert_allclose(deleted.mean(), mean, rtol=1e-10, atol=1e-12)
    return mean, deleted


def pair_u_with_deletions(kernel_matrix):
    """Order-two U and delete-one values from a symmetric kernel matrix.

    Diagonal values are ignored. The declared global range is supplied to the
    certificate separately; the observed range is not a valid replacement.
    """
    matrix = np.asarray(kernel_matrix, float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 3:
        raise ValueError('Square matrix with at least three observations required')
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=0., atol=1e-12):
        raise ValueError('Finite symmetric kernel matrix required')
    n = len(matrix)
    rows = matrix.sum(axis=1) - np.diag(matrix)
    all_pairs = rows.sum() / 2
    return float(all_pairs / math.comb(n, 2)), (all_pairs - rows) / math.comb(n - 1, 2)


def interaction_bounds(size, order, width):
    """M(U_size) and J(U_size), for a symmetric order-m kernel of width R."""
    return order * width / size, 2 * order * (order - 1) * width / (size - 1)


def mp2018_full_u_certificate(mean, delete_one_values, order, kernel_lower,
                              kernel_upper, bias_upper=0., delta=0., alpha=.05):
    """All-N corollary of Maurer-Pontil (2018), Eq.9 and Theorem 2.

    Uses all N observations, not a leave-one-out center and not an additional
    observation. REPORT.md proves the Efron-Stein monotonicity step needed to
    combine their results. This is a classical-bound composition, not the
    published Theorem 6 quoted verbatim.
    """
    deleted = np.asarray(delete_one_values, float)
    if deleted.ndim != 1 or not np.isfinite(deleted).all():
        raise ValueError('Finite delete-one values required')
    total = len(deleted)
    if not isinstance(order, (int, np.integer)) or order < 1 or total < order + 1:
        raise ValueError('Need N >= kernel order + 1')
    if not 0 <= delta < alpha < 1 or not np.isfinite([mean, kernel_lower, kernel_upper, bias_upper]).all():
        raise ValueError('Finite inputs and 0 <= delta < alpha < 1 required')
    width = kernel_upper - kernel_lower
    if width <= 0:
        raise ValueError('A positive declared kernel width is required')
    if not kernel_lower - 1e-12 <= mean <= kernel_upper + 1e-12:
        raise ValueError('Mean outside declared kernel range')
    if np.any(deleted < kernel_lower - 1e-12) or np.any(deleted > kernel_upper + 1e-12):
        raise ValueError('Delete-one mean outside declared kernel range')
    if not np.isclose(mean, deleted.mean(), rtol=1e-10, atol=1e-12):
        raise ValueError('Delete-one U means must average to the full U mean')
    small = total - 1
    m_small, j_small = interaction_bounds(small, order, width)
    m_full, j_full = interaction_bounds(total, order, width)
    vf = float(np.square(deleted - mean).sum())
    variance_proxy = small * vf / total
    coefficient = (np.sqrt(2 * small / total) * np.sqrt(2 * m_small ** 2 + 8 * j_small ** 2)
                   + 2 * m_full / 3 + j_full)
    x = np.log(2 / (alpha - delta))
    radius = np.sqrt(2 * variance_proxy * x) + coefficient * x
    margin = max(float(mean) - bias_upper, 0.)
    a = np.sqrt(2 * variance_proxy)
    root = 2 * margin / (a + np.sqrt(a * a + 4 * coefficient * margin)) if margin else 0.
    probability = min(1., delta + 2 * np.exp(-root * root))
    return dict(mean=float(mean), observations=total, order=int(order),
                jackknife_variance_proxy=variance_proxy, mp2018_vf=vf,
                linear_coefficient=float(coefficient), radius=float(radius),
                bias_upper=float(bias_upper), delta=float(delta), alpha=float(alpha),
                lower_bound=float(mean - bias_upper - radius), p=float(probability),
                reject=bool(probability < alpha), kernel_lower=float(kernel_lower),
                kernel_upper=float(kernel_upper),
                attribution='Derived all-N composition of Maurer-Pontil 2018 Eq.9 and Theorem 2')


def mp2018_published_theorem6(mean_on_first_n, delete_one_values, order, width,
                             failure_probability=.05):
    """Published Theorem 6 with exactly its n+1-observation accounting.

    The center uses the first n=N-1 observations; the variance estimator uses
    all N delete-one statistics. Returns only the evaluation radius.
    """
    deleted = np.asarray(delete_one_values, float)
    n = len(deleted) - 1
    if n < order or n < 2 or width <= 0 or not 0 < failure_probability < 1:
        raise ValueError('Invalid size, order, width or failure probability')
    if not np.isclose(mean_on_first_n, deleted[-1], rtol=1e-10, atol=1e-12):
        raise ValueError('Center must omit the last observation')
    vf = float(np.square(deleted - deleted.mean()).sum())
    a = order * width
    b = 2 * order * (order - 1) * width * n / (n - 1)
    x = np.log(2 / failure_probability)
    return float(np.sqrt(2 * vf * x) + (8 * a / 3 + 5 * b) * x / n)


def mixture_betting_pvalue(independent_scores, kernel_lower, kernel_upper,
                           bias_upper=0., delta=0., fractions=None):
    """Classical one-sided fixed-mixture betting test on independent kernels.

    This test must not receive dependent row products or overlapping triples.
    The grid must be fixed before evaluation; no best-bet post-selection.
    Uses a conservative delta + 1/E conversion for the validation event.
    """
    values = np.asarray(independent_scores, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Nonempty independent bounded scores required')
    width = kernel_upper - kernel_lower
    if width <= 0 or not 0 <= delta < 1 or not np.isfinite(bias_upper):
        raise ValueError('Invalid range, bias or failure budget')
    if np.any(values < kernel_lower - 1e-12) or np.any(values > kernel_upper + 1e-12):
        raise ValueError('Scores outside declared range')
    threshold = (bias_upper - kernel_lower) / width
    if threshold < 0:
        return dict(p=float(delta), log_evalue=None, note='Null impossible on valid bias event')
    if threshold >= 1:
        return dict(p=1., log_evalue=0.)
    if threshold == 0:
        return dict(p=float(delta) if np.any(values > kernel_lower) else 1., log_evalue=None)
    fractions = np.geomspace(1e-4, .99, 64) if fractions is None else np.asarray(fractions, float)
    if fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions >= 1)):
        raise ValueError('A fixed nonempty fraction grid in [0,1) is required')
    scaled = (values - kernel_lower) / width
    capitals = np.array([np.log1p(fraction * (scaled / threshold - 1)).sum()
                         for fraction in fractions])
    log_evalue = float(logsumexp(capitals) - np.log(len(capitals)))
    probability = 1. if log_evalue <= 0 else min(1., delta + np.exp(-log_evalue))
    return dict(p=float(probability), log_evalue=log_evalue, fixed_mixture_components=len(fractions))
