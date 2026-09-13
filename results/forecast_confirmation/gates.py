"""Fixed finite-archive gates and explicitly classical comparison bounds.

Condition on the complete selection archive. Independent uniform index draws
give a finite-population sampling experiment, not a temporal generalization
guarantee. Full forecast metadata are used only by the known-forecast method.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent / 'helpers'))
from reference_certificate_efficiency import (signed_category_budget,
    triple_scores, one_sided_certificate, family_validation_delta)
from peer_rank_products import peer_rank_products
from classical_reference_comparators import mixture_betting_pvalue

METHODS = ['reference_u', 'reference_betting', 'known_forecast_u', 'known_forecast_pairs', 'loss_gain']
TOTAL = 32768
CELLS = 32


def means(c, values):
    count = np.bincount(c, minlength=CELLS)
    return np.divide(np.bincount(c, weights=values, minlength=CELLS), count,
                     out=np.full(CELLS, .5), where=count > 0)


def exact_forecast_residual(x, category):
    """Center population midranks with exact integer numerators.

    Category-constant forecasts have identically zero residuals, not floating
    noise subsequently magnified by a near-zero certificate range. The input
    length bound also prevents signed-int64 products and sums from overflow.
    """
    x,category=np.asarray(x,float),np.asarray(category,int)
    n=len(x)
    if not 1<=n<=1_000_000_000 or category.shape!=x.shape:
        raise ValueError('Aligned nonempty vectors of at most one billion rows required')
    if not np.isfinite(x).all() or np.any((category<0)|(category>=CELLS)):
        raise ValueError('Finite forecasts and declared categories required')
    q=(2*rankdata(x,method='average')-1).astype(np.int64)
    counts=np.bincount(category,minlength=CELLS).astype(np.int64)
    sums=np.zeros(CELLS,dtype=np.int64)
    np.add.at(sums,category,q)
    numerator=counts[category]*q-sums[category]
    denominator=2*n*counts[category]
    return numerator/denominator


def by_values(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p, kind='stable')
    scaled = p[order] * n * np.sum(1 / np.arange(1, n + 1)) / np.arange(1, n + 1)
    result = np.ones(n)
    result[order] = np.minimum(1, np.minimum.accumulate(scaled[::-1])[::-1])
    return result


def classical(mean, h, width, full=False, alpha=.05):
    """MP2009 independent mean, or permutation-MGF full-U construction.

    The two constructions use the same fixed disjoint scores. They have
    different statistics and constants. The latter follows the proved finite-sample
    variance argument for a bounded kernel of order two, without learning bias.
    """
    h = np.asarray(h, float)
    j = len(h)
    assert j >= 2 and width >= 0 and np.isfinite(h).all()
    if not full:
        assert np.isclose(mean, h.mean(), rtol=1e-12, atol=1e-14)
    if width == 0:
        assert np.max(np.abs(h)) < 1e-14 and abs(mean) < 1e-14
        return dict(mean=0., radius=0., lower=0., p=1., variance=0., width=0.)
    assert np.ptp(h) <= width + 1e-12
    variance = float(h.var(ddof=1))
    x = np.log(2 / alpha)
    a = np.sqrt(2 * variance / j)
    c = (2 * width / np.sqrt(j * (j - 1)) + width / (3*j)
         if full else 7 * width / (3 * (j - 1)))
    radius = a * np.sqrt(x) + c * x
    d = max(mean, 0.)
    root = 2*d / (a + np.sqrt(a*a + 4*c*d)) if d else 0.
    exponent = root*root
    if full:
        radius = min(radius, width * np.sqrt(x / (2*j)))
        exponent = max(exponent, 2*j*(d/width)**2)
    return dict(mean=float(mean), radius=float(radius), lower=float(mean-radius),
                p=float(min(1., 2*np.exp(-exponent))), variance=variance, width=float(width))


def draw_indices(n, seed):
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    train = streams[0].integers(n, size=4096)
    validate = streams[1].integers(n, size=(8192, 2))
    evaluate = streams[2].integers(n, size=12288)
    all_rows = np.concatenate([train, validate.ravel(), evaluate])
    assert len(all_rows) == TOTAL
    return train, validate, evaluate, all_rows


def certify(x, y, category, base, blend, scale, indices, family=8):
    """All five fixed methods for one member of a complete candidate family."""
    x, y, base, blend = [np.asarray(v, float) for v in (x, y, base, blend)]
    category = np.asarray(category, int)
    assert len(x) == len(y) and np.isfinite([x, y, base, blend]).all()
    assert np.all((category >= 0) & (category < CELLS)) and scale > 0
    train, validate, evaluate, all_rows = indices
    zt, zv, ze = category[train], category[validate[:, 0]], category[evaluate]
    f = means(zt, (rankdata(x[train])-1)/(len(train)-1))
    g = means(zt, (rankdata(y[train])-1)/(len(train)-1))
    compare = lambda v: (v[validate[:, 0]] > v[validate[:, 1]]).astype(float) + .5*(v[validate[:, 0]] == v[validate[:, 1]])
    delta = family_validation_delta(family)
    budget = signed_category_budget(f, g, np.bincount(zv, minlength=CELLS),
                                   means(zv, compare(x)), means(zv, compare(y)), delta)
    xx, yy, ff, gg = x[evaluate], y[evaluate], f[ze], g[ze]
    h = triple_scores(*[v[None, :] for v in (xx, yy, ff, gg)])
    mean = float(peer_rank_products(xx, yy, ff, gg)['corrected_residual_product'].mean())
    result = one_sided_certificate(mean, h, len(h), f, g, budget['bias_upper'], delta=delta)
    rows = [dict(method='reference_u', mean=mean, radius=result['radius'],
                 lower=result['lower_bound'], p=result['p'], bias=budget['bias_upper'],
                 variance=result['kernel_sample_variance'], width=result['kernel_upper']-result['kernel_lower'])]
    betting = mixture_betting_pvalue(h, result['kernel_lower'], result['kernel_upper'],
                                    bias_upper=budget['bias_upper'], delta=delta)
    # A test-only comparator. Do not present the U-bound interval as its own.
    rows.append(dict(method='reference_betting', mean=float(h.mean()), radius=None,
                     lower=None, p=betting['p'], bias=budget['bias_upper'],
                     variance=float(h.var(ddof=1)), width=result['kernel_upper']-result['kernel_lower']))
    # Full-X residual metadata use no outcome values. Their category centering
    # is exact for the finite archive, including ties and empty declared cells.
    a=exact_forecast_residual(x,category)
    assert abs(a.mean()) < 1e-13
    ay, ya = a[all_rows], y[all_rows]
    pair = .25*(ay[::2]-ay[1::2])*np.sign(ya[::2]-ya[1::2])
    full = float(np.sum(ay*(rankdata(ya)-(len(ya)+1)/2))/(len(ya)*(len(ya)-1)))
    width = float(np.ptp(a)/2)
    for name, center, use_full in [('known_forecast_u', full, True),
                                   ('known_forecast_pairs', float(pair.mean()), False)]:
        rows.append(dict(method=name, bias=0., **classical(center, pair, width, full=use_full)))
    yy = np.clip(y, 0, scale)/scale
    bb = np.clip(base, 0, scale)/scale
    pp = np.clip(blend, 0, scale)/scale
    gain = ((yy-bb)**2-(yy-pp)**2)[all_rows]
    rows.append(dict(method='loss_gain', bias=0., **classical(float(gain.mean()), gain, 2.)))
    # Exact archive quantities are diagnostic only; no gate uses these values.
    ry = (rankdata(y)-.5)/len(y)
    exact_rank = float(np.mean(a*(ry-means(category, ry)[category])))
    exact_gain = float(np.mean((yy-bb)**2-(yy-pp)**2))
    for row in rows:
        row.update(exact_target=exact_gain if row['method']=='loss_gain' else exact_rank,
                   query_calls=TOTAL, distinct_labels=len(np.unique(all_rows)),
                   archive_rows=len(y), delta=delta if row['method'].startswith('reference_') else 0.)
    return rows, dict(f=f, g=g, **budget)
