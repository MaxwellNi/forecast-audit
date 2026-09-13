#!/usr/bin/env python3
"""Independently rebuild the fixed archive certificates without writing files.

No producer/helper implementation is imported. Literal pair comparisons rebuild
both full-U means; binomial-CDF inversion and linear programming rebuild the
validation allowance. This verifies exposed arithmetic, not statistical coverage
across repeated applications or the historical truth of local timestamps.
"""
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.optimize import brentq, linprog
from scipy.stats import binom

HERE = Path(__file__).resolve().parent
RTOL, ATOL = 1e-10, 1e-12
CATEGORIES = 4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def hashes():
    return {str(p.relative_to(HERE)): sha(p) for p in HERE.rglob('*') if p.is_file()}


def midranks(x, leave_self=False):
    ordered = np.sort(x)
    below = np.searchsorted(ordered, x, side='left')
    through = np.searchsorted(ordered, x, side='right')
    numerator = below + .5 * (through - below) - (.5 if leave_self else 0.)
    return numerator / (len(x) - int(leave_self))


def cell_means(c, values):
    return np.array([values[c == j].mean() if np.any(c == j) else .5
                     for j in range(CATEGORIES)])


def comparison(focal, reference):
    return (focal > reference) + .5 * (focal == reference)


def full_u_mean(x, y, f, g, block_size=128):
    """Literal comparisons, in blocks to avoid an n-by-n memory allocation.

    The product of reference sums includes coincident reference positions;
    subtract their joint sum to keep the two references distinct. Self terms
    contribute exactly 1/2 and 1/4 and are removed before that subtraction.
    """
    n = len(x)
    assert n >= 3
    values = np.empty(n)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        a = comparison(x[start:stop, None], x[None, :])
        b = comparison(y[start:stop, None], y[None, :])
        sa, sb = a.sum(axis=1) - .5, b.sum(axis=1) - .5
        joint = np.einsum('ij,ij->i', a, b) - .25
        values[start:stop] = ((sa * sb - joint) / ((n - 1) * (n - 2))
                             - f[start:stop] * sb / (n - 1)
                             - g[start:stop] * sa / (n - 1)
                             + f[start:stop] * g[start:stop])
    return float(values.mean())


def validation_budget(f, g, counts, mean_x, mean_y, delta):
    """Rebuild confidence rectangles and capped-simplex extrema independently."""
    radii = np.full(CATEGORIES, np.inf)
    occupied = counts > 0
    radii[occupied] = np.sqrt(np.log(8 * CATEGORIES / delta) / (2 * counts[occupied]))
    lx, ux = np.maximum(0., mean_x - radii), np.minimum(1., mean_x + radii)
    ly, uy = np.maximum(0., mean_y - radii), np.minimum(1., mean_y + radii)
    total, tail = int(counts.sum()), delta / (2 * CATEGORIES)
    # Invert P_p[Binomial(total,p) <= observed_count] = tail directly.
    # This is independent of the producer's beta inverse implementation.
    caps = np.array([1. if count == total else brentq(
        lambda mass: binom.cdf(int(count), total, mass) - tail,
        0., 1., xtol=1e-14) for count in counts])
    corners = np.array([(a - f) * (b - g) for a in (lx, ux) for b in (ly, uy)])
    cell_upper, cell_lower = corners.max(axis=0), corners.min(axis=0)
    absolute = np.maximum(abs(lx - f), abs(ux - f)) * np.maximum(abs(ly - g), abs(uy - g))

    def extremum(values, maximize):
        sign = -1. if maximize else 1.
        # Scaling prevents the LP solver from discarding small objective gaps.
        fit = linprog(sign * values * 1e6, A_eq=np.ones((1, CATEGORIES)),
                      b_eq=[1.], bounds=list(zip(np.zeros(CATEGORIES), caps)),
                      method='highs', options={'primal_feasibility_tolerance': 1e-10,
                                               'dual_feasibility_tolerance': 1e-10})
        assert fit.success, fit.message
        assert abs(fit.x.sum() - 1.) < 1e-10
        assert np.all(fit.x >= -1e-10) and np.all(fit.x <= caps + 1e-10)
        return float(np.dot(fit.x, values))

    return dict(lower_x=lx, upper_x=ux, lower_y=ly, upper_y=uy,
                category_mass_upper=caps, signed_cell_upper=cell_upper,
                signed_upper=extremum(cell_upper, True),
                signed_lower=extremum(cell_lower, False),
                absolute_upper=extremum(absolute, True))


def certificate(mean, h, f, g, bias, alpha, delta, kind):
    """Direct range/variance formulas and their one-sided probability inversion."""
    corners = np.array([(a - f) * (b - g) for a in (0., 1.) for b in (0., 1.)])
    lo, hi = float(corners.min()), float(corners.max())
    width, count = hi - lo, len(h)
    assert width > 0 and count >= 2 and 0 < delta < alpha < 1
    assert np.all((h >= lo - ATOL) & (h <= hi + ATOL))
    margin = max(mean - bias, 0.)
    sample_variance = np.nan
    if kind == 'range':
        radius = width * np.sqrt(np.log(1 / (alpha - delta)) / (2 * count))
        probability = min(1., delta + np.exp(-2 * count * (margin / width) ** 2))
    else:
        assert kind in ('variance', 'independent_bernstein')
        if kind == 'independent_bernstein':
            assert np.isclose(mean, h.mean(), rtol=1e-12, atol=1e-14)
        sample_variance = float(np.var(h, ddof=1))
        a = np.sqrt(2 * sample_variance / count)
        c = (2 * width / np.sqrt(count * (count - 1)) + width / (3 * count)
             if kind == 'variance' else 7 * width / (3 * (count - 1)))
        log_term = np.log(2 / (alpha - delta))
        radius = a * np.sqrt(log_term) + c * log_term
        if kind == 'variance':
            radius = min(radius, width * np.sqrt(log_term / (2 * count)))
        root = 2 * margin / (a + np.sqrt(a * a + 4 * c * margin)) if margin else 0.
        exponent = root ** 2
        if kind == 'variance':
            exponent = max(exponent, 2 * count * (margin / width) ** 2)
        probability = min(1., delta + 2 * np.exp(-exponent))
    return dict(mean=mean, bias_upper=bias, kernel_lower=lo, kernel_upper=hi,
                effective_triples=count, delta=delta, alpha=alpha, radius=radius,
                lower_bound=mean - bias - radius, p=probability,
                reject=probability < alpha, kernel_sample_variance=sample_variance,
                bound=kind)


def main():
    before, errors = hashes(), {}

    def close(label, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL,
                                   equal_nan=True, err_msg=label)
        left, right = np.asarray(actual, float), np.asarray(expected, float)
        differences = np.abs(left - right)
        finite = differences[np.isfinite(differences)]
        errors[label] = max(errors.get(label, 0.), float(finite.max()) if finite.size else 0.)

    def fields(label, actual, expected):
        for key, value in expected.items():
            if isinstance(value, (bool, str)):
                assert actual[key] == value, (label, key, actual[key], value)
            else:
                close(label + '.' + key, actual[key], value)

    p, seal, receipt = read('protocol.json'), read('archive_seal.json'), read('audit_receipt.json')
    protocol_hash = sha(HERE / 'protocol.json')
    assert (HERE / 'protocol.sha256').read_text().split()[0] == protocol_hash
    assert sha(HERE / 'study.py') == p['script_sha256']
    assert seal['protocol_sha256'] == receipt['protocol_sha256'] == protocol_hash
    assert sha(HERE / 'forecast_archive.npz') == seal['archive_sha256'] == receipt['archive_sha256']
    assert sha(HERE / 'forecast_model.pkl') == seal['model_sha256']
    assert receipt['source_sha256'] == p['source_sha256']
    for name, entry in p['helper_sources'].items():
        assert sha(HERE / 'helpers' / name) == entry['sha256']
    for name, value in receipt['output_sha256'].items():
        assert sha(HERE / name) == value
    started, sampled, written = read('audit_started.json'), read('sampling_receipt.json'), read('certificate_written.json')
    assert started['protocol_sha256'] == protocol_hash and started['archive_sha256'] == seal['archive_sha256']
    assert sampled['sampling_indices_sha256'] == sha(HERE / 'sampling_indices.npz')
    assert written['certificate_results_sha256'] == sha(HERE / 'certificate_results.csv')
    assert written['census_not_yet_computed'] and receipt['census_computed_after_certificate_written']
    times = [p['frozen_utc'], seal['sealed_utc'], started['started_utc'], sampled['drawn_utc'],
             written['written_utc'], receipt['completed_utc']]
    assert list(map(datetime.fromisoformat, times)) == sorted(map(datetime.fromisoformat, times))

    archive = np.load(HERE / 'forecast_archive.npz', allow_pickle=False)
    draws = np.load(HERE / 'sampling_indices.npz', allow_pickle=False)
    training, validation, evaluation = [draws[key] for key in ('training', 'validation', 'evaluation')]
    n = len(archive['outcome'])
    assert n == 70080
    assert training.shape == (p['nuisance_training_rows'],)
    assert validation.shape == (p['validation_pairs'], 2)
    assert evaluation.shape == (p['evaluation_groups'], p['evaluation_peers'])
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(p['seed']).spawn(3)]
    for stream, indices in zip(streams, (training, validation, evaluation)):
        assert np.array_equal(stream.integers(0, n, size=indices.shape), indices)
    all_indices = np.concatenate([training, validation.ravel(), evaluation.ravel()])
    query_cost = dict(training_queries=len(training), validation_pair_queries=validation.size,
                      evaluation_queries=evaluation.size, total_queries=len(all_indices),
                      unique_archive_rows_queried=len(np.unique(all_indices)), archive_census_rows=n,
                      query_to_census_ratio=len(all_indices) / n,
                      unique_to_census_ratio=len(np.unique(all_indices)) / n,
                      feature_archive_construction_already_accessed_full_archive=True)
    fields('receipt.query_cost', receipt['query_cost'], query_cost)
    fields('sampling.query_cost', sampled, query_cost)
    fit = pd.read_csv(HERE / 'nuisance_validation.csv').set_index(['candidate', 'category'])
    results = pd.read_csv(HERE / 'certificate_results.csv').set_index(['candidate', 'method'])
    triples = pd.read_csv(HERE / 'triple_scores.csv.gz').set_index(['candidate', 'triple'])
    census = pd.read_csv(HERE / 'census_results.csv').set_index(['candidate', 'method'])
    expected_index = set(itertools.product(p['candidates'], p['methods']))
    assert results.index.is_unique and set(results.index) == expected_index
    assert census.index.is_unique and set(census.index) == expected_index
    assert fit.index.is_unique and set(fit.index) == set(itertools.product(p['candidates'], range(CATEGORIES)))
    rebuilt, candidate_checks = {}, {}
    for name in p['candidates']:
        x, y, categories = archive[name], archive['outcome'], archive['category']
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert set(np.unique(categories)) <= set(range(CATEGORIES))
        f = np.clip(cell_means(categories[training], midranks(x[training], True)), 0., 1.)
        g = np.clip(cell_means(categories[training], midranks(y[training], True)), 0., 1.)
        zval = categories[validation[:, 0]]
        counts = np.bincount(zval, minlength=CATEGORIES)
        mean_x = cell_means(zval, comparison(x[validation[:, 0]], x[validation[:, 1]]))
        mean_y = cell_means(zval, comparison(y[validation[:, 0]], y[validation[:, 1]]))
        budget = validation_budget(f, g, counts, mean_x, mean_y, p['delta'])
        fitted = fit.loc[name].reindex(range(CATEGORIES))
        fit_values = dict(f=f, g=g, training_count=np.bincount(categories[training], minlength=CATEGORIES),
                          validation_count=counts, validation_mean_x=mean_x, validation_mean_y=mean_y)
        fit_values.update({key: budget[key] for key in ('lower_x', 'upper_x', 'lower_y', 'upper_y',
                                                       'category_mass_upper', 'signed_cell_upper')})
        fields('validation', fitted, fit_values)
        # Partition within raw groups. Remainder rows, if any, stay in full U.
        used = 3 * (evaluation.shape[1] // 3)
        positions = evaluation[:, :used].reshape(-1, 3)
        h = np.zeros(len(positions))
        for focal_role, left_role, right_role in itertools.permutations(range(3)):
            focal, left, right = positions[:, focal_role], positions[:, left_role], positions[:, right_role]
            h += ((comparison(x[focal], x[left]) - f[categories[focal]])
                  * (comparison(y[focal], y[right]) - g[categories[focal]])) / 6
        recorded_h = triples.loc[name]
        assert recorded_h.index.is_unique and set(recorded_h.index) == set(range(len(h)))
        close('triple_scores', recorded_h.reindex(range(len(h))).score, h)
        grouped_means = np.array([full_u_mean(x[idx], y[idx], f[categories[idx]], g[categories[idx]])
                                  for idx in evaluation])
        idx = evaluation.ravel()
        pooled_mean = full_u_mean(x[idx], y[idx], f[categories[idx]], g[categories[idx]])
        means = {'grouped': float(grouped_means.mean()), 'pooled': pooled_mean,
                 'independent': float(h.mean())}
        ux, uy = midranks(x), midranks(y)
        mx, my = cell_means(categories, ux), cell_means(categories, uy)
        mass = np.bincount(categories, minlength=CATEGORIES) / n
        theta = float(np.mean((ux - mx[categories]) * (uy - my[categories])))
        exact_bias = float(np.dot(mass, (mx - f) * (my - g)))
        fitted_target = float(np.mean((ux - f[categories]) * (uy - g[categories])))
        close('census.bias_identity', fitted_target, theta + exact_bias)
        if name == 'categorical_copy_control':
            close('census.copy_control', theta, 0.)
        assert budget['signed_lower'] - ATOL <= exact_bias <= budget['signed_upper'] + ATOL
        candidate_checks[name] = dict(grouped_mean=means['grouped'], pooled_mean=pooled_mean,
                                      independent_triple_mean=means['independent'],
                                      signed_bias_lower=budget['signed_lower'],
                                      signed_bias_upper=budget['signed_upper'],
                                      absolute_bias_upper=budget['absolute_upper'], census_theta=theta)
        for method in p['methods']:
            kind = ('independent_bernstein' if method.endswith('_bernstein') else
                    'variance' if method.endswith('_variance') else 'range')
            bias = budget['absolute_upper'] if '_absolute_' in method else budget['signed_upper']
            values = certificate(means[method.split('_')[0]], h, f, g, bias, p['alpha'], p['delta'], kind)
            values.update(query_cost)
            values['primary'] = method == p['primary'].split(':')[0]
            fields('certificate', results.loc[(name, method)], values)
            assert np.isfinite(results.loc[(name, method), 'computational_seconds'])
            assert results.loc[(name, method), 'computational_seconds'] >= 0
            census_values = dict(census_theta=theta, census_fitted_moment=fitted_target,
                                  exact_learning_bias=exact_bias, signed_allowance=budget['signed_upper'],
                                  absolute_allowance=budget['absolute_upper'],
                                  signed_bias_covered=exact_bias <= budget['signed_upper'],
                                  confidence_lower_bound=values['lower_bound'],
                                  observed_lower_covers=values['lower_bound'] <= theta, p=values['p'])
            fields('census', census.loc[(name, method)], census_values)
            rebuilt[(name, method)] = values
    assert len(triples) == len(p['candidates']) * len(h)
    family = p['family_size']
    assert family == len(p['candidates'])
    harmonic = sum(1 / k for k in range(1, family + 1))
    for method in p['methods']:
        ordered = sorted(p['candidates'], key=lambda name: rebuilt[(name, method)]['p'])
        probabilities = np.array([rebuilt[(name, method)]['p'] for name in ordered])
        scaled = probabilities * family * harmonic / np.arange(1, family + 1)
        adjusted = np.minimum(1., np.minimum.accumulate(scaled[::-1])[::-1])
        for name, value in zip(ordered, adjusted):
            for label, table in (('certificate', results), ('census', census)):
                fields(label, table.loc[(name, method)], dict(p_by=value, by_retain=value <= p['by_q']))

    # Fresh source reproductions intentionally have no release manifest.
    manifest_checked = (HERE / 'MANIFEST.json').exists()
    if manifest_checked:
        manifest = read('MANIFEST.json')['files']
        assert set(manifest) == set(before) - {'MANIFEST.json'}
        for name, entry in manifest.items():
            assert sha(HERE / name) == entry['sha256'], ('manifest hash', name)
            assert (HERE / name).stat().st_size == entry['bytes'], ('manifest size', name)
    assert before == hashes(), 'Read-only verification changed files'
    print(json.dumps(dict(
        passed=True, files_unchanged=True, release_manifest_checked=manifest_checked,
        producer_helpers_imported=False, archive_rows=n, independent_draw_indices_reproduced=True,
        methods=len(p['methods']), candidates=len(p['candidates']), certificate_rows_rebuilt=len(rebuilt),
        validation_cells_rebuilt=len(fit), bias_linear_programs_solved=3 * len(p['candidates']),
        symmetric_triple_scores_checked=len(triples),
        direct_group_means_rebuilt=len(p['candidates']) * evaluation.shape[0],
        direct_grouped_focal_products_rebuilt=len(p['candidates']) * evaluation.size,
        direct_pooled_focal_products_rebuilt=len(p['candidates']) * evaluation.size,
        raw_probabilities_and_decisions_rebuilt=len(rebuilt),
        numeric_comparison_tolerances=dict(rtol=RTOL, atol=ATOL),
        maximum_absolute_discrepancies=errors, candidate_checks=candidate_checks,
        all_observed_lower_bounds_cover_census=bool(census.observed_lower_covers.all()),
        scope='Deterministic reconstruction of this exposed application; timings are hash-checked, not reproduced. '
              'Recorded local provenance is checked for consistency, not externally authenticated. '
              'No new independent statistical application or repeated-sample coverage claim.'), indent=2))


if __name__ == '__main__':
    main()
