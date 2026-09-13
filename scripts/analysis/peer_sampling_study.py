"""Reproduce the paired resampled- and fixed-peer null experiment.

Known conditional means isolate reference reuse. The fixed-peer arms condition
on the entity attributes and use independent noises in the two channels.
No source observations or restricted incidence patterns are required.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import math
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm
import peer_rank_study as prs

def group_t(scores_by_group):
    m = len(scores_by_group)
    mean = float(math.fsum(map(float, scores_by_group)) / m)
    se = float(np.std(scores_by_group, ddof=1) / math.sqrt(m))
    return mean / se if se else float('nan')

def one_replication(rep):
    out = []
    protocol = {'seed_base': 202609059830000}
    rng = np.random.default_rng(protocol['seed_base'] + rep)
    (b, ex, ey) = rng.normal(size=(3, 400, 128))
    rng.choice([-1.0, 1.0], size=(400, 128))
    (x, y, mx, my) = prs.make_condition('gaussian_null', b[:, :32], ex[:, :32], ey[:, :32], None)
    (naive, corrected) = prs.vectorized_scores(x, y, mx, my)
    for m in (25, 100, 400):
        out.append({'replication': rep, 'peer_regime': 'resampled_peers', 'periods': m, 'shared_statistic': group_t(naive[:m]), 'distinct_statistic': group_t(corrected[:m]), 'shared_mean': float(naive[:m].mean()), 'distinct_mean': float(corrected[:m].mean())})
    N = 32
    a = norm.ppf((np.arange(1, N + 1) - 0.5) / N)
    P = ndtr((a[:, None] - a[None, :]) / math.sqrt(2))
    np.fill_diagonal(P, 0.0)
    f = P.sum(1) / (N - 1)
    rng = np.random.default_rng(202609109830000 + rep)
    (eps, eta) = rng.normal(size=(2, 400, N))
    for (arm, sign) in (('fixed_peers_aligned_effects', 1.0), ('fixed_peers_opposed_effects', -1.0)):
        V = a[None, :] + eps
        W = sign * a[None, :] + eta
        g = f if sign > 0 else 1.0 - f
        (naive, corrected) = prs.vectorized_scores(V, W, np.broadcast_to(f, V.shape).copy(), np.broadcast_to(g, V.shape).copy())
        for m in (25, 100, 400):
            out.append({'replication': rep, 'peer_regime': arm, 'periods': m, 'shared_statistic': group_t(naive[:m]), 'distinct_statistic': group_t(corrected[:m]), 'shared_mean': float(naive[:m].mean()), 'distinct_mean': float(corrected[:m].mean())})
    return out

def wilson(rejections, replications):
    z = norm.ppf(.975)
    p = rejections / replications
    denominator = 1 + z * z / replications
    center = (p + z * z / (2 * replications)) / denominator
    half = z * math.sqrt(p * (1-p) / replications + z*z / (4*replications**2)) / denominator
    return max(0., center-half), min(1., center+half)


def run(output, replications=300, workers=1, verify=None):
    if replications < 2 or workers < 1:
        raise ValueError('At least two replications and one worker are required.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = [row for batch in pool.map(one_replication, range(replications)) for row in batch]
    draws = pd.DataFrame(records)
    summary = []
    for (regime, periods), cell in draws.groupby(['peer_regime', 'periods'], sort=True):
        for method, column, mean_column in [('shared_references', 'shared_statistic', 'shared_mean'),
                                            ('distinct_references', 'distinct_statistic', 'distinct_mean')]:
            count = int((norm.sf(cell[column]) < .05).sum())
            low, high = wilson(count, len(cell))
            summary.append(dict(peer_regime=regime, periods=periods, method=method,
                                replications=len(cell), rejections=count, rate=count/len(cell),
                                wilson_low=low, wilson_high=high, mean_T=float(cell[column].mean()),
                                sd_T=float(cell[column].std(ddof=1)), mean_score=float(cell[mean_column].mean())))
    summaries = pd.DataFrame(summary)
    checks = {}
    if verify is not None:
        verify = Path(verify)
        keys = ['replication', 'peer_regime', 'periods']
        expected = pd.read_csv(verify / 'replications.csv', float_precision='round_trip').sort_values(keys)
        actual = draws.sort_values(keys)
        if actual[keys].to_dict('records') != expected[keys].to_dict('records'):
            raise AssertionError('Replication identities differ.')
        values = ['shared_statistic','distinct_statistic','shared_mean','distinct_mean']
        error = float(np.max(np.abs(actual[values].to_numpy()-expected[values].to_numpy())))
        np.testing.assert_allclose(actual[values], expected[values], rtol=1e-12, atol=1e-13)
        cell_keys = ['peer_regime','periods','method']
        expected_summary = pd.read_csv(verify / 'summary.csv', float_precision='round_trip').sort_values(cell_keys)
        actual_summary = summaries.sort_values(cell_keys)
        if actual_summary[cell_keys].to_dict('records') != expected_summary[cell_keys].to_dict('records'):
            raise AssertionError('Summary cell identities differ.')
        np.testing.assert_array_equal(actual_summary.rejections, expected_summary.rejections)
        summary_values = ['rate','wilson_low','wilson_high','mean_T','sd_T','mean_score']
        np.testing.assert_allclose(actual_summary[summary_values], expected_summary[summary_values], rtol=1e-12, atol=1e-13)
        checks = dict(replications_checked=len(draws), cells_checked=len(summaries), maximum_draw_difference=error)
    draws.to_csv(output / 'replications.csv', index=False)
    summaries.to_csv(output / 'summary.csv', index=False)
    protocol = dict(replications=replications, entities=32, groups=[25,100,400],
                    resampled_seed_base=202609059830000, fixed_seed_base=202609109830000,
                    resampled_draw_shape=[3,400,128], resampled_used_entities=32,
                    notes='The wider resampled draw array preserves the recorded random stream; only the first 32 entities are used. Groups are nested within replication. Both methods share each panel.',
                    nuisance_means='known conditional means', null_threshold='one-sided normal 0.05',
                    numpy=np.__version__, replay_checks=checks)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps(checks or dict(replications=len(draws),cells=len(summaries))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replications', type=int, default=300)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--verify', type=Path, help='Directory with recorded replications.csv and summary.csv.')
    args = parser.parse_args()
    run(args.output, args.replications, args.workers, args.verify)
