"""Retrospective oracle-bank comparison under an explicitly frozen split.

This consumes existing simulation records. It does not regenerate observations,
fit a model, choose a favorable split, or establish real-panel calibration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(k, n):
    z = 1.959963984540054
    den = 1 + z*z/n
    center = (k/n + z*z/(2*n))/den
    half = z*np.sqrt((k/n)*(1-k/n)/n + z*z/(4*n*n))/den
    return [max(0., center-half), min(1., center+half)]


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    config = json.loads(args.protocol.read_text())
    root = args.root.resolve()
    source = root / config['source_file']
    assert sha(source) == config['source_sha256']
    output = root / config['output_directory']
    output.mkdir(parents=True, exist_ok=False)
    with source.open() as f:
        raw = list(csv.DictReader(f))
    methods, regimes = config['methods'], config['regimes']
    rows = [x for x in raw if x['regime'] in regimes and
            x['alternative'] in ['null', 'positive_covariance']]
    assert len(rows) == 4*2*300*6
    indexed = {(x['regime'], x['alternative'], x['method'], int(x['replication'])): x
               for x in rows}
    assert len(indexed) == len(rows)
    # Verify unique generating draws and identical data across methods.
    seeds, hashes = [], []
    for regime, alternative in itertools.product(regimes, ['null', 'positive_covariance']):
        for rep in range(300):
            group = [indexed[regime, alternative, method, rep] for method in methods]
            for field in ['sample_sha256', 'seed', 'fold_sha256']:
                assert len({x[field] for x in group}) == 1
            seeds.append(int(group[0]['seed']))
            hashes.append(group[0]['sample_sha256'])
    assert len(seeds) == len(set(seeds)) == len(set(hashes))
    assert config['calibration_null_replications'] == [0, 149]
    assert config['validation_null_replications'] == [150, 299]
    assert config['power_positive_covariance_replications'] == [0, 299]
    assert config['alpha'] == .05
    evaluated, summaries, banks, comparisons = [], [], [], []
    decisions = {}
    for regime, method in itertools.product(regimes, methods):
        bank = np.array([float(indexed[regime, 'null', method, i]['pvalue']) for i in range(150)])
        assert np.all(np.isfinite(bank)) and np.all((bank >= 0) & (bank <= 1))
        banks.append(dict(regime=regime, method=method, size=150,
                          distinct_values=int(np.unique(bank).size),
                          boundary_native_p=float(np.sort(bank)[6])))
        for alternative, reps in [('null', range(150, 300)), ('positive_covariance', range(300))]:
            values = []
            for rep in reps:
                row = indexed[regime, alternative, method, rep]
                native = float(row['pvalue'])
                if not np.isfinite(native) or not 0 <= native <= 1:
                    raise ValueError('Evaluation reference value must be finite and in [0, 1]')
                numerator = 1 + int(np.count_nonzero(bank <= native))
                reject = int(numerator <= 7)
                values.append(reject)
                evaluated.append(dict(regime=regime, alternative=alternative,
                                      method=method, replication=rep, seed=int(row['seed']),
                                      sample_sha256=row['sample_sha256'], native_pvalue=native,
                                      rank_numerator=numerator, rank_denominator=151,
                                      calibrated_pvalue=numerator/151, reject=reject))
            n, k = len(values), sum(values)
            lo, hi = wilson(k, n)
            summaries.append(dict(regime=regime, alternative=alternative, method=method,
                                  rejections=k, n=n, rate=k/n,
                                  conditional_bank_wilson95_low=lo,
                                  conditional_bank_wilson95_high=hi))
            if alternative == 'positive_covariance':
                decisions[regime, method] = np.array(values)
    for regime in regimes:
        for first, second in itertools.combinations(methods, 2):
            a, b = decisions[regime, first], decisions[regime, second]
            a_only = int(np.count_nonzero((a == 1) & (b == 0)))
            b_only = int(np.count_nonzero((a == 0) & (b == 1)))
            p = float(binomtest(a_only, a_only+b_only, .5).pvalue) if a_only+b_only else 1.
            comparisons.append(dict(regime=regime, first=first, second=second,
                                    first_only=a_only, second_only=b_only, n=300,
                                    conditional_bank_rate_difference=float(np.mean(a-b)),
                                    paired_pvalue=p))
    assert len(comparisons) == 60
    order = sorted(range(60), key=lambda i: comparisons[i]['paired_pvalue'])
    last = 0.
    for rank, idx in enumerate(order):
        last = min(1., max(last, (60-rank)*comparisons[idx]['paired_pvalue']))
        comparisons[idx]['holm60_pvalue'] = last
    assert len(evaluated) == 10800 and len(summaries) == 48
    write_csv(output/'evaluated.csv', evaluated)
    write_csv(output/'summary.csv', summaries)
    write_csv(output/'calibration_banks.csv', banks)
    write_csv(output/'paired_power.csv', comparisons)
    receipt = dict(status='COMPLETE', protocol_sha256=sha(args.protocol),
                   source_sha256=sha(source), script_sha256=sha(__file__),
                   unique_source_draws=2400, calibration_draws=600,
                   validation_null_draws=600, positive_draws=1200,
                   evaluated_method_rows=len(evaluated), method_summary_cells=len(summaries),
                   paired_comparisons=len(comparisons), marginal_rank_bound=7/151,
                   interpretation=config['interpretation'], timing=config['timing'],
                   interval_scope=config['summaries'], cost=config['cost'],
                   files={p.name:sha(p) for p in sorted(output.glob('*.csv'))})
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
