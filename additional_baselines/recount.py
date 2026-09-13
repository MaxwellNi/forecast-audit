"""Portable per-decision recount; no raw data generation or model fit."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,hashlib,json,math
import numpy as np

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def table(name, rows):
    with (ROOT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0.0, center-half), min(1.0, center+half)

def main():
    complete = json.loads((ROOT / 'RUN_COMPLETED.json').read_text())
    protocol = json.loads(PROTOCOL.read_text())
    assert complete['status'] == 'ALL_PRESCRIBED_RUNS_COMPLETE'
    assert complete['protocol_sha256'] in {sha(PROTOCOL),protocol['original_protocol_sha256']}
    for source in protocol['sources']:
        assert sha(PACKAGE/source['path']) == source['sha256']
    all_rows, raw_by_suite, completeness = [], {}, {}
    adaptive_pairs = []
    for suite in protocol['suites']:
        with (ROOT / (suite + '.raw.csv')).open() as f:
            rows = list(csv.DictReader(f))
        raw_by_suite[suite] = rows
        spec = protocol['designs'][suite]
        assert len(rows) == spec['expected_raw_rows'], (suite, len(rows))
        assert sha(ROOT / (suite + '.raw.csv')) == next(
            x['raw_sha256'] for x in complete['results'] if x['suite'] == suite)
        groups = defaultdict(list)
        for row in rows:
            key = (row.get('setting', ''), row['regime'], row['metric'], row.get('overlap', ''))
            groups[key].append(row)
        expected_groups = {'adaptive_beta': 8, 'ablation': 72,
                           'generalization': 320, 'stress': 8}[suite]
        assert len(groups) == expected_groups
        assert {k[2] for k in groups} == {'typeI', 'power'}
        expected_regimes = set(spec['seeds_per_cell']) if suite == 'stress' else set(spec['regimes'])
        assert {k[1] for k in groups} == expected_regimes
        if suite == 'ablation':
            assert {k[0] for k in groups} == {x['label'] for x in spec['configurations']}
        if suite == 'generalization':
            assert {k[0] for k in groups} == set(spec['methods'])
            assert {float(k[3]) for k in groups} == set(spec['overlap'])
        for (setting, regime, metric, overlap), group in groups.items():
            if suite == 'adaptive_beta':
                n, seed0 = 50, 505
            elif suite == 'stress':
                n, seed0 = spec['seeds_per_cell'][regime], 505
            else:
                n = spec['seeds_per_cell'][regime]
                seed0 = spec['null_seed0'] if metric == 'typeI' else spec['alternative_seed0']
            assert len(group) == n
            assert sorted(int(x['seed']) for x in group) == list(range(seed0, seed0+n))
            expected_dim = ((160, 160) if regime == 'large_n' else (120, 100))
            if suite == 'stress':
                expected_dim = {'high_dim_fe': (600, 60), 'serial': (120, 100),
                                'overlap': (120, 98), 'small_T': (120, 24)}[regime]
            assert {(int(x['N']), int(x['T'])) for x in group} == {expected_dim}
            methods = ['fixed_beta1', 'adaptive_heuristic'] if suite == 'adaptive_beta' else [setting]
            for method in methods:
                decision = ('fixed_reject' if method == 'fixed_beta1' else 'adaptive_reject') if suite == 'adaptive_beta' else 'reject'
                values = [int(x[decision]) for x in group]
                assert set(values) <= {0, 1}
                k = sum(values)
                lo, hi = wilson(k, n)
                out = dict(suite=suite, setting=method, regime=regime,
                           signal_delta=0.0 if metric == 'typeI' else 0.15,
                           legacy_metric=metric, overlap=float(overlap) if overlap else 0.8,
                           N=expected_dim[0], T=expected_dim[1], rejections=k,
                           replicates=n, rate=k/n, wilson95_low=lo, wilson95_high=hi)
                all_rows.append(out)
            if suite == 'adaptive_beta':
                betas = np.asarray([float(x['beta_hat']) for x in group])
                adaptive_pairs.append(dict(regime=regime, legacy_metric=metric, replicates=n,
                    fixed_only=sum(int(x['fixed_reject']) == 1 and int(x['adaptive_reject']) == 0 for x in group),
                    adaptive_only=sum(int(x['fixed_reject']) == 0 and int(x['adaptive_reject']) == 1 for x in group),
                    both=sum(int(x['fixed_reject']) == 1 and int(x['adaptive_reject']) == 1 for x in group),
                    neither=sum(int(x['fixed_reject']) == 0 and int(x['adaptive_reject']) == 0 for x in group),
                    beta_mean=float(betas.mean()), beta_sd=float(betas.std(ddof=1)),
                    beta_min=float(betas.min()), beta_max=float(betas.max()),
                    beta_clipped_at_half=int(sum(betas == 0.5)), beta_clipped_at_two=int(sum(betas == 2.0)),
                    mean_adaptive_weight_norm_squared=float(np.mean([float(x['adaptive_weight_norm_squared']) for x in group]))))
                for x in group:
                    means = np.asarray(json.loads(x['spectrum_means']))
                    q = np.asarray([8, 12, 16, 24, 32], float)
                    for prefix, beta in [('fixed', 1.0), ('adaptive', float(x['beta_hat']))]:
                        design = np.column_stack([np.ones(5), q**(-beta)])
                        w = np.linalg.solve(design.T @ design, design.T)[0]
                        assert abs(w @ means - float(x[prefix+'_mean'])) < 1e-12
                        assert abs(w @ w - float(x[prefix+'_weight_norm_squared'])) < 1e-10
                        t = float(x[prefix+'_mean'])/(float(x[prefix+'_se']) + 1e-18)
                        assert abs(t-float(x[prefix+'_statistic'])) < 1e-12
                        assert int(t > 1.6448536269514722) == int(x[prefix+'_reject'])
            if suite == 'ablation':
                for x in group:
                    t = float(x['product_mean'])/(float(x['cluster_se']) + 1e-18)
                    assert abs(t-float(x['statistic'])) < 1e-12
                    assert int(t > 1.6448536269514722) == int(x['reject'])
        completeness[suite] = dict(raw_rows=len(rows), groups=len(groups),
                                  exact_seed_sets=True, exact_dimensions=True,
                                  sha256=sha(ROOT / (suite+'.raw.csv')))
    assert len(all_rows) == 416
    table('all_cell_rates.csv', all_rows)
    table('adaptive_paired_summary.csv', adaptive_pairs)
    by_key = {(r['suite'], r['setting'], r['regime'], r['legacy_metric'], r['overlap']): r for r in all_rows}
    ablation_summary, grid_summary = [], []
    for suite, destination in [('ablation', ablation_summary), ('generalization', grid_summary)]:
        for setting in dict.fromkeys(r['setting'] for r in all_rows if r['suite'] == suite):
            cells = [r for r in all_rows if r['suite'] == suite and r['setting'] == setting]
            nulls = [r for r in cells if r['signal_delta'] == 0]
            alts = [r for r in cells if r['signal_delta'] > 0]
            pairs = [((r['regime'], r['overlap']), r['rate'], by_key[(suite, setting, r['regime'], 'power', r['overlap'])]['rate']) for r in nulls]
            destination.append(dict(setting=setting, design_cells=len(pairs),
                maximum_zero_signal_rejection=max(r['rate'] for r in nulls),
                minimum_positive_signal_rejection=min(r['rate'] for r in alts),
                legacy_threshold_failed_cells=sum(a > .18 or b < .90 for _, a, b in pairs),
                stronger_prose_threshold_failed_cells=sum(a >= .15 or b <= .95 for _, a, b in pairs)))
    table('ablation_summary.csv', ablation_summary)
    table('generalization_summary.csv', grid_summary)
    expected={(x['suite'],x['setting'],x['regime'],x['legacy_metric'],x['overlap']):x for x in protocol['reference_rates']}
    comparisons=[]
    for x in all_rows:
        key=(x['suite'],x['setting'],x['regime'],x['legacy_metric'],x['overlap'])
        y=expected.pop(key)
        comparisons.append(dict(suite=x['suite'],setting=x['setting'],regime=x['regime'],
            legacy_metric=x['legacy_metric'],overlap=x['overlap'],
            reference_rejections=y['rejections'],replay_rejections=x['rejections'],
            replicates=x['replicates'],matches=x['rejections']==y['rejections'] and x['replicates']==y['replicates']))
    assert not expected
    table('reference_count_comparison.csv',comparisons)
    result=dict(scope='Legacy sensitivity only; not calibrated ranked conditional-null inference',
        protocol_sha256=sha(PROTOCOL),original_protocol_sha256=protocol['original_protocol_sha256'],
        completion_sha256=sha(ROOT/'RUN_COMPLETED.json'),recount_sha256=sha(__file__),
        completeness=completeness,raw_rows=sum(x['raw_rows'] for x in completeness.values()),
        binary_decisions=sum(x['raw_rows'] for x in completeness.values())+400,
        reported_rate_cells=len(all_rows),reference_rate_cells=len(comparisons),
        reference_count_mismatches=[x for x in comparisons if not x['matches']],
        intervals='Pointwise Wilson95% Monte Carlo intervals, not simultaneous or scientific-null guarantees',
        ablation_summary=ablation_summary,generalization_summary=grid_summary,
        adaptive_summary=adaptive_pairs,cells=all_rows)
    dump('recount.json',result)
    print(json.dumps({k:result[k] for k in ['raw_rows','binary_decisions','reported_rate_cells','reference_count_mismatches']},indent=2))



if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Recount complete supplementary synthetic comparisons')
    parser.add_argument('--results',required=True,type=Path)
    args=parser.parse_args()
    ROOT=args.results.resolve()
    PACKAGE=Path(__file__).resolve().parent
    PROTOCOL=PACKAGE/'protocol.json'
    main()
