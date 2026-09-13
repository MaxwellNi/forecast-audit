"""Recount current design experiments and complete-family decisions.

This verifies stored aggregate arithmetic. It does not regenerate trajectories
or privately held observation patterns. Separate simulation commands provide
seed-to-result replay for the fully synthetic studies.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


def verify(root):
    root = Path(root)
    checks = []
    peer = root / 'results/design_validation/peer_sampling'
    draws = pd.read_csv(peer / 'replications.csv', float_precision='round_trip')
    summary = pd.read_csv(peer / 'summary.csv', float_precision='round_trip')
    assert len(draws) == 2700 and len(summary) == 18
    for row in summary.itertuples():
        cell = draws[(draws.peer_regime == row.peer_regime) & (draws.periods == row.periods)]
        kind = 'shared' if row.method == 'shared_references' else 'distinct'
        values = cell[kind + '_statistic']
        count = int((norm.sf(values) < .05).sum())
        assert len(cell) == row.replications == 300 and count == row.rejections
        np.testing.assert_allclose([row.rate,row.mean_T,row.sd_T,row.mean_score],
                                   [count/len(cell),values.mean(),values.std(ddof=1),cell[kind+'_mean'].mean()],
                                   rtol=1e-12,atol=1e-14)
        z=norm.ppf(.975);p=count/len(cell);den=1+z*z/len(cell)
        center=(p+z*z/(2*len(cell)))/den
        half=z*np.sqrt(p*(1-p)/len(cell)+z*z/(4*len(cell)**2))/den
        np.testing.assert_allclose([row.wilson_low,row.wilson_high],
                                   [max(0.,center-half),min(1.,center+half)],rtol=0,atol=1e-14)
    checks.append(dict(study='peer_sampling',records=len(draws),cells=len(summary),arithmetic='PASS'))

    raw = root / 'results/equal_entity_directional'
    draws = pd.read_csv(raw / 'replications.csv.gz', float_precision='round_trip')
    summary = pd.read_csv(raw / 'summary.csv', float_precision='round_trip')
    keys = ['entities','times','lookback','injected_association','method']
    groups = draws.groupby(keys)
    assert len(draws) == 42000 and len(summary) == 84
    for row in summary.itertuples():
        cell = groups.get_group(tuple(getattr(row,key) for key in keys))
        count = int((cell.statistic > norm.ppf(.95)).sum())
        assert len(cell) == row.replications == 500 and count == row.rejections
        np.testing.assert_allclose([row.rejection_rate,row.mean_statistic,row.sd_statistic,row.mean_score],
                                   [count/len(cell),cell.statistic.mean(),cell.statistic.std(ddof=1),cell.mean_entity_score.mean()],
                                   rtol=1e-12,atol=1e-14)
        assert cell.evaluation_rows_per_entity.eq(3*row.times//5).all()
    checks.append(dict(study='equal_entity_directional',records=len(draws),cells=len(summary),arithmetic='PASS'))

    directional = root / 'results/directional_comparison'
    models = pd.read_csv(directional/'model_comparison.csv',float_precision='round_trip',keep_default_na=False,na_values=[''])
    domain_summary = pd.read_csv(directional/'domain_summary.csv')
    assert len(models) == 164 and len(domain_summary) == 16
    for (domain,configuration),cell in models.groupby(['domain','configuration']):
        summary_row = domain_summary[(domain_summary.domain==domain)&(domain_summary.configuration==configuration)].iloc[0]
        assert cell.family_size.eq(len(cell)).all()
        np.testing.assert_allclose(cell.guarded_p,np.where(cell.final_label.eq('ABSTAIN'),1.,cell.normal_p_one_sided),rtol=0,atol=1e-15)
        reject,adjusted,_,_=multipletests(cell.guarded_p,alpha=.05,method='fdr_by')
        np.testing.assert_allclose(cell.BY_adjusted_p,adjusted,rtol=1e-12,atol=1e-14)
        np.testing.assert_array_equal(cell.final_label.eq('RETAIN'),reject)
        for label in ['RETAIN','NOT_RETAINED','ABSTAIN']:
            assert int(cell.final_label.eq(label).sum()) == int(summary_row[label])
        assert cell.loc[cell.standard_error<=1e-10,'final_label'].eq('ABSTAIN').all()
        assert cell.loc[cell.standard_error<=1e-10,'guarded_p'].eq(1).all()
    for (_, _),model in models.groupby(['domain','model']):
        indexed=model.set_index('configuration')
        for support in ['all','middle']:
            assert indexed.loc['complementary_'+support,'evaluation_rows']==indexed.loc['directional_'+support,'evaluation_rows']
    checks.append(dict(study='directional_comparison',model_configurations=len(models),families=len(domain_summary),full_family_BY='PASS',matched_support='PASS'))

    discrepancies=pd.read_csv(root/'results/restricted_aggregate/solver_comparison.csv')
    assert len(discrepancies)==16 and discrepancies.maximum_absolute_difference.max()<3.2e-10
    checks.append(dict(study='stored_cross_solver_comparison',quantities=16,maximum=float(discrepancies.maximum_absolute_difference.max()),source_observations_replayed=False))
    return checks


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=dict(status='PASS',checks=verify(args.root),scope='Stored aggregate verification, not source-data retraining or error-rate validation.')
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
