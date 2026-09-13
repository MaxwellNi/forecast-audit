"""Verify frozen provenance, BY, selection, identity and aggregate arithmetic."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out, write_result=False):
    protocol = json.loads((out/'protocol.json').read_text())
    source = json.loads((out/'source_receipt.json').read_text())
    selection = json.loads((out/'selection.json').read_text())
    started = json.loads((out/'confirmation_started.json').read_text())
    completed = json.loads((out/'confirmation_receipt.json').read_text())
    timing = json.loads((out/'forecast_timing_check.json').read_text())
    chronology = [protocol['frozen_utc'], source['downloaded_utc'], selection['selection_frozen_utc'], timing['verified_utc'],
                  started['started_utc'], completed['completed_utc']]
    assert all(datetime.fromisoformat(a) < datetime.fromisoformat(b) for a, b in zip(chronology, chronology[1:]))
    assert sha(out/'study.py') == protocol['script_sha256'] == completed['script_sha256']
    assert sha(out/'source.zip') == source['sha256'] == completed['source_sha256']
    assert sha(out/'fitted_models.pkl') == selection['models_sha256']
    assert sha(out/'selection.json') == completed['selection_sha256']
    protocol_sha = sha(out/'protocol.json')
    selection_sha = sha(out/'selection.json')
    assert protocol_sha == completed['protocol_sha256']
    assert all(receipt['protocol_sha256'] == protocol_sha for receipt in
               [source, selection, timing, started, completed])
    assert selection['source_sha256'] == sha(out/'source.zip')
    assert selection['selection_csv_sha256'] == sha(out/'selection_all_candidates.csv')
    assert timing['script_sha256'] == sha(out/'study.py')
    assert started['selection_sha256'] == selection_sha
    assert source['bytes'] == (out/'source.zip').stat().st_size
    assert source['source_url'] == protocol['source_url']
    for name in ('protocol', 'selection'):
        digest, filename = (out/(name+'.sha256')).read_text().split()
        assert filename == name+'.json' and digest == sha(out/filename)
    original = json.loads((out/'ORIGINAL_RUN_HASHES.json').read_text())
    preserved = ['source.zip', 'fitted_models.pkl'] + sorted(p.name for p in out.glob('*.csv'))
    for filename in preserved:
        assert sha(out/filename) == original['files'][filename]['sha256']
    for receipt in [protocol, source, selection, timing, started, completed]:
        assert receipt['public_packaging']['original_run_hashes'] == 'ORIGINAL_RUN_HASHES.json'
    assert not selection['confirmation_outcomes_parsed']
    assert not timing['confirmation_outcomes_parsed']
    for filename, digest in completed['outputs_sha256'].items():
        assert sha(out/filename) == digest
    s = pd.read_csv(out/'selection_all_candidates.csv')
    assert len(s) == 16 and len(s[s.guard]) == 5 and np.all(s.loc[s.guard, 'p_raw'] == 1.)
    ordered = s.sort_values('p_raw', kind='stable')
    h = sum(1/j for j in range(1, 17))
    raw = ordered.p_raw.to_numpy()
    # Literal minimum over every admissible larger rank: independent BY implementation.
    adjusted = [min(1., min(16*h*raw[j]/(j+1) for j in range(i, 16))) for i in range(16)]
    assert np.allclose(adjusted, ordered.p_by, rtol=1e-10, atol=1e-14)
    assert np.array_equal(s.gate, (s.p_by <= .05)&(s.theta_select > 0)&~s.guard)
    assert int(s.gate.sum()) == 1
    for baseline, choices in selection['choices'].items():
        family = s[s.baseline == baseline]
        b = family.mse_baseline_select.iloc[0]
        for strategy, metric in [('convex','mse_convex_select'),('augment','mse_augment_select'),('gated','mse_augment_select')]:
            eligible = family[family.gate] if strategy == 'gated' else family
            winner = eligible.sort_values(metric,kind='stable').iloc[0] if len(eligible) else None
            expected = winner.candidate if winner is not None and winner[metric] < b else 'baseline'
            assert choices[strategy] == expected
    identity = pd.read_csv(out/'gain_identity.csv')
    assert len(identity) == 16
    assert np.allclose(identity.gain_confirm, identity.quadratic_identity, rtol=1e-10, atol=1e-6)
    metrics = pd.read_csv(out/'confirmation_metrics.csv')
    trace = pd.read_csv(out/'weekly_trace.csv')
    assert len(metrics) == 74 and len(trace) == 74*12
    aggregated = trace.groupby(['baseline','method'])[['mse','mae']].mean()
    for row in metrics.itertuples():
        independent = aggregated.loc[(row.baseline,row.method)]
        assert np.allclose([row.mse,row.mae],independent,rtol=1e-12,atol=1e-8)
        assert np.isclose(row.rmse**2,row.mse,rtol=1e-12)
    paired = pd.read_csv(out/'paired_risk_improvements.csv')
    assert len(paired) == 82
    ix = metrics.set_index(['baseline','method'])
    for row in paired.itertuples():
        expected = ix.loc[(row.baseline,row.comparator),'mse']-ix.loc[(row.baseline,row.method),'mse']
        assert np.isclose(row.gain,expected,rtol=1e-10,atol=1e-7)
        w = trace[trace.baseline == row.baseline].pivot(index='week',columns='method',values='mse')
        delta=(w[row.comparator]-w[row.method]).to_numpy()
        centered=delta-delta.mean()
        # Bartlett HAC lag1 on12 weekly means, including12/11 correction.
        variance=max(0.,(np.mean(centered**2)+np.dot(centered[:-1],centered[1:])/12)/11)
        assert np.isclose(row.weekly_se,np.sqrt(variance),rtol=1e-10,atol=1e-7)
    result={'passed':True,'chronology':chronology,'checks':{
        'public_receipt_hash_links':True,'sidecar_hash_files':2,
        'source_model_and_result_files_preserved':len(preserved),'full_family_BY':16,'copy_guard_family_members':5,
        'fixed_selection_choices':6,'literal_gain_identities':16,'risk_metric_rows':74,
        'weekly_trace_rows':888,'paired_gain_and_HAC_rows':82},
        'scope':'Public receipt-link and aggregate arithmetic verification. Timestamps refer to the recorded original run; public packaging edits are identified in ORIGINAL_RUN_HASHES.json. No new outcome selection, population nuisance oracle, temporal independence or external preregistration is established.'}
    if write_result:
        (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--write-result',action='store_true',help='Refresh verification.json; default is read-only')
    args=parser.parse_args()
    main(args.output,args.write_result)
