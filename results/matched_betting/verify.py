#!/usr/bin/env python3
"""Read-only independent checks of the matched betting extension.

Reuses the earlier independent audit's primitive simulator/rank reconstruction,
which imports no producer. Betting is recomputed by frequency-weighted log
products, without importing the new betting implementation.
"""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import binom

HERE = Path(__file__).resolve().parent
RESULTS = HERE/'recorded'
ORIGINAL = HERE.parent/'reference_certificate_efficiency/replications.csv.gz'
spec = importlib.util.spec_from_file_location('independent_primitives',
        HERE/'independent_primitives.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
REPS = [0, 1, 17, 67, 123, 211, 257, 299]


def snapshot():
    return {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in HERE.rglob('*') if p.is_file()}


def direct_betting(h, lower, upper, budget, delta, fractions):
    assert np.isfinite(h).all() and np.all(h >= lower-1e-12) and np.all(h <= upper+1e-12)
    threshold = (budget-lower)/(upper-lower)
    assert 0 < threshold < 1  # True for all declared estimated-fit cases.
    values, counts = np.unique(h, return_counts=True)
    capitals = np.array([np.dot(counts, np.log(1+fraction*((values-lower)/(upper-lower)/threshold-1)))
                         for fraction in fractions])
    top = capitals.max()
    log_evalue = float(top+np.log(np.mean(np.exp(capitals-top))))
    probability = min(1., delta+np.exp(-log_evalue)) if log_evalue > 0 else 1.
    return log_evalue, probability


def validate_export():
    manifest = json.loads((HERE/'MANIFEST.json').read_text())
    for name, wanted in manifest['files'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == wanted, name


def main():
    before = snapshot()
    protocol = json.loads((RESULTS/'protocol.json').read_text())
    validate_export()
    frame = pd.read_csv(RESULTS/'replications.csv.gz')
    assert len(frame) == 48000 and frame.replication.nunique() == 300
    keys = ['replication','design','signal','validation_pairs','groups','method']
    assert not frame.duplicated(keys).any()
    assert frame.groupby(keys[1:]).size().eq(300).all()
    assert (frame.total_observations == frame.training_rows+2*frame.validation_pairs+64*frame.groups).all()
    assert np.array_equal(frame.reject, frame.p < .05)
    expected_triples = np.where(frame.method.str.startswith('pooled'), frame.groups*64//3, frame.groups*21)
    assert np.array_equal(frame.effective_triples, expected_triples)
    fractions = np.array(protocol['fixed_betting_fractions'])
    np.testing.assert_array_equal(fractions, np.geomspace(1e-4,.99,64))
    original = pd.read_csv(ORIGINAL)
    original = original[(original.replication < 300) & original.signal.isin(protocol['signals'])
                        & original.groups.isin(protocol['groups']) & original.fit.eq('estimated')]
    original['method'] = original.method.replace({'independent_triples':'independent_bernstein'})
    matched = frame[~frame.method.str.endswith('betting')].merge(original,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(matched) == 28800
    errors = {}
    for column in ['mean','p','radius','lower_bound','kernel_sample_variance','signed_allowance',
                   'kernel_lower','kernel_upper','exact_bias','target']:
        error = float(np.max(abs(matched[column+'_new']-matched[column+'_old'])))
        assert error < 1e-10, (column,error)
        errors[column] = error
    assert np.array_equal(matched.reject_new, matched.reject_old)
    all_bets = frame[frame.method.str.endswith('betting')]
    expected_p = np.minimum(1., .0001+np.exp(-all_bets.log_evalue))
    np.testing.assert_allclose(all_bets.p, expected_p, rtol=1e-10, atol=1e-14)
    max_log_error, max_p_error, rebuilt = 0., 0., 0
    for rep in REPS:
      for di, (design, probabilities, success) in enumerate(audit.DESIGNS):
       for si, signal in enumerate(protocol['signals']):
        seed = [protocol['seed'],rep,di,si]
        z,v,w = audit.random_sample(seed+[0],(1600,64),probabilities,success,signal)
        tz,tv,tw = audit.random_sample(seed+[1],(8192,),probabilities,success,signal)
        vz,vv,vw = audit.random_sample(seed+[2],(8192,2),probabilities,success,signal)
        f,g = audit.fitted_means(tz,tv,len(probabilities)),audit.fitted_means(tz,tw,len(probabilities))
        corners = np.array([(a-f)*(b-g) for a,b in itertools.product((0.,1.),repeat=2)])
        lower,upper = float(corners.min()),float(corners.max())
        for groups in protocol['groups']:
          h = audit.blocks(v[:groups],w[:groups],f[z[:groups]],g[z[:groups]])
          # Explicit index partition: all six roles share one disjoint triple.
          positions = np.arange(groups*64).reshape(groups,64)[:,:63].reshape(-1,3)
          assert np.unique(positions).size == positions.size == 3*len(h)
          pv,pw,pz = [value[:groups].reshape(1,-1) for value in (v,w,z)]
          pooled_h = audit.blocks(pv,pw,f[pz],g[pz])
          for validation in protocol['validation_pairs']:
            budget = audit.budget(f,g,vz[:validation],vv[:validation],vw[:validation])['signed_allowance']
            for method,scores in [('independent_betting',h),('pooled_betting',pooled_h)]:
              selected = frame[(frame.replication == rep) & (frame.design == design) & (frame.signal == signal)
                               & (frame.groups == groups) & (frame.validation_pairs == validation) & (frame.method == method)]
              assert len(selected) == 1
              row = selected.iloc[0]
              log_evalue,probability = direct_betting(scores,lower,upper,budget,.0001,fractions)
              log_error,p_error = abs(log_evalue-row.log_evalue),abs(probability-row.p)
              assert log_error < 1e-7 and p_error < 1e-9, (rep,design,signal,log_error,p_error)
              max_log_error,max_p_error = max(max_log_error,log_error),max(max_p_error,p_error)
              rebuilt += 1
    summary = pd.read_csv(RESULTS/'summary.csv')
    assert len(summary) == 640
    max_ci_error = 0.
    for row in summary.itertuples():
        sub = frame[(frame.design==row.design) & (frame.signal==row.signal)
                    & (frame.groups==row.groups) & (frame.validation_pairs==row.validation_pairs) & (frame.method==row.method)]
        count = int((sub.p<row.level).sum())
        assert count == row.rejections and len(sub) == row.replications == 300
        if count:
            max_ci_error = max(max_ci_error,abs(binom.sf(count-1,300,row.ci_lower)-.025))
        else:
            assert row.ci_lower == 0.
        if count<300:
            max_ci_error = max(max_ci_error,abs(binom.cdf(count,300,row.ci_upper)-.025))
        else:
            assert row.ci_upper == 1.
    assert max_ci_error < 1e-10
    assert before == snapshot(), 'Verifier changed files'
    print(json.dumps(dict(passed=True,read_only=True,records=48000,original_rows_matched=28800,
                          original_row_maximum_discrepancies=errors,stored_betting_probabilities_checked=len(all_bets),
                          independent_primitive_replications=REPS,independent_betting_rows_rebuilt=rebuilt,
                          maximum_rebuilt_log_evalue_error=max_log_error,maximum_rebuilt_p_error=max_p_error,
                          summary_cells_checked=640,maximum_binomial_cdf_interval_error=max_ci_error,
                          scope='Deterministic audit of exposed-stream comparator extension, not an independent power study'),indent=2))


if __name__ == '__main__':
    main()
