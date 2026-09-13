"""Independent factorial verification, without analysis or producer imports.

Binomial confidence limits are inverted using binomial tails and root finding,
not the analysis beta-quantile implementation. Selected original validation
primitives use only the earlier independent audit's numeric comparisons/LP.
"""
from pathlib import Path
from functools import lru_cache
import argparse,hashlib,importlib.util,json,platform,time
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binom

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE/'inputs/original_efficiency'
BETTING=HERE/'inputs/matched_betting'
SETTINGS=['design','signal','fit','training_rows','validation_pairs','groups','peers','total_observations']
INDEX=['replication']+SETTINGS
METHODS=['absolute_range','signed_range','absolute_variance','signed_variance']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@lru_cache(None)
def cp(k,n,alpha=.05):
    lower=0. if k==0 else brentq(lambda p:binom.sf(k-1,n,p)-alpha/2,0.,1.,xtol=5e-15)
    upper=1. if k==n else brentq(lambda p:binom.cdf(k,n,p)-alpha/2,0.,1.,xtol=5e-15)
    return lower,upper


def paired_limits(plus,minus,n):
    lp,up=cp(plus,n,.025);lm,um=cp(minus,n,.025)
    return max(-1.,lp-um),min(1.,up-lm)


def expected_inference(frame):
    mu=frame['mean'].to_numpy();b=frame.bias_upper.to_numpy();j=frame.effective_triples.to_numpy()
    r=(frame.kernel_upper-frame.kernel_lower).to_numpy();margin=np.maximum(mu-b,0.)
    variance=frame.method.str.endswith('variance').to_numpy();rad=np.empty(len(frame));p=np.empty(len(frame))
    rad[~variance]=r[~variance]*np.sqrt(-np.log(.05-.0001)/(2*j[~variance]))
    p[~variance]=np.minimum(1,.0001+np.exp(-2*j[~variance]*(margin[~variance]/r[~variance])**2))
    jj,rr,d=j[variance],r[variance],margin[variance];s2=frame.kernel_sample_variance.to_numpy()[variance]
    x=np.log(2/(.05-.0001));aa=np.sqrt(2*s2/jj);cc=rr*(2/np.sqrt(jj*(jj-1))+1/(3*jj))
    rad[variance]=np.minimum(rr*np.sqrt(x/(2*jj)),aa*np.sqrt(x)+cc*x)
    exponent=((np.sqrt(aa*aa+4*cc*d)-aa)/(2*cc))**2
    exponent=np.maximum(exponent,2*jj*(d/rr)**2)
    p[variance]=np.minimum(1,.0001+2*np.exp(-exponent))
    return p,rad,mu-b-rad


def verify_pairs(frame,table):
    maxima=0.
    for key,g in frame.groupby(SETTINGS):
        use=np.ones(len(table),bool)
        for name,value in zip(SETTINGS,key):use&=table[name]==value
        rows=table[use];pivot=g.pivot(index='replication',columns='method',values='reject').astype(int)
        n=len(pivot)
        for row in rows.itertuples():
            a,b=pivot[row.before],pivot[row.after];plus=int(((a==0)&(b==1)).sum());minus=int(((a==1)&(b==0)).sum())
            assert (row.replications,row.after_only,row.before_only)==(n,plus,minus)
            assert row.both==int(((a==1)&(b==1)).sum()) and row.neither==int(((a==0)&(b==0)).sum())
            assert row.before_rejections==int(a.sum()) and row.after_rejections==int(b.sum())
            lo,hi=paired_limits(plus,minus,n)
            errors=[abs(row.delta-(plus-minus)/n),abs(row.ci_lower-lo),abs(row.ci_upper-hi)]
            maxima=max(maxima,*errors)
    assert maxima<1e-11,maxima
    return maxima


def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();protocol=json.loads((HERE/'provenance/protocol.json').read_text());receipt=json.loads((HERE/'provenance/completion_receipt.json').read_text())
    assert protocol['code_sha256']==json.loads((HERE/'provenance/export_trace.json').read_text())['historical_analysis_sha256']
    assert sha(HERE/'provenance/protocol.json')==receipt['protocol_sha256']
    for name,digest in receipt['output_sha256'].items():assert sha(HERE/name)==digest,name
    for name,digest in protocol['input_sha256'].items():
        path=HERE/json.loads((HERE/'provenance/input_path_map.json').read_text())[name]
        assert sha(path)==digest,name
    started=json.loads((HERE/'provenance/run_started.json').read_text())
    assert protocol['frozen_utc']<started['started_utc']<receipt['completed_utc']
    old=pd.read_csv(ORIGINAL/'replications.csv.gz');full=pd.read_csv(HERE/'factorial_rows.csv.gz')
    assert len(full)==400000 and full.replication.nunique()==1000 and len(full[SETTINGS].drop_duplicates())==100
    assert not full.duplicated(INDEX+['method']).any() and set(full.method)==set(METHODS)
    assert full.groupby(SETTINGS+['method']).size().eq(1000).all()
    copied=full[full.method!='absolute_variance'].set_index(INDEX+['method']).sort_index()
    original=old[old.method.isin(copied.index.get_level_values('method'))].set_index(INDEX+['method']).sort_index()
    pd.testing.assert_frame_equal(copied,original,check_exact=False,rtol=1e-12,atol=1e-13)
    added=full[full.method=='absolute_variance'].set_index(INDEX).sort_index()
    signed=old[old.method=='signed_variance'].set_index(INDEX).sort_index()
    for col in ['mean','kernel_lower','kernel_upper','effective_triples','kernel_sample_variance','absolute_allowance','signed_allowance']:
        np.testing.assert_allclose(added[col],signed[col],rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(added.bias_upper,added.absolute_allowance,rtol=0,atol=0)
    assert (full.signed_allowance<=full.absolute_allowance+1e-14).all()
    p,rad,lower=expected_inference(full)
    errors={name:float(np.max(abs(a-full[name]))) for name,a in [('p',p),('radius',rad),('lower_bound',lower)]}
    assert max(errors.values())<2e-11,errors
    np.testing.assert_array_equal(full.reject,p<.05)
    np.testing.assert_array_equal(full.lower_covers,lower<=full.target)
    np.testing.assert_array_equal(full.allowance_covers,full.exact_bias<=full.bias_upper)
    assert (abs(full.exact_bias)<=full.absolute_allowance).all()
    summary=pd.read_csv(HERE/'summary.csv');assert len(summary)==400
    summary_error=0.
    for key,g in full.groupby(SETTINGS+['method']):
        use=np.ones(len(summary),bool)
        for name,value in zip(SETTINGS+['method'],key):use&=summary[name]==value
        rows=summary[use];assert len(rows)==1;row=rows.iloc[0]
        n=len(g);events=int(g.reject.sum());lo,hi=cp(events,n)
        assert row.replications==n and row.rejections==events
        assert row.coverage_failures==int((~g.lower_covers).sum()) and row.allowance_failures==int((~g.allowance_covers).sum())
        for name,value in [('power',events/n),('power_ci_lower',lo),('power_ci_upper',hi),('target',g.target.iloc[0]),
              ('mean',g['mean'].mean()),('bias',g.exact_bias.mean()),('bias_allowance',g.bias_upper.mean()),
              ('absolute_allowance',g.absolute_allowance.mean()),('signed_allowance',g.signed_allowance.mean()),
              ('radius',g.radius.mean()),('lower',g.lower_bound.mean())]:
            summary_error=max(summary_error,abs(float(value)-row[name]))
    assert summary_error<1e-11
    contrasts=pd.read_csv(HERE/'paired_differences.csv');assert len(contrasts)==600
    contrast_error=verify_pairs(full,contrasts)
    attribution=pd.read_csv(HERE/'attribution.csv');assert len(attribution)==100;interaction_error=0.
    for key,g in full.groupby(SETTINGS):
        use=np.ones(len(attribution),bool)
        for name,value in zip(SETTINGS,key):use&=attribution[name]==value
        row=attribution[use].iloc[0];pivot=g.pivot(index='replication',columns='method',values='reject').astype(int)
        ar,sr,av,sv=[pivot[m] for m in METHODS];assert (sr>=ar).all() and (sv>=av).all()
        d=(sv-av)-(sr-ar);assert set(d).issubset({-1,0,1})
        plus,minus=int((d==1).sum()),int((d==-1).sum());lo,hi=paired_limits(plus,minus,len(d))
        assert row.interaction_plus==plus and row.interaction_minus==minus
        for field,value in [('interaction',d.mean()),('interaction_ci_lower',lo),('interaction_ci_upper',hi)]:
            interaction_error=max(interaction_error,abs(row[field]-value))
        low=g.pivot(index='replication',columns='method',values='lower_bound');rr=g.pivot(index='replication',columns='method',values='radius')
        bb=g.pivot(index='replication',columns='method',values='bias_upper')
        learning=bb.absolute_range-bb.signed_range;sampling=rr.absolute_range-rr.absolute_variance
        total=low.signed_variance-low.absolute_range
        assert np.max(abs(total-learning-sampling))<1e-12
        n=len(d);la=float(((sr-ar+sv-av)/2).mean());sa=float(((av-ar+sv-sr)/2).mean())
        lr=np.sqrt(np.log(40)/(2*n));sradius=np.sqrt(2*np.log(40)/n)
        for field,value in [('learning_lcb_gain',learning.mean()),('sampling_lcb_gain',sampling.mean()),('total_lcb_gain',total.mean()),
                ('learning_average_delta',la),('sampling_average_delta',sa),('learning_average_ci_lower',max(0,la-lr)),
                ('learning_average_ci_upper',min(1,la+lr)),('sampling_average_ci_lower',max(-1,sa-sradius)),
                ('sampling_average_ci_upper',min(1,sa+sradius))]:interaction_error=max(interaction_error,abs(row[field]-value))
    assert interaction_error<1e-11
    # Independent reconstruction of original training/validation only, using
    # separate exact seed suffixes. Evaluation streams need not be consumed.
    spec=importlib.util.spec_from_file_location('old_independent_audit',HERE/'independent_primitives.py')
    ind=importlib.util.module_from_spec(spec);spec.loader.exec_module(ind)
    budget_rows=[];budget_error=0.
    for rep in [0,1,17,123,317,503,777,999]:
      for di,(design,prob,success) in enumerate(ind.DESIGNS):
       for si,signal in enumerate(ind.SIGNALS):
        seed=[202609121457,rep,di,si]
        tz,tv,tw=ind.random_sample(seed+[1],(8192,),prob,success,signal)
        z,v,w=ind.random_sample(seed+[2],(8192,2),prob,success,signal)
        fv,fw=ind.fitted_means(tz,tv,len(prob)),ind.fitted_means(tz,tw,len(prob))
        for fit in ['estimated']+(['opposed_shifts'] if di==0 else []):
         f,g=(fv,fw) if fit=='estimated' else (np.clip(fv+.15,0,1),np.clip(fw-.15,0,1))
         for nv in ([512,8192] if fit=='estimated' else [8192]):
          with np.errstate(divide='ignore',invalid='ignore'):b=ind.budget(f,g,z[:nv],v[:nv],w[:nv])
          sub=full[(full.replication==rep)&(full.design==design)&(full.signal==signal)&(full.fit==fit)&(full.validation_pairs==nv)]
          assert len(sub)==16
          for col in ['absolute_allowance','signed_allowance','bias_interval_width']:
            budget_error=max(budget_error,float(np.max(abs(sub[col]-b[col]))))
          budget_rows.append(dict(replication=rep,design=design,signal=signal,fit=fit,validation_pairs=nv,**b))
    assert len(budget_rows)==200 and budget_error<1e-11
    pd.DataFrame(budget_rows).to_csv(output/'independent_validation_rebuild.csv',index=False,float_format='%.17g')
    # Existing 300-rep comparator stream remains a separately labelled context.
    matched=pd.read_csv(BETTING/'replications.csv.gz');matched['fit']='estimated'
    subset=full[(full.fit=='estimated')&(full.replication<300)&full.signal.isin([-.125,0.,.125,.25])&full.groups.isin([100,400])]
    context=pd.concat([subset,matched[matched.method!='signed_variance']],ignore_index=True)
    sec=pd.read_csv(HERE/'matched300_comparator_summary.csv');assert len(sec)==256
    for key,g in context.groupby(SETTINGS+['method']):
        use=np.ones(len(sec),bool)
        for name,value in zip(SETTINGS+['method'],key):use&=sec[name]==value
        row=sec[use].iloc[0];assert row.replications==300 and row.rejections==int(g.reject.sum())
        lo,hi=cp(int(g.reject.sum()),300)
        assert abs(row.power_ci_lower-lo)<1e-11 and abs(row.power_ci_upper-hi)<1e-11
        if row.method.endswith('betting'):assert np.isnan(row.coverage_failures) and np.isnan(row.radius)
    secpair=pd.read_csv(HERE/'matched300_paired_differences.csv');assert len(secpair)==128
    secondary_error=verify_pairs(context,secpair)
    for name,specifications in [('main_figure_data.csv',protocol['figure_settings']),('persistent_table.csv',protocol['persistent_table_settings'])]:
        displayed=pd.read_csv(HERE/name);assert len(displayed)==4*len(specifications)
        for order,definition in enumerate(specifications):
            d=displayed[displayed.display_order==order];assert len(d)==4
            for k,v in definition.items():assert (d[k]==v).all()
            for row in d.itertuples():
                subset=summary
                for k in SETTINGS+['method']:subset=subset[subset[k]==getattr(row,k)]
                assert len(subset)==1 and row.rejections==subset.rejections.iloc[0]
    result=dict(status='PASS',scope='Independent stored-row arithmetic, direct binomial-tail CI inversion and200original primitive training/validation reconstructions; no producer or factorial analysis imports.',
        source_rows_copied=300000,new_absolute_variance_rows=100000,total_rows=400000,total_method_cells=400,
        nonpositive_method_cells=int((summary.target<=0).sum()),nonpositive_rejections=int(summary.loc[summary.target<=0,'rejections'].sum()),
        coverage_failures=int(summary.coverage_failures.sum()),absolute_and_signed_validation_rebuilds=200,
        max_allowance_rebuild_error=budget_error,max_inference_errors=errors,max_summary_error=summary_error,
        paired_contrast_cells=600,max_paired_ci_error=contrast_error,interaction_cells=100,max_interaction_error=interaction_error,
        secondary_cells=256,secondary_paired_cells=128,max_secondary_ci_error=secondary_error,
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),elapsed_seconds=time.perf_counter()-start,
        protocol_sha256=sha(HERE/'provenance/protocol.json'),completion_receipt_sha256=sha(HERE/'provenance/completion_receipt.json'),verifier_sha256=sha(Path(__file__)))
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.output)
