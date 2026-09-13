"""Complete the missing original absolute-allowance/variance-U factorial cell.

No random generation: retain the three original constructions, add one using
the exact same stored U center, fitted range, disjoint variance and validation
allowance. This is post-exposure analysis of original streams.
"""
from pathlib import Path
from datetime import datetime,timezone
from itertools import combinations
import argparse,hashlib,json,platform,time
import numpy as np
import pandas as pd
from scipy.stats import beta

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE/'inputs/original_efficiency'
BETTING=HERE/'inputs/matched_betting'
SETTINGS=['design','signal','fit','training_rows','validation_pairs','groups','peers','total_observations']
INDEX=['replication']+SETTINGS
METHODS=['absolute_range','signed_range','absolute_variance','signed_variance']
PAIR_NAMES={('absolute_range','signed_range'):'learning_at_range',
 ('absolute_range','absolute_variance'):'sampling_at_absolute',
 ('absolute_range','signed_variance'):'total_upgrade',
 ('signed_range','absolute_variance'):'crossed_comparison',
 ('signed_range','signed_variance'):'sampling_at_signed',
 ('absolute_variance','signed_variance'):'learning_at_variance'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def write_new(path,data):
    with Path(path).open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
def csv(frame,path):
    frame.to_csv(path,index=False,float_format='%.17g',
        compression={'method':'gzip','mtime':0} if str(path).endswith('.gz') else None)


def cp(events,n,alpha=.05):
    return (0. if events==0 else float(beta.ppf(alpha/2,events,n-events+1)),
            1. if events==n else float(beta.ppf(1-alpha/2,events+1,n-events)))


def paired_interval(plus,minus,n):
    """At least 95% coverage: two 97.5% CP marginals for discordant cells.

    N+ and N- are dependent multinomial cells, each marginal binomial(n,p).
    Bonferroni covers both probabilities; subtract their confidence limits.
    Never use an independent two-proportion standard error.
    """
    assert 0<=plus and 0<=minus and plus+minus<=n
    lp,up=cp(plus,n,.025);lm,um=cp(minus,n,.025)
    return max(-1.,lp-um),min(1.,up-lm)


def probabilities(mean,bias,s2,j,width,variance):
    margin=np.maximum(mean-bias,0.)
    if not variance:
        radius=width*np.sqrt(np.log(1/(.05-.0001))/(2*j))
        p=np.minimum(1.,.0001+np.exp(-2*j*(margin/width)**2))
    else:
        a=np.sqrt(2*s2/j);c=2*width/np.sqrt(j*(j-1))+width/(3*j)
        x=np.log(2/(.05-.0001))
        radius=np.minimum(width*np.sqrt(x/(2*j)),a*np.sqrt(x)+c*x)
        root=np.divide(2*margin,a+np.sqrt(a*a+4*c*margin),out=np.zeros_like(margin),where=margin>0)
        exponent=np.maximum(root*root,2*j*(margin/width)**2)
        p=np.minimum(1.,.0001+2*np.exp(-exponent))
    return p,radius,mean-bias-radius


def summary(frame):
    rows=[]
    for key,g in frame.groupby(SETTINGS+['method'],sort=False):
        n=len(g);events=int(g.reject.sum());lo,hi=cp(events,n)
        rows.append(dict(zip(SETTINGS+['method'],key),replications=n,rejections=events,power=events/n,
            power_ci_lower=lo,power_ci_upper=hi,target=float(g.target.iloc[0]),
            mean=float(g['mean'].mean()),bias=float(g.exact_bias.mean()),
            bias_allowance=float(g.bias_upper.mean()),absolute_allowance=float(g.absolute_allowance.mean()),
            signed_allowance=float(g.signed_allowance.mean()),radius=float(g.radius.mean()),
            lower=float(g.lower_bound.mean()),coverage_failures=int((~g.lower_covers).sum()) if g.lower_bound.notna().all() else np.nan,
            allowance_failures=int((~g.allowance_covers).sum())))
    return pd.DataFrame(rows)


def paired(frame,pairs,include_interaction=False):
    rows=[];attributions=[]
    for key,g in frame.groupby(SETTINGS,sort=False):
        definitions=dict(zip(SETTINGS,key));pivot=g.pivot(index='replication',columns='method',values='reject').astype(int)
        n=len(pivot)
        for before,after in pairs:
            a,b=pivot[before],pivot[after];plus=int(((a==0)&(b==1)).sum());minus=int(((a==1)&(b==0)).sum())
            lo,hi=paired_interval(plus,minus,n)
            rows.append(dict(**definitions,contrast=PAIR_NAMES.get((before,after),after+'_minus_'+before),
                before=before,after=after,replications=n,before_rejections=int(a.sum()),after_rejections=int(b.sum()),
                after_only=plus,before_only=minus,both=int(((a==1)&(b==1)).sum()),neither=int(((a==0)&(b==0)).sum()),
                delta=(plus-minus)/n,ci_lower=lo,ci_upper=hi,confidence=.95,
                ci_method='paired discordant multinomial; Bonferroni of two 97.5% exact CP marginals; pointwise'))
        if include_interaction:
            ar,sr,av,sv=[pivot[m] for m in METHODS]
            assert np.all(sr>=ar) and np.all(sv>=av)
            d=(sv-av)-(sr-ar)
            assert set(d.unique()).issubset({-1,0,1})
            plus,minus=int((d==1).sum()),int((d==-1).sum());lo,hi=paired_interval(plus,minus,n)
            low=g.pivot(index='replication',columns='method',values='lower_bound')
            rad=g.pivot(index='replication',columns='method',values='radius')
            allowance=g.pivot(index='replication',columns='method',values='bias_upper')
            learning=allowance.absolute_range-allowance.signed_range
            sampling=rad.absolute_range-rad.absolute_variance
            total=low.signed_variance-low.absolute_range
            assert np.max(abs(total-learning-sampling))<1e-12
            learning_average=.5*((sr-ar)+(sv-av));sampling_average=.5*((av-ar)+(sv-sr))
            assert np.array_equal(learning_average+sampling_average,sv-ar)
            lr=np.sqrt(np.log(40)/(2*n));sradius=np.sqrt(2*np.log(40)/n)
            la,sa=float(learning_average.mean()),float(sampling_average.mean())
            attributions.append(dict(**definitions,replications=n,target=float(g.target.iloc[0]),
                learning_lcb_gain=float(learning.mean()),sampling_lcb_gain=float(sampling.mean()),
                total_lcb_gain=float(total.mean()),lcb_additivity_max_error=float(np.max(abs(total-learning-sampling))),
                interaction=float(d.mean()),interaction_plus=plus,interaction_minus=minus,
                interaction_ci_lower=lo,interaction_ci_upper=hi,
                learning_average_delta=la,learning_average_ci_lower=max(0.,la-lr),learning_average_ci_upper=min(1.,la+lr),
                sampling_average_delta=sa,sampling_average_ci_lower=max(-1.,sa-sradius),sampling_average_ci_upper=min(1.,sa+sradius),
                average_ci_method='pointwise bounded Hoeffding on paired per-replication factor averages; widths1 and2',
                interaction_ci_method='paired ternary contrast; Bonferroni of two97.5%CP marginal intervals; pointwise'))
    return pd.DataFrame(rows),pd.DataFrame(attributions)


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();protocol=json.loads((HERE/'provenance/protocol.json').read_text())
    mapping=json.loads((HERE/'provenance/input_path_map.json').read_text())
    files=[HERE/path for path in mapping.values()];before={str(p):sha(p) for p in files}
    for key,path in mapping.items():assert sha(HERE/path)==protocol['input_sha256'][key]
    write_new(output/'run_started.json',dict(started_utc=utc(),protocol_sha256=sha(HERE/'provenance/protocol.json')))
    old=pd.read_csv(ORIGINAL/'replications.csv.gz')
    assert len(old)==500000 and old.replication.nunique()==1000
    base=old[old.method=='signed_variance'].copy();assert len(base)==100000
    assert not base.duplicated(INDEX).any() and len(base[SETTINGS].drop_duplicates())==100
    width=(base.kernel_upper-base.kernel_lower).to_numpy();j=base.effective_triples.to_numpy();s2=base.kernel_sample_variance.to_numpy()
    p,radius,lower=probabilities(base['mean'].to_numpy(),base.absolute_allowance.to_numpy(),s2,j,width,True)
    np.testing.assert_allclose(radius,base.radius,rtol=1e-13,atol=1e-14)
    base['method']='absolute_variance';base['bias_upper']=base.absolute_allowance
    base['p']=p;base['lower_bound']=lower;base['reject']=p<.05
    base['lower_covers']=base.lower_bound<=base.target;base['allowance_covers']=base.exact_bias<=base.bias_upper
    original=old[old.method.isin(['absolute_range','signed_range','signed_variance'])]
    full=pd.concat([original,base],ignore_index=True).sort_values(INDEX+['method'],kind='stable')
    assert len(full)==400000
    csv(full,output/'factorial_rows.csv.gz')
    sums=summary(full);csv(sums,output/'summary.csv')
    paired_rows,attribution=paired(full,list(combinations(METHODS,2)),True)
    csv(paired_rows,output/'paired_differences.csv');csv(attribution,output/'attribution.csv')
    original_summary=pd.read_csv(ORIGINAL/'summary.csv')
    csv(original_summary[original_summary.method.isin(['independent_triples','pooled_variance'])],output/'classical_context.csv')
    def select_settings(specs):
        pieces=[]
        for number,spec in enumerate(specs):
            use=np.ones(len(sums),dtype=bool)
            for name,value in spec.items():use&=sums[name]==value
            piece=sums.loc[use].copy();assert len(piece)==4;piece['display_order']=number;pieces.append(piece)
        return pd.concat(pieces,ignore_index=True)
    csv(select_settings(protocol['figure_settings']),output/'main_figure_data.csv')
    csv(select_settings(protocol['persistent_table_settings']),output/'persistent_table.csv')
    # All existing matched-matched comparator cells, no source-generating call and no new tuning.
    matched=pd.read_csv(BETTING/'replications.csv.gz')
    assert len(matched)==48000
    matched['fit']='estimated';matched['bias_upper']=matched.signed_allowance
    matched['absolute_allowance']=np.nan;matched['lower_covers']=matched.lower_bound<=matched.target
    matched['allowance_covers']=matched.exact_bias<=matched.bias_upper
    subset=full[(full.fit=='estimated')&(full.replication<300)&full.signal.isin([-.125,0.,.125,.25])&full.groups.isin([100,400])]
    check=subset[subset.method=='signed_variance'].merge(matched[matched.method=='signed_variance'],on=INDEX,suffixes=('_original','_matched'),validate='one_to_one')
    assert len(check)==9600
    np.testing.assert_allclose(check.p_original,check.p_matched,rtol=1e-10,atol=1e-12)
    context=pd.concat([subset,matched[matched.method!='signed_variance']],ignore_index=True)
    assert len(context)==76800
    csv(summary(context),output/'matched300_comparator_summary.csv')
    context_pairs,_=paired(context,[tuple(x) for x in protocol['secondary_pairs']])
    csv(context_pairs,output/'matched300_paired_differences.csv')
    outputs=['factorial_rows.csv.gz','summary.csv','paired_differences.csv','attribution.csv','classical_context.csv',
             'main_figure_data.csv','persistent_table.csv','matched300_comparator_summary.csv','matched300_paired_differences.csv']
    assert before=={str(p):sha(p) for p in files}
    write_new(output/'completion_receipt.json',dict(completed_utc=utc(),elapsed_seconds=time.perf_counter()-start,
        source_bytes_unchanged=True,factorial_rows=400000,method_cells=400,paired_contrast_cells=len(paired_rows),
        attribution_cells=len(attribution),matched300_method_cells=256,matched300_paired_cells=len(context_pairs),
        output_sha256={name:sha(output/name) for name in outputs},protocol_sha256=sha(HERE/'provenance/protocol.json')))
    print('Complete:',len(full),'rows;',len(sums),'cells;',len(paired_rows),'paired contrasts')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New directory for reconstructed factorial outputs')
    args=parser.parse_args();run(args.output)
