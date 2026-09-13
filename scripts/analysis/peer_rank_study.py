"""Frozen synthetic study of conditional rank products and reference reuse.

The iid-row experiments have known conditional nuisance means. A separate
fixed-grid experiment intentionally violates the iid-reference assumption.
These results do not replace the published real-data audit target.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm
from peer_rank_products import peer_rank_products


CONDITIONS = ['gaussian_null', 'gaussian_alternative', 'linear_copy',
              'nonmonotone_copy', 'interaction_copy']
SOURCE = Path(__file__).resolve()
DEFAULT_OUTPUT = SOURCE.parents[2] / 'results/reference_rank_audit'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'replications': 300, 'seed_base': 202609059830000,
        'entities': [8,32,128], 'clusters': [25,100,400],
        'conditions': CONDITIONS, 'normal_tail': 'one-sided; sample cluster SD with ddof=1',
        'fixed_grid_boundary': {'entities':8,'clusters':6400,'noise_sd':.1,
                                'description':'X=B+noise,Y=-B+noise; fixed grid, iid-peer premise false'},
        'methods': ['naive_normal','distinct_reference_normal','distinct_reference_finite'],
        'numeric_score_allowance': 1e-10,
        'nuisance_status': 'known means; copy conditions use exact forecast mean and outcome mean .5',
        'estimand': 'conditional covariance of population marginal mid-CDF transforms given raw controls',
        'finite_scope': 'honest fixed nuisance means; iid rows within independent clusters',
        'hypothesis': 'iid conditional-null reference reuse adds bias, distinct references remove it; fixed-grid boundary need not be valid',
        'full_disclosure': 'all settings, paired draws, means, SDs, rates, intervals; no tuning or exclusions',
        'source_sha256': {SOURCE.name:sha(SOURCE),'peer_rank_products.py':sha(SOURCE.with_name('peer_rank_products.py'))},
        'software': {'numpy':np.__version__, 'pandas':pd.__version__},
    }
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    return protocol


def vectorized_scores(x,y,mx,my):
    """All ordered reference pairs, batched by cluster for the fixed study sizes."""
    if x.shape != y.shape or x.shape != mx.shape or x.shape != my.shape:
        raise ValueError('unaligned study arrays')
    m,n=x.shape;k=n-1
    naive=np.empty(m);corrected=np.empty(m)
    # Bound memory independently of the stress-study cluster count.
    for start in range(0,m,128):
        stop=min(start+128,m);xx=x[start:stop];yy=y[start:stop]
        a=(xx[:,:,None]>xx[:,None,:]).astype(np.int16)*2
        a+=(xx[:,:,None]==xx[:,None,:]).astype(np.int16)
        b=(yy[:,:,None]>yy[:,None,:]).astype(np.int16)*2
        b+=(yy[:,:,None]==yy[:,None,:]).astype(np.int16)
        indices=np.arange(n);a[:,indices,indices]=0;b[:,indices,indices]=0
        sx=a.sum(axis=2,dtype=np.int64);sy=b.sum(axis=2,dtype=np.int64)
        diagonal=(a*b).sum(axis=2,dtype=np.int64)
        ux=sx/(2.*k);uy=sy/(2.*k)
        q=(sx*sy-diagonal)/(4.*k*(k-1))
        raw=(ux-mx[start:stop])*(uy-my[start:stop])
        adjusted=raw+q-ux*uy
        naive[start:stop]=raw.mean(axis=1);corrected[start:stop]=adjusted.mean(axis=1)
    return naive,corrected


def make_condition(condition,b,ex,ey,sign):
    if condition in ['gaussian_null','gaussian_alternative']:
        x=b+ex;y=b+ey if condition=='gaussian_null' else b+.6*ex+.8*ey
        mx=ndtr(b/math.sqrt(3));my=mx.copy()
    elif condition=='linear_copy':
        x=b;y=b+ey;mx=ndtr(b);my=np.full_like(b,.5)
    elif condition=='nonmonotone_copy':
        x=b*b;y=b*b+ey;mx=2*ndtr(np.abs(b))-1;my=np.full_like(b,.5)
    elif condition=='interaction_copy':
        x=sign*b;y=x+ey;mx=ndtr(x);my=np.full_like(b,.5)
    else: raise ValueError(condition)
    return x,y,mx,my


def decisions(replication,condition,n,naive,corrected,ms,allowance):
    result=[]
    for m in ms:
        for method,scores in [('naive_normal',naive[:m]),('distinct_reference_normal',corrected[:m])]:
            mean=float(math.fsum(map(float,scores))/m)
            se=float(np.std(scores,ddof=1)/math.sqrt(m))
            statistic=mean/se if se else float('nan')
            p=float(norm.sf(statistic)) if se else 1.
            result.append({'replication':replication,'condition':condition,'entities':n,'clusters':m,
                           'method':method,'mean_score':mean,'standard_error':se,'statistic':statistic,
                           'p_value':p,'reject':p<.05,'effective_disjoint_triples':m*(n//3)})
        mean=float(math.fsum(map(float,corrected[:m]))/m);effective=m*(n//3)
        excess=max(0.,mean-allowance)
        logp=-.5*effective*excess*excess
        p=math.exp(logp) if logp>-745 else 0.
        lower=mean-allowance-math.sqrt(2*math.log(20)/effective)
        result.append({'replication':replication,'condition':condition,'entities':n,'clusters':m,
                       'method':'distinct_reference_finite','mean_score':mean,'standard_error':float('nan'),
                       'statistic':float('nan'),'p_value':p,'reject':p<.05,
                       'lower_bound':lower,'effective_disjoint_triples':effective})
    return result


def one_replication(arguments):
    replication,protocol=arguments
    rng=np.random.default_rng(protocol['seed_base']+replication)
    b,ex,ey=rng.normal(size=(3,400,128));sign=rng.choice([-1.,1.],size=(400,128))
    rows=[];checks=[]
    for n in protocol['entities']:
        for condition in CONDITIONS:
            arrays=make_condition(condition,b[:,:n],ex[:,:n],ey[:,:n],sign[:,:n])
            naive,corrected=vectorized_scores(*arrays)
            rows.extend(decisions(replication,condition,n,naive,corrected,protocol['clusters'],protocol['numeric_score_allowance']))
            if replication==0:
                for cluster in [0,399]:
                    exact=peer_rank_products(*(a[cluster] for a in arrays))
                    raw=float(exact['naive_residual_product'].mean());new=float(exact['corrected_residual_product'].mean())
                    error=max(abs(raw-naive[cluster]),abs(new-corrected[cluster]))
                    if error>5e-16:raise AssertionError('vectorized/fast exact-count disagreement')
                    checks.append({'condition':condition,'entities':n,'cluster':cluster,'max_difference':error})
    n=8;m=protocol['fixed_grid_boundary']['clusters'];grid=np.linspace(-1,1,n)
    # Every row's grid position is sampled without replacement within its cluster.
    permutations=np.argsort(rng.uniform(size=(m,n)),axis=1);baseline=grid[permutations]
    x=baseline+.1*rng.normal(size=(m,n));y=-baseline+.1*rng.normal(size=(m,n))
    comparisons=ndtr((grid[:,None]-grid[None,:])/(.1*math.sqrt(2)))
    oracle=(comparisons.sum(axis=1)-.5)/(n-1);mx=oracle[permutations];my=1-mx
    naive,corrected=vectorized_scores(x,y,mx,my)
    rows.extend(decisions(replication,'fixed_grid_opposite_boundary',n,naive,corrected,[m],protocol['numeric_score_allowance']))
    if replication==0:
        for cluster in [0,m-1]:
            exact=peer_rank_products(x[cluster],y[cluster],mx[cluster],my[cluster])
            error=max(abs(float(exact['naive_residual_product'].mean())-naive[cluster]),abs(float(exact['corrected_residual_product'].mean())-corrected[cluster]))
            if error>5e-16:raise AssertionError('boundary fast-count disagreement')
            checks.append({'condition':'fixed_grid_opposite_boundary','entities':n,'cluster':cluster,'max_difference':error})
    return rows,checks


def wilson(count,n):
    z=norm.ppf(.975);p=count/n;denom=1+z*z/n
    center=(p+z*z/(2*n))/denom;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return max(0.,center-half),min(1.,center+half)


def summarize(frame):
    rows=[]
    for key,part in frame.groupby(['condition','entities','clusters','method'],sort=True):
        condition,n,m,method=key;count=int(part.reject.sum());low,high=wilson(count,len(part))
        row={'condition':condition,'entities':int(n),'clusters':int(m),'method':method,
             'replications':len(part),'rejections':count,'rejection_rate':count/len(part),
             'wilson_low':low,'wilson_high':high,'mean_score':float(part.mean_score.mean()),
             'empirical_score_sd':float(part.mean_score.std(ddof=1)),
             'mean_reported_se':float(part.standard_error.mean()),
             'mean_statistic':float(part.statistic.mean()),'sd_statistic':float(part.statistic.std(ddof=1))}
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['freeze','benchmark','run','verify'])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();output=args.output
    if args.command=='freeze':
        freeze(output);print(json.dumps({'frozen':str(output),'protocol_sha256':sha(output/'protocol.json')}));return
    protocol=json.loads((output/'protocol.json').read_text())
    if protocol['source_sha256']!={SOURCE.name:sha(SOURCE),'peer_rank_products.py':sha(SOURCE.with_name('peer_rank_products.py'))}:
        raise RuntimeError('frozen sources changed')
    if args.command=='benchmark':
        start=time.perf_counter();rows,checks=one_replication((0,protocol))
        record={'seconds':time.perf_counter()-start,'rows':len(rows),'fast_count_checks':checks,
                'outcomes_not_reported_before_production':True}
        (output/'benchmark.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k!='fast_count_checks'}));return
    if args.command=='run':
        if (output/'draws.csv').exists():raise FileExistsError('recorded run already exists')
        (output/'run_start.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(output/'protocol.json')})+'\n')
        start=time.perf_counter();rows=[];checks=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result,check in pool.map(one_replication,[(r,protocol) for r in range(protocol['replications'])],chunksize=1):
                rows.extend(result);checks.extend(check)
        frame=pd.DataFrame(rows);frame.to_csv(output/'draws.csv',index=False)
        summary=summarize(frame);summary.to_csv(output/'summary.csv',index=False)
        (output/'fast_comparisons.json').write_text(json.dumps(checks,indent=2)+'\n')
        receipt={'status':'PASS','elapsed_seconds':time.perf_counter()-start,'protocol_sha256':sha(output/'protocol.json'),
                 'audit_rows':len(frame),'cells':len(summary),'independent_primitive_repetitions':protocol['replications'],
                 'output_sha256':{n:sha(output/n) for n in ['draws.csv','summary.csv','fast_comparisons.json']}}
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2));return
    frame=pd.read_csv(output/'draws.csv');summary=pd.read_csv(output/'summary.csv')
    pd.testing.assert_frame_equal(summary,summarize(frame),check_dtype=False,rtol=2e-12,atol=2e-14)
    assert len(frame)==41400 and len(summary)==138
    assert (summary.replications==300).all()
    assert frame.groupby(['condition','entities','clusters','method']).replication.nunique().eq(300).all()
    assert (frame.reject==(frame.p_value<.05)).all()
    print(json.dumps({'status':'PASS','rows':len(frame),'cells':len(summary)}))


if __name__=='__main__':main()
