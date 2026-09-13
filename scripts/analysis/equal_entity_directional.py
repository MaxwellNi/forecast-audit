"""Equal-entity directional residual products on raw, unranked trajectories.

This benchmark isolates nuisance-training feedback. Its observations and
entity trajectories are synthetic. It does not implement the ranked-panel
audit, HAC inference, or historical forecasting evaluation.
"""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import math
import time

import numpy as np
from scipy.stats import norm


def make_folds(length, count=5):
    if length < count or count < 3:
        raise ValueError('At least three nonempty folds are required.')
    folds=np.empty(length,dtype=int)
    for label,part in enumerate(np.array_split(np.arange(length),count)):
        folds[part]=label
    return folds


def entity_scores(x,y,folds,eligible=None,gap=None):
    """Return two raw scores per entity on the same eligible middle rows.

    Eligibility and folds must be fixed independently of the simulated shocks.
    Both methods use equal entity weights. Within an entity, each evaluated
    row gets equal weight. Every included entity must have an eligible row
    with nonempty earlier and later training sets.
    """
    x,y=np.asarray(x,float),np.asarray(y,float)
    folds=np.asarray(folds)
    if x.shape!=y.shape or x.ndim!=2 or folds.shape!=(x.shape[1],):
        raise ValueError('x and y must have shape (entities, times), aligned with folds.')
    if not np.issubdtype(folds.dtype,np.number) or np.iscomplexobj(folds):
        raise ValueError('Fold codes must be finite, integer, and in temporal order.')
    fold_values=folds.astype(float)
    if (not np.isfinite(fold_values).all() or not np.equal(fold_values,np.floor(fold_values)).all()
            or np.any(np.diff(fold_values)<0)):
        raise ValueError('Fold codes must be finite, integer, and in temporal order.')
    labels=np.unique(fold_values)
    if len(labels)<3 or not np.array_equal(labels,np.arange(len(labels))):
        raise ValueError('Use at least three contiguous fold codes starting at zero.')
    if gap is not None and (not isinstance(gap,(int,np.integer)) or gap<0):
        raise ValueError('gap must be a nonnegative integer.')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Supply finite trajectories and explicit deterministic eligibility.')
    mask=np.ones(x.shape,dtype=bool) if eligible is None else np.broadcast_to(np.asarray(eligible,bool),x.shape)
    scores={name:np.zeros(x.shape[0]) for name in ('standard','directional')}
    if gap is not None:scores['gapped_complementary']=np.zeros(x.shape[0])
    used=np.zeros(x.shape,dtype=bool)
    for fold in np.unique(folds)[1:-1]:
        before=mask&(folds<fold)
        after=mask&(folds>fold)
        other=mask&(folds!=fold)
        n_before,n_after=before.sum(axis=1),after.sum(axis=1)
        take=mask&(folds==fold)&((n_before>0)&(n_after>0))[:,None]
        used|=take
        means={}
        for label,tr in [('standard',other),('earlier',before),('later',after)]:
            count=tr.sum(axis=1)
            means[label]=(np.divide((x*tr).sum(axis=1),count,out=np.zeros(len(x)),where=count>0),
                          np.divide((y*tr).sum(axis=1),count,out=np.zeros(len(y)),where=count>0))
        scores['standard']+=(((x-means['standard'][0][:,None])*(y-means['standard'][1][:,None]))*take).sum(axis=1)
        scores['directional']+=(((x-means['earlier'][0][:,None])*(y-means['later'][1][:,None]))*take).sum(axis=1)
        if gap is not None:
            block=np.flatnonzero(folds==fold)
            times=np.arange(x.shape[1])
            train=mask&((times<block[0]-gap)|(times>block[-1]+gap))
            n_train=train.sum(axis=1)
            if np.any((n_train==0)&take.any(axis=1)):
                raise ValueError('Gapped comparison has no training data on the common evaluation support.')
            mx=np.divide((x*train).sum(axis=1),n_train,out=np.zeros(len(x)),where=n_train>0)
            my=np.divide((y*train).sum(axis=1),n_train,out=np.zeros(len(y)),where=n_train>0)
            scores['gapped_complementary']+=(((x-mx[:,None])*(y-my[:,None]))*take).sum(axis=1)
    counts=used.sum(axis=1)
    if np.any(counts==0):
        raise ValueError('Every entity requires an eligible middle row and both training sides.')
    return {name:score/counts for name,score in scores.items()},used


def studentize(scores):
    scores=np.asarray(scores,float)
    if scores.ndim!=1 or len(scores)<2 or not np.isfinite(scores).all():
        raise ValueError('At least two finite entity scores are required.')
    mean=float(scores.mean())
    se=float(scores.std(ddof=1)/math.sqrt(len(scores)))
    return mean,se,(mean/se if se>0 else float('nan'))


def exact_mean(length,lookback,loading,method,signal=0.,middle_only=True):
    """Gaussian-design expectation from the full X-Y cross-covariance matrix.

    Unit-variance innovations and deterministic complete eligibility are used.
    X_t = loading * sum of previous lookback Y innovations + noise
          + signal * current Y innovation.
    Common entity intercepts disappear from both residuals.
    """
    folds=make_folds(length)
    lag=np.arange(length)[:,None]-np.arange(length)[None,:]
    cross=loading*((lag>0)&(lag<=lookback)).astype(float)+signal*np.eye(length)
    values=[]
    for t in range(length):
        f=folds[t]
        if middle_only and f in (folds.min(),folds.max()):continue
        if method=='standard':
            tx=ty=folds!=f
        elif method=='directional':
            tx,ty=folds<f,folds>f
        elif method=='gapped_complementary':
            block=np.flatnonzero(folds==f);times=np.arange(length)
            tx=ty=(times<block[0]-lookback)|(times>block[-1]+lookback)
        else:raise ValueError(method)
        if not tx.any() or not ty.any():continue
        wx=-tx.astype(float)/tx.sum();wy=-ty.astype(float)/ty.sum()
        wx[t]+=1;wy[t]+=1
        values.append(float(wx@cross@wy))
    if not values:raise ValueError('No eligible rows.')
    return float(np.mean(values))


def wilson(count,n):
    z=norm.ppf(.975);p=count/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-half),min(1.,center+half)


def run(output,repetitions=500,seed=914273):
    if repetitions<2:raise ValueError('At least two replications are required.')
    groups=(25,100,400);lengths=(20,40);lookbacks=(1,5);signals=(0.,.1,.3);loading=.2
    output=Path(output)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}. Choose a new directory to preserve recorded results.")
    output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(seed);rows=[];start=time.perf_counter()
    for replication in range(repetitions):
        innovation=rng.normal(size=(max(groups),max(lengths)+max(lookbacks)))
        noise=rng.normal(size=(max(groups),max(lengths)))
        entity_y=rng.normal(size=(max(groups),1))
        entity_x=rng.normal(size=(max(groups),1))
        for length in lengths:
            current=innovation[:,5:5+length]
            y=entity_y+current
            for lookback in lookbacks:
                past=sum(innovation[:,5-h:5+length-h] for h in range(1,lookback+1))
                base=entity_x+loading*(lookback*entity_y+past)+noise[:,:length]
                for signal in signals:
                    scores,used=entity_scores(base+signal*current,y,make_folds(length),gap=lookback if signal==0 else None)
                    for count in groups:
                        for method in scores:
                            mean,se,statistic=studentize(scores[method][:count])
                            rows.append({'replication':replication,'entities':count,'times':length,
                                         'lookback':lookback,'injected_association':signal,'method':method,
                                         'mean_entity_score':mean,'standard_error':se,'statistic':statistic,
                                         'evaluation_rows_per_entity':int(used[0].sum())})
    summary=[]
    for count in groups:
        for length in lengths:
            for lookback in lookbacks:
                for signal in signals:
                    for method in (('standard','directional','gapped_complementary') if signal==0 else ('standard','directional')):
                        cell=[r for r in rows if r['entities']==count and r['times']==length and r['lookback']==lookback and r['injected_association']==signal and r['method']==method]
                        t=np.array([r['statistic'] for r in cell]);mu=np.array([r['mean_entity_score'] for r in cell]);ses=np.array([r['standard_error'] for r in cell])
                        count_reject=int(np.sum(t>norm.ppf(.95)))
                        lo,hi=wilson(count_reject,len(cell));spread=float(mu.std(ddof=1))
                        exact=exact_mean(length,lookback,loading,method,signal)
                        summary.append({'entities':count,'times':length,'lookback':lookback,'injected_association':signal,
                                        'method':method,'replications':len(cell),'rejections':count_reject,
                                        'rejection_rate':count_reject/len(cell),'wilson_low':lo,'wilson_high':hi,
                                        'mean_statistic':float(t.mean()),'sd_statistic':float(t.std(ddof=1)),
                                        'mean_score':float(mu.mean()),'exact_mean_score':exact,
                                        'mean_score_mc_se':spread/math.sqrt(len(cell)),
                                        'mean_reported_se':float(ses.mean()),'empirical_mean_score_sd':spread,
                                        'reported_se_over_empirical_sd':float(ses.mean()/spread),
                                        'undefined_statistics':int(np.sum(~np.isfinite(t)))})
    with gzip.open(output/'replications.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    with (output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');writer.writeheader();writer.writerows(summary)
    protocol={'seed':seed,'repetitions':repetitions,'entities':groups,'times':lengths,'lookbacks':lookbacks,'injected_association':signals,
              'past_loading':loading,'folds':5,'evaluation':'Same three middle folds and equal entity weights for both methods.',
              'data':'Independent entity trajectories; independent standard normal outcome innovations, forecast noise, and two entity intercepts. Five presample innovations supply the lookbacks.',
              'null':'Injected association zero; forecast uses only past outcomes and independent forecast noise.',
              'positive_controls':'Current outcome innovation is added to the score. These are injected associations, not prospective forecasts.',
              'gapped_comparator':'Null cells only. Both nuisance means use complementary blocks after excluding lookback-many rows before and after the evaluation fold. This is a simple gapped mean comparator, not a complete implementation of a prior estimator.',
              'inference':'One-sided normal threshold 1.6448536269514722; pointwise 95% Wilson intervals. Classical independent-entity studentization, not a ranked-panel or HAC guarantee.',
              'coupling':'Common random numbers across entity counts, trajectory lengths, lookbacks, methods, and injected associations.',
              'elapsed_seconds':time.perf_counter()-start,'numpy_version':np.__version__,
              'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--repetitions',type=int,default=500)
    ap.add_argument('--seed',type=int,default=914273)
    args=ap.parse_args()
    values=run(args.output,args.repetitions,args.seed)
    print(json.dumps({'cells':len(values),'replications_per_cell':args.repetitions,'output':str(args.output)},indent=2))
