"""Finite population-midrank inference with a measured categorical bias budget.

The validation sample consists of independent, disjoint pairs of observations.
The first member supplies controls and focal values; the second supplies a
marginal rank reference. Training, validation and evaluation are independent.
The method uses no true conditional means, class probabilities or null law.
Its guarantee concerns iid resampled peers, not repeated fixed panel entities.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

SEED = 202609121140
DELTA = 0.0001


def _fits(f, g):
    f, g = np.asarray(f, dtype=float), np.asarray(g, dtype=float)
    if f.ndim != 1 or f.shape != g.shape or len(f) == 0:
        raise ValueError("Fits must be nonempty vectors for the same fixed categories")
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(g)):
        raise ValueError("Fits must be finite")
    if np.any((f < 0) | (f > 1) | (g < 0) | (g > 1)):
        raise ValueError("Fits must be clipped to [0, 1]")
    return f, g


def category_bias_budget(f, g, counts, mean_v, mean_w, delta=DELTA):
    """Uniform cell-mean confidence rectangles and a worst-cell bias bound.

    mean_v/mean_w average midrank comparisons from disjoint validation pairs.
    Empty categories receive [0,1]; they are never silently dropped.
    """
    f, g = _fits(f, g)
    n = np.asarray(counts)
    av, aw = np.asarray(mean_v, float), np.asarray(mean_w, float)
    if n.shape != f.shape or av.shape != f.shape or aw.shape != f.shape:
        raise ValueError("Counts and comparison means must match the fits")
    if not np.all(np.isfinite(n)) or np.any(n < 0) or np.any(n != np.floor(n)):
        raise ValueError("Counts must be nonnegative integers")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")
    occupied = n > 0
    if np.any(~np.isfinite(av[occupied])) or np.any(~np.isfinite(aw[occupied])):
        raise ValueError("Occupied comparison means must be finite")
    if np.any((av[occupied] < 0) | (av[occupied] > 1) |
              (aw[occupied] < 0) | (aw[occupied] > 1)):
        raise ValueError("Comparison means must lie in [0,1]")
    radius = np.full(f.shape, np.inf)
    radius[occupied] = np.sqrt(np.log(4*len(f)/delta)/(2*n[occupied]))
    lv, uv, lw, uw = [np.zeros_like(f), np.ones_like(f),
                       np.zeros_like(g), np.ones_like(g)]
    lv[occupied] = np.maximum(0, av[occupied]-radius[occupied])
    uv[occupied] = np.minimum(1, av[occupied]+radius[occupied])
    lw[occupied] = np.maximum(0, aw[occupied]-radius[occupied])
    uw[occupied] = np.minimum(1, aw[occupied]+radius[occupied])
    ev = np.maximum(np.abs(lv-f), np.abs(uv-f))
    ew = np.maximum(np.abs(lw-g), np.abs(uw-g))
    return dict(bias_upper=float(np.max(ev*ew)), error_v=ev, error_w=ew,
                lower_v=lv, upper_v=uv, lower_w=lw, upper_w=uw,
                counts=n.astype(int), delta=float(delta))


def kernel_range(f, g):
    """Exact rectangle bounds for (a-f_c)(b-g_c), a,b in [0,1]."""
    f, g = _fits(f, g)
    corners = np.stack([f*g, -f*(1-g), -(1-f)*g, (1-f)*(1-g)])
    return float(corners.min()), float(corners.max())


def finite_reference_decision(group_scores, peers, f, g, bias_upper,
                              delta=DELTA, alpha=.05):
    s = np.asarray(group_scores, float)
    if s.ndim != 1 or s.size == 0 or not np.all(np.isfinite(s)):
        raise ValueError("Group scores must be a nonempty finite vector")
    if not isinstance(peers, (int, np.integer)) or peers < 3:
        raise ValueError("Each iid peer group must contain at least three rows")
    if not 0 < delta < alpha < 1 or not np.isfinite(bias_upper) or bias_upper < 0:
        raise ValueError("Need 0 < delta < alpha < 1 and a nonnegative bias budget")
    lo, hi = kernel_range(f, g)
    if np.any(s < lo-1e-12) or np.any(s > hi+1e-12):
        raise ValueError("Group scores lie outside the declared kernel range")
    effective = len(s)*(peers//3)
    width = hi-lo
    margin = max(float(s.mean())-bias_upper, 0.)
    radius = width*np.sqrt(np.log(1/(alpha-delta))/(2*effective))
    p = min(1., delta+np.exp(-2*effective*(margin/width)**2))
    lower = float(s.mean())-bias_upper-radius
    se = (float(s.std(ddof=1)/np.sqrt(len(s))) if np.ptp(s)>0 else 0.) if len(s)>1 else float('nan')
    return dict(mean=float(s.mean()), bias_upper=float(bias_upper), radius=float(radius),
                lower_bound=float(lower), p=float(p), reject=bool(p < alpha),
                effective_triples=effective, kernel_lower=lo, kernel_upper=hi,
                group_se=se, bias_to_se=float(bias_upper/se) if se>0 else None)


def sample(rng, shape, probabilities, success, signal):
    z = rng.choice(len(probabilities), size=shape, p=probabilities)
    p = success[z]
    u = rng.random(shape)
    p11 = p*p+signal*p*(1-p)
    off = p-p11
    v = (u < p11+off).astype(float)
    w = ((u < p11) | ((u >= p11+off) & (u < p11+2*off))).astype(float)
    return z, v, w


def empirical_fit(z, v, categories):
    n = len(v)
    ranks = .5+.5*n/(n-1)*(v-v.mean())
    return np.array([ranks[z==c].mean() if np.any(z==c) else .5
                     for c in range(categories)])


def distinct_scores(v, w, f, g):
    n = v.shape[1]
    mv = (v.sum(1, keepdims=True)-v)/(n-1)
    mw = (w.sum(1, keepdims=True)-w)/(n-1)
    joint_mean = ((v*w).sum(1, keepdims=True)-v*w)/(n-1)
    uv, uw = .5+.5*(v-mv), .5+.5*(w-mw)
    ab = .25*((1+v)*(1+w)-(1+v)*mw-(1+w)*mv+joint_mean)
    return ((uv-f)*(uw-g)+(uv*uw-ab)/(n-2)).mean(1)


def one(rep):
    rows = []
    for design, probabilities, success in [
        ('two_balanced', np.array([.5,.5]), np.array([.2,.8])),
        ('eight_balanced', np.full(8,1/8), np.linspace(.2,.8,8)),
        ('eight_rare', np.array([.005]+[.995/7]*7), np.linspace(.2,.8,8))]:
        c = len(probabilities)
        true_means = .5-.5*np.dot(probabilities,success)+.5*success
        for gamma in (0., .25, .5):
            seed = [SEED, rep, c, int(design=='eight_rare'), int(100*gamma)]
            z,v,w = sample(np.random.default_rng(np.random.SeedSequence(seed+[0])),
                           (400,64), probabilities, success, gamma)
            tz,tv,tw = sample(np.random.default_rng(np.random.SeedSequence(seed+[1])),
                              (8192,), probabilities, success, gamma)
            vz,vv,vw = sample(np.random.default_rng(np.random.SeedSequence(seed+[2])),
                              (8192,2), probabilities, success, gamma)
            av = .5+.5*(vv[:,0]-vv[:,1])
            aw = .5+.5*(vw[:,0]-vw[:,1])
            target = .25*gamma*np.dot(probabilities,success*(1-success))
            for training in (32,512,8192):
                f,g = empirical_fit(tz[:training],tv[:training],c), empirical_fit(tz[:training],tw[:training],c)
                scores = distinct_scores(v,w,f[z],g[z])
                exact_bias = float(np.dot(probabilities,(true_means-f)*(true_means-g)))
                for validation in (512,8192):
                    counts = np.bincount(vz[:validation,0],minlength=c)
                    means_v = np.divide(np.bincount(vz[:validation,0],weights=av[:validation],minlength=c),
                                        counts,out=np.zeros(c),where=counts>0)
                    means_w = np.divide(np.bincount(vz[:validation,0],weights=aw[:validation],minlength=c),
                                        counts,out=np.zeros(c),where=counts>0)
                    budget = category_bias_budget(f,g,counts,means_v,means_w)
                    for groups in (100,400):
                        result = finite_reference_decision(scores[:groups],64,f,g,budget['bias_upper'])
                        se=result['group_se']
                        normal_p=float(norm.sf(result['mean']/se)) if se>0 else 1.
                        rows.append(dict(replication=rep,design=design,signal=gamma,
                            training_rows=training,validation_pairs=validation,validation_rows=2*validation,
                            groups=groups,peers=64,total_observations=training+2*validation+groups*64,
                            target=target,exact_bias=exact_bias,budget_covers=abs(exact_bias)<=budget['bias_upper'],
                            lower_covers=result['lower_bound']<=target,normal_p=normal_p,**result))
    return rows


def run(output, replications, workers):
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,
        replications=replications,delta=DELTA,alpha=.05,
        designs=['two_balanced','eight_balanced','eight_rare'],signals=[0,.25,.5],
        training_rows=[32,512,8192],validation_pairs=[512,8192],groups=[100,400],peers=64,
        validation='Disjoint iid focal/reference pairs, independent of training and evaluation; both members counted.',
        bound='Two-sided cell-mean Hoeffding rectangles, worst-cell product, bounded order-three U-statistic concentration.',
        target='Population marginal-midrank residual covariance; all means/variance truths used only as diagnostics.',
        scope='Finite categorical controls and iid resampled peers. No conditional-independence requirement, no variance floor, no estimated null law.',
        provenance='A new computable nuisance envelope and kernel-range specialization of the previously retained finite-reference bound; classical concentration is not claimed as novel.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records=[row for batch in pool.map(one,range(replications)) for row in batch]
    df=pd.DataFrame(records)
    df.to_csv(output/'replications.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    summary=[]
    keys=['design','signal','training_rows','validation_pairs','groups','peers']
    for key, cell in df.groupby(keys,sort=True):
        n=len(cell); rejected=int(cell.reject.sum()); rate=rejected/n
        zz=norm.ppf(.975); den=1+zz*zz/n
        mid=(rate+zz*zz/(2*n))/den
        half=zz*np.sqrt(rate*(1-rate)/n+zz*zz/(4*n*n))/den
        summary.append(dict(zip(keys,key),replications=n,rejections=rejected,rate=rate,
            wilson_low=mid-half,wilson_high=mid+half,normal_rejections=int((cell.normal_p<.05).sum()),
            budget_failures=int((~cell.budget_covers).sum()),coverage_failures=int((~cell.lower_covers).sum()),
            mean_bias_budget=cell.bias_upper.mean(),mean_sampling_radius=cell.radius.mean(),
            mean_lower_bound=cell.lower_bound.mean(),mean_p=cell.p.mean(),target=cell.target.iloc[0],
            total_observations=int(cell.total_observations.iloc[0])))
    pd.DataFrame(summary).to_csv(output/'summary.csv',index=False)
    print(json.dumps(dict(records=len(df),cells=len(summary))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replications',type=int,default=1000)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if args.replications<2 or args.workers<1:
        p.error('Need at least two replications and one worker')
    run(args.output,args.replications,args.workers)
