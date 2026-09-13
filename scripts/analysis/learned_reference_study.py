"""Paired learned-mean rank-reference experiment with an exactly known null.

All data are synthetic. Z is Bernoulli(1/2); conditional on Z, V and W
are independent Bernoulli(.2 + .6 Z). Thus population midranks are .25/.75,
their conditional means are .35/.65, the residual target is zero,
Gamma=.0225, and the first-projection variance is .0016.
Training computes empirical leave-one-out midranks and their group means;
it does not receive population ranks or true conditional means.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

SEED = 202609120730


def sample(rng, shape):
    z = rng.integers(0, 2, size=shape)
    p = .2 + .6 * z
    return z, (rng.random(shape) < p).astype(float), (rng.random(shape) < p).astype(float)


def fit_means(z, x):
    n = len(z)
    ranks = .5 + .5 * n / (n - 1) * (x - x.mean())
    return np.array([ranks[z == c].mean() if np.any(z == c) else .5 for c in (0, 1)])


def binary_scores(v, w, f, g):
    """Exact O(MN) specialization of pairwise midrank comparisons."""
    n = v.shape[1]
    mv = (v.sum(axis=1, keepdims=True) - v) / (n - 1)
    mw = (w.sum(axis=1, keepdims=True) - w) / (n - 1)
    mvw = ((v*w).sum(axis=1, keepdims=True) - v*w) / (n - 1)
    u = .5 + .5*(v-mv)
    r = .5 + .5*(w-mw)
    joint = .25*((1+v)*(1+w) - (1+v)*mw - (1+w)*mv + mvw)
    shared = (u-f)*(r-g)
    distinct = shared + (u*r-joint)/(n-2)
    return shared.mean(axis=1), distinct.mean(axis=1)


def one(rep):
    z, v, w = sample(np.random.default_rng(np.random.SeedSequence([SEED, rep, 0])), (400, 64))
    true = np.array([.35, .65])
    fits = [(0, true, true)]
    for size in (32, 512, 8192):
        tz, tv, tw = sample(np.random.default_rng(np.random.SeedSequence([SEED, rep, size])), (size,))
        fits.append((size, fit_means(tz, tv), fit_means(tz, tw)))
    records = []
    for n in (16, 64):
        zz, vv, ww = z[:, :n], v[:, :n], w[:, :n]
        for size, f, g in fits:
            b = float(np.mean((true-f)*(true-g)))
            ef, eg = float(np.sqrt(np.mean((true-f)**2))), float(np.sqrt(np.mean((true-g)**2)))
            shared, distinct = binary_scores(vv, ww, f[zz], g[zz])
            for m in (25, 100, 400):
                for method, scores, expected in [('shared', shared[:m], b+.0225/(n-1)), ('distinct', distinct[:m], b)]:
                    mean = float(scores.mean())
                    se = float(scores.std(ddof=1)/np.sqrt(m))
                    t = mean/se if se > 0 else 0.
                    records.append(dict(replication=rep, entities=n, groups=m, training_rows=size,
                        method=method, mean=mean, se=se, statistic=t, p=float(norm.sf(t)),
                        expected_mean=expected, nuisance_bias=b, epsilon_f=ef, epsilon_g=eg,
                        bias_bound=ef*eg, scaled_bias_bound=np.sqrt(m*n)*ef*eg,
                        centered_statistic=(mean-expected)/se if se > 0 else 0.))
    return records


def run(output, reps, workers):
    output=Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows=[r for batch in pool.map(one, range(reps)) for r in batch]
    draws=pd.DataFrame(rows)
    summaries=[]
    for key, cell in draws.groupby(['entities','groups','training_rows','method'], sort=True):
        rejects=int((cell.p < .05).sum()); rate=rejects/reps
        z=norm.ppf(.975); den=1+z*z/reps
        mid=(rate+z*z/(2*reps))/den
        half=z*np.sqrt(rate*(1-rate)/reps+z*z/(4*reps*reps))/den
        summaries.append(dict(zip(['entities','groups','training_rows','method'],key),
            replications=reps,rejections=rejects,rate=rate,wilson_low=mid-half,wilson_high=mid+half,
            mean_statistic=cell.statistic.mean(),sd_statistic=cell.statistic.std(ddof=1),
            mean_score=cell['mean'].mean(),mean_exact_expectation=cell.expected_mean.mean(),
            mean_nuisance_bias=cell.nuisance_bias.mean(),mean_error_bound=cell.bias_bound.mean(),
            mean_scaled_error_bound=cell.scaled_bias_bound.mean(),
            mean_centered_statistic=cell.centered_statistic.mean(),
            centered_rejections=int((cell.centered_statistic > norm.isf(.05)).sum())))
    draws.to_csv(output/'replications.csv',index=False)
    pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False)
    protocol=dict(seed=SEED,replications=reps,training_rows=[32,512,8192],groups=[25,100,400],
        entities=[16,64],target='Population-midrank residual covariance, exactly zero',
        sampling='Independent resampled peer groups; independent training shared across evaluation groups',
        truth=dict(conditional_midrank_means=[.35,.65],Gamma=.0225,sigma_squared=.0016),
        training='Empirical leave-one-out midranks, then two category means; no population CDF supplied',
        inference='Normal reference across independent evaluation group scores; alpha .05, no fitted threshold',
        error_bounds='Exact population L2 errors are diagnostic in this known synthetic law, not supplied to the test',
        pairing='All methods and training sizes share each evaluation panel; group/entity counts nested',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numpy=np.__version__)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(records=len(draws),cells=len(summaries))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replications',type=int,default=1000)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if args.replications < 2 or args.workers < 1: p.error('Need >=2 replications and >=1 worker')
    run(args.output,args.replications,args.workers)
