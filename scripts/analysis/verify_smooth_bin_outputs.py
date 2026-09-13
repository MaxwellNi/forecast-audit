"""Independent deterministic checks of the smooth-bin expansion and MC receipt."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
BASE=args.input
OUT=args.output
if OUT.exists(): raise FileExistsError(OUT)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integrate_bins(fun, q):
    node, weight = leggauss(64)
    u = (np.arange(q)[:, None] + (node[None, :] + 1)/2)/q
    means = fun(u) @ (weight/2)
    return means


def residual_values(fun, u, q):
    idx = np.minimum((u*q).astype(int), q-1)
    return fun(u) - integrate_bins(fun,q)[idx]


def one_case(fun, analytic, n):
    q = int(np.ceil(n**.2)); qs=q*np.array([1,2,4])
    w=np.linalg.lstsq(np.vstack([np.ones(3),qs.astype(float)**-2]),[1,0],rcond=None)[0]
    node, weights = leggauss(64)
    u=((np.arange(qs[-1])[:,None]+(node[None,:]+1)/2)/qs[-1]).ravel()
    weights=np.tile(weights/(2*qs[-1]),qs[-1])
    residuals=np.column_stack([residual_values(fun,u,int(b)) for b in qs])
    bias=(residuals**2).T@weights
    expected=np.array([analytic(int(b)) for b in qs])
    assert np.max(abs(bias-expected)) < 1e-12
    score_mean=float(bias@w)
    linear_term=residuals@w
    mean_term=residuals**2@w
    # Under independent N(0,1) errors: s=epsilon*eta+(epsilon+eta)A(U)+B(U).
    score_variance=float(1+2*(linear_term**2@weights)+(mean_term**2@weights)-score_mean**2)
    shifted_normal_rate=float(norm.sf(norm.ppf(.95)-np.sqrt(n)*score_mean/np.sqrt(score_variance)))
    return {'n':n,'q':qs.tolist(),'weights':w.tolist(),'exact_oracle_mean':score_mean,
            'oracle_variance_quadrature':score_variance,'sqrt_n_mean':np.sqrt(n)*score_mean,
            'oracle_normal_approximation_only':shifted_normal_rate,
            'max_bias_identity_error':float(np.max(abs(bias-expected)))}


cases=[]
for name,fun,analytic in [
 ('linear',lambda u:12*u,lambda q:144/(12*q*q)),
 ('quadratic',lambda u:12*u*u,lambda q:144/(9*q*q)-144/(45*q**4)),
 ('sine',lambda u:6*np.sin(2*np.pi*u),lambda q:18*(1-np.sinc(1/q)**2))]:
 for n in [2000,8000,32000]:
    cases.append({'design':name,**one_case(fun,analytic,n)})

# A nonspecial smooth pair independently checks the O(q^-4) remainder order.
f=lambda u:np.exp(u)
g=lambda u:np.cos(u)+u**3
nodes,weights=leggauss(96)
u=(nodes+1)/2
coefficient=float(np.exp(u)*(-np.sin(u)+3*u*u)@(weights/2)/12)
orders=[]
for q in [4,8,16,32,64]:
 nodes,weights=leggauss(64)
 u=((np.arange(q)[:,None]+(nodes[None,:]+1)/2)/q).ravel()
 ww=np.tile(weights/(2*q),q)
 b=float((residual_values(f,u,q)*residual_values(g,u,q))@ww)
 orders.append({'q':q,'bias':b,'q4_remainder':(b-coefficient/q**2)*q**4})
assert max(abs(o['q4_remainder']) for o in orders)<1

raw=pd.read_csv(BASE/'draws.csv')
summary=pd.read_csv(BASE/'summary.csv')
assert len(raw)==21600
assert len(raw.drop_duplicates(['n','design','seed','noise_correlation']))==5400
assert len(raw.drop_duplicates(['n','design','seed']))==2700
assert not raw.duplicated(['n','design','seed','noise_correlation','method']).any()
assert np.equal(raw.reject,raw.statistic>norm.ppf(.95)).all()
for _,row in summary.iterrows():
 data=raw[(raw.n==row.n)&(raw.design==row.design)&(raw.noise_correlation==row.noise_correlation)&(raw.method==row.method)]
 assert len(data)==300
 assert int(data.reject.sum())==int(row.hits)
 np.testing.assert_allclose([row.ci_lower,row.ci_upper],proportion_confint(int(row.hits),300,method='wilson'),atol=1e-14)
result={'status':'PASS','draws_csv_sha256':sha(BASE/'draws.csv'),'summary_csv_sha256':sha(BASE/'summary.csv'),
        'method_evaluations':21600,'generated_datasets':5400,'independent_noise_draws':2700,
        'analytic_cases':cases,'generic_smooth_pair':orders,
        'limits':'Deterministic moment checks and independent result recount; shifted normal probabilities are approximations, not finite-sample type-I guarantees.'}
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
