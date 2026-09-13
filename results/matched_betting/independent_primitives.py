"""Independent audit: this file never imports the producer or its helpers.

Rebuilds all stored inference arithmetic and selected complete replications.
The rank statistics use numeric comparison matrices or unique-pair counts,
not the producer's Bernoulli closed form. Bias bounds use scipy linprog.
"""
from pathlib import Path
from itertools import permutations, product
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import beta, rankdata

HERE = Path(__file__).resolve().parent
KEYS = ['design','signal','fit','training_rows','validation_pairs','groups','peers','total_observations','method']
REPS = [0, 1, 17, 123, 317, 503, 777, 999]
DESIGNS = [('two_balanced',np.array([.5,.5]),np.array([.2,.8])),
           ('eight_rare',np.array([.005]+[.995/7]*7),np.linspace(.2,.8,8))]
SIGNALS = [-.125,0.,.125,.25,.5]
METHODS = ['absolute_range','signed_range','signed_variance','independent_triples','pooled_variance']


def compare(a,b):
    return np.greater(a,b).astype(float) + .5*np.equal(a,b)


def random_sample(seed,shape,prob,success,signal):
    rng=np.random.default_rng(np.random.SeedSequence(seed))
    z=rng.choice(len(prob),size=shape,p=prob)
    p=success[z]
    u=rng.random(shape)
    joint=p*p+signal*p*(1-p)
    off=p-joint
    code=(u>=joint).astype(int)+(u>=joint+off)+(u>=joint+2*off)
    v=np.isin(code,[0,1]).astype(float)
    w=np.isin(code,[0,2]).astype(float)
    return z,v,w


def fitted_means(z,v,categories):
    # Average ranks equal 1 + comparison sum over all other observations.
    peer_means=(rankdata(v,method='average')-1)/(len(v)-1)
    return np.array([peer_means[z==k].mean() if np.any(z==k) else .5 for k in range(categories)])


def rank_sums(v,w):
    """Generic all-reference comparison sums for equal-size groups."""
    a=compare(v[:,:,None],v[:,None,:])
    b=compare(w[:,:,None],w[:,None,:])
    return a.sum(2)-.5,b.sum(2)-.5,(a*b).sum(2)-.25


def pooled_rank_sums(v,w):
    """Generic numeric pair compression, no binary identities."""
    v,w=np.ravel(v),np.ravel(w)
    pairs,inv,counts=np.unique(np.column_stack([v,w]),axis=0,return_inverse=True,return_counts=True)
    a=compare(pairs[:,0,None],pairs[None,:,0])
    b=compare(pairs[:,1,None],pairs[None,:,1])
    return (a@counts)[inv]-.5,(b@counts)[inv]-.5,((a*b)@counts)[inv]-.25


def full_u(sums,f,g,n):
    a,b,ab=sums
    # All ordered j != k != i pairs: product of sums minus its diagonal.
    centered_diag=ab-g*a-f*b+(n-1)*f*g
    result=((a-(n-1)*f)*(b-(n-1)*g)-centered_diag)/((n-1)*(n-2))
    return result.mean(axis=-1)


def blocks(v,w,f,g):
    end=3*(v.shape[1]//3)
    v,w,f,g=[a[:,:end].reshape(-1,3) for a in (v,w,f,g)]
    terms=[]
    for i,j,k in permutations(range(3)):
        terms.append((compare(v[:,i],v[:,j])-f[:,i])*(compare(w[:,i],w[:,k])-g[:,i]))
    return np.mean(terms,axis=0)


def solve_lp(values,caps,maximize):
    scale=1e6
    ans=linprog(scale*(-values if maximize else values),A_eq=np.ones((1,len(values))),b_eq=[1.],
                bounds=list(zip(np.zeros(len(values)),caps)),method='highs',
                options={'dual_feasibility_tolerance':1e-10,'primal_feasibility_tolerance':1e-10})
    assert ans.success,ans.message
    return float((-1 if maximize else 1)*ans.fun/scale)


def budget(f,g,z,v,w,delta=1e-4):
    c=len(f);n=np.bincount(z[:,0],minlength=c);m=len(z)
    av=compare(v[:,0],v[:,1]);aw=compare(w[:,0],w[:,1])
    mv=np.array([av[z[:,0]==k].mean() if n[k] else 0. for k in range(c)])
    mw=np.array([aw[z[:,0]==k].mean() if n[k] else 0. for k in range(c)])
    rad=np.full(c,np.inf); rad[n>0]=np.sqrt(np.log(8*c/delta)/(2*n[n>0]))
    lv,uv=np.maximum(0,mv-rad),np.minimum(1,mv+rad)
    lw,uw=np.maximum(0,mw-rad),np.minimum(1,mw+rad)
    # Independent inverse-CDF orientation, equivalent to beta.isf.
    caps=np.array([beta.ppf(1-delta/(2*c),k+1,m-k) if k<m else 1. for k in n])
    if not m:caps[:]=1.
    corners=np.array([(a-f)*(b-g) for a,b in product([lv,uv],[lw,uw])])
    qp,qm=corners.max(0),corners.min(0)
    ea=np.maximum(abs(lv-f),abs(uv-f))*np.maximum(abs(lw-g),abs(uw-g))
    bp,bm,ba=solve_lp(qp,caps,True),solve_lp(qm,caps,False),solve_lp(ea,caps,True)
    return {'signed_allowance':bp,'absolute_allowance':ba,'bias_interval_width':bp-bm}


def inference(mean,s2,j,lo,hi,bias,kind,alpha=.05,delta=1e-4):
    mean,s2,j,lo,hi,bias=[np.asarray(a) for a in [mean,s2,j,lo,hi,bias]]
    r=hi-lo;d=np.maximum(mean-bias,0.)
    if kind=='range':
        radius=r*np.sqrt(-np.log(alpha-delta)/(2*j))
        pv=np.minimum(1,delta+np.exp(-2*j*(d/r)**2))
    else:
        x=np.log(2/(alpha-delta));a=np.sqrt(2*s2/j)
        c=(2*r/np.sqrt(j*(j-1))+r/(3*j)) if kind=='variance' else 7*r/(3*(j-1))
        radius=a*np.sqrt(x)+c*x
        exponent=((np.sqrt(a*a+4*c*d)-a)/(2*c))**2
        if kind=='variance':
            radius=np.minimum(radius,r*np.sqrt(x/(2*j)))
            exponent=np.maximum(exponent,2*j*(d/r)**2)
        pv=np.minimum(1,delta+2*np.exp(-exponent))
    lower=mean-bias-radius
    return {'radius':radius,'p':pv,'lower_bound':lower,'reject':pv<alpha}

