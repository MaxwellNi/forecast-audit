"""Correctness and limitation checks; no archived outcome data are read.

The independent formulation uses four corner masses per category, rather
than the producer's perspective inequalities. True finite-frame lifts and
dual certificates are also checked in exact rational arithmetic.
"""
import argparse
import copy
from fractions import Fraction as F
from itertools import combinations, product
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog

import joint_bias as jb


def corner_lp(data, means=True):
    """Independent extended formulation: four nonnegative corner masses."""
    cells=len(data['f']);n=4*cells
    objective=np.zeros(n);a=[];b=[];e=[np.ones(n)];d=[1.]
    meanx=np.zeros(n);meany=np.zeros(n)
    for c in range(cells):
        row=np.zeros(n);row[4*c:4*c+4]=1.
        a.extend([row,-row]);b.extend([data['mass_upper'][c],-data['mass_lower'][c]])
        for j,(x,y) in enumerate(product([data['lower_x'][c],data['upper_x'][c]],
                                          [data['lower_y'][c],data['upper_y'][c]])):
            k=4*c+j;objective[k]=(x-data['f'][c])*(y-data['g'][c])
            meanx[k]=x;meany[k]=y
    if means:e.extend([meanx,meany]);d.extend([.5,.5])
    fit=linprog(-objective,A_ub=a,b_ub=b,A_eq=e,b_eq=d,bounds=(0,None),method='highs')
    assert fit.success,fit.message
    return float(-fit.fun)


def check_lift(data,p,mx,my):
    x=[]
    for pc,xc,yc in zip(p,mx,my):x.extend([pc,pc*xc,pc*yc,pc*xc*yc])
    obj,a,b,e,d=jb.model(data)
    dot=lambda row:sum(v*x[k] for k,v in row.items())
    assert all(dot(row)<=rhs for row,rhs in zip(a,b))
    assert all(dot(row)==rhs for row,rhs in zip(e,d))
    return sum(q*v for q,v in zip(obj,x)),x


def finite_frame(rng,cells):
    """Tied integer values; exact uniform-frame ranks and category means."""
    n=64;cat=np.arange(n)%cells;rng.shuffle(cat)
    x,y=rng.integers(0,9,(2,n))
    ranks=[]
    for values in [x,y]:
        ranks.append([F(2*int(np.sum(values<v))+int(np.sum(values==v)),2*n) for v in values])
    p=[F(int(np.sum(cat==c)),n) for c in range(cells)]
    mus=[[sum(rr[i] for i in range(n) if cat[i]==c)/int(np.sum(cat==c))
          for c in range(cells)] for rr in ranks]
    assert sum(p[c]*mus[0][c] for c in range(cells))==F(1,2)
    assert sum(p[c]*mus[1][c] for c in range(cells))==F(1,2)
    # Dyadic grid enclosures ensure exact containment of possibly non-dyadic means.
    bounds=[]
    for mu in mus:
        lower=[];upper=[]
        for v in mu:
            floor=(v*32).numerator//(v*32).denominator
            lower.append(max(0,(floor-int(rng.integers(1,9)))/32))
            upper.append(min(1,(floor+1+int(rng.integers(1,9)))/32))
        bounds.extend([lower,upper])
    f,g=rng.integers(0,33,(2,cells))/32
    lo=[max(0,float(pc)-.125) for pc in p];hi=[min(1,float(pc)+.125) for pc in p]
    return jb.inputs(f,g,*bounds,hi,lo),p,*mus


def run():
    cases=[]
    def known(name,args,expected,sep):
        result=jb.joint_bias_upper(**args)
        assert result['status']=='certified',result
        assert jb.verify_certificate(result)
        assert result['bias_upper']==float(expected),(name,result)
        assert result['separable_upper']==float(sep)
        assert abs(corner_lp(result['inputs'])-float(expected))<1e-12
        cases.append(dict(name=name,bias_upper=result['bias_upper'],separable=result['separable_upper']))
        return result
    known('attainable_strict_improvement',dict(f=[0,0],g=[0,0],lower_x=[.25,.625],
          upper_x=[.375,.75],lower_y=[.25,.625],upper_y=[.375,.75],
          mass_upper=[.5,.5],mass_lower=[.5,.5]),F(5,16),F(45,128))
    single=known('single_cell_strict_relaxation',dict(f=[.5],g=[.5],lower_x=[0],upper_x=[1],
           lower_y=[0],upper_y=[1],mass_upper=[1]),F(1,4),F(1,4))
    three=known('three_cell_strict_relaxation',dict(f=[.5]*3,g=[.5]*3,lower_x=[.375]*3,
           upper_x=[.625]*3,lower_y=[.375]*3,upper_y=[.625]*3,
           mass_upper=[.25,.375,.375],mass_lower=[.25,.375,.375]),F(1,64),F(1,64))
    weights=[F(1,4),F(3,8),F(3,8)];vertices=[]
    for fixed in combinations(range(3),2):
        free=next(c for c in range(3) if c not in fixed)
        for endpoints in product([-F(1),F(1)],repeat=2):
            t=[F(0)]*3
            for c,v in zip(fixed,endpoints):t[c]=v
            t[free]=-sum(weights[c]*t[c] for c in fixed)/weights[free]
            if abs(t[free])<=1:vertices.append(sum(weights[c]*t[c]**2 for c in range(3)))
    assert max(vertices)==F(3,4)
    assert F(1,64)*max(vertices)==F(3,256)<F(1,64)
    # Integrals under the explicit real-rank law in DERIVATION: E Z=1/2,
    # E[Z(Z-1/2)]=1/12; resulting means .5,.625,.375.
    density_intercepts=[F(1,4),F(3,8),F(3,8)]
    slopes=[F(0),F(9,16),-F(9,16)]
    means=[(p*F(1,2)+s*F(1,12))/p for p,s in zip(density_intercepts,slopes)]
    assert means==[F(1,2),F(5,8),F(3,8)]
    assert all(0<=p+s*t<=1 for p,s in zip(density_intercepts,slopes) for t in [-F(1,2),F(1,2)])
    true,_=check_lift(three['inputs'],weights,means,means)
    assert true==F(3,256)
    # Empty and numerical-failure behavior are explicit and conservative.
    empty=jb.joint_bias_upper([.5],[.5],[0],[.4],[0],[1],[1])
    assert empty['status']=='fallback_joint_infeasible' and jb.verify_certificate(empty)
    probempty=jb.joint_bias_upper([.5],[.5],[0],[1],[0],[1],[.9])
    assert probempty['status']=='fallback_empty_probability_box' and probempty['bias_upper']==1.
    with patch.object(jb,'linprog',side_effect=RuntimeError('injected solver failure')):
        failed=jb.joint_bias_upper(**{k:single['inputs'][k] for k in jb.NAMES})
    assert failed['status']=='fallback_solver_failure' and jb.verify_certificate(failed)
    for changed in ['allowance','multiplier','fraction']:
        bad=copy.deepcopy(single)
        if changed=='allowance':bad['bias_upper']-=.01
        elif changed=='multiplier':bad['audit']['equality_multipliers'][0]+=.01
        else:bad['audit']['chosen_upper_fraction']='0'
        assert not jb.verify_certificate(bad),changed
    # A sub-ulp objective discrepancy must be rounded toward the safe direction.
    almost=F.from_float(.1)+F(1,2**200)
    assert F.from_float(jb.upward(almost))>=almost and jb.upward(almost)>.1
    badargs=dict(f=[.5],g=[.5],lower_x=[0],upper_x=[1],lower_y=[0],upper_y=[1],mass_upper=[1])
    for key,value in [('f',[np.nan]),('upper_x',[-.1]),('lower_x',[1.1]),('mass_upper',[2.])]:
        try:jb.joint_bias_upper(**dict(badargs,**{key:value}))
        except ValueError:pass
        else:raise AssertionError('Malformed input accepted')
    rng=np.random.default_rng(202609122111)
    max_gap=0.;strict=0
    for rep in range(120):
        data,p,mx,my=finite_frame(rng,1+rep%8)
        truth,lift=check_lift(data,p,mx,my)
        answer=jb.joint_bias_upper(**data);plain=jb.joint_bias_upper(**data,mean_constraints=False)
        assert jb.verify_certificate(answer) and jb.verify_certificate(plain)
        assert answer['status']=='certified'
        assert truth<=F.from_float(answer['bias_upper'])
        assert answer['bias_upper']<=answer['separable_upper']
        assert abs(plain['bias_upper']-answer['separable_upper'])<2e-12
        independent=corner_lp(data)
        assert independent-2e-12<=answer['bias_upper']<=independent+2e-12
        max_gap=max(max_gap,abs(answer['bias_upper']-independent))
        strict+=answer['separable_upper']-answer['bias_upper']>1e-10
        # Arbitrary nonnegative multiplier proposal, not an approximately
        # feasible dual solution: the exact residual correction still works.
        _,a,_,e,_=jb.model(data)
        yy=rng.random(len(a));zz=rng.normal(size=len(e))
        upper,_,_=jb.certified_dual_fraction(data,yy,zz)
        assert truth<=upper
    return dict(status='PASS',seed=202609122111,known_cases=cases,
                three_cell_coherent_maximum=str(F(3,256)),three_cell_relaxation_gap=str(F(1,256)),
                finite_frame_cases=120,strict_improvement_cases=strict,
                max_independent_corner_formulation_gap=max_gap,
                fallback_checks=3,tamper_rejections=3,invalid_input_rejections=4,
                exact_lift_and_arbitrary_dual_checks=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=run()
    if args.output:
        with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
