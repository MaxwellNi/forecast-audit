"""Certified LP upper envelope for categorical rank-mean learning bias.

All numerical LP coefficients are reconstructed from exact binary-rational
input floats. A floating solver proposes dual multipliers; exact arithmetic
and a box-residual correction certify an upper bound. The primal objective
is diagnostic only. Confidence-set coverage is an upstream responsibility.

McCormick relaxation, LP duality and the residual correction are classical.
The extra statistical information is E(population midrank)=1/2, including
ties, applied jointly to the two category-mean vectors.
"""
from fractions import Fraction
import math

import numpy as np
from scipy.optimize import linprog

NAMES = ['f','g','lower_x','upper_x','lower_y','upper_y','mass_lower','mass_upper']
ZERO, ONE, HALF = Fraction(0), Fraction(1), Fraction(1,2)


def rational(value):
    return Fraction.from_float(float(value))


def upward(value):
    """Least adjacent binary64 rounding known to lie above exact value."""
    result = float(value)
    if not math.isfinite(result):
        raise ArithmeticError('Exact upper bound cannot be represented finitely')
    if rational(result) < value:
        result = float(np.nextafter(result,np.inf))
    assert rational(result) >= value
    return result


def inputs(f,g,lower_x,upper_x,lower_y,upper_y,mass_upper,mass_lower=None):
    values = dict(f=f,g=g,lower_x=lower_x,upper_x=upper_x,lower_y=lower_y,
                  upper_y=upper_y,mass_upper=mass_upper)
    values['mass_lower'] = np.zeros(len(f)) if mass_lower is None else mass_lower
    arrays={name:np.asarray(values[name],dtype=float) for name in NAMES}
    shape=arrays['f'].shape
    if len(shape)!=1 or shape[0]==0 or any(a.shape!=shape for a in arrays.values()):
        raise ValueError('Matching nonempty one-dimensional arrays required')
    if any(not np.isfinite(a).all() or np.any((a<0)|(a>1)) for a in arrays.values()):
        raise ValueError('All fits, intervals and masses must be finite in [0,1]')
    if any(np.any(arrays[lo]>arrays[hi]) for lo,hi in
           [('lower_x','upper_x'),('lower_y','upper_y'),('mass_lower','mass_upper')]):
        raise ValueError('Lower endpoints must not exceed upper endpoints')
    return {name:arrays[name].tolist() for name in NAMES}


def separable_fraction(data):
    """Exact capped-simplex optimum with the same input probability box."""
    v={name:[rational(x) for x in data[name]] for name in NAMES}
    lower,upper=v['mass_lower'],v['mass_upper']
    if sum(lower)>ONE or sum(upper)<ONE:
        return None
    q=[max((x-v['f'][c])*(y-v['g'][c])
           for x in [v['lower_x'][c],v['upper_x'][c]]
           for y in [v['lower_y'][c],v['upper_y'][c]]) for c in range(len(lower))]
    result=sum(p*z for p,z in zip(lower,q));left=ONE-sum(lower)
    for c in sorted(range(len(q)),key=lambda k:q[k],reverse=True):
        amount=min(left,upper[c]-lower[c]);result+=amount*q[c];left-=amount
    assert left==0
    return result


def model(data,mean_constraints=True):
    """Sparse exact Ax<=b, Ex=d, 0<=x<=1; variables [p,u,v,w] per cell."""
    z={name:[rational(x) for x in data[name]] for name in NAMES}
    cells=len(z['f']);objective=[ZERO]*(4*cells);a=[];b=[]
    def add(row,rhs=ZERO):
        a.append({k:value for k,value in row.items() if value});b.append(rhs)
    for c in range(cells):
        p,u,v,w=range(4*c,4*c+4)
        lx,hx,ly,hy=[z[name][c] for name in ['lower_x','upper_x','lower_y','upper_y']]
        f,g=z['f'][c],z['g'][c]
        objective[p]=f*g;objective[u]=-g;objective[v]=-f;objective[w]=ONE
        add({p:ONE},z['mass_upper'][c]);add({p:-ONE},-z['mass_lower'][c])
        add({u:ONE,p:-hx});add({u:-ONE,p:lx})
        add({v:ONE,p:-hy});add({v:-ONE,p:ly})
        add({w:-ONE,u:ly,v:lx,p:-lx*ly})
        add({w:-ONE,u:hy,v:hx,p:-hx*hy})
        add({w:ONE,u:-ly,v:-hx,p:hx*ly})
        add({w:ONE,u:-hy,v:-lx,p:lx*hy})
    e=[{4*c:ONE for c in range(cells)}];d=[ONE]
    if mean_constraints:
        e.extend([{4*c+1:ONE for c in range(cells)},{4*c+2:ONE for c in range(cells)}])
        d.extend([HALF,HALF])
    return objective,a,b,e,d


def certified_dual_fraction(data,inequality_multipliers,equality_multipliers,mean_constraints=True):
    """For ANY y>=0,z, upper=c·x <= b·y+d·z+sum positive residuals."""
    c,a,b,e,d=model(data,mean_constraints)
    if len(inequality_multipliers)!=len(a) or len(equality_multipliers)!=len(e):
        raise ValueError('Multiplier dimensions do not match exact model')
    values=list(inequality_multipliers)+list(equality_multipliers)
    if not np.isfinite(values).all() or any(y<0 for y in inequality_multipliers):
        raise ValueError('Finite multipliers and nonnegative inequality multipliers required')
    y=[rational(x) for x in inequality_multipliers];z=[rational(x) for x in equality_multipliers]
    residual=list(c)
    for rows,multipliers in [(a,y),(e,z)]:
        for row,mult in zip(rows,multipliers):
            for k,value in row.items():residual[k]-=mult*value
    correction=sum(max(ZERO,x) for x in residual)
    value=sum(rhs*mult for rhs,mult in zip(b,y))+sum(rhs*mult for rhs,mult in zip(d,z))+correction
    return value,correction,residual


def joint_bias_upper(f,g,lower_x,upper_x,lower_y,upper_y,mass_upper,mass_lower=None,
                     mean_constraints=True):
    """Return a certified upper allowance, never worse than same-input separable.

    An infeasible probability box returns the universal bound 1. Solver
    failure or reported infeasible joint constraints return the exact old
    separable allowance, with an explicit fallback reason. No infeasibility
    result is interpreted as statistical evidence.
    """
    data=inputs(f,g,lower_x,upper_x,lower_y,upper_y,mass_upper,mass_lower)
    sep=separable_fraction(data)
    if sep is None:
        return dict(bias_upper=1.,separable_upper=1.,status='fallback_empty_probability_box',
                    inputs=data,mean_constraints=bool(mean_constraints),audit=None,
                    solver_objective=None,certified_dual_upper=None)
    result=dict(bias_upper=upward(sep),separable_upper=upward(sep),
                status='fallback_solver_failure',inputs=data,mean_constraints=bool(mean_constraints),
                audit=None,solver_objective=None,certified_dual_upper=None,
                separable_fraction=str(sep))
    c,a,b,e,d=model(data,mean_constraints)
    def dense(rows):
        out=np.zeros((len(rows),len(c)))
        for i,row in enumerate(rows):
            for k,value in row.items():out[i,k]=float(value)
        return out
    try:
        fit=linprog(-np.array([float(x) for x in c]),A_ub=dense(a),b_ub=[float(x) for x in b],
                    A_eq=dense(e),b_eq=[float(x) for x in d],bounds=[(0.,1.)]*len(c),method='highs',
                    options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        result['solver_status']=int(fit.status);result['solver_message']=str(fit.message)
        if not fit.success:
            result['status']='fallback_joint_infeasible' if fit.status==2 else 'fallback_solver_failure'
            return result
        y=np.maximum(0.,-np.asarray(fit.ineqlin.marginals,float)).tolist()
        z=(-np.asarray(fit.eqlin.marginals,float)).tolist()
        upper,correction,residual=certified_dual_fraction(data,y,z,mean_constraints)
        if upper < -ONE:
            result['status']='fallback_suspicious_empty_relaxation';return result
        chosen=min(sep,upper)
        result.update(bias_upper=upward(chosen),certified_dual_upper=upward(upper),
                      solver_objective=float(-fit.fun),status='certified',
                      audit=dict(inequality_multipliers=y,equality_multipliers=z,
                                 certified_upper_fraction=str(upper),
                                 positive_residual_fraction=str(correction),
                                 max_absolute_residual=float(max(abs(x) for x in residual)),
                                 chosen_upper_fraction=str(chosen),
                                 primal_candidate=fit.x.tolist()))
        assert result['bias_upper']<=result['separable_upper']
        return result
    except (ArithmeticError,ValueError,RuntimeError) as error:
        result['solver_message']=str(error);return result


def verify_certificate(result):
    """Recompute exact dual/residual certificate without invoking a solver."""
    data=result['inputs'];sep=separable_fraction(data)
    if sep is None:
        return result['bias_upper']==result['separable_upper']==1.
    if result['separable_upper']!=upward(sep):return False
    if result['status']!='certified':return result['bias_upper']==upward(sep)
    audit=result['audit']
    value,correction,_=certified_dual_fraction(data,audit['inequality_multipliers'],
                                              audit['equality_multipliers'],result['mean_constraints'])
    chosen=min(value,sep)
    return (str(value)==audit['certified_upper_fraction'] and
            str(correction)==audit['positive_residual_fraction'] and
            str(chosen)==audit['chosen_upper_fraction'] and
            result['certified_dual_upper']==upward(value) and
            result['bias_upper']==upward(chosen) and result['bias_upper']<=result['separable_upper'])
