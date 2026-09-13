"""One-sided learning allowances and variance bounds for reference U statistics.

The validation intervals and the probability confidence set are shared across
signed and absolute comparisons. The full-U variance bound uses an independent
triple partition to bound kernel variance; it never plugs variance into an
unproved Hoeffding expression. Training, validation and evaluation must follow
the independent common-law design in the paper.
"""
import numpy as np
from category_reference_certificate import weighted_category_bias_budget, kernel_range


def capped_simplex(values, upper, maximize=True):
    values,upper=np.asarray(values,float),np.asarray(upper,float)
    if values.ndim!=1 or values.shape!=upper.shape or not len(values):
        raise ValueError('Need matching nonempty vectors')
    if not np.isfinite(values).all() or not np.isfinite(upper).all() or np.any(upper<0) or np.any(upper>1):
        raise ValueError('Invalid objective or probability capacities')
    left,answer=1.,0.
    for c in np.argsort(-values if maximize else values,kind='stable'):
        mass=min(max(0.,left),float(upper[c]));answer+=mass*values[c];left-=mass
    if left>1e-10:raise ValueError('Infeasible probability confidence set')
    return float(answer)


def signed_category_budget(f,g,counts,mean_v,mean_w,delta=.0001):
    out=weighted_category_bias_budget(f,g,counts,mean_v,mean_w,delta)
    f,g=np.asarray(f,float),np.asarray(g,float)
    corners=np.stack([(out[x]-f)*(out[y]-g)
                      for x in ('lower_v','upper_v') for y in ('lower_w','upper_w')])
    upper=corners.max(0);lower=corners.min(0);caps=out['category_mass_upper']
    out.update(absolute_bias_upper=out['bias_upper'],
               bias_upper=capped_simplex(upper,caps),
               bias_lower=capped_simplex(lower,caps,maximize=False),
               cell_signed_upper=upper,cell_signed_lower=lower)
    if out['bias_upper']>out['absolute_bias_upper']+1e-12:
        raise ArithmeticError('Signed allowance exceeds its absolute counterpart')
    return out


def triple_scores(v,w,f,g):
    """Fixed consecutive disjoint triples, symmetrized over all six roles.

    Arrays have shape (groups, peers); f,g contain the row-specific fits.
    The last peers%3 rows per group are unused only in this variance estimate.
    They remain in the full U statistic.
    """
    v,w,f,g=[np.asarray(x,float) for x in (v,w,f,g)]
    if v.ndim!=2 or any(x.shape!=v.shape for x in (w,f,g)) or v.shape[1]<3:
        raise ValueError('Matching group-by-peer arrays with at least three peers required')
    if not all(np.isfinite(x).all() for x in (v,w,f,g)) or np.any((f<0)|(f>1)|(g<0)|(g>1)):
        raise ValueError('Finite values and fits in [0,1] required')
    n=3*(v.shape[1]//3);v,w,f,g=[x[:,:n].reshape(-1,3) for x in (v,w,f,g)]
    h=np.zeros(len(v))
    for i in range(3):
        j,k=[x for x in range(3) if x!=i]
        avj=(v[:,i]>v[:,j])+.5*(v[:,i]==v[:,j]);avk=(v[:,i]>v[:,k])+.5*(v[:,i]==v[:,k])
        bwj=(w[:,i]>w[:,j])+.5*(w[:,i]==w[:,j]);bwk=(w[:,i]>w[:,k])+.5*(w[:,i]==w[:,k])
        h+=((avj-f[:,i])*(bwk-g[:,i])+(avk-f[:,i])*(bwj-g[:,i]))/6
    return h


def one_sided_certificate(mean,variance_scores,effective,f,g,bias_upper,delta=.0001,alpha=.05,kind='variance'):
    if not 0<delta<alpha<1 or not np.isfinite(mean) or not np.isfinite(bias_upper):
        raise ValueError('Need finite mean/bias and 0 < delta < alpha < 1')
    if not isinstance(effective,(int,np.integer)) or effective<1:raise ValueError('Positive effective count required')
    lo,hi=kernel_range(f,g);R=hi-lo
    if not lo-1e-12<=mean<=hi+1e-12:raise ValueError('Mean outside fitted kernel range')
    margin=max(float(mean)-float(bias_upper),0.)
    result=dict(mean=float(mean),bias_upper=float(bias_upper),kernel_lower=lo,kernel_upper=hi,effective_triples=int(effective),delta=float(delta),alpha=float(alpha))
    if kind=='range':
        radius=R*np.sqrt(np.log(1/(alpha-delta))/(2*effective));p=min(1.,delta+np.exp(-2*effective*(margin/R)**2));s2=None
    elif kind in ('variance','independent_bernstein'):
        h=np.asarray(variance_scores,float)
        if h.ndim!=1 or len(h)!=effective or len(h)<2 or not np.isfinite(h).all():
            raise ValueError('At least two disjoint triple scores matching effective count required')
        if np.any(h<lo-1e-12) or np.any(h>hi+1e-12):raise ValueError('Triple scores outside kernel range')
        if kind=='independent_bernstein' and not np.isclose(mean,h.mean(),rtol=1e-12,atol=1e-14):
            raise ValueError('Independent-triple inference requires the mean of those same triples')
        s2=float(h.var(ddof=1));x=np.log(2/(alpha-delta));A=np.sqrt(2*s2/effective)
        C=(2*R/np.sqrt(effective*(effective-1))+R/(3*effective)) if kind=='variance' else 7*R/(3*(effective-1))
        rb=A*np.sqrt(x)+C*x
        # The full-U route includes a proved deterministic Hoeffding cap.
        radius=min(R*np.sqrt(x/(2*effective)),rb) if kind=='variance' else rb
        root=2*margin/(A+np.sqrt(A*A+4*C*margin)) if margin else 0.
        exponent=root*root
        if kind=='variance':exponent=max(exponent,2*effective*(margin/R)**2)
        p=min(1.,delta+2*np.exp(-exponent))
    else:raise ValueError('Unknown fixed bound construction')
    result.update(radius=float(radius),lower_bound=float(mean-bias_upper-radius),p=float(p),reject=bool(p<alpha),kernel_sample_variance=s2,bound=kind)
    return result


def planned_triples(min_effect,bias_width,f,g,alpha=.05,delta=.0001,beta=.2):
    """Sufficient range-bound sample plan after validation, before evaluation.

    The declared effect must exceed the validation bias interval width. This
    plans conditional detection on the simultaneous validation event, and does
    not infer that the actual population effect exceeds the declared effect.
    """
    if not 0<delta<alpha<1 or not 0<beta<1 or not min_effect>bias_width>=0:
        raise ValueError('Need effect > bias width >=0 and valid error probabilities')
    lo,hi=kernel_range(f,g)
    amount=(hi-lo)**2*(np.sqrt(np.log(1/(alpha-delta)))+np.sqrt(np.log(1/beta)))**2/(2*(min_effect-bias_width)**2)
    return int(np.floor(amount)+1)
