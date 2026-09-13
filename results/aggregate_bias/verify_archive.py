"""Independent archive reconstruction; no production/statistical helper imports.

The full rank U mean is recovered using a global Kendall tie identity, not the
producer's focal Fenwick counts. Probability inversion uses scalar bisection.
The files audited are existing exposed-task diagnostics, not new confirmation.
"""
from pathlib import Path
import csv, json, math, hashlib, itertools, sys, argparse
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.stats import rankdata, kendalltau, beta, binom

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'forecast_confirmation'
EXPECTED = HERE/'archive/all_candidates.csv'
CANDIDATES = ['persistence','seasonal_day','seasonal_week','ridge_full','hist_gradient_boosting','extra_trees','category_copy','independent_noise']
C=32
DELTA = .1*.05/(8*math.fsum(1/j for j in range(1,9)))

def category_mean(z,values):
    return np.array([math.fsum(values[z==c])/sum(z==c) if np.any(z==c) else .5 for c in range(C)])

def comparison(v,i,j):
    return .5*(1+np.sign(v[i]-v[j]))

def aggregate(p,aa,bb,na,nb,absent):
    keep=[c for c in range(C) if p[c]>0 and na[c]>0 and nb[c]>0]
    x=math.log(3/DELTA)
    center=math.fsum(p[c]*aa[c]*bb[c] for c in keep)
    va=math.fsum((p[c]*bb[c])**2/(4*na[c]) for c in keep)
    vb=math.fsum((p[c]*aa[c])**2/(4*nb[c]) for c in keep)
    dc=[p[c]/(4*math.sqrt(na[c]*nb[c])) for c in keep]
    ra=math.sqrt(2*x*va);rb=math.sqrt(2*x*vb)
    r2=math.sqrt(2*x*math.fsum(d*d for d in dc))+x*max(dc,default=0.)
    missing=math.fsum(p[c]*absent[c] for c in range(C) if p[c]>0 and c not in keep)
    return dict(center=center,linear_a_radius=ra,linear_b_radius=rb,product_radius=r2,absent_allowance=missing,upper=center+ra+rb+r2+missing,categories_used=len(keep),categories_absent=int(sum(p[c]>0 and c not in keep for c in range(C))))

def rectangles(p, f,g,zv,xc,yc, original):
    n=np.bincount(zv,minlength=C);n_total=len(zv)
    mx=category_mean(zv,xc);my=category_mean(zv,yc)
    radius=np.full(C,np.inf)
    radius[n>0]=np.sqrt(math.log((8 if original else 4)*C/DELTA)/(2*n[n>0]))
    L=np.maximum(0,mx-radius)-f;U=np.minimum(1,mx+radius)-f
    S=np.maximum(0,my-radius)-g;T=np.minimum(1,my+radius)-g
    q=np.max(np.stack([L*S,L*T,U*S,U*T]),axis=0)
    if not original:return float(p@q)
    caps=np.ones(C)
    for c in range(C):
        if n[c]<n_total:caps[c]=beta.isf(DELTA/(2*C),n[c]+1,n_total-n[c])
    total=0.;left=1.
    for c in sorted(range(C),key=lambda c:-q[c]):
        take=min(left,caps[c]);total+=take*q[c];left-=take
        if left<=0:break
    assert left<1e-10
    return total

def full_u_global(x,y,f,g):
    n=len(x);sx=rankdata(x)-1;sy=rankdata(y)-1
    # Sum of all same-reference products, using concordance rather than
    # per-focal joint dominance counts. All tie counts are exact integers.
    pairs=n*(n-1)//2
    tx=sum(int(k)*(int(k)-1)//2 for k in np.unique(x,return_counts=True)[1])
    ty=sum(int(k)*(int(k)-1)//2 for k in np.unique(y,return_counts=True)[1])
    denominator=math.sqrt((pairs-tx)*(pairs-ty))
    signed_pairs=0 if denominator==0 else int(round(float(kendalltau(x,y).statistic)*denominator))
    joint=.5*pairs+.5*signed_pairs
    return float((np.dot(sx,sy)-joint)/(n*(n-1)*(n-2))
                 -(np.dot(f,sy)+np.dot(g,sx))/(n*(n-1))+np.dot(f,g)/n)

def disjoint_kernels(x,y,f,g):
    n=3*(len(x)//3)
    xx,yy,ff,gg=[a[:n].reshape(-1,3) for a in [x,y,f,g]]
    h=np.zeros(n//3)
    for i,j,k in itertools.permutations(range(3)):
        h+=.5*(1+np.sign(xx[:,i]-xx[:,j]))*.5*(1+np.sign(yy[:,i]-yy[:,k]))
        h-=ff[:,i]*.5*(1+np.sign(yy[:,i]-yy[:,k]))
        h-=gg[:,i]*.5*(1+np.sign(xx[:,i]-xx[:,j]))
        h+=ff[:,i]*gg[:,i]
    return h/6

def infer(mean,h,f,g,B):
    corners=np.concatenate([f*g,-f*(1-g),-(1-f)*g,(1-f)*(1-g)])
    lo=float(corners.min());hi=float(corners.max());width=hi-lo;j=len(h)
    s=math.sqrt(math.fsum((h-float(h.mean()))**2)/(j-1))
    a=s*math.sqrt(2/j);c=2*width/math.sqrt(j*(j-1))+width/(3*j)
    lx=math.log(2/(.05-DELTA))
    radius=min(width*math.sqrt(lx/(2*j)),a*math.sqrt(lx)+c*lx)
    excess=max(mean-B,0.)
    left=0.;right=1.
    while a*math.sqrt(right)+c*right<excess:right*=2
    for _ in range(100):
        mid=(left+right)/2
        if a*math.sqrt(mid)+c*mid>excess:right=mid
        else:left=mid
    exponent=max((left+right)/2,2*j*(excess/width)**2)
    pu=min(1.,DELTA+2*math.exp(-exponent))
    # The original finite mixture is reconstructed independently; this is a
    # test p-value with validation failure added, not an unconditional e-value.
    threshold=(B-lo)/width
    if threshold<0:pb=DELTA
    elif threshold>=1:pb=1.
    elif threshold==0:pb=DELTA if np.any(h>lo) else 1.
    else:
        fractions=np.exp(np.linspace(math.log(1e-4),math.log(.99),64))
        logs=np.log1p(fractions[:,None]*((h[None,:]-lo)/(width*threshold)-1)).sum(axis=1)
        maximum=float(logs.max())
        logcapital=maximum+math.log(float(np.exp(logs-maximum).sum()))-math.log(64)
        pb=1. if logcapital<=0 else min(1.,DELTA+math.exp(-logcapital))
    return dict(U_radius=radius,U_lower=mean-B-radius,U_p=pu,betting_p=pb)

def by(p):
    p=np.asarray(p,float)
    n=len(p);H=math.fsum(1/i for i in range(1,n+1));order=np.argsort(p,kind='stable')
    answer=np.ones(n);last=1.
    for rank in range(n,0,-1):
        idx=order[rank-1];last=min(last,float(p[idx])*n*H/rank);answer[idx]=last
    return answer

def small_checks():
    rng=np.random.default_rng(13092644);worst=0.;cases=0
    for n in [3,4,7,11,24]:
        for _ in range(5):
            x=rng.integers(0,4,n).astype(float);y=rng.integers(0,3,n).astype(float)
            f=rng.uniform(0,1,n);g=rng.uniform(0,1,n)
            direct=math.fsum((.5*(1+np.sign(x[i]-x[j]))-f[i])*(.5*(1+np.sign(y[i]-y[k]))-g[i]) for i,j,k in itertools.permutations(range(n),3))/(n*(n-1)*(n-2))
            err=abs(direct-full_u_global(x,y,f,g));worst=max(worst,err);cases+=1
            assert err<2e-15
    mgfcases=0;maxslack=-math.inf
    for na,nb in [(1,1),(2,3),(8,11),(40,25)]:
        for pa,pb in itertools.product([.02,.3,.5,.95],repeat=2):
            ea=np.arange(na+1)/na-pa;eb=np.arange(nb+1)/nb-pb
            probs=binom.pmf(np.arange(na+1),na,pa)[:,None]*binom.pmf(np.arange(nb+1),nb,pb)[None,:]
            d=1/(4*math.sqrt(na*nb))
            for z in [-.8,-.3,.2,.8]:
                exact=float((probs*np.exp((z/d)*ea[:,None]*eb[None,:])).sum())
                bound=1/math.sqrt(1-z*z);assert exact<=bound+1e-12
                maxslack=max(maxslack,exact-bound);mgfcases+=1
    return dict(exact_full_U_cases=cases,max_full_U_absolute_error=worst,exact_Bernoulli_product_MGF_cases=mgfcases,max_MGF_minus_bound=maxslack)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output;assert not out.exists(),'Use a new output directory';out.mkdir(parents=True)
    checks=small_checks();records=[];vectors=[];expected=pd.read_csv(EXPECTED)
    for task in ['appliances','metro']:
        ar=np.load(SOURCE/task/'forecast_archive.npz');ix=np.load(SOURCE/task/'sampling_indices.npz')
        tr=ix['training'];va=ix['validation'];ev=ix['evaluation'];mask=ar['selection'];y=ar['y'][mask]
        for base in ['ridge','seasonal_day']:
            z=ar[base+'__category'][mask].astype(int);p=np.bincount(z,minlength=C)/len(z)
            g=category_mean(z[tr],(rankdata(y[tr])-1)/(len(tr)-1))
            bv=comparison(y,va[:,0],va[:,1]);zv=z[va[:,0]]
            nA=np.bincount(zv[:4096],minlength=C);nB=np.bincount(zv[4096:],minlength=C)
            true_my=category_mean(z,(rankdata(y)-.5)/len(y))
            for candidate in CANDIDATES:
                x=ar[base+'__'+candidate][mask]
                f=category_mean(z[tr],(rankdata(x[tr])-1)/(len(tr)-1))
                av=comparison(x,va[:,0],va[:,1])
                aa=category_mean(zv[:4096],av[:4096])-f;bb=category_mean(zv[4096:],bv[4096:])-g
                absent=np.max(np.stack([f*g,-f*(1-g),-(1-f)*g,(1-f)*(1-g)]),axis=0)
                ag=aggregate(p,aa,bb,nA,nB,absent)
                oracle=float(p@((category_mean(z,(rankdata(x)-.5)/len(x))-f)*(true_my-g)))
                xx,yy,ff,gg=x[ev],y[ev],f[z[ev]],g[z[ev]]
                center=full_u_global(xx,yy,ff,gg);h=disjoint_kernels(xx,yy,ff,gg)
                budgets={'original_rectangle':rectangles(p,f,g,zv,av,bv,True),'known_mass_rectangle':rectangles(p,f,g,zv,av,bv,False),'known_mass_aggregate':ag['upper'],'oracle_diagnostic':oracle}
                for method,B in budgets.items():
                    row=dict(task=task,baseline=base,candidate=candidate,budget_method=method,bias=B,true_bias=oracle,mean=center,**infer(center,h,f,g,B))
                    if method=='known_mass_aggregate':row.update(ag)
                    records.append(row)
                vectors.append(dict(task=task,baseline=base,candidate=candidate,known_p=p.tolist(),training_f=f.tolist(),training_g=g.tolist(),count_A=nA.tolist(),count_B=nB.tolist(),residual_mean_A=aa.tolist(),residual_mean_B=bb.tolist(),true_bias=oracle,U_mean=center,kernel_variance=float(h.var(ddof=1)),aggregate=ag))
    frame=pd.DataFrame(records)
    for method in ['U','betting']:
        frame[method+'_BY']=frame.groupby(['task','baseline','budget_method'],sort=False)[method+'_p'].transform(by)
        frame[method+'_retained']=frame[method+'_BY']<=.05
    keys=['task','baseline','candidate','budget_method']
    paired=frame.merge(expected,on=keys,suffixes=('_independent','_producer'),validate='one_to_one')
    errors={};numeric=['bias','true_bias','mean','U_radius','U_lower','U_p','betting_p','U_BY','betting_BY','center','linear_a_radius','linear_b_radius','product_radius','absent_allowance','upper','categories_used','categories_absent']
    for col in numeric:
        a=paired[col+'_independent'].to_numpy();b=paired[col+'_producer'].to_numpy();occupied=np.isfinite(a)&np.isfinite(b)
        assert np.array_equal(np.isfinite(a),np.isfinite(b)),col
        errors[col]=float(np.max(np.abs(a[occupied]-b[occupied])))
        np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-12,equal_nan=True,err_msg=col)
    for col in ['U_retained','betting_retained']:
        assert np.array_equal(paired[col+'_independent'],paired[col+'_producer']),col
    frame.to_csv(out/'all_candidates.csv',index=False)
    (out/'reconstructed_vectors.json').write_text(json.dumps(vectors,indent=2)+'\n')
    sources={'stored_archive_results':hashlib.sha256(EXPECTED.read_bytes()).hexdigest(),'independent_verifier':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for task in ['appliances','metro']:
        for name in ['forecast_archive.npz','sampling_indices.npz']:
            p=SOURCE/task/name;sources[task+'/'+name]=hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ['aggregate_bias_bound.py','THEORY.md']:
        p=HERE/name;sources[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=dict(status='PASS',scope='Independent mathematical and numeric reconstruction on exposed archives. No producer/helper imports, no new forecast retraining or unseen-data confirmation.',source_hashes=sources,versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),candidate_count=32,result_rows=len(frame),small_checks=checks,max_absolute_errors=errors,all_U_and_betting_BY_decisions_match=True,retained_counts=frame.groupby('budget_method')[['U_retained','betting_retained']].sum().to_dict(),numerical_scope='Binary floating-point checks with declared tolerance; not an exact-arithmetic bound certificate. Category mass contract is exact mathematical proportions, evaluated numerically here.')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
