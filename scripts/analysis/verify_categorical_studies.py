"""Independent stored-output and primitive replay; imports no producer module."""
import argparse, hashlib, itertools, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog, brentq
from scipy.stats import beta, binom, norm, rankdata

D=.0001; A=.05; SEED=202609121140
KEY=['replication','design','signal','training_rows','validation_pairs','groups','peers']
CELL=KEY[1:]
DESIGNS={'two_balanced':(np.array([.5,.5]),np.array([.2,.8])),
 'eight_balanced':(np.full(8,.125),np.linspace(.2,.8,8)),
 'eight_rare':(np.array([.005]+[.995/7]*7),np.linspace(.2,.8,8))}
ERR={};CHECKS={};LPERR=0.;CPERR=0.;COUNTS=0

def check(name,got,want,tol=5e-12):
    x=np.asarray(got);y=np.asarray(want)
    if x.dtype.kind in 'biu' and y.dtype.kind in 'biu':
        assert np.array_equal(x,y),name
        CHECKS[name]=CHECKS.get(name,0)+x.size
        return
    assert x.shape==y.shape,(name,x.shape,y.shape)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)),name
    e=float(np.max(np.abs(x-y),initial=0));ERR[name]=max(ERR.get(name,0),e)
    assert e<=tol,(name,e)
    CHECKS[name]=CHECKS.get(name,0)+x.size

def primitive(rep,design,gamma,part,shape):
    prob,p=DESIGNS[design]
    seed=[SEED,rep,len(prob),int(design=='eight_rare'),int(100*gamma),part]
    rng=np.random.default_rng(np.random.SeedSequence(seed))
    z=rng.choice(len(prob),size=shape,p=prob)
    u=rng.random(shape)
    # Four state masses conditional on the focal category, in producer RNG order.
    c=p[z];joint=c*c+gamma*c*(1-c)
    thresholds=np.stack([joint,c,2*c-joint],axis=-1)
    state=np.sum(u[...,None]>=thresholds,axis=-1)
    v=np.isin(state,[0,1]).astype(float)
    w=np.isin(state,[0,2]).astype(float)
    return z,v,w

def fitted_means(z,v,c):
    # Generic comparison ranks, rather than the producer's Bernoulli shortcut.
    r=(rankdata(v,method='average')-1)/(len(v)-1)
    ans=np.full(c,.5)
    for i in range(c):
        if np.any(z==i):ans[i]=r[z==i].mean()
    return ans

def comparisons(x):
    # Explicit (focal, reference) comparison matrix, with diagonal removed.
    a=(x[:,:,None]>x[:,None,:]).astype(float)+.5*(x[:,:,None]==x[:,None,:])
    a[:,np.arange(x.shape[1]),np.arange(x.shape[1])]=0
    return a

def envelopes(f,g,n,av,aw,mode,lp=False):
    c=len(n);L=int(sum(n));spend=D if mode=='maxcell' else D/2
    rv=np.sqrt(np.divide(np.log(4*c/spend),2*n,out=np.full(c,np.inf),where=n>0))
    lv=np.where(n>0,np.maximum(0,av-rv),0);uv=np.where(n>0,np.minimum(1,av+rv),1)
    lw=np.where(n>0,np.maximum(0,aw-rv),0);uw=np.where(n>0,np.minimum(1,aw+rv),1)
    ev=np.maximum(abs(lv-f),abs(uv-f));ew=np.maximum(abs(lw-g),abs(uw-g));q=ev*ew
    if mode=='maxcell':return max(q)
    upper=np.ones(c)
    if L:
        use=n<L;upper[use]=beta.ppf(1-D/(2*c),n[use]+1,L-n[use])
    # Independently solve using a sorted capacity cumulative-sum formula.
    order=np.argsort(q)[::-1];caps=upper[order]
    prefix=np.r_[0,np.cumsum(caps)[:-1]]
    mass=np.minimum(caps,np.maximum(0,1-prefix))
    assert abs(mass.sum()-1)<1e-10
    value=float(q[order]@mass)
    if lp:
        global LPERR,CPERR
        solution=linprog(-q,A_eq=np.ones((1,c)),b_eq=[1.],bounds=[(0,u) for u in upper],method='highs')
        assert solution.success
        LPERR=max(LPERR,abs(value+solution.fun));assert abs(value+solution.fun)<1e-10
        # Independent binomial-CDF inversion on one category per sampled cell.
        idx=int(np.argmin(n))
        if n[idx]<L:
            u=brentq(lambda p:binom.cdf(int(n[idx]),L,p)-D/(2*c),0,1,xtol=1e-14)
            CPERR=max(CPERR,abs(u-upper[idx]));assert abs(u-upper[idx])<1e-10
    return value

def record_algebra(df,label):
    c=len(df)
    assert c==108000 and not df.duplicated(KEY).any()
    cellcounts=df.groupby(CELL).size();assert len(cellcounts)==108 and (cellcounts==1000).all()
    check(label+'.validation_rows',df.validation_rows.to_numpy(),2*df.validation_pairs.to_numpy())
    check(label+'.cost',df.total_observations.to_numpy(),(df.training_rows+2*df.validation_pairs+df.groups*df.peers).to_numpy())
    j=(df.groups*(df.peers//3)).to_numpy();check(label+'.effective_triples',df.effective_triples.to_numpy(),j)
    width=(df.kernel_upper-df.kernel_lower).to_numpy();assert np.all(width>0)
    radius=width*np.sqrt(np.log(1/(A-D))/(2*j))
    check(label+'.radius',df.radius.to_numpy(),radius)
    lower=df['mean'].to_numpy()-df.bias_upper.to_numpy()-radius
    check(label+'.lower',df.lower_bound.to_numpy(),lower)
    margin=np.maximum(df['mean'].to_numpy()-df.bias_upper.to_numpy(),0)
    p=np.minimum(1,D+np.exp(-2*j*(margin/width)**2))
    check(label+'.p',df.p.to_numpy(),p)
    check(label+'.reject',df.reject.to_numpy(),p<A)
    check(label+'.lower_reject_equivalence',df.reject.to_numpy(),lower>0)
    check(label+'.budget_coverage',df.budget_covers.to_numpy(),abs(df.exact_bias.to_numpy())<=df.bias_upper.to_numpy())
    check(label+'.lower_coverage',df.lower_covers.to_numpy(),df.lower_bound.to_numpy()<=df.target.to_numpy())
    check(label+'.normal_p',df.normal_p.to_numpy(),norm.sf(df['mean'].to_numpy()/df.group_se.to_numpy()))
    check(label+'.bias_to_se',df.bias_to_se.to_numpy(),df.bias_upper.to_numpy()/df.group_se.to_numpy(),1e-8)
    for design,(prob,pv) in DESIGNS.items():
        s=df[df.design==design]
        target=s.signal.to_numpy()*(prob@(pv*(1-pv)))/4
        check(label+'.known_target',s.target.to_numpy(),target)
    assert df.p.min()>=D and np.all(df.kernel_lower<=df['mean']) and np.all(df['mean']<=df.kernel_upper)

def summary_check(df,path,label):
    saved=pd.read_csv(path).set_index(CELL).sort_index()
    rows=[]
    for key,x in df.groupby(CELL):
        n=len(x);k=int(x.reject.sum());p=k/n;z=norm.ppf(.975)
        center=(p+z*z/(2*n))/(1+z*z/n)
        half=z/(1+z*z/n)*np.sqrt(p*(1-p)/n+z*z/(4*n*n))
        rows.append(dict(zip(CELL,key),replications=n,rejections=k,rate=p,wilson_low=center-half,wilson_high=center+half,
          normal_rejections=int((x.normal_p<A).sum()),budget_failures=int((~x.budget_covers).sum()),coverage_failures=int((~x.lower_covers).sum()),
          mean_bias_budget=x.bias_upper.mean(),mean_sampling_radius=x.radius.mean(),mean_lower_bound=x.lower_bound.mean(),mean_p=x.p.mean(),target=x.target.iloc[0],total_observations=x.total_observations.iloc[0]))
    calc=pd.DataFrame(rows).set_index(CELL).sort_index()
    assert list(saved.columns)==list(calc.columns)
    for col in calc:check(label+'.summary.'+col,saved[col].to_numpy(),calc[col].to_numpy())
    return saved

def execute(base,out):
    start=time.time()
    files={}; frames={}; summaries={}
    for name,folder,src in [('maxcell','maximum_cell','category_reference_certificate_maxcell.py'),('weighted','probability_weighted','category_reference_certificate.py')]:
        protocol=json.loads((base/folder/'protocol.json').read_text())
        h=hashlib.sha256((base.parents[1]/'scripts'/'analysis'/src).read_bytes()).hexdigest();assert h==protocol['source_sha256']
        assert protocol['replications']==1000 and protocol['seed']==SEED
        frame=pd.read_csv(base/folder/'replications.csv.gz');frames[name]=frame
        record_algebra(frame,name);summaries[name]=summary_check(frame,base/folder/'summary.csv',name)
        for p in [base.parents[1]/'scripts'/'analysis'/src,base/folder/'protocol.json',base/folder/'replications.csv.gz',base/folder/'summary.csv']:
            files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    lhs=frames['maxcell'].set_index(KEY).sort_index();rhs=frames['weighted'].set_index(KEY).sort_index()
    invariant=['validation_rows','total_observations','target','exact_bias','normal_p','mean','radius','effective_triples','kernel_lower','kernel_upper','group_se']
    for col in invariant:check('paired.invariant.'+col,lhs[col].to_numpy(),rhs[col].to_numpy())
    full=set(np.linspace(0,999,20,dtype=int).tolist())
    # Every training/validation primitive is rebuilt. Twenty replications additionally
    # replay all evaluation scores with literal comparison matrices.
    indexed={name:df.set_index(KEY).sort_index() for name,df in frames.items()}
    unseen=0;empty_ranges=[]
    for rep in range(1000):
        for design,(prob,succ) in DESIGNS.items():
            c=len(prob)
            true=.5+.5*(succ-prob@succ)
            for gamma in (0.,.25,.5):
                tz,tv,tw=primitive(rep,design,gamma,1,(8192,))
                vz,vv,vw=primitive(rep,design,gamma,2,(8192,2))
                av=(vv[:,0]>vv[:,1])+.5*(vv[:,0]==vv[:,1])
                aw=(vw[:,0]>vw[:,1])+.5*(vw[:,0]==vw[:,1])
                if rep in full:
                    ez,ev,ew=primitive(rep,design,gamma,0,(400,64));am=comparisons(ev);bm=comparisons(ew)
                    asum=am.sum(-1);bsum=bm.sum(-1);diagprod=(am*bm).sum(-1)
                    distinct_ab=(asum*bsum-diagprod)/(63*62)
                for train in (32,512,8192):
                    f=fitted_means(tz[:train],tv[:train],c);g=fitted_means(tz[:train],tw[:train],c)
                    corners=np.array([[(a-f[i])*(b-g[i]) for i in range(c)] for a,b in itertools.product([0.,1.],repeat=2)])
                    lo=float(corners.min());hi=float(corners.max())
                    truebias=float(prob@((true-f)*(true-g)))
                    if rep in full:
                        row=distinct_ab-f[ez]*bsum/63-g[ez]*asum/63+f[ez]*g[ez]
                        gs=row.mean(-1)
                    for validation in (512,8192):
                        count=np.array([np.sum(vz[:validation,0]==i) for i in range(c)])
                        unseen+=int(np.sum(count==0));assert count.sum()==validation
                        mv=np.array([av[:validation][vz[:validation,0]==i].mean() if count[i] else 0. for i in range(c)])
                        mw=np.array([aw[:validation][vz[:validation,0]==i].mean() if count[i] else 0. for i in range(c)])
                        for method in ('maxcell','weighted'):
                            bud=envelopes(f,g,count,mv,mw,method,lp=rep in full and method=='weighted')
                            for groups in (100,400):
                                key=(rep,design,gamma,train,validation,groups,64)
                                rec=indexed[method].loc[key]
                                check(method+'.fresh.budget',np.array(rec.bias_upper),np.array(bud))
                                check(method+'.fresh.exact_bias',np.array(rec.exact_bias),np.array(truebias))
                                check(method+'.fresh.kernel_lower',np.array(rec.kernel_lower),np.array(lo))
                                check(method+'.fresh.kernel_upper',np.array(rec.kernel_upper),np.array(hi))
                                if rep in full:
                                    m=float(gs[:groups].mean());se=float(gs[:groups].std(ddof=1)/np.sqrt(groups))
                                    check(method+'.fresh.mean',np.array(rec['mean']),np.array(m))
                                    check(method+'.fresh.se',np.array(rec.group_se),np.array(se))
                                    p=min(1,D+np.exp(-2*groups*21*(max(m-bud,0)/(hi-lo))**2))
                                    check(method+'.fresh.p',np.array(rec.p),np.array(p))
                                    check(method+'.fresh.reject',np.array(rec.reject),np.array(p<A))
        if rep%100==99:print('Rebuilt training/validation replications',rep+1,'elapsed',round(time.time()-start,1),flush=True)
    paired=pd.read_csv(base/'paired_budget_comparison.csv').set_index(CELL).sort_index()
    merged=summaries['maxcell'].add_suffix('_max').join(summaries['weighted'].add_suffix('_weighted'))
    assert set(paired.columns)==set(merged.columns)
    for col in paired:check('paired.summary.'+col,paired[col].to_numpy(),merged[col].to_numpy())
    useful=summaries['weighted'].reset_index().query('training_rows==8192 and validation_pairs==8192 and groups==400')
    differences={}
    for key,x in frames['weighted'].groupby(CELL):
        y=frames['maxcell'].set_index(KEY).loc[x.set_index(KEY).index].reset_index()
        d=x.reject.to_numpy().astype(int)-y.reject.to_numpy().astype(int)
        differences[str(key)]={'weighted_only':int((d==1).sum()),'maxcell_only':int((d==-1).sum()),'paired_difference':float(d.mean()),'paired_se':float(d.std(ddof=1)/np.sqrt(len(d)))}
    report=dict(status='PASS',producer_sources=files,checks=CHECKS,max_absolute_errors=ERR,full_primitive_replications=sorted(full),all_training_validation_replications=1000,
      record_rows_per_method=108000,summary_cells_per_method=108,linprog_calls=20*3*3*3*2,linprog_max_error=LPERR,independent_binomial_inversion_max_error=CPERR,
      validation_empty_cells_counted_with_repeated_training=unseen,elapsed_seconds=time.time()-start,
      null_rejections={k:int(df.query('signal==0').reject.sum()) for k,df in frames.items()},
      budget_failures={k:int((~df.budget_covers).sum()) for k,df in frames.items()},coverage_failures={k:int((~df.lower_covers).sum()) for k,df in frames.items()},
      useful_cells=useful.to_dict(orient='records'),paired_differences=differences,
      caveats=['20 complete evaluation primitive replications; every training/validation primitive and all recorded-row algebra reconstructed.','Weighted design was adapted after the maxcell study and uses exactly matched primitives; it is not independent confirmation.','No empirical zero-failure count proves a probability guarantee; the proof is separately audited.','The finite target is declared-category population-midrank covariance, not a generic observed ranked/HAC-panel target.'])
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('AUDIT PASS',round(time.time()-start,1),'seconds',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();execute(a.base,a.output)
