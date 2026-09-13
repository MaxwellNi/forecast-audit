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
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import beta, rankdata

ROOT = Path(__file__).resolve().parents[2]
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
    rad=np.full(c,np.inf);np.sqrt(np.log(8*c/delta)/(2*n),where=n>0,out=rad)
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


def rebuild(rep,seed_base=202609121457):
    rows=[]
    for di,(design,prob,success) in enumerate(DESIGNS):
      c=len(prob)
      # Truth derived by enumerating the categorical joint law and ranks.
      for si,signal in enumerate(SIGNALS):
        law=np.array([[s*s+signal*s*(1-s),s-(s*s+signal*s*(1-s)),
                       s-(s*s+signal*s*(1-s)),1-2*s+s*s+signal*s*(1-s)] for s in success])
        states=np.array([[1,1],[1,0],[0,1],[0,0]],float)
        marginal=np.sum(prob[:,None]*law,axis=0)
        rv=compare(states[:,0,None],states[None,:,0])@marginal
        rw=compare(states[:,1,None],states[None,:,1])@marginal
        muv=law@rv;muw=law@rw
        target=float(np.sum(prob[:,None]*law*(rv[None,:]-muv[:,None])*(rw[None,:]-muw[:,None])))
        seed=[seed_base,rep,di,si]
        z,v,w=random_sample(seed+[0],(1600,64),prob,success,signal)
        tz,tv,tw=random_sample(seed+[1],(8192,),prob,success,signal)
        vz,vv,vw=random_sample(seed+[2],(8192,2),prob,success,signal)
        fv,fw=fitted_means(tz,tv,c),fitted_means(tz,tw,c)
        sums=rank_sums(v,w)
        psums={ng:pooled_rank_sums(v[:ng],w[:ng]) for ng in [25,100,400,1600]}
        for fit in ['estimated']+(['opposed_shifts'] if c==2 else []):
          f,g=(fv,fw) if fit=='estimated' else (np.clip(fv+.15,0,1),np.clip(fw-.15,0,1))
          group=full_u(sums,f[z],g[z],64);h=blocks(v,w,f[z],g[z])
          corners=np.array([(a-f)*(b-g) for a,b in product([0.,1.],repeat=2)])
          lo,hi=float(corners.min()),float(corners.max())
          bias=float(prob@((muv-f)*(muw-g)))
          for val in ([512,8192] if fit=='estimated' else [8192]):
            bud=budget(f,g,vz[:val],vv[:val],vw[:val])
            for ng in [25,100,400,1600]:
              hh=h[:ng*21];j=len(hh);s2=hh.var(ddof=1)
              pz=z[:ng].reshape(1,-1);pv=v[:ng].reshape(1,-1);pw=w[:ng].reshape(1,-1)
              ph=blocks(pv,pw,f[pz],g[pz]);pm=float(full_u(psums[ng],f[pz.ravel()],g[pz.ravel()],64*ng))
              means=[float(group[:ng].mean())]*3+[float(hh.mean()),pm]
              for method,mean in zip(METHODS,means):
                kind='range' if method.endswith('_range') else ('independent_bernstein' if method=='independent_triples' else 'variance')
                b=bud['absolute_allowance'] if method=='absolute_range' else bud['signed_allowance']
                jj,ss=(len(ph),ph.var(ddof=1)) if method=='pooled_variance' else (j,s2)
                out=inference(mean,ss,jj,lo,hi,b,kind)
                rows.append(dict(replication=rep,design=design,signal=signal,fit=fit,training_rows=8192,
                  validation_pairs=val,groups=ng,peers=64,total_observations=8192+2*val+64*ng,method=method,
                  target=target,exact_bias=bias,mean=mean,bias_upper=b,kernel_lower=lo,kernel_upper=hi,
                  effective_triples=jj,kernel_sample_variance=ss if kind!='range' else np.nan,
                  lower_covers=bool(out['lower_bound']<=target),allowance_covers=bool(bias<=b),
                  **bud,**{k:np.asarray(x).item() for k,x in out.items()}))
    return pd.DataFrame(rows)


def maxima(expected,actual,columns):
    out={}
    for col in columns:
        a,b=np.asarray(expected[col]),np.asarray(actual[col])
        if a.dtype==bool or a.dtype.kind in 'OUS':
            assert np.array_equal(a,b),col
            out[col]=0
        else:
            assert np.array_equal(np.isnan(a),np.isnan(b)),col
            error=float(np.nanmax(abs(a-b))) if np.any(~np.isnan(a)) else 0.
            assert error<1e-10,(col,error)
            out[col]=error
    return out


def summary(df):
    return df.groupby(KEYS,dropna=False).agg(replications=('replication','size'),rejections=('reject','sum'),
      power=('reject','mean'),target=('target','first'),mean=('mean','mean'),bias=('exact_bias','mean'),
      bias_allowance=('bias_upper','mean'),absolute_allowance=('absolute_allowance','mean'),
      signed_allowance=('signed_allowance','mean'),bias_width=('bias_interval_width','mean'),radius=('radius','mean'),
      lower=('lower_bound','mean'),coverage_failures=('lower_covers',lambda a:int((~a).sum())),
      allowance_failures=('allowance_covers',lambda a:int((~a).sum()))).reset_index()


def main(source,output):
    SOURCE=Path(source)
    HERE=Path(output)
    HERE.mkdir(parents=True,exist_ok=False)
    started=time.time()
    df=pd.read_csv(SOURCE/'replications.csv.gz')
    assert len(df)==500000 and df.replication.nunique()==1000
    assert not df.duplicated(['replication']+KEYS).any()
    assert df.groupby(KEYS).size().eq(1000).all()
    assert len(df.groupby(KEYS))==500
    arithmetic={}
    for kind,part in df.groupby('bound'):
        expected=inference(part['mean'],part.kernel_sample_variance,part.effective_triples,part.kernel_lower,
                           part.kernel_upper,part.bias_upper,kind,part.alpha,part.delta)
        arithmetic[kind]=maxima(expected,part,['radius','p','lower_bound','reject'])
    assert np.array_equal(df.lower_covers,df.lower_bound<=df.target)
    assert np.array_equal(df.allowance_covers,df.exact_bias<=df.bias_upper)
    assert (df.signed_allowance<=df.absolute_allowance+1e-12).all()
    assert (df.bias_interval_width>=0).all()
    absolute_bias_failures=int((abs(df.exact_bias)>df.absolute_allowance).sum())
    signed_lower_bias_failures=int((df.exact_bias<df.signed_allowance-df.bias_interval_width).sum())
    assert (df.total_observations==df.training_rows+2*df.validation_pairs+df.groups*df.peers).all()
    assert (df.effective_triples==np.where(df.method=='pooled_variance',(df.groups*df.peers)//3,df.groups*(df.peers//3))).all()
    paired=df.groupby(['replication']+[k for k in KEYS if k!='method'])
    for col in ['training_rows','validation_pairs','groups','peers','total_observations','signed_allowance','absolute_allowance']:
        assert paired[col].nunique().eq(1).all(),col
    sm=summary(df).sort_values(KEYS).reset_index(drop=True)
    original=pd.read_csv(SOURCE/'summary.csv').sort_values(KEYS).reset_index(drop=True)
    summary_errors=maxima(sm,original,[c for c in sm if c not in KEYS])
    sm.to_csv(HERE/'summary_recomputed.csv',index=False)
    full=[];rep_errors={}
    for rep in REPS:
        rebuilt=rebuild(rep)
        actual=df[df.replication==rep].sort_values(KEYS).reset_index(drop=True)
        rebuilt=rebuilt.sort_values(KEYS).reset_index(drop=True)
        assert len(rebuilt)==500
        rep_errors[str(rep)]=maxima(rebuilt,actual,[c for c in rebuilt if c not in KEYS+['replication']])
        full.append(rebuilt)
        print('Independent full replication passed:',rep,flush=True)
    pd.concat(full).to_csv(HERE/'rebuilt_replications.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    coverage=sm[KEYS+['replications','coverage_failures','allowance_failures','rejections']].copy()
    coverage['coverage_failure_cp_upper_95']=beta.ppf(.95,coverage.coverage_failures+1,coverage.replications-coverage.coverage_failures)
    coverage.to_csv(HERE/'all_cell_coverage.csv',index=False)
    null=df[df.signal<=0]
    pivot=df.pivot(index=['replication','design','signal','fit','training_rows','validation_pairs','groups','peers','total_observations'],columns='method',values=['reject','lower_bound','radius','p'])
    comparisons={}
    for other in ['absolute_range','signed_range','independent_triples','pooled_variance']:
        a=pivot['reject']['signed_variance'].astype(bool);b=pivot['reject'][other].astype(bool)
        diff=pivot['lower_bound']['signed_variance']-pivot['lower_bound'][other]
        comparisons[other]={'signed_variance_only_rejections':int((a&~b).sum()),'comparator_only_rejections':int((b&~a).sum()),
                            'signed_variance_higher_lower':int((diff>1e-12).sum()),'comparator_higher_lower':int((diff<-1e-12).sum()),
                            'median_radius_ratio':float((pivot['radius']['signed_variance']/pivot['radius'][other]).median())}
    pd.DataFrame(comparisons).T.to_csv(HERE/'paired_method_comparisons.csv')
    po=sm[sm.signal>0].pivot(index=[k for k in KEYS if k!='method'],columns='method',values='power').reset_index()
    po['variance_gain_over_absolute']=po.signed_variance-po.absolute_range
    po['variance_gain_over_triples']=po.signed_variance-po.independent_triples
    po['variance_gain_over_pool']=po.signed_variance-po.pooled_variance
    po.to_csv(HERE/'positive_cell_method_comparison.csv',index=False)
    # BY arithmetic is theoretical/prospective; study delta remains fixed 1e-4.
    k,q,rho=100,.05,.1;hk=sum(1/j for j in range(1,k+1));t=q/(k*hk)
    family={'K':k,'q':q,'rho':rho,'harmonic':hk,'first_threshold':t,'second_threshold':2*t,
            'study_delta':1e-4,'proposed_delta':rho*t,'rank_one_blocked':bool(1e-4>=t),'rank_two_floor_blocked':bool(1e-4>=2*t)}
    original_protocol=json.loads((SOURCE/'protocol_original.json').read_text())
    replay_protocol=json.loads((SOURCE/'protocol_hardened_replay.json').read_text())
    original_helper_sha256=original_protocol['source_sha256']['reference_certificate_efficiency.py']
    provenance=json.loads((SOURCE/'provenance.json').read_text())
    assert original_protocol['seed']==replay_protocol['seed']==202609121457
    assert original_protocol['replications']==replay_protocol['replications']==1000
    for name,entry in provenance['outputs'].items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==entry['sha256']
        assert entry['original_equals_hardened_replay'] is True
    files=[SOURCE/name for name in ['replications.csv.gz','summary.csv','protocol_original.json','protocol_hardened_replay.json','provenance.json']]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result={'status':'PASS','stored_rows':len(df),'cells':len(sm),'replications_per_cell':1000,'all_row_arithmetic_max_errors':arithmetic,
            'summary_max_errors':summary_errors,'full_rebuilt_replications':REPS,'full_rebuilt_rows':sum(map(len,full)),
            'full_rebuild_max_errors':rep_errors,'coverage_failures':int((~df.lower_covers).sum()),
            'allowance_failures':int((~df.allowance_covers).sum()),'null_rows':len(null),'null_rejections':int(null.reject.sum()),
            'absolute_two_sided_bias_failures':absolute_bias_failures,'signed_lower_bias_failures':signed_lower_bias_failures,
            'max_cell_coverage_failures':int(sm.coverage_failures.max()),'paired_comparisons':comparisons,
            'family_floor':family,'hashes':hashes,'original_helper_sha256':original_helper_sha256,'elapsed_seconds':time.time()-started,
            'independence':'No producer or producer-helper imports. Generic comparisons; rankdata training; linprog validation LPs; enumerated population law.'}
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'results/reference_certificate_efficiency')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.study,args.output)
