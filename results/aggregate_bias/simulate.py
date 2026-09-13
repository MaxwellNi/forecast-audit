"""Matched validation-only simulation for known category masses.

This new random-stream experiment tests the proposed finite bias bound. It is
method-development evidence, not a new public-domain forecasting confirmation.
"""
from pathlib import Path
import sys,json,hashlib,datetime,itertools,gzip,io,argparse
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
REPS=2000
SEED=1309202601

def categorical_counts(n,p,rng,reps):
    left=np.full(reps,n,dtype=np.int64);out=[];remaining=1.
    for q in p[:-1]:
        d=rng.binomial(left,float(np.clip(q/remaining,0,1)));out.append(d);left-=d;remaining-=q
    out.append(left)
    return np.stack(out,axis=1)

def moments(n,p,q,reference,rng):
    reps=len(n);c=len(p)
    sx=np.zeros((reps,c));sy=sx.copy()
    outcomes=list(itertools.product([0,1],repeat=4))
    hx=np.array([float(x0>x1)+.5*(x0==x1) for x0,y0,x1,y1 in outcomes])
    hy=np.array([float(y0>y1)+.5*(y0==y1) for x0,y0,x1,y1 in outcomes])
    for k in range(c):
        weights=np.array([(q[k] if x0 else 1-q[k])*(q[k] if y0 else 1-q[k])*reference[x1,y1] for x0,y0,x1,y1 in outcomes]);weights/=weights.sum()
        left=n[:,k].copy();remaining=1.
        for i,w in enumerate(weights):
            d=left if i==len(weights)-1 else rng.binomial(left,float(np.clip(w/remaining,0,1)))
            sx[:,k]+=d*hx[i];sy[:,k]+=d*hy[i]
            left=left-d;remaining-=w
    return sx,sy

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output;assert not out.exists();out.mkdir(parents=True)
    protocol={'scope':'New fixed random-stream validation-budget experiment; no forecast-gating or temporal generalization claim','seed':SEED,'repetitions_per_cell':REPS,'categories':[2,8,32,128],'validation_pairs':[512,2048,8192],'mass_laws':['balanced','one_heavy'],'fit_errors':['exact','same_direction','opposite_direction'],'delta':.05,'sampling':'two independent half streams of independent focal/reference pairs; two raw channels conditionally independent given class; common marginal references','methods':['known_mass_full_rectangle','known_mass_aggregate_product'],'prior_exposure':'Settings designed after prior archive diagnostics; no result-dependent grid or seed selection','measure':'Bias-upper coverage and slack, no claim of discovery power','comparison':'Both methods use same known category probabilities and 2*m raw calls. Full rectangle uses both channels from all m pairs; aggregate uses X on first half and Y on second.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    source=Path(__file__);(out/'execution_seal.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
    rows=[]
    for C,m,mass,fit in itertools.product(protocol['categories'],protocol['validation_pairs'],protocol['mass_laws'],protocol['fit_errors']):
        rng=np.random.default_rng(np.random.SeedSequence([SEED,C,m,int(mass=='one_heavy'),protocol['fit_errors'].index(fit)]))
        p=np.full(C,1/C) if mass=='balanced' else np.r_[.7,np.full(C-1,.3/(C-1))]
        t=np.linspace(-1,1,C);t=t-p@t;q=.5+.35*t/np.max(np.abs(t))
        mu=.25+.5*q
        f=mu.copy();g=mu.copy()
        if fit=='same_direction':f+=.05;g+=.05
        elif fit=='opposite_direction':f+=.15;g-=.15
        b=float(p@((mu-f)*(mu-g)))
        reference=np.array([[p@((1-q)**2),p@((1-q)*q)],[p@(q*(1-q)),p@(q*q)]])
        nA=categorical_counts(m//2,p,rng,REPS);nB=categorical_counts(m//2,p,rng,REPS)
        ax,ay=moments(nA,p,q,reference,rng);bx,by=moments(nB,p,q,reference,rng)
        n=nA+nB
        mx=np.divide(ax+bx,n,out=np.full_like(ax,.5),where=n>0);my=np.divide(ay+by,n,out=np.full_like(ay,.5),where=n>0)
        rad=np.sqrt(np.log(4*C/.05)/(2*np.maximum(n,1)))
        lx=np.where(n>0,np.maximum(0,mx-rad),0);ux=np.where(n>0,np.minimum(1,mx+rad),1)
        ly=np.where(n>0,np.maximum(0,my-rad),0);uy=np.where(n>0,np.minimum(1,my+rad),1)
        rect=np.maximum.reduce([(lx-f)*(ly-g),(lx-f)*(uy-g),(ux-f)*(ly-g),(ux-f)*(uy-g)])@p
        aa=np.divide(ax,nA,out=np.zeros_like(ax),where=nA>0)-f
        bb=np.divide(by,nB,out=np.zeros_like(by),where=nB>0)-g
        valid=(nA>0)&(nB>0);pp=p*valid;xlog=np.log(3/.05)
        center=np.sum(pp*aa*bb,axis=1)
        la=np.sqrt(xlog/2*np.sum(pp**2*bb**2/np.maximum(nA,1),axis=1));lb=np.sqrt(xlog/2*np.sum(pp**2*aa**2/np.maximum(nB,1),axis=1))
        d=pp/(4*np.sqrt(np.maximum(nA,1)*np.maximum(nB,1)))
        bil=np.sqrt(2*xlog*np.sum(d*d,axis=1))+xlog*np.max(d,axis=1)
        absent=np.sum(p*(~valid)*np.maximum(f*g,(1-f)*(1-g)),axis=1)
        agg=center+la+lb+bil+absent
        for rep in range(REPS):
            rows.append(dict(categories=C,validation_pairs=m,mass=mass,fit_error=fit,replicate=rep,true_bias=b,rectangle_upper=rect[rep],aggregate_upper=agg[rep],center=center[rep],linear_x_radius=la[rep],linear_y_radius=lb[rep],product_radius=bil[rep],absent_allowance=absent[rep]))
    data=pd.DataFrame(rows)
    with gzip.GzipFile(filename='',mode='wb',fileobj=(out/'replications.csv.gz').open('wb'),mtime=0) as stream:stream.write(data.to_csv(index=False).encode())
    summaries=[]
    for keys,d in data.groupby(['categories','validation_pairs','mass','fit_error'],sort=False):
        for method in ['rectangle','aggregate']:
            u=d[method+'_upper'];summaries.append(dict(zip(['categories','validation_pairs','mass','fit_error'],keys),method=method,repetitions=len(d),noncoverage=int((u+1e-14<d.true_bias).sum()),median_slack=float((u-d.true_bias).median()),mean_slack=float((u-d.true_bias).mean()),negative_upper=int((u<0).sum())))
    summary=pd.DataFrame(summaries);summary.to_csv(out/'summary.csv',index=False)
    print(summary[(summary.categories==32)&(summary.validation_pairs==8192)].to_string(index=False),flush=True)
    print('Rows',len(data),'cells',len(summary),'maximum noncoverage',summary.noncoverage.max(),flush=True)
    receipt={'scope':protocol['scope'],'rows':len(data),'cells':len(summary),'maximum_observed_noncoverage':int(summary.noncoverage.max()),'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
