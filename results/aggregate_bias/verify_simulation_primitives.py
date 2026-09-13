"""Independent validation of the aggregate-budget simulation.

Replays the published random stream's multinomial primitive law without
importing its implementation, verifies every summary, and checks selected
replicates against the callable theorem implementation. A second, presealed
seed generates raw focal/reference observations directly, without the compressed
16-state generator, to check stochastic-law implementation independently.
"""
from pathlib import Path
import datetime
import hashlib
import itertools
import json
import math
import argparse
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd

from aggregate_bias_bound import aggregate_upper_bound

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'simulation'
FITS=('exact','same_direction','opposite_direction')


def law(C,mass):
    p=np.full(C,1/C) if mass=='balanced' else np.r_[.7,np.full(C-1,.3/(C-1))]
    locations=np.linspace(-1,1,C);locations=locations-p@locations
    q=.5+.35*locations/np.max(np.abs(locations))
    assert abs(p@q-.5)<1e-14
    return p,q,.25+.5*q


def sample_multinomial_totals(total,probabilities,rng):
    left=total.copy();remaining=1.;counts=[]
    for k,w in enumerate(probabilities):
        draw=left.copy() if k==len(probabilities)-1 else rng.binomial(left,float(np.clip(w/remaining,0,1)))
        counts.append(draw);left-=draw;remaining-=w
    return np.stack(counts,axis=1)


def replay_moments(counts,p,q,rng):
    support=list(itertools.product((0,1),repeat=4))
    reference={(x,y):sum(p[c]*(q[c] if x else 1-q[c])*(q[c] if y else 1-q[c]) for c in range(len(p))) for x,y in itertools.product((0,1),repeat=2)}
    # Use NumPy reduction here to match the declared generator's floating-point
    # category probability table exactly, while checking its independent sum.
    table={(0,0):p@((1-q)**2),(0,1):p@((1-q)*q),(1,0):p@(q*(1-q)),(1,1):p@(q*q)}
    assert max(abs(reference[k]-table[k]) for k in table)<1e-14
    sums_x=np.zeros_like(counts,dtype=float);sums_y=np.zeros_like(sums_x)
    for c in range(len(p)):
        weights=np.asarray([(q[c] if x else 1-q[c])*(q[c] if y else 1-q[c])*table[u,v] for x,y,u,v in support]);weights/=weights.sum()
        draws=sample_multinomial_totals(counts[:,c],weights,rng)
        cmp_x=np.asarray([int(x>u)+.5*int(x==u) for x,y,u,v in support])
        cmp_y=np.asarray([int(y>v)+.5*int(y==v) for x,y,u,v in support])
        for i in range(16):
            sums_x[:,c]+=draws[:,i]*cmp_x[i];sums_y[:,c]+=draws[:,i]*cmp_y[i]
    return sums_x,sums_y


def fit_values(mu,fit):
    f=mu.copy();g=mu.copy()
    if fit=='same_direction':f+=.05;g+=.05
    if fit=='opposite_direction':f+=.15;g-=.15
    assert ((0<=f)&(f<=1)&(0<=g)&(g<=1)).all()
    return f,g


def scalar_bounds(p,f,g,na,nb,ax,ay,bx,by,delta=.05):
    mean=lambda s,n:np.divide(s,n,out=np.full(len(n),.5),where=n>0)
    a=mean(ax,na)-f;b=mean(by,nb)-g
    absent=np.maximum(f*g,(1-f)*(1-g))
    result=aggregate_upper_bound(p,a,b,na,nb,delta,absent_upper=absent)
    count=na+nb;mx=mean(ax+bx,count);my=mean(ay+by,count)
    rect=0.
    for c in range(len(p)):
        if count[c]==0:lo_x,hi_x,lo_y,hi_y=0.,1.,0.,1.
        else:
            radius=math.sqrt(math.log(4*len(p)/delta)/(2*count[c]))
            lo_x,hi_x=max(0,mx[c]-radius),min(1,mx[c]+radius)
            lo_y,hi_y=max(0,my[c]-radius),min(1,my[c]+radius)
        rect+=p[c]*max((u-f[c])*(v-g[c]) for u,v in itertools.product((lo_x,hi_x),(lo_y,hi_y)))
    return result,rect


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output;out.mkdir(parents=True,exist_ok=False)
    protocol={'scope':'Independent arithmetic and raw-observation-law audit; no power or new domain confirmation','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'second_seed':1309202699,'second_seed_categories':[2,32],'second_seed_pairs':[512,8192],'mass_laws':['balanced','one_heavy'],'fit_errors':list(FITS),'raw_repetitions_per_law':200,'selected_original_replicates':[0,37,499,1999],'delta':.05}
    protocol['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    seal=out/'audit_protocol.json'
    assert not seal.exists(),'Do not overwrite a sealed audit.'
    seal.write_text(json.dumps(protocol,indent=2)+'\n')
    data=pd.read_csv(SOURCE/'replications.csv.gz')
    summary=pd.read_csv(SOURCE/'summary.csv')
    component_error=0.;checks=0
    assert len(data)==144000 and len(summary)==144
    for keys,group in data.groupby(['categories','validation_pairs','mass','fit_error'],sort=False):
        C,m,mass,fit=keys;p,q,mu=law(C,mass);f,g=fit_values(mu,fit)
        bias=float(p@((mu-f)*(mu-g)))
        assert np.max(np.abs(group.true_bias-bias))<1e-14
        for method in ('rectangle','aggregate'):
            old=summary[(summary.categories==C)&(summary.validation_pairs==m)&(summary.mass==mass)&(summary.fit_error==fit)&(summary.method==method)].iloc[0]
            slack=group[method+'_upper']-group.true_bias
            assert old.repetitions==len(group)
            assert old.noncoverage==int((slack < -1e-14).sum())
            assert old.negative_upper==int((group[method+'_upper']<0).sum())
            assert math.isclose(old.mean_slack,float(slack.mean()),abs_tol=1e-13)
            assert math.isclose(old.median_slack,float(slack.median()),abs_tol=1e-13)
        rng=np.random.default_rng(np.random.SeedSequence([1309202601,C,m,int(mass=='one_heavy'),FITS.index(fit)]))
        na=sample_multinomial_totals(np.full(2000,m//2,dtype=np.int64),p,rng)
        nb=sample_multinomial_totals(np.full(2000,m//2,dtype=np.int64),p,rng)
        ax,ay=replay_moments(na,p,q,rng);bx,by=replay_moments(nb,p,q,rng)
        for rep in protocol['selected_original_replicates']:
            rebuilt,rect=scalar_bounds(p,f,g,na[rep],nb[rep],ax[rep],ay[rep],bx[rep],by[rep])
            old=group[group.replicate==rep].iloc[0]
            mapping={'upper':'aggregate_upper','center':'center','linear_a_radius':'linear_x_radius','linear_b_radius':'linear_y_radius','product_radius':'product_radius','absent_allowance':'absent_allowance'}
            for new_key,old_key in mapping.items():
                difference=abs(rebuilt[new_key]-old[old_key]);component_error=max(component_error,difference)
                assert difference<1e-12,(keys,rep,old_key,rebuilt[new_key],old[old_key])
            assert abs(rect-old.rectangle_upper)<1e-12
            checks+=1
    second=[]
    for C,m,mass in itertools.product(protocol['second_seed_categories'],protocol['second_seed_pairs'],protocol['mass_laws']):
        p,q,mu=law(C,mass)
        rng=np.random.default_rng(np.random.SeedSequence([protocol['second_seed'],C,m,int(mass=='one_heavy')]))
        for rep in range(protocol['raw_repetitions_per_law']):
            z=rng.choice(C,size=m,p=p);r=rng.choice(C,size=m,p=p)
            # Direct independent Bernoulli outcomes conditional on each raw
            # observation's class; no 16-state compressed sampling is used.
            xv=rng.random(m)<q[z];yv=rng.random(m)<q[z]
            xr=rng.random(m)<q[r];yr=rng.random(m)<q[r]
            cx=(xv>xr).astype(float)+.5*(xv==xr);cy=(yv>yr).astype(float)+.5*(yv==yr)
            half=m//2
            na=np.bincount(z[:half],minlength=C);nb=np.bincount(z[half:],minlength=C)
            ax=np.bincount(z[:half],weights=cx[:half],minlength=C);ay=np.bincount(z[:half],weights=cy[:half],minlength=C)
            bx=np.bincount(z[half:],weights=cx[half:],minlength=C);by=np.bincount(z[half:],weights=cy[half:],minlength=C)
            for fit in FITS:
                f,g=fit_values(mu,fit);result,rect=scalar_bounds(p,f,g,na,nb,ax,ay,bx,by)
                bias=float(p@((mu-f)*(mu-g)))
                second.append(dict(categories=C,validation_pairs=m,mass=mass,fit_error=fit,replicate=rep,true_bias=bias,aggregate_upper=result['upper'],rectangle_upper=rect))
    second=pd.DataFrame(second)
    second.to_csv(out/'second_seed_raw_rows.csv',index=False)
    second_summary=[]
    for keys,group in second.groupby(['categories','validation_pairs','mass','fit_error']):
        for method in ('aggregate','rectangle'):
            second_summary.append(dict(zip(['categories','validation_pairs','mass','fit_error'],keys),method=method,repetitions=len(group),noncoverage=int((group[method+'_upper']+1e-14<group.true_bias).sum()),median_slack=float((group[method+'_upper']-group.true_bias).median())))
    pd.DataFrame(second_summary).to_csv(out/'second_seed_summary.csv',index=False)
    report={'status':'PASS','original_rows_reaggregated':len(data),'original_summary_cells_checked':len(summary),'original_settings_with_primitive_replay':72,'original_primitive_replicates_checked':checks,'original_component_max_absolute_error':component_error,'second_seed':protocol['second_seed'],'second_seed_raw_pair_replicates':1600,'second_seed_fit_rows':len(second),'second_seed_method_cells':len(second_summary),'second_seed_max_noncoverage':max(r['noncoverage'] for r in second_summary),'scope':'Finite validation-bias coverage and width only. Positive power, future-panel validity and scientific novelty are not established by these tests.','generation_review':'Reference-channel dependence through common reference category is included. Focal channels are conditionally independent, overall Bernoulli marginals are exactly 1/2, and midrank category means are .25+.5q.','original_producer_imported':False}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
