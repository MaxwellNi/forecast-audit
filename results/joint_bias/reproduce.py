"""Post-exposure joint-bias diagnosis of all 32 corrected frozen candidates.

Only the existing training and validation draws construct the new allowance.
No new draw, fitting, category, hyperparameter, selection rule or outcome is
chosen. Existing full-U means/radii are retained; disjoint triples and raw
p-values are reconstructed. Census ranks are computed only after allowances,
for labeled diagnostics. This script never writes to its study directory.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import beta,rankdata

from joint_bias import joint_bias_upper,verify_certificate
from integrity import check_manifest

CELLS=32
CANDIDATES=['persistence','seasonal_day','seasonal_week','ridge_full',
            'hist_gradient_boosting','extra_trees','category_copy','independent_noise']
DELTA=.1*.05/(8*sum(1/k for k in range(1,9)))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def means(cat,values):
    n=np.bincount(cat,minlength=CELLS)
    return np.divide(np.bincount(cat,weights=values,minlength=CELLS),n,
                     out=np.full(CELLS,.5),where=n>0)


def triples(x,y,f,g):
    x,y,f,g=[a.reshape(-1,3) for a in [x,y,f,g]]
    answer=np.zeros(len(x))
    compare=lambda a,b:(a>b).astype(float)+.5*(a==b)
    for i in range(3):
        j,k=[z for z in range(3) if z!=i]
        answer+=((compare(x[:,i],x[:,j])-f[:,i])*(compare(y[:,i],y[:,k])-g[:,i])+
                 (compare(x[:,i],x[:,k])-f[:,i])*(compare(y[:,i],y[:,j])-g[:,i]))/6
    return answer


def validation(x,y,cat,train,val):
    f=means(cat[train],(rankdata(x[train])-1)/(len(train)-1))
    g=means(cat[train],(rankdata(y[train])-1)/(len(train)-1))
    cc=cat[val[:,0]];n=np.bincount(cc,minlength=CELLS)
    compare=lambda v:(v[val[:,0]]>v[val[:,1]]).astype(float)+.5*(v[val[:,0]]==v[val[:,1]])
    mx,my=means(cc,compare(x)),means(cc,compare(y))
    r=np.full(CELLS,np.inf);r[n>0]=np.sqrt(np.log(8*CELLS/DELTA)/(2*n[n>0]))
    lx,hx=np.maximum(0,mx-r),np.minimum(1,mx+r)
    ly,hy=np.maximum(0,my-r),np.minimum(1,my+r)
    caps=np.ones(CELLS);active=n<n.sum()
    caps[active]=beta.isf(DELTA/(2*CELLS),n[active]+1,n.sum()-n[active])
    return f,g,lx,hx,ly,hy,caps,n


def variance_p(mean,h,width,bias):
    j=len(h);s2=float(h.var(ddof=1));a=np.sqrt(2*s2/j)
    c=2*width/np.sqrt(j*(j-1))+width/(3*j)
    margin=max(mean-bias,0.)
    root=2*margin/(a+np.sqrt(a*a+4*c*margin)) if margin else 0.
    exponent=max(root*root,2*j*(margin/width)**2)
    return float(min(1.,DELTA+2*np.exp(-exponent)))


def betting_p(h,lo,hi,bias):
    if bias>=hi:return 1.
    if bias<lo:raise ValueError('Unexpected allowance below fitted kernel support')
    if bias==lo:return 1. if np.all(h==lo) else float(DELTA)
    threshold=(bias-lo)/(hi-lo);scaled=(h-lo)/(hi-lo)
    capitals=[np.log1p(q*(scaled/threshold-1)).sum() for q in np.geomspace(1e-4,.99,64)]
    loge=float(logsumexp(capitals)-np.log(64))
    return 1. if loge<=0 else float(min(1.,DELTA+np.exp(-loge)))


def by(p):
    p=np.asarray(p);order=np.argsort(p,kind='stable');out=np.ones(len(p))
    scaled=p[order]*len(p)*sum(1/k for k in range(1,len(p)+1))/np.arange(1,len(p)+1)
    out[order]=np.minimum(1,np.minimum.accumulate(scaled[::-1])[::-1])
    return out


def run(study,output):
    assert not output.exists(),'Refuse to overwrite an earlier diagnostic'
    check_manifest()
    sourcefiles=[study/'selection_certificates.csv']
    sourcefiles += [study/task/name for task in ['appliances','metro']
                    for name in ['forecast_archive.npz','sampling_indices.npz']]
    before={str(p.relative_to(study)):digest(p) for p in sourcefiles}
    output.mkdir(parents=True);(output/'certificates').mkdir()
    protocol=dict(started_utc=datetime.now(timezone.utc).isoformat(),
        classification='POST-EXPOSURE EXPLORATION; not confirmation; all 32 candidates retained',
        construction='Frozen before new joint-LP outcomes; exact same old rectangles, mass upper caps, fits and delta',
        public_manifest_sha256=digest(Path(__file__).parent/'MANIFEST.json'),
        study_sha256=before,cells=CELLS,delta=DELTA,
        training_rows=4096,validation_pairs=8192,evaluation_rows=12288,total_query_calls=32768,
        evaluation='Recorded frozen full-U mean/radius; independently rebuilt disjoint triples, range and p inversion',
        census='Used only after allowance calculation for target and true-bias diagnostics',
        unchanged='No refits, new draws, category changes, evaluation tuning or confirmation-outcome remapping')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    old=pd.read_csv(study/'selection_certificates.csv',float_precision='round_trip')
    rows=[];max_old_bias_error=0.;max_old_p_error=0.;begun=time.perf_counter()
    for task in ['appliances','metro']:
        data=np.load(study/task/'forecast_archive.npz',allow_pickle=False)
        draws=np.load(study/task/'sampling_indices.npz',allow_pickle=False)
        train,val,ev=[draws[key] for key in ['training','validation','evaluation']]
        assert len(train)==4096 and val.shape==(8192,2) and len(ev)==12288
        mask=data['selection'];y=data['y'][mask]
        for baseline in ['seasonal_day','ridge']:
            cat=data[f'{baseline}__category'][mask]
            for candidate in CANDIDATES:
                key=dict(task=task,baseline=baseline,candidate=candidate)
                source=old[(old.task==task)&(old.baseline==baseline)&(old.candidate==candidate)]
                ru=source[source.method=='reference_u'].iloc[0]
                rb=source[source.method=='reference_betting'].iloc[0]
                x=data[f'{baseline}__{candidate}'][mask]
                f,g,lx,hx,ly,hy,caps,n=validation(x,y,cat,train,val)
                result=joint_bias_upper(f,g,lx,hx,ly,hy,caps)
                assert verify_certificate(result)
                certificate_name=f'{task}__{baseline}__{candidate}.json'
                (output/'certificates'/certificate_name).write_text(json.dumps(result,indent=2)+'\n')
                sep,joint=result['separable_upper'],result['bias_upper']
                max_old_bias_error=max(max_old_bias_error,abs(sep-float(ru.bias)))
                assert abs(sep-float(ru.bias))<2e-12
                h=triples(x[ev],y[ev],f[cat[ev]],g[cat[ev]])
                corners=np.array([f*g,-f*(1-g),-(1-f)*g,(1-f)*(1-g)])
                lo,hi=float(corners.min()),float(corners.max());width=hi-lo
                assert abs(width-float(ru.width))<2e-14
                assert abs(h.var(ddof=1)-float(ru.variance))<2e-14
                assert abs(h.mean()-float(rb['mean']))<2e-14
                oldpu=variance_p(float(ru['mean']),h,width,sep)
                oldpb=betting_p(h,lo,hi,sep)
                max_old_p_error=max(max_old_p_error,abs(oldpu-float(ru.p)),abs(oldpb-float(rb.p)))
                assert abs(oldpu-float(ru.p))<2e-11 and abs(oldpb-float(rb.p))<2e-11
                pu=variance_p(float(ru['mean']),h,width,joint);pb=betting_p(h,lo,hi,joint)
                assert pu<=oldpu+2e-13 and pb<=oldpb+2e-13
                # Census information begins here, after both allowances and p-values.
                rx,ry=(rankdata(x)-.5)/len(x),(rankdata(y)-.5)/len(y)
                tx,ty=means(cat,rx),means(cat,ry)
                probability=np.bincount(cat,minlength=CELLS)/len(cat)
                bias=float(np.dot(probability,(tx-f)*(ty-g)))
                theta=float(np.mean((rx-tx[cat])*(ry-ty[cat])))
                rows.append(dict(**key,status=result['status'],old_bias=sep,joint_bias=joint,
                    improvement=sep-joint,relative_improvement=(sep-joint)/sep if sep else 0.,
                    true_bias_diagnostic=bias,exact_target_diagnostic=theta,
                    validation_event_diagnostic=bool(np.all(probability<=caps)&
                        np.all(tx[n>0]>=lx[n>0])&np.all(tx[n>0]<=hx[n>0])&
                        np.all(ty[n>0]>=ly[n>0])&np.all(ty[n>0]<=hy[n>0])),
                    full_u_mean=float(ru['mean']),radius=float(ru.radius),
                    old_lower=float(ru['mean'])-sep-float(ru.radius),
                    joint_lower=float(ru['mean'])-joint-float(ru.radius),
                    old_variance_p=oldpu,joint_variance_p=pu,
                    old_betting_p=oldpb,joint_betting_p=pb,
                    dual_max_absolute_residual=float(result['audit']['max_absolute_residual']) if result['audit'] else None,
                    certificate=certificate_name))
    frame=pd.DataFrame(rows)
    for _,index in frame.groupby(['task','baseline'],sort=False).groups.items():
        for name in ['old_variance','joint_variance','old_betting','joint_betting']:
            frame.loc[index,name+'_by']=by(frame.loc[index,name+'_p'].values)
    frame.to_csv(output/'all_candidates.csv',index=False)
    assert len(frame)==32 and not frame[['task','baseline','candidate']].duplicated().any()
    for p in sourcefiles:assert digest(p)==before[str(p.relative_to(study))],('Input changed',p)
    summary=dict(status='PASS',classification=protocol['classification'],rows=len(frame),
        old_bias_max_error=max_old_bias_error,old_p_max_error=max_old_p_error,
        max_improvement=float(frame.improvement.max()),median_improvement=float(frame.improvement.median()),
        strict_improvements_above_1e_10=int((frame.improvement>1e-10).sum()),
        median_relative_improvement=float(frame.relative_improvement.median()),
        maximum_relative_improvement=float(frame.relative_improvement.max()),
        statuses=frame.status.value_counts().to_dict(),
        all_observed_validation_events=bool(frame.validation_event_diagnostic.all()),
        all_32_receipts_verified=True,source_inputs_unchanged=True,
        variance_rejections=int((frame.joint_variance_by<=.05).sum()),
        betting_rejections=int((frame.joint_betting_by<=.05).sum()),
        elapsed_seconds=time.perf_counter()-begun)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,default=Path(__file__).resolve().parent/'inputs');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.inputs.resolve(),args.output.resolve())
