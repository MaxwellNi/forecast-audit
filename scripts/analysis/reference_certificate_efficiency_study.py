"""Paired reference bounds at identical total observation budgets."""
from pathlib import Path
BASE=Path(__file__).resolve().parent
import argparse,hashlib,json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from category_reference_certificate import sample,empirical_fit,distinct_scores
from reference_certificate_efficiency import signed_category_budget,triple_scores,one_sided_certificate

SEED=202609121457
DESIGNS={'two_balanced':(np.array([.5,.5]),np.array([.2,.8])),
         'eight_rare':(np.array([.005]+[.995/7]*7),np.linspace(.2,.8,8))}
SIGNALS=[-.125,0.,.125,.25,.5]
GROUPS=[25,100,400,1600]
TRAIN=8192
VALIDATION=[512,8192]


def one(rep):
 rows=[]
 for di,(design,(prob,success)) in enumerate(DESIGNS.items()):
  c=len(prob);mu=.5-.5*prob@success+.5*success
  for si,gamma in enumerate(SIGNALS):
   seed=[SEED,rep,di,si]
   z,v,w=sample(np.random.default_rng(np.random.SeedSequence(seed+[0])),(max(GROUPS),64),prob,success,gamma)
   tz,tv,tw=sample(np.random.default_rng(np.random.SeedSequence(seed+[1])),(TRAIN,),prob,success,gamma)
   vz,vv,vw=sample(np.random.default_rng(np.random.SeedSequence(seed+[2])),(max(VALIDATION),2),prob,success,gamma)
   fitv,fitw=empirical_fit(tz,tv,c),empirical_fit(tz,tw,c)
   target=.25*gamma*np.dot(prob,success*(1-success))
   fit_cases=['estimated']+(['opposed_shifts'] if design=='two_balanced' else [])
   for fit_case in fit_cases:
    f,g=(fitv,fitw) if fit_case=='estimated' else (np.clip(fitv+.15,0,1),np.clip(fitw-.15,0,1))
    gs=distinct_scores(v,w,f[z],g[z]);hs=triple_scores(v,w,f[z],g[z]);bias=float(prob@((mu-f)*(mu-g)))
    for validation in VALIDATION:
     if fit_case=='opposed_shifts' and validation!=8192:continue
     zz=vz[:validation,0];count=np.bincount(zz,minlength=c)
     av=.5+.5*(vv[:validation,0]-vv[:validation,1]);aw=.5+.5*(vw[:validation,0]-vw[:validation,1])
     mv=np.divide(np.bincount(zz,weights=av,minlength=c),count,out=np.zeros(c),where=count>0)
     mw=np.divide(np.bincount(zz,weights=aw,minlength=c),count,out=np.zeros(c),where=count>0)
     bud=signed_category_budget(f,g,count,mv,mw)
     for groups in GROUPS:
      h=hs[:groups*21];mean=float(gs[:groups].mean());J=groups*21
      setting=dict(replication=rep,design=design,signal=gamma,fit=fit_case,training_rows=TRAIN,validation_pairs=validation,
                   groups=groups,peers=64,total_observations=TRAIN+2*validation+groups*64,target=target,exact_bias=bias,
                   absolute_allowance=bud['absolute_bias_upper'],signed_allowance=bud['bias_upper'],bias_interval_width=bud['bias_upper']-bud['bias_lower'])
      for method,kind,b,stat in [('absolute_range','range',bud['absolute_bias_upper'],mean),
                                ('signed_range','range',bud['bias_upper'],mean),
                                ('signed_variance','variance',bud['bias_upper'],mean),
                                ('independent_triples','independent_bernstein',bud['bias_upper'],float(h.mean()))]:
       result=one_sided_certificate(stat,h,J,f,g,b,kind=kind)
       rows.append(dict(**setting,method=method,allowance_covers=bias<=b,lower_covers=result['lower_bound']<=target,**result))
      pv,pw,pz=[x[:groups].reshape(1,-1) for x in (v,w,z)]
      ph=triple_scores(pv,pw,f[pz],g[pz]);pm=float(distinct_scores(pv,pw,f[pz],g[pz])[0])
      result=one_sided_certificate(pm,ph,len(ph),f,g,bud['bias_upper'],kind='variance')
      rows.append(dict(**setting,method='pooled_variance',allowance_covers=bias<=bud['bias_upper'],lower_covers=result['lower_bound']<=target,**result))
 return rows


def run(out,reps,workers):
 out.mkdir(parents=True,exist_ok=False)
 protocol=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,replications=reps,designs=list(DESIGNS),signals=SIGNALS,
   groups=GROUPS,peers=64,training=TRAIN,validation_pairs=VALIDATION,alpha=.05,delta=.0001,
   methods=['absolute_range','signed_range','signed_variance','independent_triples','pooled_variance'],
   fixed_fits='Empirical category means; a separate two-category robustness study adds +.15/-.15 persistent shifts and clips to[0,1].',
   comparison='All methods share training, validation and evaluation primitive rows. Pooling is admissible under the common-law design and is a strong comparator, not assumed inferior.',
   scope='Synthetic efficiency and coverage study fixed before these outputs; motivated by previous visible certificate conservatism. Not independent real-world confirmation.',
   source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),BASE/'reference_certificate_efficiency.py',BASE/'category_reference_certificate.py']})
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 with ProcessPoolExecutor(max_workers=workers) as ex:
  batches=[]
  for i,batch in enumerate(ex.map(one,range(reps))):
   batches.extend(batch)
   if (i+1)%50==0:print('Replications',i+1,flush=True)
 df=pd.DataFrame(batches);df.to_csv(out/'replications.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 keys=['design','signal','fit','training_rows','validation_pairs','groups','peers','total_observations','method']
 summary=df.groupby(keys,dropna=False).agg(replications=('replication','size'),rejections=('reject','sum'),power=('reject','mean'),
   target=('target','first'),mean=('mean','mean'),bias=('exact_bias','mean'),bias_allowance=('bias_upper','mean'),
   absolute_allowance=('absolute_allowance','mean'),signed_allowance=('signed_allowance','mean'),bias_width=('bias_interval_width','mean'),
   radius=('radius','mean'),lower=('lower_bound','mean'),coverage_failures=('lower_covers',lambda x:int((~x).sum())),
   allowance_failures=('allowance_covers',lambda x:int((~x).sum()))).reset_index()
 summary.to_csv(out/'summary.csv',index=False)
 print('Completed',len(df),'rows',len(summary),'cells',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--replications',type=int,default=1000);p.add_argument('--workers',type=int,default=8);a=p.parse_args();run(a.output,a.replications,a.workers)
