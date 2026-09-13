"""Post-exposure bias-budget attribution on all frozen forecast candidates."""
from pathlib import Path
import sys,json,hashlib,shutil,argparse
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.stats import rankdata
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'forecast_confirmation'
sys.path.insert(0,str(SOURCE))
from gates import means,by_values
from helpers.reference_certificate_efficiency import one_sided_certificate,triple_scores,family_validation_delta
from helpers.peer_rank_products import peer_rank_products
from helpers.classical_reference_comparators import mixture_betting_pvalue
from aggregate_bias_bound import aggregate_upper_bound

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output
    assert not out.exists(), 'Each diagnostic result directory is immutable'
    out.mkdir()
    old=pd.read_csv(SOURCE/'selection_certificates.csv')
    records=[]
    for task in ['appliances','metro']:
        archive=np.load(SOURCE/task/'forecast_archive.npz')
        idx=np.load(SOURCE/task/'sampling_indices.npz')
        train,validation,evaluation=[idx[k] for k in ['training','validation','evaluation']]
        mask=archive['selection'];y=archive['y'][mask]
        for baseline in ['ridge','seasonal_day']:
            z=archive[baseline+'__category'][mask].astype(int)
            p=np.bincount(z,minlength=32)/len(z)
            zt,zv,ze=z[train],z[validation[:,0]],z[evaluation]
            g=means(zt,(rankdata(y[train])-1)/(len(train)-1))
            comp=lambda a: (a[validation[:,0]]>a[validation[:,1]]).astype(float)+.5*(a[validation[:,0]]==a[validation[:,1]])
            yc=comp(y)
            delta=family_validation_delta(8)
            for candidate in ['persistence','seasonal_day','seasonal_week','ridge_full','hist_gradient_boosting','extra_trees','category_copy','independent_noise']:
                x=archive[baseline+'__'+candidate][mask]
                f=means(zt,(rankdata(x[train])-1)/(len(train)-1))
                xc=comp(x)
                nA=np.bincount(zv[:4096],minlength=32);nB=np.bincount(zv[4096:],minlength=32)
                a=means(zv[:4096],xc[:4096])-f;b=means(zv[4096:],yc[4096:])-g
                absent=np.maximum.reduce([f*g,-f*(1-g),-(1-f)*g,(1-f)*(1-g)])
                agg=aggregate_upper_bound(p,a,b,nA,nB,delta,absent_upper=absent)
                n=np.bincount(zv,minlength=32)
                radius=np.sqrt(np.log(4*32/delta)/(2*np.maximum(n,1)))
                mx,my=means(zv,xc),means(zv,yc)
                lx=np.where(n>0,np.maximum(0,mx-radius),0);ux=np.where(n>0,np.minimum(1,mx+radius),1)
                ly=np.where(n>0,np.maximum(0,my-radius),0);uy=np.where(n>0,np.minimum(1,my+radius),1)
                rect=float(p@np.maximum.reduce([(lx-f)*(ly-g),(lx-f)*(uy-g),(ux-f)*(ly-g),(ux-f)*(uy-g)]))
                rx,ry=(rankdata(x)-.5)/len(x),(rankdata(y)-.5)/len(y)
                oracle=float(p@((means(z,rx)-f)*(means(z,ry)-g)))
                matching=old[(old.task==task)&(old.baseline==baseline)&(old.candidate==candidate)&(old.method=='reference_u')].iloc[0]
                h=triple_scores(*[v[None,:] for v in (x[evaluation],y[evaluation],f[ze],g[ze])])
                mean=float(peer_rank_products(x[evaluation],y[evaluation],f[ze],g[ze])['corrected_residual_product'].mean())
                assert abs(mean-matching['mean'])<1e-12
                values={'original_rectangle':float(matching['bias']),'known_mass_rectangle':rect,'known_mass_aggregate':agg['upper'],'oracle_diagnostic':oracle}
                for method,B in values.items():
                    u=one_sided_certificate(mean,h,len(h),f,g,B,delta=delta)
                    bet=mixture_betting_pvalue(h,u['kernel_lower'],u['kernel_upper'],bias_upper=B,delta=delta)
                    row=dict(task=task,baseline=baseline,candidate=candidate,budget_method=method,bias=B,true_bias=oracle,mean=mean,U_radius=u['radius'],U_lower=u['lower_bound'],U_p=u['p'],betting_p=bet['p'],bias_covers=bool(B+1e-14>=oracle),queries=32768,known_category_metadata_rows=len(z))
                    if method=='known_mass_aggregate':row.update(agg)
                    records.append(row)
    frame=pd.DataFrame(records)
    for cols in ['U','betting']:
        frame[cols+'_BY']=frame.groupby(['task','baseline','budget_method'],sort=False)[cols+'_p'].transform(by_values)
        frame[cols+'_retained']=frame[cols+'_BY']<=.05
    frame.to_csv(out/'all_candidates.csv',index=False)
    table=frame[(frame.baseline=='ridge')&(frame.candidate=='extra_trees')]
    print(table[['task','budget_method','mean','bias','U_lower','U_p','U_BY','betting_p','betting_BY']].to_string(index=False))
    print(frame.groupby('budget_method')[['U_retained','betting_retained']].sum().to_string())
    receipt={'scope':'Post-exposure development diagnostic; no new independent confirmation; oracle rows are not deployable tests','records':len(frame),'candidates':32,'all_aggregate_biases_cover_true_archive_bias':bool(frame[frame.budget_method=='known_mass_aggregate'].bias_covers.all()),'aggregate_source_sha256':hashlib.sha256((HERE/'aggregate_bias_bound.py').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((HERE/'archive/protocol.json').read_bytes()).hexdigest(),'results_sha256':hashlib.sha256((out/'all_candidates.csv').read_bytes()).hexdigest()}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
