"""Independent reconstruction of the corrected public confirmation arithmetic.

Imports only independent mathematical helpers, not execution or gate producers.
"""
from pathlib import Path
from itertools import product
import argparse,hashlib,io,json,math,zipfile
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE=Path(__file__).resolve().parent
SRC=HERE
OUT=None
import independent_betting as helper
ind=helper.ind
METHODS=['reference_u','reference_betting','known_forecast_u','known_forecast_pairs','loss_gain']
CANDIDATES=['persistence','seasonal_day','seasonal_week','ridge_full','hist_gradient_boosting','extra_trees','category_copy','independent_noise']
DELTA=.1*.05/(8*sum(1/k for k in range(1,9)))


def centered_ranks(x,z):
    # Integer arithmetic is local to each category, independent of producer
    # np.add.at implementation. Half-integer average ranks are exact here.
    q=np.rint(2*rankdata(x)-1).astype(np.int64)
    a=np.empty(len(x))
    for c in np.unique(z):
        use=z==c;n=int(use.sum())
        a[use]=(n*q[use]-sum(map(int,q[use])))/(2*len(x)*n)
    return a


def kendall_sum(x,y):
    # Generic exact concordant-minus-discordant count with ties; Fenwick tree.
    _,rank=np.unique(y,return_inverse=True);rank=rank+1
    tree=np.zeros(rank.max()+1,dtype=np.int64)
    def prefix(k):
        total=0
        while k:
            total+=int(tree[k]);k-=k&-k
        return total
    order=np.argsort(x,kind='stable');s=0;seen=0;start=0
    while start<len(x):
        end=start+1
        while end<len(x) and x[order[end]]==x[order[start]]:end+=1
        for i in order[start:end]:s+=prefix(rank[i]-1)-(seen-prefix(rank[i]))
        for i in order[start:end]:
            k=int(rank[i])
            while k<len(tree):tree[k]+=1;k+=k&-k
        seen+=end-start;start=end
    return s


def order_three(x,y,f,g):
    n=len(x);sx=rankdata(x)-1;sy=rankdata(y)-1
    # Sum of same-reference diagonal products is (Kendall S + number of pairs)/2.
    joint_sum=(kendall_sum(x,y)+n*(n-1)/2)/2
    return float((np.dot(sx,sy)-joint_sum)/(n*(n-1)*(n-2))-
                 np.mean(f*sy+g*sx)/(n-1)+np.mean(f*g))


def classical(mean,h,width,full,delta=0,bias=0):
    if width==0:return dict(mean=0.,radius=0.,lower=0.,p=1.,variance=0.,width=0.)
    o=ind.inference(mean,float(h.var(ddof=1)),len(h),0,width,bias,
                    'variance' if full else 'independent_bernstein',delta=delta)
    return dict(mean=mean,radius=float(o['radius']),lower=float(o['lower_bound']),p=float(o['p']),
                variance=float(h.var(ddof=1)),width=width)


def by(p):
    p=np.asarray(p);order=np.argsort(p,kind='stable');q=np.ones(len(p));running=1.
    for j in range(len(p)-1,-1,-1):
        running=min(running,p[order[j]]*len(p)*sum(1/k for k in range(1,len(p)+1))/(j+1))
        q[order[j]]=running
    return q


def loss(y,p,s):return ((np.clip(y,0,s)-np.clip(p,0,s))/s)**2


def gate(x,y,z,base,blend,scale,indices):
    tr,va,ev,allrows=indices
    f,g=ind.fitted_means(z[tr],x[tr],32),ind.fitted_means(z[tr],y[tr],32)
    b=ind.budget(f,g,z[va],x[va],y[va],DELTA)['signed_allowance']
    xx,yy,ff,gg=x[ev],y[ev],f[z[ev]],g[z[ev]]
    h=ind.blocks(*[v[None,:] for v in [xx,yy,ff,gg]])
    mean=order_three(xx,yy,ff,gg)
    corners=np.array([(a-f)*(b-g) for a,b in product([0.,1.],repeat=2)])
    lo,hi=corners.min(),corners.max();width=float(hi-lo)
    rows=[dict(method='reference_u',bias=b,**classical(mean,h,width,True,DELTA,b))]
    p,log_e=helper.betting(h,lo,hi,b,DELTA)
    rows.append(dict(method='reference_betting',mean=float(h.mean()),p=p,radius=np.nan,lower=np.nan,
                     variance=float(h.var(ddof=1)),width=width,bias=b))
    a=centered_ranks(x,z);aa,ys=a[allrows],y[allrows]
    hp=.25*(aa[::2]-aa[1::2])*np.sign(ys[::2]-ys[1::2])
    mu=float(np.dot(aa,rankdata(ys)-(len(ys)+1)/2)/(len(ys)*(len(ys)-1)))
    for method,center,full in [('known_forecast_u',mu,True),('known_forecast_pairs',float(hp.mean()),False)]:
        rows.append(dict(method=method,bias=0.,**classical(center,hp,float(np.ptp(a)/2),full)))
    gain=(loss(y,base,scale)-loss(y,blend,scale))[allrows]
    rows.append(dict(method='loss_gain',bias=0.,**classical(float(gain.mean()),gain,2.,False)))
    ry=centered_ranks(y,z);exactrank=float(np.mean(a*ry));exactgain=float(np.mean(loss(y,base,scale)-loss(y,blend,scale)))
    for r in rows:
        r.update(exact_target=exactgain if r['method']=='loss_gain' else exactrank,query_calls=32768,
                 distinct_labels=len(np.unique(allrows)),archive_rows=len(x),delta=DELTA if r['method'].startswith('reference') else 0.)
    return rows


def source_check(task,data):
    with zipfile.ZipFile(SRC/'data'/f'{task}.zip') as zf:
        member=zf.namelist()[0];raw=pd.read_csv(io.BytesIO(zf.read(member)),compression='gzip' if member.endswith('.gz') else None)
    stamps=pd.to_datetime(raw['date' if task=='appliances' else 'date_time'])
    y=raw['Appliances' if task=='appliances' else 'traffic_volume']
    pairs=pd.DataFrame({'timestamp':stamps,'value':y}).groupby('timestamp').value.mean()
    if task=='appliances':
        hourly=pairs.groupby(pairs.index.floor('h')).agg(['sum','count'])
        expected=hourly['sum'].where(hourly['count']==6)
    else:expected=pairs
    index=pd.DatetimeIndex(data['time'])
    np.testing.assert_equal(expected.reindex(index).to_numpy(),data['y'])
    s=data['y'];scale=max(1.,float(np.quantile(s[data['train']],.99)))
    assert scale==float(data['scale'])
    # Direct array shifts check the three forecast controls and their fallback.
    median=float(np.nanmedian(s[data['train']]))
    lag1=np.r_[np.nan,s[:-1]]
    for name,lag in [('persistence',1),('seasonal_day',24),('seasonal_week',168)]:
        prediction=np.r_[np.full(lag,np.nan),s[:-lag]]
        prediction=np.where(np.isnan(prediction),lag1,prediction)
        prediction=np.where(np.isnan(prediction),median,prediction)
        np.testing.assert_equal(np.clip(prediction,0,scale),data[name])
    for baseline in ['seasonal_day','ridge']:
        breaks=np.quantile(data[baseline][data['train']],np.arange(1,16)/16)
        np.testing.assert_equal(breaks,data[baseline+'__breaks'])
        cells=2*np.searchsorted(breaks,data[baseline],side='left')+(index.dayofweek>=5)
        np.testing.assert_equal(cells,data[baseline+'__category'])
        np.testing.assert_equal(scale*(cells+.5)/32,data[baseline+'__category_copy'])
    return dict(raw_rows=len(raw),calendar_rows=len(index),scale=scale,
                split_rows={k:int(data[k].sum()) for k in ['train','calibration','selection','confirmation']})


def main():
    # Exhaustive generic test of the independent aggregate U expression.
    rng=np.random.default_rng(922);max_kernel=0.
    for n in [3,4,7,21]:
      for rep in range(12):
        x,y=rng.integers(0,4,(2,n));f,g=rng.uniform(0,1,(2,n))
        expected=float(ind.full_u(ind.rank_sums(x[None,:],y[None,:]),f[None,:],g[None,:],n)[0])
        max_kernel=max(max_kernel,abs(expected-order_three(x,y,f,g)))
    assert max_kernel<1e-13
    expected_rows=[];choices=[];results=[];bounds=[];weeks=[];candidates=[];sources={}
    for ti,task in enumerate(['appliances','metro']):
        data=np.load(SRC/task/'forecast_archive.npz');blends=pd.read_csv(SRC/task/'frozen_blends.csv')
        sources[task]=source_check(task,data)
        ids=np.load(SRC/task/'sampling_indices.npz')
        streams=[np.random.default_rng(s) for s in np.random.SeedSequence(2026091201+1000+ti).spawn(3)]
        n=int(data['selection'].sum())
        independent_indices=[streams[0].integers(n,size=4096),streams[1].integers(n,size=(8192,2)),streams[2].integers(n,size=12288)]
        independent_indices.append(np.concatenate([independent_indices[0],independent_indices[1].ravel(),independent_indices[2]]))
        for name,a in zip(['training','validation','evaluation','all_rows'],independent_indices):np.testing.assert_equal(a,ids[name])
        for baseline in ['seasonal_day','ridge']:
            mask=data['selection'];y=data['y'][mask];s=float(data['scale']);base=data[baseline][mask];z=data[baseline+'__category'][mask]
            local=[];mse={'baseline':float(loss(y,base,s).mean())};cmse=dict(mse)
            for name in CANDIDATES:
                b=blends[(blends.baseline==baseline)&(blends.candidate==name)].iloc[0]
                raw=data[baseline+'__'+name];x=raw[mask]
                # Independently reselect calibration weights, never selection outcomes.
                cal=data['calibration'];grid=np.linspace(0,2,41)
                cal_loss=loss(data['y'][cal,None],data[baseline][cal,None]+grid*(raw[cal,None]-data[baseline][cal,None]),s).mean(0)
                assert abs(grid[np.argmin(cal_loss)]-b.weight)<2e-15
                assert abs(grid[:21][np.argmin(cal_loss[:21])]-b.convex_weight)<2e-15
                rows=gate(x,y,z,base,base+b.weight*(x-base),s,independent_indices)
                for row in rows:row.update(task=task,baseline=baseline,candidate=name,weight=b.weight)
                local.extend(rows);mse[name]=float(loss(y,base+b.weight*(x-base),s).mean())
                cmse[name]=float(loss(y,base+b.convex_weight*(x-base),s).mean())
            frame=pd.DataFrame(local);selected={}
            for method in METHODS:
                use=frame.method==method;frame.loc[use,'p_by']=by(frame.loc[use,'p'])
                retained=frame.loc[use&(frame.p_by<=.05),'candidate'].tolist()
                chosen=min(['baseline']+retained,key=lambda k:mse[k]);selected[method]=chosen
                choices.append(dict(task=task,baseline=baseline,method=method,selected=chosen,selection_loss=mse[chosen],retained=';'.join(retained)))
            for method,values in [('ungated',mse),('convex',cmse),('baseline',{'baseline':mse['baseline']})]:
                chosen=min(values,key=values.get);selected[method]=chosen
                choices.append(dict(task=task,baseline=baseline,method=method,selected=chosen,selection_loss=values[chosen],retained=';'.join(CANDIDATES) if method!='baseline' else ''))
            expected_rows.extend(frame.to_dict('records'))
            mask=data['confirmation'];y=data['y'][mask];base=data[baseline][mask];times=pd.to_datetime(data['time'][mask])
            pred={'baseline':base};cpred={'baseline':base}
            for name in CANDIDATES:
                b=blends[(blends.baseline==baseline)&(blends.candidate==name)].iloc[0];x=data[baseline+'__'+name][mask]
                pred[name]=np.clip(base+b.weight*(x-base),0,s);cpred[name]=np.clip(base+b.convex_weight*(x-base),0,s)
                candidates.append(dict(task=task,baseline=baseline,candidate=name,weight=b.weight,bounded_mse=float(loss(y,pred[name],s).mean()),raw_mse=float(np.mean((y-pred[name])**2))))
            ug=pred[selected['ungated']];cv=cpred[selected['convex']]
            lbase,lug,lcv=[loss(y,p,s) for p in [base,ug,cv]]
            block=((times-times.min())/pd.Timedelta(days=7)).astype(int)
            for method,chosen in selected.items():
                p=(cpred if method=='convex' else pred)[chosen];lp=loss(y,p,s)
                results.append(dict(task=task,baseline=baseline,method=method,selected=chosen,confirmation_rows=len(y),clipping_scale=s,
                    bounded_mse=float(lp.mean()),raw_mse=float(np.mean((y-p)**2)),baseline_bounded_mse=float(lbase.mean()),
                    gain_vs_baseline=float(np.mean(lbase-lp)),gain_vs_ungated=float(np.mean(lug-lp)),gain_vs_convex=float(np.mean(lcv-lp)),
                    raw_mae=float(np.mean(abs(y-p))),bounded_mae=float(np.mean(abs(np.clip(y,0,s)-p)/s)),
                    raw_gain_vs_baseline=float(np.mean((y-base)**2-(y-p)**2)),raw_gain_vs_ungated=float(np.mean((y-ug)**2-(y-p)**2))))
                for k in np.unique(block):
                    use=block==k
                    weeks.append(dict(task=task,baseline=baseline,method=method,block=int(k),rows=int(use.sum()),first=str(times[use].min()),last=str(times[use].max()),
                        bounded_gain_vs_baseline=float(np.mean((lbase-lp)[use])),bounded_gain_vs_ungated=float(np.mean((lug-lp)[use])),bounded_gain_vs_convex=float(np.mean((lcv-lp)[use]))))
            if baseline=='ridge':
                primary=loss(y,pred[selected['reference_u']],s);t=1008 if task=='appliances' else 8760
                for name,other in [('ungated',lug),('convex',lcv)]:
                    center=float(np.sum(other-primary)/t);radius=math.sqrt(2*math.log(80)/t)
                    bounds.append(dict(task=task,baseline=baseline,primary='reference_u',comparison=name,nominal_hours=t,
                        observed_hours=len(y),availability_weighted_gain=center,alpha=.0125,radius=radius,
                        conditional_average_gain_lower=center-radius,positive_conditional_gain_certified=center-radius>0))
            print('Verified primitives',task,baseline,flush=True)
    datasets=[('selection_certificates.csv',expected_rows,['task','baseline','candidate','method']),
              ('frozen_selections.csv',choices,['task','baseline','method']),('confirmation_results.csv',results,['task','baseline','method']),
              ('confirmation_dependent_bounds.csv',bounds,['task','comparison']),('confirmation_weeks.csv',weeks,['task','baseline','method','block']),
              ('confirmation_all_candidates.csv',candidates,['task','baseline','candidate'])]
    comparisons={}
    for filename,rows,keys in datasets:
        expected=pd.DataFrame(rows).set_index(keys).sort_index();actual=pd.read_csv(SRC/filename,keep_default_na=False).set_index(keys).sort_index()
        assert expected.index.equals(actual.index)
        errors={}
        for c in expected:
            if expected[c].dtype.kind in 'fiu':
                actualcol=pd.to_numeric(actual[c],errors='coerce');e=float(np.nanmax(abs(expected[c]-actualcol))) if not expected[c].isna().all() else 0.
                errors[c]=e
                np.testing.assert_allclose(expected[c],actualcol,rtol=1e-10,atol=1e-12,equal_nan=True,err_msg=filename+':'+c)
            else:assert np.array_equal(expected[c].to_numpy(),actual[c].to_numpy()),(filename,c)
        comparisons[filename]=dict(rows=len(expected),max_errors=errors)
        expected.reset_index().to_csv(OUT/('independent_'+filename),index=False,float_format='%.17g')
    out=dict(status='PASS',scope='Independent primitive/inference/selection/confirmation arithmetic; no producer imports',
        generic_order_three_cases=48,generic_order_three_max_error=max_kernel,source_checks=sources,comparisons=comparisons,
        reviewed_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
             [HERE/'protocol.json',HERE/'gates.py',SRC/'selection_certificates.csv',SRC/'confirmation_results.csv']})
    (OUT/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=HERE,help='Bundled or independently reproduced tables and arrays')
    parser.add_argument('--output',type=Path,required=True,help='New directory for independently rebuilt tables and receipt')
    args=parser.parse_args()
    SRC=args.source.resolve();OUT=args.output.resolve()
    if OUT == HERE or HERE in OUT.parents or OUT == SRC:parser.error('Output must be separate from source and outside the bundle')
    OUT.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((HERE/'MANIFEST.json').read_text())
    for name,digest in manifest['sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest, ('Export input changed',name)
    main()
