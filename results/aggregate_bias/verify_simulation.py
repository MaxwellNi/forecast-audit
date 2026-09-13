"""Check every stored simulation row and summary without importing its generator."""
from pathlib import Path
import argparse, hashlib, itertools, json, math
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output;out.mkdir(parents=True,exist_ok=False)
    path=HERE/'simulation/replications.csv.gz';data=pd.read_csv(path,float_precision='round_trip')
    summary=pd.read_csv(HERE/'simulation/summary.csv',float_precision='round_trip')
    assert len(data)==144000 and len(summary)==144
    expected=set(itertools.product([2,8,32,128],[512,2048,8192],['balanced','one_heavy'],['exact','same_direction','opposite_direction']))
    rebuilt=[];maximum_component_error=0.
    for keys,group in data.groupby(['categories','validation_pairs','mass','fit_error'],sort=False):
        assert keys in expected;expected.remove(keys)
        C,m,mass,fit=keys
        p=np.full(C,1/C) if mass=='balanced' else np.r_[.7,np.full(C-1,.3/(C-1))]
        t=np.linspace(-1,1,C);t-=p@t;q=.5+.35*t/np.max(np.abs(t));mu=.25+.5*q
        f=mu.copy();g=mu.copy()
        if fit=='same_direction':f+=.05;g+=.05
        elif fit=='opposite_direction':f+=.15;g-=.15
        truth=float(p@((mu-f)*(mu-g)))
        assert len(group)==2000 and set(group.replicate)==set(range(2000))
        np.testing.assert_allclose(group.true_bias,truth,rtol=0,atol=2e-15)
        parts=group[['center','linear_x_radius','linear_y_radius','product_radius','absent_allowance']].sum(axis=1)
        error=float(np.max(np.abs(parts-group.aggregate_upper)));maximum_component_error=max(maximum_component_error,error)
        assert error<1e-13
        assert (group[['linear_x_radius','linear_y_radius','product_radius','absent_allowance']].to_numpy()>=0).all()
        for method in ['rectangle','aggregate']:
            up=group[method+'_upper'];slack=up-group.true_bias
            old=summary[(summary.categories==C)&(summary.validation_pairs==m)&(summary.mass==mass)&(summary.fit_error==fit)&(summary.method==method)]
            assert len(old)==1;old=old.iloc[0]
            row=dict(zip(['categories','validation_pairs','mass','fit_error'],keys),method=method,repetitions=len(group),noncoverage=int((slack<-1e-14).sum()),median_slack=float(slack.median()),mean_slack=float(slack.mean()),negative_upper=int((up<0).sum()))
            for name in ['repetitions','noncoverage','negative_upper']:assert row[name]==old[name]
            for name in ['median_slack','mean_slack']:assert abs(row[name]-old[name])<1e-13
            rebuilt.append(row)
    assert not expected
    pd.DataFrame(rebuilt).to_csv(out/'summary.csv',index=False)
    report=dict(status='PASS',rows=len(data),settings=72,method_summaries=len(rebuilt),maximum_component_error=maximum_component_error,maximum_observed_noncoverage=int(summary.noncoverage.max()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope='Complete stored-row arithmetic and summary reconstruction; not new primitive draws or a proof of coverage.')
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
