"""Retrain from the bundled public ZIPs and independently verify all results.

Example: python reproduce.py --output /tmp/forecast-confirmation-reproduction
No network or historical model pickle is needed. All writes use the new output.
"""
from pathlib import Path
import argparse,hashlib,json,os,platform,shutil,subprocess,sys
import numpy as np
import pandas as pd
import scipy,sklearn

HERE=Path(__file__).resolve().parent
TABLES=['selection_certificates.csv','frozen_selections.csv','confirmation_results.csv',
        'confirmation_weeks.csv','confirmation_all_candidates.csv','confirmation_dependent_bounds.csv']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def compare_table(a,b):
    left,right=pd.read_csv(a),pd.read_csv(b)
    assert list(left)==list(right) and left.shape==right.shape,(a.name,'schema')
    numerical={};exact={}
    for c in left:
        if c=='five_gate_seconds':continue
        x,y=left[c].to_numpy(),right[c].to_numpy()
        if left[c].dtype.kind in 'fiu':
            same=bool(np.array_equal(x,y,equal_nan=True))
            error=float(np.max(np.abs(x-y),where=np.isfinite(x)&np.isfinite(y),initial=0.))
            np.testing.assert_allclose(x,y,rtol=1e-10,atol=1e-12,equal_nan=True,err_msg=a.name+':'+c)
            numerical[c]=error
        else:
            same=bool(left[c].fillna('<missing>').equals(right[c].fillna('<missing>')))
            assert same,(a.name,c)
        exact[c]=same
    return dict(rows=len(left),all_compared_columns_exact=all(exact.values()),
        exact_by_column=exact,max_absolute_errors=numerical,
        byte_equal=Path(a).read_bytes()==Path(b).read_bytes(),
        runtime_columns_excluded=['five_gate_seconds'] if 'five_gate_seconds' in left else [])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out==HERE or HERE in out.parents:parser.error('Choose a new output outside the evidence directory')
    manifest=json.loads((HERE/'MANIFEST.json').read_text())
    for name,digest in manifest['sha256'].items():assert sha(HERE/name)==digest,('Export input changed',name)
    out.mkdir(parents=True,exist_ok=False);(out/'data').mkdir()
    sources=json.loads((HERE/'PROVENANCE.json').read_text())['source_archives']
    for task,record in sources.items():
        src=HERE/'data'/(task+'.zip');assert sha(src)==record['sha256']
        shutil.copy2(src,out/'data'/src.name)
    (out/'download_receipt.json').write_text(json.dumps(dict(
        status='Bundled source copy for reproduction; no new data retrieval or pre-exposure claim',sources=sources),indent=2)+'\n')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for phase in ['prepare','select','confirm']:
        print('Running',phase,flush=True)
        with (out/(phase+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(HERE/'experiment.py'),phase,'--output',str(out)],
                env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
    arrays={}
    for task in ['appliances','metro']:
        arrays[task]={}
        for file in ['forecast_archive.npz','sampling_indices.npz']:
            a,b=np.load(HERE/task/file),np.load(out/task/file)
            assert a.files==b.files
            details={}
            for name in a.files:
                x,y=a[name],b[name];assert x.shape==y.shape and x.dtype==y.dtype
                exact=bool(np.array_equal(x,y,equal_nan=True))
                if x.dtype.kind=='f':
                    np.testing.assert_allclose(x,y,rtol=1e-10,atol=1e-12,equal_nan=True,err_msg=task+':'+name)
                    error=float(np.max(abs(x-y),where=np.isfinite(x)&np.isfinite(y),initial=0.))
                else:
                    assert exact,(task,name)
                    error=0.
                details[name]=dict(exact=exact,max_absolute_error=error)
            arrays[task][file]=details
        compare_table(HERE/task/'frozen_blends.csv',out/task/'frozen_blends.csv')
    comparisons={name:compare_table(HERE/name,out/name) for name in TABLES}
    print('Running independent full reconstruction',flush=True)
    with (out/'independent_verify.log').open('w') as log:
        subprocess.run([sys.executable,str(HERE/'verify.py'),'--source',str(out),'--output',str(out/'independent')],
            env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
    result=dict(status='PASS',scope='Source retraining, all archive arrays, all non-runtime scientific table fields, and independent primitive reconstruction',
        dependencies=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,scikit_learn=sklearn.__version__),
        numerical_tolerance=dict(rtol=1e-10,atol=1e-12),
        exact_equality_reported_separately=True,archive_arrays=arrays,tables=comparisons,
        all_archive_arrays_exact=all(v['exact'] for task in arrays.values() for archive in task.values() for v in archive.values()),
        all_compared_scientific_columns_exact=all(v['all_compared_columns_exact'] for v in comparisons.values()),
        export_manifest_sha256=sha(HERE/'MANIFEST.json'),
        independent_verification='independent/verification.json',
        model_compatibility='New models fitted in this environment; no historical pickle loaded.')
    (out/'reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['archive_arrays','tables']},indent=2))

if __name__=='__main__':main()
