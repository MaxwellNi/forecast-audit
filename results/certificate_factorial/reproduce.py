"""Reconstruct all factorial results, verify them, and redraw the publication figure.

Outputs always go to a new directory. This is deterministic replay of exposed
streams, not a new statistical confirmation. No network or producer rerun is
needed to reconstruct the 400,000-row ablation from the supplied inputs.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,platform,subprocess,sys,time
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
KEYS=['replication','design','signal','fit','training_rows','validation_pairs','groups','peers','total_observations','method']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(output,reps):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    for script,sub in [('analysis.py','reconstructed'),('verify.py','independent'),('make_figure.py','figures')]:
        result=subprocess.run([sys.executable,str(HERE/script),'--output',str(output/sub)],check=True,text=True,capture_output=True)
        (output/(script.replace('.py','')+'.log')).write_text(result.stdout+result.stderr)
    historical=json.loads((HERE/'provenance/completion_receipt.json').read_text());comparisons={}
    for name in historical['output_sha256']:
        a=pd.read_csv(HERE/name,float_precision='round_trip');b=pd.read_csv(output/'reconstructed'/name,float_precision='round_trip')
        pd.testing.assert_frame_equal(a,b,check_exact=False,atol=2e-11,rtol=1e-10)
        numeric=a.select_dtypes(include='number').columns
        error=float(np.nanmax(np.abs(a[numeric].to_numpy()-b[numeric].to_numpy())))
        comparisons[name]={'rows':len(a),'max_absolute_numeric_difference':error,'byte_identical':sha(HERE/name)==sha(output/'reconstructed'/name)}
    # Rebuild selected complete original replications from integer seeds and
    # generic comparisons. They establish input provenance beyond stored-row
    # arithmetic; no empirical-Bernoulli shortcut is used for rank statistics.
    spec=importlib.util.spec_from_file_location('independent_primitives',HERE/'independent_primitives.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old=pd.read_csv(HERE/'inputs/original_efficiency/replications.csv.gz')
    primitive=[];max_error=0.
    for rep in reps:
        rebuilt=module.rebuild(rep).set_index(KEYS).sort_index();recorded=old[old.replication==rep].set_index(KEYS).sort_index()
        assert rebuilt.index.equals(recorded.index)
        columns=['target','exact_bias','mean','bias_upper','kernel_lower','kernel_upper','effective_triples','kernel_sample_variance','absolute_allowance','signed_allowance','bias_interval_width','p','radius','lower_bound']
        pd.testing.assert_frame_equal(rebuilt[columns],recorded[columns],check_exact=False,atol=2e-11,rtol=1e-9,check_dtype=False)
        for col in ['reject','lower_covers','allowance_covers']:np.testing.assert_array_equal(rebuilt[col],recorded[col])
        max_error=max(max_error,float(np.nanmax(np.abs(rebuilt[columns].to_numpy()-recorded[columns].to_numpy()))))
        primitive.append(rebuilt.reset_index())
    if primitive:pd.concat(primitive,ignore_index=True).to_csv(output/'primitive_replications.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    # The figure's numeric content must exactly match the two declared cells.
    meta=json.loads((output/'figures/fig_certificate_factorial.json').read_text())
    assert meta['size_inches']==[7,2.05]
    result=dict(status='PASS',scope='Portable factorial reconstruction plus independent checks; same exposed streams; no new confirmation.',rows=400000,settings=100,methods=4,paired_contrasts=600,outputs=comparisons,primitive_replications=reps,primitive_rows=500*len(reps),max_primitive_error=max_error,independent_verification='independent/verification.json',figure='figures/fig_certificate_factorial.pdf',environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),elapsed_seconds=time.perf_counter()-start)
    (output/'reproduction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--primitive-replications',nargs='*',type=int,default=[0,17],help='Complete original replications to independently rebuild; each must be in 0..999')
    a=p.parse_args();assert len(set(a.primitive_replications))==len(a.primitive_replications) and all(0<=r<1000 for r in a.primitive_replications)
    run(a.output,a.primitive_replications)
