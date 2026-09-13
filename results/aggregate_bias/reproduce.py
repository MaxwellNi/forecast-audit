"""Reconstruct aggregate bias diagnostics and validate all simulation records."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent

def check_manifest():
    manifest=json.loads((HERE/'MANIFEST.json').read_text())
    for item in manifest['files']:
        p=HERE/item['path'];assert p.is_file(),item['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],item['path']
    for item in json.loads((HERE/'PROVENANCE.json').read_text())['upstream_inputs']:
        p=HERE/item['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],item['path']
    return len(manifest['files'])

def compare_numeric(actual,expected,keys):
    a=pd.read_csv(actual,float_precision='round_trip');b=pd.read_csv(expected,float_precision='round_trip')
    a=a.sort_values(keys).reset_index(drop=True);b=b.sort_values(keys).reset_index(drop=True)
    assert list(a.columns)==list(b.columns) and len(a)==len(b)
    details={};exact=True
    for col in a.columns:
        x,y=a[col],b[col]
        if pd.api.types.is_bool_dtype(x) or pd.api.types.is_bool_dtype(y):
            assert np.array_equal(x.to_numpy(),y.to_numpy()),col
        elif pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            np.testing.assert_allclose(x,y,rtol=2e-10,atol=2e-12,equal_nan=True,err_msg=col)
            finite=np.isfinite(x)&np.isfinite(y)
            details[col]=float(np.max(np.abs(x[finite]-y[finite]))) if finite.any() else 0.
            exact &= bool(np.array_equal(x,y,equal_nan=True))
        else:assert x.fillna('').equals(y.fillna('')),col
    return dict(rows=len(a),all_numeric_values_exact=exact,max_absolute_errors=details)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--regenerate-simulation',action='store_true',help='Regenerate every fixed-stream validation replication and compare all numeric values.')
    args=parser.parse_args();entries=check_manifest();args.output.mkdir(parents=True,exist_ok=False)
    commands=[('diagnose.py','archive'),('verify_archive.py','independent_archive'),('verify_simulation.py','simulation_arithmetic')]
    for script,folder in commands:
        subprocess.run([sys.executable,str(HERE/script),'--output',str(args.output/folder)],check=True)
    comparison=compare_numeric(args.output/'archive/all_candidates.csv',HERE/'archive/all_candidates.csv',['task','baseline','candidate','budget_method'])
    independent=json.loads((args.output/'independent_archive/receipt.json').read_text())
    simulation=json.loads((args.output/'simulation_arithmetic/verification.json').read_text())
    report=dict(status='PASS',manifest_entries=entries,archive_comparison=comparison,independent_archive=dict(status=independent['status'],candidate_count=independent['candidate_count'],result_rows=independent['result_rows'],all_U_and_betting_BY_decisions_match=independent['all_U_and_betting_BY_decisions_match']),simulation_arithmetic=simulation,full_simulation_regenerated=args.regenerate_simulation,scope='All exposed-archive candidates reconstructed and independently checked; stored simulation arithmetic checked. Original forecasting confirmation remains unchanged.')
    if args.regenerate_simulation:
        subprocess.run([sys.executable,str(HERE/'simulate.py'),'--output',str(args.output/'simulation_regenerated')],check=True)
        report['simulation_replications']=compare_numeric(args.output/'simulation_regenerated/replications.csv.gz',HERE/'simulation/replications.csv.gz',['categories','validation_pairs','mass','fit_error','replicate'])
        report['simulation_summaries']=compare_numeric(args.output/'simulation_regenerated/summary.csv',HERE/'simulation/summary.csv',['categories','validation_pairs','mass','fit_error','method'])
    (args.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
