"""Independent aggregate checks; no source-data access or nuisance refitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
TERMS=['outcome_fit_shift','forecast_fit_shift','joint_fit_shift']
LABELS=['complementary_product','directional_product']+TERMS


def main(directory=HERE, write_result=False):
    global HERE
    HERE = Path(directory).resolve()
    receipt=json.loads((HERE/'receipt.json').read_text())
    for name,digest in receipt['files_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    protocol=json.loads((HERE/'protocol.json').read_text())
    original=json.loads((HERE/'ORIGINAL_RUN_HASHES.json').read_text())
    sha=lambda name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    assert sha('protocol.json')==receipt['protocol_sha256']
    assert sha('decompose.py')==receipt['script_sha256']==protocol['script_sha256']
    assert protocol['input_sha256']==receipt['input_sha256']
    assert receipt['public_packaging']['original_run_hashes']=='ORIGINAL_RUN_HASHES.json'
    preserved=sorted(p.name for p in HERE.glob('*.csv'))+['verification.json']
    for name in preserved:
        assert sha(name)==original['files'][name]['sha256']
    source_cov=pd.read_csv(HERE/'cross_resolution_term_hac_covariance.csv')
    summary=pd.read_csv(HERE/'summary.csv')
    resolution=pd.read_csv(HERE/'per_resolution.csv')
    weekly=pd.read_csv(HERE/'weekly_aggregates.csv')
    cov=pd.read_csv(HERE/'combined_hac_covariance.csv')
    checked=[]
    for row in summary.itertuples():
        model=row.model
        r=resolution[resolution.model==model]
        assert len(r)==5 and np.allclose(r[TERMS].sum(axis=1),r.mean_change,atol=1e-14)
        for term in TERMS+['mean_complementary','mean_directional','mean_change']:
            assert np.isclose(np.dot(r.weight,r[term]),getattr(row,term),atol=1e-14,rtol=1e-12)
        assert np.isclose(row.mean_directional-row.mean_complementary,sum(getattr(row,t) for t in TERMS),atol=1e-14)
        w=weekly[weekly.model==model].sort_values('period')
        assert len(w)==17 and np.all(w.rows==4000)
        assert np.allclose(w[LABELS].mean(),[row.mean_complementary,row.mean_directional]+[getattr(row,t) for t in TERMS],atol=1e-14,rtol=1e-12)
        z=w[LABELS].to_numpy()
        assert np.allclose(z[:,1]-z[:,0],z[:,2:].sum(axis=1),atol=1e-14)
        centered=z-z.mean(axis=0)
        # Independent literal lag2 Bartlett covariance of17 equally sized week means.
        matrix=centered.T@centered
        for lag in (1,2):
            cross=centered[lag:].T@centered[:-lag]
            matrix+=(1-lag/3)*(cross+cross.T)
        matrix/=17**2
        for cell in cov[cov.model==model].itertuples():
            assert np.isclose(matrix[LABELS.index(cell.left),LABELS.index(cell.right)],cell.hac_covariance,rtol=1e-10,atol=1e-17)
        cells=source_cov[source_cov.model==model]
        assert len(cells)==225
        q=list(protocol['resolution_ladder'])
        full=np.empty((15,15))
        seen=set()
        for cell in cells.itertuples():
            left=3*q.index(cell.left_q)+TERMS.index(cell.left_term)
            right=3*q.index(cell.right_q)+TERMS.index(cell.right_term)
            assert (left,right) not in seen
            seen.add((left,right))
            full[left,right]=cell.hac_covariance
        assert np.all(np.isfinite(full)) and np.allclose(full,full.T,atol=1e-17,rtol=1e-10)
        mapping=np.kron(np.asarray(protocol['intercept_weights']).reshape(1,-1),np.eye(3))
        assert np.allclose(mapping@full@mapping.T,matrix[2:,2:],atol=1e-17,rtol=1e-10)
        assert np.isclose(np.sqrt(matrix[0,0]),row.se_complementary,rtol=1e-10)
        assert np.isclose(np.sqrt(matrix[1,1]),row.se_directional,rtol=1e-10)
        change_var=np.ones(3)@matrix[2:,2:]@np.ones(3)
        direct_var=matrix[0,0]+matrix[1,1]-2*matrix[0,1]
        assert np.isclose(change_var,direct_var,rtol=1e-10,atol=1e-17)
        assert np.isclose(np.sqrt(change_var),row.se_change_with_all_cross_covariances,rtol=1e-10)
        checked.append({'model':model,'five_resolution_identities':True,'original_weights_replay':True,
                        'seventeen_week_identities':True,'all25_combined_HAC_covariances':True,
                        'variance_of_sum_includes_cross_terms':True,
                        'all225_resolution_covariances_mapped_to_combined_terms':True})
    result={'passed':True,'source_data_accessed':False,
            'current_public_receipt_links_verified':True,
            'original_result_files_preserved':len(preserved),'checks':checked,
            'scope':'Independent aggregate arithmetic only; no causal, population or calibration conclusion.'}
    if write_result:
        (HERE/'aggregate_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=HERE)
    parser.add_argument('--write-result',action='store_true',help='Refresh aggregate_verification.json; default is read-only')
    args=parser.parse_args()
    main(args.directory,args.write_result)
