"""Independent replay of the frozen Gaussian benchmark, without producer imports.

Reconstructs the recorded primitive arrays, dense bin/spline prediction maps,
literal leave-cluster regressions, symmetric whitening and orthogonal residuals.
This is a deterministic verification replay, not a new experiment or calibration
claim for the real panels. The spline feature library is shared, while its ridge
prediction map is solved here directly rather than importing the study fitter.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.integrate import quad
from scipy.special import betainc, ndtr
from scipy.stats import beta, binomtest, chi2, nct, norm, t
from sklearn.preprocessing import SplineTransformer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY = ROOT / "results/gaussian_reference"
METHODS = ["coarse", "fine", "extrapolation_beta1", "extrapolation_beta2", "spline"]
SPECS = ["correct", "omitted_quadratic_mean", "omitted_dependence"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, tolerance, label):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)), initial=0))
    if not error <= tolerance:
        raise AssertionError(f"{label}: absolute difference {error} > {tolerance}")
    return error


def t_survival(value, degrees):
    value = np.asarray(value, float)
    positive_tail = 0.5 * betainc(degrees / 2, 0.5, degrees / (degrees + value * value))
    tail = np.where(value >= 0, positive_tail, 1 - positive_tail)
    # Near zero, degrees/(degrees+t*t) can round to one and lose
    # the entire difference from one half. Use the complementary
    # incomplete-beta identity with its small argument in that region.
    central = 0.5 - 0.5*np.sign(value)*betainc(0.5, degrees/2, value*value/(degrees+value*value))
    return np.where(np.abs(value) < 1, central, tail)


def wilson(success, size):
    z = norm.ppf(.975)
    denominator = size + z*z
    center = (success + z*z/2) / denominator
    half = z*np.sqrt(success*(size-success)/size + z*z/4) / denominator
    return [center-half, center+half]


def bin_map(z, folds, bins):
    prediction = np.zeros((len(z), len(z)))
    for fold in sorted(set(folds)):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        edges = np.quantile(z[train], np.arange(1, bins) / bins)
        train_bins = np.searchsorted(edges, z[train], side="right")
        test_bins = np.searchsorted(edges, z[test], side="right")
        for row, group in zip(test, test_bins):
            donors = train[train_bins == group]
            if not len(donors):
                donors = train
            prediction[row, donors] = 1 / len(donors)
    return prediction


def spline_map(z, folds):
    prediction = np.zeros((len(z), len(z)))
    for fold in sorted(set(folds)):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        features = SplineTransformer(n_knots=8, degree=3, knots="quantile",
                                     extrapolation="linear", include_bias=False)
        train_basis = features.fit_transform(z[train, None])
        test_basis = features.transform(z[test, None])
        center = train_basis.mean(axis=0)
        centered = train_basis - center
        ridge_map = np.linalg.solve(centered.T @ centered + .001*np.eye(centered.shape[1]), centered.T)
        prediction[np.ix_(test, train)] = 1/len(train) + (test_basis-center) @ ridge_map
    return prediction


def weights(exponent):
    u = np.asarray([8, 12, 16, 24, 32], float)**(-exponent)
    return (np.dot(u,u)-u*u.sum()) / (len(u)*np.dot(u,u)-u.sum()**2)


def leave_map(phi, clusters):
    size = len(phi)
    result = np.eye(size) / size
    for group in sorted(set(clusters)):
        held = np.flatnonzero(clusters == group)
        train = np.flatnonzero(clusters != group)
        if np.linalg.matrix_rank(phi[train]) != phi.shape[1]:
            raise ValueError("The independently reconstructed leave-cluster basis is unsupported")
        qr, upper = np.linalg.qr(phi[train], mode="reduced")
        prediction = phi[held] @ np.linalg.solve(upper, qr.T)
        result[np.ix_(held, train)] = -prediction / size
    return result


def geometry(phi, covariance):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    assert eigenvalues.min() > 0
    white = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    unwhite = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    q, _ = np.linalg.qr(white @ phi, mode="reduced")
    return white, unwhite, q


def direct_test(y, directions, geom):
    white, unwhite, basis = geom
    whitened = white @ y
    residual = whitened - basis @ (basis.T @ whitened)
    projected = unwhite @ directions
    projected -= basis @ (basis.T @ projected)
    length = np.linalg.norm(projected, axis=0)
    assert np.min(length) > 1e-9
    unit = projected / length
    numerator = np.sum(unit * residual, axis=0)
    orthogonal = residual - unit * numerator
    degrees = len(y) - basis.shape[1] - 1
    statistic = numerator / np.sqrt(np.sum(orthogonal*orthogonal, axis=0) / degrees)
    return statistic, t_survival(statistic, degrees), projected, degrees


def read_rows(path, repetitions):
    rows = {}
    count = 0
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["design"]), row["specification"], float(row["effect"]), row["method"])
            current = rows.setdefault(key, {"statistic": [], "pvalue": [], "reject": [], "cosine": []})
            assert int(row["replication"]) == len(current["pvalue"]), (key, row["replication"])
            current["statistic"].append(float(row["statistic"]))
            current["pvalue"].append(float(row["pvalue"]))
            current["reject"].append(int(row["reject"]))
            current["cosine"].append(float(row["direction_cosine"]) if row["direction_cosine"] else np.nan)
            count += 1
    for key, row in rows.items():
        assert len(row["pvalue"]) == repetitions
        for name in row:
            row[name] = np.asarray(row[name])
        assert np.isfinite(row["statistic"]).all() and np.isfinite(row["pvalue"]).all()
        assert ((row["pvalue"] >= 0) & (row["pvalue"] <= 1)).all()
        np.testing.assert_array_equal(row["reject"], row["pvalue"] <= .05)
    return rows, count


def run(study):
    start = time.perf_counter()
    protocol = json.loads((study / "protocol.json").read_text())
    runtime = {name: importlib.metadata.version(name) for name in protocol['dependencies']}
    if runtime != protocol['dependencies']:
        raise ValueError(f"Use the frozen numerical runtime: observed {runtime}, required {protocol['dependencies']}")
    summary = json.loads((study / "summary.json").read_text())
    assert summary["protocol_sha256"] == sha(study / "protocol.json")
    assert summary["raw_sha256"] == sha(study / "replications.csv.gz")
    assert summary["fixture_sha256"] == sha(study / "verification_fixtures.npz")
    assert summary["source_hashes"] == protocol["sources"]
    for name, digest in protocol["sources"].items():
        assert sha(ROOT / "scripts/analysis" / name) == digest
    rows, count = read_rows(study / "replications.csv.gz", protocol["replications"])
    expected_methods = {x+"_normal" for x in METHODS} | {x+"_gaussian" for x in METHODS+['leave_cluster_raw','leave_cluster_learned','gls']}
    assert len(rows) == 468 and count == 468000
    assert {key[3] for key in rows} == expected_methods
    lookup = {(r['design'],r['specification'],r['effect'],r['method']):r for r in summary['rows']}
    assert set(lookup) == set(rows)
    nulls = []
    for key, row in rows.items():
        record = lookup[key]
        success = int(row['reject'].sum())
        assert record['rejections'] == success and record['replications'] == len(row['reject'])
        close(record['rate'],success/len(row['reject']),1e-15,'recount rate')
        close(record['wilson95'],wilson(success,len(row['reject'])),3e-15,'Wilson interval')
        close(record['mean_statistic'],row['statistic'].mean(),1e-12,'mean statistic')
        if key[1]=='correct' and key[2]==0 and key[3].endswith('_gaussian'):
            tail=.01/(2*32); n=len(row['reject'])
            ci=[float(beta.ppf(tail,success,n-success+1)) if success else 0.,
                float(beta.ppf(1-tail,success+1,n-success)) if success<n else 1.]
            close(record['simultaneous99_binomial_interval'],ci,1e-15,'simultaneous exact binomial interval')
            assert record['contains_nominal_point05'] == (ci[0] <= .05 <= ci[1])
            nulls.append({'design':key[0],'method':key[3],'rejections':success,'interval':ci,'contains_005':ci[0]<=.05<=ci[1]})
    assert len(nulls)==32

    errors={k:0. for k in ['fixture','leave_operator','annihilation','statistic','pvalue','direction_cosine','normal_statistic','GLS_power','nct_quadrature']}
    fixture=np.load(study/'verification_fixtures.npz',allow_pickle=False)
    designs=[];power=[];index=0
    for periods in protocol['periods']:
        for rho in protocol['serial_correlations']:
            units=protocol['units'];n=periods*units;reps=protocol['replications']
            rng=np.random.default_rng(protocol['seed']+100000*index)
            entity=np.tile(np.arange(units),periods);groups=np.repeat(np.arange(periods),units)
            z=rng.normal(size=n)+.25*(entity-(units-1)/2)
            phi_full=np.column_stack([np.eye(units)[entity],z,z*z])
            covariance=np.kron(rho**np.abs(np.arange(periods)[:,None]-np.arange(periods)[None,:]),.4*np.ones((units,units))+.6*np.eye(units))
            lower=np.linalg.cholesky(covariance)
            ex,ey,ep=[lower@rng.normal(size=(n,reps)) for _ in range(3)]
            gx=np.r_[np.linspace(-1,1,units),2.,3.];gy=np.r_[np.linspace(.5,-.5,units),.3,.8]
            x=phi_full@gx[:,None]+ex;pilot=phi_full@gx[:,None]+ep
            for name,value in [('x',x),('pilot',pilot)]:
                errors['fixture']=max(errors['fixture'],close(fixture[f'd{index}_{name}'],value[:,:3],2e-12,'primitive replay fixture'))
            folds=(groups>=periods//2).astype(int)
            residuals=[np.eye(n)-bin_map(z,folds,q) for q in [8,12,16,24,32]]
            spline=np.eye(n)-spline_map(z,folds)
            operators={'coarse':residuals[0].T@residuals[0]/n,'fine':residuals[-1].T@residuals[-1]/n,
                       'extrapolation_beta1':sum(w*r.T@r for w,r in zip(weights(1),residuals))/n,
                       'extrapolation_beta2':sum(w*r.T@r for w,r in zip(weights(2),residuals))/n,
                       'spline':spline.T@spline/n}
            distances=np.abs(np.arange(periods)[:,None]-np.arange(periods)[None,:])
            bartlett=np.maximum(0.,1-distances/3)
            for spec in SPECS:
                phi=phi_full[:,:-1] if spec=='omitted_quadratic_mean' else phi_full
                assumed=np.eye(n) if spec=='omitted_dependence' else covariance
                geom=geometry(phi,assumed)
                leave=leave_map(phi,groups)
                errors['leave_operator']=max(errors['leave_operator'],close(leave,fixture[f'd{index}_{spec}_leave'],3e-13,'literal QR leave-cluster operator'))
                errors['annihilation']=max(errors['annihilation'],close(leave@phi,np.zeros_like(phi),2e-13,'mean annihilation'))
                for g in range(periods):
                    selected=np.flatnonzero(groups==g)
                    close(leave[np.ix_(selected,selected)],np.eye(units)/n,1e-15,'whole within-cluster block')
                qr,upper=np.linalg.qr(phi,mode='reduced')
                learned=phi@np.linalg.solve(upper,qr.T@pilot)
                directions={name:a@x for name,a in operators.items()}
                directions.update(leave_cluster_raw=leave.T@x,leave_cluster_learned=leave.T@(x-learned),gls=np.linalg.solve(assumed,x))
                signal=geom[0]@x;signal-=geom[2]@(geom[2].T@signal);signalnorm=np.linalg.norm(signal,axis=0)
                for name,direction in directions.items():
                    errors['fixture']=max(errors['fixture'],close(direction[:,:3],fixture[f'd{index}_{spec}_{name}_direction'],2e-10,'independent direction fixture'))
                for effect in protocol['effects']:
                    y=phi_full@gy[:,None]+effect*ex+ey
                    errors['fixture']=max(errors['fixture'],close(y[:,:3],fixture[f'd{index}_{spec}_e{effect}_y'],2e-12,'paired outcomes across specifications'))
                    for name,direction in directions.items():
                        statistic,pvalue,projected,degrees=direct_test(y,direction,geom)
                        observed=rows[(index,spec,effect,name+'_gaussian')]
                        errors['statistic']=max(errors['statistic'],close(statistic,observed['statistic'],2e-8,'symmetric-whitening orthogonal-residual t'))
                        errors['pvalue']=max(errors['pvalue'],close(pvalue,observed['pvalue'],2e-10,'incomplete-beta t survival'))
                        np.testing.assert_array_equal(pvalue<=.05,observed['reject'])
                        cos=np.sum(projected*signal,axis=0)/(np.linalg.norm(projected,axis=0)*signalnorm)
                        errors['direction_cosine']=max(errors['direction_cosine'],close(cos,observed['cosine'],2e-10,'direction cosine'))
                        assert lookup[(index,spec,effect,name+'_gaussian')]['df']==degrees
                    if spec=='correct':
                        critical=t.ppf(.95,n-phi.shape[1]-1)
                        conditional=nct.sf(critical,n-phi.shape[1]-1,effect*signalnorm)
                        errors['GLS_power']=max(errors['GLS_power'],close(conditional.mean(),lookup[(index,spec,effect,'gls_gaussian')]['mean_exact_conditional_power'],2e-12,'aligned GLS conditional power'))
                        degrees=n-phi.shape[1]-1;noncentrality=effect*signalnorm[0]
                        # Integrate over probability mass, not [0,infinity):
                        # generic infinite-interval quadrature can miss the
                        # concentrated chi-square density at larger degrees.
                        integrated,quaderror=quad(lambda probability:ndtr(noncentrality-critical*np.sqrt(chi2.ppf(probability,degrees)/degrees)),0,1,epsabs=1e-11,epsrel=1e-10)
                        errors['nct_quadrature']=max(errors['nct_quadrature'],close(integrated,conditional[0],2e-10,'independent chi-square integration for aligned nct'))
                        power.append({'design':index,'effect':effect,'GLS_mean_exact_power':float(conditional.mean()),'first_replication_quadrature_error':quaderror})
                    baseproducts=[(r@x)*(r@y) for r in residuals]
                    scores={'coarse':baseproducts[0],'fine':baseproducts[-1],
                            'extrapolation_beta1':sum(w*p for w,p in zip(weights(1),baseproducts)),
                            'extrapolation_beta2':sum(w*p for w,p in zip(weights(2),baseproducts)),
                            'spline':(spline@x)*(spline@y)}
                    for name,scoreset in scores.items():
                        mean=scoreset.mean(axis=0);cluster=(scoreset-mean).reshape(periods,units,reps).sum(axis=1)
                        variance=np.einsum('ir,ij,jr->r',cluster,bartlett,cluster)/n**2
                        statistic=mean/np.sqrt(variance)
                        observed=rows[(index,spec,effect,name+'_normal')]
                        errors['normal_statistic']=max(errors['normal_statistic'],close(statistic,observed['statistic'],2e-8,'dense Bartlett covariance'))
                        close(ndtr(-statistic),observed['pvalue'],2e-10,'normal survival')
                        np.testing.assert_array_equal(ndtr(-statistic)<=.05,observed['reject'])
            designs.append({'design':index,'periods':periods,'rho':rho,'fixed_Phi_sha256':hashlib.sha256(phi_full.tobytes()).hexdigest(),'X_primitive_replay_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'pilot_replay_sha256':hashlib.sha256(pilot.tobytes()).hexdigest(),'no_evaluation_Y_in_directions':True})
            print(f'Independently replayed design {index}',flush=True)
            index+=1

    paired=[];primary=[]
    originals={(x['design'],x['specification'],x['effect']):x for x in summary['paired_learning_comparison']}
    for design in range(4):
        for spec in SPECS:
            for effect in protocol['effects']:
                one=rows[(design,spec,effect,'leave_cluster_learned_gaussian')]['reject'].astype(bool)
                two=rows[(design,spec,effect,'leave_cluster_raw_gaussian')]['reject'].astype(bool)
                counts={'learned_only':int(np.sum(one&~two)),'raw_only':int(np.sum(two&~one)),
                        'both':int(np.sum(one&two)),'neither':int(np.sum(~one&~two))}
                original=originals[(design,spec,effect)]
                assert all(original[k]==v for k,v in counts.items())
                difference=one.astype(float)-two.astype(float);se=difference.std(ddof=1)/np.sqrt(len(difference))
                close([difference.mean()-1.96*se,difference.mean()+1.96*se],original['paired_normal_mc95'],1e-15,'paired descriptive Monte Carlo interval')
                discordant=counts['learned_only']+counts['raw_only']
                probability=float(binomtest(counts['learned_only'],discordant,.5,alternative='greater').pvalue) if discordant else 1.
                item={'design':design,'specification':spec,'effect':effect,**counts,'raw_rate':float(two.mean()),'learned_rate':float(one.mean()),'difference':float(difference.mean()),'paired_normal_mc95':original['paired_normal_mc95'],'one_sided_exact_discordance_p':probability}
                paired.append(item)
                if spec=='correct' and effect==.15:primary.append(item.copy())
    order=sorted(range(4),key=lambda i:primary[i]['one_sided_exact_discordance_p']);running=0.
    for rank,index in enumerate(order):
        running=max(running,(4-rank)*primary[index]['one_sided_exact_discordance_p'])
        primary[index]['holm_adjusted_p']=min(1.,running)
        primary[index]['holm_reject_005']=running<=.05
        primary[index]['holm_order']=rank+1
    return {'status':'PASS','recorded_utc':datetime.now(timezone.utc).isoformat(),'review_type':'Independent numerical verifier, not an additional experiment or novelty claim','verifier_sha256':sha(__file__),'runtime':runtime,'protocol_sha256':sha(study/'protocol.json'),'sources':protocol['sources'],'raw_sha256':sha(study/'replications.csv.gz'),'summary_sha256':sha(study/'summary.json'),'fixture_sha256':sha(study/'verification_fixtures.npz'),'all_raw_rows_independently_recomputed':count,'method_cells':len(rows),'independent_primitive_draws':4000,'distinct_evaluated_panels':12000,'paired_reference_specifications':3,'all_final_decisions_exactly_match':True,'maximum_absolute_errors':errors,'all_32_correct_null_simultaneous_intervals_contain_005':all(x['contains_005'] for x in nulls),'correct_null_intervals':nulls,'primary_learning_comparisons_with_Holm':primary,'all_paired_learning_comparisons':paired,'GLS_aligned_power_checks':power,'design_identity_receipts':designs,'seconds':time.perf_counter()-start,'numerical_independence':'No producer or estimator modules imported. Dense prediction maps, QR leave-out fits, eigen-whitening, explicit orthogonal residual norms, incomplete-beta tails and chi-square quadrature used. SplineTransformer library is shared for basis features, but Ridge and producer prediction functions are not used.','limits':['Four fixed designs, not random-design generalization.','Normal z-only bin/spline operators are scalar-adjustment probes on panel data, not the frozen main entity-adjusted implementation.','Only correct-null Gaussian references have the stated exact size guarantee; negative-control failures do not invalidate that theorem and successful negative controls do not extend it.','Known covariance shape and exact conditional-mean span are supplied; no real-data finite certificate is established.','Learning comparison is against raw leave-out with the same pilot bank charged; it is not a novel mechanism or superiority to GLS.','Generic off-axis alternatives have a noncentral denominator; ordinary noncentral-t power was checked only for the aligned GLS direction.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=DEFAULT_STUDY)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new verification receipt; existing results are preserved')
    try:
        result=run(args.study)
    except BaseException as error:
        result={'status':'FAIL','error':repr(error),'traceback':traceback.format_exc(),'verifier_sha256':sha(__file__)}
        with args.output.open('x') as handle:json.dump(result,handle,indent=2)
        raise
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({k:result[k] for k in ['status','all_raw_rows_independently_recomputed','maximum_absolute_errors','primary_learning_comparisons_with_Holm','seconds']},indent=2))
