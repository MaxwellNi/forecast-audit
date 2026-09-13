"""Verify recorded family inference and baseline-adjustment experiments."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from statsmodels.stats.multitest import multipletests


SOURCE = Path(__file__).resolve().parent
PUBLIC = SOURCE.parents[1]
HERE = PUBLIC / "results/inference_validation"
OUTPUT = None


def verify_family():
    directory = HERE/"raw_family_study"
    protocol = json.loads((directory/"protocol.json").read_text())
    assert protocol["source_sha256"] == hashlib.sha256((SOURCE/"independent_entity_family.py").read_bytes()).hexdigest()
    data = np.load(directory/"evaluation_statistics.npz", allow_pickle=False)
    reference = np.load(directory/"independent_null_calibration.npz", allow_pickle=False)["statistics"]
    p = data["pvalue"]; ts = data["statistic"]
    assert not data["guarded"].any()
    np.testing.assert_allclose(p[..., 0], norm.sf(ts), rtol=1e-13, atol=1e-14)
    for g, count in enumerate(protocol["entities"]):
        np.testing.assert_allclose(p[:, :, :, g, :, 1], student_t.sf(ts[:, :, :, g], count-1), rtol=1e-13, atol=1e-14)
        # Brute force comparisons independently check the searchsorted tail.
        for m in range(3):
            for s in range(4):
                values = ts[:20, s, m, g, :]
                rank_p = (1+(reference[:, m, g, None, None] >= values[None]).sum(axis=0))/(len(reference)+1)
                np.testing.assert_array_equal(p[:20, s, m, g, :, 2], rank_p)
    summary = pd.read_csv(directory/"family_summary.csv")
    validated = 0
    for row in summary.itertuples(index=False):
        s = protocol["signals_for_first_three_models"].index(row.signal)
        g = protocol["entities"].index(row.entities)
        m = protocol["methods"].index(row.method)
        j = protocol["inference"].index(row.inference)
        k = row.family_size
        values = p[:, s, m, g, :k, j]
        ordered = np.sort(values, axis=1)
        cutoffs = .05*np.arange(1, k+1)/(k*np.sum(1/np.arange(1, k+1)))
        positions = np.where(ordered <= cutoffs, np.arange(1, k+1), 0).max(axis=1)
        threshold = np.where(positions > 0, cutoffs[np.maximum(positions-1, 0)], -1.)
        decisions = values <= threshold[:, None]
        for rep in (0, 41, 509, 999):
            np.testing.assert_array_equal(decisions[rep], multipletests(values[rep], .05, method="fdr_by")[0])
        null = np.ones(k, bool)
        if row.signal > 0: null[:3] = False
        fdp = decisions[:, null].sum(axis=1)/np.maximum(decisions.sum(axis=1), 1)
        np.testing.assert_allclose(fdp.mean(), row.FDR, rtol=0, atol=1e-14)
        if row.signal > 0:
            np.testing.assert_allclose(decisions[:, :3].mean(), row.power, rtol=0, atol=1e-14)
        validated += 1
    expected = pd.read_csv(directory/"analytic_expectations.csv")
    max_z = 0.
    for row in expected.itertuples(index=False):
        s = protocol["signals_for_first_three_models"].index(row.signal)
        m = protocol["methods"].index(row.method)
        for g, count in enumerate(protocol["entities"]):
            values = data["mean"][:, s, m, g, 0]
            z = abs(values.mean()-row.exact_mean_raw_score)/(values.std(ddof=1)/np.sqrt(len(values)))
            max_z = max(max_z, z)
            assert z < 5
    width = np.sqrt(np.log(40.)/(2*len(p)))
    np.testing.assert_allclose(summary.FDR_hoeffding_low, np.maximum(0.,summary.FDR-width), atol=1e-14)
    np.testing.assert_allclose(summary.FDR_hoeffding_high, np.minimum(1.,summary.FDR+width), atol=1e-14)
    return {"status": "PASS", "family_summary_cells": validated,
            "independent_statsmodels_BY_comparisons": validated*4,
            "brute_force_calibration_tail_cells": 3*3*4*20*11,
            "maximum_analytic_mean_MC_z": max_z,
            "interval_scope": "Evaluation Monte Carlo intervals condition on the one independent calibration bank; bank uncertainty is additional."}



def verify_learned():
    directory=PUBLIC/'results/learned_references'
    protocol=json.loads((directory/'protocol.json').read_text())
    assert protocol['source_sha256']==hashlib.sha256((SOURCE/'learned_reference_study.py').read_bytes()).hexdigest()
    draws=pd.read_csv(directory/'replications.csv')
    summary=pd.read_csv(directory/'summary.csv')
    np.testing.assert_allclose(draws.p,norm.sf(draws.statistic),atol=2e-15,rtol=1e-12)
    assert (draws.nuisance_bias.abs()<=draws.bias_bound+1e-14).all()
    validated=0
    for row in summary.itertuples(index=False):
        cell=draws[(draws.entities==row.entities)&(draws.groups==row.groups)&
                   (draws.training_rows==row.training_rows)&(draws.method==row.method)]
        assert len(cell)==row.replications
        assert int((cell.p<.05).sum())==row.rejections
        assert abs(row.rate-row.rejections/row.replications)<1e-14
        expected=cell.nuisance_bias+(0.0225/(row.entities-1) if row.method=='shared' else 0.)
        np.testing.assert_allclose(cell.expected_mean,expected,atol=1e-14,rtol=0)
        validated+=1
    return {'status':'PASS','replication_records':len(draws),'summary_cells':validated,
            'error_bound_scope':'Diagnostics under the known synthetic law; not confidence bounds under an unknown law.'}

def historical_checks():
    results = PUBLIC/"results"
    rows = []
    for source, method in [(results/"current_core_sensitivity/draws.csv", "extrapolation"),
                           (results/"spline_panel/panel_spline_draws.csv", "spline")]:
        frame = pd.read_csv(source)
        groupcols = ["condition", "beta"] if method == "extrapolation" else ["condition"]
        for key, group in frame[(frame.clusters == 25)&frame.condition.str.endswith("null")].groupby(groupcols):
            label = f"beta{key[1]}" if method == "extrapolation" else "spline"
            values = group.statistic.to_numpy()
            corrected = values*np.sqrt(24/25)
            rows.append({"method": label, "condition": key[0], "replications": len(values),
                         "historical_normal": int((values > norm.ppf(.95)).sum()),
                         "cr1_normal": int((corrected > norm.ppf(.95)).sum()),
                         "cr1_t24": int((corrected > student_t.ppf(.95, 24)).sum())})
    pd.DataFrame(rows).to_csv(OUTPUT/"short_cluster_reference_sensitivity.csv", index=False)
    variance_rows = []
    draws = pd.read_csv(results/"current_core_sensitivity/draws.csv")
    selected = draws[(draws.study == "cluster_count_exponent")&(draws.beta == 1)]
    qs = np.array([8, 12, 16, 24, 32])
    for (count, condition), group in selected.groupby(["clusters", "condition"]):
        means = group[[f"product_mean_q{q}" for q in qs]].to_numpy()
        covariance = np.cov(means, rowvar=False, ddof=1)
        for beta in (1, 2):
            weights = np.linalg.pinv(np.column_stack([np.ones(5), qs.astype(float)**(-beta)]))[0]
            combined = means@weights
            np.testing.assert_allclose(np.var(combined, ddof=1), weights@covariance@weights, rtol=1e-12)
            variance_rows.append({"clusters": count, "condition": condition, "beta": beta,
                "empirical_variance_ratio": float(weights@covariance@weights/covariance[-1, -1]),
                "weight_l1": float(np.abs(weights).sum()), "replications": len(group),
                "scope": "Monte Carlo covariance of the five fitted mean products; ratio to q32. Not a guarantee or a within-dataset standard-error ratio."})
    pd.DataFrame(variance_rows).to_csv(OUTPUT/"extrapolation_empirical_variance_ratio.csv", index=False)
    near = pd.read_csv(results/"design_validation/near_copy/summary.csv")
    focus = near[(near["shape"] == "linear")&(near.noise_scale == .01)]
    focus.to_csv(OUTPUT/"historical_near_copy_cell.csv", index=False)
    assert focus[focus.method == "bins32"].period_cluster_statistic_rate.iloc[0] == .93
    assert focus[focus.method == "spline"].period_cluster_statistic_rate.iloc[0] == .11
    assert focus[focus.method == "spline"].two_way_cluster_statistic_rate.iloc[0] == .08
    assert (focus[focus.method.str.startswith("extrapolation")].period_cluster_statistic_rate == 0.).all()
    raw = pd.read_csv(results/"equal_entity_directional/summary.csv")
    null = raw[(raw.method == "directional")&(raw.injected_association == 0)]
    assert null.rejection_rate.min() == .054 and null.rejection_rate.max() == .088
    return {"status": "PASS", "short_cluster_cells": len(rows),
            "near_copy_note": "93/100 is bins32; spline is 11/100, two-way spline8/100; extrapolation is 0/100 with shifted negative statistics, not proof of calibration. These are fixed restricted-pattern noise redraws, not fully regenerated rank-null panels.",
            "raw_directional_old_null_range": [float(null.rejection_rate.min()), float(null.rejection_rate.max())],
            "source_hashes": {str(path.relative_to(PUBLIC)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [results/"current_core_sensitivity/draws.csv", results/"spline_panel/panel_spline_draws.csv",
                             results/"design_validation/near_copy/summary.csv", results/"equal_entity_directional/summary.csv"]}}


def verify_near_copy():
    import near_copy_adjustment as current
    checks = []
    for name, source in (("near_copy_weak_study", "near_copy_adjustment.py"),):
        directory = HERE/name
        protocol = json.loads((directory/"protocol.json").read_text())
        assert protocol["source_sha256"] == hashlib.sha256((SOURCE/source).read_bytes()).hexdigest()
        data = np.load(directory/"evaluation.npz", allow_pickle=False)
        ref = np.load(directory/"calibration.npz", allow_pickle=False)
        p = np.load(directory/"pvalues.npz", allow_pickle=False)["pvalue"]
        t = data["statistic"]; guard = data["guarded"]
        expected = np.empty_like(p)
        expected[..., 0] = norm.sf(t*np.sqrt(25/24))
        expected[..., 1] = student_t.sf(t, 24)
        configs = protocol["configs_noise_correlation"]
        for m in range(6):
            for c, (sigma, rho) in enumerate(configs):
                ref_index = configs.index([sigma, 0.])
                bank = ref["statistic"][:, m, ref_index]
                expected[:, m, c, 2] = (1+(bank[:, None] >= t[None, :, m, c]).sum(axis=0))/(len(bank)+1)
        expected[guard] = 1.
        np.testing.assert_allclose(expected, p, rtol=1e-13, atol=1e-14)
        np.testing.assert_array_equal(expected <= .05, p <= .05)
        summary = pd.read_csv(directory/"summary.csv")
        for row in summary.itertuples(index=False):
            m = protocol["methods"].index(row.method)
            c = configs.index([row.noise_scale, row.injected_correlation])
            j = ["historical_normal", "cr1_student_t", "independent_supplied_null"].index(row.inference)
            assert int((p[:, m, c, j] <= .05).sum()) == row.rejections
            assert float(guard[:, m, c].mean()) == row.guarded_fraction
        assert guard[:, :, 0].all()
        checks.append({"run": name, "status": "PASS", "summary_cells": len(summary),
                       "brute_force_pvalue_checks": int(p.size),
                       "all_exact_copy_attempts_abstain": True})
    # Reproduce non-selected final-run evaluation seeds directly from source.
    protocol = json.loads((HERE/"near_copy_weak_study/protocol.json").read_text())
    data = np.load(HERE/"near_copy_weak_study/evaluation.npz", allow_pickle=False)
    for rep in (0, 17, 509, 999):
        result = current.replication(protocol["evaluation_seed_start"]+rep)
        for key, actual in zip(("mean", "standard_error", "statistic", "guarded"), result):
            np.testing.assert_array_equal(actual, data[key][rep])
    return {"runs": checks, "final_generator_seed_replays": 4,
            "scope": "The weak grid adds exploratory alternatives on original primitive seeds; repeated null cells are not independent new confirmation."}


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    OUTPUT=args.output
    OUTPUT.mkdir(parents=True,exist_ok=False)
    receipt = {"learned_references": verify_learned(), "family": verify_family(), "historical": historical_checks(), "near_copy": verify_near_copy()}
    (OUTPUT/"verification.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))
