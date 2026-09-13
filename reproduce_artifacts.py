"""Verify recorded results, replay designated calculations and redraw figures.

This command checks the studies listed in the reproduction map. Study-specific
commands handle full simulation regeneration and forecast retraining; aggregate
replay alone does not establish observational-panel calibration.
"""
import argparse, csv, hashlib, json, math, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent

def table(path, columns, rows):
    pd.DataFrame(rows, columns=columns).to_csv(path.with_suffix('.csv'), index=False)
    lines=['| '+' | '.join(map(str,columns))+' |', '| '+' | '.join(['---']*len(columns))+' |']
    lines += ['| '+' | '.join(map(str,row))+' |' for row in rows]
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-figures', action='store_true')
    parser.add_argument('--gaussian-python', help='Python interpreter with requirements-gaussian.txt for the full synthetic replay')
    parser.add_argument('--confirmation-python', help='Interpreter with results/seoul_confirmation/requirements.txt for frozen-model and timing reconstruction')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks = []
    for study in ['canonical_baselines', 'rough_control', 'canonical_beta2']:
        folder = ROOT / 'results' / study
        frame = pd.read_csv(folder / 'replications.csv', keep_default_na=False)
        summary = json.loads((folder / 'summary.json').read_text())
        rows = []
        for cell in summary['rows']:
            subset = frame[(frame.regime == cell['regime']) &
                           (frame.alternative == cell['alternative']) &
                           (frame.method == cell['method'])]
            assert len(subset) == 300 and int(subset.reject.sum()) == cell['rejections']
            assert abs(cell['rate'] - subset.reject.mean()) < 1e-15
            rows.append([cell[k] for k in ['regime', 'alternative', 'method', 'rejections', 'replications', 'rate']])
        table(args.output / study, ['Regime', 'Condition', 'Method', 'Rejections', 'Replications', 'Rate'], rows)
        checks.append({'study': study, 'rows': len(frame), 'cells': len(rows)})
    methods = [('gcm_coarse', 'GCM (8 bins)'), ('gcm_fine', 'GCM (32 bins)'),
               ('gcm_spline', 'Spline GCM'), ('extrapolated', 'Extrapolation'), ('kci_gamma', 'KCI')]
    canonical = json.loads((ROOT / 'results/canonical_beta2/summary.json').read_text())['rows']
    lookup = {(r['regime'], r['alternative'], r['method']): r['rate'] for r in canonical}
    rows = []
    canonical_methods = methods[:-1] + [('extrapolated_beta2', 'Extrapolation beta 2')] + methods[-1:]
    for key, label in canonical_methods:
        null = [lookup[(regime, 'null', key)] for regime in ['linear', 'smooth', 'nonlinear', 'heavy_tail']]
        power = min(lookup[(regime, 'positive_covariance', key)] for regime in ['linear', 'smooth', 'nonlinear', 'heavy_tail'])
        zero = lookup[('smooth', 'zero_covariance_dependence', key)]
        rows.append([label] + [f'{value:.3f}' for value in null + [power, zero]])
    table(args.output / 'tab_canonical_comparison', ['Method', 'Linear', 'Smooth', 'Quadratic', 'Heavy tail', 'Minimum power', 'Zero covariance'], rows)
    rough = json.loads((ROOT / 'results/rough_control/summary.json').read_text())['rows']
    lookup = {(r['regime'], r['alternative'], r['method']): r['rate'] for r in rough}
    cells = [(regime, 'null') for regime in ['kink', 'step', 'oscillation']] + [(regime, 'positive_covariance') for regime in ['kink', 'step', 'oscillation']] + [('kink', 'zero_covariance_dependence')]
    table(args.output / 'tab_rough_comparison', ['Method', 'Kink null', 'Step null', 'Oscillation null', 'Kink positive', 'Step positive', 'Oscillation positive', 'Zero covariance'], [[label] + [f'{lookup[(regime, condition, key)]:.3f}' for regime, condition in cells] for key, label in methods])
    public = pd.read_csv(ROOT / 'results/spline_panel/public_all_models.csv')
    assert len(public) == 123
    for (domain, method), family in public.groupby(['domain', 'method']):
        assert family.policy_family_size.eq(len(family)).all()
        np.testing.assert_allclose(family.p_policy, np.where(family.policy_abstain, 1.0, family.raw_p_one_sided))
        for values, flags, adjusted_column in [('raw_p_one_sided', 'raw_by_reject_recomputed', 'raw_by_adjusted_p_recomputed'), ('p_policy', 'policy_by_reject', 'policy_by_adjusted_p')]:
            rejected, adjusted, _, _ = multipletests(family[values].fillna(1.0), method='fdr_by')
            np.testing.assert_array_equal(rejected, family[flags])
            np.testing.assert_allclose(adjusted, family[adjusted_column], atol=1e-14, rtol=1e-12)
    checks.append({'study': 'all_public_audits', 'model_method_rows': 123, 'full_families': 12, 'raw_and_guarded_BY': 'PASS'})
    spline_draws = pd.read_csv(ROOT / 'results/spline_panel/panel_spline_draws.csv')
    spline_summary = pd.read_csv(ROOT / 'results/spline_panel/panel_spline_summary.csv')
    paired_draws = pd.read_csv(ROOT / 'results/spline_panel/panel_all_33_comparisons.csv')
    assert len(spline_draws) == 2700 and len(spline_summary) == 9 and len(paired_draws) == 9900
    for row in spline_summary.itertuples():
        cell = spline_draws[(spline_draws.condition == row.condition) & (spline_draws.clusters == row.clusters)]
        assert len(cell) == row.replications == 300 and int(cell.reject.sum()) == row.rejections
        assert abs(cell.reject.mean() - row.rejection_rate) < 1e-14
    joins = paired_draws.merge(spline_draws, on=['seed', 'replication', 'condition', 'clusters'], validate='many_to_one', suffixes=('_paired', '_original'))
    assert len(joins) == 9900
    np.testing.assert_allclose(joins.statistic_spline, joins.statistic, atol=1e-14, rtol=1e-12)
    np.testing.assert_array_equal(joins.reject_spline, joins.reject)
    np.testing.assert_array_equal(joins.n_spline, joins.n)
    checks.append({'study': 'spline_panel_pairing', 'datasets': 2700, 'cells': 9, 'paired_settings': 9900, 'all_rates_and_joins': 'PASS'})
    primary = public[public.method == 'extrapolated_beta_2']
    master = []
    for domain, family in primary.groupby('domain'):
        rejected, adjusted, _, _ = multipletests(family.p_policy, method='fdr_by')
        np.testing.assert_array_equal(rejected, family.policy_by_reject)
        np.testing.assert_allclose(adjusted, family.policy_by_adjusted_p, atol=1e-14, rtol=1e-12)
        master.append([domain, len(family), int(family.policy_nominal_positive.sum()), int(rejected.sum()), int(family.policy_abstain.sum())])
    factors = pd.read_csv(ROOT / 'results/factor_family/profile.csv')
    for lag, family in factors.groupby('lag'):
        assert len(family) == 212
        rejected, adjusted, _, _ = multipletests(family.p_two_sided, method='fdr_by')
        np.testing.assert_array_equal(rejected, family.by_reject)
        np.testing.assert_allclose(adjusted, family.by_adjusted_p, atol=1e-13, rtol=1e-12)
        if lag == 12:
            master.append(['Public factor alpha', 212, int((family.p_two_sided < .05).sum()), int(rejected.sum()), 0])
    table(args.output / 'tab_master', ['Task', 'Models', 'Nominal', 'BY', 'Abstain'], master)
    quality = pd.read_csv(ROOT / 'results/public_score_link/public_all_41_models.csv', float_precision='round_trip', keep_default_na=False, na_values=[''])
    cluster_quality = pd.read_csv(ROOT / 'results/public_score_link/public_all_cluster_correlations.csv', float_precision='round_trip', keep_default_na=False, na_values=[''])
    assert len(quality) == 41 and len(cluster_quality) == 12409
    for row in quality.itertuples():
        cluster_rows = cluster_quality[(cluster_quality.domain == row.domain) & (cluster_quality.model == row.model)]
        audit = primary[(primary.domain == row.domain) & (primary.model == row.model)].iloc[0]
        assert len(cluster_rows) == row.clusters_total and int(cluster_rows.n.sum()) == row.n_rows == audit.n
        defined = cluster_rows.status.eq('defined')
        baseline_defined = cluster_rows.baseline_status.eq('defined')
        paired = defined & baseline_defined
        assert int(defined.sum()) == row.clusters_defined
        assert int((~defined).sum()) == row.clusters_undefined
        expected = [cluster_rows.loc[defined, 'spearman_manual'].mean(), cluster_rows.loc[baseline_defined, 'baseline_spearman_manual'].mean(), (cluster_rows.loc[paired, 'spearman_manual'] - cluster_rows.loc[paired, 'baseline_spearman_manual']).mean()]
        np.testing.assert_allclose(expected, [row.mean_cluster_spearman, row.baseline_mean_cluster_spearman, row.paired_mean_spearman_difference], atol=1e-14, rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose([row.audit_raw_T, row.audit_raw_p, row.audit_guarded_p, row.audit_guarded_BY_p], [audit.statistic, audit.p_one_sided, audit.p_policy, audit.policy_by_adjusted_p], atol=1e-14, rtol=1e-11, equal_nan=True)
        final = 'abstain' if audit.policy_abstain else 'retain' if audit.policy_by_reject else 'null'
        assert row.audit_final_label == final
        if row.domain == 'ratings' and row.model == 'SVD interaction':
            assert row.mae_status == 'interaction_only_score_not_rating_prediction'
            assert np.isnan(row.original_scale_mae)
        else:
            assert row.mae_status == 'computed' and np.isfinite(row.original_scale_mae)
            assert row.original_scale_mae >= 0
        assert cluster_rows.loc[~defined, 'spearman_manual'].isna().all()
        assert np.isfinite(row.baseline_original_scale_mae)
    quality.to_csv(args.output / 'all_41_public_scores.csv', index=False)
    from scripts.analysis.render_public_model_roster import render_roster
    roster, roster_check = render_roster(ROOT / 'results/public_score_link/public_all_41_models.csv',
                                         ROOT / 'results/public_score_link/public_model_roster.md')
    assert not roster_check['mismatches'], roster_check['mismatches']
    (args.output / 'public_model_roster.md').write_text(roster)
    checks.append({'study': 'public_model_roster', **roster_check})
    checks.append({'study': 'public_score_link', 'models': 41, 'cluster_rows': 12409, 'means_and_final_audit_labels': 'PASS'})
    for name in ['panel_spline_summary', 'public_summary']:
        frame = pd.read_csv(ROOT / 'results/spline_panel' / (name + '.csv'))
        table(args.output / name, list(frame.columns), frame.to_numpy().tolist())
    beta = json.loads((ROOT / 'results/theory/beta_reference_verification.json').read_text())
    rows = [[row['assumed_beta']] + [f"{cell['rho']:.3f}" for cell in row['rho']] + [f"{row['weight_l2']:.3f}"] for row in beta['exponent_table']['table']]
    table(args.output / 'tab_beta_direction', ['Assumed beta', '.5', '1', '1.5', '2', '3', 'Weight norm'], rows)

    # Restricted observations are unavailable; verify only authorized aggregates.
    from scipy.stats import norm
    restricted_root = ROOT / 'results/restricted_aggregate'
    restricted_summary = json.loads((restricted_root / 'summary.json').read_text())
    for variant in ['historical_full_support', 'target_only_corrected', 'complete_calendar_primary']:
        frame = pd.read_csv(restricted_root / (variant + '.csv'), keep_default_na=False, float_precision='round_trip')
        records = json.loads((restricted_root / (variant + '.json')).read_text())
        assert len(frame) == len(records) == 30 and frame.model.nunique() == 30
        assert frame.model.tolist() == [row['model'] for row in records]
        for column in frame.columns:
            expected = [row[column] for row in records]
            if pd.api.types.is_numeric_dtype(frame[column]):
                np.testing.assert_allclose(frame[column], expected, rtol=1e-13, atol=1e-14)
            else:
                assert frame[column].tolist() == expected
        for exponent in [1, 2]:
            statistic = frame['T_beta' + str(exponent)].to_numpy(float)
            se = frame['se_beta' + str(exponent)].to_numpy(float)
            mean = frame['mean_beta' + str(exponent)].to_numpy(float)
            p = frame['p_beta' + str(exponent)].to_numpy(float)
            np.testing.assert_allclose(mean / se, statistic, rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(norm.sf(statistic), p, rtol=1e-11, atol=1e-13)
            retained, adjusted, _, _ = multipletests(p, method='fdr_by', alpha=.05)
            np.testing.assert_array_equal(retained, frame['by_retained_beta' + str(exponent)])
            np.testing.assert_allclose(adjusted, frame['by_adjusted_p_beta' + str(exponent)], rtol=1e-11, atol=1e-12)
            assert np.array_equal(np.where(retained, 'RETAIN', 'NULL'), frame['family_decision_beta' + str(exponent)])
            counts = restricted_summary['counts_by_variant'][variant][str(exponent)]
            assert int(retained.sum()) == counts['by_retained'] == 0
            assert int((p < .05).sum()) == counts['nominal_positive'] == 1
        shown = frame.copy()
        for column in shown.columns:
            if 'decision' in column:
                shown[column] = shown[column].replace({'NULL': 'NOT_RETAINED'})
        table(args.output / ('restricted_' + variant), list(shown.columns), shown.to_numpy().tolist())
    checks.append({'study': 'restricted_model_aggregates', 'variants': 3, 'models_per_variant': 30,
                   'full_families': 6, 'statistic_p_and_BY': 'PASS', 'raw_observations_available': False,
                   'private_training_information_vintage_certified': False})
    cli_models = 0
    cli_labels = []
    for domain in ['electricity', 'ratings', 'retail', 'portfolios']:
        folder = ROOT / 'results/forecast_cli' / domain
        profile = pd.read_csv(folder / 'profile.csv', keep_default_na=False, na_values=[''], float_precision='round_trip')
        resolution = pd.read_csv(folder / 'resolution_trace.csv', keep_default_na=False, na_values=[''], float_precision='round_trip')
        ordered = pd.read_csv(folder / 'by_trace.csv', keep_default_na=False, float_precision='round_trip')
        assert profile.model.nunique() == len(profile) and profile.policy_family_size.eq(len(profile)).all()
        assert not profile.finite_certificate_issued.any()
        expected = np.where(profile.policy_abstain, 1., profile.raw_p_one_sided)
        np.testing.assert_allclose(profile.p_policy, expected, atol=0, rtol=0)
        retained, adjusted, _, _ = multipletests(expected, method='fdr_by', alpha=.05)
        np.testing.assert_array_equal(retained, profile.policy_by_reject)
        np.testing.assert_allclose(adjusted, profile.policy_by_adjusted_p, atol=1e-14, rtol=1e-12)
        final = np.where(profile.policy_abstain, 'ABSTAIN', np.where(retained, 'RETAIN', 'NULL'))
        np.testing.assert_array_equal(final, profile.final_label)
        for model, group in resolution.groupby('model'):
            row = profile[profile.model == model].iloc[0]
            assert len(group) == 5
            np.testing.assert_allclose(group.weight.sum(), 1., atol=1e-12)
            np.testing.assert_allclose(group.weight @ (group.resolution_q.to_numpy(float)**-2), 0., atol=1e-12)
            np.testing.assert_allclose(group.weighted_mean_contribution.sum(), row.mean_product, atol=1e-12)
        for screen, group in ordered.groupby('screen'):
            group = group.sort_values('ordered_rank')
            m = len(profile)
            assert len(group) == m
            threshold = .05 * np.arange(1, m+1) / (m * np.sum(1 / np.arange(1, m+1)))
            np.testing.assert_allclose(group.rank_cutoff, threshold, atol=1e-15)
            candidate = np.flatnonzero(group.pvalue.to_numpy() <= threshold)
            k = int(candidate[-1]+1) if len(candidate) else 0
            assert group.step_up_last_passing_rank.eq(k).all()
            decision = group.pvalue.to_numpy() <= threshold[k-1] if k else np.zeros(m, bool)
            np.testing.assert_array_equal(decision, group.retained_by_step_up)
        cli_models += len(profile)
        cli_labels.extend(final.tolist())
    assert cli_models == 41
    assert {label: cli_labels.count(label) for label in ['RETAIN', 'ABSTAIN', 'NULL']} == {'RETAIN':23, 'ABSTAIN':8, 'NULL':10}
    subprocess.run([sys.executable, str(ROOT / 'scripts/analysis/forecast_audit_cli.py'),
                    '--input', str(ROOT / 'examples/forecast_audit_cli/synthetic_forecasts.csv'),
                    '--output-dir', str(args.output / 'cli_example'), '--frequency', 'cluster',
                    '--lag', '0', '--beta', '2', '--ladder', '2,3,4'], check=True)
    for expected in (ROOT / 'examples/forecast_audit_cli/expected').glob('*.csv'):
        assert expected.read_bytes() == (args.output / 'cli_example' / expected.name).read_bytes()
    checks.append({'study':'guarded_forecast_CLI', 'models':41, 'full_families':4,
                   'resolution_and_BY_trace':'PASS', 'synthetic_CSV_entrypoint':'PASS', 'finite_certificate_issued':False})


    # New synthetic study: stored arithmetic is portable across the two
    # declared runtimes; exact primitive replay uses --gaussian-python.
    gaussian = ROOT / 'results/gaussian_reference'
    gs = json.loads((gaussian / 'summary.json').read_text())
    gf = pd.read_csv(gaussian / 'replications.csv.gz', keep_default_na=False)
    assert len(gf) == 468000 and len(gs['rows']) == 468
    assert hashlib.sha256((gaussian / 'replications.csv.gz').read_bytes()).hexdigest() == gs['raw_sha256']
    assert hashlib.sha256((gaussian / 'verification_fixtures.npz').read_bytes()).hexdigest() == gs['fixture_sha256']
    grouped = gf.groupby(['design','specification','effect','method'], sort=False)
    for cell in gs['rows']:
        subset = grouped.get_group(tuple(cell[k] for k in ['design','specification','effect','method']))
        assert len(subset) == cell['replications'] == 1000
        assert int(subset.reject.sum()) == cell['rejections']
        assert np.array_equal(subset.reject.to_numpy(), (subset.pvalue <= .05).to_numpy())
    audit = json.loads((gaussian / 'independent_verification.json').read_text())
    assert audit['raw_sha256'] == gs['raw_sha256']
    assert all(x['holm_reject_005'] for x in audit['primary_learning_comparisons_with_Holm'])
    checks.append({'study':'known_covariance_gaussian_reference','method_rows':len(gf),'method_cells':468,
                   'independent_primitive_draws':4000,'distinct_evaluated_panels':12000,
                   'stored_arithmetic':'PASS','full_primitive_replay_requested':bool(args.gaussian_python),
                   'real_panel_finite_certificate':False})
    if args.gaussian_python:
        subprocess.run([args.gaussian_python,str(ROOT/'scripts/analysis/verify_gaussian_panel_reference.py'),
                        '--study',str(gaussian),'--output',str(args.output/'gaussian_full_verification.json')],check=True)
    # Exactly three safe aggregate files; no private source reconstruction.
    temporal = ROOT / 'results/restricted_temporal_aggregate'
    ts = json.loads((temporal/'summary.json').read_text())
    tf = pd.read_csv(temporal/'model_metrics.csv',keep_default_na=False)
    assert len(tf) == len(ts['models']) == 7
    assert set(tf.model) == {row['model'] for row in ts['models']}
    for row in ts['models']:
        actual = tf[tf.model == row['model']].iloc[0]
        for key in ['equal_month_mean_spearman','pooled_spearman','within_month_spearman_standard_deviation']:
            assert abs(float(actual[key])-row[key]) < 1e-15
    assert ts['scope']['common_evaluation_security_months'] == 18036
    assert ts['scope']['common_eligible_evaluation_periods'] == 132
    assert ts['historical_vintages_verified'] is False and ts['finite_certificate'] is False
    checks.append({'study':'retrospective_restricted_model_aggregates','model_rows':7,
                   'common_observations':18036,'common_periods':132,'CSV_JSON_arithmetic':'PASS',
                   'private_source_reproduction':False,'historical_vintages_verified':False})

    # Current design studies and matched-support family decisions.
    import importlib.util
    spec = importlib.util.spec_from_file_location('design_results', ROOT/'scripts/analysis/verify_design_results.py')
    design_results = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(design_results)
    checks.extend(design_results.verify(ROOT))

    subprocess.run([sys.executable,str(ROOT/"scripts/analysis/verify_inference_studies.py"),"--output",str(args.output/"inference_studies")],check=True)
    subprocess.run([sys.executable,str(ROOT/"scripts/analysis/fod_comparison.py"),"--verify",str(ROOT/"results/inference_validation/fod_comparison")],check=True)
    checks.append({"study":"estimated_references_and_family_inference","receipt":"inference_studies/verification.json","learned_cells":48,"family_cells":216,"near_copy_cells":234,"classical_comparison_cells":288})
    subprocess.run([sys.executable,str(ROOT/'scripts/analysis/verify_categorical_studies.py'),
                    '--base',str(ROOT/'results/category_reference'),
                    '--output',str(args.output/'categorical_verification.json')],check=True)
    # Verifiers run on a copy so the distributed release remains unchanged.
    for name, verifier in [('seoul_confirmation','verify.py'),('retail_fit_changes','verify_aggregates.py')]:
        study_copy=args.output/name
        shutil.copytree(ROOT/'results'/name,study_copy)
        subprocess.run([sys.executable,str(study_copy/verifier)],check=True)
        if name=='seoul_confirmation' and args.confirmation_python:
            subprocess.run([args.confirmation_python,str(study_copy/'verify_predictions.py')],check=True)
    checks.append({'study':'categorical_validation_and_confirmation','categorical_rows':216000,
                   'categorical_cells':216,'rental_risk_rows':74,'rental_paired_comparisons':82,
                   'retail_models':2,'frozen_rental_predictions_checked':bool(args.confirmation_python),'scope':'Recorded arithmetic and specified primitive replays; not general panel calibration'})
    subprocess.run([sys.executable,str(ROOT/'scripts/analysis/verify_reference_certificate_efficiency.py'),
                    '--study',str(ROOT/'results/reference_certificate_efficiency'),
                    '--output',str(args.output/'reference_certificate_efficiency')],check=True)
    checks.append({'study':'signed_reference_certificate_efficiency','stored_rows':500000,
                   'cells':500,'independently_rebuilt_rows':4000,'full_primitive_replications':8,
                   'scope':'Independent all-row arithmetic and selected complete synthetic replays; identical historical and hardened output bytes.'})
    subprocess.run([sys.executable,str(ROOT/'results/public_archive_certificate/verify.py')],check=True)
    checks.append({'study':'fixed_public_archive_certificate','archive_rows':70080,
                   'candidate_method_rows':21,'census_comparison_rows':21,
                   'scope':'Read-only frozen archive, draws, kernel arithmetic, finite bounds and BY; fixed-archive categorical target, not future forecasting gain.'})
    subprocess.run([sys.executable,str(ROOT/'scripts/analysis/verify_seoul_gate_tables.py'),
                    '--output',str(args.output/'seoul_gate_tables.json')],check=True)
    if args.confirmation_python:
        subprocess.run([args.confirmation_python,str(ROOT/'results/seoul_gate_diagnosis/diagnose_gate.py'),
                        '--study-dir',str(ROOT/'results/seoul_confirmation')],check=True)
    checks.append({'study':'seoul_gate_diagnosis','csv_tables':6,'candidate_rows':16,
                   'weekly_moment_rows':128,'selector_rows':6,'posthoc_arithmetic_rows':112,
                   'frozen_model_replay':bool(args.confirmation_python),
                   'scope':'Read-only original decision, all-candidate loss and gate arithmetic; no new gate selection or confirmation.'})
    for study in ['certificate_factorial', 'joint_bias', 'forecast_confirmation']:
        subprocess.run([sys.executable, str(ROOT/'results'/study/'verify.py'),
                        '--output', str(args.output/study)], check=True)
        checks.append({'study':study, 'receipt':study+'/verification.json',
                       'scope':'Independent recorded arithmetic and specified primitive reconstruction; source retraining is a separate command.'})
    with (args.output/'matched_betting_verification.json').open('w') as receipt:
        subprocess.run([sys.executable, str(ROOT/'results/matched_betting/verify.py')],
                       check=True, stdout=receipt)
    checks.append({'study':'matched_betting','receipt':'matched_betting_verification.json',
                   'stored_rows':48000, 'scope':'All-row arithmetic and 512 independent primitive betting reconstructions.'})
    subprocess.run([sys.executable, str(ROOT/'results/aggregate_bias/reproduce.py'),
                    '--output', str(args.output/'aggregate_bias')], check=True)
    checks.append({'study':'aggregate_learning_bias', 'receipt':'aggregate_bias/verification.json',
                   'archive_candidates':32, 'archive_method_rows':128, 'simulation_rows':144000,
                   'simulation_method_summaries':144,
                   'scope':'Independent known-category-mass bias validation; exposed-archive diagnosis and complete simulation arithmetic, not a new forecast confirmation.'})
    if not args.skip_figures:
        subprocess.run([sys.executable, str(ROOT/'results/aggregate_bias/make_figure.py'),
                        '--output', str(args.output/'aggregate_figure')], check=True)
        checks.append({'study':'aggregate_validation_figure', 'settings':72,
                       'rectangle_median_slack_wins':10,
                       'scope':'All plotted ratios reconstructed from recorded summaries; deterministic vector export.'})
        subprocess.run([sys.executable, str(ROOT/'results/certificate_factorial/make_figure.py'),
                        '--output', str(args.output/'factorial_figure')], check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/figures/make_efficiency_figure.py'),
                        '--summary',str(ROOT/'results/reference_certificate_efficiency/summary.csv'),
                        '--output-dir',str(args.output/'figures')],check=True)
        subprocess.run([sys.executable,str(ROOT/"scripts/figures/make_reference_training_figure.py"),"--output-dir",str(args.output/"reference_training_figure")],check=True)
        subprocess.run([sys.executable,str(ROOT/"scripts/figures/make_directional_model_figure.py"),"--input",str(ROOT/"results/directional_comparison/model_comparison.csv"),"--output-dir",str(args.output/"directional_figure")],check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/figures/make_gaussian_reference.py'),'--output-dir',str(args.output/'gaussian_figure')],check=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/figures/make_figures.py'), '--evidence-root', str(ROOT / 'results/figure_evidence'), '--output-dir', str(args.output / 'figures'), '--primary-beta', '2'], check=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/figures/make_publication_figures.py'), '--source-root', str(ROOT), '--sources-json', 'results/publication_figure_assets/source_map.json', '--output-dir', str(args.output / 'figures_current')], check=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/figures/verify_publication_figures.py'), '--source-root', str(ROOT), '--output-dir', str(args.output / 'figures_current')], check=True)
    report = {'status': 'PASS', 'scope': 'Stored aggregate arithmetic and deterministic artifacts; not refitted raw observations', 'checks': checks, 'public_forecasters': len(primary), 'paper_included': (ROOT / 'paper.pdf').is_file()}
    (args.output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
