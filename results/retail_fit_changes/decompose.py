"""Exact descriptive decomposition of the recorded M5 middle-fold fit change.

This is retrospective accounting of known reversals, not a new confirmatory
experiment and not estimation of the population feedback terms in paper Eq6.
Provider observations, predictions, residuals and entity IDs are not exported.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
core = direction = cli = None


def load_analysis(directory):
    """Load the released analysis modules from an explicit or package-local path."""
    global core, direction, cli
    sys.path.insert(0, str(directory.resolve()))
    import audit_panel_predictions
    import directional_audit
    import forecast_audit_cli
    core, direction, cli = audit_panel_predictions, directional_audit, forecast_audit_cli


FILES = {'Past price and calendar': 'model_03.parquet', 'Croston SBA': 'model_08.parquet'}
QS = (8, 12, 16, 24, 32)
TERMS = ['outcome_fit_shift', 'forecast_fit_shift', 'joint_fit_shift']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def hac(values, periods):
    scores, gaps = core.calendar_scores(values-values.mean(axis=0), periods, 'W-FRI')
    assert gaps == 0 and len(scores) == 17
    return core.bartlett_score_covariance(scores, 2)/len(values)**2


def run(out, inputs, analysis_dir, comparison, comparison_receipt):
    load_analysis(analysis_dir)
    assert not (out/'receipt.json').exists(), 'Do not overwrite an existing decomposition'
    out.mkdir(parents=True, exist_ok=True)
    published = pd.read_csv(comparison)
    source_receipt = json.loads(comparison_receipt.read_text())
    expected_hashes = {r['filename']: r['sha256'] for r in source_receipt['source_input_hashes']['retail']}
    imports = [Path(core.__file__), Path(direction.__file__), Path(cli.__file__),
               analysis_dir/'audit_decisions.py', analysis_dir/'cluster_covariance_reference.py']
    recorded = json.loads((HERE/'protocol.json').read_text())
    expected_code = {Path(name).name: digest for name, digest in recorded['sources_sha256'].items()}
    assert all(sha(path) == expected_code[path.name] for path in imports), 'Analysis module bytes differ from the recorded implementation'
    assert sha(comparison) == recorded['published_comparison_sha256'], 'Comparison table differs from the recorded support'
    source_hashes = {name: sha(inputs[name]) for name in FILES.values()}
    assert source_hashes == recorded['input_sha256'], 'Input bytes differ from the recorded source cohort'
    assert all(source_hashes[name] == expected_hashes[name] for name in source_hashes)
    ladder, weights = cli.validate_specification('W-FRI', 2, 2., QS, .05)
    protocol = {
        'written_utc': utc(), 'script_sha256': sha(__file__),
        'scope': 'Retrospective exact empirical accounting on already published public M5 support. No independence, causal interpretation, calibrated inference or prospective discovery claim.',
        'models': FILES, 'input_sha256': source_hashes, 'resolution_ladder': list(ladder),
        'beta': 2, 'intercept_weights': weights.tolist(), 'frequency': 'W-FRI', 'lag': 2,
        'support': 'Original all28weeks train nuisance fits; evaluate only original contiguous folds1,2,3, exactly68000rows in17weeks. Original standardized within-week ranks and baseline bins are unchanged.',
        'identity': 'rXc=X-fXc; rYc=Y-fYc; dx=fXd-fXc; dy=fYd-fYc. Directional minus complementary product = -rXc*dy-rYc*dx+dx*dy, row by row at each resolution. Combine products and each term linearly with the same beta2 weights; never multiply separately extrapolated residuals.',
        'terms': {'outcome_fit_shift': '-mean(rXc*dy)', 'forecast_fit_shift': '-mean(rYc*dx)', 'joint_fit_shift': 'mean(dx*dy)'},
        'covariance': 'Compute lag2 Bartlett HAC on centered week sums of all15 resolution/term columns, retaining every cross-resolution and cross-term covariance. Also compute full5x5 covariance of complementary product, directional product and3 combined terms. No extra small-sample correction is introduced into comparison with old SEs.',
        'verification': 'Check all input hashes against the recorded comparison receipt, exact full cohort hash,68000rows/17weeks, no boundary fallback or unseen entities on middle folds, row-level identity, same-support mean/SE replay, cross-resolution covariance mapping, and dense versus sparse complementary-fit agreement.',
        'export_scope': 'Model, resolution and week aggregates and opaque hashes only; no provider observations, forecasts, residuals, or entity labels.',
        'sources_sha256': {'scripts/analysis/'+p.name: sha(p) for p in imports},
        'published_comparison_sha256': sha(comparison),
    }
    dump(out/'protocol.json', protocol)
    resolutions, summaries, weeks, covariance_rows, cross_resolution = [], [], [], [], []
    verification = []
    for name, filename in FILES.items():
        raw = pd.read_parquet(inputs[filename])
        frame, cohort = cli.validate_frame(raw, 'W-FRI', 2)
        frame = frame.reset_index(drop=True)
        assert frame.model.nunique() == 1 and frame.model.iloc[0] == name
        original = published[(published.domain == 'retail') & (published.model == name)]
        assert len(original) == 4 and cohort[name] == original.cohort_sha256.iloc[0]
        entities = pd.factorize(frame.entity.astype(str), sort=True)[0]
        products_c, products_d, terms, masks = [], [], [], []
        max_identity, max_solver = 0., 0.
        for q in QS:
            fit = core.panel_residuals(frame, 'contiguous', q=q)
            folds, bins = fit['folds'], fit['baseline_bins']
            middle = np.isin(folds, (1, 2, 3))
            assert int(middle.sum()) == 68000 and frame.loc[middle, 'period'].nunique() == 17
            xr, xf, xu = direction.directional_residuals(fit['forecast_rank'], entities, bins, folds, 'forecast')
            yr, yf, yu = direction.directional_residuals(fit['outcome_rank'], entities, bins, folds, 'outcome')
            assert not (xf[middle].any() or xu[middle].any() or yf[middle].any() or yu[middle].any())
            xc, yc = fit['forecast_residual'], fit['outcome_residual']
            xd, yd = xr[:, 0], yr[:, 0]
            dx, dy = xc-xd, yc-yd
            term = np.column_stack([-xc*dy, -yc*dx, dx*dy])
            c, d = xc*yc, xd*yd
            error = float(np.max(np.abs((d-c)-term.sum(axis=1))))
            assert error < 1e-12
            max_identity = max(max_identity, error)
            # Fit differences must arise from changed training support, not the two LS solvers.
            for fold in (1, 2, 3):
                evaluation = folds == fold
                values = np.column_stack([fit['forecast_rank'], fit['outcome_rank']])
                dense = direction.additive_prediction(values, entities, bins, folds != fold, evaluation)
                sparse = np.column_stack([fit['forecast_fit'], fit['outcome_fit']])[evaluation]
                max_solver = max(max_solver, float(np.max(np.abs(dense-sparse))))
            masks.append(middle)
            products_c.append(c[middle])
            products_d.append(d[middle])
            terms.append(term[middle])
            resolutions.append({'model': name, 'q': q, 'weight': float(weights[len(terms)-1]),
                'rows': int(middle.sum()), 'weeks': 17, 'mean_complementary': float(c[middle].mean()),
                'mean_directional': float(d[middle].mean()), 'mean_change': float((d[middle]-c[middle]).mean()),
                **{label: float(term[middle, j].mean()) for j, label in enumerate(TERMS)},
                'identity_max_absolute_error': error})
        assert max_solver < 1e-7
        assert all(np.array_equal(masks[0], m) for m in masks[1:])
        period = frame.loc[middle, 'period'].reset_index(drop=True)
        c = np.column_stack(products_c)@weights
        d = np.column_stack(products_d)@weights
        # dimensions: row x q x term. Weight q while preserving3 terms.
        combined_terms = np.einsum('nqt,q->nt', np.stack(terms, axis=1), weights)
        np.testing.assert_allclose(d-c, combined_terms.sum(axis=1), atol=1e-12, rtol=1e-10)
        components = np.column_stack([c, d, combined_terms])
        cov = hac(components, period)
        means = components.mean(axis=0)
        labels = ['complementary_product', 'directional_product']+TERMS
        for j, left in enumerate(labels):
            for k, right in enumerate(labels):
                covariance_rows.append({'model': name, 'left': left, 'right': right, 'hac_covariance': float(cov[j, k])})
        all_terms = np.column_stack(terms)
        full_cov = hac(all_terms, period)
        for j in range(15):
            for k in range(15):
                cross_resolution.append({'model': name, 'left_q': QS[j//3], 'left_term': TERMS[j%3],
                    'right_q': QS[k//3], 'right_term': TERMS[k%3], 'hac_covariance': float(full_cov[j, k])})
        mapping = np.kron(weights.reshape(1, -1), np.eye(3))
        mapped = mapping@full_cov@mapping.T
        np.testing.assert_allclose(mapped, cov[2:, 2:], atol=1e-18, rtol=1e-10)
        change_se = float(np.sqrt(np.ones(3)@cov[2:, 2:]@np.ones(3)))
        direct_change_se = direction.score_summary(d-c, period, 'W-FRI', 2)['standard_error']
        assert np.isclose(change_se, direct_change_se, rtol=1e-10, atol=1e-14)
        replay = {}
        for configuration, values in [('complementary_middle', c), ('directional_middle', d)]:
            computed = direction.score_summary(values, period, 'W-FRI', 2)
            old = original[original.configuration == configuration].iloc[0]
            for field in ['mean_product', 'standard_error', 'statistic']:
                assert np.isclose(computed[field], old[field], rtol=1e-8, atol=1e-11), (name, configuration, field, computed[field], old[field])
            replay[configuration] = {'mean_recomputed': computed['mean_product'], 'mean_published': float(old.mean_product),
                'se_recomputed': computed['standard_error'], 'se_published': float(old.standard_error),
                'statistic_recomputed': computed['statistic'], 'statistic_published': float(old.statistic)}
        row = {'model': name, 'rows': len(c), 'weeks': 17, 'entities': int(frame.entity.nunique()),
            'cohort_sha256': cohort[name], 'mean_complementary': float(means[0]), 'mean_directional': float(means[1]),
            'mean_change': float((d-c).mean()), **{label: float(means[j+2]) for j, label in enumerate(TERMS)},
            'sum_terms': float(means[2:].sum()), 'se_complementary': float(np.sqrt(cov[0, 0])),
            'se_directional': float(np.sqrt(cov[1, 1])), 'se_change_with_all_cross_covariances': change_se,
            **{label+'_se': float(np.sqrt(cov[j+2, j+2])) for j, label in enumerate(TERMS)},
            'dominant_absolute_term': TERMS[int(np.argmax(np.abs(means[2:])))],
            'middle_cohort_sha256': cli.cohort_sha256(frame.loc[middle])}
        summaries.append(row)
        aggregate = pd.DataFrame(components, columns=labels).assign(period=period)
        for week, part in aggregate.groupby('period', sort=True):
            weeks.append({'model': name, 'period': str(week), 'rows': len(part),
                          **{label: float(part[label].mean()) for label in labels}})
        verification.append({'model': name, 'input_sha256_matches_original_comparison': True,
            'full_cohort_sha256_matches_original_comparison': True, 'middle_rows': len(c), 'middle_weeks': 17,
            'source_entity_count': int(frame.entity.nunique()), 'row_identity_max_error': max_identity,
            'complementary_dense_sparse_max_error': max_solver, 'mean_and_se_replay': replay,
            'covariance_cross_resolution_mapping_verified': True,
            'change_se_from_terms': change_se, 'change_se_from_direct_difference': direct_change_se,
            'no_middle_boundary_fallback_or_unseen_entity': True})
        print(json.dumps(row))
    for filename, rows in [('summary.csv', summaries), ('per_resolution.csv', resolutions),
                           ('weekly_aggregates.csv', weeks), ('combined_hac_covariance.csv', covariance_rows),
                           ('cross_resolution_term_hac_covariance.csv', cross_resolution)]:
        pd.DataFrame(rows).to_csv(out/filename, index=False, float_format='%.17g')
    dump(out/'verification.json', {'passed': True, 'checks': verification,
        'interpretation': 'Exact empirical accounting and replay, not population feedback identification.'})
    dump(out/'receipt.json', {'completed_utc': utc(), 'protocol_sha256': sha(out/'protocol.json'),
        'script_sha256': sha(__file__), 'input_sha256': source_hashes,
        'files_sha256': {p.name: sha(p) for p in sorted(out.glob('*')) if p.is_file() and p.name != 'receipt.json'},
        'public_observation_rows_exported': False, 'private_sources_accessed': False,
        'published_means_and_standard_errors_preserved': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--price-calendar-input', type=Path, required=True,
                        help='Authorized model_03.parquet with the recorded source SHA-256')
    parser.add_argument('--croston-input', type=Path, required=True,
                        help='Authorized model_08.parquet with the recorded source SHA-256')
    parser.add_argument('--output', type=Path, required=True, help='Fresh destination directory')
    parser.add_argument('--analysis-dir', type=Path, default=PACKAGE/'scripts/analysis')
    parser.add_argument('--comparison', type=Path,
                        default=PACKAGE/'results/directional_comparison/model_comparison.csv')
    parser.add_argument('--comparison-receipt', type=Path,
                        default=PACKAGE/'results/directional_comparison/receipt.json')
    args = parser.parse_args()
    for path in (args.price_calendar_input, args.croston_input, args.comparison, args.comparison_receipt):
        if not path.is_file():
            parser.error('Required input file does not exist: '+str(path))
    inputs = {'model_03.parquet': args.price_calendar_input, 'model_08.parquet': args.croston_input}
    with threadpool_limits(limits=1):
        run(args.output, inputs, args.analysis_dir, args.comparison, args.comparison_receipt)
