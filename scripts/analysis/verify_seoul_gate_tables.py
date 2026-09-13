#!/usr/bin/env python3
"""Read-only verification of six Seoul gate tables; no sklearn or model loading."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(directory):
    return {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*') if p.is_file()}


def adjusted(values, harmonic=True):
    values = np.asarray(values, float)
    order = np.argsort(values, kind='stable')
    factor = len(values) * (sum(1 / i for i in range(1, len(values) + 1)) if harmonic else 1.)
    scaled = values[order] * factor / np.arange(1, len(values) + 1)
    out = np.empty(len(values))
    out[order] = np.minimum(1., np.minimum.accumulate(scaled[::-1])[::-1])
    return out


def close(a, b, label):
    assert np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-9, atol=1e-8), label


def choice(rows, eligible, metric='mse_augment_select'):
    out = {}
    for base, group in rows.groupby('baseline', sort=False):
        allowed = group[np.asarray(eligible)[group.index]]
        best = allowed.sort_values(metric, kind='stable').iloc[0] if len(allowed) else None
        out[base] = best.candidate if best is not None and best[metric] < group.iloc[0].mse_baseline_select else 'baseline'
    return out


def verify(gate_dir, original_dir):
    gate_dir, original_dir = Path(gate_dir).resolve(), Path(original_dir).resolve()
    before_gate, before_source = snapshot(gate_dir), snapshot(original_dir)
    manifest = json.loads((gate_dir / 'MANIFEST.json').read_text())['files']
    names = ['gate_candidates.csv', 'selection_weekly_moments.csv', 'candidate_losses.csv',
             'predeclared_selector_comparison.csv', 'selected_opportunity_loss.csv', 'posthoc_gate_arithmetic.csv']
    for name in names:
        entry = manifest[name]
        assert sha(gate_dir / name) == entry['sha256']
        assert (gate_dir / name).stat().st_size == entry['bytes']
    provenance = json.loads((gate_dir / 'verification.json').read_text())
    for name, value in provenance['input_sha256'].items():
        assert sha(original_dir / name) == value, ('original input hash', name)
    rows, weeks, losses, selected, opportunities, arithmetic = [pd.read_csv(gate_dir / name) for name in names]
    assert len(rows) == len(losses) == 16 and len(weeks) == 128 and len(arithmetic) == 112
    assert len(selected) == 6 and len(opportunities) == 2
    original = pd.read_csv(original_dir / 'selection_all_candidates.csv')
    frozen = json.loads((original_dir / 'selection.json').read_text())
    metrics = pd.read_csv(original_dir / 'confirmation_metrics.csv').set_index(['baseline', 'method'])
    keys = ['baseline', 'candidate']
    assert rows[keys].equals(original[keys])
    numeric = ['theta_select', 'v_select', 't', 'alpha', 'p_raw', 'moment_se', 'blocks', 'predicted_gain',
               'mse_baseline_select', 'mse_augment_select', 'mse_convex_select', 'p_by']
    close(rows[numeric], original[numeric], 'original selection columns')
    assert rows.guard.equals(original.guard) and rows.gate.equals(original.gate)
    for row in rows.itertuples():
        weekly = weeks[(weeks.baseline == row.baseline) & (weeks.candidate == row.candidate)].sort_values('week')
        assert np.array_equal(weekly.week, np.arange(1, 9))
        w = weekly.residual_product_mean.to_numpy()
        centered = w - w.mean()
        gamma0, gamma1 = np.mean(centered ** 2), np.dot(centered[:-1], centered[1:]) / 8
        se = np.sqrt(max(0., (gamma0 + gamma1) / 7))
        iid_se = np.sqrt(gamma0 / 7)
        p = float(student_t.sf(w.mean() / se, 7)) if se > 1e-12 else 1.
        close([w.mean(), gamma0, gamma1, se, iid_se],
              [row.theta_select, row.gamma0, row.gamma1, row.moment_se, row.se_iid_weeks], 'weekly moments/HAC')
        close(p, row.p_nominal_before_guard, 'unprotected nominal p')
        close(1. if row.guard else p, row.p_raw, 'guarded p')
        close(weekly.residual_second_moment.mean(), row.v_select, 'weekly residual variance')
        close(weekly.frozen_augmentation_gain.mean(), row.predicted_gain, 'weekly selection gain')
        close(0. if row.guard else max(row.theta_select, 0.) / row.v_select, row.t, 'coefficient')
        close(w, [getattr(row, f'week_{j}_mean') for j in range(1, 9)], 'wide weekly table')
        assert row.guard == (row.guard_raw_baseline_copy or row.guard_zero_residual_variance)
    by = adjusted(rows.p_raw)
    close(by, rows.p_by, 'BY values')
    order = np.argsort(rows.p_raw.to_numpy(), kind='stable')
    factor = 16 * sum(1 / j for j in range(1, 17))
    close(rows.iloc[order].by_order, np.arange(1, 17), 'BY order')
    close(rows.iloc[order].by_rank_threshold_q05, .05 * np.arange(1, 17) / factor, 'BY thresholds')
    assert np.array_equal((by <= .05) & (rows.theta_select > 0) & ~rows.guard, rows.gate)
    for row in losses.itertuples():
        baseline = metrics.loc[(row.baseline, 'baseline'), 'mse']
        close(baseline, row.mse_baseline_confirm, 'baseline confirmation')
        for method in ['direct', 'convex', 'augment', 'gated']:
            close(metrics.loc[(row.baseline, method + '__' + row.candidate), 'mse'],
                  getattr(row, 'mse_' + method + '_confirm'), 'candidate confirmation loss')
        gain = baseline - row.mse_augment_confirm
        close(gain, row.frozen_augmentation_gain_confirm, 'augmentation gain')
        close(gain, 2 * row.t_frozen * row.theta_confirm - row.t_frozen ** 2 * row.v_confirm, 'quadratic identity')
        close(max(gain, 0.) if not row.gate else 0., row.candidate_level_missed_gain, 'missed candidate benefit')
        close(max(-gain, 0.) if not row.gate else 0., row.candidate_level_avoided_harm, 'avoided candidate harm')
    for strategy in ['augment', 'convex', 'gated']:
        chosen = choice(rows, rows.gate if strategy == 'gated' else np.ones(16, bool),
                        'mse_convex_select' if strategy == 'convex' else 'mse_augment_select')
        for base, candidate in chosen.items():
            assert frozen['choices'][base][strategy] == candidate
            saved = selected[(selected.baseline == base) & (selected.strategy == strategy)].iloc[0]
            assert saved.selected_candidate == candidate
            close(saved.confirmation_mse, metrics.loc[(base, 'selected_' + strategy), 'mse'], 'selected MSE')
    for row in opportunities.itertuples():
        group = losses[losses.baseline == row.baseline]
        baseline = float(group.iloc[0].mse_baseline_confirm)
        best = min(baseline, float(group.mse_augment_confirm.min()))
        gated = metrics.loc[(row.baseline, 'selected_gated'), 'mse']
        ungated = metrics.loc[(row.baseline, 'selected_augment'), 'mse']
        convex = metrics.loc[(row.baseline, 'selected_convex'), 'mse']
        close([gated - ungated, gated - convex, best, gated - best, ungated - best,
               group.candidate_level_missed_gain.sum(), group.candidate_level_avoided_harm.sum()],
              [row.gate_minus_ungated_mse, row.gate_minus_convex_mse, row.best_frozen_augmentation_mse,
               row.selected_gate_ex_post_opportunity_loss, row.selected_ungated_ex_post_opportunity_loss,
               row.candidate_level_sum_missed_gain_nonadditive, row.candidate_level_sum_avoided_harm_nonadditive],
              'selector opportunity accounting')
        assert row.guarded_zero_augmentation_count == int(group.guard.sum())
        assert row.blocked_beneficial_candidate_count == int(group.realized_gain_disposition.eq('blocked_beneficial').sum())
        assert row.blocked_harmful_candidate_count == int(group.realized_gain_disposition.eq('blocked_harmful').sum())
    q11, q8 = np.ones(16), np.ones(16)
    q11[~rows.guard] = adjusted(rows.loc[~rows.guard, 'p_raw'])
    for _, group in rows.groupby('baseline', sort=False):
        q8[group.index] = adjusted(group.p_raw)
    rules = {'original_full16_BY': by, 'nominal_p05_without_family_adjustment': rows.p_raw.to_numpy(),
             'BH_full16_without_harmonic_penalty': adjusted(rows.p_raw, False),
             'BY_drop_five_guards_from_family': q11, 'BY_separate_families_of_eight': q8,
             'BY_full16_iid_week_t7': adjusted(np.where(rows.guard, 1., rows.p_iid_weeks_t7)),
             'BY_full16_normal_same_hac_se': adjusted(np.where(rows.guard, 1., rows.p_normal_same_hac_se))}
    for rule, values in rules.items():
        table = arithmetic[arithmetic.rule == rule].reset_index(drop=True)
        assert len(table) == 16 and table[keys].equals(rows[keys])
        close(values, table.adjusted_or_nominal_p, 'posthoc arithmetic')
        eligible = (values <= .05) & (rows.theta_select > 0) & ~rows.guard
        assert np.array_equal(eligible, table.eligible)
        choices = choice(rows, eligible)
        assert all(r.resulting_choice == choices[r.baseline] for r in table.itertuples())
        expected = 'original' if rule == 'original_full16_BY' else 'posthoc_arithmetic_only_not_new_confirmation'
        assert table.status.eq(expected).all()
    close(np.where(rows.se_iid_weeks > 1e-12, student_t.sf(rows.theta_select / rows.se_iid_weeks, 7), 1.),
          rows.p_iid_weeks_t7, 'iid-week diagnostic p')
    close(np.where(rows.moment_se > 1e-12, ndtr(-rows.theta_select / rows.moment_se), 1.),
          rows.p_normal_same_hac_se, 'normal-tail diagnostic p')
    assert before_gate == snapshot(gate_dir) and before_source == snapshot(original_dir)
    return dict(passed=True, read_only=True, sklearn_or_frozen_models_loaded=False,
                manifest_tables_checked=6, original_input_hashes_checked=len(provenance['input_sha256']),
                candidates=16, weekly_moments=128, original_selectors=6, posthoc_arithmetic_rows=112,
                scope='Saved-table arithmetic and provenance verification only; no new confirmation, temporal calibration or prediction refit.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate-dir', type=Path, default=ROOT / 'results/seoul_gate_diagnosis')
    parser.add_argument('--original-dir', type=Path, default=ROOT / 'results/seoul_confirmation')
    parser.add_argument('--output', type=Path, help='Optional new JSON file; refuse overwrite')
    args = parser.parse_args()
    result = verify(args.gate_dir, args.original_dir)
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2))
