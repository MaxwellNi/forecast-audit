#!/usr/bin/env python3
"""Read-only default forensic replay of the exposed, frozen rental-demand study.

No model fitting, coefficient selection, study execution, data download, or
independent confirmation occurs. CLI execution is strictly read-only.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
DEFAULT_STUDY = HERE.parent / 'seoul_confirmation'


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file()}


def adjusted(p, harmonic=True):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind='stable')
    factor = len(p) * (sum(1 / j for j in range(1, len(p) + 1)) if harmonic else 1.)
    scaled = p[order] * factor / np.arange(1, len(p) + 1)
    q = np.empty(len(p))
    q[order] = np.minimum(1., np.minimum.accumulate(scaled[::-1])[::-1])
    return q


def block_statistics(moment):
    weeks = np.asarray(moment).reshape(8, 168).mean(axis=1)
    centered = weeks - weeks.mean()
    gamma0 = np.dot(centered, centered) / 8
    gamma1 = np.dot(centered[:-1], centered[1:]) / 8
    se_hac = np.sqrt(max(0., (gamma0 + gamma1) / 7))
    se_iid = np.sqrt(gamma0 / 7)
    statistic = weeks.mean() / se_hac if se_hac > 1e-12 else None
    nominal = float(student_t.sf(statistic, 7)) if statistic is not None else 1.
    return weeks, dict(gamma0=float(gamma0), gamma1=float(gamma1),
                      se_hac=float(se_hac), se_iid_weeks=float(se_iid),
                      t_statistic=statistic, p_nominal_before_guard=nominal,
                      p_iid_weeks_t7=float(student_t.sf(weeks.mean() / se_iid, 7))
                      if se_iid > 1e-12 else 1.,
                      p_normal_same_hac_se=float(norm.sf(statistic))
                      if statistic is not None else 1.)


def choose(rows, eligibility, metric='mse_augment_select'):
    choices = {}
    for baseline, group in rows.groupby('baseline', sort=False):
        eligible = group[np.asarray(eligibility)[group.index]]
        best = eligible.sort_values(metric, kind='stable').iloc[0] if len(eligible) else None
        choices[baseline] = (best.candidate if best is not None and
                             best[metric] < group.iloc[0].mse_baseline_select else 'baseline')
    return choices


def main(study_dir=DEFAULT_STUDY, write=False):
    source = Path(study_dir).resolve()
    assert source != HERE and not HERE.is_relative_to(source)
    original = hashes(source)
    spec = importlib.util.spec_from_file_location('seoul_frozen_study', source / 'study.py')
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    frozen = json.loads((source / 'selection.json').read_text())
    protocol = json.loads((source / 'protocol.json').read_text())
    models, fits = pickle.loads((source / 'fitted_models.pkl').read_bytes())
    frame = study.table(source)
    panels, _ = study.build(frame, models, fits)
    selection_mask = study.mask(frame, 'select')
    confirmation_mask = study.mask(frame, 'confirm')
    assert selection_mask.sum() == 1344 and confirmation_mask.sum() == 2016
    dates = frame.date[selection_mask].reset_index(drop=True)
    ys = frame.y.to_numpy()[selection_mask]
    yc = frame.y.to_numpy()[confirmation_mask]
    rows = pd.DataFrame(frozen['rows'])
    assert len(rows) == 16 and rows.blocks.eq(8).all()
    recorded = pd.read_csv(source / 'confirmation_metrics.csv').set_index(['baseline', 'method'])
    candidate_rows, weekly_rows, loss_rows, predictions = [], [], [], {}
    largest_error = 0.
    for index, row in rows.iterrows():
        panel = panels[row.baseline]
        candidate = panel['candidates'][row.candidate]
        my_s, my_c = panel['mY'][selection_mask], panel['mY'][confirmation_mask]
        rx_s, rx_c = candidate['rX'][selection_mask], candidate['rX'][confirmation_mask]
        x_s, x_c = candidate['x'][selection_mask], candidate['x'][confirmation_mask]
        moment = rx_s * (ys - my_s)
        weeks, stats = block_statistics(moment)
        raw_copy = bool(np.allclose(x_s, panel['b'][selection_mask], rtol=1e-12, atol=1e-8))
        zero_variance = bool(np.mean(rx_s ** 2) <= 1e-12)
        assert row.guard == (raw_copy or zero_variance)
        theta, variance = float(np.mean(moment)), float(np.mean(rx_s ** 2))
        coefficient = max(theta, 0.) / variance if not row.guard else 0.
        delta = x_s - my_s
        alpha = float(np.clip(np.mean(delta * (ys - my_s)) / np.mean(delta ** 2), 0, 1))
        reconstructed = np.array([theta, variance, stats['se_hac'], coefficient, alpha])
        saved = row[['theta_select', 'v_select', 'moment_se', 't', 'alpha']].to_numpy(float)
        assert np.allclose(reconstructed, saved, rtol=1e-11, atol=1e-7)
        largest_error = max(largest_error, float(np.max(abs(reconstructed - saved))))
        assert np.isclose(row.p_raw, 1. if row.guard else stats['p_nominal_before_guard'],
                          rtol=1e-11, atol=1e-13)
        detail = {**row.to_dict(), **stats, 'family_original_index': index + 1,
                  'guard_raw_baseline_copy': raw_copy, 'guard_zero_residual_variance': zero_variance,
                  'positive_weeks': int(sum(weeks > 0))}
        for week, value in enumerate(weeks):
            detail[f'week_{week + 1}_mean'] = float(value)
            weekly_rows.append(dict(baseline=row.baseline, candidate=row.candidate,
                                    week=week + 1, start=str(dates.iloc[week * 168]),
                                    end=str(dates.iloc[(week + 1) * 168 - 1]),
                                    residual_product_mean=float(value),
                                    residual_second_moment=float(np.mean(rx_s.reshape(8, 168)[week] ** 2)),
                                    frozen_augmentation_gain=float(np.mean(
                                        2 * row.t * moment.reshape(8, 168)[week] -
                                        row.t ** 2 * rx_s.reshape(8, 168)[week] ** 2))))
        candidate_rows.append(detail)
        if row.baseline not in predictions:
            predictions[row.baseline] = {'baseline': my_c, 'raw_baseline': panel['b'][confirmation_mask]}
        methods = {'direct': x_c, 'convex': my_c + row.alpha * (x_c - my_c),
                   'augment': my_c + row.t * rx_c,
                   'gated': my_c + (row.t * rx_c if row.gate else 0.)}
        loss = dict(baseline=row.baseline, candidate=row.candidate, guard=bool(row.guard),
                    gate=bool(row.gate), t_frozen=row.t, alpha_frozen=row.alpha,
                    mse_baseline_select=float(np.mean((ys - my_s) ** 2)),
                    mse_direct_select=float(np.mean((ys - x_s) ** 2)),
                    mse_augment_select=float(np.mean((ys - my_s - row.t * rx_s) ** 2)),
                    mse_convex_select=float(np.mean((ys - my_s - row.alpha * (x_s - my_s)) ** 2)),
                    mse_baseline_confirm=float(np.mean((yc - my_c) ** 2)),
                    theta_confirm=float(np.mean(rx_c * (yc - my_c))),
                    v_confirm=float(np.mean(rx_c ** 2)))
        for strategy, prediction in methods.items():
            key = strategy + '__' + row.candidate
            predictions[row.baseline][key] = prediction
            mse = float(np.mean((yc - prediction) ** 2))
            assert np.isclose(mse, recorded.loc[(row.baseline, key), 'mse'], rtol=1e-12, atol=1e-8)
            loss[f'mse_{strategy}_confirm'] = mse
        gain = loss['mse_baseline_confirm'] - loss['mse_augment_confirm']
        identity = 2 * row.t * loss['theta_confirm'] - row.t ** 2 * loss['v_confirm']
        assert np.isclose(gain, identity, rtol=1e-11, atol=1e-7)
        loss.update(frozen_augmentation_gain_confirm=gain,
                    quadratic_identity=identity,
                    candidate_level_missed_gain=max(gain, 0.) if not row.gate else 0.,
                    candidate_level_avoided_harm=max(-gain, 0.) if not row.gate else 0.,
                    realized_gain_disposition=('guard_zero_augmentation' if row.guard else
                    'retained_beneficial' if row.gate and gain > 0 else
                    'retained_harmful' if row.gate and gain < 0 else
                    'blocked_beneficial' if gain > 1e-7 else
                    'blocked_harmful' if gain < -1e-7 else 'blocked_zero_coefficient'))
        loss_rows.append(loss)
    details = pd.DataFrame(candidate_rows)
    harmonic = sum(1 / j for j in range(1, 17))
    order = np.argsort(rows.p_raw.to_numpy(), kind='stable')
    by = adjusted(rows.p_raw)
    assert np.allclose(by, rows.p_by, rtol=1e-13, atol=1e-13)
    for rank, index in enumerate(order, 1):
        scaled = [(rows.iloc[later].p_raw * 16 * harmonic / later_rank, later_rank)
                  for later_rank, later in enumerate(order, 1) if later_rank >= rank]
        best_q, best_rank = min(scaled)
        details.loc[index, 'by_order'] = rank
        details.loc[index, 'by_rank_threshold_q05'] = .05 * rank / (16 * harmonic)
        details.loc[index, 'by_adjustment_source_order'] = best_rank if best_q < 1 else np.nan
        details.loc[index, 'by_rank_critical_t7'] = student_t.isf(.05 * rank / (16 * harmonic), 7)
    details['nominal_p05_pass'] = ~details.guard & (details.theta_select > 0) & (details.p_raw <= .05)
    details['gate_reason'] = np.select(
        [details.guard, details.theta_select <= 0, details.p_raw > .05, details.p_by > .05],
        ['exact_copy_and_zero_variance', 'nonpositive_selection_moment',
         'fails_nominal_p05', 'passes_nominal_p05_but_fails_full16_BY'], default='retained')
    assert np.array_equal((by <= .05) & (rows.theta_select > 0) & ~rows.guard, rows.gate)
    selected_rows, opportunity = [], []
    for strategy, metric in [('augment', 'mse_augment_select'), ('convex', 'mse_convex_select'),
                             ('gated', 'mse_augment_select')]:
        eligibility = rows.gate if strategy == 'gated' else np.ones(16, dtype=bool)
        choices = choose(rows, eligibility, metric)
        for baseline, candidate in choices.items():
            assert candidate == frozen['choices'][baseline][strategy]
            name = 'baseline' if candidate == 'baseline' else strategy + '__' + candidate
            prediction = predictions[baseline][name]
            predictions[baseline]['selected_' + strategy] = prediction
            mse = float(np.mean((yc - prediction) ** 2))
            assert np.isclose(mse, recorded.loc[(baseline, 'selected_' + strategy), 'mse'], atol=1e-8)
            selected_rows.append(dict(baseline=baseline, strategy=strategy, selected_candidate=candidate,
                                      confirmation_mse=mse,
                                      confirmation_gain_vs_adjusted_baseline=float(np.mean(
                                          (yc - predictions[baseline]['baseline']) ** 2)) - mse,
                                      status='original_predeclared_and_frozen'))
    loss_table = pd.DataFrame(loss_rows)
    for baseline, panel in predictions.items():
        group = loss_table[loss_table.baseline == baseline]
        base_mse = group.iloc[0].mse_baseline_confirm
        available = {'baseline': base_mse, **dict(zip(group.candidate, group.mse_augment_confirm))}
        best = min(available, key=available.get)
        gated_mse = float(np.mean((yc - panel['selected_gated']) ** 2))
        ungated_mse = float(np.mean((yc - panel['selected_augment']) ** 2))
        convex_mse = float(np.mean((yc - panel['selected_convex']) ** 2))
        opportunity.append(dict(baseline=baseline,
                                gate_minus_ungated_mse=gated_mse - ungated_mse,
                                gate_minus_convex_mse=gated_mse - convex_mse,
                                best_frozen_augmentation_ex_post=best,
                                best_frozen_augmentation_mse=available[best],
                                selected_gate_ex_post_opportunity_loss=gated_mse - available[best],
                                selected_ungated_ex_post_opportunity_loss=ungated_mse - available[best],
                                candidate_level_sum_missed_gain_nonadditive=float(group.candidate_level_missed_gain.sum()),
                                candidate_level_sum_avoided_harm_nonadditive=float(group.candidate_level_avoided_harm.sum()),
                                blocked_beneficial_candidate_count=int(group.realized_gain_disposition.eq('blocked_beneficial').sum()),
                                blocked_harmful_candidate_count=int(group.realized_gain_disposition.eq('blocked_harmful').sum()),
                                guarded_zero_augmentation_count=int(group.guard.sum()),
                                max_absolute_selected_gate_minus_ungated_prediction=float(np.max(abs(
                                    panel['selected_gated'] - panel['selected_augment'])))))
    # These replay already observed p values; none is a new endorsed selector.
    nonguarded = ~rows.guard
    q11 = np.ones(16)
    q11[nonguarded] = adjusted(rows.loc[nonguarded, 'p_raw'])
    q8 = np.ones(16)
    for _, group in rows.groupby('baseline', sort=False):
        q8[group.index] = adjusted(group.p_raw)
    rules = {'original_full16_BY': by, 'nominal_p05_without_family_adjustment': rows.p_raw.to_numpy(),
             'BH_full16_without_harmonic_penalty': adjusted(rows.p_raw, harmonic=False),
             'BY_drop_five_guards_from_family': q11, 'BY_separate_families_of_eight': q8,
             'BY_full16_iid_week_t7': adjusted(np.where(rows.guard, 1., details.p_iid_weeks_t7)),
             'BY_full16_normal_same_hac_se': adjusted(np.where(rows.guard, 1., details.p_normal_same_hac_se))}
    arithmetic = []
    for rule, values in rules.items():
        eligible = (values <= .05) & (rows.theta_select > 0) & ~rows.guard
        choices = choose(rows, eligible)
        for index, row in rows.iterrows():
            arithmetic.append(dict(rule=rule, baseline=row.baseline, candidate=row.candidate,
                                   adjusted_or_nominal_p=float(values[index]), eligible=bool(eligible[index]),
                                   resulting_choice=choices[row.baseline],
                                   status='original' if rule == 'original_full16_BY' else
                                   'posthoc_arithmetic_only_not_new_confirmation'))
    assert original == hashes(source), 'The released study changed during the audit'
    summary = dict(passed=True, source_directory=str(source),
                   scope='Retrospective deterministic reconstruction on the previously evaluated rental data. No new independent confirmation or calibrated temporal inference.',
                   source_files_unchanged=True, input_file_count=len(original),
                   selection_hours=1344, selection_weekly_blocks=8, selection_df=7,
                   confirmation_hours=2016, confirmation_weekly_blocks=12, confirmation_df=11,
                   confirmation_fortnight_blocks=6, confirmation_fortnight_df=5,
                   family_size=16, harmonic_H16=harmonic, family_factor=16 * harmonic,
                   guarded_members=int(rows.guard.sum()), retained_members=int(rows.gate.sum()),
                   largest_reconstruction_absolute_error=largest_error,
                   selected_opportunity_accounting=opportunity,
                   windows=protocol['windows_inclusive'],
                   input_sha256=original)
    outputs = {'gate_candidates.csv': details, 'selection_weekly_moments.csv': pd.DataFrame(weekly_rows),
               'candidate_losses.csv': loss_table, 'predeclared_selector_comparison.csv': pd.DataFrame(selected_rows),
               'selected_opportunity_loss.csv': pd.DataFrame(opportunity),
               'posthoc_gate_arithmetic.csv': pd.DataFrame(arithmetic)}
    if write:
        for name, table in outputs.items():
            table.to_csv(HERE / name, index=False, float_format='%.17g')
        (HERE / 'verification.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'input_sha256'}, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=DEFAULT_STUDY)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.study_dir, False)
