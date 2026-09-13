"""Audit raw forecasts with directional entity means on a common panel.

Input columns: model, entity, period, prediction, y. Periods are consecutive
integer observation indices. Every model must contain exactly the same
complete entity-by-period panel and identical outcomes. Forecasts are inputs;
this program does not train prospective forecasting models.

Inference assumes independent entity trajectories and the predictability and
innovation conditions stated in the paper. Input validation cannot establish
these scientific assumptions. Normal and Student reference probabilities
are asymptotic choices, not exact finite-sample guarantees.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t


REQUIRED_COLUMNS = ('model', 'entity', 'period', 'prediction', 'y')
SE_FLOOR = 1e-10


def load_panel(path):
    """Read identifiers and period indices without changing their spelling."""
    return pd.read_csv(path, dtype={'model': str, 'entity': str, 'period': str},
                       keep_default_na=False)


def validate_panel(frame, folds=5):
    """Return aligned model matrices after strict common-cohort checks."""
    if not isinstance(folds, (int, np.integer)) or isinstance(folds, bool) or folds < 3:
        raise ValueError('Use an integer fold count of at least three.')
    if len(frame) == 0 or not set(REQUIRED_COLUMNS).issubset(frame.columns):
        raise ValueError('Input requires nonempty model, entity, period, prediction, and y columns.')
    data = frame.loc[:, REQUIRED_COLUMNS].copy()
    for column in ('model', 'entity'):
        if data[column].isna().any():
            raise ValueError(f'{column} identifiers must be nonempty.')
        data[column] = data[column].astype(str)
        if data[column].str.strip().eq('').any():
            raise ValueError(f'{column} identifiers must be nonempty.')
        if data[column].ne(data[column].str.strip()).any():
            raise ValueError(f'{column} identifiers may not have surrounding whitespace.')
    period_text = data['period'].astype(str)
    if not period_text.map(lambda value: re.fullmatch(r'[+-]?\d+', value) is not None).all():
        raise ValueError('Periods must be consecutive integer observation indices, not dates or fractional times.')
    data['period'] = period_text.map(int)
    for column in ('prediction', 'y'):
        try:
            numeric = pd.to_numeric(data[column], errors='raise')
            if np.iscomplexobj(numeric.to_numpy()):
                raise ValueError('Complex values are not raw real-valued forecasts.')
            data[column] = numeric.astype(float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f'{column} values must be finite real numbers.') from exc
        if not np.isfinite(data[column].to_numpy()).all():
            raise ValueError(f'{column} values must be finite real numbers.')
    if data.duplicated(['model', 'entity', 'period']).any():
        raise ValueError('Duplicate model-entity-period rows are not allowed.')
    entities = sorted(data['entity'].unique())
    periods = sorted(data['period'].unique())
    models = sorted(data['model'].unique())
    if len(entities) < 2:
        raise ValueError('At least two independent entity trajectories are required.')
    if len(periods) < folds:
        raise ValueError('Every declared fold must contain at least one period.')
    if any(right-left != 1 for left, right in zip(periods, periods[1:])):
        raise ValueError('The declared observation index has missing periods; supply a complete consecutive panel.')
    index = pd.MultiIndex.from_product([entities, periods], names=['entity', 'period'])
    matrices = {}
    common_y = None
    for model in models:
        block = data.loc[data['model'].eq(model)].set_index(['entity', 'period'])
        if len(block) != len(index) or not block.index.sort_values().equals(index):
            raise ValueError('Every model must have the same complete entity-by-period cohort; rows are never silently dropped.')
        block = block.reindex(index)
        y = block['y'].to_numpy().reshape(len(entities), len(periods))
        if common_y is None:
            common_y = y
        elif not np.array_equal(y, common_y):
            raise ValueError('Outcomes must be identical across models on every entity-period row.')
        matrices[model] = block['prediction'].to_numpy().reshape(len(entities), len(periods))
    fold_ids = np.empty(len(periods), dtype=int)
    for label, positions in enumerate(np.array_split(np.arange(len(periods)), folds)):
        fold_ids[positions] = label
    return matrices, common_y, fold_ids, entities, periods


def directional_scores(prediction, outcome, fold_ids):
    """Return one equally weighted raw score per entity on the middle folds.

    Earlier folds fit the prediction mean; later folds fit the outcome mean.
    Every included period has both training sides, with no fallback fit.
    """
    x, y = np.asarray(prediction, float), np.asarray(outcome, float)
    fold_ids = np.asarray(fold_ids)
    if x.ndim != 2 or x.shape != y.shape or fold_ids.shape != (x.shape[1],):
        raise ValueError('Prediction, outcome, and fold dimensions must agree.')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Supply finite raw trajectories.')
    if (not np.issubdtype(fold_ids.dtype, np.integer) or
            np.any(np.diff(fold_ids) < 0) or
            len(np.unique(fold_ids)) < 3 or
            not np.array_equal(np.unique(fold_ids), np.arange(len(np.unique(fold_ids))))):
        raise ValueError('Fold identifiers must be contiguous ordered integers starting at zero, with at least three folds.')
    totals = np.zeros(len(x))
    eligible = (fold_ids > fold_ids.min()) & (fold_ids < fold_ids.max())
    try:
        with np.errstate(over='raise', invalid='raise'):
            for label in np.unique(fold_ids)[1:-1]:
                take, earlier, later = fold_ids == label, fold_ids < label, fold_ids > label
                residual_x = x[:, take] - x[:, earlier].mean(axis=1, keepdims=True)
                residual_y = y[:, take] - y[:, later].mean(axis=1, keepdims=True)
                totals += np.sum(residual_x*residual_y, axis=1)
            scores = totals/int(eligible.sum())
    except FloatingPointError as exc:
        raise ValueError('Raw products overflowed; declare and apply a numerically usable analysis scale before auditing.') from exc
    if not np.isfinite(scores).all():
        raise ValueError('The resulting entity scores must be finite.')
    return scores, eligible


def by_adjust(p_values):
    """Benjamini-Yekutieli adjusted probabilities for the complete family."""
    p_values = np.asarray(p_values, float)
    if p_values.ndim != 1 or len(p_values) == 0 or not np.isfinite(p_values).all():
        raise ValueError('Supply a nonempty vector of finite probabilities.')
    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError('Probabilities must lie in [0,1].')
    count = len(p_values)
    order = np.argsort(p_values, kind='stable')
    harmonic = np.sum(1/np.arange(1, count+1))
    ranked = p_values[order]*count*harmonic/np.arange(1, count+1)
    adjusted = np.empty(count)
    adjusted[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return adjusted


def audit_panel(frame, folds=5, alpha=.05, reference='normal'):
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between zero and one.')
    if reference not in ('normal', 't'):
        raise ValueError('reference must be normal or t.')
    matrices, y, fold_ids, entities, periods = validate_panel(frame, folds)
    records = []
    for model, x in matrices.items():
        scores, eligible = directional_scores(x, y, fold_ids)
        try:
            with np.errstate(over='raise', invalid='raise'):
                mean = float(scores.mean())
                se = float(scores.std(ddof=1)/np.sqrt(len(scores)))
        except FloatingPointError as exc:
            raise ValueError('Entity-score moments overflowed; use a declared numerically usable analysis scale.') from exc
        if not np.isfinite(mean) or not np.isfinite(se):
            raise ValueError('Entity-score moments must be finite.')
        constant = bool(np.all(x == x[:, :1]))
        reason = 'constant_prediction_within_entities' if constant else ('standard_error_floor' if se <= SE_FLOOR else '')
        raw_t = mean/se if se > 0 else float('nan')
        if not np.isfinite(raw_t) and se > 0:
            raise ValueError('Studentization overflowed; use a declared numerically usable analysis scale.')
        raw_p = float(norm.sf(raw_t) if reference == 'normal' else student_t.sf(raw_t, len(scores)-1)) if se > 0 else 1.
        records.append(dict(model=model, mean=mean, se=se, statistic=raw_t,
                            p=raw_p, decision_p=1. if reason else raw_p,
                            reason=reason, entities=len(entities), periods=len(periods),
                            input_rows=len(entities)*len(periods),
                            evaluation_rows=len(entities)*int(eligible.sum()),
                            rows_per_entity=int(eligible.sum()),
                            support_fraction=float(eligible.mean()), folds=folds,
                            family_size=len(matrices), reference=reference,
                            reference_df=(len(entities)-1 if reference == 't' else ''),
                            nuisance_fit='earlier_prediction_mean_later_outcome_mean'))
    profile = pd.DataFrame(records)
    profile['BY_adjusted_p'] = by_adjust(profile['decision_p'])
    profile['label'] = np.where(profile['reason'].ne(''), 'ABSTAIN',
                                np.where(profile['BY_adjusted_p'] <= alpha, 'RETAIN', 'NOT_RETAINED'))
    receipt = dict(
        target='Mean across independent entities of within-entity mean directional raw residual products.',
        hypothesis='Positive raw residual association; null centering requires the declared innovation and predictability assumptions.',
        sampling='Independent entity trajectories; dependence within an entity is allowed only under the stated centering and moment conditions.',
        controls='Entity means only. No baseline ranks, extra covariates, additive bins, HAC, or data-selected control class.',
        fitting='Retrospective audit: earlier folds fit prediction means and later folds fit outcome means. Input forecasts are not retrained.',
        scientific_assumptions_verified_from_data=False,
        scientific_conditions=[
            'Outcomes equal an entity effect plus innovations that are conditionally mean zero given the stated past information and forecast noises.',
            'Each input forecast is predictable from that entity effect, forecast noises, and earlier outcomes; future outcomes do not enter forecast construction.',
            'Entity trajectories, including their effects and forecast noises, are independent across entities.',
            'The common panel, fold rule, model family, eligibility and analysis scale are fixed independently of the evaluated innovations.',
            'For the asymptotic reference, fourth moments are uniformly bounded and average entity-score variance tends to a positive limit.'
        ],
        reference=reference,
        reference_scope=('Normal limit as the number of independent entities grows; no exact finite-sample guarantee.' if reference == 'normal' else
                         'Student reference with entities minus one degrees of freedom, reported as an asymptotic sensitivity choice; no exact t or finite-sample guarantee.'),
        multiple_testing='BY includes every input model, including abstentions with decision p=1; its guarantee requires valid marginal inputs.',
        alpha=alpha, standard_error_floor=SE_FLOOR, model_count=len(matrices),
        entities=len(entities), periods=len(periods), folds=folds,
        fold_lengths=np.bincount(fold_ids).tolist(),
        evaluation_fold_indices=list(range(1, folds-1)),
        evaluation_rows_per_entity=int(np.sum((fold_ids > 0) & (fold_ids < folds-1))),
        outcome_alignment='Exact equality across every model on the common complete panel.',
        fallback_rows=0,
        interpretation='RETAIN is a positive test decision under the stated assumptions; NOT_RETAINED does not establish absence of predictive value. ABSTAIN declines numerical interpretation.',
        output_privacy='Only model-level summaries and protocol counts are written; source rows, entity identifiers, period values and predictions are not copied into the output.'
    )
    return profile, receipt


def run(input_path, output_dir, folds=5, alpha=.05, reference='normal'):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('Output directory already exists; choose a new directory to preserve prior results.')
    input_path = Path(input_path)
    input_bytes = input_path.read_bytes()
    profile, receipt = audit_panel(load_panel(io.BytesIO(input_bytes)), folds=folds, alpha=alpha, reference=reference)
    receipt['input_sha256'] = hashlib.sha256(input_bytes).hexdigest()
    receipt['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=False)
    profile.to_csv(output_dir/'profile.csv', index=False)
    (output_dir/'scope.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    return profile, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--reference', choices=('normal', 't'), default='normal')
    args = parser.parse_args()
    try:
        profile, receipt = run(args.input, args.output, args.folds, args.alpha, args.reference)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps({'models': len(profile), 'entities': receipt['entities'],
                      'evaluation_rows_per_entity': receipt['evaluation_rows_per_entity'],
                      'labels': profile['label'].value_counts().to_dict(),
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
