"""Locked, historical public forecasting confirmation; no outcome innovations.

Phases are separate and refuse overwrites. Frozen protocol hashes this file.
Forecast training, nuisance estimation, selection, and confirmation use disjoint
calendar windows. Confirmation lagged observations are available sequentially;
they do not update model weights or any choice. Not an external preregistration.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import scipy
from scipy.stats import t as student_t
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
SEED = 26091273
SOURCE = 'https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip'
WINDOWS = {
    'train': ['2017-12-08', '2018-04-27'],
    'nuisance': ['2018-05-05', '2018-06-29'],
    'select': ['2018-07-07', '2018-08-31'],
    'confirm': ['2018-09-08', '2018-11-30'],
}
LAGS = [1, 2, 3, 24, 25, 48, 168]
CANDIDATES = ['persistence_1h', 'seasonal_24h', 'seasonal_168h',
              'ridge_lags_calendar', 'hist_gradient_boosting',
              'mirror_ridge_control', 'exact_baseline_copy', 'noise_control']
BASELINES = ['seasonal_24h', 'ridge_lags_calendar']


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')


def check(out):
    p = json.loads((out/'protocol.json').read_text())
    assert p['script_sha256'] == sha(__file__), 'Do not edit frozen study code'
    assert (out/'protocol.sha256').read_text().split()[0] == sha(out/'protocol.json')
    return p


def freeze(out):
    assert not (out/'protocol.json').exists(), 'Frozen protocol is immutable'
    assert not (out/'source.zip').exists(), 'Protocol must predate data download'
    out.mkdir(parents=True, exist_ok=True)
    p = {
        'frozen_utc': utc(), 'script_sha256': sha(__file__), 'seed': SEED,
        'source_url': SOURCE,
        'source_page': 'https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand',
        'source_doi': 'https://doi.org/10.24432/C5F62R',
        'attribution': 'Seoul Bike Sharing Demand (2020). UCI Machine Learning Repository.',
        'license': 'CC BY 4.0; https://creativecommons.org/licenses/by/4.0/',
        'scope': 'Prospectively locked reanalysis on newly obtained historical public source. Not third-party validation, not statistically independent time samples, not a new broad cross-domain result. Prior Capital Bike Sharing 2012 support was already revealed and is excluded.',
        'prior_inventory_search': 'No matching dataset name or DOI was found in the prior local inventory before freeze; this records no discovered prior project use, not proof that no human ever saw the data.',
        'source_assertions': 'Exactly 8760 consecutive hourly records, 2017-12-01 through 2018-11-30; any schema/date failure stops the experiment without selecting another favorable source.',
        'windows_inclusive': WINDOWS,
        'embargo': 'Seven whole calendar days between each fitting/selection/confirmation window. Lags may use earlier embargo outcomes as forecast-time information; embargo is not a proof of independence.',
        'forecast_origin': 'One hour ahead, after the previous hour count is observed. Available features: lags 1,2,3,24,25,48,168; rolling past-only means 24 and168; target hour sin/cos first and second harmonics, weekday sin/cos, day-of-year sin/cos, weekend. No weather, holiday indicator, functioning-day indicator, contemporaneous count, random innovation, or future response feature.',
        'model_fit': 'All model weights and feature scaling use train window only; no model refit during nuisance, select, or confirm. Past counts may update lag features sequentially. All rows including operational closure zeros retained.',
        'models': {'ridge': 'StandardScaler then Ridge(alpha=10.0)', 'hgb': 'HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,learning_rate=.05,min_samples_leaf=32,l2_regularization=1,early_stopping=False,random_state=seed)'},
        'baselines': BASELINES, 'candidates_per_baseline': CANDIDATES,
        'controls': 'mirror_ridge=2*raw_baseline-ridge; exact copy=raw_baseline; noise=raw_baseline+N(0,100^2) from fixed stream independent of outcomes. These are prespecified diagnostic controls, not asserted scientific null truths. Some duplicate/copy targets are deliberate; keep all16 family members.',
        'nuisance': 'For each baseline separately, regress outcome and each forecast on intercept, standardized baseline linear term and cubic B-splines (5 quantile knots; linear extrapolation). Knots and all least-squares coefficients fit only nuisance window. Unclipped count-scale predictions preserve the quadratic identity. mY and mX are fitted projections, not known true conditional means.',
        'target': 'Raw rental-count residual-product mean theta_hat=mean[(X-mX(B))*(Y-mY(B))]. Count-squared units. No rank transformation or implication from ranks to raw risk.',
        'selection': 'Fit nonnegative augmentation t=max(theta_hat,0)/mean(rX^2), no upper cap. Also conventional convex blend mY+a*(X-mY), a clipped to[0,1] minimizing select MSE. Guard exact raw baseline copies and numerically zero residual variance with p=1,t=0. Complete-family BY at q=.05 over all16 baseline/candidate pairs. Gate uses positive theta and adjusted p<=.05. Select MSE-minimizing candidate within each strategy (ungated residual augmentation, gated residual augmentation, conventional convex blend); baseline remains an option; ties choose baseline then declared model order. Selection records frozen before confirm labels are evaluated.',
        'gate_inference': 'Eight nonoverlapping seven-day selection blocks. HAC lag1 on block means, Bartlett weight1/2, finite8/7 correction; Student t7 nominal upper tail. BY is arithmetic over dependent temporal diagnostics and does not establish calibrated FDR. Undefined standard error gives p=1.',
        'primary_comparison': 'For each of two baseline adjustments: selected gated augmentation minus selected conventional convex blend in final raw count MSE; also report each vs mY baseline, ungated augmentation and direct trained strong-model predictions. Positive risk improvement means comparator MSE minus method MSE.',
        'metrics': 'All candidate raw/direct, conventional, ungated augmentation and gated augmentation MSE, RMSE and MAE, plus all negative results. Minimum meaningful gain is1% of confirmation mY-baseline MSE; this is a reporting threshold, not a test-tuned choice.',
        'gain_identity': 'Report selection oracle plug-in theta_plus^2/v, selected coefficient t, confirmation theta and v, actual risk improvement2*t*theta_confirm-t^2*v_confirm, and ex-post confirmation oracle for diagnosis only. The selection plug-in is optimistic and is not a forecast guarantee. No confirmation coefficient is deployed.',
        'uncertainty': 'Primary pointwise95% paired intervals from12 nonoverlapping weekly block means with HAC lag1, Bartlett1/2 and12/11 correction, t11. Report six fortnight-block sensitivity with t5 and4000 paired weekly-block percentile bootstrap draws; seed fixed. These are dependence approximations, not exact or joint coverage guarantees. No hourly IID intervals.',
        'confirmation': 'Download occurs only after freeze. Prepare parses train/nuisance/select outcomes through usecols/nrows and builds no confirmation outcomes. Confirm phase checks immutable selection hashes, then computes confirmation predictions from only earlier raw observations and evaluates exactly once. No post-confirm retuning or source exclusion. Script/schema repairs, if needed, must be documented as protocol deviations.',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
    }
    dump(out/'protocol.json', p)
    (out/'protocol.sha256').write_text(sha(out/'protocol.json')+'  protocol.json\n')
    print(json.dumps({'frozen_utc': p['frozen_utc'], 'protocol_sha256': sha(out/'protocol.json')}))


def download(out):
    check(out)
    assert not (out/'source.zip').exists()
    with urllib.request.urlopen(SOURCE, timeout=90) as response:
        data = response.read()
    (out/'source.zip').write_bytes(data)
    dump(out/'source_receipt.json', {'downloaded_utc': utc(), 'source_url': SOURCE,
         'bytes': len(data), 'sha256': sha(out/'source.zip'),
         'protocol_sha256': sha(out/'protocol.json')})
    print('Downloaded source after protocol freeze; no outcomes summarized.')


def table(out, through=None):
    receipt = json.loads((out/'source_receipt.json').read_text())
    assert sha(out/'source.zip') == receipt['sha256']
    with zipfile.ZipFile(out/'source.zip') as z:
        name = next(name for name in z.namelist() if name.endswith('SeoulBikeData.csv'))
        raw = z.read(name)
    meta = pd.read_csv(io.BytesIO(raw), encoding='latin1', usecols=['Date', 'Hour'])
    dates = pd.to_datetime(meta.Date, dayfirst=True)+pd.to_timedelta(meta.Hour, unit='h')
    assert len(dates) == 8760 and dates.iloc[0] == pd.Timestamp('2017-12-01')
    assert dates.iloc[-1] == pd.Timestamp('2018-11-30 23:00')
    assert np.all(np.diff(dates.to_numpy()) == np.timedelta64(1, 'h'))
    n = len(dates) if through is None else int((dates < pd.Timestamp(through)+pd.Timedelta(days=1)).sum())
    y = pd.read_csv(io.BytesIO(raw), encoding='latin1', usecols=['Rented Bike Count'], nrows=n).iloc[:, 0].to_numpy(float)
    assert np.all(np.isfinite(y)) and np.all(y >= 0)
    return pd.DataFrame({'date': dates.iloc[:n], 'y': y})


def features(frame):
    # Every count-based feature is shifted by at least one complete hour.
    y = frame.y
    d = frame.date.dt
    x = pd.DataFrame({f'lag_{lag}': y.shift(lag) for lag in LAGS})
    x['mean_24'] = y.shift(1).rolling(24).mean()
    x['mean_168'] = y.shift(1).rolling(168).mean()
    for name, value, period in [('hour', d.hour, 24), ('hour2', 2*d.hour, 24),
                                ('weekday', d.dayofweek, 7), ('year', d.dayofyear, 365.25)]:
        x[name+'_sin'] = np.sin(2*np.pi*value/period)
        x[name+'_cos'] = np.cos(2*np.pi*value/period)
    x['weekend'] = (d.dayofweek >= 5).astype(float)
    return x.to_numpy()


def mask(frame, window):
    lo, hi = WINDOWS[window]
    return ((frame.date >= pd.Timestamp(lo)) & (frame.date < pd.Timestamp(hi)+pd.Timedelta(days=1))).to_numpy()


def nominal(values, block=168):
    assert len(values) % block == 0
    means = np.asarray(values).reshape(-1, block).mean(axis=1)
    n = len(means)
    centered = means-means.mean()
    gamma0 = np.mean(centered**2)
    gamma1 = np.dot(centered[1:], centered[:-1])/n
    var = max(0., (gamma0+gamma1)/(n-1))
    se = float(np.sqrt(var))
    mu = float(means.mean())
    p = float(student_t.sf(mu/se, n-1)) if se > 1e-12 else 1.
    rad = float(student_t.ppf(.975, n-1)*se)
    return {'mean': mu, 'se': se, 'blocks': n, 'df': n-1, 'p': p,
            'low': mu-rad, 'high': mu+rad}


def nuisance_design(b, center, scale, spline, fit=False):
    z = ((b-center)/scale).reshape(-1, 1)
    s = spline.fit_transform(z) if fit else spline.transform(z)
    return np.column_stack([np.ones(len(b)), z, s])


def fit_training(frame):
    x = features(frame)
    tr = mask(frame, 'train')
    assert np.isfinite(x[tr]).all()
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.))
    hgb = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15,
        learning_rate=.05, min_samples_leaf=32, l2_regularization=1,
        early_stopping=False, random_state=SEED)
    for model in [ridge, hgb]:
        model.fit(x[tr], frame.y.to_numpy()[tr])
    return ridge, hgb


def predict(frame, models):
    x = features(frame)
    safe = np.nan_to_num(x)
    return {'persistence_1h': frame.y.shift(1).to_numpy(),
            'seasonal_24h': frame.y.shift(24).to_numpy(),
            'seasonal_168h': frame.y.shift(168).to_numpy(),
            'ridge_lags_calendar': models[0].predict(safe),
            'hist_gradient_boosting': models[1].predict(safe)}


def build(frame, models, fitted=None):
    pred = predict(frame, models)
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0, 100., len(frame))
    nu = mask(frame, 'nuisance')
    result, fits = {}, {} if fitted is None else fitted
    for baseline in BASELINES:
        b = pred[baseline]
        cand = {**pred, 'mirror_ridge_control': 2*b-pred['ridge_lags_calendar'],
                'exact_baseline_copy': b.copy(), 'noise_control': b+noise}
        if fitted is None:
            center, scale = float(np.mean(b[nu])), float(np.std(b[nu]))
            spline = SplineTransformer(n_knots=5, degree=3, include_bias=False,
                                       knots='quantile', extrapolation='linear')
            hn = nuisance_design(b[nu], center, scale, spline, fit=True)
            coef_y = np.linalg.lstsq(hn, frame.y.to_numpy()[nu], rcond=None)[0]
            coef_x = {k: np.linalg.lstsq(hn, cand[k][nu], rcond=None)[0] for k in CANDIDATES}
            fits[baseline] = (center, scale, spline, coef_y, coef_x)
        center, scale, spline, coef_y, coef_x = fits[baseline]
        h = nuisance_design(np.nan_to_num(b), center, scale, spline)
        result[baseline] = {'mY': h@coef_y, 'b': b, 'candidates': {
            k: {'x': cand[k], 'rX': cand[k]-h@coef_x[k]} for k in CANDIDATES}}
    return result, fits


def prepare(out):
    check(out)
    assert not (out/'selection.json').exists()
    frame = table(out, through=WINDOWS['select'][1])
    models = fit_training(frame)
    panels, fits = build(frame, models)
    se = mask(frame, 'select')
    y = frame.y.to_numpy()[se]
    rows = []
    for base in BASELINES:
        panel = panels[base]
        my = panel['mY'][se]
        for name in CANDIDATES:
            x = panel['candidates'][name]['x'][se]
            rx = panel['candidates'][name]['rX'][se]
            guard = bool(np.allclose(x, panel['b'][se], rtol=1e-12, atol=1e-8))
            v = float(np.mean(rx**2))
            moment = rx*(y-my)
            stats = nominal(moment)
            guard = guard or v <= 1e-12
            theta = stats['mean']
            t = max(theta, 0)/v if not guard else 0.
            delta = x-my
            alpha = float(np.clip(np.mean(delta*(y-my))/np.mean(delta**2), 0, 1)) if np.mean(delta**2) > 1e-12 else 0.
            rows.append({'baseline': base, 'candidate': name, 'guard': guard,
                         'theta_select': theta, 'v_select': v, 't': t, 'alpha': alpha,
                         'p_raw': 1. if guard else stats['p'], 'moment_se': stats['se'],
                         'blocks': stats['blocks'], 'predicted_gain': max(theta, 0)**2/v if not guard else 0.,
                         'mse_baseline_select': float(np.mean((y-my)**2)),
                         'mse_augment_select': float(np.mean((y-(my+t*rx))**2)),
                         'mse_convex_select': float(np.mean((y-(my+alpha*delta))**2))})
    order = np.argsort([row['p_raw'] for row in rows], kind='stable')
    count = len(rows)
    harmonic = sum(1/i for i in range(1, count+1))
    scaled = np.array([rows[j]['p_raw']*count*harmonic/(i+1) for i, j in enumerate(order)])
    adj = np.minimum(1., np.minimum.accumulate(scaled[::-1])[::-1])
    for j, q in zip(order, adj):
        rows[j]['p_by'] = float(q)
        rows[j]['gate'] = bool(q <= .05 and rows[j]['theta_select'] > 0 and not rows[j]['guard'])
    choices = {}
    for base in BASELINES:
        candidates = [row for row in rows if row['baseline'] == base]
        base_mse = candidates[0]['mse_baseline_select']
        choices[base] = {}
        for strategy, metric in [('convex', 'mse_convex_select'), ('augment', 'mse_augment_select'), ('gated', 'mse_augment_select')]:
            eligible = [r for r in candidates if strategy != 'gated' or r['gate']]
            best = min(eligible, key=lambda r: r[metric]) if eligible else None
            choices[base][strategy] = best['candidate'] if best and best[metric] < base_mse else 'baseline'
    # Models are frozen by serialized bytes; confirmation never refits them.
    import pickle
    (out/'fitted_models.pkl').write_bytes(pickle.dumps((models, fits)))
    rows_frame = pd.DataFrame(rows)
    rows_frame.to_csv(out/'selection_all_candidates.csv', index=False)
    record = {'selection_frozen_utc': utc(), 'protocol_sha256': sha(out/'protocol.json'),
              'source_sha256': sha(out/'source.zip'), 'models_sha256': sha(out/'fitted_models.pkl'),
              'selection_csv_sha256': sha(out/'selection_all_candidates.csv'),
              'confirmation_outcomes_parsed': False, 'choices': choices,
              'counts': {w: int(mask(frame, w).sum()) for w in ['train', 'nuisance', 'select']},
              'rows': rows}
    dump(out/'selection.json', record)
    (out/'selection.sha256').write_text(sha(out/'selection.json')+'  selection.json\n')
    print(json.dumps({'selection_frozen_utc': record['selection_frozen_utc'], 'choices': choices, 'gated': int(rows_frame.gate.sum())}))


def interval(delta, bootstrap_indices):
    primary = nominal(delta)
    fortnight = nominal(delta, block=336)
    weeks = delta.reshape(-1, 168).mean(axis=1)
    boot = weeks[bootstrap_indices].mean(axis=1)
    return {'gain': primary['mean'], 'ci_low': primary['low'], 'ci_high': primary['high'],
            'weekly_se': primary['se'], 'weekly_blocks': primary['blocks'],
            'fortnight_low': fortnight['low'], 'fortnight_high': fortnight['high'],
            'bootstrap_low': float(np.quantile(boot, .025)), 'bootstrap_high': float(np.quantile(boot, .975))}


def confirm(out):
    check(out)
    assert not (out/'confirmation_receipt.json').exists(), 'Confirmation may run once only'
    assert not (out/'confirmation_started.json').exists(), 'Interrupted confirmation needs explicit deviation, not silent rerun'
    assert (out/'selection.sha256').read_text().split()[0] == sha(out/'selection.json')
    selection = json.loads((out/'selection.json').read_text())
    assert sha(out/'fitted_models.pkl') == selection['models_sha256']
    assert sha(out/'selection_all_candidates.csv') == selection['selection_csv_sha256']
    dump(out/'confirmation_started.json', {'started_utc': utc(), 'selection_sha256': sha(out/'selection.json'),
                                         'protocol_sha256': sha(out/'protocol.json')})
    import pickle
    models, fits = pickle.loads((out/'fitted_models.pkl').read_bytes())
    frame = table(out)
    panels, _ = build(frame, models, fits)
    co = mask(frame, 'confirm')
    y = frame.y.to_numpy()[co]
    assert len(y) == 2016
    bootstrap_indices = np.random.default_rng(SEED+1).integers(0, 12, size=(4000, 12))
    metrics, gains, identity, trace = [], [], [], []
    for base in BASELINES:
        panel = panels[base]
        my = panel['mY'][co]
        baseline_loss = (y-my)**2
        forecasts = {'baseline': my, 'raw_baseline': panel['b'][co]}
        for name in CANDIDATES:
            row = next(r for r in selection['rows'] if r['baseline'] == base and r['candidate'] == name)
            x = panel['candidates'][name]['x'][co]
            rx = panel['candidates'][name]['rX'][co]
            forecasts['direct__'+name] = x
            forecasts['convex__'+name] = my+row['alpha']*(x-my)
            forecasts['augment__'+name] = my+row['t']*rx
            forecasts['gated__'+name] = forecasts['augment__'+name] if row['gate'] else my
            theta, v = float(np.mean(rx*(y-my))), float(np.mean(rx**2))
            gain = float(np.mean(baseline_loss-(y-forecasts['augment__'+name])**2))
            expression = 2*row['t']*theta-row['t']**2*v
            assert np.isclose(gain, expression, rtol=1e-10, atol=1e-6)
            identity.append({'baseline': base, 'candidate': name, 'gate': row['gate'], 't_frozen': row['t'],
                             'predicted_gain_select': row['predicted_gain'], 'theta_confirm': theta, 'v_confirm': v,
                             'gain_confirm': gain, 'quadratic_identity': expression,
                             'confirm_oracle_gain_diagnostic_only': max(theta, 0)**2/v if v > 1e-12 else 0.,
                             'oracle_coefficient_diagnostic_only': max(theta, 0)/v if v > 1e-12 else 0.})
        for strategy, name in selection['choices'][base].items():
            forecasts['selected_'+strategy] = my if name == 'baseline' else forecasts[strategy+'__'+name]
        for name, pred in forecasts.items():
            loss = (y-pred)**2
            mse = float(np.mean(loss))
            metrics.append({'baseline': base, 'method': name, 'n': len(y), 'mse': mse,
                            'rmse': float(np.sqrt(mse)), 'mae': float(np.mean(np.abs(y-pred))),
                            'negative_forecast_fraction': float(np.mean(pred < 0)),
                            'relative_mse_gain': float(1-mse/np.mean(baseline_loss)),
                            'meaningful_1pct_gain': bool(mse <= .99*np.mean(baseline_loss))})
            gains.append({'baseline': base, 'method': name, 'comparator': 'baseline',
                          **interval(baseline_loss-loss, bootstrap_indices)})
            for j in range(12):
                sl = slice(j*168, (j+1)*168)
                trace.append({'baseline': base, 'method': name, 'week': j,
                              'start': str(frame.date[co].iloc[j*168]),
                              'mse': float(np.mean(loss[sl])), 'mae': float(np.mean(np.abs(y[sl]-pred[sl])))})
        for comparator in ['selected_convex', 'selected_augment', 'direct__ridge_lags_calendar', 'direct__hist_gradient_boosting']:
            delta = (y-forecasts[comparator])**2-(y-forecasts['selected_gated'])**2
            gains.append({'baseline': base, 'method': 'selected_gated', 'comparator': comparator,
                          **interval(delta, bootstrap_indices)})
    for name, rows in [('confirmation_metrics', metrics), ('paired_risk_improvements', gains),
                       ('gain_identity', identity), ('weekly_trace', trace)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'), index=False)
    outputs = ['confirmation_metrics.csv', 'paired_risk_improvements.csv', 'gain_identity.csv', 'weekly_trace.csv']
    dump(out/'confirmation_receipt.json', {'completed_utc': utc(), 'protocol_sha256': sha(out/'protocol.json'),
         'selection_sha256': sha(out/'selection.json'), 'script_sha256': sha(__file__),
         'source_sha256': sha(out/'source.zip'), 'confirm_count': len(y), 'weekly_blocks': 12,
         'outputs_sha256': {name: sha(out/name) for name in outputs},
         'no_post_confirmation_refit': True, 'all_candidates_retained': True,
         'interpretation': 'Historical one-hour-ahead forecasting under immediate lagged-count availability. Raw data publication vintages are not reconstructed. Temporal block uncertainty is approximate. This source is not proof of independent external replication or general audit superiority.'})
    selected = pd.DataFrame(metrics)
    print(selected[selected.method.isin(['baseline', 'selected_convex', 'selected_augment', 'selected_gated', 'direct__ridge_lags_calendar', 'direct__hist_gradient_boosting'])].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['freeze', 'download', 'prepare', 'confirm'])
    parser.add_argument('--output', type=Path, default=HERE)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        globals()[args.phase](args.output)
