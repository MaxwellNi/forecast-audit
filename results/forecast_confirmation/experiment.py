"""Rebuild fixed hourly forecasting tasks from bundled public source archives.

Writes only to a supplied reproduction directory. The current code is a portable
adaptation of the corrected study; it does not claim a new pre-exposure seal.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import pickle
import platform
import resource
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from gates import METHODS, TOTAL, by_values, draw_indices, certify

BUNDLE = Path(__file__).resolve().parent
HERE = None
CANDIDATES = ['persistence', 'seasonal_day', 'seasonal_week', 'ridge_full',
              'hist_gradient_boosting', 'extra_trees', 'category_copy', 'independent_noise']
BASELINES = ['seasonal_day', 'ridge']


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def read(path):
    return json.loads(Path(path).read_text())


def check():
    manifest = read(BUNDLE / 'MANIFEST.json')
    for name, digest in manifest['sha256'].items():
        assert sha(BUNDLE/name) == digest, f'Export input changed: {name}'
    return read(BUNDLE/'protocol.json')


def features(series, period):
    """All outcome-dependent features use strictly earlier calendar slots."""
    index = series.index
    part = pd.DataFrame(index=index)
    assert period == 24
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        part[f'lag_{lag}'] = series.shift(lag)
    for size in [period, 7*period]:
        prior = series.shift(1).rolling(size, min_periods=1)
        part[f'rolling_mean_{size}'] = prior.mean()
        part[f'rolling_std_{size}'] = prior.std(ddof=0)
        part[f'rolling_observed_{size}'] = prior.count()/size
    for name, values, cycle in [('hour', index.hour+index.minute/60, 24),
                                 ('week', index.dayofweek, 7),
                                 ('year', index.dayofyear, 365.25)]:
        part[f'{name}_sin'] = np.sin(2*np.pi*values/cycle)
        part[f'{name}_cos'] = np.cos(2*np.pi*values/cycle)
    part['weekend'] = (index.dayofweek>=5).astype(int)
    return part


def source_series(task, path):
    with zipfile.ZipFile(path) as archive:
        choices = [x for x in archive.namelist() if x.endswith(('.csv', '.csv.gz'))]
        assert len(choices) == 1, choices
        body = archive.read(choices[0])
        raw = pd.read_csv(io.BytesIO(body), compression='gzip' if choices[0].endswith('.gz') else None)
    stamps = pd.to_datetime(raw[task['time_column']])
    y = pd.to_numeric(raw[task['outcome_column']], errors='raise')
    assert np.isfinite(y).all() and (y >= 0).all()
    table = pd.DataFrame({'time': stamps, 'y': y})
    duplicate_spread = table.groupby('time').y.agg(['size','min','max'])
    if task['name']=='appliances':
        allowed=['T'+str(i) for i in range(1,10)]+['RH_'+str(i) for i in range(1,10)]+['lights']
        numeric=raw[allowed].astype(float).set_axis(pd.DatetimeIndex(stamps))
        numeric=numeric.groupby(level=0).mean()
        native=table.groupby('time',sort=True).y.mean()
        complete=pd.date_range(native.index.min(),native.index.max(),freq='10min')
        assert native.index.isin(complete).all()
        native=native.reindex(complete)
        sizes=native.resample('h').count()
        series=native.resample('h').sum(min_count=6).where(sizes==6)
        extras=numeric.resample('h').mean()
        extras['lights']=numeric['lights'].resample('h').sum(min_count=6)
        extras=extras.reindex(series.index).shift(1)
        origin=str(pd.Timestamp(stamps.min()).ceil('D'))
    else:
        grouped=table.groupby('time',sort=True).y.mean()
        complete=pd.date_range(grouped.index.min(),grouped.index.max(),freq='h')
        assert grouped.index.isin(complete).all()
        series=grouped.reindex(complete)
        allowed=['temp','rain_1h','snow_1h','clouds_all']
        numeric=raw[allowed].astype(float).set_axis(pd.DatetimeIndex(stamps))
        extras=numeric.groupby(level=0).mean().reindex(complete).shift(1)
        # Weather text is omitted by the prefetch adjudication; no vocabulary
        # from future periods or post-hoc text selection enters a model.
        origin=str(series.index.min())
    extras.columns=['previous_hour_'+name for name in extras.columns]
    meta = dict(raw_rows=len(raw), unique_observed_slots=int(series.notna().sum()), calendar_slots=len(series),
                absent_calendar_slots=int(series.isna().sum()), duplicate_rows=int(stamps.duplicated().sum()),
                duplicate_slots_with_different_outcomes=int((duplicate_spread['min']!=duplicate_spread['max']).sum()),
                first=str(series.index.min()), last=str(series.index.max()), archive_member=choices[0],calendar_origin=origin)
    return series, extras, meta


def split_masks(index, task, origin):
    if task['name']=='appliances':
        t0=pd.Timestamp(origin)
        bounds=[t0+pd.Timedelta(days=k) for k in [0,56,70,84,126]]
    else:
        bounds=[pd.Timestamp(s) for s in ['2012-01-01','2016-01-01','2016-07-01','2017-01-01','2018-01-01']]
    return {key:(index>=start)&(index<stop) for key,start,stop in zip(
        ['train','calibration','selection','confirmation'],bounds[:-1],bounds[1:])}, bounds


def prepare():
    p = check()
    download = read(HERE/'download_receipt.json')
    assert not (HERE/'preparation_receipt.json').exists()
    records = {}
    for ti, task in enumerate(p['tasks']):
        started = time.perf_counter()
        folder = HERE/task['name']
        folder.mkdir()
        source = HERE/'data'/f"{task['name']}.zip"
        assert sha(source) == download['sources'][task['name']]['sha256']
        series, extras, meta = source_series(task, source)
        X = features(series, 24)
        full_X = pd.concat([X,extras],axis=1)
        masks,bounds = split_masks(series.index, task, meta['calendar_origin'])
        for key in masks:
            masks[key] &= series.notna().to_numpy()
            if key=='train':
                assert masks[key].sum() >= 500
            else:
                ki=['train','calibration','selection','confirmation'].index(key)
                nominal=int((bounds[ki+1]-bounds[ki])/pd.Timedelta(hours=1))
                assert masks[key].sum() >= .75*nominal, (task['name'],key,int(masks[key].sum()),nominal)
        train = masks['train']
        scale = max(1.,float(np.quantile(series[train], .99)))
        median = float(series[train].median())
        assert scale>0
        imputer=lambda: SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)
        ridge=lambda: make_pipeline(imputer(),StandardScaler(),Ridge(alpha=10.,solver='svd'))
        models = {
          'hist_gradient_boosting': make_pipeline(imputer(),HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=15,
             learning_rate=.05,min_samples_leaf=20,l2_regularization=.01,early_stopping=False,
             random_state=p['seed']+ti+10)),
          'extra_trees':make_pipeline(imputer(),ExtraTreesRegressor(n_estimators=256,max_depth=12,
             min_samples_leaf=10,max_features=1.,n_jobs=1,random_state=p['seed']+ti+20)),
          'ridge':ridge(),'ridge_full':ridge()}
        forecasts={}
        for name,model in models.items():
            xx=X if name=='ridge' else full_X
            model.fit(xx.loc[train],np.clip(series[train],0,scale))
            forecasts[name]=np.clip(model.predict(xx),0,scale)
        lag = lambda n: X[f'lag_{n}'].fillna(X['lag_1']).fillna(median).to_numpy()
        forecasts.update(persistence=lag(1),seasonal_day=lag(24),seasonal_week=lag(168))
        forecasts={name:np.clip(values,0,scale) for name,values in forecasts.items()}
        rng = np.random.default_rng(p['seed']+100+ti)
        forecasts['independent_noise'] = rng.uniform(0,scale,len(series))
        with (folder/'models.pkl').open('wb') as f:
            pickle.dump(models, f)
        arrays = dict(time=series.index.to_numpy(dtype='datetime64[ns]'), y=series.to_numpy(),
                      scale=np.array(scale), **masks)
        arrays.update(forecasts)
        rules = []
        for bi, baseline in enumerate(BASELINES):
            base = forecasts[baseline]
            breaks = np.quantile(np.clip(base[train],0,scale),np.arange(1,16)/16)
            category = 2*np.searchsorted(breaks,np.clip(base,0,scale),side='left')+(series.index.dayofweek>=5).astype(int)
            # Copy scores depend only on the declared control category and
            # training data, making their exact conditional rank target zero.
            copied=scale*(np.arange(32)+.5)/32
            arrays[f'{baseline}__category'] = category
            for name in CANDIDATES:
                pred = copied[category] if name=='category_copy' else forecasts[name]
                arrays[f'{baseline}__{name}'] = pred
                cal = masks['calibration']
                # Fit an augmentation weight using calibration only; the grid and
                # clipped loss match the primary prospective decision metric.
                grid = np.linspace(0,2,41)
                blends = base[cal,None]+grid[None,:]*(pred[cal,None]-base[cal,None])
                losses = ((np.clip(series[cal].to_numpy()[:,None],0,scale)-np.clip(blends,0,scale))/scale)**2
                weight = float(grid[np.argmin(losses.mean(0))])
                convex_grid=grid<=1
                convex_weight=float(grid[convex_grid][np.argmin(losses[:,convex_grid].mean(0))])
                rules.append(dict(baseline=baseline,candidate=name,weight=weight,
                                  convex_weight=convex_weight,
                                  calibration_loss=float(losses.mean(0).min())))
            arrays[f'{baseline}__breaks'] = breaks
        # Numeric timing check changes current/future labels, compares all
        # features through the cutoff, and requires exact equality including NaN.
        timing = []
        for key in ['calibration','selection','confirmation']:
            cutoff = series.index[masks[key]][0]
            changed = series.copy()
            changed.loc[cutoff:] += 1_000_000
            alternate = features(changed,24)
            np.testing.assert_equal(X.loc[:cutoff].to_numpy(),alternate.loc[:cutoff].to_numpy())
            timing.append(dict(first_target=str(cutoff),current_future_feature_changes=0))
        np.savez_compressed(folder/'forecast_archive.npz', **arrays)
        pd.DataFrame(rules).to_csv(folder/'frozen_blends.csv',index=False,float_format='%.17g')
        records[task['name']] = dict(source=meta, split_rows={k:int(v.sum()) for k,v in masks.items()},
          split_start={k:str(series.index[v][0]) for k,v in masks.items()},
          split_end={k:str(series.index[v][-1]) for k,v in masks.items()},
          clipping_scale=scale,train_median=median,timing_checks=timing,
          base_feature_names=list(X),full_feature_names=list(full_X),
          archive_sha256=sha(folder/'forecast_archive.npz'), blends_sha256=sha(folder/'frozen_blends.csv'),
          models_sha256=sha(folder/'models.pkl'), preparation_seconds=time.perf_counter()-started,
          all_source_labels_materialized=True, selection_or_confirmation_metrics_computed=False)
    write_new(HERE/'preparation_receipt.json',dict(completed_utc=utc(),tasks=records))


def loss(y,pred,scale):
    return ((np.clip(y,0,scale)-np.clip(pred,0,scale))/scale)**2


def select():
    p = check()
    prepared = read(HERE/'preparation_receipt.json')
    assert not (HERE/'selection_receipt.json').exists()
    write_new(HERE/'selection_started.json',dict(started_utc=utc(), protocol_sha256=sha(BUNDLE/'protocol.json')))
    all_certificates, choices = [], []
    for ti, task in enumerate(p['tasks']):
        folder = HERE/task['name']
        record = prepared['tasks'][task['name']]
        assert sha(folder/'forecast_archive.npz')==record['archive_sha256']
        assert sha(folder/'frozen_blends.csv')==record['blends_sha256']
        data = np.load(folder/'forecast_archive.npz',allow_pickle=False)
        blends = pd.read_csv(folder/'frozen_blends.csv')
        mask = data['selection']
        y,scale = data['y'][mask],float(data['scale'])
        indices = draw_indices(len(y),p['seed']+1000+ti)
        np.savez_compressed(folder/'sampling_indices.npz', training=indices[0],validation=indices[1],
                            evaluation=indices[2],all_rows=indices[3])
        for baseline in BASELINES:
            base,cat = data[baseline][mask],data[f'{baseline}__category'][mask]
            group,selection_losses,convex_losses = [],{},{}
            for name in CANDIDATES:
                pred = data[f'{baseline}__{name}'][mask]
                weight = float(blends[(blends.baseline==baseline)&(blends.candidate==name)].weight.iloc[0])
                blend = base+weight*(pred-base)
                convex_weight=float(blends[(blends.baseline==baseline)&(blends.candidate==name)].convex_weight.iloc[0])
                started=time.perf_counter()
                rows,budget = certify(pred,y,cat,base,blend,scale,indices)
                for row in rows:
                    row.update(task=task['name'],baseline=baseline,candidate=name,weight=weight,
                               five_gate_seconds=time.perf_counter()-started)
                group.extend(rows)
                selection_losses[name] = float(loss(y,blend,scale).mean())
                convex_losses[name]=float(loss(y,base+convex_weight*(pred-base),scale).mean())
            frame = pd.DataFrame(group)
            for method in METHODS:
                subset = frame.method==method
                frame.loc[subset,'p_by'] = by_values(frame.loc[subset,'p'])
                retained = frame.loc[subset&(frame.p_by<=.05),'candidate'].tolist()
                # Baseline is always available; a significant association does
                # not force adoption of a worse fitted augmentation.
                eligible = ['baseline']+retained
                values = dict(selection_losses,baseline=float(loss(y,base,scale).mean()))
                choice = min(eligible,key=lambda name:values[name])
                choices.append(dict(task=task['name'],baseline=baseline,method=method,
                  selected=choice,selection_loss=values[choice],retained=';'.join(retained)))
            values = dict(selection_losses,baseline=float(loss(y,base,scale).mean()))
            choice = min(['baseline']+CANDIDATES,key=lambda name:values[name])
            choices.append(dict(task=task['name'],baseline=baseline,method='ungated',
                selected=choice,selection_loss=values[choice],retained=';'.join(CANDIDATES)))
            convex_values=dict(convex_losses,baseline=float(loss(y,base,scale).mean()))
            convex_choice=min(['baseline']+CANDIDATES,key=lambda name:convex_values[name])
            choices.append(dict(task=task['name'],baseline=baseline,method='convex',selected=convex_choice,
                selection_loss=convex_values[convex_choice],retained=';'.join(CANDIDATES)))
            choices.append(dict(task=task['name'],baseline=baseline,method='baseline',
                selected='baseline',selection_loss=values['baseline'],retained=''))
            all_certificates.extend(frame.to_dict('records'))
    pd.DataFrame(all_certificates).to_csv(HERE/'selection_certificates.csv',index=False,float_format='%.17g')
    pd.DataFrame(choices).to_csv(HERE/'frozen_selections.csv',index=False,float_format='%.17g')
    write_new(HERE/'selection_receipt.json',dict(completed_utc=utc(),
      certificate_sha256=sha(HERE/'selection_certificates.csv'), selections_sha256=sha(HERE/'frozen_selections.csv'),
      peak_resident_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      confirmation_loss_not_computed=True))


def confirm():
    p = check()
    selected = read(HERE/'selection_receipt.json')
    prepared=read(HERE/'preparation_receipt.json')
    assert sha(HERE/'frozen_selections.csv')==selected['selections_sha256']
    assert not (HERE/'confirmation_started.json').exists()
    write_new(HERE/'confirmation_started.json',dict(started_utc=utc(), selection_receipt_sha256=sha(HERE/'selection_receipt.json')))
    decisions = pd.read_csv(HERE/'frozen_selections.csv',keep_default_na=False)
    summary,weekly,candidate_rows,dependent_bounds = [],[],[],[]
    for task in p['tasks']:
        folder = HERE/task['name']
        record=prepared['tasks'][task['name']]
        assert sha(folder/'forecast_archive.npz')==record['archive_sha256']
        assert sha(folder/'frozen_blends.csv')==record['blends_sha256']
        data = np.load(folder/'forecast_archive.npz',allow_pickle=False)
        blends = pd.read_csv(folder/'frozen_blends.csv')
        mask = data['confirmation']
        y,scale = data['y'][mask],float(data['scale'])
        stamps = pd.to_datetime(data['time'][mask])
        block = ((stamps-stamps.min())/pd.Timedelta(days=7)).astype(int)
        for baseline in BASELINES:
            base = data[baseline][mask]
            predictions,convex_predictions={'baseline':base},{'baseline':base}
            for name in CANDIDATES:
                raw = data[f'{baseline}__{name}'][mask]
                weight=float(blends[(blends.baseline==baseline)&(blends.candidate==name)].weight.iloc[0])
                predictions[name] = np.clip(base+weight*(raw-base),0,scale)
                convex_weight=float(blends[(blends.baseline==baseline)&(blends.candidate==name)].convex_weight.iloc[0])
                convex_predictions[name]=np.clip(base+convex_weight*(raw-base),0,scale)
                candidate_rows.append(dict(task=task['name'],baseline=baseline,candidate=name,weight=weight,
                  bounded_mse=float(loss(y,predictions[name],scale).mean()),raw_mse=float(((y-predictions[name])**2).mean())))
            rule = decisions[(decisions.task==task['name'])&(decisions.baseline==baseline)]
            ungated = predictions[rule[rule.method=='ungated'].selected.iloc[0]]
            convex=convex_predictions[rule[rule.method=='convex'].selected.iloc[0]]
            base_loss,ungated_loss=loss(y,base,scale),loss(y,ungated,scale)
            convex_loss=loss(y,convex,scale)
            if baseline=='ridge':
                primary_pred=predictions[rule[rule.method=='reference_u'].selected.iloc[0]]
                primary_loss=loss(y,primary_pred,scale)
                nominal=1008 if task['name']=='appliances' else 8760
                assert len(y)<=nominal
                alpha=.05/4
                radius=np.sqrt(2*np.log(1/alpha)/nominal)
                for comparison,comparison_loss in [('ungated',ungated_loss),('convex',convex_loss)]:
                    # Missing nominal hours contribute zero, preserving a
                    # fixed horizon and the bounded martingale target.
                    observed_sum=float(np.sum(comparison_loss-primary_loss))
                    center=observed_sum/nominal
                    dependent_bounds.append(dict(task=task['name'],baseline=baseline,
                      primary='reference_u',comparison=comparison,nominal_hours=nominal,
                      observed_hours=len(y),availability_weighted_gain=center,
                      alpha=alpha,radius=radius,conditional_average_gain_lower=center-radius,
                      positive_conditional_gain_certified=bool(center-radius>0)))
            for row in rule.itertuples():
                pred=(convex_predictions if row.method=='convex' else predictions)[row.selected]
                bounded=loss(y,pred,scale)
                summary.append(dict(task=task['name'],baseline=baseline,method=row.method,selected=row.selected,
                   confirmation_rows=len(y),clipping_scale=scale,bounded_mse=float(bounded.mean()),
                   raw_mse=float(((y-pred)**2).mean()),baseline_bounded_mse=float(base_loss.mean()),
                   gain_vs_baseline=float(np.mean(base_loss-bounded)),gain_vs_ungated=float(np.mean(ungated_loss-bounded)),
                   gain_vs_convex=float(np.mean(convex_loss-bounded)),raw_mae=float(np.mean(abs(y-pred))),
                   bounded_mae=float(np.mean(abs(np.clip(y,0,scale)-np.clip(pred,0,scale))/scale)),
                   raw_gain_vs_baseline=float(np.mean((y-base)**2-(y-pred)**2)),
                   raw_gain_vs_ungated=float(np.mean((y-ungated)**2-(y-pred)**2))))
                for b in np.unique(block):
                    use=block==b
                    weekly.append(dict(task=task['name'],baseline=baseline,method=row.method,block=int(b),rows=int(use.sum()),
                       first=str(stamps[use].min()),last=str(stamps[use].max()),
                       bounded_gain_vs_baseline=float(np.mean((base_loss-bounded)[use])),
                       bounded_gain_vs_ungated=float(np.mean((ungated_loss-bounded)[use])),
                       bounded_gain_vs_convex=float(np.mean((convex_loss-bounded)[use]))))
    # These historical week summaries are descriptive; no calibrated temporal
    # confidence interval or superpopulation significance claim is attached.
    for name,rows in [('confirmation_results.csv',summary),('confirmation_weeks.csv',weekly),
                      ('confirmation_all_candidates.csv',candidate_rows),
                      ('confirmation_dependent_bounds.csv',dependent_bounds)]:
        pd.DataFrame(rows).to_csv(HERE/name,index=False,float_format='%.17g')
    write_new(HERE/'confirmation_receipt.json',dict(completed_utc=utc(),
       output_sha256={name:sha(HERE/name) for name in ['confirmation_results.csv','confirmation_weeks.csv','confirmation_all_candidates.csv','confirmation_dependent_bounds.csv']},
       peak_resident_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       scope='Locally frozen chronological confirmation. Observed-support losses are descriptive. Four separately frozen martingale Hoeffding bounds cover availability-weighted conditional average gains over fixed nominal horizons under adapted forecasts, with family failure probability at most .05; no iid/stationary-time or later-period guarantee.'))
    print(pd.DataFrame(summary)[['task','baseline','method','selected','gain_vs_baseline','gain_vs_ungated']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','select','confirm'])
    parser.add_argument('--output',type=Path,required=True,help='Existing reproduction directory with data and download_receipt.json')
    args=parser.parse_args()
    HERE=args.output.resolve()
    if HERE == BUNDLE or BUNDLE in HERE.parents:
        parser.error("Use a separate output directory; the bundled evidence is read-only")
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    with threadpool_limits(limits=1):
        globals()[args.phase]()
