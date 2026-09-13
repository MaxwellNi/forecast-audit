#!/usr/bin/env python3
"""Locally frozen finite-archive certificate with independent replacement draws.

freeze -> prepare -> audit are separate, refuse overwrites, and hash their inputs.
The inferential population is the uniform distribution over fixed archive rows,
not a temporal superpopulation or future forecast performance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform
import shutil
import sys
import time
import zipfile

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE
SOURCE = HERE / 'data/source.zip'
SEED = 2609123147
FEATURES = ['lag1', 'lag2', 'lag7', 'lag14', 'lag28', 'mean7', 'mean28',
            'target_dow_sin', 'target_dow_cos', 'target_year_sin', 'target_year_cos']
CANDIDATES = ['hist_gradient_boosting', 'seasonal_7d', 'categorical_copy_control']


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def dump_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load(path):
    return json.loads(Path(path).read_text())


def freeze(raw_source):
    assert not (HERE / 'protocol.json').exists(), 'Protocol already frozen'
    assert not (HERE / 'forecast_archive.npz').exists()
    helper_sources = {
        'reference_certificate_efficiency.py': HERE / 'helpers/reference_certificate_efficiency.py',
        'category_reference_certificate.py': HERE / 'helpers/category_reference_certificate.py',
        'peer_rank_products.py': HERE / 'helpers/peer_rank_products.py'}
    (HERE / 'helpers').mkdir(exist_ok=True)
    helper_hashes = {}
    for name, source in helper_sources.items():
        if source.resolve() != (HERE / 'helpers' / name).resolve():
            shutil.copyfile(source, HERE / 'helpers' / name)
        helper_hashes[name] = dict(source=str(source.relative_to(ROOT)), sha256=sha(source))
    protocol = dict(frozen_utc=utc(), script_sha256=sha(__file__), seed=SEED,
        source_sha256=sha(raw_source), source_bytes=Path(raw_source).stat().st_size,
        source_url='https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip',
        source_doi='https://doi.org/10.24432/C58C86',
        attribution='Trindade, A. (2015). ElectricityLoadDiagrams20112014 [Dataset]. UCI Machine Learning Repository. CC BY 4.0.',
        license_url='https://creativecommons.org/licenses/by/4.0/',
        source_inventory='Existing previously used public archive; this is a new locally frozen design-based analysis, not a newly acquired or previously unseen dataset.',
        population='Uniform finite empirical law over all frozen 2013-2014 target-day x first96-meter archive rows. Condition on this complete archive.',
        target='For each fixed candidate X and actual daily mean Y: theta_A=E_A[(F_mid,X,A(X)-E_A[F_mid,X,A(X)|C])*(F_mid,Y,A(Y)-E_A[F_mid,Y,A(Y)|C])]. Marginal population midranks, no within-date ranking, no claim about full continuous baseline adjustment.',
        archive='First96 header meters MT_001...MT_096; nominal daily arithmetic kW means of exactly96 supplied quarter-hour slots. Exclude partial edge days; fail on interior missing data. All zeros retained. Fixed targets2013-01-01 through2014-12-31 inclusive, expected70080 rows.',
        timing='One-day-ahead daily mean forecast after previous nominal calendar day is observed. Lags target-minus1,2,7,14,28 and previous7/28-day means; target calendar deterministic. All forecast model training targets strictly before2013-01-01. No forecast refit in archive.',
        features=FEATURES,
        model=dict(name='HistGradientBoostingRegressor', max_iter=160, max_leaf_nodes=31,
                   learning_rate=.05, min_samples_leaf=32, l2_regularization=1.,
                   early_stopping=False, random_state=SEED),
        categories='C=2*target_weekend+I(seasonal_7d > median seasonal_7d on chronological forecast-training rows). Four fixed categories. Numeric threshold fitted only on pre2013 rows and sealed with forecasts. Binned baseline/calendar target; residual within-bin baseline information may remain.',
        candidates=CANDIDATES, copy_control='categorical_copy_control=C as a numeric score, so its exact archive conditional residual target is zero. Keep it in each complete three-member family; no outcome-based exclusion.',
        nuisance_training_rows=4096, validation_pairs=8192, validation_rows=16384,
        evaluation_groups=128, evaluation_peers=96, evaluation_rows=12288,
        total_sample_row_queries=32768,
        sampling='Independent SeedSequence child streams for4096 training rowindices,8192 disjoint focal/reference validation pairs, and128x96 evaluation rowindices; every index drawn independently uniformly WITH replacement from the frozen archive. Repeated physical rows within/across samples are retained; independence is that of the randomized indices conditional on the fixed archive.',
        nuisance_fit='Candidate/outcome marginal leave-one-out midranks in the independent4096-row training sample, averaged by fixed C and clipped[0,1]; empty training categories get0.5. No census ranks or means enter training.',
        validation='Each iid validation pair supplies a focal category and focal/reference midcomparison for X and Y. Signed and absolute budgets share identical cell-mean rectangles and Clopper-Pearson category mass bounds; unobserved cells retained.',
        delta=.0001, alpha=.05, family_size=3, by_q=.05,
        primary='pooled_full_u_signed_variance: one full order-three U statistic over all12288 evaluation draws, signed learning allowance, proved variance bound with fixed disjoint-triple variance scores; familyBY across3 candidates.',
        methods=['grouped_full_u_absolute_range', 'grouped_full_u_signed_range',
                 'grouped_full_u_signed_variance', 'pooled_full_u_signed_range',
                 'pooled_full_u_signed_variance', 'independent_triples_signed_range',
                 'independent_triples_signed_bernstein'],
        comparator_contract='All methods share exactly the same training, validation and evaluation draws, nuisance fits and candidate family. Grouped fullU uses128 groups of96; pooled fullU uses all12288; independent triples use4096 disjoint triples. No extra observations. Each method has its own reported fixed3-target BY family; no selection among methods by results.',
        census='Compute exact full-archive midranks, category means and theta only after certificate_results.csv is written. Census is a diagnostic and exact alternative, never used for fitting, choosing budgets, method, categories, or certificate. Benchmark32768 sampled rowqueries and their unique union against70080-row census, plus end-to-end archive construction and runtimes. No claimed overall access saving: preparation materializes the whole archive.',
        inference_scope='Conditional design-based finite-archive statement. Original meter/time dependence is irrelevant to iid index sampling given fixed archive, but results do not certify future forecasting, real-time risk, continuous-baseline incremental information, or causal provenance. Historical raw vintages are not reconstructed.',
        locking='Protocol and source/helper hashes precede preparation; model, numeric category threshold and archive hash sealed before any audit sampling/outcomes; all results retained; no retuning after audit. Local timing is not external preregistration.',
        helper_sources=helper_hashes,
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                         scipy=scipy.__version__, sklearn=sklearn.__version__))
    dump_new(HERE / 'protocol.json', protocol)
    (HERE / 'protocol.sha256').write_text(sha(HERE / 'protocol.json') + '  protocol.json\n')
    print(json.dumps(dict(frozen_utc=protocol['frozen_utc'], protocol_sha256=sha(HERE / 'protocol.json'),
                         candidates=CANDIDATES, planned_sample_queries=32768, planned_archive_rows=70080)))


def check():
    p = load(HERE / 'protocol.json')
    assert p['script_sha256'] == sha(__file__), 'Frozen script changed'
    assert (HERE / 'protocol.sha256').read_text().split()[0] == sha(HERE / 'protocol.json')
    for name, entry in p['helper_sources'].items():
        assert sha(HERE / 'helpers' / name) == entry['sha256']
    return p


def source_daily(source):
    with zipfile.ZipFile(source) as archive, archive.open('LD2011_2014.txt') as stream:
        raw = pd.read_csv(stream, sep=';', decimal=',', usecols=range(97))
    stamps = pd.to_datetime(raw.iloc[:, 0])
    assert stamps.is_monotonic_increasing and not stamps.duplicated().any()
    assert stamps.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all()
    values = raw.iloc[:, 1:].astype(float).set_axis(pd.DatetimeIndex(stamps))
    assert list(values) == [f'MT_{i:03d}' for i in range(1, 97)]
    grouped = values.resample('D', closed='left', label='left')
    sizes = grouped.size()
    daily = grouped.mean().where(grouped.count().eq(96))
    good = sizes.eq(96)
    first, last = sizes[good].index[[0, -1]]
    daily = daily.loc[first:last]
    assert not daily.isna().any().any() and np.isfinite(daily.to_numpy()).all()
    assert daily.index.equals(pd.date_range(first, last, freq='D'))
    return daily, dict(raw_quarter_hour_rows=len(raw), meters=96, daily_rows=len(daily),
                       daily_start=str(first.date()), daily_end=str(last.date()),
                       retained_zero_cells=int(daily.eq(0).sum().sum()),
                       excluded_partial_edge_dates={str(k.date()): int(v) for k, v in sizes[~good].items()})


def forecast_frame(daily):
    rows = []
    for meter, series in daily.items():
        target_dates = daily.index
        part = pd.DataFrame({'date': target_dates, 'meter': meter, 'y': series.to_numpy()})
        for lag in [1, 2, 7, 14, 28]:
            part[f'lag{lag}'] = series.shift(lag).to_numpy()
        for n in [7, 28]:
            part[f'mean{n}'] = series.shift(1).rolling(n).mean().to_numpy()
        for name, value, period in [('dow', target_dates.dayofweek, 7),
                                    ('year', target_dates.dayofyear, 365.25)]:
            part[f'target_{name}_sin'] = np.sin(2 * np.pi * value / period)
            part[f'target_{name}_cos'] = np.cos(2 * np.pi * value / period)
        rows.append(part)
    frame = pd.concat(rows, ignore_index=True).dropna(subset=FEATURES + ['y'])
    return frame.sort_values(['date', 'meter']).reset_index(drop=True)


def prepare(raw_source):
    protocol = check()
    assert not (HERE / 'archive_seal.json').exists() and not (HERE / 'forecast_archive.npz').exists()
    assert sha(raw_source) == protocol['source_sha256']
    started = time.perf_counter()
    daily, source_meta = source_daily(raw_source)
    frame = forecast_frame(daily)
    train = frame[frame.date < '2013-01-01']
    archive = frame[(frame.date >= '2013-01-01') & (frame.date <= '2014-12-31')].copy()
    assert len(archive) == 70080 and len(train) > 0
    params = dict(protocol['model'])
    del params['name']
    model = HistGradientBoostingRegressor(**params)
    model.fit(train[FEATURES], train.y)
    threshold = float(train.lag7.median())
    high = (archive.lag7 > threshold).to_numpy(int)
    category = 2 * (archive.date.dt.dayofweek >= 5).to_numpy(int) + high
    predicted = model.predict(archive[FEATURES])
    model_bytes = pickle.dumps(model)
    (HERE / 'forecast_model.pkl').write_bytes(model_bytes)
    np.savez_compressed(HERE / 'forecast_archive.npz',
        date=archive.date.to_numpy(dtype='datetime64[D]'), meter=archive.meter.to_numpy(dtype='U6'),
        outcome=archive.y.to_numpy(float), category=category,
        hist_gradient_boosting=predicted, seasonal_7d=archive.lag7.to_numpy(float),
        categorical_copy_control=category.astype(float))
    timing_checks = []
    for cutoff in ['2013-01-01', '2014-01-01', '2014-12-31']:
        altered = daily.copy()
        altered.loc[pd.Timestamp(cutoff):] += 1_000_000.
        other = forecast_frame(altered)
        original_features = frame.loc[frame.date <= cutoff, FEATURES].to_numpy()
        modified_features = other.loc[other.date <= cutoff, FEATURES].to_numpy()
        assert np.array_equal(original_features, modified_features)
        timing_checks.append(dict(cutoff=cutoff, current_and_future_outcome_feature_changes=0))
    seal = dict(preparation_started_before_audit=True, sealed_utc=utc(),
        protocol_sha256=sha(HERE / 'protocol.json'), model_sha256=sha(HERE / 'forecast_model.pkl'),
        archive_sha256=sha(HERE / 'forecast_archive.npz'), archive_rows=len(archive),
        forecast_training_rows=len(train), latest_forecast_training_target=str(train.date.max()),
        earliest_archive_target=str(archive.date.min()), latest_archive_target=str(archive.date.max()),
        baseline_category_training_median=threshold, category_count=4,
        audit_statistics_computed=False, nuisance_fits_computed=False,
        source_meta=source_meta, feature_timing_checks=timing_checks,
        preparation_seconds=time.perf_counter() - started,
        archive_construction_access='All source quarter-hour observations for96 meters and all70080 archive outcomes materialized; no end-to-end access-saving claim.')
    dump_new(HERE / 'archive_seal.json', seal)
    print(json.dumps(dict(sealed_utc=seal['sealed_utc'], archive_rows=len(archive),
                         forecast_training_rows=len(train), audit_statistics_computed=False)))


def group_means(categories, values):
    counts = np.bincount(categories, minlength=4)
    return np.divide(np.bincount(categories, weights=values, minlength=4), counts,
                     out=np.full(4, .5), where=counts > 0)


def audit():
    p = check()
    seal = load(HERE / 'archive_seal.json')
    assert seal['protocol_sha256'] == sha(HERE / 'protocol.json')
    assert seal['archive_sha256'] == sha(HERE / 'forecast_archive.npz')
    assert seal['model_sha256'] == sha(HERE / 'forecast_model.pkl')
    assert not (HERE / 'audit_started.json').exists(), 'No silent rerun or retuning'
    dump_new(HERE / 'audit_started.json', dict(started_utc=utc(), protocol_sha256=sha(HERE / 'protocol.json'),
                                            archive_sha256=seal['archive_sha256']))
    sys.path.insert(0, str(HERE / 'helpers'))
    from reference_certificate_efficiency import signed_category_budget, triple_scores, one_sided_certificate
    from peer_rank_products import peer_rank_products
    data = np.load(HERE / 'forecast_archive.npz', allow_pickle=False)
    length = len(data['outcome'])
    streams = [np.random.default_rng(seed) for seed in np.random.SeedSequence(SEED).spawn(3)]
    training = streams[0].integers(0, length, size=4096)
    validation = streams[1].integers(0, length, size=(8192, 2))
    evaluation = streams[2].integers(0, length, size=(128, 96))
    np.savez_compressed(HERE / 'sampling_indices.npz', training=training,
                        validation=validation, evaluation=evaluation)
    all_indices = np.concatenate([training, validation.ravel(), evaluation.ravel()])
    query_cost = dict(training_queries=len(training), validation_pair_queries=validation.size,
                     evaluation_queries=evaluation.size, total_queries=len(all_indices),
                     unique_archive_rows_queried=len(np.unique(all_indices)), archive_census_rows=length,
                     query_to_census_ratio=len(all_indices) / length,
                     unique_to_census_ratio=len(np.unique(all_indices)) / length,
                     feature_archive_construction_already_accessed_full_archive=True)
    dump_new(HERE / 'sampling_receipt.json', dict(drawn_utc=utc(),
              sampling_indices_sha256=sha(HERE / 'sampling_indices.npz'), **query_cost))
    y, c = data['outcome'], data['category']
    ztrain, zval, zeval = c[training], c[validation[:, 0]], c[evaluation]
    gy = np.clip(group_means(ztrain, (rankdata(y[training], method='average') - 1) / (len(training) - 1)), 0, 1)
    val_y = (y[validation[:, 0]] > y[validation[:, 1]]) + .5 * (y[validation[:, 0]] == y[validation[:, 1]])
    val_counts = np.bincount(zval, minlength=4)
    val_mean_y = group_means(zval, val_y)
    results, fit_rows, diagnostic_inputs, component_rows = [], [], {}, []
    for name in CANDIDATES:
        x = data[name]
        fx = np.clip(group_means(ztrain, (rankdata(x[training], method='average') - 1) / (len(training) - 1)), 0, 1)
        val_x = (x[validation[:, 0]] > x[validation[:, 1]]) + .5 * (x[validation[:, 0]] == x[validation[:, 1]])
        val_mean_x = group_means(zval, val_x)
        budget = signed_category_budget(fx, gy, val_counts, val_mean_x, val_mean_y, delta=p['delta'])
        xx, yy, ff, gg = x[evaluation], y[evaluation], fx[zeval], gy[zeval]
        started = time.perf_counter()
        h = triple_scores(xx, yy, ff, gg)
        triple_seconds = time.perf_counter() - started
        started = time.perf_counter()
        groups = np.array([peer_rank_products(a, b, f, g)['corrected_residual_product'].mean()
                           for a, b, f, g in zip(xx, yy, ff, gg)])
        grouped_seconds = time.perf_counter() - started
        started = time.perf_counter()
        pooled = float(peer_rank_products(xx.ravel(), yy.ravel(), ff.ravel(), gg.ravel())['corrected_residual_product'].mean())
        pooled_seconds = time.perf_counter() - started
        definitions = [
            ('grouped_full_u_absolute_range', float(groups.mean()), budget['absolute_bias_upper'], 'range', grouped_seconds),
            ('grouped_full_u_signed_range', float(groups.mean()), budget['bias_upper'], 'range', grouped_seconds),
            ('grouped_full_u_signed_variance', float(groups.mean()), budget['bias_upper'], 'variance', grouped_seconds + triple_seconds),
            ('pooled_full_u_signed_range', pooled, budget['bias_upper'], 'range', pooled_seconds),
            ('pooled_full_u_signed_variance', pooled, budget['bias_upper'], 'variance', pooled_seconds + triple_seconds),
            ('independent_triples_signed_range', float(h.mean()), budget['bias_upper'], 'range', triple_seconds),
            ('independent_triples_signed_bernstein', float(h.mean()), budget['bias_upper'], 'independent_bernstein', triple_seconds)]
        for method, mean, allowance, kind, seconds in definitions:
            result = one_sided_certificate(mean, h, len(h), fx, gy, allowance,
                                           delta=p['delta'], alpha=p['alpha'], kind=kind)
            results.append(dict(candidate=name, method=method, primary=method==p['primary'].split(':')[0],
                                computational_seconds=seconds, **query_cost, **result))
        for category in range(4):
            fit_rows.append(dict(candidate=name, category=category, f=float(fx[category]), g=float(gy[category]),
                                 training_count=int(np.bincount(ztrain, minlength=4)[category]),
                                 validation_count=int(val_counts[category]), validation_mean_x=float(val_mean_x[category]),
                                 validation_mean_y=float(val_mean_y[category]),
                                 lower_x=float(budget['lower_v'][category]), upper_x=float(budget['upper_v'][category]),
                                 lower_y=float(budget['lower_w'][category]), upper_y=float(budget['upper_w'][category]),
                                 category_mass_upper=float(budget['category_mass_upper'][category]),
                                 signed_cell_upper=float(budget['cell_signed_upper'][category])))
        for index, value in enumerate(h):
            component_rows.append(dict(candidate=name, triple=index, score=float(value)))
        diagnostic_inputs[name] = (fx, gy, budget)
    output = pd.DataFrame(results)
    harmonic = 1 + 1/2 + 1/3
    for method, group in output.groupby('method', sort=False):
        order = group.sort_values('p', kind='stable').index.to_numpy()
        scaled = output.loc[order, 'p'].to_numpy() * 3 * harmonic / np.arange(1, 4)
        q = np.minimum(1., np.minimum.accumulate(scaled[::-1])[::-1])
        output.loc[order, 'p_by'] = q
        output.loc[order, 'by_retain'] = q <= .05
    output.to_csv(HERE / 'certificate_results.csv', index=False, float_format='%.17g')
    pd.DataFrame(fit_rows).to_csv(HERE / 'nuisance_validation.csv', index=False, float_format='%.17g')
    pd.DataFrame(component_rows).to_csv(HERE / 'triple_scores.csv.gz', index=False,
                                      compression={'method': 'gzip', 'mtime': 0}, float_format='%.17g')
    dump_new(HERE / 'certificate_written.json', dict(written_utc=utc(),
        certificate_results_sha256=sha(HERE / 'certificate_results.csv'),
        census_not_yet_computed=True))
    # Exact census is deliberately last; its ranks and means never enter fits.
    census_started = time.perf_counter()
    uy = (rankdata(y, method='average') - .5) / length
    my = group_means(c, uy)
    mass = np.bincount(c, minlength=4) / length
    census_rows = []
    for name in CANDIDATES:
        ux = (rankdata(data[name], method='average') - .5) / length
        mx = group_means(c, ux)
        theta = float(np.mean((ux - mx[c]) * (uy - my[c])))
        fx, gy, budget = diagnostic_inputs[name]
        exact_bias = float(np.dot(mass, (mx - fx) * (my - gy)))
        fitted_target = float(np.mean((ux - fx[c]) * (uy - gy[c])))
        assert np.isclose(fitted_target, theta + exact_bias, atol=1e-14)
        if name == 'categorical_copy_control':
            assert abs(theta) < 1e-14
        for row in output[output.candidate == name].itertuples():
            census_rows.append(dict(candidate=name, method=row.method, census_theta=theta,
                census_fitted_moment=fitted_target, exact_learning_bias=exact_bias,
                signed_allowance=float(budget['bias_upper']), absolute_allowance=float(budget['absolute_bias_upper']),
                signed_bias_covered=exact_bias <= budget['bias_upper'],
                confidence_lower_bound=row.lower_bound, observed_lower_covers=row.lower_bound <= theta,
                p=row.p, p_by=row.p_by, by_retain=bool(row.by_retain)))
    pd.DataFrame(census_rows).to_csv(HERE / 'census_results.csv', index=False, float_format='%.17g')
    census_seconds = time.perf_counter() - census_started
    files = ['sampling_indices.npz', 'sampling_receipt.json', 'certificate_results.csv',
             'nuisance_validation.csv', 'triple_scores.csv.gz', 'certificate_written.json', 'census_results.csv']
    receipt = dict(completed_utc=utc(), protocol_sha256=sha(HERE / 'protocol.json'),
        archive_sha256=seal['archive_sha256'], source_sha256=p['source_sha256'],
        output_sha256={name: sha(HERE / name) for name in files},
        census_rank_and_diagnostic_seconds=census_seconds, query_cost=query_cost,
        no_retuning=True, census_computed_after_certificate_written=True,
        scope='One randomized finite-archive certificate application, not future forecasting certification or an empirical calibration experiment.')
    dump_new(HERE / 'audit_receipt.json', receipt)
    print(output[['candidate', 'method', 'mean', 'bias_upper', 'radius', 'lower_bound', 'p', 'p_by', 'by_retain']].to_string(index=False))
    print(json.dumps(dict(census_seconds=census_seconds, query_cost=query_cost)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['freeze', 'prepare', 'audit'])
    parser.add_argument('--source', type=Path, default=SOURCE)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        globals()[args.phase](args.source) if args.phase in ('freeze', 'prepare') else audit()
