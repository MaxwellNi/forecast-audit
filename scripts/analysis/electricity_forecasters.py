#!/usr/bin/env python3
"""Daily UCI load forecasts with annual fitting and chronological features.

Freeze a local protocol with --freeze before --run. Both refuse overwrites.
Annual models learn only labels observed before January 1; daily histories may
then update through each origin. This establishes a temporal data contract,
not calibrated inference or an independently verified historical download.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/electricityloaddiagrams20112014.zip"
MEMBER = "LD2011_2014.txt"
SEED = 20260905
MODELS = {
    "seasonal_naive_7": "Seasonal naive (7 days)",
    "naive_1": "Previous daily mean",
    "drift": "Previous daily mean plus training drift",
    "ridge_lag": "Ridge on lag, rolling mean and calendar features",
    "histgbr_lag": "Histogram gradient boosting on historical features",
    "linear_lag": "Pooled ordinary least squares on five lags",
    "lasso_lag": "Pooled lasso with chronological training validation",
    "dlinear": "Two linear heads after moving-average decomposition",
    "ses": "Simple exponential smoothing with constant-training rule",
    "holt_winters": "Additive weekly smoothing with constant-training rule",
    "theta": "Expanding weekly Theta with exact constant-history rule",
}
LAGS = ["lag1", "lag2", "lag7", "lag14", "lag28"]
FEATURES = ["lag1", "lag7", "lag14", "lag28", "roll7", "roll28", "dow", "month"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def source_daily(path):
    """Read fixed header meters and retain complete nominal clock-date bins.

    Portuguese local labels are kept naive because the source already encodes
    clock changes on a fixed 96-slot grid. Zero readings are data, not missing.
    Partial edge dates are excluded; interior bad days remain missing, for a
    training-fitted feature imputer. Targets are never imputed.
    """
    with zipfile.ZipFile(path) as z, z.open(MEMBER) as f:
        raw = pd.read_csv(f, sep=";", decimal=",", usecols=range(49))
    stamp = pd.to_datetime(raw.iloc[:, 0], errors="raise")
    if stamp.duplicated().any() or not stamp.is_monotonic_increasing:
        raise ValueError("raw timestamps must be unique and sorted")
    if not (stamp.diff().iloc[1:] == pd.Timedelta(minutes=15)).all():
        raise ValueError("source must use a complete nominal quarter-hour grid")
    values = raw.iloc[:, 1:].astype(float).set_axis(pd.DatetimeIndex(stamp))
    if list(values.columns) != [f"MT_{i:03d}" for i in range(1, 49)]:
        raise ValueError("unexpected fixed header meter identifiers")
    grouped = values.resample("D", closed="left", label="left")
    size, counts = grouped.size(), grouped.count()
    daily = grouped.mean().where(counts.eq(96))
    valid_calendar = size.eq(96)
    first, last = size[valid_calendar].index[[0, -1]]
    daily = daily.loc[first:last]
    if not daily.index.equals(pd.date_range(first, last, freq="D")):
        raise ValueError("daily index has a calendar gap")
    legacy = grouped.mean()
    filled = legacy.ffill().bfill()
    meta = {
        "raw_rows": len(raw), "meters": list(values.columns),
        "first_timestamp": str(stamp.iloc[0]), "last_timestamp": str(stamp.iloc[-1]),
        "raw_missing_cells": int(values.isna().sum().sum()),
        "raw_zero_cells": int(values.eq(0).sum().sum()),
        "historical_fill_changed_cells": int((legacy.ne(filled) & ~(legacy.isna() & filled.isna())).sum().sum()),
        "nominal_quarter_hour_grid_verified": True,
        "raw_rows_by_day_count": {str(k): int(v) for k, v in size.value_counts().items()},
        "excluded_incomplete_edges": {str(k.date()): int(v) for k, v in size[~valid_calendar].items() if k < first or k > last},
        "daily_start": str(first.date()), "daily_end": str(last.date()),
        "daily_rows": len(daily), "daily_missing_cells": int(daily.isna().sum().sum()),
        "source_page": "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014",
        "source_doi": "10.24432/C58C86", "source_license": "CC BY 4.0",
        "source_page_checked_utc_date": "2026-09-05",
        "clock_contract": "Nominal Portuguese local timestamp-date mean over [00:00, 24:00); 96 supplied slots, without timezone/DST conversion.",
        "value_contract": "Arithmetic mean of supplied kW measurements; zeros retained. This is not a reconstructed physical 23/25-hour energy total.",
    }
    return daily, meta


def prepare_split(daily, year):
    cutoff = pd.Timestamp(year, 1, 1)
    training_daily = daily.loc[daily.index < cutoff]
    medians = training_daily.median()
    if medians.isna().any():
        raise ValueError("meter lacks training observations; fixed population cannot silently change")
    history = daily.fillna(medians)
    parts = []
    for meter in daily:
        s = history[meter]
        f = pd.DataFrame({"date": daily.index, "target_date": daily.index + pd.Timedelta(days=1),
                          "meter": meter, "y": daily[meter].shift(-1).to_numpy(),
                          "roll7": s.rolling(7).mean().to_numpy(),
                          "roll28": s.rolling(28).mean().to_numpy(),
                          "dow": daily.index.dayofweek, "month": daily.index.month})
        for lag in [1, 2, 7, 14, 28]:
            f[f"lag{lag}"] = s.shift(lag - 1).to_numpy()
        parts.append(f)
    frame = pd.concat(parts, ignore_index=True).dropna(subset=["y"] + FEATURES + LAGS)
    frame = frame.sort_values(["date", "meter"]).reset_index(drop=True)
    train = frame.loc[frame.target_date < cutoff].copy()
    test = frame.loc[frame.date.dt.year.eq(year)].copy()
    assert train.target_date.max() < cutoff <= test.date.min()
    meta = {"year": year, "fit_cutoff_exclusive": str(cutoff),
            "latest_training_origin": str(train.date.max()),
            "latest_training_target": str(train.target_date.max()),
            "training_rows": len(train), "test_rows": len(test),
            "test_origin_start": str(test.date.min()), "test_origin_end": str(test.date.max()),
            "test_target_start": str(test.target_date.min()), "test_target_end": str(test.target_date.max()),
            "training_medians": {str(k): float(v) for k, v in medians.items()},
            "feature_imputed_cells": int(daily.isna().sum().sum())}
    return history, train, test, meta


def fit_dlinear(history, train, test, year):
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    means = history.loc[history.index.year < year].mean()
    scales = history.loc[history.index.year < year].std() + 1e-8
    arr = ((history - means) / scales).to_numpy(dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(arr, 28, axis=0)
    meter_index = {m: j for j, m in enumerate(history.columns)}
    def x(rows):
        i = history.index.get_indexer(rows.date) - 27
        j = np.array([meter_index[m] for m in rows.meter])
        assert (i >= 0).all()
        return np.ascontiguousarray(windows[i, j])
    Xtr, Xte = x(train), x(test)
    ytr = ((train.y.to_numpy() - train.meter.map(means).to_numpy()) / train.meter.map(scales).to_numpy()).astype(np.float32)
    class DLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.trend, self.season = nn.Linear(28, 1), nn.Linear(28, 1)
        def forward(self, z):
            padded = torch.cat([z[:, :1].repeat(1, 12), z, z[:, -1:].repeat(1, 12)], dim=1)
            trend = nn.functional.avg_pool1d(padded[:, None, :], 25, stride=1).squeeze(1)
            return self.trend(trend) + self.season(z - trend)
    model = DLinear()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    xx, yy = torch.tensor(Xtr), torch.tensor(ytr)[:, None]
    generator = torch.Generator().manual_seed(SEED)
    model.train()
    for _ in range(30):
        perm = torch.randperm(len(xx), generator=generator)
        for start in range(0, len(xx), 512):
            idx = perm[start:start + 512]
            optimizer.zero_grad()
            nn.functional.mse_loss(model(xx[idx]), yy[idx]).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xte)).ravel().numpy()
    state = {k: v.detach().numpy().tolist() for k, v in model.state_dict().items()}
    return out * test.meter.map(scales).to_numpy() + test.meter.map(means).to_numpy(), state


def train_pooled(history, train, test, year):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    outputs = {"seasonal_naive_7": test.lag7.to_numpy(), "naive_1": test.lag1.to_numpy()}
    slopes = history.loc[history.index.year < year].diff().mean()
    outputs["drift"] = test.lag1.to_numpy() + test.meter.map(slopes).to_numpy()
    models = {
        "ridge_lag": (make_pipeline(StandardScaler(), Ridge(alpha=10.0)), FEATURES),
        "histgbr_lag": (make_pipeline(StandardScaler(), HistGradientBoostingRegressor(max_iter=160, max_leaf_nodes=31, learning_rate=.05, random_state=SEED, early_stopping=False)), FEATURES),
        "linear_lag": (LinearRegression(), LAGS),
    }
    # Split dates, not individual rows: no day straddles a CV train/validation boundary.
    dates = pd.DatetimeIndex(sorted(train.date.unique()))
    folds = [(np.flatnonzero(train.date.isin(dates[a]).to_numpy()),
              np.flatnonzero(train.date.isin(dates[b]).to_numpy()))
             for a, b in TimeSeriesSplit(n_splits=5, gap=1).split(dates)]
    lasso = GridSearchCV(make_pipeline(StandardScaler(), Lasso(max_iter=20000, tol=1e-5)),
                        {"lasso__alpha": np.logspace(-4, 2, 40)}, cv=folds,
                        scoring="neg_mean_squared_error", n_jobs=1, refit=True)
    models["lasso_lag"] = (lasso, LAGS)
    info = {"drift": {str(k): float(v) for k, v in slopes.items()}}
    for name, (model, features) in models.items():
        started = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(train[features], train.y)
        outputs[name] = np.asarray(model.predict(test[features]), float)
        info[name] = {"fit_seconds": time.monotonic() - started,
                      "warning_counts": dict(pd.Series([type(w.message).__name__ for w in caught], dtype=str).value_counts().astype(int).items())}
        if name == "lasso_lag":
            info[name]["alpha"] = float(model.best_params_["lasso__alpha"])
    outputs["dlinear"], info["dlinear_state"] = fit_dlinear(history, train, test, year)
    return outputs, info


def state_forecast(values, train_n, name):
    """Train initial states and coefficients once, then explicitly recurse."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
    if np.ptp(values[:train_n]) == 0:
        return np.full(len(values), float(values[0])), {
            "fit_status": "constant_training_exact_forecast_no_optimizer",
            "training_count": train_n, "training_range": 0.,
            "constant": float(values[0]), "optimizer_attempted": False,
            "forecast_rule": "Frozen annual constant, including origins after variation first appears."}
    kw = {"initialization_method": "estimated"}
    fit = (SimpleExpSmoothing(values[:train_n], **kw) if name == "ses" else
           ExponentialSmoothing(values[:train_n], trend="add", seasonal="add", seasonal_periods=7, **kw)).fit(optimized=True)
    if not fit.mle_retvals.get("success", True):
        # This deterministic rule uses only training histories, not forecast
        # errors. It preserves a finite causal path and marks it.
        if name == "ses":
            fallback = np.asarray(values, float).copy()
        else:
            fallback = np.asarray(values, float)[np.maximum(np.arange(len(values)) - 6, 0)]
        return fallback, {"fit_status": "nonconstant_optimizer_failure_causal_naive_fallback",
                          "optimizer_attempted": True, "optimizer_success": False,
                          "training_count": train_n, "training_range": float(np.ptp(values[:train_n])),
                          "fallback": "naive_1" if name == "ses" else "seasonal_naive_7"}
    p = fit.params
    alpha, level = float(p["smoothing_level"]), float(p["initial_level"])
    pred = np.empty(len(values))
    if name == "ses":
        for i, value in enumerate(values):
            level = alpha * value + (1 - alpha) * level
            pred[i] = level
    else:
        beta, gamma, trend = float(p["smoothing_trend"]), float(p["smoothing_seasonal"]), float(p["initial_trend"])
        seasons = list(np.asarray(p["initial_seasons"], float))
        for i, value in enumerate(values):
            old_level, old_trend, old_season = level, trend, seasons[i % 7]
            level = alpha * (value - old_season) + (1 - alpha) * (old_level + old_trend)
            trend = beta * (level - old_level) + (1 - beta) * old_trend
            seasons[i % 7] = gamma * (value - old_level - old_trend) + (1 - gamma) * old_season
            pred[i] = level + trend + seasons[(i + 1) % 7]
    # Library fit is an independent check of the hand-coded recursion on training data.
    np.testing.assert_allclose(pred[:train_n - 1], np.asarray(fit.fittedvalues)[1:], rtol=1e-9, atol=1e-7)
    np.testing.assert_allclose(pred[train_n - 1], np.asarray(fit.forecast(1))[0], rtol=1e-9, atol=1e-7)
    info = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v))
            for k, v in p.items() if k in ["smoothing_level", "smoothing_trend", "smoothing_seasonal", "initial_level", "initial_trend", "initial_seasons"] and (isinstance(v, np.ndarray) or np.isfinite(v))}
    info["optimizer_success"] = bool(fit.mle_retvals.get("success", True))
    info["optimizer_attempted"] = True
    info["fit_status"] = "optimized_training_fit"
    return pred, info


def series_job(job):
    year, meter, values, dates, origins = job
    from statsmodels.tsa.forecasting.theta import ThetaModel
    train_n = int((dates.year < year).sum())
    indices = dates.get_indexer(origins)
    out, metadata = {}, {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name in ["ses", "holt_winters"]:
            predictions, info = state_forecast(values, train_n, name)
            out[name], metadata[name] = predictions[indices], info
        predictions, constant_origins = [], 0
        for index in indices:
            if np.ptp(values[:index + 1]) == 0:
                predictions.append(float(values[index]))
                constant_origins += 1
                continue
            fit = ThetaModel(values[:index + 1], period=7).fit()
            predictions.append(float(np.asarray(fit.forecast(1))[0]))
        out["theta"] = np.asarray(predictions)
        metadata["theta"] = {"constant_history_no_optimizer_origins": constant_origins,
                             "nonconstant_expanding_fits": len(indices) - constant_origins}
    metadata["warnings"] = dict(pd.Series([type(w.message).__name__ for w in caught], dtype=str).value_counts().astype(int).items())
    return year, meter, out, metadata


def versions():
    import scipy, sklearn, statsmodels, torch
    return {"python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "statsmodels": statsmodels.__version__, "torch": torch.__version__}


def freeze(path, raw, workers):
    _, provenance = source_daily(raw)
    with zipfile.ZipFile(raw) as z, z.open(MEMBER) as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    extracted = raw.parent / MEMBER
    protocol = {
        "frozen_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "scope": "Locally frozen descriptive reconstruction after historical results were already known; not external preregistration and not statistical calibration.",
        "script_sha256": sha(__file__), "raw_zip_sha256": sha(raw),
        "archive_member_sha256": h.hexdigest(),
        "local_extracted_sha256": sha(extracted) if extracted.exists() else None,
        "raw_path": str(raw.resolve()), "seed": SEED, "years": [2013, 2014],
        "workers": workers, "models": MODELS, "environment": versions(), "source": provenance,
        "prediction_contract": "End of nominal origin date t forecasts arithmetic daily mean at t+1. Keep complete target days; origin year defines annual test split. Exclude incomplete first and last raw calendar days.",
        "training_contract": "Fixed annual pooled-model parameters use target_date < January 1. Meter population is fixed to first48 header IDs. Imputation medians and DLinear standardization use only daily values before January1. Histories update through each origin; targets are never imputed.",
        "training_and_evaluation_rules": ["Drop partial day targets", "Remove full-period population filtering", "Exclude annually immature labels", "Use complete historical rolling windows", "Chronological lasso CV with date blocks and one-day gap", "Explicit frozen-initial-state SES and Holt-Winters recursion", "CPU deterministic DLinear with seed20260905", "Disable histogram boosting random early stopping", "Exact constant annual-training SES/HW forecast and exact constant expanding-history Theta forecast; no unidentifiable optimizer"],
        "hyperparameters": {"ridge_alpha": 10., "histgbr": {"max_iter": 160, "max_leaf_nodes": 31, "learning_rate": .05, "early_stopping": False},
            "lasso": {"alphas": np.logspace(-4, 2, 40).tolist(), "folds": 5, "date_gap": 1, "scale_inside_cv": True, "max_iter": 20000, "tol": 1e-5},
            "dlinear": {"context": 28, "moving_average": 25, "epochs": 30, "batch": 512, "learning_rate": .001, "device": "cpu"},
            "theta": {"period": 7, "refit": "every nonconstant origin expanding through t; exact constant forecast for a constant history", "failure_fallback": None},
            "smoothing": "Constant annual training history gives that constant for the whole test year with no optimizer. Otherwise statsmodels train-only optimized initial states and coefficients; hand-coded filter verified against training fit. Nonconstant optimization failure uses an explicitly recorded causal naive1(SES) or seasonal_naive7(HW) rule."},
        "failure_policy": "Constant training histories use exact constant rules without optimization. Nonconstant SES/HW optimizer success=False uses labeled causal naive fallback; no test-error-based selection. Raised exceptions or non-finite predictions abort execution. Theta exception has no fallback.",
        "constant_history_policy_scope": "The constant-history policy was specified after observed optimization failures on all-zero training histories. It uses a deterministic training-only rule; this retrospective policy specification is not a prospective preregistration.",
        "verification_plan": ["Independent raw archive/member provenance hash", "Quarter-hour and complete-day calendar checks", "Target labels and all model row sets aligned", "Training-target maturity assertion", "Independent target perturbation tests for all eleven algorithms", "Explicit smoothing recursion checked against statsmodels training fit"],
        "auditor_plan": "Audit the generated forecasts using seasonal_naive_7 as the baseline; interpret all normal-reference outputs as nominal diagnostics.",
    }
    if protocol["archive_member_sha256"] != protocol["local_extracted_sha256"]:
        raise ValueError("local raw text differs from archived member")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, protocol)
    print(json.dumps({"protocol": str(path), "sha256": sha(path), "status": "frozen before new forecast generation"}), flush=True)


def run(protocol_path, out):
    started = time.monotonic()
    protocol = json.loads(protocol_path.read_text())
    if sha(__file__) != protocol["script_sha256"] or sha(protocol["raw_path"]) != protocol["raw_zip_sha256"]:
        raise ValueError("frozen implementation or raw-input hash mismatch")
    if versions() != protocol["environment"]:
        raise ValueError("environment differs from frozen protocol")
    out.mkdir(parents=True, exist_ok=False)
    daily, provenance = source_daily(protocol["raw_path"])
    daily.to_csv(out / "complete_daily_means.csv.gz", compression={"method": "gzip", "mtime": 0}, index_label="date")
    frames, splits, all_info, jobs, bases = [], [], {}, [], {}
    for year in protocol["years"]:
        history, train, test, meta = prepare_split(daily, year)
        splits.append(meta)
        base = test[["date", "target_date", "meter", "y"]].copy()
        base["year"] = year
        base["baseline"] = test.lag7.to_numpy()
        bases[year] = base
        forecasts, info = train_pooled(history, train, test, year)
        all_info[str(year)] = info
        for name, predictions in forecasts.items():
            f = base.copy()
            f["prediction"], f["model"], f["model_description"] = predictions, name, MODELS[name]
            frames.append(f)
        origins = pd.DatetimeIndex(sorted(test.date.unique()))
        for meter in history:
            jobs.append((year, meter, history[meter].to_numpy(), history.index, origins))
        print(json.dumps({"year": year, "pooled_models_refitted": len(forecasts), "elapsed_seconds": time.monotonic() - started}), flush=True)
    with ProcessPoolExecutor(max_workers=protocol["workers"]) as pool:
        for year, meter, forecasts, info in pool.map(series_job, jobs):
            all_info[f"{year}/{meter}"] = info
            base = bases[year].loc[bases[year].meter.eq(meter)].copy()
            for name, predictions in forecasts.items():
                f = base.copy()
                f["prediction"], f["model"], f["model_description"] = predictions, name, MODELS[name]
                frames.append(f)
    predictions = pd.concat(frames, ignore_index=True).sort_values(["model", "date", "meter"]).reset_index(drop=True)
    if not np.isfinite(predictions[["y", "prediction", "baseline"]].to_numpy()).all():
        raise ValueError("non-finite output")
    checks = {}
    reference = predictions.loc[predictions.model.eq("seasonal_naive_7"), ["date", "target_date", "meter", "y", "baseline"]].reset_index(drop=True)
    for model, rows in predictions.groupby("model"):
        pd.testing.assert_frame_equal(rows[["date", "target_date", "meter", "y", "baseline"]].reset_index(drop=True), reference)
        checks[model] = {"rows": len(rows), "same_rowset_target_and_baseline": True}
    if set(checks) != set(MODELS):
        raise ValueError("eleven-model family incomplete")
    predictions.to_csv(out / "new_predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    metrics = []
    for (name, year), rows in predictions.groupby(["model", "year"]):
        err = rows.y - rows.prediction
        metrics.append({"model": name, "year": int(year), "rows": len(rows), "days": rows.date.nunique(),
                        "mae": float(err.abs().mean()), "rmse": float(np.sqrt((err**2).mean()))})
    pd.DataFrame(metrics).to_csv(out / "error_metrics.csv", index=False)
    write_new(out / "fit_metadata.json", all_info)
    write_new(out / "receipt.json", {"status": "all eleven algorithms refitted or causally updated from raw observations",
        "elapsed_seconds": time.monotonic() - started, "protocol_sha256": sha(protocol_path),
        "raw_source": provenance, "splits": splits, "row_contracts": checks,
        "artifacts": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        "inference_boundary": "Forecast reconstruction and nominal diagnostic inputs only; no calibrated discoveries asserted."})
    if sha(__file__) != protocol["script_sha256"]:
        raise ValueError("implementation changed during execution")
    print(json.dumps({"status": "complete", "models": len(checks), "rows_per_model": len(reference), "elapsed_seconds": time.monotonic() - started}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", type=Path)
    group.add_argument("--run", type=Path, help="frozen protocol")
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze, args.raw, args.workers)
    elif args.output_dir is None:
        parser.error("--run requires --output-dir")
    else:
        run(args.run, args.output_dir)


if __name__ == "__main__":
    main()
