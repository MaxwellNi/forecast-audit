#!/usr/bin/env python3
"""Independent raw/calendar recount plus future-target perturbation checks.

Recounts use the original quarter-hour rows and a positional 96-slot reshape,
without calling the producer's daily resampler. Isolation checks really refit
all eleven model algorithms twice on each of two raw-data subpanels, changing
only future observations. Reports are new files and refuse overwrite.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRODUCER = ROOT / "scripts/analysis/electricity_forecasters.py"
spec = importlib.util.spec_from_file_location("electricity_reconstruction", PRODUCER)
producer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = producer
spec.loader.exec_module(producer)


def independent_daily(raw_path):
    with zipfile.ZipFile(raw_path) as archive, archive.open("LD2011_2014.txt") as handle:
        raw = pd.read_csv(handle, sep=";", decimal=",", usecols=range(49))
    timestamps = pd.to_datetime(raw.iloc[:, 0])
    assert len(raw) == 140256
    assert timestamps.iloc[0] == pd.Timestamp("2011-01-01 00:15:00")
    assert timestamps.iloc[95] == pd.Timestamp("2011-01-02 00:00:00")
    assert timestamps.iloc[-2] == pd.Timestamp("2014-12-31 23:45:00")
    assert timestamps.iloc[-1] == pd.Timestamp("2015-01-01 00:00:00")
    assert timestamps.equals(pd.Series(pd.date_range(timestamps.iloc[0], timestamps.iloc[-1], freq="15min"), name=timestamps.name))
    matrix = raw.iloc[95:-1, 1:].to_numpy().reshape(1460, 96, 48)
    assert np.isfinite(matrix).all()
    daily = pd.DataFrame(matrix.mean(axis=1), index=pd.date_range("2011-01-02", "2014-12-31", freq="D"), columns=raw.columns[1:])
    # Ordinary resampling uses compensated summation; the independent reduction
    # may differ by floating rounding. These are arithmetic equivalence checks.
    return daily, {"raw_rows": len(raw), "days": len(daily), "meters": 48,
                   "excluded_leading_rows": 95, "excluded_trailing_rows": 1,
                   "method": "Drop explicitly checked edge slots; reshape middle rows into1460x96x48 and take the96-slot arithmetic mean."}


def check_output(daily, output):
    predictions = pd.read_csv(output / "new_predictions.csv.gz", parse_dates=["date", "target_date"])
    dates = pd.DatetimeIndex(sorted(predictions.date.unique()))
    expected = pd.date_range("2013-01-01", "2014-12-30", freq="D")
    assert dates.equals(expected)
    assert len(dates) == 729
    assert (predictions.target_date - predictions.date).eq(pd.Timedelta(days=1)).all()
    assert set(predictions.model) == set(producer.MODELS)
    assert not predictions.duplicated(["model", "date", "meter"]).any()
    model_counts = predictions.groupby("model").size()
    assert model_counts.eq(729 * 48).all()
    row = daily.index.get_indexer(predictions.target_date)
    col = daily.columns.get_indexer(predictions.meter)
    assert (row >= 0).all() and (col >= 0).all()
    target = daily.to_numpy()[row, col]
    np.testing.assert_allclose(predictions.y, target, rtol=1e-13, atol=1e-9)
    baseline_row = daily.index.get_indexer(predictions.target_date - pd.Timedelta(days=7))
    baseline = daily.to_numpy()[baseline_row, col]
    np.testing.assert_allclose(predictions.baseline, baseline, rtol=1e-13, atol=1e-9)
    seasonal = predictions.loc[predictions.model.eq("seasonal_naive_7")]
    np.testing.assert_array_equal(seasonal.prediction, seasonal.baseline)
    fit_metadata = json.loads((output / "fit_metadata.json").read_text())
    constant_meters = [m for m in daily if daily.loc[daily.index.year < 2013, m].nunique() == 1]
    assert constant_meters == ["MT_012", "MT_015", "MT_030", "MT_039", "MT_041"]
    constant_cases, optimized_cases, theta_constant_origins = 0, 0, 0
    for key, metadata in fit_metadata.items():
        if "/" not in key:
            continue
        year, meter = key.split("/")
        train_values = daily.loc[daily.index.year < int(year), meter]
        for name in ["ses", "holt_winters"]:
            record = metadata[name]
            if train_values.nunique() == 1:
                assert record["fit_status"] == "constant_training_exact_forecast_no_optimizer"
                assert record["optimizer_attempted"] is False
                selected = predictions.loc[predictions.model.eq(name) & predictions.year.eq(int(year)) & predictions.meter.eq(meter)]
                np.testing.assert_array_equal(selected.prediction, np.full(len(selected), train_values.iloc[0]))
                constant_cases += 1
            else:
                assert record["fit_status"] == "optimized_training_fit"
                assert record["optimizer_success"] is True
                optimized_cases += 1
        assert not metadata["warnings"], (key, metadata["warnings"])
        theta_constant_origins += metadata["theta"]["constant_history_no_optimizer_origins"]
    errors = []
    recorded = pd.read_csv(output / "error_metrics.csv")
    for (model, year), frame in predictions.groupby(["model", "year"]):
        err = frame.y.to_numpy() - frame.prediction.to_numpy()
        record = recorded.loc[recorded.model.eq(model) & recorded.year.eq(year)].iloc[0]
        np.testing.assert_allclose([np.abs(err).mean(), np.sqrt(np.mean(err * err))], [record.mae, record.rmse], rtol=1e-13)
        errors.append(float(np.max(np.abs(frame.y.to_numpy() - target[frame.index]))))
    return {"model_count": len(model_counts), "rows_per_model": int(model_counts.iloc[0]),
            "origin_days": len(dates), "target_calendar_verified": True,
            "seasonal_baseline_calendar_verified": True, "all22_error_metric_rows_recomputed": True,
            "max_daily_target_difference_from_independent_raw_mean": max(errors),
            "predictions_sha256": producer.sha(output / "new_predictions.csv.gz"),
            "constant_training_meters": constant_meters,
            "constant_no_optimizer_annual_smoothing_cases": constant_cases,
            "successful_nonconstant_annual_smoothing_optimizations": optimized_cases,
            "nonconstant_failed_optimization_fallbacks_used": 0,
            "theta_constant_origin_shortcuts": theta_constant_origins,
            "univariate_warning_count": 0}


def all_model_forecasts(daily, year):
    history, train, test, metadata = producer.prepare_split(daily, year)
    outputs, _ = producer.train_pooled(history, train, test, year)
    keys = test[["date", "meter"]].reset_index(drop=True)
    for name in ["ses", "holt_winters", "theta"]:
        outputs[name] = np.empty(len(test))
    for meter in history:
        origins = pd.DatetimeIndex(test.loc[test.meter.eq(meter), "date"])
        _, _, forecasts, _ = producer.series_job((year, meter, history[meter].to_numpy(), history.index, origins))
        positions = np.flatnonzero(test.meter.eq(meter))
        for name, values in forecasts.items():
            outputs[name][positions] = values
    return keys, outputs, train, test, metadata


def perturbation_tests(daily):
    results = []
    # Two annual refit boundaries and two real meters. All historical training
    # observations are retained; only the held-out January horizon is shortened.
    for year in [2013, 2014]:
        panel = daily.loc[:f"{year}-01-15", ["MT_001", "MT_012"]].copy()
        change_day = pd.Timestamp(year, 1, 8)
        altered = panel.copy()
        future = altered.index >= change_day
        altered.loc[future] = altered.loc[future] * 17.0 + 100000.0
        akeys, af, atrain, atest, ameta = all_model_forecasts(panel, year)
        bkeys, bf, btrain, btest, bmeta = all_model_forecasts(altered, year)
        pd.testing.assert_frame_equal(akeys, bkeys)
        pd.testing.assert_frame_equal(atrain, btrain)
        assert ameta["training_medians"] == bmeta["training_medians"]
        earlier = akeys.date < change_day
        # This includes origin January7, whose target January8 was changed.
        target_changed = atest.date.eq(change_day - pd.Timedelta(days=1))
        assert np.all(atest.loc[target_changed, "y"].to_numpy() != btest.loc[target_changed, "y"].to_numpy())
        changed_later = []
        for name in producer.MODELS:
            np.testing.assert_array_equal(af[name][earlier], bf[name][earlier], err_msg=f"{year}/{name}: future target changed earlier prediction")
            changed_later.append(bool(np.any(af[name][~earlier] != bf[name][~earlier])))
            results.append({"year": year, "model": name, "earlier_predictions_checked": int(earlier.sum()),
                            "target_perturbation_start": str(change_day.date()),
                            "earlier_predictions_bitwise_identical": True,
                            "later_predictions_changed": changed_later[-1]})
        assert any(changed_later)
        # Missing history uses a training-only median, even if early observations
        # are absent. Neither backward-fill nor any later observed value is used.
        missing = panel.copy()
        missing.iloc[:3, 0] = np.nan
        missing.loc[change_day - pd.Timedelta(days=1), "MT_001"] = np.nan
        missing_b = missing.copy()
        missing_b.loc[missing_b.index >= change_day] += 10000000
        h1, tr1, te1, m1 = producer.prepare_split(missing, year)
        h2, tr2, te2, m2 = producer.prepare_split(missing_b, year)
        np.testing.assert_array_equal(h1.loc[h1.index < change_day], h2.loc[h2.index < change_day])
        assert m1["training_medians"] == m2["training_medians"]
        assert h1.iloc[0, 0] == missing.loc[missing.index.year < year, "MT_001"].median()
        assert h1.loc[change_day - pd.Timedelta(days=1), "MT_001"] == m1["training_medians"]["MT_001"]
        assert not te1.loc[te1.meter.eq("MT_001"), "target_date"].eq(change_day - pd.Timedelta(days=1)).any()
        pd.testing.assert_frame_equal(tr1, tr2)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    protocol = json.loads(args.protocol.read_text())
    assert producer.sha(PRODUCER) == protocol["script_sha256"]
    assert producer.sha(protocol["raw_path"]) == protocol["raw_zip_sha256"]
    daily, raw_check = independent_daily(protocol["raw_path"])
    perturbations = perturbation_tests(daily)
    output_check = check_output(daily, args.reconstruction)
    result = {"status": "passed", "elapsed_seconds": time.monotonic() - started,
              "verification_script_sha256": producer.sha(__file__), "producer_script_sha256": producer.sha(PRODUCER),
              "protocol_sha256": producer.sha(args.protocol), "raw_calendar": raw_check,
              "full_output_checks": output_check, "all11_future_target_perturbation": perturbations,
              "training_median_and_missing_target_policy_checks": "passed in both annual splits",
              "scope": "All production rows checked against independent raw daily means; all eleven algorithms actually refitted twice on two-meter real-data subpanels for each annual boundary. Perturbation results establish the tested temporal contract, not universal deployment guarantees or inferential calibration."}
    producer.write_new(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
