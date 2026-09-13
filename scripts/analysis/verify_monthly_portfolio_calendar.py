"""Independent raw-key/window and saved-prediction checks for monthly portfolios.

No forecasting/preparation helper is imported. Numeric reconstruction uses
explicit calendar-key lookups, direct products, and stored fitted parameters.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def sha(path):
    d = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            d.update(b)
    return d.hexdigest()


def equal(a, b, tolerance=1e-8):
    a, b = np.asarray(a, float), np.asarray(b, float)
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=tolerance, equal_nan=True)
    valid = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else 0.0


def main(args):
    frame = pd.read_parquet(args.output / "calendar_panel.parquet")
    protocol = json.loads((args.output / "protocol.json").read_text())
    assert sha(args.input) == protocol["input"]["sha256"]
    raw = pd.read_parquet(args.input, columns=["signalname", "port", "date", "ret"])
    raw = raw.loc[raw.port.eq("LS")].copy()
    raw["period"] = pd.to_datetime(raw.date).dt.to_period("M").dt.to_timestamp("M")
    assert not raw.duplicated(["signalname", "period"]).any()
    series = raw.set_index(["signalname", "period"]).ret
    lags = []
    discrepancies = {}
    for i in range(24):
        keys = pd.MultiIndex.from_arrays([frame.entity, frame.period - pd.offsets.MonthEnd(i)])
        values = series.reindex(keys).to_numpy()
        discrepancies[f"return_lag_{i}"] = equal(values, frame[f"return_lag_{i}"])
        lags.append(values)
    future = pd.MultiIndex.from_arrays([frame.entity, frame.period + pd.offsets.MonthEnd(1)])
    discrepancies["next_calendar_month_target"] = equal(series.reindex(future), frame.y)
    assert frame.target_maturity.equals(frame.period + pd.offsets.MonthEnd(1))
    x = np.column_stack(lags)
    for h in [3, 6, 12, 24]:
        discrepancies[f"compound_return_{h}m"] = equal(100 * (np.prod(1 + x[:, :h] / 100, axis=1) - 1), frame[f"compound_return_{h}m"])
    for h in [12, 24]:
        discrepancies[f"volatility_{h}m"] = equal(np.std(x[:, :h], axis=1, ddof=1), frame[f"volatility_{h}m"])
    weights = .8 ** np.arange(12)
    weights /= weights.sum()
    discrepancies["exponential_12m"] = equal(x[:, :12] @ weights, frame.exponential_12m)
    equal(frame.baseline, frame.compound_return_12m, tolerance=0)
    tests = ["Every target equals an explicit raw next-calendar-month key lookup", "Every one of thirty features equals direct past-calendar-key formulas", "Baseline uses exactly twelve known monthly returns through origin", "Unavailable windows and targets remain missing"]
    fit_checks, forecast_checks, audit_checks = [], {}, []
    if (args.output / "receipt.json").exists():
        receipt = json.loads((args.output / "receipt.json").read_text())
        forecasts = {q.model.iloc[0]: q for q in (pd.read_parquet(p) for p in sorted((args.output / "local_predictions").glob("*.parquet")))}
        assert len(forecasts) == 10
        expected = frame.loc[frame.period.dt.year.between(2013, 2023) & frame.y.notna() & frame.baseline.notna()]
        for name, q in forecasts.items():
            assert set(q.row_id) == set(expected.row_id)
            assert not q.row_id.duplicated().any()
            check = q.merge(frame[["row_id", "y", "baseline", "period"]], on="row_id", suffixes=("_saved", "_raw"), validate="one_to_one")
            equal(check.y_saved, check.y_raw, 0)
            equal(check.baseline_saved, check.baseline_raw, 0)
            assert check.period_saved.equals(check.period_raw)
        for fold in receipt["folds"]:
            year = fold["test_year"]
            cutoff = pd.Timestamp(f"{year-1}-12-31")
            train = frame.loc[frame.period.ge("1990-01-31") & frame.period.le(cutoff) & frame.target_maturity.le(cutoff) & frame.y.notna() & frame.baseline.notna()]
            test = expected.loc[expected.period.dt.year.eq(year)]
            assert len(train) == fold["training_rows"] and len(test) == fold["test_rows"]
            assert train.period.max().month == 11 and train.target_maturity.max() <= cutoff
            state = json.loads((args.output / "fitted_parameters" / f"{year}_preprocessing.json").read_text())
            train_x = train[state["features"]].replace([np.inf, -np.inf], np.nan)
            medians = train_x.median().fillna(0)
            filled = train_x.fillna(medians)
            means = filled.mean()
            scales = filled.std(ddof=0).replace(0, 1)
            equal(state["medians"], medians)
            equal(state["means"], means)
            equal(state["scales"], scales)
            equal([state["target_mean"], state["target_scale"]], [train.y.mean(), train.y.std(ddof=0)])
            xt = (test[state["features"]].replace([np.inf, -np.inf], np.nan).fillna(medians) - means) / scales
            learned = ["Ridge", "Elastic net", "LightGBM", "Small MLP"]
            values = {
                "Last month": test.return_current,
                "Trailing three-month return": test.compound_return_3m,
                "Trailing twelve-month return": test.baseline,
                "Seasonal twelve-month lag": test.return_lag_11,
                "Training portfolio mean": test.entity.map(train.groupby("entity").y.mean()).fillna(train.y.mean()),
                "Exponential twelve-month mean": test.exponential_12m,
            }
            for index, name in enumerate(learned, start=7):
                # Trusted locally produced model files whose source and outputs
                # are part of this protocol, never arbitrary user pickle input.
                with (args.output / "fitted_parameters" / f"{year}_model_{index:02}.pkl").open("rb") as f:
                    model = pickle.load(f)
                values[name] = model.predict(xt.to_numpy()) * state["target_scale"] + state["target_mean"]
            for name, prediction in values.items():
                saved = forecasts[name].set_index("row_id").loc[test.row_id].prediction
                error = equal(saved, prediction)
                forecast_checks[name] = max(forecast_checks.get(name, 0.0), error)
            fit_checks.append({"year": year, "training_rows": len(train), "test_rows": len(test), "cutoff": str(cutoff.date()), "latest_training_origin": str(train.period.max().date())})
        for filename, groups in [("extrapolated_profile.csv", ["beta", "lag", "fold_scheme"]), ("single_resolution_profile.csv", ["lag", "fold_scheme"])]:
            table = pd.read_csv(args.output / filename)
            for spec, group in table.groupby(groups):
                assert len(group) == group.model.nunique() == 10
                ordered = group.sort_values("p_one_sided", kind="stable")
                p = ordered.p_one_sided.to_numpy()
                q = np.minimum.accumulate((p * 10 * np.sum(1 / np.arange(1, 11)) / np.arange(1, 11))[::-1])[::-1].clip(max=1)
                equal(ordered.by_adjusted_p, q)
                assert np.array_equal(ordered.by_reject.to_numpy(), q <= .05)
                audit_checks.append({"file": filename, "specification": str(spec), "models": len(group), "nominal_rejections": int(group.by_reject.sum())})
        tests += ["All ten saved vectors have exactly the prespecified common raw targets and keys", "Every annual fit uses only labels matured by preceding December", "Every imputation and scaling parameter independently recomputed from training data", "All ten predictor outputs reproduced from rules or saved actual fitted parameters", "Every nominal BY adjustment independently recomputed over the full ten-model family"]
    result = {"protocol_sha256": sha(args.output / "protocol.json"), "verifier_sha256": sha(Path(__file__)), "rows_verified": len(frame), "raw_long_short_rows": len(raw), "checks": tests, "independent_calendar_discrepancies": discrepancies, "annual_fit_checks": fit_checks, "saved_prediction_max_absolute_discrepancies": forecast_checks, "complete_family_audit_checks": audit_checks, "status": "Passed independent raw-key/calendar/parameter checks; no inferential calibration claim"}
    suffix = "independent_final.json" if fit_checks else "independent_pre_fit.json"
    (args.output / suffix).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checks": len(tests), "annual_fits_checked": len(fit_checks), "maximum_calendar_discrepancy": max(discrepancies.values()), "status": "passed"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        main(args)
