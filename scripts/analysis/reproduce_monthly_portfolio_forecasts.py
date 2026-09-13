"""Freeze, construct, fit and audit a public monthly portfolio benchmark."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from monthly_portfolio_forecasters import FEATURES, MODEL_NAMES, SEED, calendar_frame, matured_training, preprocessing, make_learners, rule_predictions
from public_audit import audit_family, sha


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def code_hashes():
    folder = Path(__file__).resolve().parent
    return {name: sha(folder / name) for name in [Path(__file__).name, "monthly_portfolio_forecasters.py"]}


def audit_code_hashes():
    folder = Path(__file__).resolve().parent
    return {name: sha(folder / name) for name in ["public_audit.py", "audit_panel_predictions.py", "cluster_covariance_reference.py"]}


def freeze(args):
    args.output.mkdir(parents=True, exist_ok=False)
    schema = pd.read_parquet(args.input, columns=["signalname", "port", "date"])
    ls = schema.loc[schema.port.eq("LS")]
    protocol = {
        "task": "Next-calendar-month public long-short portfolio return forecasting",
        "dataset": "Open Source Asset Pricing published portfolio returns",
        "dataset_relationship": "A forecasting task on the same public source used by the separate factor-alpha task; not an independent additional dataset.",
        "recorded_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "record_type": "Local specification frozen before new target evaluation or model fitting, not an external preregistration",
        "input": {"filename": args.input.name, "sha256": sha(args.input), "bytes": args.input.stat().st_size, "long_short_rows": len(ls), "long_short_series": int(ls.signalname.nunique()), "schema": {c: str(t) for c, t in schema.dtypes.items()}},
        "source_hashes": code_hashes(),
        "provenance": json.loads(args.provenance.read_text()) if args.provenance else {"official_home": "https://www.openassetpricing.com/", "official_exporter": "https://github.com/OpenSourceAP/CrossSection/blob/master/Portfolios/Code/20_PredictorPorts.R", "exact_release_identity": "not established by a filename"},
        "population": "All LS signal names in the fixed public extract, no selection on evaluation activity or outcome values. The historical discovery and revision vintages of this published signal roster are not certified.",
        "return_units": "Percent, unchanged from the official portfolio-return export; compound factors use 1 + ret/100.",
        "forecast_origin": "End of calendar month t, after that month's portfolio return is observed. This is an explicit retrospective observability assumption, not a verified vendor publication timestamp.",
        "target": "The supplied simple return at exactly calendar month t+1 in percent; missing calendar months remain missing. No future-price, future-factor, contemporaneous next-month return or accounting inputs.",
        "baseline": "Complete compounded return through origin over t-11,...,t: 100*(product(1+r/100)-1). Exact baseline predictor is retained in the ten-model family.",
        "features": FEATURES,
        "window_policy": "Each compound/volatility/exponential window requires every calendar return. Explicit monthly grid, no gap compaction. Thirty features: returns t through t-23, 3/6/12/24-month compounded returns, and 12/24-month sample volatility.",
        "training_start": "1990-01-31", "test_years": list(range(2013, 2024)),
        "annual_fit_cutoff": "Preceding December 31 after month-end. Train only origins with nonmissing target matured at or before cutoff and observed twelve-month baseline. Latest eligible training origin is preceding November. Annual fitted parameters remain fixed during the following year.",
        "evaluation_rows": "All origins in 2013-2023 with an observed exact next-calendar-month target and complete twelve-month baseline, shared by every model. No return clipping, no performance-based cohort choice, no missing-label imputation.",
        "preprocessing": "Feature finite-value medians, means and population standard deviations fitted on current training only; all-missing median=0 and zero standard deviation=1. Training target mean/std for learned regressors only, inverted for predictions.",
        "models": {
            "Last month": "r[t]", "Trailing three-month return": "complete compounded r[t-2:t]", "Trailing twelve-month return": "exact declared baseline", "Seasonal twelve-month lag": "r[t-11], twelve months before target t+1", "Training portfolio mean": "per-portfolio mean of eligible matured training targets, global training mean fallback", "Exponential twelve-month mean": "normalized weights proportional to 0.8**j on r[t-j], j=0,...,11; a finite twelve-month exponentially weighted mean",
            "Ridge": {"alpha": 10.0}, "Elastic net": {"alpha": .01, "l1_ratio": .5, "max_iter": 5000, "tol": 1e-6, "selection": "cyclic"},
            "LightGBM": {"n_estimators": 200, "num_leaves": 15, "min_child_samples": 40, "learning_rate": .03, "seed": SEED, "deterministic": True},
            "Small MLP": {"hidden": [32, 16], "activation": "ReLU", "optimizer": "Adam", "alpha": .001, "batch_size": 512, "learning_rate": .001, "epochs": 50, "early_stopping": False, "seed": SEED},
        },
        "budget": "44 learned-regressor fits (four models x eleven years), plus six fixed-rule/training-mean predictors; no hyperparameter search or seed selection",
        "threads": 4,
        "audit": {"family_size": 10, "calendar_frequency": "M", "whole_month_folds": 5, "fold_schemes": ["contiguous", "interleaved"], "hac_lag": 12, "betas": [1, 2], "primary_beta": 1, "qs": [8, 12, 16, 24, 32], "single_q": 10, "score_standard_error_floor": 1e-10, "floor_policy": "Numerically near-zero standard error in normalized rank-product units is undefined and assigned p=1; not a statistical calibration correction.", "source_lock": "Forecast sources are frozen here; audit source hashes are recorded separately at run start and checked again at completion.", "multiple_testing": "Benjamini-Yekutieli separately within the complete ten-model family for each prespecified specification", "interpretation": "Nominal normal-reference diagnostics only. Neither covariance assumptions nor nuisance bias or real-data inferential calibration is established."},
        "metrics": "Full-family MAE/RMSE in percentage points and pooled/equal-month Spearman; exact-baseline diagnostic retained regardless of result.",
    }
    dump(args.output / "protocol.json", protocol)
    print(json.dumps({"protocol_sha256": sha(args.output / "protocol.json"), "frozen_utc": protocol["recorded_utc"]}), flush=True)


def checked_protocol(args):
    p = json.loads((args.output / "protocol.json").read_text())
    if p["source_hashes"] != code_hashes() or p["input"]["sha256"] != sha(args.input):
        raise ValueError("Source or public input changed after protocol freeze")
    return p


def build(args):
    protocol = checked_protocol(args)
    output = args.output / "calendar_panel.parquet"
    if output.exists():
        raise FileExistsError("Existing prepared panel is preserved")
    frame = calendar_frame(pd.read_parquet(args.input))
    frame = frame.loc[frame.period.ge("1990-01-31") & frame.period.le("2023-12-31")].reset_index(drop=True)
    frame["row_id"] = np.arange(len(frame))
    frame.to_parquet(output, index=False)
    evaluation = frame.period.dt.year.between(2013, 2023)
    admissible = evaluation & frame.y.notna() & frame.baseline.notna()
    receipt = {"protocol_sha256": sha(args.output / "protocol.json"), "input_sha256": protocol["input"]["sha256"], "panel_sha256": sha(output), "panel_rows": len(frame), "series": int(frame.entity.nunique()), "evaluation_origins_on_calendar": int(evaluation.sum()), "evaluation_rows_with_observed_target_and_baseline": int(admissible.sum()), "evaluation_rows_without_target": int((evaluation & frame.y.isna()).sum()), "evaluation_rows_without_baseline": int((evaluation & frame.baseline.isna()).sum()), "evaluation_series": int(frame.loc[admissible].entity.nunique()), "evaluation_months": int(frame.loc[admissible].period.nunique()), "evaluation_rows_with_all_thirty_features": int(frame.loc[admissible, FEATURES].notna().all(axis=1).sum()), "source_release_vs_historical_observability": "A public downloaded release can establish numeric provenance; it does not establish which release or signal definitions were available at each historical origin."}
    dump(args.output / "preparation.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def run(args):
    protocol = checked_protocol(args)
    audit_sources = audit_code_hashes()
    started = time.perf_counter()
    prediction_dir = args.output / "local_predictions"
    prediction_dir.mkdir(exist_ok=False)
    fit_dir = args.output / "fitted_parameters"
    fit_dir.mkdir(exist_ok=False)
    prep = json.loads((args.output / "preparation.json").read_text())
    if sha(args.output / "calendar_panel.parquet") != prep["panel_sha256"]:
        raise ValueError("Prepared input changed")
    frame = pd.read_parquet(args.output / "calendar_panel.parquet")
    outputs = {name: [] for name in MODEL_NAMES}
    folds = []
    with threadpool_limits(limits=protocol["threads"]):
        for year in protocol["test_years"]:
            cutoff = pd.Timestamp(f"{year-1}-12-31")
            train = frame.loc[matured_training(frame, cutoff)]
            test = frame.loc[frame.period.dt.year.eq(year) & frame.y.notna() & frame.baseline.notna()]
            if len(train) < 1000 or test.empty:
                raise ValueError("Prespecified fold has insufficient data")
            assert train.target_maturity.max() <= cutoff < test.period.min()
            x, xt, y, state = preprocessing(train, test)
            state_path = fit_dir / f"{year}_preprocessing.json"
            dump(state_path, state)
            predictions = rule_predictions(train, test)
            runtimes, warning_log = {}, {}
            for index, (name, model) in enumerate(make_learners(protocol["threads"]).items()):
                tic = time.perf_counter()
                with warnings.catch_warnings(record=True) as notices:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(x, y)
                warning_log[name] = [str(w.message) for w in notices]
                predictions[name] = model.predict(xt) * state["target_scale"] + state["target_mean"]
                runtimes[name] = time.perf_counter() - tic
                with (fit_dir / f"{year}_model_{index + 7:02}.pkl").open("wb") as f:
                    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            np.testing.assert_array_equal(predictions["Trailing twelve-month return"], test.baseline)
            for name, prediction in predictions.items():
                if not np.isfinite(prediction).all():
                    raise ValueError(f"Nonfinite output: {name}")
                q = test[["row_id", "entity", "period", "target_maturity", "y", "baseline"]].copy()
                q["prediction"] = prediction
                q["model"] = name
                q["fit_cutoff"] = cutoff
                outputs[name].append(q)
            fold = {"test_year": year, "fit_cutoff": str(cutoff.date()), "training_rows": len(train), "test_rows": len(test), "latest_training_origin": str(train.period.max().date()), "latest_training_label_maturity": str(train.target_maturity.max().date()), "preprocessing_sha256": sha(state_path), "fit_runtime_seconds": runtimes, "fit_warnings": warning_log}
            folds.append(fold)
            print(json.dumps({k: fold[k] for k in ("test_year", "training_rows", "test_rows", "fit_runtime_seconds")}), flush=True)
        panels, errors = [], []
        for i, name in enumerate(MODEL_NAMES):
            q = pd.concat(outputs[name], ignore_index=True)
            q.to_parquet(prediction_dir / f"model_{i+1:02}.parquet", index=False)
            panels.append((name, q))
            ic = q.groupby("period").apply(lambda d: spearmanr(d.y, d.prediction).statistic, include_groups=False)
            errors.append({"model": name, "n": len(q), "mae": float(np.mean(np.abs(q.prediction-q.y))), "rmse": float(np.sqrt(np.mean((q.prediction-q.y)**2))), "pooled_spearman": float(spearmanr(q.y, q.prediction).statistic), "mean_monthly_spearman": float(ic.mean()), "months": int(q.period.nunique()), "prediction_sha256": hashlib.sha256(q.prediction.to_numpy(dtype="<f8").tobytes()).hexdigest()})
        print(json.dumps({"stage": "all_ten_models_fitted", "test_keys": len(panels[0][1])}), flush=True)
        extrapolated, single = audit_family(panels, lag=12, frequency="M")
    extrapolated.to_csv(args.output / "extrapolated_profile.csv", index=False)
    single.to_csv(args.output / "single_resolution_profile.csv", index=False)
    pd.DataFrame(errors).to_csv(args.output / "model_errors.csv", index=False)
    if code_hashes() != protocol["source_hashes"]:
        raise ValueError("Source mutated during run")
    if audit_sources != audit_code_hashes():
        raise ValueError("Audit source mutated during run")
    receipt = {"task": protocol["task"], "dataset": protocol["dataset"], "dataset_relationship": protocol["dataset_relationship"], "protocol_sha256": sha(args.output / "protocol.json"), "source_hashes": code_hashes(), "audit_source_hashes": audit_sources, "input_sha256": protocol["input"]["sha256"], "prepared_panel_sha256": prep["panel_sha256"], "preparation": prep, "folds": folds, "learned_regressor_fits": 44, "model_family_size": 10, "model_errors": errors, "audit": protocol["audit"], "dependencies": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "pandas", "scikit-learn", "lightgbm", "statsmodels", "pyarrow"]}, "elapsed_seconds": time.perf_counter()-started, "interpretation": "Retrospective chronological public-data forecasting and nominal full-family diagnostics; no historical signal-discovery-vintage or statistical-certification claim", "numeric_provenance": protocol["provenance"], "row_level_outputs_published": False}
    dump(args.output / "receipt.json", receipt)
    print(json.dumps({"complete": True, "elapsed_seconds": receipt["elapsed_seconds"], "learned_regressor_fits": 44, "family_size": 10}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "build", "run"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()
    {"freeze": freeze, "build": build, "run": run}[args.stage](args)
