"""Calendar-aligned monthly forecasting of public long-short portfolio returns.

Forecasts target the next calendar month's published portfolio return. All
predictors use returns observed through the forecast month; fitted models use
only labels matured by their preceding-December annual training cutoff.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb

SEED = 2026090506
FEATURES = [f"return_lag_{i}" for i in range(24)] + [f"compound_return_{h}m" for h in (3, 6, 12, 24)] + ["volatility_12m", "volatility_24m"]
MODEL_NAMES = ["Last month", "Trailing three-month return", "Trailing twelve-month return", "Seasonal twelve-month lag", "Training portfolio mean", "Exponential twelve-month mean", "Ridge", "Elastic net", "LightGBM", "Small MLP"]


def calendar_frame(raw):
    required = ["signalname", "port", "date", "ret"]
    if any(c not in raw for c in required):
        raise ValueError("Expected signalname, port, date and ret")
    data = raw.loc[raw.port.eq("LS"), required].copy()
    if data.empty or data[["signalname", "date"]].isna().any().any():
        raise ValueError("Long-short portfolio keys must be present")
    data["month"] = pd.to_datetime(data.date, errors="raise").dt.to_period("M")
    data["ret"] = pd.to_numeric(data.ret, errors="raise")
    if np.isinf(data.ret).any() or data.duplicated(["signalname", "month"]).any():
        raise ValueError("Returns cannot be infinite and portfolio/month keys must be unique")
    result = []
    weights = .8 ** np.arange(11, -1, -1)
    weights /= weights.sum()
    for name, group in data.groupby("signalname", sort=True):
        indexed = group.set_index("month").ret.sort_index()
        dates = pd.period_range(indexed.index.min(), indexed.index.max(), freq="M")
        ret = indexed.reindex(dates)
        frame = pd.DataFrame({"entity": name, "period": dates.to_timestamp("M"), "return_current": ret.to_numpy(), "y": ret.shift(-1).to_numpy()})
        frame["target_maturity"] = frame.period + pd.offsets.MonthEnd(1)
        for i in range(24):
            frame[f"return_lag_{i}"] = ret.shift(i).to_numpy()
        for h in (3, 6, 12, 24):
            frame[f"compound_return_{h}m"] = ((1 + ret / 100).rolling(h, min_periods=h).apply(np.prod, raw=True).to_numpy() - 1) * 100
        for h in (12, 24):
            frame[f"volatility_{h}m"] = ret.rolling(h, min_periods=h).std(ddof=1).to_numpy()
        frame["exponential_12m"] = ret.rolling(12, min_periods=12).apply(lambda x: x @ weights, raw=True).to_numpy()
        frame["baseline"] = frame.compound_return_12m
        result.append(frame)
    return pd.concat(result, ignore_index=True).sort_values(["period", "entity"]).reset_index(drop=True)


def matured_training(frame, cutoff):
    return frame.period.ge("1990-01-31") & frame.period.le(cutoff) & frame.target_maturity.le(cutoff) & frame.y.notna() & frame.baseline.notna()


def preprocessing(train, test):
    x = train[FEATURES].replace([np.inf, -np.inf], np.nan)
    xt = test[FEATURES].replace([np.inf, -np.inf], np.nan)
    medians = x.median().fillna(0)
    x, xt = x.fillna(medians), xt.fillna(medians)
    means = x.mean()
    scales = x.std(ddof=0).replace(0, 1)
    target_mean = float(train.y.mean())
    target_scale = float(train.y.std(ddof=0)) or 1.0
    state = {"features": FEATURES, "medians": medians.tolist(), "means": means.tolist(), "scales": scales.tolist(), "target_mean": target_mean, "target_scale": target_scale}
    return ((x - means) / scales).to_numpy(), ((xt - means) / scales).to_numpy(), (train.y.to_numpy() - target_mean) / target_scale, state


def make_learners(threads=4):
    return {
        "Ridge": Ridge(alpha=10.0),
        "Elastic net": ElasticNet(alpha=.01, l1_ratio=.5, max_iter=5000, tol=1e-6, selection="cyclic"),
        "LightGBM": lgb.LGBMRegressor(n_estimators=200, num_leaves=15, min_child_samples=40, learning_rate=.03, n_jobs=threads, random_state=SEED, deterministic=True, force_col_wise=True, verbose=-1),
        "Small MLP": MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", solver="adam", alpha=.001, batch_size=512, learning_rate_init=.001, max_iter=50, shuffle=True, random_state=SEED, tol=0.0, n_iter_no_change=1000, early_stopping=False),
    }


def rule_predictions(train, test):
    means = train.groupby("entity").y.mean()
    result = {
        "Last month": test.return_lag_0.to_numpy(),
        "Trailing three-month return": test.compound_return_3m.to_numpy(),
        "Trailing twelve-month return": test.baseline.to_numpy().copy(),
        "Seasonal twelve-month lag": test.return_lag_11.to_numpy(),
        "Training portfolio mean": test.entity.map(means).fillna(train.y.mean()).to_numpy(),
        "Exponential twelve-month mean": test.exponential_12m.to_numpy(),
    }
    assert list(result) == MODEL_NAMES[:6]
    return result
