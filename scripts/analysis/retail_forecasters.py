"""Calendar and causal historical-series helpers used by the lagged-price weekly task."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb

import numpy as np

import pandas as pd

from sklearn.linear_model import Ridge

def lagged_average(frame: pd.DataFrame, width: int) -> pd.Series:
    return frame.groupby('series', sort=False)['sales'].transform(lambda values: values.shift(1).rolling(width, min_periods=width).mean())

def complete_calendar(calendar: pd.DataFrame, day_columns: list[str]) -> tuple[pd.DataFrame, dict]:
    calendar = calendar.copy()
    if 'd' not in calendar:
        dates = pd.to_datetime(calendar.date)
        if not dates.diff().dropna().eq(pd.Timedelta(days=1)).all():
            raise ValueError('Cannot reconstruct day IDs from a nonconsecutive calendar')
        numbers = (dates - pd.Timestamp('2011-01-29')).dt.days + 1
        calendar['d'] = [f'd_{i}' for i in numbers]
    observed = calendar[calendar.d.isin(day_columns)]
    if len(observed) != len(day_columns) or observed.d.duplicated().any() or observed.date.duplicated().any():
        raise ValueError('Every sales day needs exactly one calendar record')
    counts = observed.groupby('wm_yr_wk').size()
    incomplete = {str(k): int(v) for (k, v) in counts[counts != 7].items()}
    return (observed[observed.wm_yr_wk.isin(counts[counts == 7].index)].copy(), incomplete)

def croston_sba_series(y, alpha=0.1):
    n = len(y)
    f = np.full(n, np.nan)
    z = p = None
    q = 1
    for t in range(n):
        if z is not None:
            f[t] = (1 - alpha / 2.0) * z / p
        if y[t] > 0:
            if z is None:
                (z, p) = (float(y[t]), float(q))
            else:
                z = z + alpha * (y[t] - z)
                p = p + alpha * (q - p)
            q = 1
        else:
            q += 1
    return f

def theta_weekly_series(y, n_train):
    n = len(y)
    t_idx = np.arange(n, dtype=float)
    A = np.column_stack([np.ones(n_train), t_idx[:n_train]])
    (coef, *_) = np.linalg.lstsq(A, y[:n_train], rcond=None)
    trend = coef[0] + coef[1] * t_idx
    theta2 = 2.0 * y - trend
    alphas = np.arange(0.05, 1.0, 0.05)
    lv = np.full(len(alphas), theta2[0])
    sse = np.zeros(len(alphas))
    for t in range(1, n_train):
        sse += (theta2[t] - lv) ** 2
        lv = alphas * theta2[t] + (1 - alphas) * lv
    a = float(alphas[int(np.argmin(sse))])
    f = np.full(n, np.nan)
    level = theta2[0]
    for t in range(1, n):
        f[t] = 0.5 * (trend[t] + level)
        level = a * theta2[t] + (1 - a) * level
    return f
