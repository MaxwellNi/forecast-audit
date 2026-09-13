"""Public forecasting tasks with features and cohorts fixed by training time.

These are new temporal evaluations, not reproductions of the earlier per-user
rating split or target-week retail-price task. Source download vintages and
all possible sampling dependence remain separate questions.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
import rating_forecasters as ratings
from retail_forecasters import complete_calendar, lagged_average, croston_sba_series, theta_weekly_series


def split_ratings(frame, cutoff="2017-01-01", end="2018-01-01", min_movie=200,
                  min_user=60, max_users=6000, seed=2026090503):
    """Determine eligibility solely from rows strictly before a global cutoff."""
    required = ["userId", "movieId", "rating", "timestamp"]
    if frame[required].isna().any().any():
        raise ValueError("rating inputs must be complete")
    if frame.duplicated(["userId", "movieId"]).any():
        raise ValueError("duplicate user/movie keys")
    lower = int(pd.Timestamp(cutoff, tz="UTC").timestamp())
    upper = int(pd.Timestamp(end, tz="UTC").timestamp())
    train = frame.loc[frame.timestamp < lower].copy()
    movies = train.movieId.value_counts()
    eligible_movies = movies[movies >= min_movie].index
    train = train[train.movieId.isin(eligible_movies)]
    users = train.userId.value_counts()
    eligible_users = np.sort(users[users >= min_user].index.to_numpy())
    if len(eligible_users) > max_users:
        eligible_users = np.sort(np.random.default_rng(seed).choice(eligible_users, max_users, replace=False))
    train = train[train.userId.isin(eligible_users)].copy()
    # Keep the movie cohort fixed from pre-cutoff eligibility. Some selected
    # movies may have no selected-user training observations; model fallback
    # behavior is explicit and does not look at evaluation ratings.
    test = frame.loc[(frame.timestamp >= lower) & (frame.timestamp < upper)
                     & frame.userId.isin(eligible_users) & frame.movieId.isin(eligible_movies)].copy()
    for part in (train, test):
        part.sort_values(["userId", "timestamp", "movieId"], inplace=True)
    if train.empty or test.empty or train.timestamp.max() >= test.timestamp.min():
        raise ValueError("global temporal split is empty or overlaps")
    return train, test, {"cutoff_utc": cutoff, "evaluation_end_exclusive_utc": end,
        "training_rows": len(train), "test_rows": len(test),
        "selected_training_users": len(eligible_users), "evaluation_users": int(test.userId.nunique()),
        "eligible_training_movies": len(eligible_movies), "fitted_training_movies": int(train.movieId.nunique()),
        "maximum_training_timestamp": int(train.timestamp.max()),
        "minimum_test_timestamp": int(test.timestamp.min()),
        "singleton_evaluation_users": int((test.userId.value_counts() == 1).sum()),
        "evaluation_activity_filter": "none", "cohort_uses_evaluation_rows": False}


def fit_rating_split(train, test):
    """Use the same ten algorithms, with a frozen global chronological split."""
    np.random.seed(2026090503)
    fitted = ratings.fit_forecasters(train)
    forecasts = {ratings.MODEL_NAMES[k]: ratings.predict(test, fitted, k) for k in
        ("item_mean", "user_mean", "baseline", "svd_cf", "svd_interaction")}
    matrix, observed, *_ = ratings.dense_matrices(train, fitted["uids"], fitted["mids"])
    forecasts["Global mean"] = np.full(len(test), fitted["mu"], float)
    forecasts["Slope One"] = ratings.pred_slope_one(train, test, fitted, matrix, observed)
    forecasts["Item nearest neighbours"] = ratings.pred_knn_item(train, test, fitted, matrix, observed)
    forecasts["Alternating least squares"] = ratings.pred_als(train, test, fitted, seed=2026090503)
    forecasts["SVD (32 factors)"] = ratings.pred_svd_k32(train, test, fitted)
    baseline = test.movieId.map(fitted["item"]).fillna(fitted["mu"]).to_numpy()
    np.testing.assert_array_equal(forecasts["Item mean"], baseline)
    panel = pd.DataFrame({"entity": test.movieId.to_numpy(), "period": test.userId.to_numpy(),
        "y": test.rating.to_numpy(), "baseline": baseline})
    return panel, forecasts


def prepare_retail_tables(sales, calendar, prices, n_series=4000, test_weeks=28, seed=2026090504):
    """Complete weekly sales, training-only cohort, and strictly lagged prices."""
    day_columns = [c for c in sales if c.startswith("d_")]
    calendar, incomplete = complete_calendar(calendar, day_columns)
    weeks = np.sort(calendar.wm_yr_wk.unique())
    if len(weeks) <= test_weeks:
        raise ValueError("not enough training weeks")
    cutoff = int(weeks[-test_weeks])
    sales = sales.copy()
    sales["series"] = sales.store_id + "_" + sales.item_id
    if sales.series.duplicated().any() or prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any():
        raise ValueError("nonunique retail keys")
    prices = prices.copy()
    prices["series"] = prices.store_id + "_" + prices.item_id
    valid_price = np.isfinite(prices.sell_price) & (prices.sell_price > 0)
    if (~valid_price & prices.sell_price.notna()).any():
        raise ValueError("nonpositive or nonfinite observed price")
    train_prices = prices.loc[(prices.wm_yr_wk < cutoff) & valid_price]
    history_count = train_prices.groupby("series").wm_yr_wk.nunique()
    train_days = calendar.loc[calendar.wm_yr_wk < cutoff, "d"].tolist()
    positive_history = sales[train_days].sum(axis=1) > 0
    eligible = sales.loc[positive_history & sales.series.isin(history_count[history_count >= 8].index)].copy()
    eligible.sort_values("series", inplace=True)
    eligible = eligible.sample(min(n_series, len(eligible)), random_state=seed).reset_index(drop=True)
    if eligible.empty:
        raise ValueError("empty training-eligible cohort")
    rows=[]
    for week, days in calendar.groupby("wm_yr_wk", sort=True):
        part=eligible[["series", "store_id", "item_id"]].copy()
        part["wm_yr_wk"]=week
        part["sales"]=eligible[days.d.tolist()].sum(axis=1).to_numpy()
        part["month"]=int(pd.to_datetime(days.date).iloc[0].month)
        part["period"]=pd.to_datetime(days.date).max()
        rows.append(part)
    frame=pd.concat(rows, ignore_index=True).merge(
        prices[["series", "wm_yr_wk", "sell_price"]], on=["series", "wm_yr_wk"],
        how="left", validate="many_to_one").sort_values(["series", "wm_yr_wk"]).reset_index(drop=True)
    frame["lag1"]=frame.groupby("series").sales.shift(1)
    frame["ma4"]=lagged_average(frame, 4)
    frame["ma8"]=lagged_average(frame, 8)
    frame["ses03"]=frame.groupby("series").sales.transform(lambda s:s.ewm(alpha=.3,adjust=False).mean().shift(1))
    price_lag1=frame.groupby("series").sell_price.shift(1)
    price_lag2=frame.groupby("series").sell_price.shift(2)
    medians=train_prices.groupby("series").sell_price.median()
    fallback=float(train_prices.sell_price.median())
    scale=frame.series.map(medians).fillna(fallback)
    frame["price_lag1_rel"]=price_lag1.fillna(scale)/scale
    frame["price_change_lag1"]=(price_lag1/price_lag2-1).fillna(0.)
    frame["month_sin"]=np.sin(2*np.pi*frame.month/12)
    frame["month_cos"]=np.cos(2*np.pi*frame.month/12)
    frame=frame.loc[frame.lag1.notna()].reset_index(drop=True)
    evidence={"cutoff_week":cutoff, "complete_weeks":len(weeks), "partial_weeks_excluded":incomplete,
        "series":len(eligible), "training_rows":int((frame.wm_yr_wk<cutoff).sum()),
        "test_rows":int((frame.wm_yr_wk>=cutoff).sum()), "test_weeks":test_weeks,
        "target_week_price_used":False, "evaluation_price_missingness_filters_rows":False,
        "training_eligibility":"positive total training sales and at least eight observed training price weeks",
        "calendar_features":"month sine/cosine only; no SNAP or unpublished future price plan"}
    return frame, cutoff, evidence


def fit_retail_frame(frame, cutoff, threads=4):
    train=frame[frame.wm_yr_wk<cutoff].copy()
    test=frame[frame.wm_yr_wk>=cutoff].copy()
    calendar_features=["price_lag1_rel", "price_change_lag1", "month_sin", "month_cos"]
    full_features=calendar_features+["lag1", "ma4"]
    options=dict(n_estimators=200,num_leaves=31,verbose=-1,n_jobs=threads,
        random_state=2026090504,deterministic=True,force_col_wise=True)
    calendar_model=lgb.LGBMRegressor(**options).fit(train[calendar_features],train.sales)
    full_model=lgb.LGBMRegressor(**options).fit(train[full_features].fillna(0),train.sales)
    forecasts={"Last week":test.lag1.to_numpy(),"Four week mean":test.ma4.fillna(test.lag1).to_numpy(),
        "Past price and calendar":calendar_model.predict(test[calendar_features]),
        "Past price calendar and history":full_model.predict(test[full_features].fillna(0)),
        "Training series mean":test.series.map(train.groupby("series").sales.mean()).fillna(train.sales.mean()).to_numpy(),
        "Eight week mean":test.ma8.fillna(test.lag1).to_numpy(),
        "Exponential smoothing":test.ses03.fillna(test.lag1).to_numpy()}
    croston=np.full(len(frame),np.nan); theta=np.full(len(frame),np.nan)
    for _,group in frame.groupby("series",sort=False):
        indices=group.index.to_numpy(); values=group.sales.to_numpy(float)
        n_train=int((group.wm_yr_wk<cutoff).sum())
        croston[indices]=croston_sba_series(values)
        if n_train>=8:theta[indices]=theta_weekly_series(values,n_train)
    for name,values in [("Croston SBA",croston),("Theta",theta)]:
        forecasts[name]=np.where(np.isfinite(values[test.index]),values[test.index],test.lag1)
    forecasts["Ridge with history"]=Ridge(alpha=1).fit(train[full_features].fillna(0),train.sales).predict(test[full_features].fillna(0))
    panel=pd.DataFrame({"entity":test.series.to_numpy(),"period":test.period.to_numpy(),
        "y":test.sales.to_numpy(),"baseline":test.lag1.to_numpy()})
    return panel,forecasts
