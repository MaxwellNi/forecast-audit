"""Factor-family regressions with calendar HAC and complete-family BY.

Missing months receive zero regression scores on the calendar covariance grid.
This preserves calendar lags; it does not establish ignorable missingness or
validity of the normal approximation. Alpha signs remain explicit.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from audit_panel_predictions import by_adjust, calendar_scores, bartlett_score_covariance


def read_monthly_factors(path, columns):
    rows = []
    for line in Path(path).read_text().splitlines():
        match = re.match(r"^\s*(\d{6})\s*,(.*)$", line)
        if match:
            values = [float(v.strip()) for v in match[2].split(",")[:len(columns)]]
            rows.append([int(match[1]), *values])
    result = pd.DataFrame(rows, columns=["yyyymm", *columns])
    if result.yyyymm.duplicated().any() or not np.isfinite(result[columns]).all().all():
        raise ValueError("invalid monthly factor rows")
    if ((result[columns] == -99.99) | (result[columns] == -999)).any().any():
        raise ValueError("factor file contains a missing-value sentinel")
    return result


def audit_factor_family(data_dir, lags=(6, 12, 24)):
    data_dir = Path(data_dir)
    ff = read_monthly_factors(data_dir / "F-F_Research_Data_5_Factors_2x3.csv",
                              ["MktRF", "SMB", "HML", "RMW", "CMA", "RF"])
    momentum = read_monthly_factors(data_dir / "F-F_Momentum_Factor.csv", ["Mom"])
    factors = ff.merge(momentum, on="yyyymm", validate="one_to_one")
    ports = pd.read_parquet(data_dir / "predictor_ports_full.parquet")
    ls = ports.loc[ports.port.astype(str) == "LS"].copy()
    ls["date"] = pd.to_datetime(ls.date)
    ls["yyyymm"] = ls.date.dt.year * 100 + ls.date.dt.month
    if ls.duplicated(["signalname", "yyyymm"]).any():
        raise ValueError("duplicate signal-month input")
    columns = ["MktRF", "SMB", "HML", "RMW", "CMA", "Mom"]
    rows = []
    maximum_covariance_error = 0.0
    family = sorted(ls.signalname.unique())
    for signal in family:
        data = ls.loc[ls.signalname == signal].merge(factors, on="yyyymm", validate="one_to_one")
        data = data.dropna(subset=["ret", *columns]).sort_values("date")
        if len(data) < 60:
            raise ValueError(f"prespecified signal has fewer than 60 usable months: {signal}")
        x = np.column_stack([np.ones(len(data)), data[columns].to_numpy(float)])
        y = data.ret.to_numpy(float)
        coefficient, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        if rank != x.shape[1] or not np.isfinite(y).all():
            raise ValueError("rank-deficient factor design or invalid return")
        residual = y - x @ coefficient
        scores, gaps = calendar_scores(x * residual[:, None], data.date, "M")
        bread = np.linalg.inv(x.T @ x)
        # Independent statsmodels HAC uses the same chronological calendar:
        # absent months have a zero design row and zero outcome, hence zero score.
        index = pd.PeriodIndex(data.date, freq="M")
        grid = pd.period_range(index.min(), index.max(), freq="M")
        positions = grid.get_indexer(index)
        padded_x = np.zeros((len(grid), x.shape[1])); padded_y = np.zeros(len(grid))
        padded_x[positions] = x; padded_y[positions] = y
        for lag in lags:
            covariance = bread @ bartlett_score_covariance(scores, lag) @ bread
            oracle = sm.OLS(padded_y, padded_x, hasconst=True).fit(
                cov_type="HAC", cov_kwds={"maxlags":lag, "use_correction":False}, use_t=False)
            error = float(np.max(np.abs(covariance - oracle.cov_params())))
            maximum_covariance_error = max(maximum_covariance_error, error)
            np.testing.assert_allclose(covariance, oracle.cov_params(), rtol=2e-8, atol=2e-12)
            se = float(np.sqrt(covariance[0, 0]))
            t = float(coefficient[0] / se)
            rows.append({"signal": signal, "lag": lag, "n":len(data), "calendar_gaps":gaps,
                         "alpha":float(coefficient[0]), "alpha_sign":"positive" if coefficient[0] > 0 else "negative",
                         "standard_error":se, "statistic":t, "p_two_sided":float(2 * norm.sf(abs(t)))})
    result = pd.DataFrame(rows)
    for lag in lags:
        mask = result.lag == lag
        p = result.loc[mask, "p_two_sided"].to_numpy()
        rejected, adjusted = by_adjust(p)
        oracle = multipletests(p, method="fdr_by", alpha=0.05)
        np.testing.assert_array_equal(rejected, oracle[0])
        np.testing.assert_allclose(adjusted, oracle[1], rtol=1e-13, atol=1e-14)
        result.loc[mask, "by_reject"] = rejected
        result.loc[mask, "by_adjusted_p"] = adjusted
    return result, {"family_size":len(family), "full_family_before_screening":True,
                    "maximum_statsmodels_covariance_abs_error":maximum_covariance_error,
                    "statsmodels_by_match":True}


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    result, metadata = audit_factor_family(args.data_dir)
    result.to_csv(args.output_dir / "profile.csv",index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(metadata,indent=2)+"\n")
