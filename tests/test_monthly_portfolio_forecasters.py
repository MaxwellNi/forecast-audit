"""Independent hand-calculated calendar, target-isolation and maturity tests."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from monthly_portfolio_forecasters import FEATURES, calendar_frame, matured_training, preprocessing, rule_predictions


def fixture(periods=38):
    return pd.DataFrame({"signalname": "A", "port": "LS", "date": pd.date_range("2010-01-31", periods=periods, freq="ME"), "ret": np.arange(1, periods + 1, dtype=float)})


class MonthlyPortfolioTests(unittest.TestCase):
    def test_exact_next_month_target_and_complete_baseline_through_origin(self):
        frame = calendar_frame(fixture())
        self.assertAlmostEqual(frame.loc[11, "y"], 13)
        self.assertAlmostEqual(frame.loc[11, "baseline"], 100 * (np.prod(1 + np.arange(1, 13) / 100) - 1))
        self.assertEqual(frame.loc[11, "target_maturity"], pd.Timestamp("2011-01-31"))
        self.assertTrue(np.isnan(frame.loc[10, "baseline"]))
        self.assertTrue(np.isnan(frame.iloc[-1].y))

    def test_gap_is_retained_and_never_compacted(self):
        raw = fixture().drop(index=14)
        frame = calendar_frame(raw)
        self.assertEqual(len(frame), 38)
        self.assertTrue(np.isnan(frame.loc[13, "y"]))
        self.assertTrue(np.isnan(frame.loc[14, "return_current"]))
        self.assertTrue(frame.loc[14:25, "baseline"].isna().all())
        self.assertTrue(np.isfinite(frame.loc[26, "baseline"]))

    def test_future_returns_cannot_change_current_inputs(self):
        raw = fixture()
        frame = calendar_frame(raw)
        altered = raw.copy()
        altered.loc[24:, "ret"] = 98765
        revised = calendar_frame(altered)
        np.testing.assert_allclose(frame.loc[:23, FEATURES + ["baseline", "exponential_12m"]], revised.loc[:23, FEATURES + ["baseline", "exponential_12m"]], equal_nan=True)
        self.assertNotEqual(frame.loc[23, "y"], revised.loc[23, "y"])

    def test_annual_training_ends_at_previous_november_origin(self):
        frame = calendar_frame(fixture(50))
        cutoff = pd.Timestamp("2012-12-31")
        train = frame.loc[matured_training(frame, cutoff)]
        self.assertEqual(train.period.max(), pd.Timestamp("2012-11-30"))
        self.assertEqual(train.target_maturity.max(), cutoff)

    def test_preprocessing_uses_training_only_and_preserves_targets(self):
        frame = calendar_frame(fixture(50))
        train = frame.iloc[11:24].copy()
        test = frame.iloc[24:30].copy()
        x, _, y, state = preprocessing(train, test)
        changed = test.copy()
        changed[FEATURES] = 1e6
        changed["y"] = np.nan
        xx, _, yy, second = preprocessing(train, changed)
        np.testing.assert_array_equal(x, xx)
        np.testing.assert_array_equal(y, yy)
        self.assertEqual(state, second)
        self.assertTrue(changed.y.isna().all())

    def test_rules_exact_baseline_seasonality_and_exponential_formula(self):
        frame = calendar_frame(fixture(50))
        train, test = frame.iloc[11:24], frame.iloc[24:27]
        pred = rule_predictions(train, test)
        np.testing.assert_array_equal(pred["Trailing twelve-month return"], test.baseline)
        np.testing.assert_array_equal(pred["Seasonal twelve-month lag"], np.array([14, 15, 16]))
        weights = .8 ** np.arange(12)
        expected = np.arange(25, 13, -1) @ weights / weights.sum()
        self.assertAlmostEqual(pred["Exponential twelve-month mean"][0], expected)

    def test_duplicate_calendar_keys_are_rejected(self):
        raw = fixture()
        with self.assertRaises(ValueError):
            calendar_frame(pd.concat([raw, raw.iloc[[3]]], ignore_index=True))


if __name__ == "__main__":
    unittest.main()
