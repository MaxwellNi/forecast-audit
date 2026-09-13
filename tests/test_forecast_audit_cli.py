"""Input integrity, observed-redundancy and complete-family output checks."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import audit_panel_predictions as core
from forecast_audit_cli import (audit_frame, main, validate_frame,
                                 validate_specification, read_input, REQUIRED)


def fixture():
    rng = np.random.default_rng(908713)
    n = 12 * 8
    period = np.repeat(np.arange(12), 8)
    entity = np.tile(np.arange(8), 12)
    baseline = rng.normal(size=n)
    y = .7 * baseline + rng.normal(size=n)
    models = {"copy": baseline, "same_order": np.exp(baseline), "reverse_order": -baseline,
              "constant": np.ones(n), "signal": y + rng.normal(size=n) * .2,
              "unrelated": rng.normal(size=n)}
    return pd.concat([pd.DataFrame({"model": name, "entity": entity, "period": period,
                                    "y": y, "prediction": prediction, "baseline": baseline})
                      for name, prediction in models.items()], ignore_index=True)


class ForecastAuditCliTests(unittest.TestCase):
    def test_model_identifiers_cannot_collapse_disjoint_cohorts(self):
        frame = fixture()
        first = frame[(frame.model == "copy") & (frame.period < 6)].copy()
        second = frame[(frame.model == "signal") & (frame.period >= 6)].copy()
        first["model"], second["model"] = "01", "1"
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "models.csv"
            pd.concat([first, second]).to_csv(source, index=False)
            read = read_input(source, {name: name for name in REQUIRED})
            self.assertEqual(set(read.model), {"01", "1"})
            with self.assertRaisesRegex(ValueError, "exactly the same"):
                audit_frame(read, frequency="cluster", lag=0, beta=2, ladder=(2, 3, 4))
            mapped = {name: "custom_" + name for name in REQUIRED}
            pd.concat([first, second]).rename(columns=mapped).to_csv(source, index=False)
            read = read_input(source, mapped)
            self.assertEqual(set(read.model), {"01", "1"})
            with self.assertRaisesRegex(ValueError, "exactly the same"):
                audit_frame(read, frequency="cluster", lag=0, beta=2, ladder=(2, 3, 4))

    def test_entity_and_cluster_numeric_spellings_remain_distinct(self):
        periods = ["01", "1", "1.0", "1e0", "02", "2", "10", "9007199254740992", "9007199254740993"]
        frame = pd.DataFrame([{"model": "one", "entity": entity, "period": period,
                               "y": j, "prediction": j, "baseline": 1-j}
                              for period in periods for j, entity in enumerate(["01", "1"])])
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "labels.csv"
            frame.to_csv(source, index=False)
            read = read_input(source, {name: name for name in REQUIRED})
            self.assertEqual(set(read.entity), {"01", "1"})
            self.assertEqual(set(read.period), set(periods))
            checked, _ = validate_frame(read, "cluster", 0)
            self.assertEqual(checked.period.nunique(), 9)
            self.assertEqual(checked.entity.nunique(), 2)
            self.assertEqual(checked.period.cat.categories.tolist(), periods)
            folds = core.period_folds(checked.period)
            self.assertEqual(pd.DataFrame({"period": read.period, "fold": folds}).groupby("period").fold.nunique().max(), 1)

    def test_NA_like_model_identifiers_are_retained_and_empty_is_rejected(self):
        base = fixture().query("model == 'copy'").copy()
        frame = pd.concat([base.assign(model=name) for name in ["NA", "NULL", "null", "None"]], ignore_index=True)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "names.csv"
            frame.to_csv(source, index=False)
            read = read_input(source, {name: name for name in REQUIRED})
            self.assertEqual(set(read.model), {"NA", "NULL", "null", "None"})
            tables = audit_frame(read, frequency="cluster", lag=0, beta=2, ladder=(2, 3, 4))
            self.assertTrue(tables["profile"].policy_family_size.eq(4).all())
            self.assertTrue(tables["profile"].final_label.eq("ABSTAIN").all())
            read.loc[0, "model"] = ""
            with self.assertRaises(ValueError):
                validate_frame(read, "cluster", 0)

    def test_csv_cannot_silently_reinterpret_surplus_fields_or_duplicate_headers(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bad.csv"
            for text in ("model,entity,period,y,prediction,baseline\nextra,m,1,2,3,4,5\n",
                         "model,entity,period,y,prediction,baseline\nm,1,2,3,4\n",
                         "model,entity,period,y,prediction,prediction\nm,1,2,3,4,5\n"):
                source.write_text(text)
                with self.assertRaises(ValueError):
                    read_input(source, {name: name for name in REQUIRED})

    def test_reject_invalid_explicit_specifications(self):
        for kwargs in ({"beta": 0}, {"beta": float("inf")}, {"beta": float("nan")},
                       {"lag": -1}, {"lag": True}, {"lag": 1},
                       {"ladder": (2, 2)}, {"ladder": (4, 2)}, {"ladder": (2, 3.5)},
                       {"ladder": (2,)}, {"beta": 1e6}):
            specification = dict(frequency="cluster", lag=0, beta=2., ladder=(2, 3, 4), alpha=.05)
            specification.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validate_specification(**specification)

    def test_reject_bad_rows_missing_cohort_and_insufficient_clusters(self):
        frame = fixture()
        changes = [pd.concat([frame, frame.iloc[:1]], ignore_index=True),
                   frame.drop(index=0), frame[frame.period < 4].copy()]
        for column, value in (("prediction", float("nan")), ("baseline", float("inf")),
                              ("y", "bad-number"), ("entity", ""), ("period", float("inf"))):
            changed = frame.copy()
            changed[column] = changed[column].astype(object)
            changed.loc[0, column] = value
            changes.append(changed)
        changed = frame.copy()
        changed.loc[0, "y"] += 1
        changes.append(changed)
        for changed in changes:
            with self.subTest(rows=len(changed)), self.assertRaises((ValueError, TypeError)):
                validate_frame(changed, "cluster", 0)

    def test_reject_calendar_aliases_and_out_of_range_lag(self):
        frame = fixture()
        frame["period"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(frame.period, unit="h")
        with self.assertRaises(ValueError):
            validate_frame(frame, "D", 0)
        frame = fixture()
        frame["period"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(frame.period, unit="D")
        with self.assertRaises(ValueError):
            validate_frame(frame, "D", 12)

    def test_all_guards_full_family_and_unchanged_core_diagnostic(self):
        frame = fixture()
        tables = audit_frame(frame, frequency="cluster", lag=0, beta=2, ladder=(2, 3, 4))
        profile = tables["profile"].set_index("model")
        for model in ("copy", "same_order", "reverse_order", "constant"):
            self.assertEqual(profile.loc[model, "final_label"], "ABSTAIN")
            self.assertEqual(profile.loc[model, "p_policy"], 1.)
        self.assertEqual(profile.loc["copy", "guard_reason"], "exact_observed_baseline_copy")
        self.assertEqual(profile.loc["signal", "final_label"], "RETAIN")
        self.assertEqual(profile.loc["unrelated", "final_label"], "NOT_RETAINED")
        self.assertTrue((profile.policy_family_size == 6).all())
        oracle = multipletests(profile.p_policy, alpha=.05, method="fdr_by")
        assert_array_equal(profile.policy_by_reject, oracle[0])
        assert_allclose(profile.policy_by_adjusted_p, oracle[1], atol=1e-15)
        signal = frame[frame.model == "signal"].reset_index(drop=True)
        expected = core.audit_extrapolated_panel(signal, 0, None, qs=(2, 3, 4), betas=(2,))[0]
        assert_allclose(profile.loc["signal", ["mean_product", "standard_error", "statistic", "p_one_sided"]].to_numpy(float),
                        [expected[k] for k in ("mean_product", "standard_error", "statistic", "p_one_sided")], atol=0, rtol=0)
        traces = tables["resolution_trace"]
        for model, part in traces.groupby("model"):
            self.assertAlmostEqual(part.weighted_mean_contribution.sum(), profile.loc[model, "mean_product"], places=12)
        self.assertEqual(len(tables["by_trace"]), 12)
        self.assertFalse(profile.finite_certificate_issued.any())
        for table in tables.values():
            self.assertNotIn("entity", table.columns)
            self.assertNotIn("period", table.columns)

    def test_undefined_outcome_scale_abstains_and_complete_cli_writes_aggregates(self):
        frame = fixture()
        frame["y"] = 1.
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source, output = folder / "input.csv", folder / "audit"
            frame.to_csv(source, index=False)
            self.assertEqual(main(["--input", str(source), "--output-dir", str(output),
                                   "--frequency", "cluster", "--lag", "0", "--beta", "2",
                                   "--ladder", "2,3,4"]), 0)
            profile = pd.read_csv(output / "profile.csv", keep_default_na=False)
            self.assertTrue(profile.final_label.eq("ABSTAIN").all())
            self.assertTrue(profile.p_policy.eq(1).all())
            self.assertTrue(profile.statistic.eq("").all())
            receipt = json.loads((output / "receipt.json").read_text())
            self.assertFalse(receipt["finite_certificate_issued"])
            self.assertEqual(receipt["family_size"], 6)
            self.assertEqual(set(p.name for p in output.iterdir()),
                             {"receipt.json", "profile.csv", "resolution_trace.csv", "by_trace.csv", "guard_evidence.csv", "cohorts.csv"})
            with self.assertRaises(SystemExit):
                main(["--input", str(source), "--output-dir", str(output),
                      "--frequency", "cluster", "--lag", "0", "--beta", "2", "--ladder", "2,3,4"])


if __name__ == "__main__":
    unittest.main()
