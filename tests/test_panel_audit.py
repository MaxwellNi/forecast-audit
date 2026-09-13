"""Regression tests for fold target isolation, calendar covariance, and full-family BY."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from audit_panel_predictions import (by_adjust, panel_residuals, period_folds,
                                    calendar_scores, bartlett_score_covariance, crossfit_additive_least_squares,
                                    audit_panel, audit_extrapolated_panel)
from cluster_covariance_reference import crossfit_additive_nuisance


class AuditTests(unittest.TestCase):
    def test_sparse_solve_against_dense_lstsq_and_backfitting(self):
        rng = np.random.default_rng(912)
        n = 360
        a = rng.integers(0, 8, n); b = rng.integers(0, 5, n)
        folds = np.arange(n) % 5
        y = rng.normal(size=n) + a + .3 * b
        sparse = crossfit_additive_least_squares(y, [a,b], folds, np.arange(n))
        design = np.column_stack([np.ones(n), np.eye(8)[a], np.eye(5)[b]])
        oracle = np.empty(n)
        for f in range(5):
            tr = folds != f
            coefficient = np.linalg.lstsq(design[tr], y[tr], rcond=None)[0]
            oracle[~tr] = design[~tr] @ coefficient
        np.testing.assert_allclose(sparse, oracle, atol=1e-9, rtol=1e-10)
        reference = crossfit_additive_nuisance(y, [a,b], folds, cluster_ids=np.arange(n))
        np.testing.assert_allclose(sparse, reference, atol=1e-8, rtol=1e-9)

    def test_complete_pipeline_target_isolation(self):
        rng = np.random.default_rng(914)
        n = 240
        frame = pd.DataFrame({"entity": np.tile(np.arange(12), 20),
                              "period": np.repeat(pd.date_range("2020-01-01", periods=20), 12),
                              "y": rng.normal(size=n), "prediction": rng.normal(size=n),
                              "baseline": rng.normal(size=n)})
        first = panel_residuals(frame)
        evaluation = first["folds"] == 2
        altered = frame.copy()
        altered.loc[evaluation, "y"] = rng.normal(size=evaluation.sum()) * 200
        second = panel_residuals(altered)
        np.testing.assert_array_equal(first["outcome_fit"][evaluation], second["outcome_fit"][evaluation])
        np.testing.assert_array_equal(first["forecast_fit"], second["forecast_fit"])

    def test_calendar_gap_and_direct_quadratic_form(self):
        periods = ["2020-01-31", "2020-01-31", "2020-03-31", "2020-04-30"]
        values = np.array([[1., 2.], [3., -1.], [7., 4.], [-2., 2.]])
        scores, gaps = calendar_scores(values, periods, "M")
        self.assertEqual(gaps, 1)
        np.testing.assert_array_equal(scores[1], [0., 0.])
        for lag in [0, 1, 2, 12]:
            kernel = np.array([[max(1 - abs(i-j)/(lag+1), 0) for j in range(4)] for i in range(4)])
            np.testing.assert_allclose(bartlett_score_covariance(scores, lag), scores.T @ kernel @ scores)

    def test_by_independent_and_no_screening(self):
        pvalues = np.array([0.00001, 0.04, 0.001, 0.9, 0.014, 0.011, 0.002])
        decision, adjusted = by_adjust(pvalues)
        oracle = multipletests(pvalues, alpha=0.05, method="fdr_by")
        np.testing.assert_array_equal(decision, oracle[0])
        np.testing.assert_allclose(adjusted, oracle[1])
        self.assertEqual(len(adjusted), len(pvalues))

    def test_fold_cluster_integrity(self):
        periods = pd.to_datetime(np.repeat(pd.date_range("2020-01-01", periods=13), np.arange(1,14)))
        for scheme in ["contiguous", "interleaved"]:
            folds = period_folds(periods, scheme)
            self.assertEqual(set(folds), set(range(5)))
            self.assertEqual(pd.DataFrame({"period":periods,"fold":folds}).groupby("period").fold.nunique().max(), 1)

    def test_extrapolation_retains_cross_resolution_covariance(self):
        rng = np.random.default_rng(162)
        frame = pd.DataFrame({"entity":np.tile(np.arange(10),30),"period":np.repeat(pd.date_range("2020-01-01",periods=30),10),
                              "y":rng.normal(size=300),"prediction":rng.normal(size=300),"baseline":rng.normal(size=300)})
        qs = [3,4,5]
        result = audit_extrapolated_panel(frame,2,"D",qs=qs,betas=[1])[0]
        products = []
        for q in qs:
            residual = panel_residuals(frame,q=q)
            products.append(residual["forecast_residual"]*residual["outcome_residual"])
        matrix = np.column_stack(products)
        weights = np.linalg.lstsq(np.column_stack([np.ones(3),1/np.array(qs)]).T,np.array([1.,0.]),rcond=None)[0]
        scores = (matrix-matrix.mean(axis=0)).reshape(30,10,3).sum(axis=1)
        covariance = bartlett_score_covariance(scores,2)
        oracle_se = np.sqrt(weights@covariance@weights)/300
        self.assertAlmostEqual(result["standard_error"],oracle_se,places=12)
        self.assertAlmostEqual(result["mean_product"],matrix.mean(axis=0)@weights,places=12)

    def test_unordered_user_clusters_do_not_parse_ids_as_dates(self):
        rng = np.random.default_rng(888)
        frame = pd.DataFrame({"entity":np.tile(np.arange(8),20),"period":np.repeat(np.arange(20),8),
                              "y":rng.normal(size=160),"prediction":rng.normal(size=160),"baseline":rng.normal(size=160)})
        result = audit_panel(frame,[0],None)[0]
        self.assertEqual(result["periods"],20)
        self.assertTrue(np.isfinite(result["statistic"]))

    def test_constant_within_cluster_prediction_is_undefined(self):
        rng = np.random.default_rng(889)
        frame = pd.DataFrame({"entity":np.tile(np.arange(8),20),"period":np.repeat(np.arange(20),8),
                              "y":rng.normal(size=160),"prediction":np.repeat(np.arange(20),8),"baseline":rng.normal(size=160)})
        result = audit_panel(frame,[0],None)[0]
        self.assertEqual(result["inference_status"],"undefined_rank_or_scale")
        self.assertTrue(np.isnan(result["statistic"]))
        self.assertEqual(result["p_one_sided"],1.0)
        extrap = audit_extrapolated_panel(frame,0,None)[0]
        self.assertEqual(extrap["inference_status"],"undefined_rank_or_scale")

    def test_exactly_entity_spanned_prediction_does_not_studentise_solver_noise(self):
        rng = np.random.default_rng(899)
        frame = pd.DataFrame({"entity":np.tile(np.arange(8),20),"period":np.repeat(np.arange(20),8),
                              "y":rng.normal(size=160),"prediction":np.tile(np.arange(8),20),"baseline":rng.normal(size=160)})
        result = audit_panel(frame,[0],None)[0]
        self.assertEqual(result["inference_status"],"undefined_rank_or_scale")
        extrap = audit_extrapolated_panel(frame,0,None)[0]
        self.assertEqual(extrap["inference_status"],"undefined_rank_or_scale")


if __name__ == "__main__":
    unittest.main()
