"""Direction, fallback, and complete-family checks for the public comparison."""
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/analysis'))
from directional_audit import additive_prediction, directional_residuals, audit_directional_frame, score_summary
from test_forecast_audit_cli import fixture
import forecast_audit_cli as cli


class DirectionalAuditTests(unittest.TestCase):
    def test_one_entity_exact_means_and_empty_side(self):
        values = np.array([1., 3., 5., 9., 12.])
        entity = np.zeros(5, int)
        bins = np.zeros(5, int)
        folds = np.arange(5)
        x, fallback_x, unseen_x = directional_residuals(values, entity, bins, folds, 'forecast')
        y, fallback_y, unseen_y = directional_residuals(values, entity, bins, folds, 'outcome')
        expected_x = np.array([1.-np.mean(values[1:])] + [values[k]-np.mean(values[:k]) for k in range(1,5)])
        expected_y = np.array([values[k]-np.mean(values[k+1:]) for k in range(4)] + [12.-np.mean(values[:4])])
        np.testing.assert_allclose(x[:,0], expected_x, rtol=0, atol=1e-14)
        np.testing.assert_allclose(y[:,0], expected_y, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(fallback_x, [True,False,False,False,False])
        np.testing.assert_array_equal(fallback_y, [False,False,False,False,True])
        self.assertFalse(unseen_x.any() or unseen_y.any())

    def test_training_targets_exclude_evaluation_and_unseen_entity_fallback(self):
        values = np.array([1.,3.,7.,11.,101.])
        entities = np.array([0,0,1,1,2])
        bins = np.zeros(5,int)
        training = np.array([True,True,True,True,False])
        evaluation = ~training
        predicted = additive_prediction(values, entities, bins, training, evaluation)
        np.testing.assert_allclose(predicted, [[5.5]], atol=1e-14)
        changed = values.copy(); changed[-1] = -10000.
        np.testing.assert_array_equal(predicted, additive_prediction(changed, entities, bins, training, evaluation))
        with self.assertRaises(ValueError):
            additive_prediction(values, entities, bins, training, training)

    def test_six_model_family_and_same_support_comparison(self):
        frame = fixture()
        result = audit_directional_frame(frame, frequency='cluster', lag=0, beta=2, ladder=(2,3,4))
        self.assertEqual(len(result),24)
        self.assertEqual(result.configuration.nunique(),4)
        self.assertTrue(result.family_size.eq(6).all())
        self.assertFalse(result.calendar_direction.any())
        original = cli.audit_frame(frame, frequency='cluster',lag=0,beta=2,ladder=(2,3,4))['profile'].set_index('model')
        standard = result[result.configuration=='complementary_all'].set_index('model')
        np.testing.assert_allclose(standard.statistic, original.statistic, atol=1e-10, rtol=1e-10, equal_nan=True)
        np.testing.assert_array_equal(standard.final_label,original.final_label)
        for config, group in result.groupby('configuration'):
            self.assertEqual(set(group.model),set(original.index))
            self.assertTrue(group[group.model.isin(['copy','same_order','reverse_order','constant'])].final_label.eq('ABSTAIN').all())
            self.assertNotIn('NULL',set(group.final_label))
        paired = result.pivot(index='model',columns='configuration',values='evaluation_rows')
        np.testing.assert_array_equal(paired.complementary_middle,paired.directional_middle)
        np.testing.assert_array_equal(paired.complementary_all,paired.directional_all)
        self.assertTrue((paired.directional_middle < paired.directional_all).all())
        for forbidden in ('entity','period','prediction','y','baseline'):
            self.assertNotIn(forbidden,result.columns)

    def test_tiny_standard_error_abstains_even_with_large_raw_statistic(self):
        tiny = score_summary(1e-12*np.arange(1,11), np.arange(10), "cluster", 0)
        self.assertLess(tiny["standard_error"], 1e-10)
        self.assertGreater(tiny["raw_statistic"], 6.)
        self.assertTrue(tiny["standard_error_abstain"])
        self.assertTrue(np.isnan(tiny["statistic"]))
        ordinary = score_summary(np.arange(1,11), np.arange(10), "cluster", 0)
        self.assertFalse(ordinary["standard_error_abstain"])
        self.assertAlmostEqual(ordinary["statistic"], tiny["raw_statistic"])
        with patch("directional_audit.score_summary", return_value=tiny):
            result = audit_directional_frame(fixture(), frequency="cluster", lag=0, beta=2, ladder=(2,3,4))
        self.assertTrue(result.final_label.eq("ABSTAIN").all())
        self.assertTrue(result.guarded_p.eq(1.).all())
        self.assertTrue(result.BY_adjusted_p.eq(1.).all())
        newly_guarded = result[~result.original_policy_abstain]
        self.assertTrue(newly_guarded.guard_reason.eq("standard_error_at_or_below_numerical_floor").all())

    def test_calendar_direction_and_invalid_side(self):
        frame=fixture()
        frame.period=pd.Timestamp('2024-01-01')+pd.to_timedelta(frame.period,unit='D')
        result=audit_directional_frame(frame,frequency='D',lag=1,beta=2,ladder=(2,3,4))
        self.assertTrue(result.calendar_direction.all())
        with self.assertRaises(ValueError):
            directional_residuals(np.arange(5),np.zeros(5,int),np.zeros(5,int),np.arange(5),'invalid')


if __name__=='__main__':
    unittest.main()
