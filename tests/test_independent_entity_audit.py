"""Behavioral checks for the raw, independent-entity audit entry point."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

try:
    from scripts.analysis.independent_entity_audit import audit_panel, by_adjust, directional_scores, run, validate_panel
except ModuleNotFoundError:
    from independent_entity_audit import audit_panel, by_adjust, directional_scores, run, validate_panel


def fixture():
    x = np.array([[1,2,6,8,9,10], [0,4,1,7,2,1], [1,3,4,10,5,6]], dtype=float)
    y = np.array([[2,4,3,9,8,10], [4,1,6,0,3,5], [2,8,3,4,1,7]], dtype=float)
    rows = []
    for model in ('varying', 'constant'):
        for entity in range(3):
            for period in range(6):
                rows.append(dict(model=model, entity=f'e{entity}', period=period,
                                 prediction=x[entity,period] if model == 'varying' else entity+1.,
                                 y=y[entity,period]))
    return pd.DataFrame(rows), x, y


class TestIndependentEntityAudit(unittest.TestCase):
    def test_exact_hand_calculation_and_studentization(self):
        frame, x, y = fixture()
        score, eligible = directional_scores(x, y, np.repeat(np.arange(3), 2))
        np.testing.assert_array_equal(score, [-13.5, -11., -1.])
        np.testing.assert_array_equal(eligible, [False,False,True,True,False,False])
        profile, scope = audit_panel(frame, folds=3)
        row = profile.set_index('model').loc['varying']
        self.assertEqual(row['mean'], -8.5)
        self.assertAlmostEqual(row['se'], np.std([-13.5,-11.,-1.],ddof=1)/np.sqrt(3))
        self.assertAlmostEqual(row['p'], norm.sf(row['statistic']))
        self.assertEqual(row['evaluation_rows'], 6)
        self.assertEqual(row['support_fraction'], 1/3)
        self.assertFalse(scope['scientific_assumptions_verified_from_data'])
        self.assertEqual(scope['fallback_rows'], 0)

    def test_complete_family_retains_constant_abstention(self):
        frame, _, _ = fixture()
        profile, _ = audit_panel(frame, folds=3)
        constant = profile.set_index('model').loc['constant']
        self.assertEqual(constant['label'], 'ABSTAIN')
        self.assertEqual(constant['reason'], 'constant_prediction_within_entities')
        self.assertEqual(constant['decision_p'], 1)
        self.assertEqual(constant['family_size'], 2)
        self.assertEqual(len(profile), 2)

    def test_cohort_hole_is_rejected(self):
        frame, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'same complete'):
            audit_panel(frame.drop(index=0), folds=3)

    def test_identical_outcomes_are_required(self):
        frame, _, _ = fixture()
        frame.loc[0,'y'] += .00001
        with self.assertRaisesRegex(ValueError, 'Outcomes must be identical'):
            audit_panel(frame, folds=3)

    def test_duplicate_rows_are_rejected(self):
        frame, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            audit_panel(pd.concat([frame,frame.iloc[:1]]), folds=3)

    def test_missing_global_period_is_rejected(self):
        frame, _, _ = fixture()
        frame = frame.loc[frame.period.ne(2)]
        with self.assertRaisesRegex(ValueError, 'missing periods'):
            audit_panel(frame, folds=3)

    def test_row_order_does_not_change_results(self):
        frame, _, _ = fixture()
        a, _ = audit_panel(frame, folds=3)
        b, _ = audit_panel(frame.sample(frac=1,random_state=991), folds=3)
        pd.testing.assert_frame_equal(a,b)

    def test_numeric_period_order_not_lexical(self):
        frame, _, _ = fixture()
        frame['period'] = (frame.period+8).astype(str)
        _, _, folds, _, periods = validate_panel(frame,3)
        self.assertEqual(periods,[8,9,10,11,12,13])
        np.testing.assert_array_equal(folds,[0,0,1,1,2,2])

    def test_fractional_periods_are_rejected(self):
        frame, _, _ = fixture()
        frame['period'] = frame.period+.5
        with self.assertRaisesRegex(ValueError, 'integer observation indices'):
            audit_panel(frame,folds=3)

    def test_finite_inputs_are_required(self):
        frame, _, _ = fixture()
        frame.loc[0,'prediction'] = np.inf
        with self.assertRaisesRegex(ValueError, 'finite real'):
            audit_panel(frame,folds=3)

    def test_complex_values_are_not_silently_cast(self):
        frame, _, _ = fixture()
        frame['prediction'] = frame['prediction'].astype(complex)
        frame.loc[0,'prediction'] += 1j
        with self.assertRaisesRegex(ValueError, 'finite real'):
            audit_panel(frame,folds=3)

    def test_t_reference_is_labeled_as_sensitivity(self):
        frame, _, _ = fixture()
        profile, scope = audit_panel(frame,folds=3,reference='t')
        row=profile.set_index('model').loc['varying']
        self.assertAlmostEqual(row['p'],student_t.sf(row['statistic'],2))
        self.assertEqual(row['reference_df'],2)
        self.assertIn('no exact t',scope['reference_scope'])

    def test_by_known_values_and_ties(self):
        np.testing.assert_allclose(by_adjust([.0001,.02,.08,1]),[1/1200,1/12,2/9,1])
        np.testing.assert_allclose(by_adjust([.01,.01,.5]),[.0275,.0275,11/12])
        with self.assertRaises(ValueError):
            by_adjust([-.01,.5])

    def test_floor_abstains_without_claiming_zero_association(self):
        frame, _, _ = fixture()
        frame['prediction'] *= 1e-12
        profile, _ = audit_panel(frame,folds=3)
        row=profile.set_index('model').loc['varying']
        self.assertEqual(row['label'],'ABSTAIN')
        self.assertEqual(row['reason'],'standard_error_floor')
        self.assertTrue(np.isfinite(row['statistic']))
        self.assertEqual(row['decision_p'],1)

    def test_write_scope_and_never_overwrite(self):
        frame, _, _ = fixture()
        with tempfile.TemporaryDirectory() as temp:
            input_path=Path(temp)/'input.csv'
            output=Path(temp)/'audit'
            frame.to_csv(input_path,index=False)
            run(input_path,output,folds=3)
            before={p.name:p.read_bytes() for p in output.iterdir()}
            scope=json.loads((output/'scope.json').read_text())
            self.assertEqual(scope['model_count'],2)
            self.assertEqual(len(scope['input_sha256']),64)
            self.assertEqual(set(before),{'profile.csv','scope.json'})
            with self.assertRaises(FileExistsError):
                run(input_path,output,folds=3)
            self.assertEqual(before,{p.name:p.read_bytes() for p in output.iterdir()})

    def test_failed_validation_creates_no_output(self):
        frame, _, _ = fixture()
        with tempfile.TemporaryDirectory() as temp:
            input_path=Path(temp)/'bad.csv';output=Path(temp)/'audit'
            frame.drop(index=0).to_csv(input_path,index=False)
            with self.assertRaises(ValueError):
                run(input_path,output,folds=3)
            self.assertFalse(output.exists())

    def test_fold_requirements_and_trajectory_count(self):
        frame, _, _ = fixture()
        for count in (2,7,3.5,True):
            with self.assertRaises(ValueError):
                audit_panel(frame,folds=count)
        with self.assertRaisesRegex(ValueError, 'At least two'):
            audit_panel(frame.loc[frame.entity.eq('e0')],folds=3)


if __name__ == '__main__':
    unittest.main()
