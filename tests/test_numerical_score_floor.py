"""A solver-sized residual must not turn a degenerate score into evidence."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from audit_panel_predictions import audit_panel, audit_extrapolated_panel


class NumericalScaleTests(unittest.TestCase):
    def evaluate(self, residual_scale):
        rng = np.random.default_rng(916)
        n = 400
        frame = pd.DataFrame({"period": np.repeat(np.arange(40), 10)})
        outcome = rng.normal(size=n)
        fit = {"forecast_rank": rng.normal(size=n), "outcome_rank": outcome,
               "forecast_residual": residual_scale * (outcome + rng.normal(size=n)),
               "outcome_residual": outcome}
        with patch("audit_panel_predictions.panel_residuals", return_value=fit):
            return audit_panel(frame, [0], None) + audit_extrapolated_panel(frame, 0, None)

    def test_solver_remainder_is_undefined(self):
        for result in self.evaluate(5e-10):
            self.assertLess(result["standard_error"], 1e-10)
            self.assertEqual(result["inference_status"], "undefined_rank_or_scale")
            self.assertEqual(result["p_one_sided"], 1.0)
            self.assertTrue(np.isnan(result["statistic"]))

    def test_resolvable_signal_is_still_reported(self):
        for result in self.evaluate(5e-4):
            self.assertEqual(result["inference_status"], "computed")
            self.assertGreater(result["statistic"], 1.64485)


if __name__ == "__main__":
    unittest.main()
