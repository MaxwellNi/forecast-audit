#!/usr/bin/env python3
"""Read-only negative controls for the independent archive verifier.

CSV perturbations exist only in memory: file bytes, provenance links and the
release manifest stay valid. Each test must fail at the named arithmetic check.
"""
import contextlib
import io
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import numpy as np
import verify


class ArchiveVerificationTests(unittest.TestCase):
    def reject_changed_field(self, filename, column, check_name, method=None):
        original_read = verify.pd.read_csv
        before = verify.hashes()

        def perturbed_read(path, *args, **kwargs):
            frame = original_read(path, *args, **kwargs)
            if Path(path).name == filename:
                selected = frame.candidate.eq('hist_gradient_boosting')
                if method is not None:
                    selected &= frame.method.eq(method)
                row = frame.index[selected][0]
                frame.loc[row, column] += .01
            return frame

        with patch.object(verify.pd, 'read_csv', side_effect=perturbed_read):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(AssertionError, check_name):
                    verify.main()
        self.assertEqual(before, verify.hashes(), 'Negative control wrote package files')

    def test_changed_signed_allowance(self):
        self.reject_changed_field('certificate_results.csv', 'bias_upper', r'certificate\.bias_upper',
                                  'grouped_full_u_signed_range')

    def test_changed_raw_probability(self):
        self.reject_changed_field('certificate_results.csv', 'p', r'certificate\.p',
                                  'grouped_full_u_signed_range')

    def test_changed_grouped_mean(self):
        self.reject_changed_field('certificate_results.csv', 'mean', r'certificate\.mean',
                                  'grouped_full_u_signed_range')

    def test_changed_kernel_endpoint(self):
        self.reject_changed_field('certificate_results.csv', 'kernel_upper', r'certificate\.kernel_upper',
                                  'grouped_full_u_signed_range')

    def test_changed_validation_mass_cap(self):
        self.reject_changed_field('nuisance_validation.csv', 'category_mass_upper',
                                  r'validation\.category_mass_upper')

    def test_changed_validation_mean_interval(self):
        self.reject_changed_field('nuisance_validation.csv', 'upper_x', r'validation\.upper_x')

    def test_full_u_matches_exhaustive_ordered_triples(self):
        rng = np.random.default_rng(1907)
        for n in (3, 4, 7, 9):
            x = rng.choice([-2.4, -.7, 0., 1.2, 9.3], n)
            y = rng.choice([-3., 0., 2.7, 11.], n)
            f, g = rng.uniform(size=(2, n))
            literal = np.mean([(verify.comparison(x[i], x[j]) - f[i])
                               * (verify.comparison(y[i], y[k]) - g[i])
                               for i, j, k in itertools.permutations(range(n), 3)])
            np.testing.assert_allclose(verify.full_u_mean(x, y, f, g, block_size=2),
                                       literal, rtol=1e-12, atol=1e-14)

    def test_independent_bernstein_requires_its_triple_mean(self):
        h = np.array([-.1, .2, .05])
        f = g = np.full(4, .5)
        with self.assertRaises(AssertionError):
            verify.certificate(h.mean() + .01, h, f, g, .02, .05, .0001,
                               'independent_bernstein')


if __name__ == '__main__':
    unittest.main(verbosity=2)
