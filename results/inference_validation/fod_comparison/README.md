# Classical FOD moment on the independent-entity forecast family

This comparison adds a classical forward-orthogonal-deviation (FOD) outcome moment with a recursive predictable score instrument to the existing complementary, gapped-complementary and blocked-directional fits. It uses the same simulator, random primitives, 12 middle evaluation rows, entity weights, independent 5,000-draw calibration bank and 1,000 evaluation families. It is a specified moment comparison, **not a full panel GMM estimator**. The transformation is attributed to [Arellano and Bover (1995), equations (24)-(25), pp. 41-42](https://www.cemfi.es/~arellano/arellano-bover-1995.pdf).

The new moment is centered under the stated innovation null. In this experiment it has higher weak-signal power than the blocked-directional moment. Supplied-null-calibrated complementary fitting has still higher power at coefficient 0.06. These results do not establish general dominance of either temporal design.

## Primary results: 400 independent entities, 11 models

Every family has 11 forecasts; the alternatives inject the same current-innovation coefficient into the first three. Family retention uses BY at 0.05. Each method’s supplied-null calibration uses its own statistic values on the same independent null primitives. The methods share a nominal level; their realized FDRs are not forced to match.

| Method | All-null FDR, normal | All-null FDR, t(399) | All-null FDR, supplied null |
|---|---:|---:|---:|
| Complementary | 0.000 | 0.000 | 0.008 |
| Gapped complementary | 0.928 | 0.926 | 0.015 |
| Blocked directional | 0.011 | 0.011 | 0.009 |
| Classical FOD / recursive instrument | 0.012 | 0.011 | 0.015 |

| Method | Normal power, 0.03 | Normal power, 0.06 | Normal power, 0.10 | Supplied-null power, 0.03 | Supplied-null power, 0.06 | Supplied-null power, 0.10 |
|---|---:|---:|---:|---:|---:|---:|
| Complementary | 0.000 | 0.044 | 0.809 | 0.184 | 0.844 | 1.000 |
| Gapped complementary | 0.992 | 1.000 | 1.000 | 0.105 | 0.694 | 1.000 |
| Blocked directional | 0.110 | 0.658 | 0.994 | 0.107 | 0.666 | 0.995 |
| Classical FOD / recursive instrument | 0.138 | 0.752 | 0.998 | 0.143 | 0.752 | 0.998 |

The very high nominal power of gapped fitting accompanies severely inflated FDR and is not evidence of useful calibrated detection. At coefficient 0.06, normal-reference mixed-family FDR is 0.646 for gapped fitting, 0.00985 for blocked directional fitting and 0.01112 for FOD. With supplied-null calibration, the corresponding FDRs are 0.01150, 0.01025 and 0.01047; complementary fitting gives 0.00890.

At coefficient 0.06, the supplied-null **paired power difference, FOD minus blocked directional, is 0.08633**, with pointwise Monte Carlo interval [0.07210, 0.10056]. Its Bonferroni interval over all 27 primary contrasts is [0.06373, 0.10893]. FOD minus complementary is −0.09200, with pointwise interval [−0.10782, −0.07618] and Bonferroni interval [−0.11713, −0.06687]. Power differences are formed within each family before averaging, accounting for dependence across the three affected models. These are Monte Carlo comparisons under this fixed law and, for calibrated results, conditional on the realized banks.

FOD also improves over blocked directional fitting at coefficient 0.03 in this simulation. At coefficient 0.10 both nearly saturate; their small power differences do not exclude zero in the 27-comparison Bonferroni intervals. All effects, entity counts, family sizes and inference references are retained in the CSVs.

## Moment, target and assumptions

For t=5,...,16 out of 20 observations, the instrument is X_t minus the mean of X strictly before t. The transformed outcome is sqrt((20−t)/(21−t)) times Y_t minus its strictly future mean. Divide the sum of the 12 instrument-outcome products by the sum of those square-root factors, then weight entities equally.

The normalized FOD mean and the original directional mean are both exactly zero under the null and exactly τ under a constant current-innovation covariance injection τ. The FOD normalization is a positive design-only scalar and leaves its studentized statistic unchanged. Under time-varying effects, FOD targets a weighted average and the original directional construction targets an equal-row average. The target match here is specific to the constant-effect experiment.

The original complementary score has exact mean −0.04505208 + 1.0625τ; the gapped score has exact mean 0.06553288 + 1.15079365τ. Their positive response normalizations do not remove null bias. Original matrices and numerical outputs remain unaltered. The complete derivation and nuisance-support differences are in `DERIVATION.md`.

This null-centering argument uses independent entity trajectories and outcome innovations with the required orthogonality to past information and forecast noises. It does not validate serially dependent innovations, empirical ranks, additive baseline bins, HAC inference, or observational-panel decisions. Future means use other outcomes from the same observed entity trajectory, including later rows within an evaluation fold; this is the classical moment construction, whose within-entity dependence is handled by entity-level studentization. It uses the same total observations but different nuisance sets from blocked fitting. Current-innovation injections are synthetic positive controls, not prospective forecasts.

Supplied-null calibration requires the correct simulator law. Its marginal rank argument averages over calibration-bank randomness. Evaluation intervals condition on the realized banks. `family_summary.csv` includes conventional Monte Carlo intervals and pointwise finite-evaluation 95% Hoeffding bounds for FDP in [0,1]. With 1,000 evaluation families, the Hoeffding radius is 0.042947; for example, FOD’s normal all-null estimate 0.012 has upper bound 0.054947. These conservative finite-evaluation bounds do not establish that every conditional FDR is below 0.05, and they are not simultaneous across cells.

The comparator was specified after the earlier study had been reviewed, before this new comparison was run. This is a retrospective methodological follow-up. Its improvement in this simulator is not independent confirmation of a generally better forecasting procedure.

## Files and reproduction

`family_summary.csv` contains all 288 family cells; `family_replications.csv.gz` retains 288,000 family records. `paired_power_differences.csv` contains 162 paired contrasts. `analytic_expectations.csv` reports exact null biases and alternative response factors. The compressed arrays retain calibration statistics and evaluation means, standard errors, statistics, probabilities and guards. `score_matrices.npz` stores all four score matrices and the fixed evaluated rows. Protocol, receipt and SHA256 files record the exact run.

From the public release root:

```bash
python scripts/analysis/fod_comparison.py --verify results/inference_validation/fod_comparison
python scripts/analysis/fod_comparison.py --output /tmp/fod-reproduction
PYTHONPATH=scripts/analysis python -m unittest discover -s tests -p 'test_fod_comparison.py' -v
```

The script’s defaults locate `independent_entity_family.py` beside it and the original bank under `results/inference_validation/raw_family_study`. `--base-study` and `--legacy-results` support other layouts. A reproduction refuses to overwrite an existing result directory. The reference study and its stored arrays are inputs; they are not modified.

Six tests pass, including independent rowwise products, the classical transformation’s orthonormality, exact null/alternative response, independently reconstructed random primitives, normalization invariance and reference/BY checks. Saved-output verification reconstructs upper-tail ranks using direct inclusive comparisons and BY decisions using a separate step-up implementation. It verifies every one of the 288 family-summary cells and all Hoeffding bounds. In the recorded same-environment run, original calibration/evaluation arrays reproduce **bit for bit**, with zero maximum difference for means, standard errors, statistics and probabilities; every original BY decision agrees. Cross-environment verification permits numerical rounding tolerance; the tested Python 3.12 environment differs by at most 1.2e-16 in tail probabilities and gives identical decisions. All original result hashes are unchanged.

Final comparator source SHA256: `f3712102a18a82752c26315f24e3cd78f6812029d10878ce2b3fc4acec329922`. Base-study source SHA256: `5b351e0672a1027379ac33d71fe6e4d4fe4da402aaf936b29b20c6a030bdc10f`.
