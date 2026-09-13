# Paired exponent-two scalar comparison

The exponent-two statistic is evaluated on exactly the 2,700 original samples. This is a paired extension after the exponent-one results and separate panel results were known. It is not a new independent replication.

## Design and provenance

The local protocol was written at 2026-09-07T20:11:49.434005+00:00 before the first new exponent-two outcome. The experiment report records its original SHA256 as `db92bed6d13e4eb2c8e2441a2b14510944ef0dc89890fc0a10fb6cb1c10ad2f6`; current packaged file hashes are listed in `MANIFEST.json` at the package root. This timestamp is not an independent preregistration service.

Each of nine conditions has 300 draws of 400 independent observations. The original seeds, two-fold assignments, training-fold bin estimates, ladder {8, 12, 16, 24, 32}, normal reference and 0.05 cutoff are unchanged. The additional weights cancel the inverse-square term. No exponent, ladder or cutoff was selected after these outcomes.

The single run completed in 6.81 seconds with one process and one numerical thread. All 2,700 sample and fold hashes match the original records. Exponent-one, eight-bin and 32-bin statistics and p-values replay exactly, with maximum absolute error zero. All 13,500 original comparator rows are unchanged in their original fields, including their original timing values. Official KCI was not run again. There are 16,200 rows and 54 method cells in the extended file.

## Results

Entries report rejections out of 300 and the Wilson 95% interval for the rejection proportion. These are pointwise Monte Carlo intervals, without multiplicity adjustment. All original and added results are retained.

| Control | Condition | Method | Rejections | Rate | Wilson 95% interval |
|---|---|---|---:|---:|---|
| Linear | Conditional independence | GCM, 8 bins | 66/300 | 0.2200 | [0.1768, 0.2703] |
| Linear | Conditional independence | GCM, 32 bins | 42/300 | 0.1400 | [0.1053, 0.1838] |
| Linear | Conditional independence | Spline GCM | 30/300 | 0.1000 | [0.0709, 0.1392] |
| Linear | Conditional independence | Extrapolation, beta 1 | 36/300 | 0.1200 | [0.0880, 0.1617] |
| Linear | Conditional independence | Official KCI | 21/300 | 0.0700 | [0.0462, 0.1046] |
| Linear | Conditional independence | Extrapolation, beta 2 | 42/300 | 0.1400 | [0.1053, 0.1838] |
| Linear | Positive covariance | GCM, 8 bins | 298/300 | 0.9933 | [0.9760, 0.9982] |
| Linear | Positive covariance | GCM, 32 bins | 287/300 | 0.9567 | [0.9273, 0.9745] |
| Linear | Positive covariance | Spline GCM | 284/300 | 0.9467 | [0.9151, 0.9669] |
| Linear | Positive covariance | Extrapolation, beta 1 | 285/300 | 0.9500 | [0.9192, 0.9695] |
| Linear | Positive covariance | Official KCI | 219/300 | 0.7300 | [0.6771, 0.7771] |
| Linear | Positive covariance | Extrapolation, beta 2 | 290/300 | 0.9667 | [0.9397, 0.9818] |
| Smooth tanh | Conditional independence | GCM, 8 bins | 21/300 | 0.0700 | [0.0462, 0.1046] |
| Smooth tanh | Conditional independence | GCM, 32 bins | 29/300 | 0.0967 | [0.0682, 0.1354] |
| Smooth tanh | Conditional independence | Spline GCM | 21/300 | 0.0700 | [0.0462, 0.1046] |
| Smooth tanh | Conditional independence | Extrapolation, beta 1 | 28/300 | 0.0933 | [0.0654, 0.1316] |
| Smooth tanh | Conditional independence | Official KCI | 14/300 | 0.0467 | [0.0280, 0.0768] |
| Smooth tanh | Conditional independence | Extrapolation, beta 2 | 27/300 | 0.0900 | [0.0626, 0.1278] |
| Smooth tanh | Positive covariance | GCM, 8 bins | 294/300 | 0.9800 | [0.9571, 0.9908] |
| Smooth tanh | Positive covariance | GCM, 32 bins | 290/300 | 0.9667 | [0.9397, 0.9818] |
| Smooth tanh | Positive covariance | Spline GCM | 286/300 | 0.9533 | [0.9232, 0.9720] |
| Smooth tanh | Positive covariance | Extrapolation, beta 1 | 289/300 | 0.9633 | [0.9355, 0.9794] |
| Smooth tanh | Positive covariance | Official KCI | 218/300 | 0.7267 | [0.6736, 0.7740] |
| Smooth tanh | Positive covariance | Extrapolation, beta 2 | 293/300 | 0.9767 | [0.9526, 0.9887] |
| Quadratic | Conditional independence | GCM, 8 bins | 296/300 | 0.9867 | [0.9662, 0.9948] |
| Quadratic | Conditional independence | GCM, 32 bins | 177/300 | 0.5900 | [0.5335, 0.6442] |
| Quadratic | Conditional independence | Spline GCM | 34/300 | 0.1133 | [0.0822, 0.1542] |
| Quadratic | Conditional independence | Extrapolation, beta 1 | 63/300 | 0.2100 | [0.1677, 0.2596] |
| Quadratic | Conditional independence | Official KCI | 14/300 | 0.0467 | [0.0280, 0.0768] |
| Quadratic | Conditional independence | Extrapolation, beta 2 | 184/300 | 0.6133 | [0.5571, 0.6667] |
| Quadratic | Positive covariance | GCM, 8 bins | 300/300 | 1.0000 | [0.9874, 1.0000] |
| Quadratic | Positive covariance | GCM, 32 bins | 298/300 | 0.9933 | [0.9760, 0.9982] |
| Quadratic | Positive covariance | Spline GCM | 292/300 | 0.9733 | [0.9483, 0.9864] |
| Quadratic | Positive covariance | Extrapolation, beta 1 | 283/300 | 0.9433 | [0.9111, 0.9643] |
| Quadratic | Positive covariance | Official KCI | 239/300 | 0.7967 | [0.7475, 0.8383] |
| Quadratic | Positive covariance | Extrapolation, beta 2 | 299/300 | 0.9967 | [0.9814, 0.9994] |
| Heavy tails | Conditional independence | GCM, 8 bins | 27/300 | 0.0900 | [0.0626, 0.1278] |
| Heavy tails | Conditional independence | GCM, 32 bins | 36/300 | 0.1200 | [0.0880, 0.1617] |
| Heavy tails | Conditional independence | Spline GCM | 28/300 | 0.0933 | [0.0654, 0.1316] |
| Heavy tails | Conditional independence | Extrapolation, beta 1 | 32/300 | 0.1067 | [0.0766, 0.1467] |
| Heavy tails | Conditional independence | Official KCI | 21/300 | 0.0700 | [0.0462, 0.1046] |
| Heavy tails | Conditional independence | Extrapolation, beta 2 | 32/300 | 0.1067 | [0.0766, 0.1467] |
| Heavy tails | Positive covariance | GCM, 8 bins | 298/300 | 0.9933 | [0.9760, 0.9982] |
| Heavy tails | Positive covariance | GCM, 32 bins | 300/300 | 1.0000 | [0.9874, 1.0000] |
| Heavy tails | Positive covariance | Spline GCM | 291/300 | 0.9700 | [0.9440, 0.9841] |
| Heavy tails | Positive covariance | Extrapolation, beta 1 | 296/300 | 0.9867 | [0.9662, 0.9948] |
| Heavy tails | Positive covariance | Official KCI | 278/300 | 0.9267 | [0.8915, 0.9511] |
| Heavy tails | Positive covariance | Extrapolation, beta 2 | 297/300 | 0.9900 | [0.9710, 0.9966] |
| Smooth tanh | Zero-covariance dependence | GCM, 8 bins | 18/300 | 0.0600 | [0.0383, 0.0928] |
| Smooth tanh | Zero-covariance dependence | GCM, 32 bins | 29/300 | 0.0967 | [0.0682, 0.1354] |
| Smooth tanh | Zero-covariance dependence | Spline GCM | 15/300 | 0.0500 | [0.0305, 0.0808] |
| Smooth tanh | Zero-covariance dependence | Extrapolation, beta 1 | 28/300 | 0.0933 | [0.0654, 0.1316] |
| Smooth tanh | Zero-covariance dependence | Official KCI | 300/300 | 1.0000 | [0.9874, 1.0000] |
| Smooth tanh | Zero-covariance dependence | Extrapolation, beta 2 | 23/300 | 0.0767 | [0.0516, 0.1124] |

## Paired change from exponent one to exponent two

Every entry compares the same 300 samples. No significance or superiority rule was added after observing these counts.

| Control | Condition | Neither rejects | Beta 1 only | Beta 2 only | Both reject | Rate difference |
|---|---|---:|---:|---:|---:|---:|
| Linear | Conditional independence | 256 | 2 | 8 | 34 | +0.0200 |
| Linear | Positive covariance | 10 | 0 | 5 | 285 | +0.0167 |
| Smooth tanh | Conditional independence | 271 | 2 | 1 | 26 | -0.0033 |
| Smooth tanh | Positive covariance | 7 | 0 | 4 | 289 | +0.0133 |
| Quadratic | Conditional independence | 116 | 0 | 121 | 63 | +0.4033 |
| Quadratic | Positive covariance | 1 | 0 | 16 | 283 | +0.0533 |
| Heavy tails | Conditional independence | 265 | 3 | 3 | 29 | +0.0000 |
| Heavy tails | Positive covariance | 2 | 1 | 2 | 295 | +0.0033 |
| Smooth tanh | Zero-covariance dependence | 271 | 6 | 1 | 22 | -0.0167 |

## Interpretation

Exponent two does not repair calibration in this benchmark. Its four conditional-independence rejection counts are 42, 27, 184 and 32 out of 300 for linear, smooth tanh, quadratic and heavy-tail controls. Every corresponding pointwise Wilson interval lies above the nominal 0.05 level. For the quadratic control it rejects 184/300, compared with 63/300 for exponent one, 34/300 for spline GCM and 14/300 for KCI. The paired difference consists of 121 additional exponent-two rejections and no additional exponent-one rejections.

The increased rejection rates under positive covariance cannot be presented as an advantage at equal error control. They accompany substantial null miscalibration, and the experiment did not equalize type-I error across methods.

On the zero-covariance dependent condition, exponent two rejects 23/300 and KCI rejects 300/300. Conditional covariance is zero by construction. Rejection in this condition is detection for the omnibus conditional-independence test, but it is not power for the signed residual-covariance target.

The smooth-control inverse-square bias calculation has conditions that this fixed-sample Gaussian quantile-bin experiment does not establish. Smoothness as a function of an unbounded Gaussian control does not imply bounded smoothness after transforming that control to a uniform percentile. For the quadratic conditional mean, g(z)=0.8(z^2-1), the derivative of g(Phi^{-1}(u)) is 1.6 Phi^{-1}(u)/phi(Phi^{-1}(u)), which is unbounded near the endpoints. Estimated quantile edges and conditional means add further terms. This explains why the existing uniform-control proposition cannot justify exponent two here; it is not a proof of the effective bias exponent in this experiment.

The paired extension supplies a complete sensitivity check, not evidence of a universal exponent choice, finite-sample validity, panel validity, or improvement over spline GCM. Both exponent rows and this limitation are retained.

## Files and replay

- `protocol.json`: experimental design and packaged dependency/source hashes.
- `attempt_started.json`, `summary.json`: recorded execution and complete summary.
- `replications.csv`: all 16,200 original and added method rows.
- `paired_data_hashes.csv`: every sample, fold and seed.
- `comparison_coordinates.csv`: all 54 counts, rates, intervals and statistic means.
- `verification.json`: separate formula-based arithmetic checks, with scope stated explicitly.

From the package root, use the pinned canonical environment and cap numerical threads at one:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python scripts/analysis/canonical_beta2.py run --protocol results/canonical_beta2/protocol.json --original results/canonical_baselines --output /tmp/canonical_beta2_replay
```

The command refuses to overwrite an existing attempt. The frozen protocol binds the exact producing script, original inputs and numerical dependencies. The verifier replays every added statistic with closed-form intercept weights and an explicit standard-error formula, while sharing the maintained bin-fitting routine. It does not rerun KCI.
