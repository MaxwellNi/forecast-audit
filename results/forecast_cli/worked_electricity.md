# Worked electricity audit

The DLinear forecast and the last-week baseline copy below are audited inside the complete 11-model electricity family. The calculation retains the recorded labels and common evaluation cohort of all 11 models. The trace uses known predictions and known audit settings; it supplies an executable explanation, not new evidence of calibration.

Each model has 34,992 predictions for 48 meters across 729 observed days. Five contiguous whole-day folds contain 146, 146, 146, 146 and 145 days. The fixed choices are beta 2, ladder {8, 12, 16, 24, 32}, daily calendar HAC with Bartlett lag 14, and nominal family level 0.05.

Within each day, prediction and outcome ranks are standardized; baseline percentile ranks assign control bins. For each resolution, training folds fit joint additive meter and bin effects. The evaluation residuals are multiplied and averaged. These averages are the table entries below. The weights combine the residual products before the cluster/HAC standard error is calculated.

| Resolution q | Weight | DLinear average residual product | Weighted contribution | Baseline-copy average residual product | Weighted contribution |
|---:|---:|---:|---:|---:|---:|
| 8 | -0.204639450 | 0.028764023 | -0.005886254 | 0.012325054 | -0.002522192 |
| 12 | 0.154242546 | 0.021141801 | 0.003260965 | 0.004677663 | 0.000721495 |
| 16 | 0.279851244 | 0.019116122 | 0.005349670 | 0.002563840 | 0.000717494 |
| 24 | 0.369571743 | 0.018481846 | 0.006830368 | 0.001239194 | 0.000457971 |
| 32 | 0.400973918 | 0.017133356 | 0.006870029 | 0.000607606 | 0.000243634 |

The weights sum to one and their inner product with q to the power minus two is zero. This cancels a term of that form; it does not establish that the actual finite-rank panel has only that bias term.

| Quantity | DLinear | Last-week baseline copy |
|---|---:|---:|
| Mean within-day Spearman correlation | 0.99150090235 | 0.985220593373 |
| Original-scale MAE | 4.58504896809 | 6.30688010927 |
| Combined residual-product mean | 0.0164247786831 | -0.000381598317409 |
| Calendar HAC standard error | 0.0018743633108 | 7.1254232993e-05 |
| Studentized statistic | 8.76285754662 | -5.35544768893 |
| One-sided normal p-value | 9.51810796144e-19 | 0.999999957328 |
| Guarded p-value | 9.51810796144e-19 | 1 |
| Guarded BY adjusted p-value | 5.26964507646e-18 | 1 |
| Final operational label | RETAIN | ABSTAIN |

The last-week forecast equals the observed baseline exactly. The outcome-free copy guard therefore sets its policy p-value to one and returns ABSTAIN, irrespective of its raw statistic. All 11 family members remain in the multiple-testing calculation.

With 11 models, the harmonic factor is 3.019877344877345. The sorted rank-r cutoff is 0.05 r divided by 11 times that factor. DLinear has guarded rank 6; its own cutoff is approximately 0.00903107. The last passing rank is 10 and the common step-up cutoff is approximately 0.01505179. DLinear is retained. The baseline copy has rank 11, p-value one, and is not retained.

All ten non-copy electricity models are retained in this existing screen. This example does not demonstrate rejection of a high-scoring non-copy forecast or an advantage over spline adjustment. RETAIN is the algorithm's operational label; the calculation does not establish the sampling and bias conditions needed for a valid certificate. Forecast accuracy is shown separately because residual association and forecast quality answer different questions.

The common supplied cohort SHA256 is `12553db517213f4038de753151e273c5d0296e4b05e46c006a0a610e90696384`. The prediction file SHA256 is `3b8dd4072414ff7ba2d310480af4fcbb7da1afe2df1a3bfa371a20c6789c2e28`. Exact numeric values, both full BY traces and the five-resolution calculations are in `worked_electricity.json` and `electricity/`.

The CSV interface preserves exact identifiers and distinct numeric spellings, as documented in the example guide.
