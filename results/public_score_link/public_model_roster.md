# Public forecasting scores and audit decisions: all 41 models

The primary descriptive score is the equal-cluster mean of Spearman(prediction, outcome), computed on exactly the same admissible rows as the beta=2 audit. Average ranks retain ties. A cluster with fewer than two rows or a constant prediction/outcome has undefined correlation; it is counted and omitted only from the mean, never assigned zero. All original rows and all 41 models remain in the assets.

The metric was specified before its calculation, after the existing audit results were known. No forecaster was refitted or tuned. These are descriptive scores for the frozen held-out cohorts, not estimates with demonstrated generalization guarantees and not tests of conditional independence. The audit targets residual association beyond its named controls, not overall predictive utility. Its final `RETAIN` label is an operational screen; real-panel calibration remains unestablished.

Original-scale MAE pools rows within each domain. Units differ across domains. `SVD interaction` is an interaction-only ratings score without an intercept, so its MAE against full ratings is deliberately undefined. All other original-scale MAEs are finite. Return-series rules are compared to the one-month target as supplied forecasts; a multi-month historical return is not rescaled after evaluation.

## Baseline-copy examples

| Domain | Exact model name | Mean cluster Spearman | Defined / total clusters | Original-scale MAE | Raw beta=2 T | Raw p | Guarded BY p | Final label |
|---|---|---:|---:|---:|---:|---:|---:|---|
| electricity | seasonal_naive_7 | 0.985221 | 729/729 | 6.306880 | -5.355448 | 1 | 1 | ABSTAIN |
| portfolios | Trailing twelve-month return | 0.079192 | 132/132 | 11.791535 | 0.859809 | 0.194947 | 1 | ABSTAIN |
| ratings | Item mean | 0.315775 | 252/279 | 0.692750 | 2.611821 | 0.00450307 | 1 | ABSTAIN |
| retail | Last week | 0.792158 | 28/28 | 3.866518 | 1.930341 | 0.0267823 | 1 | ABSTAIN |

The high electricity and retail correlations show that a good ranking score can come entirely from a named baseline. Those baseline forecasts can still be useful. In ratings, the Item mean baseline has MAE 0.692750 and is withheld as a copy, whereas Alternating least squares has MAE 0.719000 and is retained under the stated screen. Retention therefore does not mean lower prediction error. The earlier retail illustration is sensitive to nuisance-training direction; see [the correction note](../../ERRATA.md).

## Complete named roster

The four tables contain all 41 forecasts. `RETAIN`, `ABSTAIN` and `NOT_RETAINED` display the final full-family beta=2 policy using unrounded values; `RETAIN` is not a validated finite certificate. The CSV also includes raw and guarded p-values, baseline scores, paired score differences, MAE status, undefined-group reasons, cohort hashes and family sizes. There is no performance-based model selection.

### electricity: 11 models; 34,992 rows per model

MAE is in original UCI daily-mean load units. Correlations use audit-period clusters.

| Exact model name | Mean Spearman | Defined / total | Undefined groups | MAE | Raw beta=2 T | Guarded BY p | Final label |
|---|---:|---:|---:|---:|---:|---:|---|
| dlinear | 0.991501 | 729/729 | 0 | 4.585049 | 8.762858 | 5.26965e-18 | RETAIN |
| drift | 0.982657 | 729/729 | 0 | 6.855457 | 9.065939 | 5.14867e-19 | RETAIN |
| histgbr_lag | 0.991109 | 729/729 | 0 | 5.559271 | 7.490403 | 1.14044e-13 | RETAIN |
| holt_winters | 0.954106 | 729/729 | 0 | 5.556864 | 9.020278 | 6.23224e-19 | RETAIN |
| lasso_lag | 0.987510 | 729/729 | 0 | 5.969079 | 8.351603 | 1.2428e-16 | RETAIN |
| linear_lag | 0.987514 | 729/729 | 0 | 5.950414 | 8.356252 | 1.2428e-16 | RETAIN |
| naive_1 | 0.982640 | 729/729 | 0 | 6.858103 | 9.065535 | 5.14867e-19 | RETAIN |
| ridge_lag | 0.989524 | 729/729 | 0 | 5.354171 | 8.532050 | 3.41149e-17 | RETAIN |
| seasonal_naive_7 | 0.985221 | 729/729 | 0 | 6.306880 | -5.355448 | 1 | ABSTAIN |
| ses | 0.947475 | 729/729 | 0 | 8.064325 | 9.422228 | 7.33492e-20 | RETAIN |
| theta | 0.989848 | 729/729 | 0 | 4.893346 | 9.239127 | 2.06385e-19 | RETAIN |

### retail: 10 models; 112,000 rows per model

MAE is in weekly sales units. Correlations use audit-period clusters.

| Exact model name | Mean Spearman | Defined / total | Undefined groups | MAE | Raw beta=2 T | Guarded BY p | Final label |
|---|---:|---:|---:|---:|---:|---:|---|
| Croston SBA | 0.703850 | 28/28 | 0 | 4.355190 | -8.590988 | 1 | NOT_RETAINED |
| Eight week mean | 0.800312 | 28/28 | 0 | 3.739556 | 3.840667 | 0.000299488 | RETAIN |
| Exponential smoothing | 0.821310 | 28/28 | 0 | 3.492279 | 4.289652 | 5.24148e-05 | RETAIN |
| Four week mean | 0.820489 | 28/28 | 0 | 3.548056 | 6.037842 | 7.62456e-09 | RETAIN |
| Last week | 0.792158 | 28/28 | 0 | 3.866518 | 1.930341 | 1 | ABSTAIN |
| Past price and calendar | 0.135860 | 28/28 | 0 | 9.129963 | 7.675661 | 2.41006e-13 | RETAIN |
| Past price calendar and history | 0.826980 | 28/28 | 0 | 3.462275 | 4.710655 | 9.04034e-06 | RETAIN |
| Ridge with history | 0.820299 | 28/28 | 0 | 3.561231 | 3.650526 | 0.000547516 | RETAIN |
| Theta | 0.833809 | 28/28 | 0 | 3.405118 | 6.362060 | 1.45764e-09 | RETAIN |
| Training series mean | 0.615479 | 28/28 | 0 | 5.634002 | undefined | 1 | ABSTAIN |

### ratings: 10 models; 12,088 rows per model

MAE is in rating points. Correlations use user clusters.

| Exact model name | Mean Spearman | Defined / total | Undefined groups | MAE | Raw beta=2 T | Guarded BY p | Final label |
|---|---:|---:|---:|---:|---:|---:|---|
| Alternating least squares | 0.273583 | 252/279 | 27 | 0.719000 | 12.743563 | 1.65278e-36 | RETAIN |
| Global mean | undefined | 0/279 | 279 | 0.747473 | undefined | 1 | ABSTAIN |
| Item mean | 0.315775 | 252/279 | 27 | 0.692750 | 2.611821 | 1 | ABSTAIN |
| Item nearest neighbours | 0.394732 | 252/279 | 27 | 0.581100 | 12.926787 | 2.32725e-37 | RETAIN |
| SVD (32 factors) | 0.338896 | 252/279 | 27 | 0.608299 | 10.075530 | 2.07597e-23 | RETAIN |
| SVD (40 factors) | 0.351815 | 252/279 | 27 | 0.608098 | 10.618683 | 8.93258e-26 | RETAIN |
| SVD interaction | 0.143548 | 252/279 | 27 | undefined | 13.597640 | 6.05656e-41 | RETAIN |
| Slope One | 0.336254 | 252/279 | 27 | 0.611640 | 7.176637 | 1.74389e-12 | RETAIN |
| User and item bias | 0.315775 | 252/279 | 27 | 0.622581 | 2.611821 | 1 | ABSTAIN |
| User mean | undefined | 0/279 | 279 | 0.715290 | undefined | 1 | ABSTAIN |

### portfolios: 10 models; 27,272 rows per model

MAE is in return percentage points. Correlations use audit-period clusters.

| Exact model name | Mean Spearman | Defined / total | Undefined groups | MAE | Raw beta=2 T | Guarded BY p | Final label |
|---|---:|---:|---:|---:|---:|---:|---|
| Elastic net | 0.056696 | 132/132 | 0 | 2.749776 | 0.946980 | 1 | NOT_RETAINED |
| Exponential twelve-month mean | 0.082389 | 132/132 | 0 | 2.850867 | 1.291076 | 1 | NOT_RETAINED |
| Last month | 0.050998 | 132/132 | 0 | 3.769198 | 1.454005 | 1 | NOT_RETAINED |
| LightGBM | 0.045043 | 132/132 | 0 | 2.751239 | -0.269110 | 1 | NOT_RETAINED |
| Ridge | 0.056215 | 132/132 | 0 | 2.757891 | 0.942420 | 1 | NOT_RETAINED |
| Seasonal twelve-month lag | 0.003749 | 132/132 | 0 | 3.834352 | -1.100405 | 1 | NOT_RETAINED |
| Small MLP | 0.016815 | 132/132 | 0 | 2.936832 | -1.137301 | 1 | NOT_RETAINED |
| Trailing three-month return | 0.032418 | 132/132 | 0 | 5.615175 | -0.203567 | 1 | NOT_RETAINED |
| Trailing twelve-month return | 0.079192 | 132/132 | 0 | 11.791535 | 0.859809 | 1 | ABSTAIN |
| Training portfolio mean | 0.047195 | 132/132 | 0 | 2.758516 | -2.111274 | 1 | NOT_RETAINED |

## Verification and limitations

- All 41 model cohorts match their saved audit row and cluster counts. Within each domain, sorted entity/cluster/outcome/baseline rows have the same hash for every model; no rows were dropped.
- All 12,409 model-cluster pairs and 37,227 rank vectors were checked. Hand-built average ranks exactly match scipy `rankdata`; manual centered-rank correlations match scipy `spearmanr` within 3.33e-16.
- All 41 raw normal tails were checked using `math.erfc` against scipy and the saved audit output. Maximum absolute difference from saved values: 1.11e-16. All eight raw/guarded complete-family BY calculations match an independent step-up calculation, statsmodels and the saved values.
- Ratings have 18 singleton user clusters and nine additional constant-outcome user clusters. Thus eight nonconstant-score models have 252 defined correlations of 279; `Global mean` and `User mean` have none. All clusters in the other domains have defined correlations. Undefined correlations are fully retained in the cluster asset.
- No uncertainty interval or hypothesis test is attached to these descriptive mean correlations or MAEs. Cross-cluster dependence, heterogeneous cluster sizes, baseline choice and evaluation-cohort selection continue to matter. The two constant-rating models have an undefined primary score, not a score of zero.
- The underlying audits still have their disclosed approximation and dependence limitations. This extension establishes a score-to-audit comparison; it does not establish a valid real-data certificate or show that an audit retention measures predictive superiority.

## Files

- `public_all_41_models.csv`: comprehensive model-level table, with full precision.
- `public_all_cluster_correlations.csv`: every cluster score, independent calculation and undefined reason.
- `public_baseline_copies.csv`: the four exact baseline-copy rows, unfiltered by result.
- `protocol.json`, `verification.json`, `receipt.json`, `provenance.json`: frozen choices, checks and hashes.
- `scripts/analysis/compute_public_score_link.py` (from the package root): computes these scores from supplied public forecast vectors; use `--help` for the required input specification and output path.
