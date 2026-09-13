# Complete public-model comparison

All 41 models under the original complementary-fold protocol. Decisions use the within-domain guarded beta=2 BY diagnostic screen. See ../../ERRATA.md for subsequent directional sensitivity.

## Electricity (MAE: original UCI load units)

| Model | Score | MAE | T (beta=2) | T (spline) | Guarded BY p | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| DLinear | 0.9915 | 4.5850 | 8.7629 | 8.9257 | 5.270e-18 | RETAIN |
| Drift | 0.9827 | 6.8555 | 9.0659 | 9.2509 | 5.149e-19 | RETAIN |
| Histogram gradient boosting | 0.9911 | 5.5593 | 7.4904 | 7.7229 | 1.140e-13 | RETAIN |
| Holt-Winters | 0.9541 | 5.5569 | 9.0203 | 8.7902 | 6.232e-19 | RETAIN |
| Lasso lag regression | 0.9875 | 5.9691 | 8.3516 | 8.5618 | 1.243e-16 | RETAIN |
| Linear lag regression | 0.9875 | 5.9504 | 8.3563 | 8.5659 | 1.243e-16 | RETAIN |
| One-day persistence | 0.9826 | 6.8581 | 9.0655 | 9.2513 | 5.149e-19 | RETAIN |
| Ridge lag regression | 0.9895 | 5.3542 | 8.5320 | 8.6949 | 3.411e-17 | RETAIN |
| Seven-day persistence * | 0.9852 | 6.3069 | -5.3554 | 4.9382 | 1 | ABSTAIN |
| Exponential smoothing | 0.9475 | 8.0643 | 9.4222 | 9.2469 | 7.335e-20 | RETAIN |
| Theta | 0.9898 | 4.8933 | 9.2391 | 9.4481 | 2.064e-19 | RETAIN |

## Ratings (MAE: rating points)

| Model | Score | MAE | T (beta=2) | T (spline) | Guarded BY p | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| Alternating least squares | 0.2736 | 0.7190 | 12.7436 | 12.6709 | 1.653e-36 | RETAIN |
| Global mean | undefined | 0.7475 | undefined | undefined | 1 | ABSTAIN |
| Item mean * | 0.3158 | 0.6927 | 2.6118 | 3.1966 | 1 | ABSTAIN |
| Item nearest neighbors | 0.3947 | 0.5811 | 12.9268 | 12.8866 | 2.327e-37 | RETAIN |
| SVD (32 factors) | 0.3389 | 0.6083 | 10.0755 | 10.1029 | 2.076e-23 | RETAIN |
| SVD (40 factors) | 0.3518 | 0.6081 | 10.6187 | 10.6352 | 8.933e-26 | RETAIN |
| SVD interaction | 0.1435 | undefined | 13.5976 | 13.5258 | 6.057e-41 | RETAIN |
| Slope One | 0.3363 | 0.6116 | 7.1766 | 7.1260 | 1.744e-12 | RETAIN |
| User and item bias * | 0.3158 | 0.6226 | 2.6118 | 3.1966 | 1 | ABSTAIN |
| User mean | undefined | 0.7153 | undefined | undefined | 1 | ABSTAIN |

## Retail (MAE: weekly sales units)

| Model | Score | MAE | T (beta=2) | T (spline) | Guarded BY p | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| Croston SBA | 0.7038 | 4.3552 | -8.5910 | -13.1536 | 1 | NOT_RETAINED |
| Eight week mean | 0.8003 | 3.7396 | 3.8407 | 3.8424 | 2.995e-04 | RETAIN |
| Exponential smoothing | 0.8213 | 3.4923 | 4.2897 | 5.6152 | 5.241e-05 | RETAIN |
| Four week mean | 0.8205 | 3.5481 | 6.0378 | 7.9978 | 7.625e-09 | RETAIN |
| Last week * | 0.7922 | 3.8665 | 1.9303 | -0.5018 | 1 | ABSTAIN |
| Past price and calendar | 0.1359 | 9.1300 | 7.6757 | 6.7605 | 2.410e-13 | RETAIN |
| Past price calendar and history | 0.8270 | 3.4623 | 4.7107 | 7.8711 | 9.040e-06 | RETAIN |
| Ridge with history | 0.8203 | 3.5612 | 3.6505 | 7.8799 | 5.475e-04 | RETAIN |
| Theta | 0.8338 | 3.4051 | 6.3621 | 9.5732 | 1.458e-09 | RETAIN |
| Training series mean | 0.6155 | 5.6340 | undefined | undefined | 1 | ABSTAIN |

## Portfolios (MAE: percentage points of return)

| Model | Score | MAE | T (beta=2) | T (spline) | Guarded BY p | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| Elastic net | 0.0567 | 2.7498 | 0.9470 | 0.9459 | 1 | NOT_RETAINED |
| Exponential twelve-month mean | 0.0824 | 2.8509 | 1.2911 | 1.2781 | 1 | NOT_RETAINED |
| Last month | 0.0510 | 3.7692 | 1.4540 | 1.4438 | 1 | NOT_RETAINED |
| LightGBM | 0.0450 | 2.7512 | -0.2691 | -0.2850 | 1 | NOT_RETAINED |
| Ridge | 0.0562 | 2.7579 | 0.9424 | 0.9417 | 1 | NOT_RETAINED |
| Seasonal twelve-month lag | 0.0037 | 3.8344 | -1.1004 | -1.1102 | 1 | NOT_RETAINED |
| Small MLP | 0.0168 | 2.9368 | -1.1373 | -1.1235 | 1 | NOT_RETAINED |
| Trailing three-month return | 0.0324 | 5.6152 | -0.2036 | -0.2081 | 1 | NOT_RETAINED |
| Trailing twelve-month return * | 0.0792 | 11.7915 | 0.8598 | -0.7906 | 1 | ABSTAIN |
| Training portfolio mean | 0.0472 | 2.7585 | -2.1113 | -2.1099 | 1 | NOT_RETAINED |

Score: mean within-cluster Spearman correlation over clusters where it is defined; descriptive only. MAE: original-scale mean absolute error, with domain-specific units. The CSV retains full stored precision.
RETAIN is the operational diagnostic screen at 0.05; real-panel calibration is not established. NOT_RETAINED means non-rejection; original CSV null codes denote the same decision. ABSTAIN denotes a guard or an undefined audit. An asterisk marks the five finite copy or weak-order guards.
Global mean and User mean have undefined descriptive scores and statistics. SVD interaction is an interaction-only score and has no rating-scale MAE. Training series mean has an undefined audit scale. All three undefined statistic pairs remain in their model families.
