# Comparing nuisance-training directions on matched observations

The 164 rows in `model_comparison.csv` report all 41 public models under four configurations. `domain_summary.csv` counts their decisions. Every domain keeps its complete model family, original abstention rule, baseline-bin assignments, resolution ladder (8, 12, 16, 24, 32), and exponent 2. The code is `scripts/analysis/directional_audit.py`.

| Configuration | Nuisance training | Evaluation observations |
|---|---|---|
| `complementary_all` | Other four folds | All five folds |
| `complementary_middle` | Other four folds | Middle three folds |
| `directional_middle` | Earlier folds for forecasts, later folds for outcomes | Middle three folds |
| `directional_all` | Earlier folds for forecasts, later folds for outcomes; reported edge fallbacks | All five folds |

Electricity uses daily HAC with 14 lags; retail uses weekly HAC with two lags; portfolios use monthly HAC with 12 lags. Ratings use user clusters with lag zero. **Ratings clusters have an identifier order, not a calendar order.** The configuration names describe a partition sensitivity in that domain.

To reproduce a domain, supply the same full-family forecast CSV and use the corresponding frequency and lag. For example:

```sh
python scripts/analysis/directional_audit.py \
  --input retail_forecasts.csv --output-dir retail_directional \
  --frequency W --lag 2 --beta 2 --ladder 8,12,16,24,32
```

The input columns are `model`, `entity`, `period`, `prediction`, `y`, and `baseline`; the primary README explains the input checks and public-data setup. Source observations are not redistributed. The output is model-level aggregates, not observation-level predictions or identifiers. This comparison reruns the audit on the stored public forecasts; it does not retrain the forecasters.

## Matched-support retail comparison

On the same middle-fold observations, complementary fitting retains five models and directional fitting retains seven. Directional fitting adds Croston SBA, Eight week mean, and Ridge with history and removes Past price and calendar. The price/calendar statistic changes from 6.41 to -0.82; Croston SBA changes from -19.43 to 6.11. Across all folds, the retained count changes from seven to six. Electricity, ratings, and portfolios retain 10, 6, and 0 models under all four configurations.

Matching evaluation observations removes one source of difference between procedures. It does not isolate a causal feedback contribution, establish error control, or prove a predictive advantage. The original ranks, baseline-bin assignments, and full-cohort redundancy guard remain fixed; the baseline channel is not retrained under the directional restriction. The original retail bootstrap uses complementary products and does not calibrate these directional products.

## Output fields and safeguards

`statistic` is unavailable when the standard error is at or below 1e-10. `raw_statistic` preserves the unguarded ratio for diagnosis only. Such models, any other undefined statistic, and originally guarded models have `guarded_p=1` and decision `ABSTAIN`. `normal_p_one_sided` is the descriptive normal tail of the guarded statistic; `BY_adjusted_p` adjusts guarded values within the complete domain and configuration. `NOT_RETAINED` means failure to pass this specified screen, not proof of no information.

The pooled-fallback and unseen-entity columns count the affected evaluation rows. A missing training side uses pooled baseline-bin means on the complementary folds; an unseen entity receives the training-weighted mean entity effect. These fallbacks are outside the exact time-order centering result. `calendar_direction` reports whether the specified order is temporal. The cohort hash identifies the model's validated input cohort without exporting its rows.

The source arithmetic and full-family adjusted values were checked against the recorded three-configuration public comparison. The fourth configuration adds the matched complementary-middle control. These checks establish numerical reproduction, not independent statistical calibration.
