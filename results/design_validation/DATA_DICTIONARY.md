# Field conventions

Each table is either a simulation replication table, a cell summary, or a model aggregate. `replication` is a simulation index, not an observation identifier. `observations`, `periods`, `months`, and `entities` are counts. A repeated model row can share the same observation pattern as other models.

| Field or prefix | Meaning |
|---|---|
| `peer_regime` | Resampled peers, fixed peers with aligned effects, or fixed peers with opposed effects. |
| `method` | Nuisance adjustment or reference-product construction. |
| `shape` | Linear or quadratic baseline function used to construct a synthetic score. |
| `forecast_past_loading` | Coefficient multiplying the forecast's dependence on past outcomes. |
| `outcome_entity_effect_fraction` | Simulation parameter controlling the persistent entity component of outcomes. |
| `noise_scale` | Standard deviation multiplier of independent score noise. |
| `delta` | Injected score coefficient in the power experiment; not itself a rank correlation. |
| `shared_statistic`, `distinct_statistic` | Studentized residual-product means under the two reference constructions. |
| `standard`, `standard_middle` | Complementary-fold nuisance fitting, evaluated on all or middle blocks. |
| `ordered_middle`, `ordered_all` | Directional nuisance fitting, evaluated on middle or all blocks. |
| `month`, `period_cluster` | Period-cluster variance; public temporal applications preserve their stated HAC lag. |
| `twoway`, `two_way` | Entity-plus-period variance with the recorded intersection subtraction and fallback. |
| `mean`, `mean_T` | Across-replication mean of a score or statistic, as indicated by the column. |
| `sd`, `std`, `sd_T` | Across-replication sample standard deviation. |
| `se`, `standard_error` | Reported standard error or Monte Carlo standard error, as indicated by the experiment. |
| `replications`, `rejections`, `rate` | Number of simulation draws, count above the positive normal threshold, and their ratio. |
| `wilson_low`, `wilson_high` | Pointwise 95% Wilson interval for a binomial rejection fraction. |
| `withheld`, `withheld_fraction` | Whether the outcome-free guard withholds a score, or its fraction across draws. |
| `withholding_reason` | Observed copy, matching baseline weak order, undefined variation, or other documented guard result. |
| `BY` | Full-family Benjamini-Yekutieli adjusted value for the stated tail and family. |
| `retained` | Adjusted screening value at most 0.05; validity still depends on the underlying input values. |
| `pstar` | Input probability after withheld or undefined cases have been assigned one. |
| `beta1`, `beta2` | Fixed resolution-extrapolation exponent 1 or 2. |
| `bins8`, `bins32` | Single-resolution nuisance adjustment with 8 or 32 baseline bins. |
| `spline` | Spline adjustment under the stated common controls and folds. |
| `size_adjusted_critical` | Empirical 95% quantile of that method's saved null statistics. |
| `size_adjusted_power` | Fraction above the empirical null quantile; see the conditional-calibration scope in the README. |
| `identity_share`, `chance_level` | In-sample rank variation attributed to entity identity and its design-specific chance comparison. |
| `design_effect` | Variance relative to an observation-independent standard-error comparison. |
| `gamma_hat` | Estimated shared-reference product offset on the recorded rank scale. |
| `replay_absolute_error` | Numerical difference between repeated implementations on the same recorded inputs. |
| `p_wild_week`, `p_wild_block2` | One-sided resampling value with independent week weights or adjacent-week-pair weights. |
| `Tstar_week_q95`, `Tstar_block2_q95` | Bootstrap 95% statistic quantile for those weight schemes. |
| `ordered_middle_residual_correlation` | Combined product mean divided by the recorded forecast and outcome residual scales. |
| `lo95`, `hi95`, `ucb95` | Pointwise normal-reference interval endpoints or one-sided upper bound; not a new coverage guarantee. |
| `seconds`, `_s` | Elapsed runtime in seconds on the recorded configuration. |

Exact-expectation tables use `window` for the number of past outcomes in a forecast and `folds` for the constructed block assignment. `part_a_exact` and `part_a` refer to the analytically computed standard-fit numerator contribution; `mc_standard` and `mc_time_ordered` are their Monte Carlo comparisons. These are numerator expectations, not calibrated test probabilities.

The untrained network in historical restricted aggregates is intentionally separated from the genuine pretrained TimesFM-2.5 rerun. A name correction does not change the saved numerical forecast audit. No comparison of model ranks across unequal supports should be read as a paired predictive leaderboard.
