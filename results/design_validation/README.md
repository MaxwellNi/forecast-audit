# Design checks for conditional forecasting audits

These results explain when a rank-based residual-association audit can change with its adjustment, reference samples, or nuisance-training folds. A retained model is a screening result under the specified procedure. These files do not establish nominal error control for every dependent panel.

The top-level `peer_sampling.csv` and `peer_sampling.pdf` contain the complete comparison of resampled and fixed peers. Recreate the figure from the repository root:

```bash
python scripts/figures/make_peer_sampling_figure.py
```

The `SHA256SUMS` file identifies the exact reviewed files in this directory and its figure generator. Check it from the repository root with `sha256sum -c results/design_validation/SHA256SUMS`. The 43 core evidence exports comprise 42 aggregate source files plus a column-selected model-level detectable-effect table. The top-level plotting table is a projection of the peer-sampling summary, and the figure files are derived assets.

The figure shows all 18 cells, with 300 simulations per cell and pointwise 95% Wilson intervals. The horizontal line is 0.05. The black open circles use a shared reference; blue filled squares use distinct references. The simulation uses exact conditional means, so this figure isolates peer sampling from errors in learning the nuisance means.

## What each directory contains

| Directory | Experiment and unit of a row | Main interpretation |
|---|---|---|
| `peer_sampling` | Simulation replication or summary cell; public or restricted model aggregate | Whether shared or distinct references are centred depends on the sampling model. |
| `feedback_nulls` | One simulation replication, design cell, or exact-expectation calculation | Ordered nuisance fits reduce a specific feedback bias under suitable innovations; strong serial dependence remains a counterexample. |
| `ordered_folds` | One model, with standard and ordered nuisance fits | Descriptive sensitivity of the fitted audit; the middle-fold and all-fold versions are distinct procedures. |
| `interaction_adjustment` | One simulated panel or summary cell | Adding an omitted interaction, changing reference construction, and changing rank-conditioned information are different operations. |
| `near_copy` | One synthetic score draw on a fixed observation pattern, or a summary cell | Exact-copy abstention and the calibration of a near-copy test are separate properties. |
| `conditional_power` | One synthetic score draw, summary cell, or model aggregate | Empirical null-quantile comparisons are conditional on a fixed observed panel; they are not independently calibrated tests for a new population. |
| `two_way_clustering` | One public model or domain summary | Changes in the variance estimator can be examined without changing the standard nuisance folds. |
| `computational_cost` | One synthetic problem size or numerical equivalence case | Measured runtime and exact implementation checks; timing slopes are not asymptotic complexity claims. |
| `retail_bootstrap` | One retail model | Wild-bootstrap sensitivity of the original standard-fold score products over 28 weeks. |
| `restricted_rerun` | One pretrained forecasting model | Retrospective forecasts with contexts ending at their forecast origins; historical checkpoint availability is not established. |

The restricted-case files contain only model-level or simulation-level aggregates. They do not contain entity identifiers, original observations, predictions, residual vectors, or per-period totals. The simulations using a restricted observation pattern cannot be regenerated from these aggregates alone. The public and synthetic procedures can be checked separately; the withheld data are not an implicit part of this code release.

## Peer sampling

For the resampled experiment, each period contains 32 fresh entities. Forecast and outcome share a Gaussian baseline and have independent innovations. Conditional means are known. For the fixed experiments, the 32 entity effects are held fixed across periods; their signs are aligned or opposed between forecast and outcome. Each focal entity has 31 possible comparison peers. The period counts are 25, 100, and 400; the same simulation replication supplies the three nested period counts.

`shared_references` reuses a comparison peer in both rank factors. `distinct_references` averages over distinct comparison peers. The expected distinct product is approximately −0.001423 with aligned fixed effects and +0.001423 with opposed fixed effects, while the expected shared product is zero. Distinct references are therefore not a universal correction. The public model comparison changes the reference construction while retaining the original fitted nuisance adjustments and variance protocol; it does not establish which population sampling model is true for a given application.

The rare-recurrence experiment has an entity pool of 32, 64, 256, or 4,096, 32 entities observed per period, and 25, 100, or 400 periods. Forecast and outcome can share persistent entity effects but have conditionally independent innovations. Both reference constructions use learned entity controls. `entity_effect_fraction` takes values 0, 0.5, or 0.9. A large in-sample `identity_share` can arise simply because most entities recur rarely; compare it with `chance_level`. This quantity is descriptive, not an applicability certificate.

## Feedback and nuisance-training folds

`standard` fits each nuisance on the complementary four blocks. `ordered_middle` fits the forecast nuisance on earlier blocks and the outcome nuisance on later blocks, and evaluates only blocks 2 through 4. `ordered_all` evaluates all five blocks. Where a side has no training block, its fallback estimates pooled baseline-bin means from the other blocks without entity effects. This fallback is outside the two-sided support condition of the stylized feedback result.

The four-support study has 50 replications per design cell. Its repeated study has 200 replications on each of two observation patterns, with forecast-past loading −0.4, 0, or 0.4 and outcome-entity-effect fraction 0 or 0.1. The one-month study has 200 replications and uses the latest simulated monthly return to predict an independent next-month return. These are null simulations conditional on their incidence patterns. The standard and ordered statistics share the specified clustering rule; the ordered-middle statistic also uses fewer evaluation observations.

`period_cluster` uses period totals. `two_way` adds entity totals and subtracts the observation-level intersection contribution. In these calculations, a nonpositive two-way variance falls back to the larger one-way variance. This finite-sample rule does not establish a joint asymptotic result for the full audit.

The serial-dependence experiment is an exploratory stress test with 100 replications per cell, correlations 0.1, 0.3, 0.6, and 0.9, and forecast-past loadings −0.4 and 0.4. Its comparator fits bin means on independent simulated copies. At correlation 0.9, the absolute mean difference between the ordered statistic and this comparator reaches 6.47 standard units. Temporal ordering is therefore not a generic solution to serial dependence.

Public ordered-fold comparisons preserve each domain's original ranks, bin assignments, and uncertainty calculation: 14 daily HAC lags for electricity, two weekly lags for retail, user clusters for ratings, and 12 monthly HAC lags for portfolios. Ordered ratings blocks are blocks of users, not time. Public two-way-clustering results are a separate standard-fold sensitivity; these files do not report a jointly calibrated ordered-plus-two-way public procedure.

## Adjustment and detection

The interaction simulation uses 80 entities, balanced fixed signs A, a Gaussian baseline B, and forecast and outcome equal to AB plus independent noise. There are 300 replications at each of 25, 100, and 400 periods. The initial adjustments are additive entity effects and baseline bins or a spline. The augmented adjustment adds sign-by-bin interaction cells. Reference-removal and exact-conditional-mean comparisons are exploratory diagnostics. The latter compare product means, not rejection rates, and distinguish conditioning on a baseline's empirical rank from conditioning on its raw level.

Near-copy simulations hold one observed panel fixed and generate a score from a linear or quadratic function of the standardized baseline rank plus independent noise. Noise scales are 0, 0.001, 0.01, 0.03, 0.1, 0.3, and 1; each has 100 replications. **Rejection rates in the summary are raw test rates before the withholding rule.** The withholding fraction is reported separately. In this tested grid, only noise zero triggers the guard. The rule also recognizes matching observed weak orders; it is not restricted to exact numerical equality.

Conditional-power simulations add an injected coefficient times the observed standardized outcome rank to synthetic scores. The injection coefficient is not itself a measured correlation increment. There are 200 replications per shape, coefficient, and adjustment. Each method's 95% null quantile is estimated from the same 200 null draws used in its reported null rate. This comparison uses knowledge of the simulated null and fixes the observed outcomes and panel structure. It is not independent out-of-sample calibration. Fine bins and splines provide meaningful comparisons; these results establish no adjusted-power advantage for extrapolation.

`conditional_power/minimum_detectable_effects.csv` is a separate 100-replication study on each model's observation pattern. Its entries interpolate the increase in mean monthly forecast-outcome rank correlation needed for 80% detection. `single` uses a one-sided 0.05 normal threshold; `family` uses the threshold 3.34 required for a first rejection in a thirty-model Benjamini-Yekutieli family. These are measured correlation increments, unlike the injection coefficient above. Blank entries mean the tested effect grid never reached 80%. Model rows are retained separately even when their observation counts match; the standard and ordered-middle procedures also evaluate different observations.

## Bootstrap, runtime, and pretrained models

The retail bootstrap uses 99,999 Webb-weight draws, with weights applied to single weeks or pairs of adjacent weeks. It resamples the standard-fold fitted score totals and studentizes with the same two-lag Bartlett HAC rule. Nuisance fits are held fixed. Full-family Benjamini-Yekutieli adjustment includes all ten models and assigns probability one to withheld or undefined cases. All seven original retentions remain. The bootstrap does not validate the ordered-fold procedure or arbitrary 25-period panels.

Runtime results use synthetic data, one computational thread, and median elapsed times over repeated calls. The reference-equivalence file compares direct pair matrices with sorting and cumulative counts, including ties. Sorting is slower at 2,048 entities per period and faster at 8,192 in the recorded grid. Runtime depends on both observation and entity counts, so no single fitted power law should be interpreted as a universal complexity result.

The six-model rerun uses raw-return contexts of 13 to 24 months ending at each forecast origin. Every model is loaded from a pretrained checkpoint, with loading checks; it is distinct from the historical untrained forecasting network in the stored family. No positive model is retained. Four negative associations meet an unadjusted two-sided normal threshold, but none survives the six-model two-sided adjustment with two-way clusters. The contexts are temporally ordered; model pretraining and historical weight availability are not certified.

See `DATA_DICTIONARY.md` for field conventions. Missing or undefined values are retained as blank cells rather than silently converted into zero.

## Generate the peer experiment from seeds

```bash
python scripts/analysis/peer_sampling_study.py --output /tmp/peer-sampling-study --workers 4 --verify results/design_validation/peer_sampling
```

This regenerates all 2,700 paired replication records and checks all 18 cells
against the saved experiment. It uses no provider observations. The protocol
retains the original wider Gaussian draw array to reproduce its random stream;
only the first 32 entities enter this experiment. `generator_protocol.json`
records the seeds and dimensions. This command is separate from redrawing the
figure or checking stored arithmetic with `verify_design_results.py`.
