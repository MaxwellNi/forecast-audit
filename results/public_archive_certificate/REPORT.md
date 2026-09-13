# A sampling-design-covered certificate on a fixed public electricity archive

The frozen study certifies positive residual rank association for both the HGB forecast and the previous-week forecast, relative to the declared four-category control, in a **fixed 70,080-row historical archive**. The exact category-copy control is not retained. Independent replacement sampling makes the finite-archive sampling design match the certificate assumptions despite temporal dependence in the original electricity records.

This is a successful application of the finite-archive guarantee, with limited practical discrimination. The predeclared training-median threshold is **zero**, so the control distinguishes zero from positive previous-week demand and weekdays from weekends. Both ordinary forecasts have large remaining rank association. All seven methods retain both, and the full census computes exact archive targets faster than the full-U sampling calculation. The experiment establishes neither HGB's incremental value beyond the continuous seasonal forecast nor a practical advantage for the proposed certificate over the strong comparators or census.

## Frozen design and provenance

The existing source is Artur Trindade's [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014), DOI [10.24432/C58C86](https://doi.org/10.24432/C58C86), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The official description explains the supplied nominal Portuguese clock grid and zeros for clients created after 2011. This study retains zeros, uses the first 96 header meters, and averages the supplied 96 quarter-hour kW records for each complete nominal calendar day. These are daily mean kW targets, not reconstructed physical 23/25-hour energy totals.

The local project had already used this dataset. It is not newly discovered or previously unseen data. Before computing these audit outputs, [PREDECLARED_PLAN.md](PREDECLARED_PLAN.md) and the executable protocol fixed the source, meter population, forecast family, categories, budgets, methods and interpretation. The recorded order is:

1. Protocol frozen at **2026-09-12 14:28:47 UTC**, SHA256 `8850d43b7e52b2f52524e17c4adbc86f99aa628b6af006d20bdaebf9bf007592`.
2. Model, numeric category threshold and forecast archive sealed at **14:29:05 UTC**, before nuisance fitting, random audit draws or audit statistics.
3. Independent indices drawn and stored; every declared certificate written before the exact census target was computed.

This is local locking, not external preregistration. The archive source was reused, and no future-time confirmation claim is made. The protocol, script, frozen model, helper implementations, archive, draw indices and outputs have hash links.

## Forecast and target

A single histogram gradient boosting model trains on **67,392 target-day/meter records strictly before 2013-01-01**, using only target-minus-1, 2, 7, 14 and 28 daily lags, previous-7/28-day means and deterministic target calendar harmonics. Its hyperparameters were fixed before fitting. Its weights remain fixed throughout 2013–2014; lagged observations update sequentially as they become available. Perturbing current and future daily outcomes at three archive cutoffs changes no feature at or before the corresponding target date.

The archive contains all **730 target days in 2013–2014 × 96 meters = 70,080 rows**. The three fixed candidate scores are HGB, previous-week mean load, and the numeric category code. The control is

`C = 2 × target_weekend + I(previous_week_forecast > training_median)`.

The pre-2013 training median is zero. The four archive cell counts are **883, 49,229, 350 and 19,618**; the overwhelmingly positive-load categories are broad. No threshold was changed after observing this result. The previous-week candidate is deliberately retained as a substantive comparator: positive association for that candidate exposes what coarse conditioning leaves unadjusted.

Let `P_A` assign probability `1/70080` to each fixed archive row. For candidate `X`, define its population marginal midrank under this finite law as

`U_X = P_A(X' < X) + 0.5 P_A(X' = X)`.

The estimand is

`theta_A = E_A[(U_X − E_A[U_X|C]) (U_Y − E_A[U_Y|C])]`.

It uses marginal population midranks and the four declared categories. It is not a within-date rank statistic, a raw-MSE gain, conditional independence, adjustment for the entire continuous seasonal forecast, a provenance claim, or a future-time population parameter. The category-copy score has `theta_A=0` because its marginal rank is a function of `C`.

## Why the actual sampling design covers the theorem

Condition on the complete fixed archive `A`. Generate every row index independently uniformly with replacement from `1,…,70080`. For any sets of archive records, the joint probability of their sampled values factorizes because the indices are independent. Thus the sampled records are iid from `P_A`, regardless of the original time-series dependence. Repeated physical rows within or across samples are retained; independently drawn indices may coincide without destroying independence. The distinction is between independent draw positions and distinct physical source records.

The experiment uses separate random-number child streams for:

| Stage | Design | Row queries |
|---|---|---:|
| Nuisance training | Independent iid rows | 4,096 |
| Validation | 8,192 independent focal/reference pairs | 16,384 |
| Evaluation | 128 groups of 96 iid rows | 12,288 |
| **Total** | Same draws shared by all methods/candidates | **32,768** |

The training sample's leave-one-out marginal midranks are averaged within the fixed categories to learn bounded `f` and `g`. Validation midcomparisons estimate the conditional marginal-rank means using disjoint iid focal/reference pairs; all pair members count in the budget. Empty categories, if any, retain the declared fallback and conservative confidence intervals. Evaluation is independent of both stages.

Conditional on the training fits, the distinct-reference order-three kernel has expectation `theta_A+b(f,g)`. Validation supplies a simultaneous bias allowance with failure budget `delta=0.0001`. The range bound or the proved variance bound then bounds the evaluation mean with **4,096 effective triples**. Pooling all evaluation draws is legitimate under this actual common-law iid design. Complete-family BY is applied to the three fixed candidate p-values for each prespecified method; the pooled signed variance method was named primary before evaluation. Comparators are not selected according to their results.

The finite-sample claim concerns repeated randomized samples from this same fixed archive. It does not convert the original temporal records into iid draws from a future population. The exact population masses and rank means could be obtained by census; they are intentionally withheld from the certificate and used only after certificate output for diagnosis.

## Results and exact census benchmark

Primary results use the signed learning allowance and pooled full-U variance bound. The displayed lower bounds are individual 95% lower bounds; BY retention uses the complete three-candidate family and is reported separately.

| Candidate | Exact census `theta_A` | Sample full-U mean | Learning allowance | Sampling radius | Lower bound | BY p | Retained |
|---|---:|---:|---:|---:|---:|---:|---|
| HGB | 0.078219 | 0.078359 | 0.004277 | 0.003579 | **0.070503** | 0.000275 | Yes |
| Previous week | 0.077414 | 0.077525 | 0.004452 | 0.003654 | **0.069418** | 0.000275 | Yes |
| Category-copy control | 0, up to rounding | −0.000012 | 0.004081 | 0.003850 | **−0.007943** | 1 | No |

All observed lower bounds cover their exact census target, and the signed learning budgets cover their exact learning bias. This single realized check is not an empirical coverage study. The two positive candidates reach the declared p-value floor `delta`; their equal displayed BY p-values do not resolve which predictor is better. Neither comparison tests the difference between their targets or forecasts.

The HGB learning bias is only about 0.000062; the certified allowance is 0.004277. The signed and absolute allowances coincide for both positive candidates in this realization. The signed construction reduces the copy-control allowance from 0.004799 to 0.004081, without changing its non-retention. It therefore provides no additional positive discovery here.

All methods use identical samples, nuisance fits and candidate families. HGB results are:

| Method | HGB lower bound | Sampling radius | Kernel computation time |
|---|---:|---:|---:|
| Grouped full U, absolute allowance, range bound | 0.050503 | 0.023568 | 0.3569 s |
| Grouped full U, signed allowance, range bound | 0.050503 | 0.023568 | 0.3569 s |
| Grouped full U, signed allowance, variance bound | 0.070492 | 0.003579 | 0.3573 s |
| Pooled full U, signed allowance, range bound | 0.050515 | 0.023568 | 0.3280 s |
| **Pooled full U, signed allowance, variance bound** | **0.070503** | **0.003579** | **0.3285 s** |
| Independent triples, signed allowance, range bound | 0.050347 | 0.023568 | 0.00046 s |
| Independent triples, signed allowance, Bernstein bound | 0.070336 | 0.003579 | 0.00046 s |

The variance bound narrows HGB's sampling radius about 6.6-fold relative to the range bound, but the range bound already retains both useful candidates. Independent triples with Bernstein give nearly the same lower bound at substantially less kernel-computation cost. One shared draw does not establish a systematic efficiency ranking, but it provides no evidence of a necessary full-U advantage on this archive.

The sample makes **32,768 queries (46.76% of census)** and touches **26,140 unique archive rows (37.30%)**. These are audit-stage query counts; querying a row returns its outcome and all fixed candidate scores. Forecast training costs and source parsing are not hidden inside that count: archive preparation took 8.39 s and already materialized every archive outcome. The exact census ranks and diagnostic computations for all three candidates took about **0.039 s**, including construction of the census output table; HGB's pooled U computation alone took 0.328 s. These are single-run timings on this machine, not a general complexity comparison. Census is both exact and cheap for this materialized archive.

Consequently this example supplies a real public archive for which the guarantee's sampling design is literal and reviewable. It does **not** establish lower total computation/access cost, uniqueness relative to independent triples, additional positive discoveries from the signed allowance, or predictive utility beyond a strong continuous baseline. A setting with an already indexed, expensive-to-query archive could motivate sampling, but that cost model is not demonstrated here.

## Reproduction and review

`study.py` exposes separate `freeze`, `prepare` and `audit` phases, each refusing overwrites. The released completed directory should be verified rather than overwritten:

```bash
python verify.py
```

Default verification is read-only and imports no producer helpers. It checks release-manifest entries and provenance hash links, regenerates every draw index, and independently rebuilds all 21 certificate records. This includes training fits and validation comparisons, all 12 candidate/category confidence rectangles, binomial-CDF inversion for category-mass caps, and nine linear programs for signed upper/lower and absolute bias extrema. Literal pair comparisons reconstruct all 384 group means and every one of the 36,864 grouped and 36,864 pooled candidate/focal products. All 12,288 candidate/triple scores, fitted kernel ranges, radii, raw probabilities, rejection flags, BY adjustments, query counts and census quantities are checked. Numeric comparisons use `rtol=1e-10, atol=1e-12`; actual discrepancies are recorded in [verification.json](verification.json). Runtime values are hash-protected and checked for finite nonnegative values, rather than reproduced by the verifier. Local provenance ordering is checked for consistency, not externally authenticated. This is deterministic verification of an already exposed application, not another independent statistical sample or an empirical coverage study.

For a fresh numerical reproduction use [reproduce.py](reproduce.py) with an existing official source zip or the explicit source-download option and a new output directory. It copies the fixed helper bytes and portable study logic, recreates the original sampling and computation, and compares numerical outputs. It records reproduction timing separately and does not relabel it as the original lock. [reproduction_comparison.json](reproduction_comparison.json) separately records exact array equality and maximum absolute numeric differences, with the interpreter and dependency versions used. Numeric-table comparisons allow `rtol=1e-10, atol=1e-12`; numeric archive-array comparisons allow `rtol=1e-10, atol=1e-10`. Passing those tolerance checks alone does not imply exact equality. Runtime columns and execution timestamps are excluded from equality claims. In the recorded Python 3.9.21 replay with the original pinned dependencies, every archive array and compared non-runtime table column is exactly equal and the maximum numeric differences are zero. Equality concerns loaded arrays/values, not model, compressed-file or timestamp bytes.

The read-only `python test_verify.py` command adds six in-memory corruption controls for the validation intervals/caps, signed allowance, grouped mean, kernel endpoint and raw probability. These leave hashes intact and require the corresponding arithmetic check to reject. It also checks the blocked full-U formula against literal ordered triples and the independent-Bernstein mean requirement.

Key files are [protocol.json](protocol.json), [archive_seal.json](archive_seal.json), [certificate_results.csv](certificate_results.csv), [census_results.csv](census_results.csv), [nuisance_validation.csv](nuisance_validation.csv), [sampling_receipt.json](sampling_receipt.json), [audit_receipt.json](audit_receipt.json) and [verify.py](verify.py). No post-result category, budget, forecast, method or source change was made. The zero median and the unfavorable census/comparator costs are retained.
