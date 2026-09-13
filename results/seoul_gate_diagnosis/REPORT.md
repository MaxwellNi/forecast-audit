# Rental-demand gate: exact selection diagnosis

The seasonal gate excluded HGB because its **nominal positive residual-association p-value passed 0.05 but its complete-family BY value did not**. HGB was not guarded, and the study uses raw rental counts throughout. Its eight weekly residual-product means were positive but uneven; with the frozen Student `t7` reference, the p-value was 0.00667738 and the BY value was 0.10940387. Consequently the software selected the adjusted seasonal baseline, losing 299,789.52 count-squared units of realized MSE improvement relative to the already frozen ungated choice. Ridge HGB passed; gated and ungated Ridge predictions are identical. This is a diagnosis of the software's decision and its losses on the exposed historical window, not a population causal or risk claim.

## Scope and reproduction

Inputs are the 27 files in `../seoul_confirmation/`, including the original source archive, frozen models, selection record, protocol, and released losses. The audit reconstructs forecasts from those model bytes without refitting. It independently reconstructs selection weekly moments, nominal statistics, every family-adjusted p-value, guards, coefficients, and all candidate losses. The largest selection reconstruction discrepancy is `2.91e-11`. All 27 input hashes and the input file set remain unchanged.

From this directory (with the unchanged Seoul study at `../seoul_confirmation/`):

```bash
python diagnose_gate.py
```

Execution is read-only and prints its checks. Use `--study-dir PATH` to select the unchanged Seoul study directory. The reconstruction uses the released `study.py` only for source parsing and frozen-model feature/prediction construction; statistics, BY, choices and losses are recomputed here. This is deterministic forensic verification on already revealed data, not independent statistical confirmation. The original producer environment is NumPy 1.26.4, pandas 2.3.3, SciPy 1.13.1 and scikit-learn 1.6.1, which matches this run.

The evidence files are:

- [gate_candidates.csv](gate_candidates.csv): all 16 rows, nominal p before and after guard, full-family BY order/threshold/source order, both guards, eight individual weekly means, HAC components, coefficients and selection losses.
- [selection_weekly_moments.csv](selection_weekly_moments.csv): all 128 candidate/week residual-product means, residual second moments, and frozen-coefficient selection gains, with dates.
- [candidate_losses.csv](candidate_losses.csv): all 16 candidates' direct, convex, augmentation and gated confirmation losses, selection losses, coefficients, gain identity, and candidate-level missed benefit/avoided harm.
- [predeclared_selector_comparison.csv](predeclared_selector_comparison.csv): the six original selections and losses.
- [selected_opportunity_loss.csv](selected_opportunity_loss.csv): actual selector differences, with nonadditive candidate inventory totals labeled separately.
- [posthoc_gate_arithmetic.csv](posthoc_gate_arithmetic.csv): narrowly labeled arithmetic changes to the frozen p-values; none is a new confirmed method.
- [verification.json](verification.json): checks, dates, counts and all input SHA256 values.

## 1. Dates and what the gate measures

Forecast training used 3,384 hours (2017-12-08–2018-04-27). Nuisance fitting used 1,344 hours (2018-05-05–06-29). Selection used **1,344 hours, eight seven-day blocks, `t7`** (2018-07-07–08-31). Confirmation used **2,016 hours, twelve seven-day blocks, `t11`** (2018-09-08–11-30); fortnight sensitivity used six blocks and `t5`. Seven calendar days separate the windows. These gaps and blocks do not prove independence.

The number of blocks differs from the Student reference degrees of freedom: eight selection weeks use seven degrees of freedom, and twelve confirmation weeks use eleven.

For separately frozen nuisance projections, the selection moment is

`W = [X − mX(B)] [Y − mY(B)]`, with `theta = mean(W)` and `v = mean([X − mX(B)]²)`.

The augmentation coefficient is `t = max(theta,0)/v`, except guarded rows have `t=0`. The gate uses a one-sided nominal test of positive mean `W`, then full-family BY at 0.05. All quantities are on the raw rental-count scale; there is **no ranking or rank-to-raw-loss conversion**. The adjusted baseline `mY(B)` is a fitted spline/linear projection, not an asserted true conditional mean. The gate is a **nominal residual-association screen used to restrict augmentation selection**, not an established certificate of predictive gain or calibrated temporal FDR.

## 2. Guards do not explain the useful candidate's exclusion

There are five guarded rows: seasonal control with seasonal-24h and exact copy; Ridge control with Ridge, reflected Ridge and exact copy. For Ridge control, `2B − Ridge = B`, explaining the extra duplicate. All five satisfy both raw-copy equality and residual second moment at most `1e-12`. Their residual moments have numerical roundoff magnitude, and all have exactly zero frozen augmentation coefficients and zero realized augmentation gain.

Seasonal HGB has residual second moment **119,083.2591**, coefficient **1.050545734**, and neither guard. Ridge HGB has residual second moment **22,295.6086**, coefficient **0.786988732**, and neither guard. Turning off the copy-label check while retaining the zero-variance guard changes neither selection. Treating near-zero floating-point residuals as real directions and dividing by their tiny second moments would not define a meaningful missing forecast.

Guarded raw copies can have nonzero *convex* gains because `mY(B)+a[B−mY(B)]` recalibrates the adjusted baseline, whereas the residual augmentation direction is zero. For example, the seasonal exact-copy convex blend has MSE 312,931.78, a 20,982.56 improvement over the adjusted baseline; its residual augmentation still has exactly zero gain. This is accounted for in the candidate loss table and is not a guard-induced missed residual augmentation.

## 3. Eight-week variability and the exact BY cutoff

For weekly means `w1,…,w8`, the recorded HAC standard error satisfies

`SE² = (gamma0 + gamma1)/7`, where `gamma0 = mean((w−mean(w))²)` and `gamma1 = sum(centered[w+1] centered[w])/8`.

This is the frozen lag-one Bartlett calculation, including its finite-block correction. The one-sided reference uses `t7`.

| Selection week, starting | Seasonal HGB mean | Ridge HGB mean |
|---|---:|---:|
| July 7 | 286,986.58 | 22,753.63 |
| July 14 | 45,290.46 | 15,042.23 |
| July 21 | 54,679.33 | 12,916.69 |
| July 28 | 55,973.36 | 12,134.30 |
| August 4 | 49,581.41 | 11,974.76 |
| August 11 | 93,603.82 | 22,030.88 |
| August 18 | 125,755.27 | 15,541.90 |
| August 25 | 288,949.05 | 27,976.75 |
| **Mean** | **125,102.41** | **17,546.39** |
| **HAC SE** | **38,057.97** | **2,099.99** |
| **Student statistic** | **3.28715** | **8.35548** |

Both candidates also have positive frozen-coefficient augmentation gains in every selection week. Seasonal HGB's larger absolute signal therefore does not imply a more precise association estimate: its high first and last weeks contrast sharply with the middle weeks. These data establish empirical heterogeneity, not its underlying cause or an externally measured power curve.

Seasonal HGB has `gamma0=9.482049674e9`, `gamma1=6.568120681e8`. Its hypothetical independent-week SE is 36,804.60; the lag-one HAC term increases SE by only **3.41%**. With the same HAC SE but a Gaussian upper tail, p would be 0.0005060 instead of 0.0066774. The short-block Student tail matters materially. These comparisons explain arithmetic; they do not license assuming independent weeks, using a Gaussian reference, or claiming that more weeks would necessarily solve the problem.

The complete family includes all deliberate copies and controls: `m=16`, `H16=3.380728993`, and `m H16=54.091663892`. Sorted raw p-values are compared by the step-up procedure against `0.05 k/(16 H16)`. Only order 1 satisfies its threshold. Seasonal HGB is order 3, with threshold **0.0027730705**; its statistic 3.28715 is below that order's `t7` critical value 3.94801. Its adjusted value is determined by the smaller suffix-adjusted value at order 4:

`p_BY(HGB | seasonal) = 0.008090257292 × 54.091663892 / 4 = 0.109403869562`.

Thus citing only HGB's own sorted threshold does not describe the full BY computation; the complete suffix minimum is verified.

| BY order | Control / candidate | Raw p after guard | BY threshold at order | Adjusted p | Disposition |
|---:|---|---:|---:|---:|---|
| 1 | Ridge / HGB | 0.0000344919 | 0.000924357 | 0.001865722 | Retained |
| 2 | Seasonal / Ridge | 0.006575418 | 0.001848714 | 0.109403870 | BY blocks |
| 3 | Seasonal / HGB | 0.006677383 | 0.002773071 | 0.109403870 | BY blocks |
| 4 | Seasonal / persistence-1h | 0.008090257 | 0.003697427 | 0.109403870 | BY blocks |
| 5 | Seasonal / seasonal-168h | 0.013852011 | 0.004621784 | 0.149855665 | BY blocks |
| 6 | Ridge / seasonal-24h | 0.131552431 | 0.005546141 | 1 | Nominal p exceeds .05 |
| 7 | Ridge / noise | 0.196069782 | 0.006470498 | 1 | Nominal p exceeds .05 |
| 8 | Ridge / seasonal-168h | 0.532507488 | 0.007394855 | 1 | Nonpositive selection moment |
| 9 | Seasonal / noise | 0.561778082 | 0.008319212 | 1 | Nonpositive selection moment |
| 10 | Ridge / persistence-1h | 0.968087020 | 0.009243568 | 1 | Nonpositive selection moment |
| 11 | Seasonal / reflected Ridge | 0.993424582 | 0.010167925 | 1 | Nonpositive selection moment |
| 12 | Seasonal / seasonal-24h | 1 | 0.011092282 | 1 | Guard |
| 13 | Seasonal / exact copy | 1 | 0.012016639 | 1 | Guard |
| 14 | Ridge / Ridge | 1 | 0.012940996 | 1 | Guard |
| 15 | Ridge / reflected Ridge | 1 | 0.013865353 | 1 | Guard |
| 16 | Ridge / exact copy | 1 | 0.014789710 | 1 | Guard |

The software cause is therefore specific: the eight-week nominal association evidence is insufficient for the predeclared 16-member BY requirement, so no seasonal candidate remains eligible. The data do not identify a unique scientific cause of low precision or prove population false negatives.

## 4. Selection and realized opportunity loss

The three strategies have identical underlying forecast fits, nuisance fits, candidate family and data windows. Each minimizes its specified selection loss with the adjusted baseline available. Ungated augmentation and convex blending choose HGB under both controls; gating chooses the seasonal baseline and Ridge HGB.

| Control | Adjusted baseline MSE | Convex HGB MSE | Ungated HGB MSE | Gated choice MSE | Gate minus ungated MSE |
|---|---:|---:|---:|---:|---:|
| Seasonal | 333,914.34 | 40,231.11 | 34,124.82 | 333,914.34 | **299,789.52** |
| Ridge | 53,554.47 | 36,746.76 | 31,123.04 | 31,123.04 | **0.00** |

Positive values in the final column are costs of gating on this fixed window. Seasonal gated MSE is 9.7851 times ungated MSE; the forgone gain is 89.7804% of adjusted-baseline MSE. Ridge's 22,431.43 improvement over its adjusted baseline is attributable to the frozen augmentation forecast shared by gated and ungated strategies. Gating adds zero improvement. The gate's Ridge advantage over convex blending is 5,623.72, with the original weekly interval [833.08, 10,414.36] and fortnight sensitivity [−897.79, 12,145.23]; these intervals are temporal approximations and do not isolate a gate benefit.

The realized *candidate* inventory must be distinguished from the *selected strategy*:

| Control / candidate | Frozen t | Augmentation gain vs adjusted baseline | Gate decision |
|---|---:|---:|---|
| Seasonal / persistence-1h | 0.643112611 | 238,951.69 | Blocks |
| Seasonal / seasonal-24h | 0 | 0.00 | Guard |
| Seasonal / seasonal-168h | 0.327947697 | 31,308.99 | Blocks |
| Seasonal / Ridge | 1.001564779 | 274,948.74 | Blocks |
| Seasonal / HGB | 1.050545734 | 299,789.52 | Blocks |
| Seasonal / reflected Ridge | 0 | 0.00 | Blocks |
| Seasonal / exact copy | 0 | 0.00 | Guard |
| Seasonal / noise | 0 | 0.00 | Blocks |
| Ridge / persistence-1h | 0 | 0.00 | Blocks |
| Ridge / seasonal-24h | 0.031261269 | −1,114.26 | Blocks |
| Ridge / seasonal-168h | 0 | 0.00 | Blocks |
| Ridge / Ridge | 0 | 0.00 | Guard |
| Ridge / HGB | 0.786988732 | 22,431.43 | Retains |
| Ridge / reflected Ridge | 0 | 0.00 | Guard |
| Ridge / exact copy | 0 | 0.00 | Guard |
| Ridge / noise | 0.033943473 | −26.16 | Blocks |

Four seasonal candidates had positive realized frozen augmentation gains and were blocked; none had negative augmentation gain. Two Ridge candidates had harmful frozen augmentations that the gate excluded, totaling 1,140.42 across alternative candidates. **That total is not a realized benefit of gating**, because ungated MSE selection also chose HGB and never deployed those harmful alternatives. Candidate opportunities are alternative forecasts on the same outcomes and must not be added into selector regret. All five guarded alternatives have zero residual-augmentation opportunity loss.

Define retrospective opportunity loss against the best *already frozen* residual augmentation or adjusted baseline, without optimizing coefficients on confirmation. HGB is that best candidate under both controls. Seasonal gated opportunity loss is 299,789.52; Ridge gated and both ungated opportunity losses are zero. This comparator is an ex-post empirical selection benchmark, not a population oracle or a deployable selection rule.

The frozen-coefficient identity reproduces every candidate's gain:

`Delta_confirm(t) = 2t theta_confirm − t² v_confirm`.

For seasonal HGB, `theta_confirm=275813.2867`, `v_confirm=253450.2320` and frozen `t=1.050545734` give 299,789.52. The ex-post empirical optimum at `t=1.088234501` gives 300,149.53; the coefficient mismatch costs just 360.01. For Ridge HGB the analogous mismatch is 626.95. Those ex-post coefficients are diagnostic only. In this example the large seasonal cost is the gate's eligibility decision, not poor frozen coefficient performance. The larger confirmation than selection plug-in gain also illustrates that the selection plug-in is not a guarantee of the gain on another window; this observation alone does not establish population nonstationarity.

## 5. Narrow arithmetic comparisons, not retrospective retuning

These calculations reuse the original p-values or the same eight weekly moments solely to identify which components matter. They are fully retained, including changes that do not alter the adverse result.

| Arithmetic modification | Seasonal HGB adjusted/nominal p | Seasonal selected candidate |
|---|---:|---|
| Original full-16 BY | 0.109404 | Adjusted baseline |
| Nominal p without family adjustment | 0.006677 | HGB |
| Full-16 BH, omit harmonic multiplier | 0.032361 | HGB |
| BY after dropping five guards, family 11 | 0.067187 | Adjusted baseline |
| BY separately within each family of eight | 0.058635 | Adjusted baseline |
| Full-16 BY, remove lag-one covariance but retain `t7` | 0.086337 | Adjusted baseline |
| Full-16 BY, Gaussian tail at the same HAC SE | 0.009124 | HGB |

The controls and duplicate rows contribute to the family penalty but removing only the five guarded rows does not fix the seasonal exclusion. The HAC lag term is also not the decisive component. Replacing BY or the Student reference would change the evidence standard; these modifications cannot be promoted as validated improvements using this same confirmation window. No new rule has been selected or deployed here.

## 6. Association screening and actual augmentation gain are different targets

For a frozen positive coefficient `t`, positive raw residual association alone is insufficient for positive gain: one needs `theta > t v/2`. To exceed a prespecified meaningful gain `Delta_min`, one needs `theta > t v/2 + Delta_min/(2t)`. A positive association implies existence of some beneficial sufficiently small coefficient under the appropriate finite-moment fixed-distribution algebra; it does not certify the coefficient actually estimated on another window. Nuisance estimation, coefficient error and temporal change remain relevant.

The Seoul gate tests the mean of `W`, while realized gain is the mean of `2tW − t²rX²`. Its nominal association test does not directly test the latter quantity or the frozen 1% reporting threshold. Conversely, a failure to retain is not evidence that every augmentation is useless. The present seasonal result illustrates the missed-use side. Ridge's two harmful but unselected frozen candidates illustrate why candidate rejection counts cannot establish selected-policy value.

A future gain-based gate cannot simply apply the same ordinary interval to selection losses after fitting `t` and choosing a candidate on those same losses. That would ignore coefficient and candidate selection. Any such comparison must predeclare a valid design for its target: for example, an internal chronological coefficient-fit/evaluation split shared by all selectors, or a fixed coefficient grid with simultaneous inference under stated dependence assumptions. With only eight selection weeks, extra splitting reduces precision and must count against the same budget. Neither ordinary BY arithmetic nor the categorical independent-reference certificate makes these temporal gain intervals valid automatically.

## 7. Feasible next comparison at the same budget

The strongest completed comparison already is the original predeclared trio: ungated selection, conventional convex blend and association-gated selection. It fully answers whether gating added realized selection value on this support: it did not.

For a future experiment, freeze the data source/window before outcomes are seen and keep the same two baselines, eight candidates, forecasting features, training budget, nuisance budget, total selection observations, confirmation observations, and candidate evaluation counts. A bounded comparison can include the original three selectors plus one **gain-based conservative selector** whose coefficient fitting and gain evaluation are separated within the fixed selection budget. Give every compared selector access to the same resulting data partitions and report both the common-partition comparison and the cost of any reduced fit/evaluation sample; otherwise a split-only competitor confounds target choice with sample allocation. Preserve the adjusted-baseline fallback, tie rules, direction constraints, all copies/controls, a prespecified practical gain threshold, and one final untouched evaluation window.

Report final selected MSE, paired gain against ungated and convex selection, empirical opportunity loss versus the best frozen eligible candidate, candidate-level harmful deployments and missed gains as separate descriptive inventories, sample counts, and fit/evaluation cost. Provide results under predeclared dependence sensitivities. A classical paired-loss comparator is appropriate for the final fixed forecasts; a nested-model or encompassing test should be used only after checking that its stated target and assumptions match the actual fitted augmentation design. There is no reason to add an unrelated literature roster to this forensic result.

The 2018 Seoul confirmation outcomes have already been inspected. They may be reused for transparent exploration, implementation checks and the diagnosis above, but cannot independently confirm any newly designed selector. A new source, later untouched period, or other genuinely unexamined evaluation support is required before making a new confirmation claim.

