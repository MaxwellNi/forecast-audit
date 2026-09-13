# Independent calibration and baseline-adjustment studies

All observations in the two new studies are generated from public, specified random-number seeds. No restricted observations, identifiers, model predictions, or data loaders are used. The historical CSVs in this directory only recount already released aggregate results.

## Raw independent-entity families

`independent_entity_family.py` runs the complete raw equal-entity inference branch with fitted entity means and the full model family retained in the Benjamini-Yekutieli correction. It compares complementary, gapped complementary, and directional mean fitting on identical middle-fold observations. The forecasts within a family share outcomes and have noise correlation 0.5. There are 10 or 11 models, either all null or with three injected alternatives, and 25, 100, or 400 independent entities.

The study uses 5,000 independent null calibration panels and 1,000 evaluation families. Both streams, all methods, model counts, signals, and entity counts are fixed in `raw_family_study/protocol.json`. Entity counts, family sizes and alternatives share random primitives within a replication. The 216 summary cells are not 216 independent studies.

The independent calibration bank uses the known simulation law. Its plus-one p-values are a comparator with additional information, not a way to obtain a null law from an arbitrary real panel. Monte Carlo intervals for FDR and power condition on that single calibration bank. They quantify variation across evaluation families and do not include variation from generating another calibration bank. Comparisons share a nominal error level; they do not impose exactly equal realized error rates.

The NPZ files contain no pickled objects:

| File | Arrays and axes |
|---|---|
| `independent_null_calibration.npz` | `statistics`: calibration replication, fitting method, entity count |
| `evaluation_statistics.npz` | `mean`, `standard_error`, `statistic`, `guarded`: evaluation replication, signal, fitting method, entity count, model; `pvalue` appends inference rule |

Axis values follow the order recorded in the protocol. `family_replications.csv.gz` contains every family-level decision count and false discovery proportion. `family_summary.csv` includes all 216 cells; `single_model_summary.csv` reports the first model without multiplicity correction. `paired_power_differences.csv` reports matched-replication comparisons; `analytic_expectations.csv` gives exact expected fitted score means, which differ from the zero target for the biased fitting rules.

The existing raw statistic already uses sample standard deviation with `ddof=1`. The Student-t comparison changes its reference distribution and does not apply a second degrees-of-freedom correction.

## Near-copy rank study

`near_copy_adjustment.py` uses 25 independent periods and 80 entities. In every period, a fixed baseline grid is randomly assigned to entities. The score and outcome have independent homogeneous Gaussian noises under the null. Conditional on a focal baseline rank, the competitor grid is fixed, so the score and outcome rank distributions factorize. This establishes the zero conditional rank target before fitting.

The six methods use identical observations and five whole-period fitting folds. They are 32 bins, the two recorded extrapolation rules, the released spline fit, a spline fit with an explicitly unpenalized linear baseline term, and known conditional rank means. The added linear term changes penalization; it is not a new general-purpose function class or a new conditional-independence test.

The initial correlation grid was zero, 0.3 and 0.6. Its positive controls saturated. The distributed full grid retains those configurations and adds 0.03, 0.06 and 0.10 on the same primitive seeds. This is an exploratory power extension. Its repeated null cells are not independent confirmation; the full grid preserves every original configuration.

The near-copy calibration and evaluation each have 1,000 panels. NPZ axes are replication, adjustment method, score configuration; the p-value array appends inference rule. Method, configuration and seed orders are in each protocol. Exact copies are a separate zero-variance boundary and are withheld from inference. Near copies remain in every denominator. Oracle conditional means and supplied-null calibration use information unavailable in a generic real application.

## Reproduction

Use Python with NumPy, pandas, SciPy, scikit-learn and statsmodels. Run from the package root after installing the documented dependencies.

```bash
python scripts/analysis/independent_entity_family.py --output /tmp/raw-family-new
python scripts/analysis/near_copy_adjustment.py --output /tmp/near-copy-new --workers 2
python -m unittest discover -s tests -v
python scripts/analysis/verify_inference_studies.py --output /tmp/inference-checks
```

Choose new output directories; generators refuse to overwrite recorded runs. `verify_inference_studies.py` checks every reported count against stored arrays, independently computes Monte Carlo tails and family decisions, and reproduces selected generator draws. `verification.json` records the checks. Source hashes and result hashes are retained with each experiment.

The family summaries also include a pointwise finite-evaluation 95% Hoeffding interval for the mean false discovery proportion, conditional on the calibration bank. Unlike the approximate normal Monte Carlo interval, this remains nonzero when no false discoveries were observed. Neither interval incorporates a newly generated calibration bank.
