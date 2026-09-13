# Audit a fixed forecast family

`forecast_audit_cli.py` accepts a CSV containing forecasts, outcomes, a baseline,
entity identifiers and period identifiers. It reports residual association across adjustment
resolutions, checks observed baseline redundancy, and applies the Benjamini-Yekutieli
(BY) calculation to the entire supplied model family. No individual observations or residuals are
written to the output directory.

The names `RETAIN`, `NOT_RETAINED` and `ABSTAIN` are operational labels. This command
does not issue a finite confidence certificate. Its normal-reference p-values
still require justified sampling, dependence and nuisance-bias conditions.
Neither the observed-redundancy guard nor multiple-testing adjustment supplies
those conditions.

## Run the included example

From the package root, after installing `requirements-artifacts.txt`, choose a
new output directory:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python scripts/analysis/forecast_audit_cli.py \
  --input examples/forecast_audit_cli/synthetic_forecasts.csv \
  --output-dir /tmp/forecast_audit_example \
  --frequency cluster --lag 0 --beta 2 --ladder 2,3,4
```

The synthetic fixture has 12 periods, eight entities and six models. It was
generated with NumPy seed 908713. Four observed copies or order-preserving
variants abstain, the signal model is retained and the unrelated model is not
retained. These are software examples, not an empirical calibration study.
The checked aggregate outputs are in `expected/`.

## Read the outputs

| File | Contents |
|---|---|
| `profile.csv` | Every model's combined mean, standard error, raw statistic and p-value, original and guarded BY results, guard reason and final label. Undefined statistics are blank. |
| `resolution_trace.csv` | Each resolution's average residual product, weight and weighted contribution, plus the combined mean, standard error and statistic. The standard error includes dependence across resolutions; it is not formed by adding their separate errors. |
| `by_trace.csv` | Raw and guarded sorted p-values, each rank's BY cutoff and the last passing step-up rank. A row's own cutoff is not by itself the final step-up rule. |
| `guard_evidence.csv` | Exact observed-copy, constant-rank and common/reversed weak-order evidence, including tie partitions. It uses predictions and baselines without outcomes. |
| `cohorts.csv` | Cohort and fold hashes, row counts and fold sizes. Individual identifiers and measurements are excluded. |
| `receipt.json` | Fixed settings, model family, source/dependency hashes, output hashes and the limits of interpretation. |

An exact baseline copy, zero within-period prediction ranks, or common/reversed
baseline weak ordering receives `ABSTAIN`. A numerically undefined residual
score also receives `ABSTAIN`. These models stay in the family with guarded
p-value one. Among the remaining models, full-family BY retention gives
`RETAIN`; the other models receive `NOT_RETAINED`. An observed agreement is a property
of these supplied observations, not proof of a population relationship.

## Input and fixed choices

The six required columns are `model`, `entity`, `period`, `y`, `prediction` and
`baseline`. The three measured columns must contain finite real values. Every
model must have exactly the same entity-period cohort, outcomes and baseline.
Duplicate observations, missing fields, nonfinite measurements, different
cohorts and inconsistent baseline values stop the calculation. Extra columns
are ignored. Custom names can be mapped with, for example,
`--prediction-column forecast` or `--y-column actual`.

Model, entity and period identifiers are parsed as exact strings. Identifiers
such as `01`, `1`, `1.0` and `1e0` remain distinct; numeric-looking identifiers
are never merged. Literal `NA`, `NULL`, `null` and `None` are valid identifiers
and stay in the family. Empty identifiers are invalid. Those same nonnumeric
tokens in measured outcome, prediction or baseline columns are invalid.

Choose these arguments before examining the resulting audit:

| Argument | Meaning |
|---|---|
| `--frequency D`, `M` or `W-FRI` | Daily, monthly or Friday-ending weekly periods for the calendar HAC calculation. Distinct labels cannot refer to the same calendar period. Missing calendar periods contribute zero score sums. |
| `--frequency cluster` | Unordered clusters with lag zero. If every identifier is a decimal number, labels sort by exact numerical value and then exact string spelling to break ties. Otherwise they sort lexically. Equivalent numerical spellings remain distinct clusters. |
| `--lag` | Fixed nonnegative Bartlett lag; it must be smaller than the complete calendar length. |
| `--beta` | Fixed finite positive extrapolation exponent. No exponent is estimated or selected. |
| `--ladder` | At least two strictly increasing, distinct integer resolutions, each at least two. |
| `--alpha` | Family screen level, default 0.05. This is a nominal level unless the input p-values meet the necessary validity conditions. |

The five contiguous folds contain whole observed periods. At least five
observed periods are necessary to execute that partition. Five is a numerical
minimum, not a sufficient sample size for inference. The output flags panels
with at most 50 observed periods because earlier diagnostics found substantial
small-panel miscalibration. This flag is not a validated cutoff; more than 50
periods also does not establish validity.

The nuisance fit can train on periods later than an evaluation period. It is
an audit partition, not a forward-only forecast-training procedure. The
supplied forecasts must already follow their intended evaluation protocol.

## Benchmark maintainer checklist

1. Fix the forecast family and baseline before screening; include every member.
2. Confirm that all forecasts use the same permitted evaluation information and
   that target values were unavailable when forecasts were created.
3. Use the exact common cohort. This command stops on unequal cohorts instead
   of silently removing observations.
4. Record frequency, lag, exponent, ladder and the reason for choosing them.
   The command supplies no guarantee that a chosen ladder preserves power.
5. Inspect the guards, undefined scales, resolution contributions and complete
   BY trace. A small p-value does not resolve nuisance bias or dependence.
6. Report ordinary forecast accuracy separately. A retained residual-association
   screen is not a forecast-quality ranking or proof of improvement.
7. Preserve the receipt and every model result, including abstentions and
   non-retentions. Publishing a model aggregate may require permission even
   though individual observations are absent.

For sensitivity to the nuisance-training direction, use
`directional_audit.py` with the same arguments. It holds baseline-bin assignments
and the original guard fixed and reports both complete and middle-fold supports.
On unordered clusters, earlier and later labels mean identifier order, not time.
