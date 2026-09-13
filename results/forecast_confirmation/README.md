# Fixed forecast selection and chronological confirmation

Two public tasks compare residual-rank certificates with direct loss tests,
ungated selection and convex forecast combinations. The task definitions,
models, controls, sample allocation and confirmation comparisons were fixed
locally before the first source download. This package reproduces the later
mechanically corrected computation; it is not a new independent confirmation
or an external preregistration.

From the repository's installed Python 3.12 environment:

```bash
python results/forecast_confirmation/verify.py --output /tmp/forecast-confirmation-check
python results/forecast_confirmation/reproduce.py --output /tmp/forecast-confirmation-reproduction
python -m unittest discover -s results/forecast_confirmation -p 'test_*.py'
```

Use new output paths. The first command independently reconstructs all 160
certificate rows, 32 selected rules, 32 confirmation rows, 944 weekly rows,
32 candidate-loss rows and four dependent confirmation bounds from saved
arrays and sampling indices. It checks source aggregation, scales, seasonal
features, categories and calibration weights, without importing the producer
or loading fitted models. The second command trains every model from the
bundled public ZIPs, rebuilds forecasts and choices, compares all numeric
arrays and non-runtime result fields, then invokes that independent verifier.
Exact equality is reported separately from tolerance-based numerical agreement.
The repository pins NumPy 1.26.4, pandas 2.2.3, SciPy 1.11.4 and scikit-learn
1.7.2. Original models were fitted with scikit-learn 1.6.1; their pickles are
not included here. A source replay fits and saves models for the installed
version in its output directory. Allow up to 40 minutes per phase, one CPU
thread and 8 GiB process address space.

## Targets, controls and information

Appliances predicts the next complete hour's appliance energy, summing six
ten-minute readings. Metro predicts hourly traffic volume; duplicate source
timestamps are averaged and missing calendar hours remain missing. Forecasts
use strictly earlier calendar-hour measurements and known calendars. The
backtest assumes measurements are available when their hour ends; historical
publication vintages were not reconstructed. Interpolated airport variables
and target-hour weather are excluded.

| Task | Training rows | Weight calibration | Selection archive | Confirmation rows |
|---|---:|---:|---:|---:|
| Appliances | 1,344 | 336 | 336 | 1,008 |
| Metro | 17,491 | 3,476 | 4,362 | 8,713 |

The rank target is the residual covariance of marginal midranks under uniform
sampling from each **fixed selection archive**, conditional on 32 declared
categories: 16 training-baseline prediction quantile bins crossed with a
weekend indicator. Equal breakpoints remain, so some cells are empty. This is
coarse adjustment, not conditioning on the complete continuous baseline.
Conditional on the fixed archive, independently drawn row indices are iid
by construction. The original hourly observations need not be iid.

Every task and baseline has the same eight candidates: persistence, daily
and weekly seasonal forecasts, richer Ridge, histogram gradient boosting,
ExtraTrees, a category-only copy and independent noise. Two baselines are
reported; Ridge is primary. Models fit clipped training outcomes. All delivered
forecasts are clipped to the training outcome's 99th percentile scale. Primary
loss is the resulting normalized squared error in [0,1]. Raw MSE and MAE use
the same delivered predictions and untruncated outcomes and are secondary.
Calibration chooses augmentation weights on the fixed grid 0, .05, ..., 2;
convex weights use 0, .05, ..., 1. Each gate applies BY to all eight candidates
and chooses the retained augmentation with the lowest selection loss, with
baseline fallback. Ungated selection uses the same fitted augmentations.
See [protocol.json](protocol.json) for every fixed choice.

## Results and costs

The reference U and reference betting methods fall back to baseline in all
four task/baseline cases. The known-forecast U and independent-pair methods
choose ExtraTrees and match ungated selection in all four. Under the primary
Ridge baseline, these augmentations reduce bounded loss by 16.95% and 73.87%
relative to baseline; the gate's improvement over ungated selection is zero.
The direct loss gate retains ExtraTrees in three cases and falls back for
Appliances/Ridge. Both tasks fail the prespecified criterion requiring the
primary reference gate to improve on both ungated and convex selection.

Every candidate and gate uses 32,768 replacement row queries. Reference methods
allocate 4,096 to mean fitting, 8,192 independent pairs to validation and
12,288 to evaluation. Known-forecast methods additionally access complete
forecast/category metadata and allocate all queries to outcome evaluation;
the direct loss method evaluates all queries. Thus equal label-call budgets
do not imply equal information or identical sample allocation. Query streams
are reused across candidates and methods; repeated calls are not new labels.
The whole source archive, calibration labels and selection archive are already
materialized. These experiments establish no label-saving or census-cost
advantage. The saved preparation and selection output fields record unique
queries and available archive sizes.

Four bounds fixed before source retrieval cover same-period average
conditional, availability-weighted gains of the primary reference rule versus
ungated and convex rules. For each fixed horizon T (1,008 or 8,760 hours), missing
hours contribute zero and the radius is sqrt(2 log(80) / T). Each level is
.0125; a union bound gives simultaneous 95% coverage under adapted forecasts,
without iid or stationary hours. All four lower bounds are nonpositive.
They do not certify later-period loss, raw unbounded MSE, or a rank-to-risk
implication. Weekly summaries are descriptive.

## Integrity and numerical correction

[PROVENANCE.json](PROVENANCE.json) records original input digests, unchanged
numeric-file digests, portable transformations and the original pre-download
timing. [MANIFEST.json](MANIFEST.json) checks this export's input bytes; its creation
does not backdate or renew the original protocol. A floating-point
category-centering error in the initial computation caused six category-copy
false rejections in the known-forecast methods. Exact integer residual
numerators remove them. Forecasts, draws, all 32 selected rules and the four
confirmation bounds remain unchanged. This correction was made after outcome
exposure, and the defective implementation is excluded from the default
reproduction path. Focused tests retain the exact-null regression cases.

Code follows the repository MIT license. Source and derived data are CC BY 4.0;
see [SOURCES_AND_LICENSE.md](SOURCES_AND_LICENSE.md). Only these public data
and their derived forecasts are included.
