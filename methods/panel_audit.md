# Inspecting an observed forecast panel

Supply one CSV containing the complete declared model family. Default columns
are `model`, `entity`, `period`, `y`, `prediction` and `baseline`. Use the
column-name options in `forecast_audit_cli.py --help` if your names differ.
Declare the period or cluster frequency, uncertainty lag, extrapolation
exponent and resolution grid before inspecting the results.

```sh
python scripts/analysis/forecast_audit_cli.py \
  --input examples/forecast_audit_cli/synthetic_forecasts.csv \
  --output-dir /tmp/panel-audit-example \
  --frequency cluster --lag 0 --beta 2 --ladder 2,3,4
```

Read the output receipt and profile together. They identify the target,
represented controls, fitting and uncertainty rules, raw statistics and p-values,
copy abstentions and full-family BY values. A non-retained model is not evidence
of no predictive information. `finite_certificate_issued` is false: this panel
screen does not implement or verify the independent-reference theorem's sampling
design. Algorithmic input validation cannot establish a central limit theorem.

To compare temporal nuisance fits on the same rows, use
`scripts/analysis/directional_audit.py --help`. Earlier score fitting and later
outcome fitting are retrospective audit choices; the future outcome fit is not
a forecasting feature. With `--frequency cluster`, ordered identifiers do not
become calendar time. Empty training sides and unseen entities are reported
separately from the raw-score centering result. See
[independent-entity inference](independent_entities.md) for that distinct design.

For a literal finite certificate from independent training, validation and
reference draws, use the [categorical reference guide](categorical_references.md).
