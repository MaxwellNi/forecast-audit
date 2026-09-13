# Forecast audit

A forecast can score highly by repeating information already present in a
baseline. This package tests the remaining association and explains how shared
rank references and fitted entity effects can create it. The paper supplies
explicit targets, corrections and error budgets; the experiments connect those
quantities to complete model comparisons and subsequent forecasting loss.
For independent categorical samples, a new aggregate validation bound estimates
the total learning bias directly. With exact category probabilities, it can
replace the sum of separate worst-case category errors.

Read the [paper](paper.pdf), *When a High Score Is an Illusion: Certifying
Genuine versus Repackaged Forecasting Skill*, or start with the
[method-to-evidence map](REPRODUCTION_MAP.md). The
[fixed release](https://github.com/MaxwellNi/forecast-audit/releases/tag/icdm2026-v7)
also includes [additional results and reproduction details](technical_companion.pdf).

## Choose the appropriate analysis

| Your question | Entry point | What the output establishes |
|---|---|---|
| Does a forecast add residual association under independent categorical sampling? | [Finite reference certificate](methods/categorical_references.md) | A lower bound for a declared marginal-rank covariance, given independent training, validation and evaluation |
| Are exact category probabilities available without forecast or outcome ranks? | [Aggregate validation](results/aggregate_bias/README.md) | A finite-sample upper bound for total learning bias, using independent validation streams and the stated category metadata |
| Can known marginal rank means improve the learning allowance? | [Joint rank-mean bound](results/joint_bias/README.md) | A conservative upper bias certificate using the same validation event |
| How do audit choices change an observed panel comparison? | [Panel command](methods/panel_audit.md) and the example below | Target-labelled statistics, abstentions and full-family decisions; a screen unless its assumptions are established |
| Does selecting an augmentation improve subsequent predictions? | [Chronological forecast study](results/forecast_confirmation/README.md) | Fixed-model and fixed-rule confirmation losses, with costs and dependence-sensitive uncertainty stated separately |

An association certificate and a future prediction gain answer different
questions. Every reported family includes copies, undefined scores and
non-retained candidates. The [data guide](DATA_ACCESS.md) states which sources
can be redistributed. The [correction note](ERRATA.md) records the identity
and target-unit qualifications for the restricted historical results.

## Install and run

Use Python 3.12 in a fresh environment:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-artifacts.txt
python reproduce_artifacts.py --output /tmp/forecast-audit-results
```

This command verifies recorded arithmetic, designated primitive replays and
figure generation. It does not retrain the original 41 predictors. Use
`--skip-figures` for arithmetic checks alone. Outputs go to a new directory;
the distributed evidence is preserved. The requirements pin versions used in
the checked Linux CPU environment.

To inspect a supplied panel:

```sh
python scripts/analysis/forecast_audit_cli.py \
  --input examples/forecast_audit_cli/synthetic_forecasts.csv \
  --output-dir /tmp/forecast-audit-example \
  --frequency cluster --lag 0 --beta 2 --ladder 2,3,4
```

Run `python scripts/analysis/forecast_audit_cli.py --help` for the input columns
and analysis arguments. Each output identifies its target, represented controls,
sampling assumptions and validity status. The command does not infer those
assumptions from data. The worked example is synthetic and outcome-free copy
checks do not calibrate an otherwise invalid p-value.

For the categorical reference example:

```sh
python scripts/analysis/categorical_reference_audit.py \
  --fits examples/categorical_reference/fits.csv \
  --validation examples/categorical_reference/validation.csv \
  --evaluation examples/categorical_reference/evaluation.csv \
  --certificate signed_variance \
  --output /tmp/reference-certificate.json
```

Choose the rule before evaluation. The small example illustrates the input
format and returns no positive lower bound. For exact category probabilities,
`python examples/aggregate_validation.py` demonstrates the aggregate allowance
from synthetic independent pairs; it supplies a bias bound only. The guide explains independent
samples, tie handling, row ordering, the declared family and counted costs.

## Reproduce the principal comparisons

| Evidence | Files and commands | Interpretation |
|---|---|---|
| Reference correction with learned controls | [48-cell study](results/learned_references/README.md) | At 8,192 training rows, 400 groups and 64 peers, matched null rejections change from 402 to 41 of 1,000. Small training samples remain difficult. |
| Learning allowance versus sampling precision | [Four-method factorial study](results/certificate_factorial/README.md) | At eight categories and 30,976 observations, signed versus absolute variance bounds detect 435 versus 274 of 1,000; the paired pointwise 95% gain interval is [0.131, 0.189]. |
| Strong classical comparators | [Five-method comparison](results/reference_certificate_efficiency/README.md), [bounded-mean betting](results/matched_betting/README.md) | Identical primitives, full costs and all outcomes. The proposed certificate does not dominate the classical alternatives. |
| Aggregate learning bias | [Validation and matched comparisons](results/aggregate_bias/README.md) | In a post-exposure diagnostic on 32 existing candidates, matched known-probability rectangle and aggregate betting rules retain 3 and 13 candidates across four separate families. A 72-setting simulation checks bound coverage and width; 10 of 72 settings have smaller median slack under the rectangle. |
| Joint rank-mean information | [Bias-bound study](results/joint_bias/README.md) | All 32 inspected allowances tighten, by median 1.24%; no reference decision changes. The LP gives an outer bound, not generally the attainable maximum. |
| Exact archive certification | [Electricity archive](results/public_archive_certificate/README.md) | A positive lower bound for a fixed archive and four declared categories; all seven matched methods retain the forecast. |
| Two forecast confirmations | [Appliances and Metro](results/forecast_confirmation/README.md) | Ridge augmentation reduces bounded loss by 16.95% and 73.87%; known-forecast gates equal ungated selection. The reference gates fall back to baseline. |
| Temporal feedback and classical moments | [Independent-entity study](results/equal_entity_directional/README.md), [full-family comparison](results/inference_validation/fod_comparison/README.md) | Exact raw-score bias identities, separate inference conditions, paired calibration and power |
| All 41 public predictors | [Same-support figure](results/publication_figure_assets/fig_public_directional.pdf), [model roster](results/public_score_link/public_model_roster.md) | Every model, with audit changes on fixed observations; sensitivity does not establish calibrated discoveries. |
| Retail changes and rental decisions | [Fit-change decomposition](results/retail_fit_changes/README.md), [Seoul confirmation](results/seoul_confirmation/README.md), [gate diagnosis](results/seoul_gate_diagnosis/README.md) | Exact observed decomposition, subsequent loss and all selected or excluded candidates |

Retrain the two new public forecast studies from their bundled source ZIPs:

```sh
python results/forecast_confirmation/reproduce.py --output /tmp/forecast-retraining
```

It fits models in the installed environment, checks all 78 saved arrays and
all non-runtime scientific table fields, and invokes an independent verifier.
Allow up to 40 minutes per phase. The separate Seoul study uses its own pinned
model environment; follow its README or pass `--confirmation-python` to the
package reproduction command. No incompatible serialized models from the two
new tasks need to be loaded.

Synthetic regeneration and study-specific verification commands are in
[REPRODUCTION_MAP.md](REPRODUCTION_MAP.md). Complete checks can be run with:

```sh
python -m pip install -r requirements-public.txt -r requirements-canonical.txt
python -m unittest discover -s tests
python -m unittest discover -s results/forecast_confirmation -p 'test_*.py'
```

## Contents and scope

`results/` contains all declared cells, including adverse outcomes and clearly
identified exploratory extensions. `scripts/analysis/` contains audit and
forecasting implementations; `scripts/figures/` contains deterministic plotting
code. `data_reference/` records provider sources and checksums. `MANIFEST.json`
identifies every distributed file. The paper and companion are supplied as
PDFs; document typesetting sources are separate from this code package.

The four original application panels are distributed as aggregates. The
Appliances, Metro and Seoul studies include their attributed public source ZIPs
under CC BY 4.0; the separate electricity study includes an attributed derived
archive. Restricted trading files contain only model-level and simulation
aggregates. Private observations, identifiers, predictions, residuals, source
loaders and model weights are excluded. Aggregate replay cannot reconstruct
restricted training or establish historical data and checkpoint availability.

Locally fixed protocols are distinguished from exploratory analyses and from
external preregistration. Hashes establish byte identity, not unobserved research
history. Forecast confirmation is distinct from fixed-archive inference; no
label-acquisition saving is established by a fully materialized archive. The
software uses the [MIT license](LICENSE); data and dependencies retain their
own terms.
