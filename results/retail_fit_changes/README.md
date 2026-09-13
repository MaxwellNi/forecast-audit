# Exact empirical accounting of the two retail reversals

The original reported reversals are reproduced on the **same 68,000 observations, 4,000 series and 17 middle-fold weeks**. Every component of the empirical fit-change identity reinforces the observed reversal. The forecast-fit component is largest for price/calendar; the outcome-fit component is largest for Croston. These are observed algebraic contributions, not identified population feedback or causal effects.

## Identity and unchanged audit construction

Write the complementary residuals as `rXc = X − fXc` and `rYc = Y − fYc`, with `X,Y` the original within-week standardized ranks. Let `dx = fXd − fXc` and `dy = fYd − fYc` denote directional minus complementary fitted values. Then, exactly for every row,

`rXd*rYd − rXc*rYc = −rXc*dy − rYc*dx + dx*dy`.

The three reported terms are:

- **Outcome-fit shift:** `−mean(rXc*dy)`.
- **Forecast-fit shift:** `−mean(rYc*dx)`.
- **Joint-fit shift:** `mean(dx*dy)`.

This identity is computed separately at each original resolution `(8,12,16,24,32)`. The resulting products and all three terms are then combined linearly using the **same exponent-two intercept weights** as the original comparison. Separately extrapolating residuals and multiplying them would create a different score; that operation is not used. The original baseline bins, whole-period five-fold partition, ranks, observations and pooled-row weights are fixed. All 28 original weeks supply the fitting supports; only original folds 1, 2 and 3 enter the reported means.

## Results on exactly matched support

| Model | Complementary mean | Directional mean | Change | Outcome-fit shift | Forecast-fit shift | Joint-fit shift |
|---|---:|---:|---:|---:|---:|---:|
| Past price and calendar | 0.0101375232 | −0.0015812759 | **−0.0117187991** | −0.0038165299 | −0.0044870051 | −0.0034152641 |
| Croston SBA | −0.0198785780 | 0.0055164499 | **0.0253950280** | 0.0124230467 | 0.0097007329 | 0.0032712484 |

For price/calendar the contributions are **32.57%, 38.29%, and 29.14%** of the total negative change. No single contribution accounts for most of the reversal. For Croston they are **48.92%, 38.20%, and 12.88%** of the positive change, with the outcome-fit shift largest. The signs agree at **all five individual resolutions** as well as after extrapolation; the reversals do not depend on extrapolation manufacturing a sign change from the opposite per-resolution pattern.

The original means, standard errors and statistics are recovered to floating-point precision:

| Model | Complementary SE | Directional SE | SE of paired change, including all cross-covariances |
|---|---:|---:|---:|
| Past price and calendar | 0.0015807678 | 0.0019394405 | 0.0015572934 |
| Croston SBA | 0.0010230692 | 0.0009035556 | 0.0010447662 |

These use the unchanged lag-two Bartlett HAC convention from the original comparison. They are descriptive paired scales on 17 observed weeks, not newly calibrated small-sample confidence statements. The full **15×15 covariance matrix** of the five-resolution-by-three-term vector is retained for each model. Its map through the original weights agrees with the directly computed three-term covariance. A separate **5×5 matrix** contains complementary score, directional score and the three combined terms. The SE of the change from the covariance of the three terms equals the SE from the direct paired score difference. Individual term SEs must not be combined as if the terms or resolutions were independent.

## What this explains; and what it does not identify

The calculation advances the earlier observation that means change more than standard errors: it determines which **empirical changes in fitted forecasts and outcomes** contribute how much. In these two models all three channels move in the same direction, with different largest contributors. It also verifies that the comparison is not an implementation discrepancy: complementary fitted values from the original sparse solver and the directional routine's dense solver, when given identical training supports, agree within **4.04×10⁻¹⁰** for price/calendar and **1.63×10⁻¹⁰** for Croston.

The paper's Eq. (6) is a population expectation identity under a specified linear temporal model. The table above is **not an estimate of its three structural feedback terms**. Here the forecast/outcome variables are standardized ranks, the baseline controls are bins, fitted projections change with training support, and the real data do not identify the temporal model's parameters or innovations. The empirical joint-fit term includes observed co-movement of both changes; its sign is not an estimate of a unique causal feedback mechanism. Forecast and outcome fitting occur jointly with entity and baseline-bin effects, so no uniquely meaningful entity-versus-bin causal attribution is supplied.

The decomposition is retrospective and focuses on the two already reported reversals. It is not an independent validation sample, new multiple-testing screen, or forecaster retraining experiment. It does not change any original decision, model family, guard, mean, standard error, or published support.

## Input provenance and reproducibility

This package releases aggregate evidence and code. The two source prediction parquets are **not included**. Source-level replay requires separately authorized access to the exact original inputs:

| Input | Model | SHA-256 |
|---|---|---|
| `model_03.parquet` | Past price and calendar | `e0b52f233a0d615c627406c599657e8b7f8bf077b101a0de6670ed3ea5fffe83` |
| `model_08.parquet` | Croston SBA | `eedc9e9a954530190dd1c26a8d75a923d1cf4720f2b29bdabed025ff7a1f4d4b` |

Both match the original directional-comparison receipt. The common full-cohort hash is `81801a5c0571a3c12620e1f20eae86f63be34859d97332266000440be3f41676`; the common middle-cohort hash is `8c9b3cdf069b547167775e04804da1cc4a9706ee4bd5e5f3fc6c594de1c0ab35`. These opaque hashes identify the recorded support; they do not disclose or reconstruct provider observations.

From this directory, run the public aggregate verifier using NumPy and pandas:

```sh
python verify_aggregates.py
```

The verifier checks public receipt links and original result identities, all ten resolution/model identities, weighted means, 34 weekly identities, both 5×5 HAC covariance matrices and both 15×15 covariance projections. It accesses no source rows and does not refit nuisance functions. `verification.json` is the original source-level execution record; `aggregate_verification.json` records the separate public aggregate checks. The latter cannot independently establish the source-row fit calculations in the former.

The portable producer requires explicit source file arguments and a fresh destination. Using the dependencies in the repository's `requirements.txt`, run:

```sh
python decompose.py \
  --price-calendar-input /path/to/authorized/model_03.parquet \
  --croston-input /path/to/authorized/model_08.parquet \
  --output ./reproduction
```

The placeholders must be replaced with authorized local files. The producer validates their recorded SHA-256 hashes before parsing them. It imports the released analysis modules and original directional-comparison aggregates from the surrounding repository. `--analysis-dir`, `--comparison` and `--comparison-receipt` allow explicit locations when running outside the full repository. `--help` lists the interface. Existing completed output directories are refused. The output contains model/resolution/week aggregates and opaque hashes; provider observations, predictions, residuals and entity identifiers are not exported.

Release preparation made the producer's source paths explicit, normalized module-location descriptions in the protocol, and updated their receipt links. [ORIGINAL_RUN_HASHES.json](ORIGINAL_RUN_HASHES.json) retains the original producer and protocol identities. Recorded timestamps describe the original execution; they are not evidence that the adapted public bytes ran at those times. All original numerical CSVs, source hashes and source-level verification results are unchanged. [MANIFEST.json](MANIFEST.json) inventories the current public package separately from the original execution receipt.

| File | Contents |
|---|---|
| `summary.csv` | Two model-level decompositions, original means/SEs, paired scales and cohort hashes |
| `per_resolution.csv` | All ten resolution/model identities and the original extrapolation weights |
| `weekly_aggregates.csv` | All 34 model/week aggregates, including all three contributions |
| `combined_hac_covariance.csv` | Two complete 5×5 covariance matrices |
| `cross_resolution_term_hac_covariance.csv` | Two complete 15×15 resolution/term covariance matrices |
| `decompose.py` | Portable source-level producer requiring explicit authorized input paths |
| `protocol.json`, `receipt.json` | Settings, original times and current public hash links |
| `verification.json` | Original source-cohort replay, row identities, dense/sparse equivalence and covariance mapping |
| `verify_aggregates.py`, `aggregate_verification.json` | Independent aggregate checks without source-data access |
| `ORIGINAL_RUN_HASHES.json`, `MANIFEST.json` | Original execution identities and current public package inventory |

The public producer was freshly replayed during packaging: all five released CSV tables reproduced byte for byte under Python 3.9.21. [portability_verification.json](portability_verification.json) records these comparisons.
