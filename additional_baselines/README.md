# Supplementary synthetic sensitivity experiments

This directory is a standalone reproduction of four supplementary diagnostic experiments: adaptive exponent selection, a nine-setting ablation, an eight-implementation overlap/regime grid, and four stress designs. It contains synthetic generators only. It needs no research repository, external dataset, real-panel loader, credentials, model download or GPU.

The implementation directory contains the complete supplementary synthetic methods. The protocol identifies their packaged source hashes and experimental settings. The results are retrospective diagnostic comparisons, not a new preregistration.

## Reproduce or recount

The completed replay used Python 3.9.21, NumPy 1.26.4, SciPy 1.13.1 and pandas 2.3.3. The four selected drivers use no torch functionality; neural functions in one bundled dependency are unused. Install these dependencies in an isolated environment if needed:

```bash
python -m pip install numpy==1.26.4 scipy==1.13.1 pandas==2.3.3
```

From this directory, check all source hashes and imports without running simulations:

```bash
PYTHONDONTWRITEBYTECODE=1 python replay.py --check
```

Run all four declared settings into a **new** destination, then recount the individual decisions:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python replay.py --output ../my_new_synthetic_run
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python recount.py --results ../my_new_synthetic_run
```

An existing replay destination is refused. Both workers use one numerical thread each. All settings and seed sets are fixed; the expected output is 53,740 replicate records, 54,140 binary decisions and 416 rate cells. A reference host completed the four-suite replay in about twenty minutes; this is an estimate, not a stopping rule. All four suites must finish before summaries are produced. Failed-run logs are retained and a failed result must not be silently replaced by a favorable retry.

`recount.py` also accepts the completed released results directory when it contains the four `*.raw.csv` files and `RUN_COMPLETED.json`. It verifies cell dimensions, exact seed sets and recorded hashes, recomputes rates and pointwise Wilson intervals, checks adaptive arithmetic, and compares integer counts against the recorded completed replay in `protocol.json`. It writes `all_cell_rates.csv`, `adaptive_paired_summary.csv`, `ablation_summary.csv`, `generalization_summary.csv`, `reference_count_comparison.csv` and `recount.json`. Copy the results to a separate directory before recounting, because the command writes the derived summary files. No simulation is run by the recount command.

## Scope of the comparisons

The following limitations govern interpretation of every supplementary comparison:

- Zero latent added signal does not prove the audited ranked conditional null with random peer ranks. The rates measure rejection under that generated condition; they are not validated Type-I error or size-corrected power.
- The adaptive experiment evaluates one misspecified, clipped log-slope heuristic which reuses evaluation products. Its failure does not prove that beta cannot be estimated. Changing beta does not follow the fixed-beta second-order weight frontier.
- The long ladder is retained and also changes rejection materially. Beta is not the uniquely sensitive setting. The recorded `.18/.90` tolerance rule is not nominal `.05` calibration.
- The grid's sign-weighted and projected-covariance implementations have outer-label feedback through their first-stage residuals. Local permutation omits firm from its conditioning cells. These are not established faithful implementations for comparisons against WGCM, PCM or a valid conditional permutation procedure.
- The RFF/distance routines in the broad dependency are not canonical KCI or partial distance correlation. They are not among the eight implementations in this replayed grid. The broad dependency's full benchmark is not invoked by `replay.py`.
- Stress designs include overlapping/current-period outcomes and retain all losses of positive-signal rejection. Their diagnostic output labels do not establish robustness, calibrated inference or forecasting superiority.
- Pointwise Monte Carlo intervals quantify simulation uncertainty. They provide neither simultaneous coverage across selected extrema nor a proof of calibration, an unavoidable error floor, valid multiple-testing control or deployable forecasting gains.

Joint nuisance fitting and balanced synthetic panels address sequential-fit feedback and unequal-cluster centering within these routines. They do not resolve the separate rank-null, dependence, adaptation or comparator issues above.

Use `replay.py` for the documented comparisons. Other module entrypoints are outside this reproduction protocol. This directory contains no real-data assets or loaders.
