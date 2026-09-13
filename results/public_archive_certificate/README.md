# Fixed public archive certificate

This package applies a finite-sample residual-rank certificate to a **fixed 70,080-row electricity forecast archive** using independent, uniform row-index sampling **with replacement**. It certifies association relative to the declared finite baseline/calendar categories in this archive. It does not certify future forecasting risk or incremental value beyond the full continuous seasonal forecast.

The primary pooled-U variance certificate retains both HGB and the previous-week forecast; the category-copy control is not retained. Exact census targets are 0.078219, 0.077414 and zero, respectively. All seven compared methods make the same retention decisions. The training-median baseline threshold is **zero**, yielding coarse zero/positive-load × weekday/weekend categories. These limitations are part of the result and were not changed after evaluation.

The audit uses 32,768 sampled row queries, including both members of each validation pair; these touch 26,140 unique rows. Forecast preparation already materializes the entire archive. The exact census costs about 0.039 seconds for all three candidates, compared with about 0.328 seconds for HGB pooled-U evaluation alone. Independent triples yield a similar lower bound with much less kernel computation. This example establishes a covered sampling design and a computable certificate, without demonstrating superiority over census or the strongest sampled comparators.

## Verify supplied results

Use Python 3.9 or later and the pinned dependencies. The recorded source replay used Python 3.9.21 and the versions listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python verify.py
python test_verify.py
```

Verification is read-only and needs no source download or model unpickling. It checks the release manifest and provenance hash links, regenerates all draw indices, and rebuilds every one of the 21 certificate records from the supplied archive. It imports no producer helpers. The independent calculations include:

- Training fits, validation comparisons and mean intervals for all 12 candidate/category cells; category-mass caps by binomial-CDF inversion; signed and absolute bias extrema by linear programming.
- All 12,288 candidate/triple scores, all 384 group means, and all 36,864 grouped plus 36,864 pooled focal products using literal pair comparisons.
- Fitted kernel ranges, sampling radii, raw probabilities, rejection flags, complete-family BY adjustments, query counts and census quantities.

Arithmetic comparisons use `rtol=1e-10` and `atol=1e-12`; [verification.json](verification.json) records the actual discrepancies. Runtime values are checked for finite nonnegative values and protected by hashes, not reproduced by this command. Local timestamps are checked for internal consistency, not externally authenticated. The eight tests include six in-memory corruption controls for signed allowances, raw probabilities, grouped means, kernel endpoints, validation mass caps and intervals; file hashes remain valid, so detection must come from arithmetic. Two additional tests check literal ordered triples and the independent-Bernstein mean requirement. This deterministic check does not estimate repeated-sample coverage.

## Reproduce forecasts from the official source

The source is Trindade, A. (2015), [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014), UCI Machine Learning Repository, DOI [10.24432/C58C86](https://doi.org/10.24432/C58C86), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [DATA_LICENSE.md](DATA_LICENSE.md). The official approximately 249 MiB zip is not duplicated in this package. Its URL, byte size and exact SHA256 are pinned in [protocol.json](protocol.json).

With an existing official zip:

```bash
python reproduce.py --source /path/to/electricityloaddiagrams20112014.zip --output /path/to/new-output
```

Or explicitly fetch the source and reproduce:

```bash
python reproduce.py --download-source --output /path/to/new-output
```

The output directory must not exist. The launcher verifies the source hash, trains the specified model chronologically, rebuilds the complete archive, repeats the fixed independent sampling, executes all declared comparisons, performs read-only verification and compares the resulting forecasts and tables with this package. It records reproduction timestamps separately. The comparison report distinguishes `np.array_equal` from tolerance checks and records maximum absolute differences for every numeric archive array and table column. Numeric tables use `rtol=1e-10, atol=1e-12`; numeric archive arrays use `rtol=1e-10, atol=1e-10`. A successful tolerance check alone does not establish exact equality. Runtime columns and execution timestamps are excluded from equality claims.

The supplied [reproduction_comparison.json](reproduction_comparison.json) records the pinned replay: all seven archive arrays and all compared non-runtime table columns are exactly equal, with zero maximum forecast or table difference. This is equality of loaded arrays/values; it does not assert identical model, compressed-file or timestamp bytes. This is a numerical reproduction of already exposed data, not a new independent confirmation or a recreated historical lock.

## Files and interpretation

- [REPORT.md](REPORT.md) explains the finite-archive target, actual sampling assumptions, full results and unfavorable comparator/census costs.
- [protocol.json](protocol.json), [archive_seal.json](archive_seal.json) and [audit_receipt.json](audit_receipt.json) record the fixed specification and provenance links.
- [certificate_results.csv](certificate_results.csv) contains all three candidates × seven methods. `primary` identifies the method named before evaluation; methods were not selected after results.
- [census_results.csv](census_results.csv) contains exact finite-archive targets, computed only after certificate output, never used to fit or tune it.
- [sampling_receipt.json](sampling_receipt.json) reports total and unique query counts, including validation references.
- [PACKAGING.json](PACKAGING.json) and [ORIGINAL_RUN_HASHES.json](ORIGINAL_RUN_HASHES.json) distinguish the original local execution from portable packaging. Numeric result, archive and model bytes are unchanged; public path/receipt hashes were updated transparently.

The individual lower bounds have their stated pointwise confidence level. Complete-family BY retention is computed separately for each fixed three-candidate method family. Original temporal dependence is not assumed absent: only randomized draws from the already fixed archive are iid. Historical raw publication vintages are not reconstructed.
