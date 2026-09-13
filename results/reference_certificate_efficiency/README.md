# Signed and variance-sensitive reference certificates

This follow-up retains all 500,000 simulation rows across 500 cells, including
negative and zero effects, rare categories, weak signals and persistent opposed
mean-fit errors. It studies population marginal-midrank residual covariance
for a declared finite category target under independent common-law sampling.
It is not evidence of future forecasting gain or arbitrary panel calibration.

Five methods share exactly the same training, validation and evaluation draws:
absolute range, signed range, signed full-U variance, classical independent
triples, and pooled full-U variance. The signed bias allowance is no larger
than its absolute counterpart on the same confidence set. The variance rule
uses a fixed disjoint-triple estimate of the symmetrized kernel variance and
retains the full group statistic. Its p-value need not dominate signed range
on every dataset. Do not select the smaller rule after viewing evaluation data.

The independent-triple empirical Bernstein comparator follows classical
concentration theory. Pooling is admissible in this common-law simulation and
is a strong comparator. Their power matches or exceeds the proposed grouped
rule in several cells; a new general empirical Bernstein inequality or broad
superiority is not demonstrated. See [the derivation](DERIVATION.md) for source
attribution, the signed bias interval, exact constants and nondegenerate and
zero-variance cases.

Each cell contains 1,000 trials. The table fixes 8,192 validation pairs and
8,192 training observations; all evaluation rows and both members of every
validation pair count toward the displayed total.

| Design and effect | Evaluation groups of 64 | Total observations | Absolute range | Signed range | Group full-U variance | Independent triples | Pooled U variance |
|---|---:|---:|---:|---:|---:|---:|---:|
| Two categories, target 0.01 | 100 | 30,976 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| Two categories, target 0.02 | 25 | 26,176 | 0.000 | 0.000 | 0.985 | 0.972 | 0.991 |
| Eight categories with a rare cell, signal 0.25 | 100 | 30,976 | 0.000 | 0.000 | 0.435 | 0.433 | 0.461 |
| Eight categories with a rare cell, signal 0.5 | 25 | 26,176 | 0.000 | 0.000 | 0.916 | 0.867 | 0.933 |

Every method has zero observed power in all positive cells with only 512
validation pairs. All 200 nonpositive-signal cells have zero observed
rejections; that is a finite Monte Carlo result, not proof of zero error
probability. The methods and nested sample sizes share primitives, so the
500,000 rows cannot be treated as independent trials of a single procedure.

Run from the package root into fresh output directories:

```sh
python scripts/analysis/reference_certificate_efficiency_study.py --output /tmp/reference-efficiency --replications 1000 --workers 4
python scripts/analysis/verify_reference_certificate_efficiency.py --output /tmp/reference-efficiency-independent
```

The first command regenerates the synthetic study with the hardened public
helper. The second uses an independent implementation with generic comparisons,
average-rank training and linear programs; it checks all stored inference
arithmetic and 500 summaries and reconstructs eight full primitive replications
(4,000 rows). It imports no producer helper. Neither command uses private data.

`protocol_original.json` preserves the first run. The later
`protocol_hardened_replay.json` identifies a complete rerun after adding an API
check that independent-triple inference uses the mean of those same triples.
Both `replications.csv.gz` and `summary.csv` are byte-identical across the two
runs; `provenance.json` records their hashes. The original study already met
the stronger condition, so no result changed. The permissive original helper
is identified by hash but is not distributed. Portable source paths and the
new family-budget helper change current source hashes; no claim is made that
the portable files were the exact bytes of either historical run.

All stored results use delta=0.0001. The optional public CSV `--family-size`
rule sets delta=rho*q/(K*H_K) before data, making first-rank BY resolution
possible. It is not an empirical result under a retuned budget, and is not a
proof of optimal power or sample allocation. Training observations must be
counted separately when using the CSV interface with externally fitted means.
