# Validate the total learning bias

A residual-rank certificate subtracts an allowance for errors in its fitted
conditional means. This study estimates their **total weighted product** from
two independent validation streams, instead of constructing a separate
worst-case rectangle for every category. The bound needs exact category
probabilities and independent training, validation pairs and evaluation draws.
[THEORY.md](THEORY.md) gives the statement, proof and classical antecedents.
Its conditioning fixes the focal validation categories only. Reference
observations, including their category labels, retain the common marginal law.

From the installed repository environment, use a new external output directory:

```bash
python results/aggregate_bias/reproduce.py --output /tmp/aggregate-bias-reproduction
```

This command rebuilds all 128 archived diagnostic rows, independently verifies
all 32 candidates and their U/betting family decisions, then checks all 144,000
stored simulation replications and 144 method summaries. It reads the immutable
forecast arrays and sampling indices in [forecast_confirmation](../forecast_confirmation/README.md).
It does not retrain forecasts or create a new chronological confirmation.
The distributed study may be read-only; all output goes to the chosen directory.

To regenerate every replication in the fixed simulation stream:

```bash
python results/aggregate_bias/reproduce.py --output /tmp/aggregate-bias-full --regenerate-simulation
```

For a separately implemented simulation-law check, including a second seed that
samples raw focal/reference pairs rather than a compressed categorical law:

```bash
python results/aggregate_bias/verify_simulation_primitives.py --output /tmp/aggregate-bias-primitive-check
```

The primitive check reconstructs 288 selected original replications and 1,600
new pair-sampling repetitions across 24 fit settings. Those produce 4,800 fit
rows and 48 method summaries. The second stream checks bias coverage and width;
it is not an independent forecasting task. Full regeneration requires NumPy,
pandas and SciPy from the root environment, one CPU thread and approximately
2 GiB available memory. Allow several minutes for the complete simulation.

## What is measured

The archive comparison holds every forecast, training draw, validation pair,
evaluation draw and candidate family fixed. It separates four budgets:

| Allowance | Information beyond validation draws | U retained | Betting retained |
|---|---|---:|---:|
| Original category rectangles | None | 0 | 0 |
| Rectangles with known masses | Complete category metadata | 1 | 3 |
| Direct aggregate | Complete category metadata | 9 | 13 |
| Exact archive bias, diagnostic only | Full archive outcomes | 14 | 16 |

Counts total four separate eight-candidate families. They are not counts of
independently confirmed discoveries. The direct aggregate passes all recorded
category-copy and independent-noise controls. In Appliances/Ridge, the
ExtraTrees allowance decreases from 0.029275 for the same-metadata rectangle
to 0.004328; combining it with the existing classical betting test gives a BY
value of 0.003062. Metro/Ridge/ExtraTrees remains unretained. Its aggregate
allowance is 0.005736, above its observed association of 0.004083.

The simulation uses 72 settings, each with 2,000 repetitions and two methods:
2, 8, 32 or 128 categories; 512, 2,048 or 8,192 validation pairs; balanced or
unequal masses; and exact, same-direction or opposite-direction fitting errors.
It measures bias-bound coverage and slack, including absent categories.
It does not measure forecast selection power. Zero observed noncoverage is
not zero error probability.

## Interpretation and provenance

Known category probabilities are a real information requirement. They are
available from the entire control column of a fixed archive; sample category
frequencies from validation alone do not justify this theorem. Both new budget
comparators receive the same metadata. The existing known-forecast methods
also know the full forecast column and remain relevant stronger-information
comparators. Knowing a named baseline function alone does not reveal either
conditional mean.

The two forecast tasks were inspected before this method was developed. This
is a post-exposure diagnostic, not external preregistration or a replacement
for the original prespecified forecasting success result, which remains 0/2.
The method improves an error allowance under its stated information model;
it does not establish universal statistical efficiency, incremental forecasting
benefit, general dependent-panel calibration or a new concentration theory.

[PROVENANCE.json](PROVENANCE.json) separates original input hashes from portable
code adaptations. [MANIFEST.json](MANIFEST.json) covers this study's files.
A simulation audit corrected the floating-point reduction order used to replay
the same seed before its separate raw-data seed was run; the distribution,
settings, seeds and bounds did not change. The earlier forecast-confirmation
centering correction remains documented in that study's original provenance.

Code follows the root MIT license. The existing public data and derived archive
arrays retain the CC BY 4.0 terms in
[forecast_confirmation/SOURCES_AND_LICENSE.md](../forecast_confirmation/SOURCES_AND_LICENSE.md).
All additional simulation outputs are provided under CC BY 4.0. This directory
contains no restricted trading observations or manuscript source.
