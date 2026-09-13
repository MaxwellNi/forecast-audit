# Matched bounded-mean betting comparison

This extension compares the signed reference certificate with classical
bounded-mean betting on the same training, validation and evaluation draws.
It reuses already exposed synthetic streams; it is not independent confirmation.
All 32 settings, five methods and 300 paired repetitions are distributed.

## Reproduce or verify

From the repository root, use the main Python requirements:

```sh
python results/matched_betting/verify.py
python results/matched_betting/reproduce.py --output /tmp/matched-betting-replay
```

The verifier checks all 48,000 rows, their 28,800 matching original rows,
640 level-specific summaries and exact binomial intervals. It independently
rebuilds 512 betting rows from eight primitive simulation replications without
importing the producer. Full reproduction writes a new directory and can take
several minutes. Row comparisons should exclude runtime and new execution time.

## Target and construction

Conditional on independent training, the distinct-reference kernel has mean
`vartheta + learning_bias`. Independent validation supplies an upper bias
allowance with failure probability at most `delta`. The same allowance is used
by all five methods. For a fixed kernel range `[lower, upper]`, let
`X = (kernel - lower) / (upper - lower)` and
`m = (allowance - lower) / (upper - lower)`. On the validation event, under
`vartheta <= 0`, the mean of `X` is at most `m`.

For each of 64 predeclared fractions, multiply
`1 + fraction * (X / m - 1)` over independent disjoint triples. The equal-weight
mean of these products is an e-value; the reported conservative p-value is
`min(1, delta + 1 / evalue)`. The two limiting threshold cases are handled
separately in the implementation. Neither kernel roles nor overlapping triples
are treated as independent observations. A fixed finite grid of fractions is
not universally consistent for arbitrarily small effects. Fits and range are
fixed independently of evaluation; the API cannot verify that sampling design.

This is the classical bounded-mean betting construction described by
[Waudby-Smith and Ramdas](https://doi.org/10.1093/jrsssb/qkad009).
It is not a new general concentration inequality. The helper also implements
the variance-sensitive U-statistic comparator of
[Maurer and Pontil](https://proceedings.mlr.press/v75/maurer18a.html).

## Results and provenance

At eight categories, 100 groups, 64 peers, 8,192 training rows and 8,192
validation pairs, the positive target 0.013655 rejects in 130/300 signed
variance runs and 296/300 independent betting runs. Classical pooled methods
are also retained. All nonpositive cells have zero observed rejections, whose
pointwise 95% binomial upper endpoint is 0.012221. Zero observed events are
not a zero-risk guarantee. With only 512 validation pairs all positive cells
remain undetected.

`recorded/protocol.json` is the original pre-execution specification for this
post-exposure extension. Its hashes refer to the original source layout.
`SOURCE_MAP.json` records original and portable source hashes. Portable changes
only relocate dependencies and add a new-output argument; they do not backdate
the adapted files or assert byte identity with the original executable.
`MANIFEST.json` covers this distributed directory. Complete paired outcomes
are in `recorded/paired_decisions.csv`.
