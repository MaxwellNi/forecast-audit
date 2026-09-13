# Paired factorial certificate study

This directory completes the absolute/signed learning allowance × range/variance sampling bound comparison on the **same previously inspected synthetic streams**. It is a post-exposure ablation, not an independent new confirmation. The analysis protocol was frozen before these additional results, and every original setting is retained.

The two-category headline has rejection counts **58 / 78 / 1000 / 1000** in the order absolute/range, signed/range, absolute/variance, signed/variance. Its large increase comes mainly from the sampling bound. A nonsaturated eight-category setting has **0 / 0 / 274 / 435**; the incremental signed effect under the variance bound is **0.161**, with a conservative exact pointwise paired 95% interval **[0.131, 0.189]**. Strong classical betting comparators can have substantially higher power; see the preserved matched comparisons below.

## One-command reproduction

With Python 3.12 and the pinned dependencies in `requirements.txt`, run from the repository root:

```sh
python results/certificate_factorial/reproduce.py --output factorial_reproduction
```

The output directory must be new. The command reconstructs all 400,000 method rows and summary/paired tables, independently verifies all rows and 200 training/validation cases, redraws the publication figure, and completely rebuilds original replications 0 and 17 from integer seeds using generic rank comparisons. It compares reconstructed outputs against the preserved evidence with explicit numerical tolerances and records whether each output is also byte-identical. No network access or new statistical experiment is involved.

For the eight full original replications used in the release check:

```sh
python results/certificate_factorial/reproduce.py --output factorial_extended_check --primitive-replications 0 1 17 123 317 503 777 999
```

Separate entries are available:

```sh
python results/certificate_factorial/analysis.py --output factorial_rebuilt
python results/certificate_factorial/verify.py --output factorial_verified
python results/certificate_factorial/make_figure.py --output factorial_figure
```

These commands write to output directories; they do not replace the preserved evidence. Historical executable hashes and a readable pre-execution correction diff are retained under `provenance`; unavailable historical source layouts are not distributed. Use the portable entries above for execution.

## Target, information and cost

Let R_X and R_Y be population marginal midranks, including ties, and let C be the declared category. The target is

`vartheta = E[(R_X - E[R_X|C]) (R_Y - E[R_Y|C])]`.

Conditional on independent fitted category means f,g, the common distinct-reference full-U statistic has mean `vartheta + b_H`, where `b_H = sum_c pi_c (muX_c-f_c)(muY_c-g_c)`. The four constructions change only the learning allowance and sampling radius. They share training draws, validation pairs, full-U center, fitted kernel range and evaluation rows. Both variance constructions use the same sample variance of disjoint symmetric triples, with `J=M*floor(N/3)`. The independent-triple and pooled comparators are kept separately.

All 100 original settings remain: two/eight categories; signals −0.125, 0, 0.125, 0.25, 0.5; M=25,100,400,1600; N=64; 8,192 training rows; validation of 512 or 8,192 pairs for estimated fits; and the original two-category opposed-shift cases at 8,192 pairs. Forty settings have nonpositive target. The total charged observations are `8192 + 2*validation_pairs + 64*M`. There are 1,000 paired replications per setting, alpha=.05 and delta=.0001.

The original absolute allowance is preserved exactly: on each original validation rectangle, multiply the largest absolute X fit error by the largest absolute Y fit error, then maximize its probability-weighted sum over the original capped simplex. The signed allowance maximizes the corner error product over that same simplex. Both use the same rectangles, probability caps and failure allocation. This does not replace the original absolute comparator with empirical weights, a worst-cell maximum, or another weaker definition.

The new absolute/variance cell copies the original signed/variance center, range, disjoint-triple variance and sampling radius, and replaces only its learning allowance with the original absolute allowance. The factorial producer makes no random draws. Three constructions retain the original numerical rows, and 100,000 new absolute/variance rows complete the grid.

## Attribution and confidence intervals

`paired_differences.csv` contains all six paired decision differences for each of the 100 fixed settings. If N+ and N− count the two discordant outcomes, each has a marginal binomial distribution even though they are dependent. Two two-sided 97.5% Clopper–Pearson intervals jointly cover their probabilities with at least 95% confidence by the union bound. Subtracting endpoints gives the reported difference interval. These are conservative exact **pointwise** paired intervals; no simultaneous coverage across 600 comparisons is claimed. They are neither independent-proportion intervals nor McNemar tests relabeled as intervals.

Every paired lower-bound difference decomposes exactly:

`L_signed,variance - L_absolute,range = (B_absolute - B_signed) + (r_range - r_variance)`.

Power is thresholded and can interact. `attribution.csv` reports the interaction `(SV-AV)-(SR-AR)` and the two order-averaged contributions. The interaction is ternary, so the same paired interval applies. Order-averaged learning and sampling effects have supports [0,1] and [−1,1], respectively, and use stated pointwise Hoeffding intervals. Grid sums are descriptive because different settings share draws.

The full-U empirical variance construction and the independent Bernstein/betting methods are classical ingredients. The factorial evidence separates their contribution from the learning allowance; it does not support a general novelty or efficiency-dominance claim.

## Stronger comparisons and negative evidence

`matched300_comparator_summary.csv` retains all 32 existing comparator settings and 300 shared replications, with eight methods; its paired table contains all 128 declared contrasts. At the existing weak two-category target .005 and cost 30,976, signed full-U, independent Bernstein, independent betting, pooled-U and pooled betting reject **1 / 10 / 263 / 2 / 273** out of 300. Absolute/variance also rejects 1/300. Independent betting minus signed full-U has paired difference .8733 [ .8094, .9132 ]. Test-only betting has no lower-bound radius, so its coverage fields are correctly unavailable. The preserved 48,000 comparator input rows are also supplied; the separate `matched_betting` study provides its full producer and dedicated checks.

Across the factorial grid, all 160 nonpositive method-setting cells have zero observed rejections and all 400 cells have zero observed coverage failures. This does not establish zero failure probability. All original weak, null, negative, rare-category and persistent-shift settings remain visible.

## File map and provenance

- `factorial_rows.csv.gz`: all 400,000 rows, including primitive center/variance/range/allowances, raw p, lower bound and decisions.
- `summary.csv`, `paired_differences.csv`, `attribution.csv`: all 400 method-setting summaries, 600 paired contrasts and 100 attribution records.
- `compact_focus_table.csv`, `COMPACT_TABLE.md`, `main_figure_data.csv`, `persistent_table.csv`: every predeclared display setting.
- `fig_certificate_factorial.pdf/png/json`: compact 7 by 2.05 inch figure and every plotted value. Only notation, leading zeros and hats differ from the earlier rendering. Lines connect the two displayed budgets; they do not estimate intermediate power.
- `inputs/original_efficiency`: unchanged original 500,000 rows, 500-cell summary and original generation protocol; old published evidence remains available at `results/reference_certificate_efficiency`.
- `inputs/matched_betting`: unchanged 48,000 existing comparator rows and frozen comparator protocol.
- `provenance`: exported original scientific protocol, pre-execution correction history, historical executable hashes, translated first-run receipts, original independent-verification receipt and the relative input-path map.
- `PROVENANCE.md`: what changed during portable export and what did not.
- `release_check`: results from executing the portable entries before publication; not a new independent confirmation.
- `MANIFEST.json`: hashes of files in this study directory.

No private data are needed. The integer seed, categorical laws, generation shapes and independent primitive reconstruction are supplied. A full original-simulation replay is also supported by the existing public reference-efficiency study; this package's default command rebuilds the additional factorial analysis and the declared bounded primitive checks.
