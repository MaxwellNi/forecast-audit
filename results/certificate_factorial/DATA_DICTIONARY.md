# Data dictionary

Rows are keyed by `replication`, `design`, `signal`, `fit`, `training_rows`, `validation_pairs`, `groups`, `peers`, `total_observations`, and `method`. Replication IDs are paired across methods and settings. The method-setting grid has 100 settings × four methods × 1,000 replications.

| Field | Meaning |
|---|---|
| `design` | `two_balanced` or `eight_rare`, with masses and Bernoulli margins fixed in the protocol. |
| `signal` | Conditional Bernoulli covariance multiplier gamma; this is not itself the rank target. |
| `fit` | Empirical category means, or the predeclared clipped +0.15/−0.15 opposed shifts. |
| `target` | True conditional category-residual covariance of population marginal midranks. |
| `exact_bias` | True mean displacement b_H for the realized independent training fits. Used for diagnostics only. |
| `mean` | Same grouped full-U center for the four factorial methods. |
| `absolute_allowance`, `signed_allowance` | Original validation bounds computed from the same rectangles and category mass caps. |
| `bias_upper` | Allowance actually used by this method. |
| `bias_interval_width` | Signed validation upper bound minus signed validation lower bound. |
| `kernel_lower`, `kernel_upper` | Fitted kernel support used by the inference rule. |
| `effective_triples` | J=M floor(N/3); used for full-U concentration and independent-triple variance calibration. |
| `kernel_sample_variance` | Unbiased sample variance of the independent symmetric triple kernels; unavailable in stored range-only rows. |
| `radius` | Sampling allowance at alpha=.05, delta=.0001 under the selected bound. |
| `lower_bound` | `mean - bias_upper - radius`. |
| `p` | Original inverse-certificate probability, including its validation error correction. |
| `reject` | Original strict decision `p < .05`. |
| `lower_covers`, `allowance_covers` | Simulation diagnostics against the exact target and true bias; not used to construct the bounds. |

In `summary.csv`, `replications` and `rejections` are integer counts and `power` is their ratio. All intervals there are pointwise binomial intervals; for zero or negative targets this is a rejection rate, not alternative power. Remaining means average the stated quantity within a fixed setting and method.

In `paired_differences.csv`, `after_only` and `before_only` are the discordant counts. `both` and `neither` complete the paired 2×2 table. `delta=(after_only-before_only)/replications`; `ci_lower` and `ci_upper` are the exact conservative pointwise paired interval described in the README. This does not treat the two methods as independent samples.

In `attribution.csv`, the `*_lcb_gain` columns split the continuous lower-bound improvement into learning and sampling components. `interaction` is `(SV-AV)-(SR-AR)` on paired binary decisions. The `*_average_delta` columns average the two factor orders. Their interval methods and supports are recorded in the file.

The matched-comparator summaries use 300 replications and eight methods. A betting method supplies a test only, so its radius, lower bound and coverage diagnostics are unavailable, not failures. Raw input rows and the separate matched-betting study preserve the complete comparator record.
