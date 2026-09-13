# A finite reference bound with categorical controls

Use this procedure when independent training, validation and evaluation samples
come from the same population, with a fixed finite set of control categories.
Each evaluation group contains the same number of iid peers. The target is the
covariance left between population midranks after their conditional means are
removed. It is a signed association target, not a test of arbitrary conditional
independence or of forecast improvement in original units.

The lower bound is observed distinct-reference association, minus an allowance
for imperfect mean fits, minus sampling uncertainty. Disjoint validation pairs
estimate category-specific means without a known marginal distribution.
One-sided binomial intervals bound category probabilities, so a rare uncertain
category need not receive the weight of the whole population. The paper's
finite-reference result proves the resulting bound under its sampling conditions.

Three input files are required:

| File | Columns | Meaning |
|---|---|---|
| Fits | `category,forecast_mean,outcome_mean` | Fitted population-midrank means in [0,1], from separate training data. Every allowed category must be declared. |
| Validation | `pair,role,category,forecast,outcome` | Each disjoint pair has one `focal` and one `reference` row; both rows count toward validation cost. Numeric raw scores may have ties. |
| Evaluation | `group,category,forecast,outcome` | Balanced groups of at least three iid peers from the same law. |

Run the command shown in the main README into a new JSON path. It reports the
mean, bias allowance, kernel range, concentration penalty, lower bound, p-value
and observation counts. `POSITIVE_LOWER_BOUND` means the declared bound exceeds
zero at the chosen level; `NO_POSITIVE_LOWER_BOUND` does not prove no value.
`sampling_assumptions_verified` remains false: CSV validation cannot establish
sampling independence or the common law. Record training costs separately.

The default `--certificate absolute_range` and validation failure budget
0.0001 preserve the earlier interface. The test level defaults to 0.05.
`--certificate signed_range` uses the largest signed bias over the same
validation confidence set. This allowance can be negative and is never larger
than the absolute allowance on that same set. `--certificate signed_variance`
adds a finite variance-sensitive penalty for the full U-statistic. The two
signed rules are separately valid; their data-dependent minimum is not an
automatically valid p-value. Fix the rule before observing evaluation data.

The variance rule needs at least two independent triple scores in total.
Within each group it takes consecutive disjoint triples in input row order,
averages all six focal/reference assignments into **one** score per triple,
and uses their sample variance. It does not use the variance of dependent row
products as though those rows were independent. Every row, including leftover
rows, remains in the full rank statistic. The input order must be fixed
independently of values; sorting by forecast or outcome before forming these
triples would invalidate this construction. The CSV checks cannot verify that
requirement.

The validation budget, rule, categories and family must all be fixed before
data inspection. A p-value has infimum `delta`, so at K=100 and q=0.05 the
default 0.0001 exceeds the first BY threshold 0.0000963878. It blocks an
isolated rank-one rejection, not all higher ranks. `--family-size 100`
chooses `delta=rho*alpha/(K*H_K)`, with `rho=0.1` by default and `alpha`
serving as the declared family FDR level. `--family-rho` can set another
prespecified fraction in (0,1). Alternatively supply `--delta`, but do not
supply both budget mechanisms. Smaller delta widens validation uncertainty;
this is a p-value resolution rule, not a proved power optimum.

The command reports each marginal result and `family_adjustment_applied=false`.
It does not collect a model family or apply BY. Apply the full-family rule
separately to the complete family of valid inputs, retaining all candidates.
Arbitrary dependence between models is allowed by BY, but each certificate
still requires its own stated marginal sampling design. Choosing categories,
budgets, methods or models from evaluation outcomes needs an additional
validity argument.

The simulation evidence is in `results/category_reference/`. It retains the
maximum-cell bound and probability-weighted follow-up on identical primitives.
The latter improves rare-cell detection in the reported strong-signal setting,
but has larger allowances on some individual draws and does not solve all weak
signals. The largest reported configuration uses 8,192 training observations,
8,192 validation pairs and 400 groups of 64 peers: 50,176 observations in total.
The [derivation](../results/category_reference/DERIVATION.md) gives the exact
formulas and the distinction between both versions.

The [signed and variance-sensitive extension](../results/reference_certificate_efficiency/README.md)
retains the old outputs and compares five methods on exactly the same
observations. Its full cost is 8,192 training rows plus twice the validation
pair count plus all evaluation rows. The stored study uses delta=0.0001;
the optional family-size helper is not a rerun of that study at smaller delta.
The classical independent-triple and admissible pooled full-U comparators
match or exceed its observed power in several settings. No minimum-cost,
uniform efficiency, fixed-panel, or original-unit forecasting guarantee follows.
