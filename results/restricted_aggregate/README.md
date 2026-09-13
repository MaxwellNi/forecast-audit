# Historical model aggregates

The three CSV files retain the recorded audit statistics and complete
30-model families for their respective eligibility rules. Matching JSON
files preserve the same rows and numerical values. These are retrospective
outputs under the original complementary-fold protocol, not a comparison
with the later directional audit or the separately rerun pretrained models.

The row now named **Untrained forecasting network** was previously mislabelled
TimesFM-2.5. The 29 excess-return-based historical forecasts also used a target
with a risk-free-rate unit error. See [the correction note](../../ERRATA.md)
for the effect and scope of both corrections.

Original `NULL` decision fields mean non-retention, not no information.
Readers and new rendered outputs use `NOT_RETAINED` for that same decision.
JSON `null` values remain missing values, distinct from decision strings.
No private observations, identifiers, predictions, residuals, loaders or
model weights are included.
