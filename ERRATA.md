# Corrections and interpretation of the revised release

This note records corrections to the September 10 manuscript and released
results. Numerical records remain available with their original analysis
settings; the revised paper distinguishes those settings from the subsequent
panel-design checks.

## Identity of one historical forecasting row

The historical row labelled TimesFM-2.5 was produced by untrained forecasting
networks. An incompatible model class left all 333 parameter tensors newly
initialized. Different random seeds produced five networks; each stored
prediction averaged only the networks that covered that observation. The row
is now named **Untrained forecasting network**, with the original audit
statistics retained. It is not evidence about the pretrained TimesFM-2.5 model.
No full-family retention decision changes as a result of this correction.

A separate rerun used the appropriate TimesFM-2.5 architecture and verified
232 matching parameter tensors under two loading seeds, alongside the
checkpoint-file records. Its results are reported separately. This loading
check does not establish the checkpoint's historical availability or
pretraining cutoff at every forecast origin. Matching two seeded loads is
an additional diagnostic; loading reports, architecture compatibility and
checkpoint identity must also be checked. Forecast outputs alone do not
establish which pretrained weights generated them.

## Units of the stored training target

The stored excess-return target used by 29 of the 30 historical forecasters
subtracted the risk-free rate in percentage-point rather than decimal-return
units. The error is common to all observations within a month. Holding the
stored predictions fixed, correction by a common monthly shift preserves
within-month outcome ranks and the corresponding rank-audit statistics.
It does not preserve a hypothetical model retrained with the corrected target,
nor does it validate the historical training series or economic interpretation.
The historical comparison remains a retrospective audit of stored outputs.

## Sensitivity of the retail illustration

Under the original complementary-fold settings, the M5 forecast named
Past price and calendar has statistic 7.68 and passes the full-family screen.
Under directional nuisance fitting, its statistic is -0.82 on the middle
folds and -1.21 on all folds; it is not retained in either case. The original
arithmetic is correct for its settings, but the example is not robust to the
training direction. The revised paper uses the ratings comparison to
illustrate why prediction accuracy and residual association answer different
questions.

A comparison restricted to the same middle-fold observations separates the
change in evaluation support from the change in nuisance fitting. The
complementary-middle procedure retains five models; the directional-middle
procedure retains seven. For Past price and calendar, the statistic changes
from 6.41 to -0.82 on that shared support. Croston SBA changes from -19.43 to
6.11. Directional fitting adds Croston SBA, Eight week mean, and Ridge with
history, and removes Past price and calendar. All 41 models and all four
configurations are reported in [the matched-support comparison](results/directional_comparison/).

The original all-fold complementary set and directional-middle set both
contain seven models, but comparing those sets also changes evaluation
support. The directional all-fold set contains six. Neither matching support
nor a change in retained sets identifies a causal feedback effect or
establishes calibration. The recorded wild-cluster bootstrap applies to the
original complementary-fold products, not the directional products. It is a
separate variance-sensitivity check.

## Reading the additional checks

Directional fitting keeps the original baseline-bin assignments and
full-cohort guard. Empty earlier or later training folds use pooled bin fits
on the complementary folds, and entities absent from a training side receive
the stated fallback prediction. These boundaries are reported explicitly.
The ratings task orders user clusters rather than calendar periods, so its
result is a partition-sensitivity comparison, not a temporal guarantee.

The reference construction depends on the sampling design. Exact centering
results for resampled peers or fixed peers require their stated assumptions;
a stable retained set is not evidence that those assumptions hold. Likewise,
new time-direction and clustering comparisons do not prove a joint calibration
guarantee. The paper reports them as diagnostic evidence.
