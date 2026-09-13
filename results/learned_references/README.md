# Estimated means for tied rank references

This fully synthetic study has an exactly zero population-midrank residual target. It compares shared and distinct references with exact means and with independently fitted means. The protocol states the Bernoulli law, seeds, nested evaluation groups and training sample sizes. The learner receives empirical training midranks, not the true marginal distribution.

`replications.csv` preserves all 48,000 outcomes. `summary.csv` gives all 48 cells, pointwise Wilson intervals, conditional expected means, training-error products and exact synthetic-law error diagnostics. These diagnostics are not confidence bounds for unknown real distributions and are not used to calibrate the tests.

From the package root:

```sh
python scripts/analysis/learned_reference_study.py --output /tmp/learned-reference-study --replications 1000 --workers 4
```

Training samples of 32 and 512 observations retain substantial failures. The largest training sample demonstrates feasibility in this finite-control sampling design; it does not establish calibration under dependent or fixed peers. There is no alternative-signal or family-power claim in this study.
