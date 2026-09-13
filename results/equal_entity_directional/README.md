# Raw equal-entity directional benchmark

This experiment isolates a specific feedback mechanism on independent entity trajectories. It uses raw scores and outcomes, with no ranks, baseline bins, HAC estimator, or data-dependent eligibility. It supplies an executable example for the independent-entity inference result; it does not establish validity of the paper's complete ranked-panel audit.

Reproduce the simulation and its four tests from the repository root:

```bash
python -m unittest discover -s tests -p test_equal_entity_directional.py -v
python scripts/analysis/equal_entity_directional.py --output /tmp/equal-entity-directional --repetitions 500 --seed 914273
```

The output directory must not exist. The script refuses to overwrite recorded results. Choose another empty destination if a previous reproduction already occupies the example path.

## Design

Each entity has independent standard normal outcome innovations, independent forecast noise, and two independent constant entity intercepts. Its outcome is its first intercept plus the current innovation. Its forecast is its second intercept plus 0.2 times the sum of its previous `lookback` outcomes, plus independent noise. Five presample innovations supply the past values. At injection zero, the forecast has no information about the current outcome innovation.

There are 25, 100, or 400 independent entities; 20 or 40 observed times per entity; and lookbacks of one or five times. All five folds are consecutive and fixed before generation. Both compared methods evaluate exactly the same three middle folds. Each entity's score is its mean evaluated residual product; the overall estimate gives every entity equal weight.

The standard method fits both entity means on the complementary folds. The directional method fits the forecast mean on earlier folds and the outcome mean on later folds. The third, simple gapped comparator also uses a common training set for both means, but excludes the evaluation block and `lookback` additional times on each side. It is included only in null cells. This is a controlled mean-fitting comparison, not a reproduction of a complete estimator from prior work.

The gap removes direct lookback links between evaluated rows and their training rows. It does not necessarily remove covariance between the two fitted means, because they still share the same training observations. The directional construction separates the relevant past and future innovations in this design. Its exact mean score is zero at the null. Independence across entity trajectories permits classical studentization as the number of entities grows with trajectory length fixed. The result does not make the finite-sample normal threshold exact.

Positive controls add 0.1 or 0.3 times the current outcome innovation to the score. These are deliberately injected associations, not forecasts that could be made before observing that outcome. Common random numbers link all entity counts, trajectory lengths, lookbacks, methods, and injection levels. All 84 cells and all 500 replications per cell are retained.

## What the recorded experiment shows

With 400 entities, 20 times, and lookback five, the standard, gapped, and directional null statistics have means −2.677, 3.148, and −0.002. Their positive rejection counts are 0, 458, and 27 out of 500. The exact corresponding raw-score expectations are −0.0450521, 0.0655329, and zero. These numbers demonstrate why removing direct training links alone can leave a product-of-fitting-errors term.

The directional null rejection rates over the full tested grid range from 0.054 to 0.088. Some pointwise intervals exclude 0.05; this experiment does not establish uniform finite-sample nominal size. At 400 entities, all tested positive directional cells reject 500 out of 500, but smaller samples have substantially lower detection rates. These are raw normal-threshold detection counts. They do not establish superiority at independently matched error rates or on other data-generating processes.

## Files and fields

- `replications.csv.gz`: one row per simulation replication, configuration, and method. The rows contain scalar summaries, not the generated trajectories.
- `summary.csv`: all 84 cells, including rejection counts, pointwise 95% Wilson intervals, statistic mean and standard deviation, exact mean score, Monte Carlo standard error, and the ratio of reported to empirical standard errors.
- `protocol.json`: seeds, dimensions, coupling, generation and inference rules, and generator hash.
- `verification.json`: a separate recount of saved replications and a comparison with an independently evaluated covariance formula.

`injected_association` is the coefficient on the current innovation. `mean_entity_score` is the average over entities of each entity's mean residual product. `standard_error` is the sample standard deviation of the entity scores divided by the square root of the entity count. `statistic` is the mean divided by that standard error. `evaluation_rows_per_entity` is 12 or 24. `reported_se_over_empirical_sd` compares the average reported standard error with the across-replication standard deviation of the estimated mean, rather than with the standard deviation of individual observations.

The function rejects nonfinite, noninteger, unordered, or noncontiguous fold codes. It also requires at least one eligible evaluated row with earlier and later training observations for every included entity. Eligibility must be fixed independently of the shocks; the API cannot establish that scientific condition from an array alone.
