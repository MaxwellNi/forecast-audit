# Temporal fitting and peer sampling

These examples check expected residual-product scores under fully specified synthetic models. They help distinguish a formula error from a change in the sampling assumptions. They do not provide calibrated p-values for the application panels.

Run from the repository root:

    python scripts/analysis/panel_design_checks.py
    python -m unittest discover -s tests -p test_panel_design_checks.py -v

The first command regenerates design_checks.json without downloading data or reading another result file. It needs NumPy and the Python standard library.

The results include:

- Independent calculations of the temporal feedback expectation: the three-term formula and dense covariance contraction agree for 40 deterministic-seed causal designs.
- A sign reversal for a one-lag forecast as the number of rows per fold changes. Values are expected sums of residual products over the entity's observed rows, not test statistics.
- Zero expected numerator under time-directed fitting and uncorrelated innovations, using only folds with both earlier and later training rows.
- An AR(0.8) counterexample with expected time-directed numerator 0.35584. This violates the uncorrelated-innovation premise and shows why temporal order alone is insufficient.
- Exhaustive fixed-peer examples with binary outcomes and ties: shared-reference scores have expectation zero, while distinct references introduce expectations of -1/90 or +1/90.
- An independently resampled conditional-null example: the distinct-reference expectation is zero and the shared-reference expectation contains the reference-reuse term.

The functions reject invalid fold ordering, noncausal linear weights, nonpositive innovation variance, and missing temporal training sides. They cannot check whether a supplied real panel satisfies the innovation, sampling, or nuisance-learning assumptions. The identity checks therefore complement the paper's empirical sensitivity analyses.
