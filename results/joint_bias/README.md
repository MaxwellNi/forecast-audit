# Joint rank-mean bias allowance

The exact marginal mean of a population midrank is one half, including ties.
Adding this identity for both rank channels couples the category confidence
rectangles. A conservative linear-programming allowance is no larger than
the same-input separable allowance. The code distinguishes the mathematical
LP optimum from the certified numerical upper bound actually returned.

This is **post-exposure exploratory analysis**, not a new confirmation study.
All 32 candidates are retained. Their allowances decrease by a median 1.24%
and a maximum 6.52%, but every variance and betting raw p-value remains one.
Every BY value is also one; both reference methods still fall back in all
four task/baseline families. The analysis provides no observed forecasting
improvement or recovery of the failed practical success criterion.

## Run and verify

With the repository's installed Python environment, run from the repository
root:

```sh
python results/joint_bias/run.py
```

The directory is also self-contained: from inside it, run `python run.py`.
Install its pinned `requirements.txt` first if using a separate environment.
No network, model pickle, private file or neighboring directory is required.
The command takes approximately 10–20 seconds on one CPU, checks all bundled
hashes, runs the mathematical checks, reconstructs all 32 rows, compares them
with the stored results and invokes the independent verifier. It writes only
to a temporary directory and then removes that directory. To retain outputs,
use a **new, nonexistent** directory:

```sh
python results/joint_bias/run.py --output joint-bias-replay
```

For read-only verification of the recorded results without solving an LP:

```sh
python results/joint_bias/verify.py
```

The independent verifier imports neither the producing implementation nor
an optimization solver. It checks all 32 rational dual certificates,
reconstructs the archive inputs and p-values, and checks 200 additional finite
rank laws and 240 exact corner-formulation conversions. The separate
`check_joint_bias.py` also exercises numerical-failure fallbacks, malformed
inputs, upward rounding, tampered receipts and 120 finite frames with ties.

`reproduce.py --output NEW_DIRECTORY` runs only the reconstruction. Both
`reproduce.py` and `verify.py` accept `--inputs` for an equivalent directory
containing the five input files. `verify.py --results DIRECTORY` checks a
reconstruction instead of the recorded rows; optional `--output` writes its
receipt into a new directory. The default verification is read-only.

## What is reconstructed

The local `inputs/` directory contains byte-identical copies of five files
from the public forecast-confirmation study: two numeric forecast archives,
their sampling indices, and the recorded certificate table. This permits an
independent copy of this directory to run without the rest of the release.
The data sources and attribution are in [SOURCES_AND_LICENSE.md](SOURCES_AND_LICENSE.md).

For each of two tasks, two baselines and eight candidates, the analysis keeps
4,096 training draws, 8,192 independent validation pairs and 12,288 evaluation
draws, totaling 32,768 row queries. It rebuilds the training category fits,
validation comparison means, mean confidence rectangles and category-mass
upper bounds. The confidence allocation, 32 categories and all draws are
unchanged. The new allowance uses only training and validation observations.
Census ranks and means are used afterward for labeled diagnostics.

Evaluation retains the previously verified full-U mean and sampling radius.
The independent checks rebuild the literal six-role disjoint triple kernels,
variance, kernel range and both raw p-value inversions. This is a reconstruction
of the allowance change, not another full forecast-training or full-U source
replay. Exact numeric equality is reported separately from tolerance-based
agreement: the comparison uses absolute and relative tolerance 10⁻¹¹ and
reports the maximum absolute discrepancy. Different solver versions may
return different valid dual proposals and slightly different certified
upper values. Every proposal is checked using exact arithmetic.

## Mathematical and numerical scope

On the existing simultaneous validation event, the true signed learning
bias is no larger than the mathematical joint LP optimum, which is no larger
than the same-input separable optimum. The half-mean identities require no
additional error allocation. This permits substitution in an already proved
finite certificate under independent sampling from one common population.
For these archives, independence is created by replacement row-index draws
conditional on the fixed archive; it is not an iid claim about hourly data.
The target remains covariance of population marginal ranks after adjustment
for the declared categories, not conditional independence or predictive loss.

The solver's primal objective is diagnostic only. The implementation uses
exact binary-rational weak duality and a positive-residual correction to
certify an upper bound. It then takes the minimum with an exact separable
upper bound and rounds upward. Solver failure or a reported empty joint
set returns the separable allowance. An empty probability box returns one.
Invalid intervals raise an error. Exact LP certification is conditional on
valid input confidence intervals; it does not certify the upstream floating
evaluation of a binomial quantile.

The LP is an **outer relaxation**, not a sharp attainable bias optimum.
With one category, both mean intervals [0,1] and both fits one half, it returns
one quarter although the true bias is zero. A nontrivial three-category
rank law has LP optimum 1/64 but attainable coherent maximum 3/256.
Conversely, an attainable two-category example improves the allowance from
45/128 to 5/16. See [DERIVATION.md](DERIVATION.md) for proofs and classical
optimization references. Neither LP duality, McCormick convexification,
concentration inequalities nor BY is presented as a new general method.

## Files and provenance

`recorded/all_candidates.csv` contains every allowance, true-bias diagnostic,
raw p-value and BY value; [ALL_CANDIDATES.md](ALL_CANDIDATES.md) shows all 32
rows. The 32 JSON files in `recorded/certificates/` preserve exact inputs,
multipliers and fraction strings. `joint_bias.py` is the unchanged core;
`verify.py` is a separate arithmetic implementation. No old result is replaced.

[PROVENANCE.json](PROVENANCE.json) records the construction freeze, first
archive run, original code digests and portable interface changes. The core
and check code were frozen before the first new allowance result. That local
freeze does not make this exposed-archive analysis prospective or externally
preregistered. The export manifest was created after the calculations and
does not backdate them. Code uses the repository MIT license; source-derived
arrays and results use CC BY 4.0 with the attribution below.
