# Figures and complete public-model table

The main figures are `reference_training.pdf` and `fig_public_directional.pdf`.
The first has 24 points: the original 18 peer-design observations plus six
estimated-mean observations. Its JSON records every count and pointwise interval.
The second preserves all 41 model identities on the same middle support, with
39 finite pairs, two undefined pairs and all eight original copy/order flags.
Its 82-row CSV retains both configurations and their means, standard errors,
support, fallbacks and decision values. See `../../REPRODUCTION_MAP.md` for
commands and exact scope.

The following paragraph describes the separate scalar and original all-fold
figures retained for the technical companion. Their configurations differ from
the main middle-support comparison.


The canonical figure shows 30 cells across six methods; all 54 source cells
are checked. The public figure shows all 41 models: 11 electricity and 10 in
each of ratings, retail and portfolios. It retains all 38 finite statistic
pairs, five finite copy/order guards and three undefined pairs. Coincident
statistics occupy separate named model rows, without numerical jitter.

The two-page table supplies the descriptive score, original-scale MAE, both
raw statistics, guarded BY p-value and diagnostic decision for every model.
Its PDF and Markdown round displayed values; the CSV retains stored precision.
RETAIN does not establish calibrated inference. NOT_RETAINED denotes non-rejection. The CSV preserves its original null decision codes.
ABSTAIN denotes a guard or an undefined audit. The SVD interaction score has
no rating-scale MAE. Pointwise Wilson intervals are not simultaneous intervals.

Run from the package root, using a fresh output directory:

```sh
python scripts/figures/make_publication_figures.py --source-root . --sources-json results/publication_figure_assets/source_map.json --output-dir rebuilt_figures
python scripts/figures/verify_publication_figures.py --source-root . --output-dir rebuilt_figures
```

The scripts use the Python dependencies in requirements-artifacts.txt.
The verifier independently checks SVG markers and error bars, all 54 Wilson
intervals and all 41 exact and rounded table rows. plot_coordinates.json
records each plotted coordinate. provenance.json binds input and output hashes.
No source observations are needed for these aggregate presentation artifacts.
