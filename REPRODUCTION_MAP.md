# Paper, code and evidence

Run commands from the package root after installing `requirements-artifacts.txt`.
Use a fresh output directory. The main paper contains the core statements and
proofs; the technical companion holds expanded experiments and protocols.

| Main item | Recorded evidence | Reproduce or verify |
|---|---|---|
| Figure 1: earlier score fit, later outcome fit | The stated temporal weights and eligibility rule | `python scripts/analysis/panel_design_checks.py` checks the finite identities; `scripts/analysis/independent_entity_audit.py` runs the raw procedure on supplied data. The timeline is a schematic. |
| Figure 2: reference sampling and fitted means | `results/design_validation/peer_sampling.csv`; `results/learned_references/summary.csv` | `python scripts/figures/make_reference_training_figure.py --output-dir /tmp/reference-figure` |
| Figure 4: all 41 models on unchanged middle folds | `results/directional_comparison/model_comparison.csv`; `results/publication_figure_assets/public_directional_models.csv` | `python scripts/figures/make_directional_model_figure.py --input results/directional_comparison/model_comparison.csv --output-dir /tmp/model-figure` |
| Table I: available information, sampling and interpretation | Main definitions, theorems and complete audit procedure | This is a logical scope map, not an experimental estimate or an automatic assumption check. |
| Table III: exact feedback expectations and paired null results | `results/equal_entity_directional/summary.csv`; `results/equal_entity_directional/verification.json` | `python scripts/analysis/equal_entity_directional.py --output /tmp/directional-study --repetitions 500 --seed 914273` |
| Table V: complete-family temporal comparison | `results/inference_validation/fod_comparison/` and `results/inference_validation/raw_family_study/` | `python scripts/analysis/verify_inference_studies.py --output /tmp/inference-check` checks stored family decisions, supplied-null calibration ranks and the matched classical comparison. Full regeneration uses the study commands below. |
| Companion: scalar methods on common draws | `results/canonical_beta2/replications.csv`; `results/canonical_beta2/summary.json` | `python reproduce_artifacts.py --output /tmp/paper-results` recounts every cell. Generator and canonical comparator dependencies are documented beside these results. |
| Estimated conditional means and ties | `results/learned_references/replications.csv`; protocol and 48-cell summary | `python scripts/analysis/learned_reference_study.py --output /tmp/learned-study --workers 4` |
| Complete correlated families, K = 10 and 11 | `results/inference_validation/raw_family_study/` | `python scripts/analysis/independent_entity_family.py --output /tmp/family-study` regenerates the independent calibration bank and evaluation families. |
| Matched classical FOD moment | `results/inference_validation/fod_comparison/` | `python scripts/analysis/fod_comparison.py --verify results/inference_validation/fod_comparison`; use `--output /tmp/fod-study` for full regeneration. |
| Near-copy controls and weak alternatives | `results/inference_validation/near_copy_weak_study/` | `python scripts/analysis/near_copy_adjustment.py --output /tmp/near-copy-study` retains the original settings and the documented exploratory extension. |
| DLinear worked mean/SE/T/p and full-family BY | `results/spline_panel/public_all_models.csv`; `results/directional_comparison/model_comparison.csv` | `python reproduce_artifacts.py --output /tmp/worked-results` checks aggregate arithmetic and full-family adjustments. The worked example uses complementary all-fold fitting and exponent two. |
| Retail identity changes and mean/SE decomposition | `results/directional_comparison/model_comparison.csv` | The same-support figure command above exports all displayed coordinates and underlying means, standard errors, support and fallbacks. |
| Restricted stored-output provenance | `ERRATA.md`; `results/restricted_temporal_aggregate/`; `results/design_validation/` | Only model and simulation aggregates are distributed. Aggregate replay does not reconstruct restricted observations or establish corrected retraining. |

The package-level `reproduce_artifacts.py` combines aggregate checks and figure
reconstruction. `scripts/analysis/verify_inference_studies.py --output /tmp/inference-check`
recounts the learned-reference, complete-family and near-copy studies, checks
normal/t tails, independently checks selected BY decisions and calibration ranks,
and verifies exact synthetic expectations. Analytic floating-point comparisons
allow library-level rounding differences; hypothesis decisions and integer counts
remain exact. The focused tests independently enumerate small tied laws and
literal reference products, and verify temporal supports and family calculations.

The older scalar/resolution, oracle-rank and original all-fold public figures
remain in `results/publication_figure_assets/` and the technical companion.
They use their own clearly identified configurations; their coordinates should
not be substituted for the main same-middle-support comparison.

There are three different levels of reproducibility:

1. **Aggregate replay:** recount released numbers and redraw figures without provider data.
2. **Synthetic regeneration:** rerun the fully specified mechanisms from independent random streams.
3. **Forecast retraining:** obtain provider inputs and follow the documented model workflows. This is separate from the default replay; restricted original training is not publicly reproducible.

The package includes the licensed Seoul source archive and the derived
electricity archive used by the finite-archive study. Original observations for
the four-domain model comparisons and all restricted observations are excluded.
See `DATA_ACCESS.md` for study-specific access and reuse conditions, and
`MANIFEST.json` for the exact distributed files and hashes.

## Reference variance, finite bounds and held-out loss

| Main result | Code and evidence | Verification scope |
|---|---|---|
| Theorem 1, full covariance-null projection | `results/reference_covariance/reference_covariance_checks.py` and `results.json` | Rebuilds exact finite-support projection moments and six simulation cells. Run on a copy to preserve recorded results. |
| Proposition 2, observable categorical bound | `scripts/analysis/categorical_reference_audit.py`; `methods/categorical_references.md` | General numeric scores, ties, category validation, and complete observed bound. Input checks do not establish iid sampling. |
| Both categorical simulation studies | `scripts/analysis/verify_categorical_studies.py --base results/category_reference --output /tmp/categorical-checks.json` | Rebuilds all training/validation primitives, all 216,000 stored-row calculations and 20 complete evaluation replications. It does not regenerate all evaluation draws. |
| Companion: rental held-out gains | `results/seoul_confirmation/README.md` | Source-data refitting is separate from the fast aggregate, frozen-prediction and timing verifiers. |
| Equation (15), retail fit shifts | `results/retail_fit_changes/README.md` | Two models, five resolutions, 17 weeks and complete covariance; provider prediction cohorts are separate inputs. |

The categorical study producers use a Bernoulli-specific simulation shortcut.
The public CSV certificate instead uses the general sorted comparison kernel
in `scripts/analysis/peer_rank_products.py`, checked against literal ordered
triples including ties. Synthetic targets and ideal conditional means are
used only for study diagnostics; the executable certificate does not receive
them.

## Signed and variance-sensitive reference certificates

| Item | Recorded evidence | Command and scope |
|---|---|---|
| Signed categorical bias and full-U variance penalty | `results/reference_certificate_efficiency/DERIVATION.md` | `scripts/analysis/categorical_reference_audit.py --certificate signed_range` or `--certificate signed_variance`; rule and input order fixed before evaluation. The default remains `absolute_range`. |
| Five matched synthetic methods | `results/reference_certificate_efficiency/replications.csv.gz` and 500-cell `summary.csv` | `python scripts/analysis/reference_certificate_efficiency_study.py --output /tmp/reference-efficiency --replications 1000 --workers 4` regenerates all 500,000 rows. All training, validation and evaluation observations count toward cost. |
| Companion: five-method same-budget efficiency | The complete matched-method summary above | `python scripts/figures/make_efficiency_figure.py --summary results/reference_certificate_efficiency/summary.csv --output-dir /tmp/efficiency-figure` writes `fig_certificate_efficiency.pdf`, `.svg`, `.png`, and the complete plotted records in `.json`. Distinct markers identify all five methods; coincident values are plotted without offsets. |
| Independent implementation check | Portable `scripts/analysis/verify_reference_certificate_efficiency.py` | `python scripts/analysis/verify_reference_certificate_efficiency.py --output /tmp/efficiency-check` checks every stored row/cell and regenerates eight complete primitive replications (4,000 rows), using no producer imports. |
| Hardened independent-triple API | `protocol_original.json`, `protocol_hardened_replay.json`, `provenance.json` beside the results | Both complete historical runs have identical compressed result and summary bytes. The portable code rejects an independent-triple mean inconsistent with those same triples. The permissive historical helper is not distributed. |
| Prespecified family failure budget | `family_validation_delta` in `scripts/analysis/reference_certificate_efficiency.py` | `--family-size K` reserves rho of the first BY threshold; it does not apply BY or establish optimal allocation. Stored study results retain their original delta. |
| Fixed public archive application | `results/public_archive_certificate/README.md` | `python results/public_archive_certificate/verify.py` checks the sealed archive, independent replacement draws, all 21 certificate rows and census comparisons without writing or downloading. Full source reconstruction is separate. |
| Seoul gate decision and opportunity loss | `results/seoul_gate_diagnosis/README.md`, all six CSVs | Read-only aggregate verification: `python scripts/analysis/verify_seoul_gate_tables.py`. Full replay: `python results/seoul_gate_diagnosis/diagnose_gate.py --study-dir results/seoul_confirmation` using that study's separately pinned environment. This is exposed-data diagnosis, not a newly selected or confirmed gate. |

The default artifact reproduction runs the study-specific independent checks
in addition to every retained aggregate check. Neither a source hash nor a complete
arithmetic replay establishes optimality, absence of earlier exploration, or
general forecasting utility.

## Factorial attribution, joint information and chronological confirmation

| Main item | Evidence | Reproduce or verify |
|---|---|---|
| Table II: notation | Declared targets and recurrent error terms | Definitions, not experimental estimates |
| Figure 3(a): matched learning comparison | `results/certificate_factorial/`, all 400,000 factorial rows | `python results/certificate_factorial/verify.py --output /tmp/factorial-check`; redraw with `python results/certificate_factorial/make_figure.py --output /tmp/factorial-figure` |
| Table IV: matched bounded-mean betting | `results/matched_betting/`, all 48,000 method rows | `python results/matched_betting/verify.py`; full regeneration uses its `reproduce.py --output /tmp/betting-replay` |
| Companion: joint rank-mean allowance | `results/joint_bias/`, exact dual certificates and all 32 inspected candidates | `python results/joint_bias/verify.py --output /tmp/joint-check`; full replay uses `run.py --output /tmp/joint-replay` |
| Table VI: all primary Ridge confirmation rules | `results/forecast_confirmation/confirmation_results.csv`, with source ZIPs, forecast arrays and sampling indices | `python results/forecast_confirmation/verify.py --output /tmp/confirmation-check`; source retraining uses `reproduce.py --output /tmp/confirmation-retraining` |
| Companion: all 32 rules and four temporal loss bounds | Complete candidate, selected-loss, weekly and conditional-gain tables in the same directory | The confirmation verifier reconstructs every record. The bounds concern the same-period conditional average, not a future population or raw-unit risk. |

The factorial and joint studies inspect already observed simulation or archive
results. Their protocols and original hashes are distinguished from portable
code adaptations. The two public-task rules were fixed before source download;
the mechanically corrected computation is disclosed and is not relabelled as
a newly preregistered experiment. The new full retraining uses provider ZIPs
and does not deserialize the original scikit-learn 1.6 models.

## Aggregate learning-bias validation

[results/aggregate_bias/](results/aggregate_bias/README.md) contains the known-category-mass bound, all 32 exposed-archive candidates under four budgets, and 144,000 validation simulation replications. Its default reproduction rebuilds 128 diagnostic rows and independently checks both U/betting BY decisions; the full simulation uses `--regenerate-simulation`. The original chronological forecasting confirmation is unchanged.

Lemma 1 and Equations (8)–(9) define the aggregate allowance. A minimal
synthetic independent-pair example is available with
`python examples/aggregate_validation.py`. It computes a learning-bias bound,
not a complete association certificate or a forecast-retention decision.

Figure 3(b) displays all 72 validation settings from `results/aggregate_bias/`.
Run `python results/aggregate_bias/make_figure.py --output /tmp/aggregate-figure`
to regenerate the vector figure and every plotted coordinate. The complete
older three-panel factorial figure remains in the companion and public files.
