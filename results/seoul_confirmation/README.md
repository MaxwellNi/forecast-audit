# Seoul hourly forecasting study

This study evaluates whether residual augmentation improves forecasts on a held-out period. Under the Ridge baseline, augmentation reduces mean squared error by 22,431. The full comparisons also show that gating selects the same forecast as ungated augmentation for Ridge and blocks a useful forecast for the seasonal baseline. Read the [study report](REPORT.md) and [protocol](protocol.json) for the design and complete results.

From this directory, install the packages in `requirements.txt` and verify the released evidence:

```sh
python -m pip install -r requirements.txt
python verify.py
python verify_predictions.py
```

Both verifiers are read-only by default. The first checks public receipt links, original source/model/result hashes, selection and aggregate arithmetic. The second also checks the complete current manifest, rebuilds all 74 forecasts and 82 uncertainty comparisons from the fixed models, and runs three current/future-outcome perturbation checks. Results are recorded in `verification.json` and `prediction_verification.json`. These checks establish computational consistency of the released evidence, rather than temporal independence or inferential coverage.

The original producer used **Python 3.9.21**, NumPy 1.26.4, pandas 2.3.3, SciPy 1.13.1 and scikit-learn 1.6.1, as recorded in `protocol.json`. Its threadpoolctl version was not recorded. `requirements.txt` pins those recorded packages and threadpoolctl 3.6.0 for current verification. Use the recorded scikit-learn version when loading the serialized fitted models. Arbitrary newer environments and serialized fitted-model bytes are not promised to be identical.

To computationally replay fitting and the fixed design, use a fresh output directory and run these stages sequentially:

```sh
python study.py freeze --output ./reproduction
python study.py download --output ./reproduction
python study.py prepare --output ./reproduction
python study.py confirm --output ./reproduction
```

The script refuses reused protocol, selection and confirmation stages. A replay uses already exposed data and is not a second independent confirmation. Compare its CSV results with the released tables; timestamps and serialized bytes can differ between environments. The included `source.zip` is the original provider download, distributed under CC BY 4.0 with attribution in [DATA_LICENSE.md](DATA_LICENSE.md).

Release preparation normalized file-location descriptions in `study.py` and `protocol.json` and updated their receipt links. [ORIGINAL_RUN_HASHES.json](ORIGINAL_RUN_HASHES.json) preserves original code and receipt identities. Recorded times describe the original run, not execution of the revised public bytes. The source, fitted models and all result CSVs are unchanged. `MANIFEST.json` describes the current public package and is distinct from the original experiment receipts.

| Files | Purpose |
|---|---|
| `study.py`, `protocol.json`, `protocol.sha256` | Public executable design and protocol, with original-run chronology identified |
| `source.zip`, `source_receipt.json` | Unchanged provider source and its receipt |
| `fitted_models.pkl`, `selection.json`, `selection.sha256` | Unchanged fitted models and frozen choices, with public hash links |
| `selection_all_candidates.csv` | All 16 family members, guards, nominal/BY values, coefficients and decisions |
| `forecast_timing_check.json` | Recorded pre-confirmation timing check |
| `confirmation_metrics.csv`, `paired_risk_improvements.csv` | All 74 forecast risks and 82 paired uncertainty comparisons |
| `gain_identity.csv`, `weekly_trace.csv` | All 16 empirical identities and 888 weekly summaries |
| `confirmation_started.json`, `confirmation_receipt.json` | Original-run times and current public evidence links |
| `verify.py`, `verify_predictions.py` | Aggregate/provenance verification and full forecast/timing reconstruction |
| `ORIGINAL_RUN_HASHES.json`, `MANIFEST.json` | Original evidence identities and current package inventory |

The columns `confirm_oracle_gain_diagnostic_only` and `oracle_coefficient_diagnostic_only` are **ex-post empirical optima on fixed confirmation residuals**, not population oracles. They were never used for selection or deployment. HAC/Student intervals approximate dependence; BY arithmetic does not establish calibrated real-panel FDR. The local recorded protocol is not external preregistration.

The public producer was freshly replayed during packaging: all five released CSV tables reproduced byte for byte under Python 3.9.21. [portability_verification.json](portability_verification.json) records these comparisons.

For interval reconstruction, the verifier checks the two recorded producer Student critical values against an independent numerical inversion of the Student CDF. This preserves the strict endpoint check across small SciPy inverse-CDF approximation changes.
