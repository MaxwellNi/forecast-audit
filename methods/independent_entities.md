# Independent-entity raw forecast audit

This entry point applies the paper's directional entity-mean construction to a supplied complete panel. It fits each prediction mean on earlier time folds and each outcome mean on later folds, evaluates only the middle folds, and averages equally across entities. Forecasts are supplied as inputs; the program does not train a forecasting model.

Use this procedure only when its scientific assumptions are justified: entities are independent, forecasts use the permitted past information, outcome innovations have the specified conditional mean-zero property, and the model family, cohort and fold rule are fixed before examining the audit outcomes. These assumptions are recorded in the output and cannot be confirmed by checking a CSV.

The input has five columns:

| Column | Meaning |
|---|---|
| `model` | Forecast identifier; every supplied model stays in the family. |
| `entity` | Entity identifier shared across all models. |
| `period` | Consecutive integer observation index, in chronological order. Use the actual prespecified observation schedule; do not conceal missing observations by renumbering after outcome-dependent selection. |
| `prediction` | Finite raw forecast value on the declared analysis scale. |
| `y` | Finite observed outcome, identical across models on each entity-period row. |

Each model must contain exactly the same complete entity-by-period panel. Duplicate rows, cohort holes, inconsistent outcomes, missing periods and nonfinite values are rejected. No rows are silently removed. At least two entities and five periods are needed for the default five-fold rule; useful inference requires more independent information than these minimal computational checks establish.

```bash
python scripts/analysis/independent_entity_audit.py \
  --input forecasts.csv \
  --output outputs/raw_entity_audit \
  --folds 5 --alpha 0.05
```

The output directory must be new. It contains:

- `profile.csv`: each model's mean score, standard error, signed statistic, raw probability, decision probability, BY-adjusted probability, label, abstention reason, complete family size and evaluation-support counts.
- `scope.json`: the target, fitting rule, assumptions, reference distribution, fold lengths, input checksum and interpretation. Source rows, entity identifiers, period values and predictions are not copied to the output.

The default uses the standard normal reference from the independent-entity asymptotic result. `--reference t` uses a Student reference with the entity count minus one degrees of freedom as a declared sensitivity choice; it supplies no exact small-sample guarantee. Both choices use sample standard deviation with `ddof=1`. Constant within-entity forecasts and standard errors at most `1e-10` cause abstention, with decision probability one. All models, including abstentions, remain in the BY calculation.

`RETAIN` means a positive decision under the stated assumptions. `NOT_RETAINED` does not establish absence of predictive value. `ABSTAIN` declines interpretation because the recorded numerical rule applies. The BY guarantee requires valid marginal inputs; choosing BY does not validate the scientific assumptions.

This entry point uses raw scores and entity means only. It does not introduce baseline ranks, general covariate adjustment, HAC inference, or a guarantee for the paper's separate observed-panel sensitivity suite.

Run the behavioral checks from the repository root with:

```bash
python -m unittest discover -s tests -p 'test_independent_entity_audit.py' -v
```
