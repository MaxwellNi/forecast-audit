# Seoul gate diagnosis

Read [REPORT.md](REPORT.md) for the exact 16-member gate decision, eight-week selection means and every candidate loss. Seasonal HGB fails BY, not a guard or rank/raw-scale mismatch. The realized gated-minus-ungated MSE is 299,789.52 for seasonal control and exactly 0 for Ridge.

Run `python diagnose_gate.py --study-dir ../seoul_confirmation` for read-only replay from unchanged source/model/results. The six CSVs preserve all candidates, including guarded copies. Counterfactual p-value arithmetic is explicitly retrospective and is not new confirmation. No rule or coefficient is retuned.
