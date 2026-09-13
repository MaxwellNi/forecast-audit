# Scope of the retrospective comparison

This table compares the actual custom adversarial residual-ranking MLP with Ridge regression and histogram gradient boosting on identical evaluation support. It also reports an unfit one-month momentum control. The neural result is the fixed arithmetic mean of three seeds; each seed is included separately, without selection.

All fitted models use the same 12 features: five complete-calendar lagged-return and volatility controls and seven disclosure count, buy-share and announcement-delay features. There are 18,036 common evaluation security/month observations across 11 expanding annual folds. All 55 predetermined fits completed: 33 custom-model, 11 Ridge and 11 histogram-gradient-boosting fits.

Training includes only observed next-calendar-month labels whose maturity is no later than the annual fit cutoff. Feature imputation and scaling use training data only. Missing labels are never imputed, model weights remain fixed within an evaluation year, and transaction rows are not replicated. The actual custom model retains its raw-rank and residual-rank losses, gradient-reversal issuer/month heads and stability penalty. Ridge and boosting fit the same training within-month rank target on the same transformed features.

The primary metric is the equal-weight mean of within-month Spearman correlations on common eligible periods containing at least ten rows and nonconstant outcomes and scores. The CSV also reports pooled Spearman and the standard deviation of the within-month correlations. These are model-level aggregates, not significance tests.

- The source extracts do not provide historical snapshots or publication/ingestion timestamps for transaction revisions, revised-primary security mappings, price adjustments, or dividends. Historical vintages are unverified.
- Transaction uniqueness is checked against the current full extract, and security identifiers and adjusted prices are current-source fields. The cohort is therefore selected retrospectively; verified train-only preprocessing and label-maturity checks do not establish historical availability of the evaluation population.
- Reported announcement days are filtered against forecast origins but have not been independently matched against archived original public announcements.
- The population is a conservative subset of covered securities with eligible disclosures and observed next-calendar-month outcomes. Vendor coverage, missing outcomes and delisting selection remain limitations; missing labels are not imputed.
- The comparison uses 12 conservative available features and is neither a reconstruction nor a reranking of the historical 30-model comparison.
- Ridge and histogram gradient boosting use fixed settings. This is not an exhaustive hyperparameter competition or an ablation of every component of the custom neural objective.
- The evaluation concerns temporal prediction within the covered security population; it does not establish generalization to held-out securities.
- Annual model weights are fixed after the fit cutoff. The assumption that the relevant data are observable after the calendar day ends is not verified by historical market-data publication records.
- The comparison was recorded before its own fits, but these reconstructed outcomes had already been used for a different simple-MLP experiment. It is not an untouched prospective holdout or an external preregistration.
- Metrics are descriptive. No confidence intervals, significance tests, finite certificate, prospective validity, economic-profitability claim or model selection is supported by this export.
- No external archive retrieval, pretrained checkpoints or reconstruction of all historical models was used.
