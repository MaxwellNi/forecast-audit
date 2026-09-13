# Data access and redistribution scope

This package contains code, synthetic records, public-task summaries and
model-level aggregates from the restricted case study. Public summaries retain
calendar group coordinates where needed; ratings groups use ordinal labels
instead of provider user IDs. These summaries are research outputs; the separately attributed sources
and derived archive below also include public observations.

The four original model-comparison panels are supplied as aggregates.
The Appliances, Metro and Seoul confirmations include their CC BY 4.0 provider
ZIPs and documented derived results. Source replay fits the new tasks
in the installed environment; Seoul also includes its separately pinned fitted parameters. The finite-archive electricity study also includes a
70,080-row derived daily-load and forecast archive, sampled indices and fitted
parameters under CC BY 4.0, with the source and transformations attributed in
`results/public_archive_certificate/DATA_LICENSE.md`.
Data licenses are separate from the software license.
Restricted-source observations, identifiers, dates, forecasts, residuals,
per-period totals, training inputs, loaders and model weights are also excluded.
Public access to a provider's download does not by itself grant redistribution
rights. The software MIT license does not license data.

| Dataset | Obtain from | Terms and released scope |
|---|---|---|
| Appliances Energy Prediction | [UCI record, DOI 10.24432/C5VC8G](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) | Luis Candanedo (2017), CC BY 4.0. Original ZIP, attributed hourly aggregation and forecast arrays are included under `results/forecast_confirmation/`; retain attribution and the documented transformations. |
| Metro Interstate Traffic Volume | [UCI record, DOI 10.24432/C5X60B](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) | John Hogue (2019), CC BY 4.0. Original ZIP, averaged duplicate hours, missing-hour representation and forecast arrays are included under `results/forecast_confirmation/`; retain attribution and the documented transformations. |
| Seoul Bike Sharing Demand | [UCI record, DOI 10.24432/C5F62R](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand) | UCI (2020), CC BY 4.0. The original source ZIP is included in `results/seoul_confirmation/source.zip`, together with derived study results and newly fitted model parameters. Retain attribution, the [license](https://creativecommons.org/licenses/by/4.0/), and the documented transformations. |
| ElectricityLoadDiagrams20112014 | [UCI record, DOI 10.24432/C58C86](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) | UCI identifies CC BY 4.0, permitting reuse with attribution. Cite Artur Trindade (2015), retain the [license](https://creativecommons.org/licenses/by/4.0/) and indicate transformations. The original model comparison is supplied as aggregates. The separate finite-archive study includes daily means and three candidate forecasts for 96 meters over 2013–2014, with draw indices and transformations documented in `results/public_archive_certificate/`. Obtain the unchanged 261 MB source ZIP from UCI using the documented checksum. |
| MovieLens 25M | [GroupLens official download page](https://grouplens.org/datasets/movielens/25m/) and its [README](https://files.grouplens.org/datasets/movielens/ml-25m-README.txt) | The dataset README permits research use subject to attribution and no implied endorsement; redistribution requires separate permission, and commercial/revenue use requires prior permission. No ratings, user/movie records or individual predictions are included. Consult the provider README when obtaining the data. |
| M5 Forecasting Accuracy | [Kaggle data](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and [competition rules](https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules) | Obtain data through the provider under its competition terms. Consult the current competition rules and account terms before downloading or redistributing provider data. No provider-data redistribution permission is granted by this package. No sales, prices, calendar rows or reversible per-record outputs are included. |
| Open Source Asset Pricing and factor inputs | [OSAP data](https://www.openassetpricing.com/data/) and [Kenneth French data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | Provider pages make downloads available; this audit did not establish an express blanket redistribution license for the underlying data. Obtain portfolio/factor observations directly under provider terms. No source return/factor rows or individual forecasts are included. |

`data_reference/` identifies provider-input checksums and the OSAP October 2025
numeric match. A current numeric match does not establish historical availability
at forecast origins. New refits must retain their own input hashes and protocol.

Restricted UK insider-trading observations combine disclosures with licensed
market data. The package supplies historical model aggregates
under `results/restricted_aggregate/`, retrospective comparison aggregates
under `results/restricted_temporal_aggregate/`, and the documented panel-design
summaries under `results/design_validation/`.
Their exact names and hashes are recorded in the current manifest. Arithmetic can be checked;
private training and row-level reconstruction cannot be performed from this
release. No permission to access or redistribute the private sources is
conferred. Private source loaders are excluded; see `DEPENDENCY_STATUS.json`.

The [correction note](ERRATA.md) identifies the untrained historical network
and the unit error in the stored excess-return target. Correcting that target
by a common within-month shift leaves ranks unchanged for fixed stored
predictions; it does not establish what models trained on corrected targets
would predict. A separate rerun checks forecast information cutoffs and
weight loading, without establishing historical availability or pretraining
cutoffs of the checkpoints. No private observations or model weights are
provided by those summaries.

The MIT `LICENSE` applies to the authors' software. Installed dependencies retain
their own licenses. No third-party model weights are bundled.
