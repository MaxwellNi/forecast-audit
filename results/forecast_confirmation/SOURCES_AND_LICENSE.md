# Public data sources and license

Source ZIPs are included without modification. Metadata and terms were checked
on 12 September 2026. To obtain replacements, download the linked ZIP and verify
its SHA256 before running any reproduction.

| Dataset and attribution | Primary source and download | ZIP SHA256 |
|---|---|---|
| Luis Candanedo (2017), Appliances Energy Prediction | [UCI dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction), [DOI](https://doi.org/10.24432/C5VC8G), [original ZIP](https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip) | `2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a` |
| John Hogue (2019), Metro Interstate Traffic Volume | [UCI dataset](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume), [DOI](https://doi.org/10.24432/C5X60B), [original ZIP](https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip) | `b99aeabcbd6cc86f642da3a79d90883425798f58abd3b3302da2fa19dda73768` |

Both datasets use [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Credit the dataset authors, retain these source and license links, and indicate
changes when redistributing. The derived hourly series, numeric forecast
archives, sampling indices and statistical result tables in this directory are
also distributed under CC BY 4.0. Changes include hourly energy summation,
duplicate traffic timestamp averaging, missing-slot reindexing, past-feature
construction, model forecasts, clipping, categories, augmentation and inference.
No endorsement by the dataset authors is implied. Code is separately covered
by the repository MIT license; the data license is not replaced by that code
license.

Appliances has 19,735 source ten-minute records and Metro has 48,204 source
records. Calendar aggregation, fixed windows and observed-target support
produce the reported row counts; source record counts are not treated as
independent sample sizes. Appliances' hourly interpolated airport variables
are excluded; Metro's duplicate hours and missing calendar slots are handled
by the declared deterministic rules.
