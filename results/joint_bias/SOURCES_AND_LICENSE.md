# Public data attribution and changes

The bundled numeric archives derive from two public datasets:

| Dataset | Attribution and primary source |
|---|---|
| Appliances Energy Prediction | Luis Candanedo (2017), [UCI dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction), [DOI](https://doi.org/10.24432/C5VC8G) |
| Metro Interstate Traffic Volume | John Hogue (2019), [UCI dataset](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume), [DOI](https://doi.org/10.24432/C5X60B) |

Both use [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The derived numeric forecast archives, sampling indices and result tables
are also distributed under CC BY 4.0. Credit the dataset authors, retain
the source and license links, and identify the transformations when
redistributing. No endorsement is implied.

The upstream forecast study sums six ten-minute Appliances readings into
hourly energy, averages duplicate Metro timestamps, retains missing calendar
slots, constructs past-observation and calendar features, fits forecasts,
clips outputs, creates categories and freezes augmentation weights. This
directory copies those derived arrays unchanged and adds a post-exposure
joint bias-allowance diagnosis. It includes no original source ZIP or fitted
model and makes no new source-replay claim. Its five input files are identical
to their counterparts in the public forecast-confirmation export.

Code is separately covered by the repository MIT license. If this directory
is redistributed alone, retain a copy of that license. The code license does
not replace the data license.
