# Groundhog Reliability: Weather-Outcome Definition + Bayesian Model

## Data sources (free)
- Groundhog predictions and metadata are pulled from the Groundhog Day API endpoints (`/api/v1/groundhogs` and `/api/v1/predictions?year=...`).
- Weather observations use Daymet v4 R1, a 1 km daily gridded dataset covering North America (Canada, U.S., Mexico) from 1980 onward.
- Daily data are retrieved via the Daymet single‑pixel REST API.
- Anomalies are computed relative to the 1991–2020 climate normals baseline, consistent with NOAA’s current standard for normals (30‑year window updated every decade).

Sources:
- https://groundhog-day.com/api
- https://daac.ornl.gov/DAYMET/guides/Daymet_Daily_V4R1.html
- https://daymet.ornl.gov/web_services
- https://www.weather.gov/grr/1991-2020ClimateNormals

## Outcome definition
We treat a groundhog’s prediction as a binary forecast about the post‑Feb‑2 weather window.

### Prediction mapping
- **Shadow (shadow = 1)** → predicts **six more weeks of winter**
- **No shadow (shadow = 0)** → predicts **early spring**

### Weather outcome (binary)
We compute the mean temperature anomaly for a window after Feb 2 (default: Feb 3–Mar 16):

- Daily anomaly = `tavg_daily - climatology_tavg(doy)`
- Window mean anomaly = average of daily anomalies in the window

Statistical significance:
- We run a one‑sample t‑test of daily anomalies vs 0.
- If **p ≤ alpha** (default 0.05) and mean anomaly > 0 → **early spring**.
- If **p ≤ alpha** and mean anomaly < 0 → **winter**.
- Otherwise → **uncertain** (excluded from accuracy modeling by default).

Missing‑data tolerance:
- The window requires at least `(window_days - tolerance_days)` valid daily values.

## Bayesian model
We model **correctness** (prediction matches outcome) using a hierarchical logistic model:

`correct ~ Bernoulli(p)`

`logit(p_i) = mu + alpha_groundhog[g[i]] + beta_year * year_z + beta_lat * lat_z + beta_lon * lon_z`

- `alpha_groundhog` is a partial‑pooling effect for each groundhog.
- Covariates reduce confounding from location and time.
- Rankings are computed from posterior samples of `sigmoid(mu + alpha_groundhog)` (i.e., mean covariates).

Outputs:
- `data/groundhog_rankings.csv` – posterior mean and 95% HDI for each groundhog
- `data/posterior_summary.csv` – model parameters
- `data/model_trace.nc` – full posterior samples
