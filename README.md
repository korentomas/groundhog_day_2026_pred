# Groundhog Day Bayesian Reliability (North America)

This repo builds a weather‑based outcome label for each groundhog prediction and fits a hierarchical Bayesian model (PyMC) to rank reliability with credible intervals.

## Quick start

1) Fetch Groundhog Day predictions (already supported):
```bash
python fetch_data.py
```

2) Fetch Daymet weather for each groundhog location (North America, free):
```bash
python scripts/fetch_daymet.py
```

3) Build weather outcomes (significance‑based anomaly label):
```bash
python scripts/build_weather_outcomes.py
```

4) Build analysis dataset (join predictions + weather):
```bash
python scripts/build_analysis_dataset.py
```

5) Fit Bayesian model and export rankings:
```bash
python models/fit_pymc.py
```

Outputs:
- `data/weather_outcomes.parquet`
- `data/analysis_dataset.parquet`
- `data/groundhog_rankings.csv`

See `docs/method.md` for details on the statistical definition and model.
