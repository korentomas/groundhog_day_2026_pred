# Groundhog Day 2026 Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/korentomas/groundhog_day_2026_pred/blob/main/notebooks/02_bayesian_hierarchical.ipynb)

A side project applying real Bayesian methods to the least serious dataset available: over a century of groundhogs guessing at the weather.

Every February, dozens of groundhogs across the US and Canada "predict" whether winter will run long (saw shadow) or end early (no shadow). The [groundhog-day.com API](https://groundhog-day.com/api/v1) has recorded these predictions back to 1886. This project treats that record as a genuine binomial time series: is the shadow rate actually 50/50, does it drift over time or by region, and which groundhogs are consistently more "shadow-prone" than others, once you account for the fact that some of them have one data point and Punxsutawney Phil has 131.

## Viewing this

GitHub renders `.ipynb` files with their outputs baked in, so a plain link into `notebooks/` is enough to see all the numbers and plots with zero setup:

- [`notebooks/01_shadow_eda.ipynb`](notebooks/01_shadow_eda.ipynb): frequentist EDA.
- [`notebooks/02_bayesian_hierarchical.ipynb`](notebooks/02_bayesian_hierarchical.ipynb): the actual Bayesian analysis.

Or open the Bayesian notebook directly in Colab with the badge above if you want to rerun or tweak the models yourself.

## Data source

All data comes from the public [groundhog-day.com API](https://groundhog-day.com/api/v1): groundhog metadata (name, location, type) and yearly shadow predictions from 1886 onward. 88 groundhogs are in the registry, and only 37 of them are real, living groundhogs. The rest are taxidermied, animatronic, or otherwise not a groundhog (there's an alligator and a lobster in there).

## Reproducing

```
pip install requests pandas numpy matplotlib seaborn scipy plotly kaleido pymc arviz
python fetch_data.py   # pulls groundhogs.json, predictions.json, combined_data.json
python run_eda.py      # runs the shadow-rate EDA, prints stats, saves plots + bayesian_input.json
jupyter nbconvert --to notebook --execute --inplace notebooks/01_shadow_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_bayesian_hierarchical.ipynb
```

`data/` is gitignored (`*.json`, `*.png`, `*.geojson`). Everything in it, including the Natural Earth admin-1 boundaries geojson used for the map in the Bayesian notebook, is fetched or generated on demand. Nothing in this repo depends on committed data files, and every plot that matters is generated inline in the notebooks rather than linked from a saved file, so the committed `.ipynb`s render correctly on their own.

## What's here

- `fetch_data.py`: pulls raw groundhog and prediction data from the API, writes `data/groundhogs.json`, `data/predictions.json`, `data/combined_data.json`.
- `run_eda.py`: script version of the core EDA. Loads `combined_data.json`, computes overall and yearly shadow rates, runs a binomial test against p=0.5, breaks results down by decade and groundhog type, saves `data/shadow_analysis.png`, `data/groundhog_type_analysis.png`, and `data/bayesian_input.json` (summary stats formatted for the downstream Bayesian model).
- `notebooks/01_shadow_eda.ipynb`: the notebook version of the above. Load, clean, and explore the shadow data: overall and yearly rates, binomial and chi-square tests, groundhog-type breakdown, and prep for the Bayesian stage.
- `notebooks/02_bayesian_hierarchical.ipynb`: the real analysis. Two hierarchical PyMC models (per-groundhog partial pooling, and a crossed groundhog x decade model to check whether the apparent post-1980 trend survives once you control for who was reporting), a posterior-based ability ranking, and a geographic map of the groundhog registry. Both models converge cleanly: r-hat ~1.0, zero divergences.

## Future work

See the Future Work section at the end of `notebooks/02_bayesian_hierarchical.ipynb` for what's next: fake vs. real groundhog comparisons, Phil-agreement rates, per-groundhog streakiness, career-length effects, and geographic clustering. Checking predictions against actual NOAA spring-onset data was investigated and dropped: the API's own temperature field only has usable values for 13 of 1761 predictions, too sparse to build anything on.

## Notes

- "Ability" ranking in the Bayesian notebook is based on the hierarchical posterior mean, i.e. how often a groundhog says shadow relative to the pooled population, not a measure of forecasting accuracy against real winter outcomes. There's no ground-truth winter-length data validated against these predictions.
- Git history starts from a single init commit; this repo was cleaned up and pinned after the fact rather than built commit by commit.
