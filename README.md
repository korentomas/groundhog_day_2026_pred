# Groundhog Day 2026 Prediction

> "When Chekhov saw the long winter, he saw a winter bleak and dark and bereft of hope. Yet we know that winter is just another step in the cycle of life." — *Groundhog Day* (1993)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/korentomas/groundhog_day_2026_pred/blob/main/notebooks/02_bayesian_hierarchical.ipynb)

A side project applying real Bayesian methods to the least serious dataset available: over a century of groundhogs guessing at the weather.

Every February, dozens of groundhogs across the US and Canada "predict" whether winter will run long (saw shadow) or end early (no shadow). The [groundhog-day.com API](https://groundhog-day.com/api/v1) has recorded these predictions back to 1886. This project treats that record as a genuine binomial time series: is the shadow rate actually 50/50, does it drift over time or by region, and which groundhogs are consistently more "shadow-prone" than others, once you account for the fact that some of them have one data point and Punxsutawney Phil has 127.

## Viewing this

Rendered site with both notebooks fully run: **[korentomas.github.io/groundhog_day_2026_pred](https://korentomas.github.io/groundhog_day_2026_pred)**. Rebuilt on every push to `main` via GitHub Actions (see `.github/workflows/publish.yml`): fetches fresh data, re-executes both notebooks, builds the Quarto site.

Or, since notebooks are committed without outputs, browse them raw:

- [`notebooks/01_shadow_eda.ipynb`](notebooks/01_shadow_eda.ipynb): frequentist EDA.
- [`notebooks/02_bayesian_hierarchical.ipynb`](notebooks/02_bayesian_hierarchical.ipynb): the actual Bayesian analysis.

Or open the Bayesian notebook directly in Colab with the badge above if you want to rerun or tweak the models yourself.

## Data source

All data comes from the public [groundhog-day.com API](https://groundhog-day.com/api/v1): groundhog metadata (name, location, type) and yearly shadow predictions from 1886 onward. 93 groundhogs are in the registry, and only 38 of them are real, living groundhogs. The rest are taxidermied, animatronic, or otherwise not a groundhog (there's an alligator and a lobster in there).

## Reproducing

```
pip install -r requirements.txt
python fetch_data.py   # pulls groundhogs.json, predictions.json, combined_data.json
python run_eda.py      # runs the shadow-rate EDA, prints stats, saves plots + bayesian_input.json
ln -s ../data notebooks/data   # nbconvert executes notebooks with their own dir as cwd
jupyter nbconvert --to notebook --execute --inplace notebooks/01_shadow_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_bayesian_hierarchical.ipynb
```

`data/` is gitignored (`*.json`, `*.png`, `*.geojson`). Everything in it, including the Natural Earth admin-1 boundaries geojson used for the map in the Bayesian notebook, is fetched or generated on demand. Nothing in this repo depends on committed data files, and every plot that matters is generated inline in the notebooks rather than linked from a saved file. Notebooks are committed with outputs stripped to keep diffs readable; run them (or use the published site above) to see plots.

## What's here

- `fetch_data.py`: pulls raw groundhog and prediction data from the API, writes `data/groundhogs.json`, `data/predictions.json`, `data/combined_data.json`.
- `run_eda.py`: script version of the core EDA. Loads `combined_data.json`, computes overall and yearly shadow rates, runs a binomial test against p=0.5, breaks results down by decade and groundhog type, saves `data/shadow_analysis.png`, `data/groundhog_type_analysis.png`, and `data/bayesian_input.json` (summary stats formatted for the downstream Bayesian model).
- `notebooks/01_shadow_eda.ipynb`: the notebook version of the above. Load, clean, and explore the shadow data: overall and yearly rates, binomial and chi-square tests, groundhog-type breakdown, and prep for the Bayesian stage.
- `notebooks/02_bayesian_hierarchical.ipynb`: the real analysis. Two hierarchical PyMC models (per-groundhog partial pooling, and a crossed groundhog x decade model to check whether the apparent post-1980 trend survives once you control for who was reporting), a posterior-based ability ranking, a geographic map of the groundhog registry, plus three follow-up models: fake vs. real groundhogs, agreement rate with Punxsutawney Phil, and geographic clustering (both a linear pass and a full Gaussian process). Five PyMC models total, all converge cleanly: r-hat ~1.0, zero divergences. Maps and geo scatterplots use a hand-drawn vector groundhog marker (no groundhog emoji exists in Unicode, and matplotlib doesn't reliably render color emoji fonts).

## Future work

Fake vs. real groundhog comparisons, Phil-agreement rates, and geographic
clustering (linear pass and a full Gaussian process) are done — see sections
4-6 of `notebooks/02_bayesian_hierarchical.ipynb`. None of the three turned
up a credible effect: real vs. fake groundhogs, Phil-agreement vs.
independence baseline, and lat/lon vs. shadow rate all have 94% HDIs
straddling zero, and the GP's amplitude posterior sits on zero too. Still
open: per-groundhog streakiness and career-length effects, see the
Future Work section at the end of that notebook. Checking predictions
against actual NOAA spring-onset data was investigated and dropped: the
API's own temperature field only has usable values for 13 of 1761
predictions, too sparse to build anything on.

## Notes

- "Ability" ranking in the Bayesian notebook is based on the hierarchical posterior mean, i.e. how often a groundhog says shadow relative to the pooled population, not a measure of forecasting accuracy against real winter outcomes. There's no ground-truth winter-length data validated against these predictions.
- Git history starts from a single init commit; this repo was cleaned up and pinned after the fact rather than built commit by commit.
