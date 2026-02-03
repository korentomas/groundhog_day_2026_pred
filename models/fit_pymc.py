#!/usr/bin/env python3
"""
Fit a hierarchical Bayesian model for groundhog prediction reliability.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


def standardize(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return series * 0.0
    return (series - mean) / std


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit PyMC hierarchical reliability model")
    parser.add_argument("--data", default="data/analysis_dataset.parquet", help="Path to analysis dataset")
    parser.add_argument("--out-trace", default="data/model_trace.nc", help="Output trace NetCDF")
    parser.add_argument("--out-summary", default="data/posterior_summary.csv", help="Output posterior summary")
    parser.add_argument("--out-rankings", default="data/groundhog_rankings.csv", help="Output rankings")
    parser.add_argument("--draws", type=int, default=1000, help="Number of draws")
    parser.add_argument("--tune", type=int, default=1000, help="Number of tuning steps")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    df = df[df["correct"].notna()].copy()
    if df.empty:
        print("No rows with defined outcomes to model")
        return 1

    df["year_z"] = standardize(df["year"])
    df["lat_z"] = standardize(df["lat"].astype(float))
    df["lon_z"] = standardize(df["lon"].astype(float))

    groundhog_codes, groundhog_index = pd.factorize(df["slug"], sort=True)
    n_groundhogs = len(groundhog_index)

    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        sigma_g = pm.HalfNormal("sigma_g", 1.0)
        alpha_g = pm.Normal("alpha_g", mu=0.0, sigma=sigma_g, shape=n_groundhogs)

        beta_year = pm.Normal("beta_year", 0.0, 1.0)
        beta_lat = pm.Normal("beta_lat", 0.0, 1.0)
        beta_lon = pm.Normal("beta_lon", 0.0, 1.0)

        logit_p = (
            mu
            + alpha_g[groundhog_codes]
            + beta_year * df["year_z"].values
            + beta_lat * df["lat_z"].values
            + beta_lon * df["lon_z"].values
        )

        pm.Bernoulli("correct", logit_p=logit_p, observed=df["correct"].astype(int).values)
        trace = pm.sample(draws=args.draws, tune=args.tune, target_accept=0.9, chains=4)

    trace_path = Path(args.out_trace)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(trace, trace_path)

    summary = az.summary(trace, var_names=["mu", "sigma_g", "beta_year", "beta_lat", "beta_lon"], round_to=4)
    summary.to_csv(args.out_summary)

    # Compute posterior reliability for each groundhog at mean covariates (year_z=0, lat_z=0, lon_z=0)
    mu_samples = trace.posterior["mu"].values.reshape(-1)
    alpha_samples = trace.posterior["alpha_g"].values.reshape(-1, n_groundhogs)

    logits = mu_samples[:, None] + alpha_samples
    p_samples = 1 / (1 + np.exp(-logits))

    ranking_rows = []
    for i, slug in enumerate(groundhog_index):
        samples = p_samples[:, i]
        mean = float(np.mean(samples))
        hdi = az.hdi(samples, hdi_prob=0.95)
        ranking_rows.append({
            "slug": slug,
            "mean_reliability": mean,
            "hdi_2_5": float(hdi[0]),
            "hdi_97_5": float(hdi[1]),
            "n_obs": int((df["slug"] == slug).sum()),
        })

    rankings = pd.DataFrame(ranking_rows).sort_values("mean_reliability", ascending=False)
    rankings.to_csv(args.out_rankings, index=False)

    print(f"Wrote trace to {trace_path}")
    print(f"Wrote summary to {args.out_summary}")
    print(f"Wrote rankings to {args.out_rankings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
