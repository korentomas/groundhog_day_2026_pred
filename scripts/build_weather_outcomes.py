#!/usr/bin/env python3
"""
Build yearly weather outcome metrics from cached Daymet raw CSV files.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "daymet_raw"


def load_groundhogs(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data.get("groundhogs", [])


def parse_coords(coord_str: str) -> tuple[float, float] | None:
    if not coord_str:
        return None
    try:
        lat_str, lon_str = coord_str.split(",")
        return float(lat_str.strip()), float(lon_str.strip())
    except Exception:
        return None


def parse_daymet_csv(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("year,"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not locate Daymet header line")
    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(pd.io.common.StringIO(csv_text))
    return df


def add_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["year"].astype(int).astype(str), format="%Y") + pd.to_timedelta(df["yday"] - 1, unit="D")
    return df


def to_tavg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns={
        "tmax (deg c)": "tmax",
        "tmin (deg c)": "tmin",
    }, inplace=True)
    if "tmax" not in df.columns or "tmin" not in df.columns:
        raise ValueError("Expected tmax/tmin columns not found")
    df["tavg"] = (df["tmax"] + df["tmin"]) / 2.0
    return df


def compute_climatology(df: pd.DataFrame, baseline_start: int, baseline_end: int) -> pd.DataFrame:
    base = df[(df["year"] >= baseline_start) & (df["year"] <= baseline_end)].copy()
    base["doy"] = base["date"].dt.dayofyear
    clim = base.groupby("doy")["tavg"].mean().reset_index().rename(columns={"tavg": "tavg_clim"})
    return clim


def compute_window_metrics(df: pd.DataFrame, clim: pd.DataFrame, year: int, start_md: str, end_md: str,
                           min_days: int, alpha: float) -> dict:
    start_month, start_day = map(int, start_md.split("-"))
    end_month, end_day = map(int, end_md.split("-"))
    start_date = pd.Timestamp(year=year, month=start_month, day=start_day)
    end_date = pd.Timestamp(year=year, month=end_month, day=end_day)

    year_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    if year_df.empty:
        return {}

    year_df["doy"] = year_df["date"].dt.dayofyear
    year_df = year_df.merge(clim, on="doy", how="left")
    year_df["anom"] = year_df["tavg"] - year_df["tavg_clim"]

    anom = year_df["anom"].dropna()
    n_days = int(anom.shape[0])
    if n_days < min_days:
        return {
            "n_days": n_days,
            "mean_anom": np.nan,
            "p_value_two_sided": np.nan,
            "p_value_one_sided": np.nan,
            "outcome": "uncertain",
            "outcome_binary": np.nan,
        }

    t_stat, p_two = stats.ttest_1samp(anom.values, popmean=0.0, nan_policy="omit")
    mean_anom = float(np.nanmean(anom))

    if math.isnan(t_stat) or math.isnan(p_two):
        p_one = np.nan
    else:
        if t_stat >= 0:
            p_one = p_two / 2.0
        else:
            p_one = 1.0 - (p_two / 2.0)

    outcome = "uncertain"
    outcome_binary = np.nan
    if not math.isnan(p_one) and p_one <= alpha:
        if mean_anom > 0:
            outcome = "early_spring"
            outcome_binary = 1
        elif mean_anom < 0:
            outcome = "winter"
            outcome_binary = 0

    return {
        "n_days": n_days,
        "mean_anom": mean_anom,
        "p_value_two_sided": p_two,
        "p_value_one_sided": p_one,
        "outcome": outcome,
        "outcome_binary": outcome_binary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build weather outcome metrics from Daymet caches")
    parser.add_argument("--groundhogs", default="data/groundhogs.json", help="Path to groundhogs.json")
    parser.add_argument("--window-start", default="02-03", help="Window start (MM-DD)")
    parser.add_argument("--window-end", default="03-16", help="Window end (MM-DD)")
    parser.add_argument("--tolerance-days", type=int, default=3, help="Allow this many missing days in window")
    parser.add_argument("--baseline-start", type=int, default=1991, help="Climatology baseline start year")
    parser.add_argument("--baseline-end", type=int, default=2020, help="Climatology baseline end year")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for anomaly")
    parser.add_argument("--out", default="data/weather_outcomes.parquet", help="Output parquet path")
    parser.add_argument("--out-csv", default="data/weather_outcomes.csv", help="Output csv path")
    args = parser.parse_args()

    groundhogs = load_groundhogs(Path(args.groundhogs))
    gh_map = {gh.get("slug"): gh for gh in groundhogs}

    rows = []
    for raw_path in RAW_DIR.glob("*.csv"):
        slug = raw_path.stem
        gh = gh_map.get(slug, {})

        try:
            df = parse_daymet_csv(raw_path.read_text())
            df = add_dates(df)
            df = to_tavg(df)
        except Exception as exc:
            print(f"ERROR parsing {raw_path.name}: {exc}")
            continue

        clim = compute_climatology(df, args.baseline_start, args.baseline_end)
        if clim.empty:
            print(f"No climatology for {slug}; skipping")
            continue

        min_days = (pd.Timestamp(2001, int(args.window_end.split('-')[0]), int(args.window_end.split('-')[1])) -
                    pd.Timestamp(2001, int(args.window_start.split('-')[0]), int(args.window_start.split('-')[1]))).days + 1
        min_days = max(1, min_days - args.tolerance_days)

        for year in sorted(df["year"].unique()):
            metrics = compute_window_metrics(
                df,
                clim,
                int(year),
                args.window_start,
                args.window_end,
                min_days=min_days,
                alpha=args.alpha,
            )
            if not metrics:
                continue

            row = {
                "slug": slug,
                "groundhog_id": gh.get("id"),
                "shortname": gh.get("shortname"),
                "name": gh.get("name"),
                "country": gh.get("country"),
                "region": gh.get("region"),
                "city": gh.get("city"),
                "year": int(year),
                "window_start": args.window_start,
                "window_end": args.window_end,
                "baseline_start": args.baseline_start,
                "baseline_end": args.baseline_end,
                "alpha": args.alpha,
            }
            row.update(metrics)
            rows.append(row)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print("No output rows generated")
        return 1

    out_path = Path(args.out)
    out_csv = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_parquet(out_path, index=False)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path} and {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
