#!/usr/bin/env python3
"""
Join groundhog predictions with weather outcomes to build an analysis dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_predictions(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    rows = []
    for year, payload in data.items():
        preds = payload.get("predictions") if isinstance(payload, dict) else payload
        if not preds:
            continue
        for pred in preds:
            shadow = pred.get("shadow")
            gh = pred.get("groundhog", {})
            row = {
                "year": int(pred.get("year") or year),
                "shadow": shadow,
                "details": pred.get("details"),
                "groundhog_id": gh.get("id"),
                "slug": gh.get("slug"),
                "shortname": gh.get("shortname"),
                "name": gh.get("name"),
                "country": gh.get("country"),
                "region": gh.get("region"),
                "city": gh.get("city"),
                "coordinates": gh.get("coordinates"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def parse_coords(coord_str: str) -> tuple[float, float] | tuple[float, float | None]:
    if not coord_str:
        return (np.nan, np.nan)
    try:
        lat_str, lon_str = coord_str.split(",")
        return float(lat_str.strip()), float(lon_str.strip())
    except Exception:
        return (np.nan, np.nan)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build analysis dataset from predictions + weather outcomes")
    parser.add_argument("--predictions", default="data/predictions.json", help="Path to predictions.json")
    parser.add_argument("--weather", default="data/weather_outcomes.parquet", help="Path to weather outcomes parquet")
    parser.add_argument("--out", default="data/analysis_dataset.parquet", help="Output parquet")
    parser.add_argument("--out-csv", default="data/analysis_dataset.csv", help="Output csv")
    args = parser.parse_args()

    preds = load_predictions(Path(args.predictions))
    if preds.empty:
        print("No predictions found")
        return 1

    preds["shadow_binary"] = preds["shadow"].apply(lambda x: 1 if x == 1 else (0 if x == 0 else np.nan))
    preds = preds[preds["shadow_binary"].notna()].copy()

    coords = preds["coordinates"].apply(parse_coords)
    preds[["lat", "lon"]] = pd.DataFrame(coords.tolist(), index=preds.index)

    weather = pd.read_parquet(args.weather)
    merged = preds.merge(weather, on=["slug", "year"], how="left", suffixes=("", "_wx"))

    merged["prediction_early_spring"] = 1 - merged["shadow_binary"]
    merged["correct"] = np.where(
        merged["outcome_binary"].notna(),
        (merged["prediction_early_spring"] == merged["outcome_binary"]).astype(int),
        np.nan,
    )

    out_path = Path(args.out)
    out_csv = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged.to_parquet(out_path, index=False)
    merged.to_csv(out_csv, index=False)
    print(f"Wrote {len(merged)} rows to {out_path} and {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
