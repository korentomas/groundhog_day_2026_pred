#!/usr/bin/env python3
"""
Fetch Daymet daily data for each groundhog location and cache raw CSV responses.

This uses the Daymet single-pixel API (free, no key required).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import requests

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "daymet_raw"
API_URL = "https://daymet.ornl.gov/single-pixel/api/data"
DEFAULT_VARS = "tmin,tmax"


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


def fetch_single_pixel(lat: float, lon: float, start: str, end: str, vars_csv: str) -> str:
    params = {
        "lat": lat,
        "lon": lon,
        "vars": vars_csv,
        "start": start,
        "end": end,
    }
    resp = requests.get(API_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.text


def iter_groundhogs(groundhogs: Iterable[dict], only_active: bool) -> Iterable[dict]:
    for gh in groundhogs:
        if only_active and not gh.get("active", 0):
            continue
        yield gh


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Daymet single-pixel data for groundhog locations")
    parser.add_argument("--groundhogs", default="data/groundhogs.json", help="Path to groundhogs.json")
    parser.add_argument("--start", default="1980-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--vars", default=DEFAULT_VARS, help="Comma-separated Daymet variables")
    parser.add_argument("--only-active", action="store_true", help="Only fetch for active groundhogs")
    parser.add_argument("--refresh", action="store_true", help="Re-download even if cached")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of groundhogs (for testing)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    groundhogs = load_groundhogs(Path(args.groundhogs))
    count = 0
    for gh in iter_groundhogs(groundhogs, args.only_active):
        coords = parse_coords(gh.get("coordinates", ""))
        if not coords:
            print(f"Skipping {gh.get('slug')} (missing/invalid coordinates)")
            continue

        slug = gh.get("slug") or f"gh_{gh.get('id')}"
        out_path = RAW_DIR / f"{slug}.csv"

        if out_path.exists() and not args.refresh:
            print(f"Cached: {slug}")
        else:
            lat, lon = coords
            try:
                text = fetch_single_pixel(lat, lon, args.start, args.end, args.vars)
            except Exception as exc:
                print(f"ERROR: {slug} ({lat}, {lon}) -> {exc}")
                continue
            out_path.write_text(text)
            print(f"Fetched: {slug}")

        count += 1
        if args.limit and count >= args.limit:
            break

    print(f"Done. Processed {count} groundhogs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
