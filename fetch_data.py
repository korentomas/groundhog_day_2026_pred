#!/usr/bin/env python3
"""
Fetch all data from the Groundhog Day API for Bayesian analysis in Julia.
"""
import requests
import json
import os
from datetime import datetime

BASE_URL = "https://groundhog-day.com/api/v1"
DATA_DIR = "data"

def fetch_all_groundhogs():
    """Fetch all groundhogs with their predictions."""
    print("Fetching all groundhogs...")
    response = requests.get(f"{BASE_URL}/groundhogs")
    response.raise_for_status()
    return response.json()

def fetch_predictions_for_year(year):
    """Fetch predictions for a specific year."""
    response = requests.get(f"{BASE_URL}/predictions", params={"year": year})
    if response.status_code == 302:
        # Follow redirect
        response = requests.get(response.headers["Location"])
    response.raise_for_status()
    return response.json()

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Fetch all groundhogs
    groundhogs_data = fetch_all_groundhogs()
    with open(f"{DATA_DIR}/groundhogs.json", "w") as f:
        json.dump(groundhogs_data, f, indent=2)
    print(f"Saved groundhogs data ({len(groundhogs_data.get('groundhogs', []))} groundhogs)")
    
    # Fetch predictions for all years (1886-2022)
    all_predictions = {}
    for year in range(1886, 2023):
        try:
            predictions = fetch_predictions_for_year(year)
            all_predictions[str(year)] = predictions
            print(f"  Year {year}: {len(predictions)} predictions")
        except Exception as e:
            print(f"  Year {year}: Error - {e}")
    
    with open(f"{DATA_DIR}/predictions.json", "w") as f:
        json.dump(all_predictions, f, indent=2)
    print(f"Saved all predictions data")
    
    # Create a combined dataset for easier Bayesian analysis
    combined = {
        "metadata": {
            "source": "https://groundhog-day.com/api/v1/",
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "year_range": {"min": 1886, "max": 2022}
        },
        "groundhogs": groundhogs_data.get("groundhogs", []),
        "predictions_by_year": all_predictions
    }
    
    with open(f"{DATA_DIR}/combined_data.json", "w") as f:
        json.dump(combined, f, indent=2)
    print("Saved combined dataset")
    
    print("\nData fetch complete! Files saved to data/:")
    print("  - groundhogs.json: All groundhogs with their metadata")
    print("  - predictions.json: All predictions organized by year")
    print("  - combined_data.json: Combined dataset for analysis")

if __name__ == "__main__":
    main()
