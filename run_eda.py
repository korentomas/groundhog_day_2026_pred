#!/usr/bin/env python3
"""
Run exploratory data analysis on Groundhog Day predictions data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

print("=" * 60)
print("GROUNDHOG DAY PREDICTIONS - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Load the combined data
with open('data/combined_data.json', 'r') as f:
    data = json.load(f)

groundhogs = data['groundhogs']
predictions_by_year = data['predictions_by_year']

print(f"\nData Overview:")
print(f"  Number of groundhogs: {len(groundhogs)}")
print(f"  Year range: {data['metadata']['year_range']['min']} - {data['metadata']['year_range']['max']}")

# Create a flat DataFrame of all predictions
all_predictions = []
for year, year_payload in predictions_by_year.items():
    year_preds = year_payload.get('predictions') if isinstance(year_payload, dict) else year_payload
    if not year_preds:
        continue
    for pred in year_preds:
        pred = dict(pred)
        pred['_raw_shadow_present'] = 'shadow' in pred
        pred['_raw_shadow_value'] = pred.get('shadow')
        pred['_raw_shadow_is_none'] = pred.get('shadow') is None
        pred['year'] = int(year)
        all_predictions.append(pred)

df = pd.DataFrame(all_predictions)
groundhog_df = pd.json_normalize(df['groundhog'])
df = pd.concat([df.drop('groundhog', axis=1), groundhog_df], axis=1)

# Create binary shadow indicator
df['shadow_binary'] = df['shadow'].apply(lambda x: 1 if x == 1 else (0 if x == 0 else np.nan))
valid_df = df[df['shadow_binary'].notna()]

# Validate NaN conversion for shadow values that were None in raw data
raw_none_count = df['_raw_shadow_is_none'].sum()
df_nan_count = df['shadow'].isna().sum()
mismatch_mask = df['_raw_shadow_is_none'] != df['shadow'].isna()
mismatch_count = mismatch_mask.sum()
print("\nNaN Validation (shadow field):")
print(f"  Raw None count: {int(raw_none_count)}")
print(f"  DataFrame NaN count: {int(df_nan_count)}")
print(f"  Mismatches: {int(mismatch_count)}")
if mismatch_count:
    print("  Sample mismatches:")
    print(df.loc[mismatch_mask, ['year', 'shadow', '_raw_shadow_is_none']].head(10).to_string(index=False))

# Inspect which rows have NaN shadow and whether the source explicitly had shadow
nan_df = df[df['shadow'].isna()].copy()
nan_with_value = nan_df[nan_df['_raw_shadow_present'] & nan_df['_raw_shadow_value'].notna()]
nan_missing_field = nan_df[~nan_df['_raw_shadow_present']]
print("\nNaN Shadow Rows:")
print(f"  Total NaN shadows: {len(nan_df)}")
print(f"  NaN but raw shadow had non-null value: {len(nan_with_value)}")
print(f"  NaN because raw shadow field missing: {len(nan_missing_field)}")
if len(nan_with_value):
    print("  Sample NaN rows with non-null raw shadow (unexpected):")
    cols = ['year', 'shadow', '_raw_shadow_present', '_raw_shadow_value', 'shortname', 'slug', 'details']
    print(nan_with_value[cols].head(10).to_string(index=False))
print("  Sample NaN rows:")
cols = ['year', 'shadow', '_raw_shadow_present', '_raw_shadow_value', 'shortname', 'slug', 'details']
print(nan_df[cols].head(10).to_string(index=False))

print(f"\nTotal predictions: {len(df)}")
print(f"Valid predictions (non-null shadow): {len(valid_df)}")

# Overall shadow statistics
overall_shadow_rate = valid_df['shadow_binary'].mean()
n_shadows = valid_df['shadow_binary'].sum()
n_total = len(valid_df)

print("\n" + "=" * 60)
print("OVERALL SHADOW STATISTICS")
print("=" * 60)
print(f"  Saw shadow: {int(n_shadows)} ({overall_shadow_rate*100:.1f}%)")
print(f"  No shadow: {int(n_total - n_shadows)} ({(1-overall_shadow_rate)*100:.1f}%)")

# Binomial test
from scipy.stats import binomtest
binom_result = binomtest(int(n_shadows), n_total, p=0.5, alternative='two-sided')
print(f"\nBinomial Test (H0: p=0.5):")
print(f"  p-value: {binom_result.pvalue:.6f}")
print(f"  95% CI: {binom_result.proportion_ci(confidence_level=0.95).low:.4f} - {binom_result.proportion_ci(confidence_level=0.95).high:.4f}")

# Yearly statistics
yearly_stats = valid_df.groupby('year').agg({
    'shadow_binary': ['sum', 'count', 'mean']
}).reset_index()
yearly_stats.columns = ['year', 'shadow_count', 'total_count', 'shadow_rate']

print("\n" + "=" * 60)
print("YEARLY STATISTICS")
print("=" * 60)
print(f"  Mean yearly shadow rate: {yearly_stats['shadow_rate'].mean():.4f}")
print(f"  Std dev: {yearly_stats['shadow_rate'].std():.4f}")
print(f"  Min: {yearly_stats['shadow_rate'].min():.4f} (Year {yearly_stats.loc[yearly_stats['shadow_rate'].idxmin(), 'year']})")
print(f"  Max: {yearly_stats['shadow_rate'].max():.4f} (Year {yearly_stats.loc[yearly_stats['shadow_rate'].idxmax(), 'year']})")

# Decade analysis
valid_df['decade'] = (valid_df['year'] // 10) * 10
print("\n" + "=" * 60)
print("DECADE-WISE SHADOW RATES")
print("=" * 60)
decade_summary = valid_df.groupby('decade')['shadow_binary'].agg(['sum', 'count', 'mean'])
decade_summary.columns = ['Shadows', 'Total', 'Rate']
print(decade_summary.to_string())

# Groundhog analysis
print("\n" + "=" * 60)
print("GROUNDHOG ANALYSIS")
print("=" * 60)
print("\nTop 10 Most Active Groundhogs:")
groundhog_stats = valid_df.groupby(['slug', 'shortname', 'type', 'country']).agg({
    'shadow_binary': ['count', 'sum', 'mean']
}).reset_index()
groundhog_stats.columns = ['slug', 'shortname', 'type', 'country', 'predictions', 'shadows', 'shadow_rate']
groundhog_stats = groundhog_stats.sort_values('predictions', ascending=False)
print(groundhog_stats.head(10)[['shortname', 'type', 'country', 'predictions', 'shadows', 'shadow_rate']].to_string(index=False))

# Create visualizations
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Shadow rate over time
ax1 = axes[0, 0]
yearly_stats['rolling_rate'] = yearly_stats['shadow_rate'].rolling(window=10, min_periods=1).mean()
ax1.plot(yearly_stats['year'], yearly_stats['shadow_rate'], 'o-', alpha=0.5, label='Yearly Rate')
ax1.plot(yearly_stats['year'], yearly_stats['rolling_rate'], '-', linewidth=2, label='10-Year Rolling Avg')
ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Baseline')
ax1.set_xlabel('Year')
ax1.set_ylabel('Shadow Rate')
ax1.set_title('Groundhog Shadow Prediction Rate Over Time')
ax1.legend()
ax1.set_ylim(0, 1)

# Plot 2: Distribution of yearly shadow rates
ax2 = axes[0, 1]
ax2.hist(yearly_stats['shadow_rate'], bins=20, edgecolor='black', alpha=0.7)
ax2.axvline(x=overall_shadow_rate, color='red', linestyle='--', linewidth=2, label=f'Overall Mean: {overall_shadow_rate:.3f}')
ax2.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='50% Baseline')
ax2.set_xlabel('Shadow Rate')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Yearly Shadow Rates')
ax2.legend()

# Plot 3: Cumulative shadow predictions
ax3 = axes[1, 0]
valid_df_sorted = valid_df.sort_values('year')
cumulative_shadows = valid_df_sorted['shadow_binary'].cumsum()
ax3.plot(valid_df_sorted['year'], cumulative_shadows, '-', label='Cumulative Shadows')
ax3.plot(valid_df_sorted['year'], np.arange(1, len(cumulative_shadows)+1)*0.5, '--', color='green', label='50% Expected')
ax3.set_xlabel('Year')
ax3.set_ylabel('Cumulative Shadow Predictions')
ax3.set_title('Cumulative Shadow Predictions Over Time')
ax3.legend()

# Plot 4: Bar chart of decades
ax4 = axes[1, 1]
decade_stats = valid_df.groupby('decade')['shadow_binary'].mean()
decade_stats.plot(kind='bar', ax=ax4, edgecolor='black')
ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Baseline')
ax4.set_xlabel('Decade')
ax4.set_ylabel('Shadow Rate')
ax4.set_title('Shadow Rate by Decade')
ax4.legend()
ax4.set_ylim(0, 1)
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('data/shadow_analysis.png', dpi=150, bbox_inches='tight')
print("  Saved: data/shadow_analysis.png")

# Groundhog type analysis
fig2, ax = plt.subplots(figsize=(12, 6))
type_stats = valid_df.groupby('type')['shadow_binary'].agg(['count', 'mean', 'std']).reset_index()
type_stats.columns = ['type', 'count', 'mean_shadow_rate', 'std']
type_stats = type_stats.sort_values('mean_shadow_rate')
colors = plt.cm.RdYlGn(type_stats['mean_shadow_rate'])
bars = ax.barh(type_stats['type'], type_stats['mean_shadow_rate'], color=colors, edgecolor='black')
ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='50% Baseline')
ax.set_xlabel('Shadow Rate')
ax.set_ylabel('Groundhog Type')
ax.set_title('Shadow Rate by Groundhog Type')
ax.legend()
ax.set_xlim(0, 1)
for i, (rate, count) in enumerate(zip(type_stats['mean_shadow_rate'], type_stats['count'])):
    ax.text(rate + 0.02, i, f'n={count}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('data/groundhog_type_analysis.png', dpi=150, bbox_inches='tight')
print("  Saved: data/groundhog_type_analysis.png")

# Save Bayesian input data
bayesian_data = {
    'overall': {
        'n_shadows': int(n_shadows),
        'n_total': n_total,
        'proportion': n_shadows/n_total,
        'binom_pvalue': binom_result.pvalue,
        'ci_95_low': binom_result.proportion_ci(confidence_level=0.95).low,
        'ci_95_high': binom_result.proportion_ci(confidence_level=0.95).high
    },
    'yearly_data': yearly_stats[['year', 'shadow_rate', 'shadow_count', 'total_count']].to_dict('records'),
    'decade_data': valid_df.groupby('decade')['shadow_binary'].agg(['sum', 'count', 'mean']).reset_index().to_dict('records'),
    'groundhog_data': groundhog_stats.to_dict('records')
}

with open('data/bayesian_input.json', 'w') as f:
    json.dump(bayesian_data, f, indent=2)
print("  Saved: data/bayesian_input.json")

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)
