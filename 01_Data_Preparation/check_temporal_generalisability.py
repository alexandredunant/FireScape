"""
Temporal Generalisability Analysis for FireScape Dataset

This script analyzes the distribution of fire observations across:
- Years
- Months
- Seasons
- Day of year

Helps assess whether the dataset is balanced across temporal dimensions
for model generalisability.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# ===================================================================
# CONFIGURATION
# ===================================================================

OUTPUT_DIR = "/mnt/CEPH_PROJECTS/Firescape/output/01_Training_Data/"
NETCDF_PATH = os.path.join(OUTPUT_DIR, "spacetime_stacks.nc")
OUTPUT_PLOT_PATH = os.path.join(OUTPUT_DIR, "temporal_generalisability_analysis.png")
OUTPUT_STATS_PATH = os.path.join(OUTPUT_DIR, "temporal_generalisability_stats.csv")

# Define seasons (Northern Hemisphere)
SEASONS = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Fall': [9, 10, 11]
}

# ===================================================================
# LOAD DATA
# ===================================================================

print(f"Loading dataset from: {NETCDF_PATH}")
ds = xr.open_dataset(NETCDF_PATH)

print(f"\nDataset dimensions: {dict(ds.dims)}")
print(f"Number of observations: {ds.dims['id_obs']}")

# Extract temporal and label information
event_dates = pd.to_datetime(ds.event_date.values)
labels = ds.label.values

print(f"\nDate range: {event_dates.min()} to {event_dates.max()}")
print(f"Fire events (label=1): {(labels == 1).sum()}")
print(f"Non-fire events (label=0): {(labels == 0).sum()}")

# ===================================================================
# CREATE TEMPORAL FEATURES
# ===================================================================

# Create a DataFrame for easier analysis
df = pd.DataFrame({
    'date': event_dates,
    'label': labels,
    'year': event_dates.year,
    'month': event_dates.month,
    'day_of_year': event_dates.dayofyear,
    'month_name': event_dates.strftime('%B')
})

# Add season
def get_season(month):
    for season, months in SEASONS.items():
        if month in months:
            return season
    return 'Unknown'

df['season'] = df['month'].apply(get_season)

# Add label name
df['label_name'] = df['label'].map({0: 'No Fire', 1: 'Fire'})

# ===================================================================
# COMPUTE STATISTICS
# ===================================================================

print("\n" + "="*70)
print("TEMPORAL DISTRIBUTION STATISTICS")
print("="*70)

# Overall statistics
print("\n--- Year Distribution ---")
year_dist = df['year'].value_counts().sort_index()
print(year_dist)

print("\n--- Month Distribution ---")
month_dist = df.groupby('month_name')['label'].count().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])
print(month_dist)

print("\n--- Season Distribution ---")
season_dist = df['season'].value_counts().reindex(['Winter', 'Spring', 'Summer', 'Fall'])
print(season_dist)

# Statistics by label
print("\n--- Fire Events by Year ---")
fire_by_year = df[df['label'] == 1].groupby('year').size()
print(fire_by_year)

print("\n--- Fire Events by Season ---")
fire_by_season = df[df['label'] == 1].groupby('season').size().reindex(['Winter', 'Spring', 'Summer', 'Fall'])
print(fire_by_season)

print("\n--- Fire Events by Month ---")
fire_by_month = df[df['label'] == 1].groupby('month_name').size().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])
print(fire_by_month)

# ===================================================================
# STATISTICAL TESTS FOR BALANCE
# ===================================================================

print("\n" + "="*70)
print("BALANCE ANALYSIS")
print("="*70)

# Coefficient of Variation (lower = more balanced)
def coefficient_of_variation(series):
    return series.std() / series.mean() if series.mean() > 0 else np.nan

print("\nCoefficient of Variation (CV) - Lower values indicate better balance:")
print(f"  Years:   CV = {coefficient_of_variation(year_dist):.3f}")
print(f"  Months:  CV = {coefficient_of_variation(month_dist):.3f}")
print(f"  Seasons: CV = {coefficient_of_variation(season_dist):.3f}")

# Fire rate by temporal period
print("\n--- Fire Rate by Year ---")
fire_rate_year = df.groupby('year')['label'].mean()
print(fire_rate_year)
print(f"  Mean fire rate: {fire_rate_year.mean():.3f}")
print(f"  Std fire rate:  {fire_rate_year.std():.3f}")

print("\n--- Fire Rate by Season ---")
fire_rate_season = df.groupby('season')['label'].mean().reindex(['Winter', 'Spring', 'Summer', 'Fall'])
print(fire_rate_season)

print("\n--- Fire Rate by Month ---")
fire_rate_month = df.groupby('month')['label'].mean()
print(fire_rate_month)

# ===================================================================
# VISUALIZATIONS
# ===================================================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig = plt.figure(figsize=(20, 12))

# ---- Row 1: Overall Distribution ----

# 1. Distribution by Year
ax1 = plt.subplot(3, 3, 1)
year_counts = df.groupby(['year', 'label_name']).size().unstack(fill_value=0)
year_counts.plot(kind='bar', stacked=True, ax=ax1, color=['#2ecc71', '#e74c3c'], alpha=0.8)
ax1.set_title('Distribution by Year', fontsize=14, fontweight='bold')
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Observations')
ax1.legend(title='Label')
ax1.grid(axis='y', alpha=0.3)

# 2. Distribution by Month
ax2 = plt.subplot(3, 3, 2)
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df['month_abbr'] = df['date'].dt.strftime('%b')
month_counts = df.groupby(['month_abbr', 'label_name']).size().unstack(fill_value=0)
month_counts = month_counts.reindex(month_order)
month_counts.plot(kind='bar', stacked=True, ax=ax2, color=['#2ecc71', '#e74c3c'], alpha=0.8)
ax2.set_title('Distribution by Month', fontsize=14, fontweight='bold')
ax2.set_xlabel('Month')
ax2.set_ylabel('Number of Observations')
ax2.legend(title='Label')
ax2.grid(axis='y', alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

# 3. Distribution by Season
ax3 = plt.subplot(3, 3, 3)
season_order = ['Winter', 'Spring', 'Summer', 'Fall']
season_counts = df.groupby(['season', 'label_name']).size().unstack(fill_value=0)
season_counts = season_counts.reindex(season_order)
season_counts.plot(kind='bar', stacked=True, ax=ax3, color=['#2ecc71', '#e74c3c'], alpha=0.8)
ax3.set_title('Distribution by Season', fontsize=14, fontweight='bold')
ax3.set_xlabel('Season')
ax3.set_ylabel('Number of Observations')
ax3.legend(title='Label')
ax3.grid(axis='y', alpha=0.3)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

# ---- Row 2: Fire Rate Analysis ----

# 4. Fire Rate by Year
ax4 = plt.subplot(3, 3, 4)
fire_rate_year.plot(kind='bar', ax=ax4, color='#e67e22', alpha=0.8)
ax4.axhline(y=fire_rate_year.mean(), color='red', linestyle='--',
            label=f'Mean: {fire_rate_year.mean():.3f}')
ax4.set_title('Fire Rate by Year', fontsize=14, fontweight='bold')
ax4.set_xlabel('Year')
ax4.set_ylabel('Fire Rate (proportion)')
ax4.set_ylim(0, 1)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# 5. Fire Rate by Month
ax5 = plt.subplot(3, 3, 5)
fire_rate_month.plot(kind='bar', ax=ax5, color='#e67e22', alpha=0.8)
ax5.axhline(y=fire_rate_month.mean(), color='red', linestyle='--',
            label=f'Mean: {fire_rate_month.mean():.3f}')
ax5.set_title('Fire Rate by Month', fontsize=14, fontweight='bold')
ax5.set_xlabel('Month')
ax5.set_ylabel('Fire Rate (proportion)')
ax5.set_ylim(0, 1)
ax5.set_xticklabels(month_order)
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# 6. Fire Rate by Season
ax6 = plt.subplot(3, 3, 6)
fire_rate_season.plot(kind='bar', ax=ax6, color='#e67e22', alpha=0.8)
ax6.axhline(y=fire_rate_season.mean(), color='red', linestyle='--',
            label=f'Mean: {fire_rate_season.mean():.3f}')
ax6.set_title('Fire Rate by Season', fontsize=14, fontweight='bold')
ax6.set_xlabel('Season')
ax6.set_ylabel('Fire Rate (proportion)')
ax6.set_ylim(0, 1)
ax6.legend()
ax6.grid(axis='y', alpha=0.3)
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

# ---- Row 3: Temporal Patterns ----

# 7. Day of Year Distribution (Fire events only)
ax7 = plt.subplot(3, 3, 7)
fire_df = df[df['label'] == 1]
ax7.hist(fire_df['day_of_year'], bins=52, color='#e74c3c', alpha=0.7, edgecolor='black')
ax7.set_title('Fire Events by Day of Year', fontsize=14, fontweight='bold')
ax7.set_xlabel('Day of Year')
ax7.set_ylabel('Number of Fire Events')
ax7.grid(axis='y', alpha=0.3)

# 8. Timeline of all observations
ax8 = plt.subplot(3, 3, 8)
fire_dates = df[df['label'] == 1]['date']
nofire_dates = df[df['label'] == 0]['date']
ax8.scatter(nofire_dates, [0]*len(nofire_dates), alpha=0.4, s=10,
           color='#2ecc71', label='No Fire')
ax8.scatter(fire_dates, [1]*len(fire_dates), alpha=0.4, s=10,
           color='#e74c3c', label='Fire')
ax8.set_title('Timeline of All Observations', fontsize=14, fontweight='bold')
ax8.set_xlabel('Date')
ax8.set_yticks([0, 1])
ax8.set_yticklabels(['No Fire', 'Fire'])
ax8.legend()
ax8.grid(axis='x', alpha=0.3)
plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45)

# 9. Observations per Year-Month heatmap
ax9 = plt.subplot(3, 3, 9)
df['year_month'] = df['date'].dt.to_period('M')
heatmap_data = df.pivot_table(index='month', columns='year',
                               values='label', aggfunc='count', fill_value=0)
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd',
            ax=ax9, cbar_kws={'label': 'Count'})
ax9.set_title('Observations Heatmap (Year × Month)', fontsize=14, fontweight='bold')
ax9.set_xlabel('Year')
ax9.set_ylabel('Month')
ax9.set_yticklabels(month_order, rotation=0)

plt.tight_layout()
print(f"Saving plot to: {OUTPUT_PLOT_PATH}")
plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches='tight')
plt.show()

# ===================================================================
# SAVE STATISTICS TO CSV
# ===================================================================

print(f"\nSaving statistics to: {OUTPUT_STATS_PATH}")

stats_summary = pd.DataFrame({
    'Metric': [
        'Total Observations',
        'Fire Events',
        'Non-Fire Events',
        'Date Range Start',
        'Date Range End',
        'Number of Years',
        'Number of Months Covered',
        'CV - Years',
        'CV - Months',
        'CV - Seasons',
        'Mean Fire Rate',
        'Std Fire Rate (by year)'
    ],
    'Value': [
        len(df),
        (df['label'] == 1).sum(),
        (df['label'] == 0).sum(),
        df['date'].min().strftime('%Y-%m-%d'),
        df['date'].max().strftime('%Y-%m-%d'),
        df['year'].nunique(),
        df['year_month'].nunique(),
        f"{coefficient_of_variation(year_dist):.3f}",
        f"{coefficient_of_variation(month_dist):.3f}",
        f"{coefficient_of_variation(season_dist):.3f}",
        f"{fire_rate_year.mean():.3f}",
        f"{fire_rate_year.std():.3f}"
    ]
})

stats_summary.to_csv(OUTPUT_STATS_PATH, index=False)

# ===================================================================
# GENERALISABILITY ASSESSMENT
# ===================================================================

print("\n" + "="*70)
print("GENERALISABILITY ASSESSMENT")
print("="*70)

# Check for temporal coverage
years_covered = df['year'].nunique()
months_covered = df['year_month'].nunique()
total_possible_months = (df['date'].max().year - df['date'].min().year + 1) * 12

print(f"\nTemporal Coverage:")
print(f"  Years covered: {years_covered}")
print(f"  Months with data: {months_covered} / {total_possible_months} possible months")
print(f"  Coverage rate: {months_covered / total_possible_months * 100:.1f}%")

# Check for seasonal balance
season_cv = coefficient_of_variation(season_dist)
print(f"\nSeasonal Balance:")
print(f"  CV = {season_cv:.3f}")
if season_cv < 0.3:
    print("  ✓ Well balanced across seasons")
elif season_cv < 0.5:
    print("  ⚠ Moderate imbalance across seasons")
else:
    print("  ✗ Significant imbalance across seasons")

# Check for yearly balance
year_cv = coefficient_of_variation(year_dist)
print(f"\nYearly Balance:")
print(f"  CV = {year_cv:.3f}")
if year_cv < 0.3:
    print("  ✓ Well balanced across years")
elif year_cv < 0.5:
    print("  ⚠ Moderate imbalance across years")
else:
    print("  ✗ Significant imbalance across years")

# Check fire rate consistency
fire_rate_std = fire_rate_year.std()
print(f"\nFire Rate Consistency:")
print(f"  Std deviation = {fire_rate_std:.3f}")
if fire_rate_std < 0.1:
    print("  ✓ Consistent fire rate across years")
elif fire_rate_std < 0.2:
    print("  ⚠ Moderate variation in fire rate")
else:
    print("  ✗ High variation in fire rate across years")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

# Close dataset
ds.close()
