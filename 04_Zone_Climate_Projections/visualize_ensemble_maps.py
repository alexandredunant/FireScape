#!/usr/bin/env python
"""
Visualize Ensemble Member Fire Risk Projections as Maps

Creates spatial maps showing fire risk projections across fire brigade zones
for different scenarios, time periods, and seasons.

Features:
- Maps for each scenario (RCP4.5, RCP8.5)
- Ensemble mean and ensemble spread (std) maps
- Multiple time periods and seasons
- Comparison maps showing change from baseline
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
from pathlib import Path
import scienceplots

plt.style.use(['science', 'no-latex'])
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

print("="*80)
print("ENSEMBLE FIRE RISK PROJECTION MAPS")
print("="*80)
print()

# ===================================================================
# CONFIGURATION
# ===================================================================

BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape")
FIRE_BRIGADE_ZONES = BASE_DIR / "Data/06_Administrative_Boundaries/Processed/FireBrigade_ResponsibilityAreas_Bolzano_clipped.gpkg"

# Input data directories
RCP45_DIR = BASE_DIR / "output/04_Zone_Climate_Projections/rcp45_ensemble_members"
RCP85_DIR = BASE_DIR / "output/04_Zone_Climate_Projections/rcp85_ensemble_members"

# Output directory
OUTPUT_DIR = BASE_DIR / "output/04_Zone_Climate_Projections/ensemble_maps"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Visualization parameters
SCENARIOS = {
    'rcp45': {'dir': RCP45_DIR, 'label': 'RCP4.5'},
    'rcp85': {'dir': RCP85_DIR, 'label': 'RCP8.5'}
}

YEARS_TO_PLOT = [2020, 2050, 2080, 2100]  # Baseline, mid-century, late-century, end-century
MONTHS_TO_PLOT = {
    3: 'March',
    8: 'August'
}

print("Configuration:")
print(f"  Scenarios: {', '.join([v['label'] for v in SCENARIOS.values()])}")
print(f"  Years: {YEARS_TO_PLOT}")
print(f"  Months: {', '.join(MONTHS_TO_PLOT.values())}")
print(f"  Output: {OUTPUT_DIR}")
print()

# ===================================================================
# LOAD DATA
# ===================================================================

print("Loading fire brigade zones...")
gdf_zones = gpd.read_file(FIRE_BRIGADE_ZONES)
print(f"✓ Loaded {len(gdf_zones)} zones")
print()

# Load ensemble results for both scenarios
ensemble_data = {}

for scenario_key, scenario_info in SCENARIOS.items():
    csv_path = scenario_info['dir'] / "zone_projections_ensemble_members.csv"

    if csv_path.exists():
        print(f"Loading {scenario_info['label']} data...")
        df = pd.read_csv(csv_path)
        ensemble_data[scenario_key] = df
        print(f"  ✓ {len(df)} records")
        print(f"  Ensemble members: {df['ensemble_member'].nunique()}")
        print(f"  Years: {sorted(df['year'].unique())}")
        print(f"  Months: {sorted(df['month'].unique())}")
    else:
        print(f"⚠️  Warning: {csv_path} not found, skipping {scenario_info['label']}")

print()

if not ensemble_data:
    print("ERROR: No data files found!")
    exit(1)

# ===================================================================
# FUNCTION: CREATE MAP
# ===================================================================

def create_risk_map(gdf_zones, risk_data, title, output_path, vmin=None, vmax=None,
                    cmap='YlOrRd', show_change=False):
    """Create a choropleth map of fire risk by zone."""

    # Merge risk data with zones
    gdf_plot = gdf_zones.merge(risk_data, on='zone_id', how='left')

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Determine color scale
    if vmin is None:
        vmin = gdf_plot['risk_value'].min()
    if vmax is None:
        vmax = gdf_plot['risk_value'].max()

    # Use diverging colormap for change maps
    if show_change:
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        cmap = 'RdBu_r'

    # Plot
    gdf_plot.plot(column='risk_value', ax=ax, cmap=cmap,
                  vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.3,
                  legend=True, legend_kwds={'label': 'Relative Fire Risk', 'shrink': 0.7})

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  ✓ Saved: {output_path.name}")

# ===================================================================
# GENERATE MAPS
# ===================================================================

print("="*80)
print("GENERATING MAPS")
print("="*80)
print()

for scenario_key, df in ensemble_data.items():
    scenario_label = SCENARIOS[scenario_key]['label']

    print(f"\n{scenario_label}:")
    print("-" * 60)

    for year in YEARS_TO_PLOT:
        for month, month_name in MONTHS_TO_PLOT.items():

            # Filter data
            data_subset = df[(df['year'] == year) & (df['month'] == month)]

            if len(data_subset) == 0:
                print(f"  ⚠️  No data for {year} {month_name}")
                continue

            # Calculate ensemble statistics
            ensemble_stats = data_subset.groupby('zone_id').agg({
                'mean_risk': ['mean', 'std']
            }).reset_index()
            ensemble_stats.columns = ['zone_id', 'ensemble_mean', 'ensemble_std']

            # 1. Ensemble Mean Map
            risk_data = ensemble_stats[['zone_id', 'ensemble_mean']].copy()
            risk_data.columns = ['zone_id', 'risk_value']

            title = f"{scenario_label} - {month_name} {year}\nEnsemble Mean Fire Risk"
            output_path = OUTPUT_DIR / f"map_{scenario_key}_{year}_{month:02d}_ensemble_mean.png"

            create_risk_map(gdf_zones, risk_data, title, output_path)

            # 2. Ensemble Spread Map (uncertainty)
            risk_data = ensemble_stats[['zone_id', 'ensemble_std']].copy()
            risk_data.columns = ['zone_id', 'risk_value']

            title = f"{scenario_label} - {month_name} {year}\nEnsemble Spread (Uncertainty)"
            output_path = OUTPUT_DIR / f"map_{scenario_key}_{year}_{month:02d}_ensemble_std.png"

            create_risk_map(gdf_zones, risk_data, title, output_path, cmap='viridis')

            # 3. Change from baseline (if not baseline year)
            if year > min(YEARS_TO_PLOT):
                baseline_year = min(YEARS_TO_PLOT)
                baseline_data = df[(df['year'] == baseline_year) & (df['month'] == month)]

                if len(baseline_data) > 0:
                    baseline_stats = baseline_data.groupby('zone_id')['mean_risk'].mean().reset_index()
                    baseline_stats.columns = ['zone_id', 'baseline_risk']

                    # Calculate change
                    change_data = ensemble_stats.merge(baseline_stats, on='zone_id')
                    change_data['risk_value'] = ((change_data['ensemble_mean'] - change_data['baseline_risk']) /
                                                  change_data['baseline_risk'] * 100)

                    title = f"{scenario_label} - {month_name} {year}\nChange from {baseline_year} (%)"
                    output_path = OUTPUT_DIR / f"map_{scenario_key}_{year}_{month:02d}_change_from_baseline.png"

                    create_risk_map(gdf_zones, change_data[['zone_id', 'risk_value']],
                                   title, output_path, show_change=True)

print()
print("="*80)
print("COMPLETE")
print("="*80)
print(f"Maps saved to: {OUTPUT_DIR}")
print("="*80)
