#!/usr/bin/env python
"""
Create Publication-Quality Climate Projection Maps and Plots

Generates:
1. SPATIAL MAPS: Fire risk by forestry zone for different time periods (2020, 2050, 2080)
2. CLEAN TIME SERIES: Scatter plots with error bars showing risk evolution
3. Proper legends and clear visualization

For manuscript Figure: Climate Change Impacts on Fire Risk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
import geopandas as gpd
from pathlib import Path
import warnings
from matplotlib.patches import Patch
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# Use publication-quality settings
plt.style.use(['science', 'no-latex'])
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

print("="*80)
print("CREATING PUBLICATION CLIMATE MAPS AND PLOTS")
print("="*80)
print()

# ===================================================================
# CONFIGURATION
# ===================================================================

BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape")
OUTPUT_FIGURES = BASE_DIR / "output/figures"
OUTPUT_FIGURES.mkdir(exist_ok=True, parents=True)

# Fire brigade zones shapefile
FIRE_BRIGADE_ZONES = BASE_DIR / "Data/06_Administrative_Boundaries/Processed/FireBrigade_ResponsibilityAreas_Bolzano_clipped.gpkg"

# Check for data
DATA_LOCATIONS = [
    BASE_DIR / "output/04_Zone_Climate_Projections",
    BASE_DIR / "Archive/Scripts/OUTPUT/04_Zone_Climate_Projections"
]

RCP45_CSV = None
RCP85_CSV = None

for loc in DATA_LOCATIONS:
    rcp45_path = loc / "rcp45/zone_projections_relative_risk.csv"
    rcp85_path = loc / "rcp85/zone_projections_relative_risk.csv"
    if rcp45_path.exists():
        RCP45_CSV = rcp45_path
    if rcp85_path.exists():
        RCP85_CSV = rcp85_path

if RCP45_CSV is None or RCP85_CSV is None:
    print("ERROR: Could not find climate projection data files")
    exit(1)

if not FIRE_BRIGADE_ZONES.exists():
    print(f"ERROR: Fire brigade zones shapefile not found: {FIRE_BRIGADE_ZONES}")
    exit(1)

print(f"✓ Found RCP4.5 data: {RCP45_CSV}")
print(f"✓ Found RCP8.5 data: {RCP85_CSV}")
print(f"✓ Found zones shapefile: {FIRE_BRIGADE_ZONES}")
print()

# ===================================================================
# LOAD DATA
# ===================================================================

print("Loading data...")
df_rcp45 = pd.read_csv(RCP45_CSV)
df_rcp45['scenario'] = 'RCP4.5'

df_rcp85 = pd.read_csv(RCP85_CSV)
df_rcp85['scenario'] = 'RCP8.5'

df = pd.concat([df_rcp45, df_rcp85], ignore_index=True)

# Load spatial zones
zones_gdf = gpd.read_file(FIRE_BRIGADE_ZONES)

print(f"✓ Loaded {len(df)} projection records")
print(f"✓ Loaded {len(zones_gdf)} fire brigade zones")
print()

# Use PLACE_IT column for Italian place names
zone_name_col = 'PLACE_IT'

print(f"Using zone name column: {zone_name_col}")
print(f"Sample zone names: {zones_gdf[zone_name_col].head(5).tolist()}")
print()

# ===================================================================
# FIGURE 1: SPATIAL MAPS OF RISK EVOLUTION
# ===================================================================

print("Creating spatial maps...")

# Focus on August (peak fire season) and median climate scenario
df_maps = df[(df['month'] == 8) & (df['climate_quantile'] == 'pctl50')]

# Select time periods for maps
TIME_PERIODS = [2020, 2050, 2080]
SCENARIOS = ['RCP4.5', 'RCP8.5']

fig, axes = plt.subplots(len(SCENARIOS), len(TIME_PERIODS),
                         figsize=(14, 8))

for i, scenario in enumerate(SCENARIOS):
    for j, year in enumerate(TIME_PERIODS):
        ax = axes[i, j]

        # Get data for this scenario and year
        df_subset = df_maps[(df_maps['scenario'] == scenario) &
                           (df_maps['year'] == year)]

        # Merge with spatial data
        zones_plot = zones_gdf.copy()
        zones_plot = zones_plot.merge(
            df_subset[['zone_name', 'mean_risk']],
            left_on=zone_name_col,
            right_on='zone_name',
            how='left'
        )

        # Plot
        zones_plot.plot(
            column='mean_risk',
            ax=ax,
            cmap='YlOrRd',
            edgecolor='black',
            linewidth=0.3,
            legend=False,
            vmin=df_maps['mean_risk'].quantile(0.05),
            vmax=df_maps['mean_risk'].quantile(0.95),
            missing_kwds={'color': 'lightgrey'}
        )

        ax.set_title(f'{scenario} - {year}', fontsize=11, fontweight='bold')
        ax.axis('off')

# Add colorbar
sm = plt.cm.ScalarMappable(
    cmap='YlOrRd',
    norm=plt.Normalize(
        vmin=df_maps['mean_risk'].quantile(0.05),
        vmax=df_maps['mean_risk'].quantile(0.95)
    )
)
sm._A = []
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Relative Fire Risk (August)', fontsize=11)

plt.tight_layout(rect=[0, 0, 0.90, 1])
output_path = OUTPUT_FIGURES / "climate_maps_spatial_risk_evolution.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Saved: {output_path.name}")
print()

# ===================================================================
# FIGURE 2: CLEAN TIME SERIES WITH ERROR BARS
# ===================================================================

print("Creating clean time series plots...")

# Select key zones
KEY_ZONES = ['Bolzano', 'Merano', 'Bressanone', 'Brunico', 'Silandro']

# Filter to August and key zones
df_ts = df[(df['month'] == 8) & (df['zone_name'].isin(KEY_ZONES))]

# Create figure with 2 subplots (one per scenario)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

colors = plt.cm.Set2(np.linspace(0, 1, len(KEY_ZONES)))
zone_colors = dict(zip(KEY_ZONES, colors))

for idx, scenario in enumerate(['RCP4.5', 'RCP8.5']):
    ax = axes[idx]

    df_scenario = df_ts[df_ts['scenario'] == scenario]

    for zone in KEY_ZONES:
        df_zone = df_scenario[df_scenario['zone_name'] == zone]

        # Get median (pctl50) as main line
        df_median = df_zone[df_zone['climate_quantile'] == 'pctl50']

        if len(df_median) > 0:
            # Get uncertainty from different climate quantiles
            years = []
            mean_risk = []
            lower_err = []
            upper_err = []

            for year in sorted(df_zone['year'].unique()):
                df_year = df_zone[df_zone['year'] == year]

                pctl25 = df_year[df_year['climate_quantile'] == 'pctl25']['mean_risk'].values
                pctl50 = df_year[df_year['climate_quantile'] == 'pctl50']['mean_risk'].values
                pctl99 = df_year[df_year['climate_quantile'] == 'pctl99']['mean_risk'].values

                if len(pctl50) > 0:
                    years.append(year)
                    mean_risk.append(pctl50[0])
                    # Ensure error bars are always positive
                    lower_err.append(abs(pctl50[0] - (pctl25[0] if len(pctl25) > 0 else pctl50[0])))
                    upper_err.append(abs((pctl99[0] if len(pctl99) > 0 else pctl50[0]) - pctl50[0]))

            # Plot with error bars
            ax.errorbar(
                years, mean_risk,
                yerr=[lower_err, upper_err],
                marker='o', markersize=6,
                linewidth=2, capsize=4, capthick=1.5,
                color=zone_colors[zone],
                label=zone,
                alpha=0.8
            )

    ax.set_xlabel('Year', fontsize=12)
    ax.set_title(scenario, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

axes[0].set_ylabel('Relative Fire Risk (August)', fontsize=12)

plt.tight_layout()
output_path = OUTPUT_FIGURES / "climate_timeseries_with_uncertainty.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Saved: {output_path.name}")
print()

# ===================================================================
# FIGURE 3: RISK CHANGE QUANTIFICATION
# ===================================================================

print("Creating risk change quantification...")

# Calculate change from 2020 to 2080
df_change = df[(df['month'] == 8) & (df['climate_quantile'] == 'pctl50')]

df_2020 = df_change[df_change['year'] == 2020][['zone_name', 'scenario', 'mean_risk']]
df_2020.columns = ['zone_name', 'scenario', 'risk_2020']

df_2080 = df_change[df_change['year'] == 2080][['zone_name', 'scenario', 'mean_risk']]
df_2080.columns = ['zone_name', 'scenario', 'risk_2080']

df_delta = df_2020.merge(df_2080, on=['zone_name', 'scenario'])
df_delta['risk_change'] = df_delta['risk_2080'] - df_delta['risk_2020']
df_delta['risk_change_pct'] = 100 * df_delta['risk_change'] / df_delta['risk_2020']

# Create spatial map of risk change
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for idx, scenario in enumerate(['RCP4.5', 'RCP8.5']):
    ax = axes[idx]

    df_subset = df_delta[df_delta['scenario'] == scenario]

    zones_plot = zones_gdf.copy()
    zones_plot = zones_plot.merge(
        df_subset[['zone_name', 'risk_change_pct']],
        left_on=zone_name_col,
        right_on='zone_name',
        how='left'
    )

    zones_plot.plot(
        column='risk_change_pct',
        ax=ax,
        cmap='RdYlBu_r',
        edgecolor='black',
        linewidth=0.3,
        legend=False,
        vmin=-10,
        vmax=max(50, df_delta['risk_change_pct'].max()),
        missing_kwds={'color': 'lightgrey'}
    )

    ax.set_title(f'{scenario}: Change 2020→2080 (%)',
                fontsize=12, fontweight='bold')
    ax.axis('off')

# Add colorbar
sm = plt.cm.ScalarMappable(
    cmap='RdYlBu_r',
    norm=plt.Normalize(vmin=-10, vmax=max(50, df_delta['risk_change_pct'].max()))
)
sm._A = []
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Fire Risk Change (%)', fontsize=11)

plt.tight_layout(rect=[0, 0, 0.90, 1])
output_path = OUTPUT_FIGURES / "climate_maps_risk_change.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Saved: {output_path.name}")
print()

# ===================================================================
# SUMMARY STATISTICS
# ===================================================================

print("="*80)
print("SUMMARY STATISTICS")
print("="*80)
print()

for scenario in ['RCP4.5', 'RCP8.5']:
    df_scen = df_delta[df_delta['scenario'] == scenario]
    print(f"{scenario}:")
    print(f"  Mean risk increase: {df_scen['risk_change_pct'].mean():.1f}%")
    print(f"  Median risk increase: {df_scen['risk_change_pct'].median():.1f}%")
    print(f"  Max risk increase: {df_scen['risk_change_pct'].max():.1f}% ({df_scen.loc[df_scen['risk_change_pct'].idxmax(), 'zone_name']})")
    print(f"  Min risk increase: {df_scen['risk_change_pct'].min():.1f}% ({df_scen.loc[df_scen['risk_change_pct'].idxmin(), 'zone_name']})")
    print()

print("="*80)
print("FIGURES COMPLETE")
print("="*80)
print()
print("Generated files in output/figures/:")
print("  • climate_maps_spatial_risk_evolution.png - Spatial maps (2020, 2050, 2080)")
print("  • climate_timeseries_with_uncertainty.png - Clean time series with error bars")
print("  • climate_maps_risk_change.png - Maps showing % change 2020→2080")
print()
