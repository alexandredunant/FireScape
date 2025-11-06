#!/usr/bin/env python
"""
Create Final Publication Climate Figures with Climate Drivers

Shows:
1. Spatial maps with quantile information
2. Time series with actual T and P trends from climate data
3. Clear labeling of which climate quantile is used

For manuscript figures showing climate change impacts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import geopandas as gpd
from pathlib import Path
import warnings
import xarray as xr

warnings.filterwarnings("ignore")

# Use publication-quality settings
plt.style.use(['science', 'no-latex'])
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

print("="*80)
print("CREATING FINAL CLIMATE FIGURES WITH DRIVERS")
print("="*80)
print()

# ===================================================================
# CONFIGURATION
# ===================================================================

BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape")
OUTPUT_FIGURES = BASE_DIR / "output/figures"
OUTPUT_FIGURES.mkdir(exist_ok=True, parents=True)

# Fire brigade zones
FIRE_BRIGADE_ZONES = BASE_DIR / "Data/06_Administrative_Boundaries/Processed/FireBrigade_ResponsibilityAreas_Bolzano_clipped.gpkg"

# Climate projection results
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
    print("ERROR: Could not find climate projection data")
    exit(1)

# Climate data directories
TEMP_RCP45_DIR = Path("/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/tas/rcp45")
TEMP_RCP85_DIR = Path("/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/tas/rcp85")
PRECIP_RCP45_DIR = Path("/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/pr/rcp45")
PRECIP_RCP85_DIR = Path("/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/pr/rcp85")

print(f"✓ RCP4.5 projections: {RCP45_CSV}")
print(f"✓ RCP8.5 projections: {RCP85_CSV}")
print()

# ===================================================================
# LOAD DATA
# ===================================================================

print("Loading projection data...")
df_rcp45 = pd.read_csv(RCP45_CSV)
df_rcp45['scenario'] = 'RCP4.5'

df_rcp85 = pd.read_csv(RCP85_CSV)
df_rcp85['scenario'] = 'RCP8.5'

df = pd.concat([df_rcp45, df_rcp85], ignore_index=True)

zones_gdf = gpd.read_file(FIRE_BRIGADE_ZONES)
zone_name_col = 'PLACE_IT'

print(f"✓ Loaded {len(df)} projection records")
print(f"✓ Loaded {len(zones_gdf)} fire brigade zones")
print()

# ===================================================================
# LOAD ACTUAL CLIMATE DATA FOR KEY MONTHS
# ===================================================================

print("Loading climate driver data...")

def load_climate_stats(scenario, variable):
    """Load mean climate values for August across years from NetCDF files."""
    if variable == 'temp':
        base_dir = TEMP_RCP45_DIR if scenario == 'RCP4.5' else TEMP_RCP85_DIR
        var_name = 'tas'
    else:
        base_dir = PRECIP_RCP45_DIR if scenario == 'RCP4.5' else PRECIP_RCP85_DIR
        var_name = 'pr'

    stats = {}
    quantiles = ['pctl25', 'pctl50', 'pctl99']
    years = [2020, 2030, 2040, 2050, 2060, 2070, 2080]

    for quantile in quantiles:
        stats[quantile] = []

        # Load the NetCDF file for this quantile
        nc_file = base_dir / f"{var_name}_EUR-11_{quantile}_{scenario.lower().replace('.', '')}.nc"

        if nc_file.exists():
            try:
                # Open NetCDF file
                ds = xr.open_dataset(nc_file)

                # Get the variable data
                data = ds[var_name]

                # Extract August values for each year
                for year in years:
                    try:
                        # Filter for all days in August of this year
                        august_data = data.sel(time=slice(f'{year}-08-01', f'{year}-08-31'))

                        # Calculate spatial mean (ignoring NaN values outside study area)
                        mean_val = float(np.nanmean(august_data.values))

                        # Convert units if needed
                        # Temperature is already in °C, no conversion needed
                        if variable == 'precip':
                            # Check if precipitation needs conversion
                            # Usually already in mm/day from QDM, but verify units
                            pass

                        stats[quantile].append(mean_val)
                    except Exception as e:
                        # If this specific year/month not found, append NaN
                        print(f"    Warning: Could not load {year}-08 for {quantile}: {e}")
                        stats[quantile].append(np.nan)

                ds.close()
            except Exception as e:
                print(f"  Warning: Could not load {nc_file}: {e}")
                stats[quantile] = [np.nan] * len(years)
        else:
            print(f"  Warning: File not found: {nc_file}")
            stats[quantile] = [np.nan] * len(years)

    return years, stats

# Load climate data for both scenarios
climate_data = {}
for scenario in ['RCP4.5', 'RCP8.5']:
    years_t, temp_stats = load_climate_stats(scenario, 'temp')
    years_p, precip_stats = load_climate_stats(scenario, 'precip')
    climate_data[scenario] = {
        'years': years_t,
        'temp': temp_stats,
        'precip': precip_stats
    }

print("✓ Loaded climate driver data")
print()

# ===================================================================
# EXPORT DATA TO CSV
# ===================================================================

print("Exporting data to CSV files...")

# Export main projection data
OUTPUT_DATA = BASE_DIR / "output/04_Zone_Climate_Projections"
OUTPUT_DATA.mkdir(exist_ok=True, parents=True)

projection_export = OUTPUT_DATA / "climate_projection_data.csv"
df.to_csv(projection_export, index=False)
print(f"✓ Exported projection data: {projection_export.name}")

# Export climate driver data in flat format
climate_records = []
for scenario in ['RCP4.5', 'RCP8.5']:
    years = climate_data[scenario]['years']
    for quantile in ['pctl25', 'pctl50', 'pctl99']:
        temp_vals = climate_data[scenario]['temp'][quantile]
        precip_vals = climate_data[scenario]['precip'][quantile]

        for i, year in enumerate(years):
            climate_records.append({
                'scenario': scenario,
                'year': year,
                'climate_quantile': quantile,
                'temperature_celsius': temp_vals[i],
                'precipitation_mm_per_day': precip_vals[i]
            })

climate_df = pd.DataFrame(climate_records)
climate_export = OUTPUT_DATA / "climate_drivers_data.csv"
climate_df.to_csv(climate_export, index=False)
print(f"✓ Exported climate driver data: {climate_export.name}")
print()

# ===================================================================
# FIGURE 1: SPATIAL MAPS WITH QUANTILE INFO
# ===================================================================

print("Creating spatial maps (median climate scenario)...")

QUANTILE = 'pctl50'
QUANTILE_LABEL = 'Median (50th percentile)'

df_maps = df[(df['month'] == 8) & (df['climate_quantile'] == QUANTILE)]

TIME_PERIODS = [2020, 2050, 2080]
SCENARIOS = ['RCP4.5', 'RCP8.5']

fig, axes = plt.subplots(len(SCENARIOS), len(TIME_PERIODS), figsize=(14, 8))

for i, scenario in enumerate(SCENARIOS):
    for j, year in enumerate(TIME_PERIODS):
        ax = axes[i, j]

        df_subset = df_maps[(df_maps['scenario'] == scenario) &
                           (df_maps['year'] == year)]
        df_subset = df_subset.groupby('zone_name')['mean_risk'].mean().reset_index()

        zones_plot = zones_gdf.copy()
        zones_plot = zones_plot.merge(
            df_subset[['zone_name', 'mean_risk']],
            left_on=zone_name_col,
            right_on='zone_name',
            how='left'
        )

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

        # Add panel label (a, b, c, etc.)
        panel_label = chr(97 + i * len(TIME_PERIODS) + j)  # a, b, c, d, e, f
        ax.text(0.02, 0.98, f'({panel_label}) {scenario} - {year}',
               transform=ax.transAxes, fontsize=11, fontweight='bold',
               va='top', ha='left', bbox=dict(boxstyle='round',
               facecolor='white', alpha=1, edgecolor='none'))
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
cbar.set_label('Relative Fire Risk (August)', fontsize=12)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout(rect=[0, 0, 0.90, 1.00])
output_path = OUTPUT_FIGURES / "climate_maps_spatial_evolution_pctl50.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Saved: {output_path.name}")
print()

# ===================================================================
# FIGURE 2: TIME SERIES WITH CLIMATE DRIVERS (COMBINED SCENARIOS)
# ===================================================================

print("Creating combined time series with climate drivers...")

KEY_ZONES = ['Bolzano', 'Merano', 'Bressanone', 'Brunico']

# Define styles for scenarios and quantiles
scenario_styles = {
    'RCP4.5': {'linestyle': '-', 'linewidth': 1.0},
    'RCP8.5': {'linestyle': '--', 'linewidth': 1.0}
}

quantile_markers = {
    'pctl25': 'v',   # down triangle
    'pctl50': 'o',   # circle
    'pctl75': 's',   # square
    'pctl99': '^'    # up triangle
}

quantile_sizes = {
    'pctl25': 3,
    'pctl50': 4,
    'pctl75': 3,
    'pctl99': 3
}

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Panel 1: Fire Risk with different quantiles as different markers
ax = axes[0]

colors = plt.cm.Set2(np.linspace(0, 1, len(KEY_ZONES)))

for idx, zone in enumerate(KEY_ZONES):
    for scenario in ['RCP4.5', 'RCP8.5']:
        for quantile in ['pctl25', 'pctl50', 'pctl99']:
            df_filtered = df[(df['scenario'] == scenario) &
                           (df['month'] == 8) &
                           (df['climate_quantile'] == quantile) &
                           (df['zone_name'] == zone)]

            if len(df_filtered) > 0:
                df_agg = df_filtered.groupby('year')['mean_risk'].mean().reset_index()

                # Label only for median of RCP8.5 to avoid legend clutter
                if quantile == 'pctl50' and scenario == 'RCP8.5':
                    label = zone
                else:
                    label = None

                ax.plot(df_agg['year'], df_agg['mean_risk'],
                       marker=quantile_markers[quantile],
                       markersize=quantile_sizes[quantile],
                       linestyle=scenario_styles[scenario]['linestyle'],
                       linewidth=scenario_styles[scenario]['linewidth'],
                       color=colors[idx], label=label, alpha=0.8, zorder=3)

ax.set_ylabel('Relative Fire Risk', fontsize=11)
ax.text(0.02, 0.98, '(a) Fire Risk Projections', transform=ax.transAxes,
       fontsize=10, fontweight='bold', va='top', ha='left')
ax.grid(True, alpha=0.3, zorder=0)

# Create legend for zones in upper right
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
zones_legend = ax.legend(loc='upper right', fontsize=9, ncol=2, title='Zones',
                         framealpha=1, fancybox=False, shadow=False,
                         facecolor='white', edgecolor='none')
ax.add_artist(zones_legend)  # Add first legend to keep it

# Create custom legend for scenarios and quantiles at bottom
legend_elements = [
    Line2D([0], [0], color='gray', linestyle='-', linewidth=1.0, label='RCP4.5'),
    Line2D([0], [0], color='gray', linestyle='--', linewidth=1.0, label='RCP8.5'),
    Line2D([0], [0], color='gray', marker='v', markersize=3, linestyle='None', label='25th pctl'),
    Line2D([0], [0], color='gray', marker='o', markersize=4, linestyle='None', label='50th pctl'),
    Line2D([0], [0], color='gray', marker='^', markersize=3, linestyle='None', label='99th pctl'),
]
ax.legend(handles=legend_elements, loc='lower center', fontsize=8, ncol=5,
         framealpha=1, fancybox=False, shadow=False,
         facecolor='white', edgecolor='none', bbox_to_anchor=(0.5, -0.15))

# Panel 2: Temperature and Precipitation on dual y-axes
ax1 = axes[1]

# Temperature on left y-axis
for scenario in ['RCP4.5', 'RCP8.5']:
    years = climate_data[scenario]['years']

    for quantile in ['pctl25', 'pctl50', 'pctl99']:
        temp_vals = climate_data[scenario]['temp'][quantile]

        if not all(np.isnan(temp_vals)):
            # Label only median
            if quantile == 'pctl50':
                label = f'Temperature {scenario}'
            else:
                label = None

            ax1.plot(years, temp_vals,
                    marker=quantile_markers[quantile],
                    markersize=quantile_sizes[quantile],
                    linestyle=scenario_styles[scenario]['linestyle'],
                    linewidth=scenario_styles[scenario]['linewidth'],
                    color='orangered', label=label, alpha=0.7, zorder=3)

ax1.set_ylabel('Temperature (°C)', fontsize=11, color='orangered')
ax1.tick_params(axis='y', labelcolor='orangered')

# Precipitation on right y-axis
ax2 = ax1.twinx()

for scenario in ['RCP4.5', 'RCP8.5']:
    years = climate_data[scenario]['years']

    for quantile in ['pctl25', 'pctl50', 'pctl99']:
        precip_vals = climate_data[scenario]['precip'][quantile]

        if not all(np.isnan(precip_vals)):
            # Label only median
            if quantile == 'pctl50':
                label = f'Precipitation {scenario}'
            else:
                label = None

            ax2.plot(years, precip_vals,
                    marker=quantile_markers[quantile],
                    markersize=quantile_sizes[quantile],
                    linestyle=scenario_styles[scenario]['linestyle'],
                    linewidth=scenario_styles[scenario]['linewidth'],
                    color='steelblue', label=label, alpha=0.7, zorder=2)

ax2.set_ylabel('Precipitation (mm/day)', fontsize=11, color='steelblue')
ax2.tick_params(axis='y', labelcolor='steelblue')

# Add panel label
ax1.text(0.02, 0.98, '(b) Climate Drivers', transform=ax1.transAxes,
        fontsize=10, fontweight='bold', va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='none'))
ax1.grid(True, alpha=0.3, zorder=0)
ax1.set_xlabel('Year', fontsize=11)

# Add combined legend for temperature and precipitation in upper right
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
climate_legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                            fontsize=8, framealpha=1, fancybox=False,
                            shadow=False, facecolor='white', edgecolor='none')
ax1.add_artist(climate_legend)  # Add first legend to keep it

# Add legend for quantiles at bottom
quantile_legend_elements = [
    Line2D([0], [0], color='gray', marker='v', markersize=3, linestyle='None', label='25th pctl'),
    Line2D([0], [0], color='gray', marker='o', markersize=4, linestyle='None', label='50th pctl'),
    Line2D([0], [0], color='gray', marker='^', markersize=3, linestyle='None', label='99th pctl'),
]
ax1.legend(handles=quantile_legend_elements, loc='lower center', fontsize=8, ncol=3,
          framealpha=1, fancybox=False, shadow=False,
          facecolor='white', edgecolor='none', bbox_to_anchor=(0.5, -0.15))

plt.tight_layout()
output_path = OUTPUT_FIGURES / "climate_drivers_combined.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Saved: {output_path.name}")
print()

# ===================================================================
# FIGURE 3: RISK CHANGE WITH QUANTILE INFO
# ===================================================================

print("Creating risk change maps (median climate scenario)...")

QUANTILE = 'pctl50'
df_change = df[(df['month'] == 8) & (df['climate_quantile'] == QUANTILE)]

df_2020 = df_change[df_change['year'] == 2020].groupby(['zone_name', 'scenario'])['mean_risk'].mean().reset_index()
df_2020.columns = ['zone_name', 'scenario', 'risk_2020']

df_2080 = df_change[df_change['year'] == 2080].groupby(['zone_name', 'scenario'])['mean_risk'].mean().reset_index()
df_2080.columns = ['zone_name', 'scenario', 'risk_2080']

df_delta = df_2020.merge(df_2080, on=['zone_name', 'scenario'])
df_delta['risk_change_pct'] = 100 * (df_delta['risk_2080'] - df_delta['risk_2020']) / df_delta['risk_2020']

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

    # Calculate symmetric scale centered on 0
    max_abs_val = df_delta['risk_change_pct'].abs().max()
    vmin_val = -max_abs_val
    vmax_val = max_abs_val

    zones_plot.plot(
        column='risk_change_pct',
        ax=ax,
        cmap='RdYlBu_r',
        edgecolor='black',
        linewidth=0.3,
        legend=False,
        vmin=vmin_val,
        vmax=vmax_val,
        missing_kwds={'color': 'lightgrey'}
    )

    median_change = df_subset['risk_change_pct'].median()
    # Add panel label with scenario and median info
    panel_label = chr(97 + idx)  # a, b
    sign = '+' if median_change >= 0 else ''
    ax.text(0.02, 0.98, f'({panel_label}) {scenario}\nMedian: {sign}{median_change:.1f}%',
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           va='top', ha='left', bbox=dict(boxstyle='round',
           facecolor='white', alpha=1, edgecolor='none'))
    ax.axis('off')

sm = plt.cm.ScalarMappable(
    cmap='RdYlBu_r',
    norm=plt.Normalize(vmin=vmin_val, vmax=vmax_val)
)
sm._A = []
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Fire Risk Change 2020-2080 (%)', fontsize=12)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout(rect=[0, 0, 0.90, 1.00])
output_path = OUTPUT_FIGURES / "climate_risk_change_pctl50.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Saved: {output_path.name}")
print()

print("="*80)
print("FIGURES COMPLETE")
print("="*80)
print()
print("Generated files:")
print("  • climate_maps_spatial_evolution_pctl50.png")
print("  • climate_drivers_RCP45_pctl50.png")
print("  • climate_drivers_RCP85_pctl50.png")
print("  • climate_risk_change_pctl50.png")
print()
print("All figures clearly labeled with 'Median (50th percentile) Climate'")
print()
