#!/usr/bin/env python
"""
Create plot comparing ACTUAL historical climate data with climate projections
Uses the same data sources as model training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path
import warnings
import xarray as xr

warnings.filterwarnings("ignore")

plt.style.use(['science', 'no-latex'])
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

print("="*80)
print("CREATING HISTORICAL vs PROJECTION COMPARISON (CORRECTED)")
print("="*80)
print()

BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape")
OUTPUT_FIGURES = BASE_DIR / "output/figures"

# Extract ACTUAL historical climate data from training data sources
print("Extracting ACTUAL historical climate data (1999-2019)...")

TEMP_DIR = Path("/mnt/CEPH_PROJECTS/CLIMATE/GRIDS/TEMPERATURE/TIME_SERIES/UPLOAD")
PRECIP_DIR = Path("/mnt/CEPH_PROJECTS/CLIMATE/GRIDS/PRECIPITATION/TIME_SERIES/UPLOAD")

historical_years = list(range(1999, 2020))
historical_records = []

for year in historical_years:
    try:
        # Load August temperature - use same logic as create_raster_stacks.py
        # Look for any file matching the pattern, excluding zip9 versions
        import glob
        temp_pattern = str(TEMP_DIR / str(year) / f"*{year}08.nc")
        temp_files = [f for f in glob.glob(temp_pattern) if 'zip9' not in f]
        temp_file = Path(temp_files[0]) if temp_files else None

        # Load August precipitation - excluding zip9 versions
        precip_pattern = str(PRECIP_DIR / str(year) / f"*{year}08.nc")
        precip_files = [f for f in glob.glob(precip_pattern) if 'zip9' not in f]
        precip_file = Path(precip_files[0]) if precip_files else None

        if temp_file.exists() and precip_file.exists():
            # Open temperature file
            ds_temp = xr.open_dataset(temp_file)
            # Get the temperature variable (skip coordinate variables)
            temp_vars = [v for v in ds_temp.data_vars if len(ds_temp[v].dims) > 1]
            temp_var = temp_vars[0]
            temp_data = ds_temp[temp_var]

            # Calculate mean temperature for August (spatial mean across all days)
            temp_mean = float(np.nanmean(temp_data.values))

            ds_temp.close()

            # Open precipitation file
            ds_precip = xr.open_dataset(precip_file)
            # Get the precipitation variable (skip coordinate variables)
            precip_vars = [v for v in ds_precip.data_vars if len(ds_precip[v].dims) > 1]
            precip_var = precip_vars[0]
            precip_data = ds_precip[precip_var]

            # Sum precipitation over all days, then take spatial mean
            # First average spatially, then sum over time to get monthly total
            spatial_dims = [d for d in precip_data.dims if d != 'DATE' and d != 'time']
            precip_monthly = float(np.nansum(precip_data.mean(dim=spatial_dims).values))

            ds_precip.close()

            historical_records.append({
                'year': year,
                'temperature': temp_mean,
                'precipitation': precip_monthly
            })

            print(f"  {year}: T={temp_mean:.1f}°C, P={precip_monthly:.1f}mm")
        else:
            print(f"  Warning: Missing files for {year}")

    except Exception as e:
        print(f"  Warning: Could not load {year}: {e}")

historical_summary = pd.DataFrame(historical_records)
print()
print(f"✓ Extracted historical data: {len(historical_summary)} years")
print(f"  Year range: {historical_summary['year'].min()} - {historical_summary['year'].max()}")
print(f"  Mean August temperature: {historical_summary['temperature'].mean():.1f}°C")
print(f"  Mean August precipitation: {historical_summary['precipitation'].mean():.1f} mm")
print()

# Load climate projection data
print("Loading climate projection data...")
climate_df = pd.read_csv(BASE_DIR / "output/04_Zone_Climate_Projections/climate_drivers_data.csv")

print("Creating comparison plot...")

# Create figure with 2 panels (RCP4.5 and RCP8.5)
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

scenarios = ['RCP4.5', 'RCP8.5']

for idx, scenario in enumerate(scenarios):
    ax1 = axes[idx]

    # Filter projection data for this scenario
    scenario_data = climate_df[climate_df['scenario'] == scenario]

    # Plot historical temperature (left y-axis)
    ax1.scatter(historical_summary['year'], historical_summary['temperature'],
               color='orangered', marker='o', s=40, alpha=0.8, edgecolors='darkred',
               linewidths=0.5, label='Historical Temperature', zorder=10)

    quantile_styles = {
        'pctl25': ('v', 0.5),
        'pctl50': ('o', 0.7),
        'pctl99': ('^', 0.5)
    }

    # Plot projected temperature quantiles
    for quantile, (marker, alpha) in quantile_styles.items():
        quant_data = scenario_data[scenario_data['climate_quantile'] == quantile]
        label = f'Projected Temp ({quantile.replace("pctl", "")}th pctl)'
        ax1.plot(quant_data['year'], quant_data['temperature_celsius'],
                 marker=marker, markersize=4, linestyle='-', linewidth=0.8,
                 color='orangered', alpha=alpha, label=label)

    ax1.set_ylabel('Temperature (°C)', fontsize=11, color='orangered')
    ax1.tick_params(axis='y', labelcolor='orangered')
    ax1.grid(True, alpha=0.3, zorder=0)

    # Plot precipitation on right y-axis
    ax2 = ax1.twinx()

    ax2.scatter(historical_summary['year'], historical_summary['precipitation'],
               color='steelblue', marker='s', s=40, alpha=0.8, edgecolors='darkblue',
               linewidths=0.5, label='Historical Precipitation', zorder=10)

    # Plot projected precipitation quantiles
    # The climate driver data is in mm/day, multiply by 31 to get monthly total
    for quantile, (marker, alpha) in quantile_styles.items():
        quant_data = scenario_data[scenario_data['climate_quantile'] == quantile]
        label = f'Projected Precip ({quantile.replace("pctl", "")}th pctl)'
        # Convert from mm/day to monthly total
        precip_monthly = quant_data['precipitation_mm_per_day'] * 31

        ax2.plot(quant_data['year'], precip_monthly,
                 marker=marker, markersize=4, linestyle='-', linewidth=0.8,
                 color='steelblue', alpha=alpha, label=label)

    ax2.set_ylabel('Precipitation (mm/month)', fontsize=11, color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue')

    # Add panel label
    panel_label = chr(97 + idx)  # a, b
    ax1.text(0.02, 0.98, f'({panel_label}) {scenario}', transform=ax1.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='none'))

    # Add combined legend for all quantiles
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    lines = lines1 + lines2
    labels = labels1 + labels2

    ax1.legend(lines, labels, loc='upper left',
              fontsize=8, framealpha=0.85, facecolor='white', edgecolor='none', ncol=4)

# Set x-axis label on bottom plot
axes[1].set_xlabel('Year', fontsize=11)

plt.tight_layout()
output_path = OUTPUT_FIGURES / "historical_vs_projections.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Saved: {output_path.name}")
print()

print("="*80)
print("SUMMARY:")
print("="*80)
print(f"Historical August precipitation (1999-2019): {historical_summary['precipitation'].mean():.1f} mm")
print(f"Historical August temperature (1999-2019): {historical_summary['temperature'].mean():.1f}°C")
print()
print("This plot shows the ACTUAL historical data used for model training")
print("compared with climate projection quantiles.")
print("="*80)
