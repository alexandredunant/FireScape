#!/usr/bin/env python
"""
Plot historical daily min/mean/max climate vs individual ensemble member projections
for the Bolzano region.

Shows:
1) Historical and projected precipitation (min, mean, max daily values)
2) Historical and projected temperature (min, mean, max daily values)

Each ensemble member is plotted as a separate colored line.
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import glob
from pathlib import Path
import warnings
import geopandas as gpd
from datetime import datetime

warnings.filterwarnings("ignore")

plt.style.use(['science', 'no-latex'])
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

print("="*80)
print("HISTORICAL vs ENSEMBLE MEMBERS - BOLZANO REGION")
print("="*80)
print()

# ===================================================================
# CONFIGURATION
# ===================================================================

BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape")
OUTPUT_DIR = BASE_DIR / "output/04_Zone_Climate_Projections"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# AOI for Bolzano region
AOI_PATH = BASE_DIR / "Data/00_QGIS/ADMIN/BOLZANO_REGION_UTM32.gpkg"

# Historical data paths (same as training data)
TEMP_DIR = Path("/mnt/CEPH_PROJECTS/CLIMATE/GRIDS/TEMPERATURE/TIME_SERIES/UPLOAD")
PRECIP_DIR = Path("/mnt/CEPH_PROJECTS/CLIMATE/GRIDS/PRECIPITATION/TIME_SERIES/UPLOAD")

# Projection data paths
PROJECTION_BASE = Path("/mnt/CEPH_PROJECTS/FACT_CLIMAX/CORDEX-Adjust/QDM")

# Settings
HISTORICAL_YEARS = range(1999, 2020)
PROJECTION_YEARS = [2020, 2030, 2040, 2050, 2060, 2070, 2080]
SCENARIO = "rcp85"  # Focus on RCP8.5
MONTH = 8  # August

# ===================================================================
# LOAD BOLZANO AOI
# ===================================================================

print("Loading Bolzano region AOI...")
aoi_gdf = gpd.read_file(AOI_PATH)
# Reproject to EPSG:3035 (LAEA Europe) to match CORDEX data
aoi_laea = aoi_gdf.to_crs("EPSG:3035")
print(f"  AOI bounds (EPSG:3035): {aoi_laea.total_bounds}")
print()

# ===================================================================
# EXTRACT HISTORICAL DATA (August daily stats)
# ===================================================================

print(f"Extracting historical August daily statistics ({HISTORICAL_YEARS.start}-{HISTORICAL_YEARS.stop-1})...")

historical_stats = []

for year in HISTORICAL_YEARS:
    try:
        # Find temperature file for August
        temp_pattern = str(TEMP_DIR / str(year) / f"*{year}08.nc")
        temp_files = [f for f in glob.glob(temp_pattern) if 'zip9' not in f]

        # Find precipitation file for August
        precip_pattern = str(PRECIP_DIR / str(year) / f"*{year}08.nc")
        precip_files = [f for f in glob.glob(precip_pattern) if 'zip9' not in f]

        if not temp_files or not precip_files:
            print(f"  Warning: Missing files for {year}")
            continue

        # Load temperature
        ds_temp = xr.open_dataset(temp_files[0])
        temp_vars = [v for v in ds_temp.data_vars if len(ds_temp[v].dims) > 1]
        temp_data = ds_temp[temp_vars[0]]

        # Calculate daily spatial mean, then get min/mean/max across days
        if 'DATE' in temp_data.dims or 'time' in temp_data.dims:
            time_dim = 'DATE' if 'DATE' in temp_data.dims else 'time'
            spatial_dims = [d for d in temp_data.dims if d not in [time_dim]]
            daily_spatial_mean = temp_data.mean(dim=spatial_dims)

            temp_min = float(daily_spatial_mean.min().values)
            temp_mean = float(daily_spatial_mean.mean().values)
            temp_max = float(daily_spatial_mean.max().values)
        else:
            temp_min = temp_mean = temp_max = float(np.nanmean(temp_data.values))

        ds_temp.close()

        # Load precipitation
        ds_precip = xr.open_dataset(precip_files[0])
        precip_vars = [v for v in ds_precip.data_vars if len(ds_precip[v].dims) > 1]
        precip_data = ds_precip[precip_vars[0]]

        # Calculate daily spatial mean, then get min/mean/max across days
        if 'DATE' in precip_data.dims or 'time' in precip_data.dims:
            time_dim = 'DATE' if 'DATE' in precip_data.dims else 'time'
            spatial_dims = [d for d in precip_data.dims if d not in [time_dim]]
            daily_spatial_mean = precip_data.mean(dim=spatial_dims)

            precip_min = float(daily_spatial_mean.min().values)
            precip_mean = float(daily_spatial_mean.mean().values)
            precip_max = float(daily_spatial_mean.max().values)
        else:
            precip_min = precip_mean = precip_max = float(np.nanmean(precip_data.values))

        ds_precip.close()

        historical_stats.append({
            'year': year,
            'temp_min': temp_min,
            'temp_mean': temp_mean,
            'temp_max': temp_max,
            'precip_min': precip_min,
            'precip_mean': precip_mean,
            'precip_max': precip_max
        })

        print(f"  {year}: T=[{temp_min:.1f}, {temp_mean:.1f}, {temp_max:.1f}]°C, "
              f"P=[{precip_min:.2f}, {precip_mean:.2f}, {precip_max:.2f}] mm/day")

    except Exception as e:
        print(f"  Error processing {year}: {e}")

historical_df = pd.DataFrame(historical_stats)
print(f"\n✓ Extracted {len(historical_df)} years of historical data")
print()

# ===================================================================
# EXTRACT ENSEMBLE MEMBER PROJECTIONS (August daily stats)
# ===================================================================

print(f"Extracting ensemble member projections for {SCENARIO.upper()}...")

# Find all ensemble member files
temp_ensemble_files = sorted(glob.glob(str(PROJECTION_BASE / "tas" / SCENARIO / "*.nc")))
precip_ensemble_files = sorted(glob.glob(str(PROJECTION_BASE / "pr" / SCENARIO / "*.nc")))

print(f"  Found {len(temp_ensemble_files)} temperature ensemble members")
print(f"  Found {len(precip_ensemble_files)} precipitation ensemble members")
print()

def extract_model_name(filepath):
    """Extract GCM-RCM model combination from filename."""
    name = Path(filepath).stem
    parts = name.split('_')
    # Format: tas_EUR-11_GCM_RCM_...
    if len(parts) >= 4:
        return f"{parts[2]}_{parts[3]}"
    return name

def extract_august_stats_for_years(ds, var_name, years, aoi_bounds):
    """Extract August daily statistics for specified years from a dataset."""
    stats = {}

    # Clip to Bolzano region approximately
    # aoi_bounds is [minx, miny, maxx, maxy] in EPSG:3035
    minx, miny, maxx, maxy = aoi_bounds

    # CORDEX uses x/y coordinates in LAEA projection
    ds_clipped = ds.sel(x=slice(minx, maxx), y=slice(miny, maxy))

    for year in years:
        try:
            # Select August of this year
            time_slice = ds_clipped.sel(time=slice(f'{year}-08-01', f'{year}-08-31'))

            if len(time_slice.time) == 0:
                continue

            data = time_slice[var_name]

            # Calculate daily spatial mean, then get min/mean/max across days
            spatial_dims = [d for d in data.dims if d != 'time']
            daily_spatial_mean = data.mean(dim=spatial_dims)

            stats[year] = {
                'min': float(daily_spatial_mean.min().values),
                'mean': float(daily_spatial_mean.mean().values),
                'max': float(daily_spatial_mean.max().values)
            }

        except Exception as e:
            print(f"    Warning: Could not extract {year} from ensemble: {e}")
            continue

    return stats

# Extract temperature projections
print("Processing temperature ensemble members...")
temp_ensemble_data = {}

for file in temp_ensemble_files:  # Process all ensemble members
    model_name = extract_model_name(file)
    print(f"  Loading {model_name}...")

    try:
        ds = xr.open_dataset(file, chunks={'time': 365})
        stats = extract_august_stats_for_years(ds, 'tas', PROJECTION_YEARS, aoi_laea.total_bounds)

        if stats:
            temp_ensemble_data[model_name] = stats
            print(f"    ✓ Extracted {len(stats)} years")

        ds.close()
    except Exception as e:
        print(f"    ✗ Error: {e}")

print()

# Extract precipitation projections
print("Processing precipitation ensemble members...")
precip_ensemble_data = {}

for file in precip_ensemble_files:  # Process all ensemble members
    model_name = extract_model_name(file)
    print(f"  Loading {model_name}...")

    try:
        ds = xr.open_dataset(file, chunks={'time': 365})
        stats = extract_august_stats_for_years(ds, 'pr', PROJECTION_YEARS, aoi_laea.total_bounds)

        if stats:
            precip_ensemble_data[model_name] = stats
            print(f"    ✓ Extracted {len(stats)} years")

        ds.close()
    except Exception as e:
        print(f"    ✗ Error: {e}")

print()
print(f"✓ Extracted {len(temp_ensemble_data)} temperature ensemble members")
print(f"✓ Extracted {len(precip_ensemble_data)} precipitation ensemble members")
print()

# ===================================================================
# CREATE VISUALIZATION
# ===================================================================

print("Creating visualization...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Color palette for ensemble members - use tab20 for more colors
colors = plt.cm.tab20(np.linspace(0, 1, 20))

# -------------------------------------------------------------------
# SUBPLOT 1: PRECIPITATION
# -------------------------------------------------------------------

ax = axes[0]

# Plot historical min/mean/max as scatter points
ax.scatter(historical_df['year'], historical_df['precip_min'],
          color='darkblue', marker='v', s=30, alpha=0.6, label='Historical Min', zorder=10)
ax.scatter(historical_df['year'], historical_df['precip_mean'],
          color='steelblue', marker='o', s=50, alpha=0.8, label='Historical Mean', zorder=10)
ax.scatter(historical_df['year'], historical_df['precip_max'],
          color='lightblue', marker='^', s=30, alpha=0.6, label='Historical Max', zorder=10)

# Plot each ensemble member projection
for idx, (model_name, stats) in enumerate(precip_ensemble_data.items()):
    years = sorted(stats.keys())
    means = [stats[y]['mean'] for y in years]

    ax.plot(years, means, marker='o', markersize=4, linestyle='-', linewidth=1.5,
           color=colors[idx % len(colors)], alpha=0.7, label=f'Proj: {model_name}', zorder=5)

ax.set_ylabel('Precipitation (mm/day)', fontsize=11, fontweight='bold')
ax.set_title('(a) August Daily Precipitation - Bolzano Region', fontsize=12, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, zorder=0)
ax.legend(loc='upper left', fontsize=7, framealpha=0.9, ncol=2)
ax.set_xlim(1995, 2085)

# -------------------------------------------------------------------
# SUBPLOT 2: TEMPERATURE
# -------------------------------------------------------------------

ax = axes[1]

# Plot historical min/mean/max as scatter points
ax.scatter(historical_df['year'], historical_df['temp_min'],
          color='darkred', marker='v', s=30, alpha=0.6, label='Historical Min', zorder=10)
ax.scatter(historical_df['year'], historical_df['temp_mean'],
          color='orangered', marker='o', s=50, alpha=0.8, label='Historical Mean', zorder=10)
ax.scatter(historical_df['year'], historical_df['temp_max'],
          color='coral', marker='^', s=30, alpha=0.6, label='Historical Max', zorder=10)

# Plot each ensemble member projection
for idx, (model_name, stats) in enumerate(temp_ensemble_data.items()):
    years = sorted(stats.keys())
    means = [stats[y]['mean'] for y in years]

    ax.plot(years, means, marker='o', markersize=4, linestyle='-', linewidth=1.5,
           color=colors[idx % len(colors)], alpha=0.7, label=f'Proj: {model_name}', zorder=5)

ax.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
ax.set_xlabel('Year', fontsize=11, fontweight='bold')
ax.set_title('(b) August Daily Temperature - Bolzano Region', fontsize=12, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, zorder=0)
ax.legend(loc='upper left', fontsize=7, framealpha=0.9, ncol=2)
ax.set_xlim(1995, 2085)

plt.tight_layout()

output_path = OUTPUT_DIR / "historical_vs_ensemble_members_bolzano.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Saved: {output_path}")
print()

print("="*80)
print("COMPLETE")
print("="*80)
print(f"Historical period: {HISTORICAL_YEARS.start}-{HISTORICAL_YEARS.stop-1}")
print(f"Projection scenario: {SCENARIO.upper()}")
print(f"Projection years: {PROJECTION_YEARS}")
print(f"Temperature ensemble members: {len(temp_ensemble_data)}")
print(f"Precipitation ensemble members: {len(precip_ensemble_data)}")
print("="*80)
