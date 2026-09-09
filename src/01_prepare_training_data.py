#!/usr/bin/env python
# %%
"""
PREPARE TRAINING FEATURES FOR EBM FIRE RISK MODEL

This script directly extracts features from raster data without creating
intermediate spatio-temporal stacks, since the EBM model only uses:
1. Static features: from 4x4 spatial window
2. Event-day weather: Temperature (mean) and precipitation from the event day
3. SPEI features: 30-day and 90-day drought indices
4. Seasonality: Day of year encoded as sin/cos

Output: training_features_spei.parquet ready for EBM training
"""

import os
import pandas as pd
import geopandas as gpd
import rioxarray
import xarray as xr
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm, trange
from datetime import datetime
import warnings
import gc

warnings.filterwarnings("ignore")

# Try to import psutil for memory monitoring
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Note: psutil not available for memory monitoring")

print("=" * 80)
print("PREPARE TRAINING FEATURES FOR EBM FIRE RISK MODEL")
print("=" * 80)
print()

# ===================================================================
# CONFIGURATION
# ===================================================================

# Set FIRESCAPE_ROOT when the data live outside this repository.
BASE_DIR = Path(os.environ.get("FIRESCAPE_ROOT", Path(__file__).resolve().parents[1]))
OUTPUT_DIR = BASE_DIR / "output/01_Training_Data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input files
SPACETIME_PARQUET = OUTPUT_DIR / "spacetime_dataset.parquet"
STATIC_RASTER_DIR = BASE_DIR / "Data/STATIC_INPUT_250m"
TEMP_DIR = BASE_DIR / "Data/05_Meteorological_Data/Temperature"
PRECIP_DIR = BASE_DIR / "Data/05_Meteorological_Data/Precipitation"
SPEI_DIR = BASE_DIR / "Data/08_SPEI/Firescape"
LIGHTNING_DIR = BASE_DIR / "Data/05_Meteorological_Data/Lightning_Standardized"

# Output file
OUTPUT_PATH = OUTPUT_DIR / "training_features_ebm_250m.parquet"

# Spatial window size for feature extraction
WINDOW_SIZE = 1  # Point extraction (was 4 for 50m data)

# Static features to extract
STATIC_FEATURES = [
    "tri",
    "northness",
    "slope",
    "aspect",
    "nasadem",
    "treecoverdensity",
    "distroads",
    "eastness",
    "flammability",
    "walking_time_to_bldg",
    "walking_time_to_elec_infra",
]

# SPEI scales (in days)
SPEI_SCALES = [30, 90]

print(f"Configuration:")
print(f"  Input parquet: {SPACETIME_PARQUET}")
print(f"  Static rasters: {STATIC_RASTER_DIR}")
print(f"  Temperature: {TEMP_DIR}")
print(f"  Precipitation: {PRECIP_DIR}")
print(f"  SPEI directory: {SPEI_DIR}")
print(f"  Lightning directory: {LIGHTNING_DIR}")
print(f"  Output: {OUTPUT_PATH}")
print(f"  Spatial window: {WINDOW_SIZE}x{WINDOW_SIZE} pixels")
print()

# ===================================================================
# LOAD DATA
# ===================================================================

print("Loading point data...")
gdf = gpd.read_parquet(SPACETIME_PARQUET)
gdf["date"] = pd.to_datetime(gdf["date"])

print(f"✓ Loaded {len(gdf)} observations")
print(f"  Fires (bin=1): {(gdf['bin'] == 1).sum()}")
print(f"  Non-fires (bin=0): {(gdf['bin'] == 0).sum()}")
print(f"  Date range: {gdf['date'].min()} to {gdf['date'].max()}")
print()

# Sort by date to minimize SPEI file operations
print("Sorting data by date to optimize memory usage...")
gdf = gdf.sort_values("date").reset_index(drop=True)
print(f"✓ Data sorted chronologically")
print()

def validate_template_resolution(template_path):
    """Verify template is at 250m resolution."""
    da = rioxarray.open_rasterio(template_path, masked=True).squeeze()
    res = abs(da.rio.resolution()[0])

    if res != 250:
        raise ValueError(f"Template resolution is {res}m, expected 250m")

    print(f"✓ Template resolution validated: {res}m")
    return da

# Load template raster for coordinate reference
print("Loading template raster...")
TEMPLATE_PATH = STATIC_RASTER_DIR / "nasadem.tif"
template_raster = validate_template_resolution(TEMPLATE_PATH)
print(f"✓ Template loaded: {template_raster.shape}")
print()

TMEAN_PATTERN = str(BASE_DIR / "Data/05_Meteorological_Data/Temperature/tmean_{year}.nc")
PREC_PATTERN = str(BASE_DIR / "Data/05_Meteorological_Data/Precipitation/prec_{year}.nc")
LIGHTNING_PATTERN = str(BASE_DIR / "Data/05_Meteorological_Data/Lightning_Standardized/lightning_density_{year}.nc")

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================


def extract_point_value(raster_data, point, template):
    """Extract value at point location using nearest neighbor."""
    # Find nearest pixel indices
    iy = np.abs(template.y - point.y).argmin().item()
    ix = np.abs(template.x - point.x).argmin().item()

    # Extract single pixel value
    value = raster_data.isel(y=iy, x=ix).values

    # Convert to float, handle NaN
    return float(value) if not np.isnan(value) else np.nan


def extract_static_features(point, static_rasters, template):
    """
    Extract static features from PRE-LOADED rasters at point location.

    Args:
        point: Shapely point geometry
        static_rasters: Dict of {var_name: reprojected_raster_data}
        template: Template raster

    Returns:
        dict: Feature name -> value at point location
    """
    features = {}

    # Extract each static variable from pre-loaded rasters
    for var_name, raster_data in static_rasters.items():
        features[var_name] = extract_point_value(
            raster_data, point, template
        )

    return features


def extract_temporal_features(point, date, template, spei_cache):
    """
    Extract ALL temporal features (weather, lightning, SPEI) for a given point and date.

    Args:
        point: Shapely point geometry
        date: Date to extract features for
        template: Template raster for reprojection
        spei_cache: Cache dictionary for SPEI datasets

    Returns:
        dict: All temporal features (T_daily, P_daily, lightning_density, SPEI_30d, SPEI_90d)
    """
    features = {}

    # ===================================================================
    # 1. TEMPERATURE (yearly standardized NetCDF)
    # ===================================================================
    # Load yearly standardized file
    temp_file = Path(TMEAN_PATTERN.format(year=date.year))

    if temp_file.exists():
        with xr.open_dataset(temp_file, decode_times=True) as ds:
            # Select specific date from yearly file
            data = (
                ds["tmean"].sel(time=pd.to_datetime(date), method="nearest").squeeze()
            )
            # Data should already be in correct CRS (UTM 32N)
            # But ensure CRS is set
            if data.rio.crs is None:
                data = data.rio.write_crs(template.rio.crs)
            # Reproject to match template if needed
            data = data.rio.reproject_match(template.squeeze())
            features["T_daily"] = extract_point_value(
                data, point, template
            )
    else:
        print(f"no temp_file found for {date.year}")
        features["T_daily"] = np.nan

    # ===================================================================
    # 2. PRECIPITATION (yearly standardized NetCDF)
    # ===================================================================
    # Load yearly standardized file
    precip_file = Path(PREC_PATTERN.format(year=date.year))

    if precip_file.exists():
        with xr.open_dataset(precip_file, decode_times=True) as ds:
            # Select specific date from yearly file
            data = (
                ds["prec"].sel(time=pd.to_datetime(date), method="nearest").squeeze()
            )
            # Data should already be in correct CRS (UTM 32N)
            if data.rio.crs is None:
                data = data.rio.write_crs(template.rio.crs)
            # Reproject to match template if needed
            data = data.rio.reproject_match(template.squeeze())
            features["P_daily"] = extract_point_value(
                data, point, template
            )
    else:
        print(f"no precip_file found for {date.year}")
        features["P_daily"] = np.nan

    # ===================================================================
    # 3. LIGHTNING (yearly standardized NetCDF, available 2012+)
    # ===================================================================
    if date.year < 2012:
        # Lightning data not available before 2012
        features["lightning_density"] = np.nan
    else:
        # Load yearly standardized file
        lightning_file = Path(LIGHTNING_PATTERN.format(year=date.year))

        if lightning_file.exists():
            with xr.open_dataset(lightning_file, decode_times=True) as ds:
                # Select specific date from yearly file
                data = (
                    ds["lightning_density"]
                    .sel(time=pd.to_datetime(date), method="nearest")
                    .squeeze()
                )
                # Data should already be in correct CRS and extent
                # Lightning values of 0 indicate no lightning (already filled in standardization)
                if data.rio.crs is None:
                    data = data.rio.write_crs(template.rio.crs)
                data = data.rio.reproject_match(template.squeeze())

                features["lightning_density"] = extract_point_value(
                    data, point, template
                )
        else:
            print(
                f"no lightning_file found for {date.year}"
            )
            # If file doesn't exist, use 0 (no lightning)
            features["lightning_density"] = 0.0

    # ===================================================================
    # 4. SPEI (yearly NetCDF with time dimension)
    # ===================================================================
    for scale in SPEI_SCALES:
        cache_key = f"{scale}_{date.year}"

        # Load from cache or open file
        if cache_key not in spei_cache:
            spei_file = SPEI_DIR / f"SPEI{scale}_{date.year}.nc"
            if spei_file.exists():
                # Limit cache size to prevent memory issues
                if len(spei_cache) >= 4:
                    oldest_key = next(iter(spei_cache))
                    spei_cache[oldest_key].close()
                    del spei_cache[oldest_key]

                spei_cache[cache_key] = xr.open_dataset(spei_file)
            else:
                print(f"no spei_file found ")
                features[f"SPEI_{scale}d"] = np.nan
                continue

        ds = spei_cache.get(cache_key)
        if ds is not None:
            spei_var = f"SPEI{scale}"
            data = ds[spei_var].sel(DATE=date, method="nearest").squeeze()

            # SPEI uses spatial window (consistent with other features)
            data = data.rio.write_crs(template.rio.crs)
            data = data.rio.reproject_match(template.squeeze())
            features[f"SPEI_{scale}d"] = extract_point_value(
                data, point, template
            )
        else:
            features[f"SPEI_{scale}d"] = np.nan

    return features


def extract_features_for_observation(row, template, spei_cache, static_rasters):
    """
    Extract all features for a single observation.

    Returns:
        dict: Dictionary with all features
    """
    point = row.geometry
    date = row.date
    features = {}

    # 1. Extract static features (from pre-loaded rasters)
    static_features = extract_static_features(
        point, static_rasters, template
    )
    features.update(static_features)

    # 2. Extract ALL temporal features (weather, lightning, SPEI)
    temporal_features = extract_temporal_features(
        point, date, template, spei_cache
    )
    features.update(temporal_features)

    # 5. Add seasonality features
    day_of_year = date.timetuple().tm_yday
    features["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # 6. Add metadata
    features["id_obs"] = row.id_obs
    features["date"] = date
    features["bin"] = row.bin

    return features


# ===================================================================
# PRE-LOAD STATIC RASTERS (LOAD ONCE, KEEP IN MEMORY)
# ===================================================================

print("=" * 80)
print("PRE-LOADING STATIC RASTERS")
print("=" * 80)
print()

# Static rasters never change - just load them once and keep in memory
static_rasters = {}
print("Loading and reprojecting static features (once for all observations)...")

for var_name in STATIC_FEATURES:
    raster_path = STATIC_RASTER_DIR / f"{var_name}.tif"
    try:
        rds = rioxarray.open_rasterio(raster_path, masked=True)
        # Reproject once and keep in memory for entire run
        rds_matched = rds.rio.reproject_match(template_raster.squeeze())
        static_rasters[var_name] = (
            rds_matched.squeeze("band") if "band" in rds_matched.dims else rds_matched
        )
        print(f"  ✓ Loaded {var_name}")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not load {var_name}: {e}")

print(f"✓ Pre-loaded {len(static_rasters)} static rasters (will stay in memory)")
print()

# ===================================================================
# MAIN EXTRACTION LOOP (CHUNKED PROCESSING)
# ===================================================================

print("=" * 80)
print("EXTRACTING FEATURES")
print("=" * 80)
print()

# Initialize SPEI cache
spei_cache = {}
all_features = []

# Show memory usage before processing
if HAS_PSUTIL:
    mem_before = psutil.Process().memory_info().rss / 1024**3
    print(f"Memory usage before processing: {mem_before:.2f} GB")
    print()

# Process all observations
print(f"Processing {len(gdf):,} observations...")
for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Extracting features"):
    features = extract_features_for_observation(
        row, template_raster, spei_cache, static_rasters
    )
    all_features.append(features)

# Clean up SPEI cache
for ds in spei_cache.values():
    ds.close()
spei_cache.clear()

# Show memory usage after processing
if HAS_PSUTIL:
    mem_after = psutil.Process().memory_info().rss / 1024**3
    print(f"Memory usage after processing: {mem_after:.2f} GB")
    print()

print()
print("✓ Feature extraction complete!")
print()

# ===================================================================
# CREATE DATAFRAME AND QUALITY CHECKS
# ===================================================================

print("Creating feature dataframe...")
features_df = pd.DataFrame(all_features)

# Merge with geometry from original gdf
result_gdf = gdf[["id_obs", "geometry"]].merge(features_df, on="id_obs")

print(f"✓ Created dataframe with {len(result_gdf)} observations")
print()

# Quality checks
print("=" * 80)
print("QUALITY CHECKS")
print("=" * 80)
print()

print("Feature completeness:")
feature_cols = [
    col
    for col in result_gdf.columns
    if col not in ["id_obs", "geometry", "date", "bin"]
]

for col in feature_cols:
    n_valid = result_gdf[col].notna().sum()
    pct_valid = (n_valid / len(result_gdf)) * 100

    if pct_valid < 100:
        print(f"  {col:<30}: {n_valid:>5}/{len(result_gdf)} ({pct_valid:>5.1f}%)")

    if n_valid > 0 and result_gdf[col].dtype in [np.float64, np.float32]:
        mean_val = result_gdf[col].mean()
        std_val = result_gdf[col].std()
        min_val = result_gdf[col].min()
        max_val = result_gdf[col].max()

        if col in ["SPEI_30d", "SPEI_90d", "T_daily", "P_daily", "sin_doy", "cos_doy"]:
            print(
                f"      Range: [{min_val:>7.2f}, {max_val:>7.2f}], "
                f"Mean: {mean_val:>6.2f} ± {std_val:>5.2f}"
            )

print()


# Filter out rows with missing SPEI_90d (critical feature)
print("Filtering rows with missing SPEI_90d...")
n_before = len(result_gdf)
n_missing_spei90 = result_gdf["SPEI_90d"].isna().sum()
print(f"  Rows with missing SPEI_90d: {n_missing_spei90}")

# Class balance check
print("Class balance:")
print(
    f"  Fires (bin=1): {(result_gdf['bin'] == 1).sum()} "
    f"({(result_gdf['bin'] == 1).sum() / len(result_gdf) * 100:.1f}%)"
)
print(
    f"  Non-fires (bin=0): {(result_gdf['bin'] == 0).sum()} "
    f"({(result_gdf['bin'] == 0).sum() / len(result_gdf) * 100:.1f}%)"
)
print()

# Final NaN check
print("Final data completeness check:")
feature_cols_final = [
    col
    for col in result_gdf.columns
    if col not in ["id_obs", "geometry", "date", "bin"]
]

total_nans = result_gdf[feature_cols_final].isna().sum().sum()
print(f"  Total NaN values in feature columns: {total_nans}")

if total_nans > 0:
    print("  NaN values present in:")
    for col in feature_cols_final:
        n_nans = result_gdf[col].isna().sum()
        if n_nans > 0:
            pct = (n_nans / len(result_gdf)) * 100
            print(f"    {col:<30}: {n_nans:>5} ({pct:>5.2f}%)")

    # Identify critical features (exclude lightning_density which is NA for pre-2012)
    critical_features = [
        col for col in feature_cols_final if col != "lightning_density"
    ]

    critical_nans = result_gdf[critical_features].isna().sum().sum()

    if critical_nans > 0:
        print()
        print(f"  ⚠️  WARNING: {critical_nans} NaN values in critical features")
        print(
            "  Removing rows with NaN in critical features (keeping lightning NaNs)..."
        )
        n_rows_before_drop = len(result_gdf)
        result_gdf.dropna(subset=critical_features, inplace=True)
        n_rows_after_drop = len(result_gdf)
        print(f"  Removed {n_rows_before_drop - n_rows_after_drop} rows.")
        print(f"  Remaining rows: {n_rows_after_drop}")
    else:
        print()
        print("  ✓ No NaN values in critical features!")
        print("  ℹ️  Lightning NaN values preserved for pre-2012 data")
else:
    print("  ✓ No NaN values - dataset is complete!")

print()

# ===================================================================
# SAVE RESULTS
# ===================================================================

print("Saving results...")
result_gdf.to_parquet(OUTPUT_PATH)
print(f"✓ Saved to: {OUTPUT_PATH}")
print()

# Save feature names for reference
feature_names = [
    col
    for col in result_gdf.columns
    if col not in ["id_obs", "geometry", "date", "bin"]
]

feature_names_file = OUTPUT_DIR / "feature_names_ebm_250m.txt"
with open(feature_names_file, "w") as f:
    f.write("\n".join(feature_names))

print(f"✓ Feature names saved to: {feature_names_file}")
print()

# ===================================================================
# SUMMARY
# ===================================================================

print("=" * 80)
print("FEATURE PREPARATION COMPLETE!")
print("=" * 80)
print()
print("Summary:")
print(f"  Output file: {OUTPUT_PATH}")
print(f"  Total observations: {len(result_gdf)}")
print(f"  Total features: {len(feature_names)}")
print()
print("Feature groups:")
print(f"  Static features: {len(STATIC_FEATURES)}")
print(f"  Event-day weather: 2 (T_daily, P_daily)")
print(f"  Lightning: 1 (lightning_density)")
print(f"  SPEI features: {len(SPEI_SCALES)}")
print(f"  Seasonality: 2 (sin_doy, cos_doy)")
