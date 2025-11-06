#!/usr/bin/env python
"""
Fire Brigade Zone Climate Projections - PURE RELATIVE RISK

Generates climate-driven fire risk projections at the fire brigade zone level.
Uses ONLY relative probability predictions - NO scaling factors applied.

Purpose:
- Project future fire risk trends by zone under climate change
- Identify zones with greatest relative risk increases
- Support fire brigade resource planning and allocation
- Provide uncertainty estimates for risk projections

Key Features:
- Pure relative risk scores (NO conversion to absolute counts)
- Multiple climate quantiles (25th, 50th, 99th percentile)
- Decadal projections (2020-2080)
- Peak fire season focus (March and August)
- Uncertainty quantification via Bayesian posteriors
- NO FUDGE FACTORS - predictions directly from trained model

Output:
- CSV with zone-level risk scores by year/month/quantile
- Maps showing spatial patterns of risk change
- Trend analysis plots for each zone
- Uncertainty bands for all projections
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import rioxarray
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xvec
import gc
import itertools
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

print("="*80)
print("FIRE BRIGADE ZONE CLIMATE PROJECTIONS - RELATIVE RISK")
print("="*80)
print()

# ===================================================================
# CONFIGURATION
# ===================================================================

# LANDCOVER FIRE RISK MAPPING:
# This script uses pre-transformed landcover_fire_risk.tif (ordinal values 0-5)
# The transformation is done ONCE by:
#   Scripts/01_Data_Preparation/create_transformed_landcover.py
#
# For the canonical Corine → Fire Risk mapping dictionary, see that script.
# Values: 0=No risk, 1=Very low, 2=Low, 3=Moderate, 4=High, 5=Very high

# Paths
BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape")
AOI_PATH = BASE_DIR / "Data/00_QGIS/ADMIN/BOLZANO_REGION_UTM32.gpkg"
FIRE_BRIGADE_ZONES = BASE_DIR / "Data/06_Administrative_Boundaries/Processed/FireBrigade_ResponsibilityAreas_Bolzano_clipped.gpkg"
STATIC_RASTER_DIR = BASE_DIR / "Data/STATIC_INPUT"

# Climate data
TARGET_SCENARIO = "rcp85"
TEMP_BASE_DIR = Path("/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/tas") / TARGET_SCENARIO
PRECIP_BASE_DIR = Path("/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/pr") / TARGET_SCENARIO

# Model files
MODEL_DIR = BASE_DIR / "Scripts/OUTPUT/02_Model_RelativeProbability"
TRACE_PATH = MODEL_DIR / "trace_relative.nc"
SCALER_PATH = MODEL_DIR / "scaler_relative.joblib"
TEMPORAL_GROUPS_PATH = MODEL_DIR / "temporal_groups.joblib"
GROUP_NAMES_PATH = MODEL_DIR / "group_names.joblib"

# Output
OUTPUT_DIR = BASE_DIR / f"Scripts/OUTPUT/04_Zone_Climate_Projections/{TARGET_SCENARIO}"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Projection parameters
PROJECTION_YEARS = list(range(2020, 2090, 10))  # Every 10 years
# One representative month per season to capture seasonal patterns
# Winter: February, Spring: March, Summer: August, Fall: October
PROJECTION_MONTHS = [2, 3, 8, 10]  # One per season
CLIMATE_QUANTILES = ["pctl25", "pctl50", "pctl99"]
SPATIAL_RESOLUTION = 50  # Meters (native resolution - must match training data!)

# Model parameters (must match training)
TIME_STEPS = 60
STATIC_VARS = [
    'tri', 'northness', 'slope', 'aspect', 'nasadem',
    'treecoverdensity', 'landcover_fire_risk', 'distroads', 'eastness',
    'flammability', 'walking_time_to_bldg', 'walking_time_to_elec_infra'
]
DYNAMIC_VARS = ['T', 'P']
DAY_WINDOWS_TO_KEEP = [1, 3, 5, 10, 15, 30, 60]

month_names = {2: 'Feb', 3: 'Mar', 8: 'Aug', 10: 'Oct'}
print(f"Configuration:")
print(f"  Scenario: {TARGET_SCENARIO}")
print(f"  Years: {PROJECTION_YEARS[0]}-{PROJECTION_YEARS[-1]} (every 10 years)")
print(f"  Months: {', '.join([month_names[m] for m in PROJECTION_MONTHS])} (one per season)")
print(f"  Climate quantiles: {CLIMATE_QUANTILES}")
print(f"  Spatial resolution: {SPATIAL_RESOLUTION}m (matches training data)")
print(f"  ⚠️  NO DOWNSAMPLING - using native resolution to match training spatial context")
print(f"  ⚠️  NO SCALING FACTORS - pure relative risk only")
print(f"  Output: {OUTPUT_DIR}")
print()

# ===================================================================
# LOAD MODEL
# ===================================================================

print("Loading trained Bayesian model...")
trace = az.from_netcdf(TRACE_PATH)
scaler = joblib.load(SCALER_PATH)
temporal_groups = joblib.load(TEMPORAL_GROUPS_PATH)
group_names = joblib.load(GROUP_NAMES_PATH)
print("✓ Model loaded")
print(f"  Model type: Relative probability (Bayesian)")
print(f"  Output interpretation: Relative fire risk scores")
print()

# ===================================================================
# LOAD FIRE BRIGADE ZONES
# ===================================================================

print("Loading fire brigade zones...")
gdf_zones = gpd.read_file(FIRE_BRIGADE_ZONES)
print(f"✓ Loaded {len(gdf_zones)} zones")

# Ensure unique zone identifiers
if 'ID' not in gdf_zones.columns:
    gdf_zones['zone_id'] = range(len(gdf_zones))
else:
    gdf_zones['zone_id'] = gdf_zones['ID']

# Get zone names if available
zone_name_col = None
for col in ['PLACE_IT', 'NAME', 'name', 'NAME_IT']:
    if col in gdf_zones.columns:
        zone_name_col = col
        gdf_zones['zone_name'] = gdf_zones[col]
        break

if zone_name_col is None:
    gdf_zones['zone_name'] = 'Zone_' + gdf_zones['zone_id'].astype(str)

print(f"  Zone ID column: zone_id")
print(f"  Zone name column: zone_name")
print()

# ===================================================================
# CREATE PREDICTION GRID
# ===================================================================

print("Creating prediction grid...")
aoi = gpd.read_file(AOI_PATH)

# Use a static raster as template
template_path = STATIC_RASTER_DIR / "nasadem.tif"
with rioxarray.open_rasterio(template_path) as template_raster:
    clipped = template_raster.rio.clip(aoi.geometry, all_touched=True, drop=True).squeeze('band')

    # Keep native resolution to match training data spatial context
    # Training used 50m pixels with 4×4 window = 200m effective spatial context
    # Any downsampling here would create mismatch with training scale
    downsample_factor = SPATIAL_RESOLUTION // 50
    if downsample_factor > 1:
        downsampled = clipped.coarsen(
            x=downsample_factor,
            y=downsample_factor,
            boundary='trim'
        ).mean()
    else:
        downsampled = clipped  # Native 50m resolution

    # Create point grid
    valid_pixels = np.argwhere(~np.isnan(downsampled.values))
    ys_idx, xs_idx = valid_pixels[:, 0], valid_pixels[:, 1]

    coords_x = downsampled.x[xs_idx].values
    coords_y = downsampled.y[ys_idx].values

    grid_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(coords_x, coords_y),
        crs=clipped.rio.crs
    )

n_points = len(grid_gdf)
print(f"✓ Created grid: {n_points:,} points at {SPATIAL_RESOLUTION}m resolution")

# Assign grid points to zones
print("Assigning grid points to fire brigade zones...")
grid_gdf = gpd.sjoin(grid_gdf, gdf_zones[['zone_id', 'geometry']],
                     how='left', predicate='within')
grid_gdf = grid_gdf.dropna(subset=['zone_id'])
grid_gdf['zone_id'] = grid_gdf['zone_id'].astype(int)
print(f"✓ Assigned {len(grid_gdf):,} points to zones")
print()

# ===================================================================
# EXTRACT STATIC FEATURES
# ===================================================================

print("Extracting static features...")
print("  Note: Applying 4x4 window averaging to match training data processing")
print("  Note: landcover_fire_risk.tif is pre-transformed (0-5 ordinal values)")
static_features_df = pd.DataFrame(index=grid_gdf.index)

# Window size must match training (see create_raster_stacks.py line 120 and
# train_relative_probability_model.py lines 84-87)
SPATIAL_WINDOW_SIZE = 4

for var_name in tqdm(STATIC_VARS, desc="  Static Variables"):
    raster_path = STATIC_RASTER_DIR / f"{var_name}.tif"
    if not raster_path.exists():
        tqdm.write(f"  Warning: {var_name}.tif not found, skipping")
        continue

    with rioxarray.open_rasterio(raster_path) as rds:
        rds_clipped = rds.rio.clip(aoi.geometry, all_touched=True, drop=True).squeeze('band')

        if downsample_factor > 1:
            rds_downsampled = rds_clipped.coarsen(
                x=downsample_factor,
                y=downsample_factor,
                boundary='trim'
            ).mean()
        else:
            rds_downsampled = rds_clipped

        # Apply 4x4 rolling window mean to match training data processing
        # This ensures feature distributions match between training and projection
        rds_windowed = rds_downsampled.rolling(
            x=SPATIAL_WINDOW_SIZE,
            y=SPATIAL_WINDOW_SIZE,
            center=True
        ).mean()

        extracted = rds_windowed.xvec.extract_points(
            grid_gdf.geometry, x_coords='x', y_coords='y'
        )
        static_features_df[var_name] = extracted.values

# landcover_fire_risk.tif already contains pre-transformed ordinal values (0-5)
# No post-processing needed - transformation done during raster creation
# See Scripts/01_Data_Preparation/create_transformed_landcover.py

print(f"✓ Static features extracted: {static_features_df.shape}")

# ===================================================================
# VALIDATE STATIC FEATURES
# ===================================================================

print("Validating static features...")

# Check for landcover_fire_risk range
if 'landcover_fire_risk' in static_features_df.columns:
    lc_values = static_features_df['landcover_fire_risk'].dropna()
    if len(lc_values) > 0:
        lc_min, lc_max = lc_values.min(), lc_values.max()
        if lc_min < 0 or lc_max > 5:
            raise ValueError(
                f"landcover_fire_risk values out of expected range [0-5]: "
                f"found [{lc_min}, {lc_max}]"
            )
        print(f"  ✓ landcover_fire_risk in valid range: [{lc_min:.0f}, {lc_max:.0f}]")

        # Show distribution
        lc_dist = static_features_df['landcover_fire_risk'].value_counts().sort_index()
        print(f"    Distribution: {dict(lc_dist.head(6))}")
else:
    print(f"  ⚠️  WARNING: landcover_fire_risk not found in features")

# Check for missing values
missing_pct = (static_features_df.isna().sum() / len(static_features_df)) * 100
high_missing = missing_pct[missing_pct > 10]
if len(high_missing) > 0:
    print(f"  ⚠️  WARNING: High missing values:")
    for feat, pct in high_missing.items():
        print(f"    {feat}: {pct:.1f}% missing")
else:
    print(f"  ✓ Missing values < 10% for all features")

print()

# ===================================================================
# CLIMATE FEATURE EXTRACTION FUNCTION
# ===================================================================

def extract_climate_features(current_date, grid_gdf, temp_file, precip_file,
                             downsampled_template):
    """Extract climate features for a specific date."""

    start_date = current_date - timedelta(days=TIME_STEPS - 1)

    # Initialize result
    feature_cols = []
    for var in ['T', 'P']:
        for window in DAY_WINDOWS_TO_KEEP:
            for op in ['mean', 'max']:
                feature_cols.append(f'{var}_cumulative_{op}_{window}d')

    result_df = pd.DataFrame(np.nan, index=grid_gdf.index, columns=feature_cols)

    # Process temperature and precipitation
    for prefix, climate_file in [('T', temp_file), ('P', precip_file)]:
        with xr.open_dataset(climate_file) as ds:
            var_name = 'tas' if prefix == 'T' else 'pr'
            data_array = ds[var_name]

            # Filter to date range
            end_date = current_date + timedelta(days=1)
            time_filtered = data_array.sel(time=slice(start_date, end_date))
            time_filtered = time_filtered.sel(time=time_filtered.time < (pd.Timestamp(current_date) + pd.Timedelta(days=1)))

            if len(time_filtered.time) < TIME_STEPS:
                tqdm.write(f"    Warning: Only {len(time_filtered.time)}/{TIME_STEPS} days for {prefix}")

            # Set CRS
            if 'lambert_azimuthal_equal_area' in ds:
                crs_var = ds['lambert_azimuthal_equal_area']
                crs_wkt = crs_var.attrs.get('spatial_ref', None)
                if crs_wkt:
                    time_filtered = time_filtered.rio.write_crs(crs_wkt)
                else:
                    proj_string = (f"+proj=laea +lat_0={crs_var.attrs['latitude_of_projection_origin']} "
                                 f"+lon_0={crs_var.attrs['longitude_of_projection_origin']} "
                                 f"+x_0={crs_var.attrs['false_easting']} "
                                 f"+y_0={crs_var.attrs['false_northing']} "
                                 f"+datum=WGS84 +units=m +no_defs")
                    time_filtered = time_filtered.rio.write_crs(proj_string)

            # Reproject to match grid
            time_filtered_reprojected = time_filtered.rio.reproject_match(downsampled_template)

            # Downsample if needed
            if downsample_factor > 1:
                time_filtered_reprojected = time_filtered_reprojected.isel(
                    y=slice(None, None, downsample_factor),
                    x=slice(None, None, downsample_factor)
                )

            # Reverse time for cumulative operations
            reversed_time_data = time_filtered_reprojected.reindex(
                time=time_filtered_reprojected.time[::-1]
            ).chunk({"time": -1})
            num_available_days = len(reversed_time_data.time)

            # Cumulative statistics
            cumulative_sum = reversed_time_data.cumsum(dim='time', skipna=True)
            divisor = xr.DataArray(np.arange(1, num_available_days + 1), dims='time',
                                  coords={'time': reversed_time_data.time})
            cumulative_mean = cumulative_sum / divisor

            cumulative_max = xr.apply_ufunc(
                np.maximum.accumulate,
                reversed_time_data.fillna(-9999),
                input_core_dims=[['time']],
                output_core_dims=[['time']],
                dask="parallelized",
                output_dtypes=[reversed_time_data.dtype]
            )

            # Extract for each window
            for day_window in DAY_WINDOWS_TO_KEEP:
                i = day_window - 1
                if i >= num_available_days:
                    continue

                for op_name, data_stack in [('mean', cumulative_mean), ('max', cumulative_max)]:
                    target_raster = data_stack.isel(time=i)
                    extracted = target_raster.xvec.extract_points(
                        grid_gdf.geometry, x_coords='x', y_coords='y'
                    ).compute()

                    column_name = f"{prefix}_cumulative_{op_name}_{day_window}d"
                    result_df[column_name] = extracted.values

            gc.collect()

    return result_df


# ===================================================================
# PREDICTION FUNCTION
# ===================================================================

def generate_relative_predictions(trace, X_scaled, temporal_groups, group_names):
    """
    Generate PURE relative probability predictions.
    NO scaling factors applied.
    """

    n_samples = 100
    alpha_samples = trace.posterior['alpha'].values.reshape(-1)[:n_samples]
    attention_samples = trace.posterior['attention_weights'].values.reshape(-1, len(group_names))[:n_samples]

    beta_samples_dict = {}
    for group_name in group_names:
        beta_key = f'beta_{group_name}'
        if beta_key in trace.posterior:
            beta_data = trace.posterior[beta_key].values
            beta_flat = beta_data.reshape(-1, beta_data.shape[-1])[:n_samples]
            beta_samples_dict[group_name] = beta_flat

    n_points = X_scaled.shape[0]
    predictions_per_sample = []

    for sample_idx in range(n_samples):
        logit_pred = np.full(n_points, alpha_samples[sample_idx])

        for group_idx, (group_name, feature_indices) in enumerate(temporal_groups.items()):
            if group_name in beta_samples_dict:
                beta_sample = beta_samples_dict[group_name][sample_idx]
                group_features = X_scaled[:, feature_indices]
                group_contrib = np.dot(group_features, beta_sample)
                attention_weight = attention_samples[sample_idx, group_idx]
                logit_pred += attention_weight * group_contrib

        # Convert to probability - PURE RELATIVE RISK
        # Use a numerically stable sigmoid implementation to avoid overflow warnings.
        # The calculation is split into two parts for positive and negative logits
        # to prevent overflow in either direction.
        prob_pred = np.empty_like(logit_pred)
        pos_mask = logit_pred >= 0

        # For non-negative logits
        prob_pred[pos_mask] = 1 / (1 + np.exp(-logit_pred[pos_mask]))

        # For negative logits
        neg_mask = ~pos_mask
        prob_pred[neg_mask] = np.exp(logit_pred[neg_mask]) / (1 + np.exp(logit_pred[neg_mask]))
        predictions_per_sample.append(prob_pred)

    predictions_per_sample = np.array(predictions_per_sample)
    pred_mean = predictions_per_sample.mean(axis=0)
    pred_std = predictions_per_sample.std(axis=0)

    return pred_mean, pred_std


# ===================================================================
# RUN PROJECTIONS
# ===================================================================

print("="*80)
print("GENERATING ZONE-LEVEL CLIMATE PROJECTIONS")
print("="*80)
print()

all_results = []

for climate_quantile in CLIMATE_QUANTILES:
    print(f"\n{'#'*80}")
    print(f"# CLIMATE QUANTILE: {climate_quantile}")
    print(f"{'#'*80}\n")

    # Load climate data
    temp_file = TEMP_BASE_DIR / f"tas_EUR-11_{climate_quantile}_{TARGET_SCENARIO}.nc"
    precip_file = PRECIP_BASE_DIR / f"pr_EUR-11_{climate_quantile}_{TARGET_SCENARIO}.nc"

    if not temp_file.exists() or not precip_file.exists():
        print(f"  ⚠️  Climate files not found, skipping {climate_quantile}")
        continue

    projection_iterator = list(itertools.product(PROJECTION_YEARS, PROJECTION_MONTHS))

    for year, month in tqdm(projection_iterator, desc=f"  {climate_quantile} projections"):

        # Use mid-month date
        current_date = datetime(year, month, 15) ## CANT BE A SINGLE DATE - need to run everyday optimally

        # Extract climate features
        climate_features = extract_climate_features(
            current_date, grid_gdf, temp_file, precip_file, downsampled
        )

        # Combine with static features
        all_features = pd.concat([static_features_df, climate_features], axis=1)

        # Ensure all expected features present
        expected_features = list(scaler.feature_names_in_)
        missing_features = [f for f in expected_features if f not in all_features.columns]
        if missing_features:
            tqdm.write(f"  ⚠️  WARNING: {len(missing_features)} features missing, filling with 0: {missing_features[:5]}")
            for feat in missing_features:
                all_features[feat] = 0

        # Reorder to match scaler's training feature order
        all_features = all_features[expected_features]

        # Validate feature consistency with trained model
        if year == PROJECTION_YEARS[0] and month == PROJECTION_MONTHS[0] and climate_quantile == CLIMATE_QUANTILES[0]:
            tqdm.write(f"  ✓ Feature validation (first iteration only):")
            tqdm.write(f"    Expected features: {len(expected_features)}")
            tqdm.write(f"    Actual features: {len(all_features.columns)}")
            tqdm.write(f"    Feature order matches: {list(all_features.columns) == expected_features}")

        # Scale features
        X_scaled = scaler.transform(all_features)

        # Generate PURE RELATIVE predictions (NO SCALING)
        pred_mean, pred_std = generate_relative_predictions(
            trace, X_scaled, temporal_groups, group_names
        )

        # Aggregate by zone
        grid_gdf['prediction'] = pred_mean
        grid_gdf['uncertainty'] = pred_std

        zone_stats = grid_gdf.groupby('zone_id').agg({
            'prediction': ['mean', 'max', 'std'],
            'uncertainty': 'mean'
        }).reset_index()

        zone_stats.columns = ['zone_id', 'mean_risk', 'max_risk', 'spatial_std', 'prediction_uncertainty']

        zone_stats['year'] = year
        zone_stats['month'] = month
        zone_stats['climate_quantile'] = climate_quantile

        all_results.append(zone_stats)

        gc.collect()

# Combine all results
print("\n" + "="*80)
print("COMPILING RESULTS")
print("="*80)

df_results = pd.concat(all_results, ignore_index=True)

# Merge with zone metadata
df_results = df_results.merge(
    gdf_zones[['zone_id', 'zone_name']],
    on='zone_id',
    how='left'
)

# Save results
output_csv = OUTPUT_DIR / "zone_projections_relative_risk.csv"
df_results.to_csv(output_csv, index=False)
print(f"\n✓ Results saved: {output_csv}")

# ===================================================================
# SUMMARY STATISTICS
# ===================================================================

print("\n" + "="*80)
print("PROJECTION SUMMARY")
print("="*80)
print()

# Calculate baseline (2020) and future change
baseline_year = 2020
for quantile in CLIMATE_QUANTILES:
    print(f"\nClimate Quantile: {quantile}")
    print("-" * 60)

    quantile_data = df_results[df_results['climate_quantile'] == quantile]

    # Group months by season (one representative month per season)
    seasons = {
        'Winter (Feb)': [2],
        'Spring (Mar)': [3],
        'Summer (Aug)': [8],
        'Fall (Oct)': [10]
    }

    for season_name, season_months in seasons.items():
        print(f"\n  {season_name}:")
        season_data = quantile_data[quantile_data['month'].isin(season_months)]

        if len(season_data) == 0:
            print(f"    No data available")
            continue

        # Calculate seasonal baseline
        baseline_data = season_data[season_data['year'] == baseline_year]
        baseline = baseline_data['mean_risk'].mean() if len(baseline_data) > 0 else 0

        print(f"    {'Year':<8} {'Mean Risk':<12} {'vs Baseline':<15}")
        print(f"    {'-'*40}")

        for year in PROJECTION_YEARS:
            year_data = season_data[season_data['year'] == year]
            if len(year_data) == 0:
                continue
            mean_risk = year_data['mean_risk'].mean()
            ratio = mean_risk / baseline if baseline > 0 else np.nan
            pct_change = (ratio - 1) * 100 if not np.isnan(ratio) else np.nan

            print(f"    {year:<8} {mean_risk:<12.6f} {ratio:.2f}x ({pct_change:+.1f}%)")

print("\n" + "="*80)
print("\nCOMPLETE")
print("="*80)
