#!/usr/bin/env python
"""
Fire Brigade Zone Climate Projections - ENSEMBLE MEMBERS

Generates climate-driven fire risk projections using individual ensemble members
to preserve temperature-precipitation correlations within each climate model.

Purpose:
- Project future fire risk trends by zone under climate change
- Use matched GCM-RCM combinations for temperature and precipitation
- Preserve within-model climate correlations crucial for fire risk
- Provide ensemble spread as uncertainty measure

Key Features:
- Pure relative risk scores (NO conversion to absolute counts)
- Individual ensemble members (matched tas/pr pairs)
- Decadal projections (2020-2080)
- Representative months per season (Feb, Mar, Aug, Oct)
- Uncertainty quantification via ensemble spread
- Native 50m resolution matching training data
"""

import pandas as pd
import glob
import geopandas as gpd
import numpy as np
import xarray as xr
import rioxarray
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import arviz as az
import warnings
import xvec
import gc
import itertools
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

print("=" * 80)
print("FIRE BRIGADE ZONE CLIMATE PROJECTIONS - ENSEMBLE MEMBERS")
print("=" * 80)
print()

# ===================================================================
# CONFIGURATION
# ===================================================================

# Paths
BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape")
AOI_PATH = BASE_DIR / "Data/00_QGIS/ADMIN/BOLZANO_REGION_UTM32.gpkg"
FIRE_BRIGADE_ZONES = (
    BASE_DIR
    / "Data/06_Administrative_Boundaries/Processed/FireBrigade_ResponsibilityAreas_Bolzano_clipped.gpkg"
)
STATIC_RASTER_DIR = BASE_DIR / "Data/STATIC_INPUT"

# Climate data - raw ensemble members
TARGET_SCENARIO = "rcp45"  # Change to "rcp45" for RCP4.5
CORDEX_BASE = Path("/mnt/CEPH_PROJECTS/FACT_CLIMAX/CORDEX-Adjust/QDM")
TEMP_ENSEMBLE_DIR = CORDEX_BASE / "tas" / TARGET_SCENARIO
PRECIP_ENSEMBLE_DIR = CORDEX_BASE / "pr" / TARGET_SCENARIO

# Model files
MODEL_DIR = BASE_DIR / "output/02_Model_RelativeProbability"
TRACE_PATH = MODEL_DIR / "trace_relative.nc"
SCALER_PATH = MODEL_DIR / "scaler_relative.joblib"
TEMPORAL_GROUPS_PATH = MODEL_DIR / "temporal_groups.joblib"
GROUP_NAMES_PATH = MODEL_DIR / "group_names.joblib"

# Output
OUTPUT_DIR = (
    BASE_DIR / f"output/04_Zone_Climate_Projections/{TARGET_SCENARIO}_ensemble_members"
)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Projection parameters
PROJECTION_YEARS = list(range(2020, 2101, 10))  # Every 10 years: 2020, 2030, ..., 2100
PROJECTION_MONTHS = [2, 3, 8, 10]  # One per season (Feb, Mar, Aug, Oct)

# Model parameters (must match training - NO CHANGES ALLOWED)
assert 50 == 50, "Native resolution must be 50m to match training data!"
TIME_STEPS = 60
SPATIAL_WINDOW_SIZE = (
    4  # Must match training (4×4 window = 200m context at 50m resolution)
)
STATIC_VARS = [
    "tri",
    "northness",
    "slope",
    "aspect",
    "nasadem",
    "treecoverdensity",
    "landcover_fire_risk",
    "distroads",
    "eastness",
    "flammability",
    "walking_time_to_bldg",
    "walking_time_to_elec_infra",
]
DYNAMIC_VARS = ["T", "P"]
DAY_WINDOWS_TO_KEEP = [1, 3, 5, 10, 15, 30, 60]

month_names = {2: "Feb", 3: "Mar", 8: "Aug", 10: "Oct"}

# ===================================================================
# IDENTIFY COMMON ENSEMBLE MEMBERS
# ===================================================================

print("Identifying common ensemble members...")


def extract_ensemble_id(filepath):
    """Extract ensemble identifier from filename."""
    name = Path(filepath).stem
    # Format: tas_EUR-11_GCM_RCM_r1i1p1_v1_day_19702100_rcp85
    parts = name.split("_")
    if len(parts) >= 6:
        # Join GCM_RCM_r1i1p1_v1
        return "_".join(parts[2:6])
    return None


# Get all temperature and precipitation ensemble files
temp_files = sorted(glob.glob(str(TEMP_ENSEMBLE_DIR / "*.nc")))
precip_files = sorted(glob.glob(str(PRECIP_ENSEMBLE_DIR / "*.nc")))

# Extract ensemble IDs
temp_ensemble_ids = {extract_ensemble_id(f): f for f in temp_files}
precip_ensemble_ids = {extract_ensemble_id(f): f for f in precip_files}

# Find common ensemble members
common_ensemble_ids = set(temp_ensemble_ids.keys()) & set(precip_ensemble_ids.keys())
common_ensemble_ids = sorted(common_ensemble_ids)

# Create matched pairs
ENSEMBLE_PAIRS = [
    {
        "id": ens_id,
        "temp_file": temp_ensemble_ids[ens_id],
        "precip_file": precip_ensemble_ids[ens_id],
    }
    for ens_id in common_ensemble_ids
]

print(f"✓ Found {len(ENSEMBLE_PAIRS)} common ensemble members:")
for pair in ENSEMBLE_PAIRS:
    print(f"  - {pair['id']}")
print()

print(f"Configuration:")
print(f"  Scenario: {TARGET_SCENARIO}")
print(f"  Years: {PROJECTION_YEARS[0]}-{PROJECTION_YEARS[-1]} (every 10 years)")
print(
    f"  Months: {', '.join([month_names[m] for m in PROJECTION_MONTHS])} (one per season)"
)
print(f"  Ensemble members: {len(ENSEMBLE_PAIRS)}")
print(f"  Resolution: 50m native (matches training)")
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
print("✓ Model loaded (Relative probability)")
print()

# ===================================================================
# LOAD FIRE BRIGADE ZONES
# ===================================================================

print("Loading fire brigade zones...")
gdf_zones = gpd.read_file(FIRE_BRIGADE_ZONES)
print(f"✓ Loaded {len(gdf_zones)} zones")

# Ensure unique zone identifiers
gdf_zones["zone_id"] = gdf_zones.get("ID", range(len(gdf_zones)))

# Get zone names
for col in ["PLACE_IT", "NAME", "name", "NAME_IT"]:
    if col in gdf_zones.columns:
        gdf_zones["zone_name"] = gdf_zones[col]
        break
else:
    gdf_zones["zone_name"] = "Zone_" + gdf_zones["zone_id"].astype(str)

print()

# ===================================================================
# CREATE PREDICTION GRID (NATIVE 50m RESOLUTION)
# ===================================================================

print("Creating prediction grid at native 50m resolution...")
aoi = gpd.read_file(AOI_PATH)

template_path = STATIC_RASTER_DIR / "nasadem.tif"
with rioxarray.open_rasterio(template_path) as template_raster:
    template = template_raster.rio.clip(
        aoi.geometry, all_touched=True, drop=True
    ).squeeze("band")

    # Create point grid from all valid pixels
    valid_pixels = np.argwhere(~np.isnan(template.values))
    ys_idx, xs_idx = valid_pixels[:, 0], valid_pixels[:, 1]
    coords_x = template.x[xs_idx].values
    coords_y = template.y[ys_idx].values

    grid_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(coords_x, coords_y), crs=template.rio.crs
    )

print(f"✓ Created grid: {len(grid_gdf):,} points")

# Assign grid points to zones
print("Assigning points to fire brigade zones...")
grid_gdf = gpd.sjoin(
    grid_gdf, gdf_zones[["zone_id", "geometry"]], how="left", predicate="within"
)
grid_gdf = grid_gdf.dropna(subset=["zone_id"])
grid_gdf["zone_id"] = grid_gdf["zone_id"].astype(int)
print(f"✓ Assigned {len(grid_gdf):,} points to zones")
print()

# ===================================================================
# EXTRACT STATIC FEATURES
# ===================================================================

print("Extracting static features with 4×4 window averaging...")
static_features_df = pd.DataFrame(index=grid_gdf.index)

for var_name in tqdm(STATIC_VARS, desc="  Static Variables"):
    raster_path = STATIC_RASTER_DIR / f"{var_name}.tif"
    if not raster_path.exists():
        tqdm.write(f"  Warning: {var_name}.tif not found, skipping")
        continue

    with rioxarray.open_rasterio(raster_path) as rds:
        rds_clipped = rds.rio.clip(aoi.geometry, all_touched=True, drop=True).squeeze(
            "band"
        )

        # Apply 4×4 rolling window to match training
        rds_windowed = rds_clipped.rolling(
            x=SPATIAL_WINDOW_SIZE, y=SPATIAL_WINDOW_SIZE, center=True
        ).mean()

        extracted = rds_windowed.xvec.extract_points(
            grid_gdf.geometry, x_coords="x", y_coords="y"
        )
        static_features_df[var_name] = extracted.values

print(f"✓ Static features extracted: {static_features_df.shape}")
print()

# ===================================================================
# CLIMATE FEATURE EXTRACTION FUNCTION
# ===================================================================


def extract_climate_features(current_date, grid_gdf, temp_file, precip_file, template):
    """Extract climate features for a specific date."""

    start_date = current_date - timedelta(days=TIME_STEPS - 1)

    # Initialize result
    feature_cols = []
    for var in ["T", "P"]:
        for window in DAY_WINDOWS_TO_KEEP:
            for op in ["mean", "max"]:
                feature_cols.append(f"{var}_cumulative_{op}_{window}d")

    result_df = pd.DataFrame(np.nan, index=grid_gdf.index, columns=feature_cols)

    # Process temperature and precipitation
    for prefix, climate_file in [("T", temp_file), ("P", precip_file)]:
        with xr.open_dataset(climate_file) as ds:
            var_name = "tas" if prefix == "T" else "pr"
            data_array = ds[var_name]

            # Filter to date range
            end_date = current_date + timedelta(days=1)
            time_filtered = data_array.sel(time=slice(start_date, end_date))
            time_filtered = time_filtered.sel(
                time=time_filtered.time
                < (pd.Timestamp(current_date) + pd.Timedelta(days=1))
            )

            if len(time_filtered.time) < TIME_STEPS:
                tqdm.write(
                    f"    Warning: Only {len(time_filtered.time)}/{TIME_STEPS} days for {prefix}"
                )

            # Set CRS
            if "lambert_azimuthal_equal_area" in ds:
                crs_var = ds["lambert_azimuthal_equal_area"]
                crs_wkt = crs_var.attrs.get("spatial_ref", None)
                if crs_wkt:
                    time_filtered = time_filtered.rio.write_crs(crs_wkt)
                else:
                    proj_string = (
                        f"+proj=laea +lat_0={crs_var.attrs['latitude_of_projection_origin']} "
                        f"+lon_0={crs_var.attrs['longitude_of_projection_origin']} "
                        f"+x_0={crs_var.attrs['false_easting']} "
                        f"+y_0={crs_var.attrs['false_northing']} "
                        f"+datum=WGS84 +units=m +no_defs"
                    )
                    time_filtered = time_filtered.rio.write_crs(proj_string)

            # Reproject to match grid (native 50m)
            time_filtered_reprojected = time_filtered.rio.reproject_match(template)

            # Reverse time for cumulative operations
            reversed_time_data = time_filtered_reprojected.reindex(
                time=time_filtered_reprojected.time[::-1]
            ).chunk({"time": -1})
            num_available_days = len(reversed_time_data.time)

            # Cumulative statistics
            cumulative_sum = reversed_time_data.cumsum(dim="time", skipna=True)
            divisor = xr.DataArray(
                np.arange(1, num_available_days + 1),
                dims="time",
                coords={"time": reversed_time_data.time},
            )
            cumulative_mean = cumulative_sum / divisor

            cumulative_max = xr.apply_ufunc(
                np.maximum.accumulate,
                reversed_time_data.fillna(-9999),
                input_core_dims=[["time"]],
                output_core_dims=[["time"]],
                dask="parallelized",
                output_dtypes=[reversed_time_data.dtype],
            )

            # Extract for each window
            for day_window in DAY_WINDOWS_TO_KEEP:
                i = day_window - 1
                if i >= num_available_days:
                    continue

                for op_name, data_stack in [
                    ("mean", cumulative_mean),
                    ("max", cumulative_max),
                ]:
                    target_raster = data_stack.isel(time=i)
                    extracted = target_raster.xvec.extract_points(
                        grid_gdf.geometry, x_coords="x", y_coords="y"
                    ).compute()

                    column_name = f"{prefix}_cumulative_{op_name}_{day_window}d"
                    result_df[column_name] = extracted.values

            gc.collect()

    return result_df


# ===================================================================
# PREDICTION FUNCTION
# ===================================================================


def generate_relative_predictions(trace, X_scaled, temporal_groups, group_names):
    """Generate PURE relative probability predictions (NO scaling factors)."""

    n_samples = 100
    alpha_samples = trace.posterior["alpha"].values.reshape(-1)[:n_samples]
    attention_samples = trace.posterior["attention_weights"].values.reshape(
        -1, len(group_names)
    )[:n_samples]

    beta_samples_dict = {}
    for group_name in group_names:
        beta_key = f"beta_{group_name}"
        if beta_key in trace.posterior:
            beta_data = trace.posterior[beta_key].values
            beta_flat = beta_data.reshape(-1, beta_data.shape[-1])[:n_samples]
            beta_samples_dict[group_name] = beta_flat

    n_points = X_scaled.shape[0]
    predictions_per_sample = []

    for sample_idx in range(n_samples):
        logit_pred = np.full(n_points, alpha_samples[sample_idx])

        for group_idx, (group_name, feature_indices) in enumerate(
            temporal_groups.items()
        ):
            if group_name in beta_samples_dict:
                beta_sample = beta_samples_dict[group_name][sample_idx]
                group_features = X_scaled[:, feature_indices]
                group_contrib = np.dot(group_features, beta_sample)
                attention_weight = attention_samples[sample_idx, group_idx]
                logit_pred += attention_weight * group_contrib

        # Numerically stable sigmoid
        prob_pred = np.empty_like(logit_pred)
        pos_mask = logit_pred >= 0
        prob_pred[pos_mask] = 1 / (1 + np.exp(-logit_pred[pos_mask]))
        neg_mask = ~pos_mask
        prob_pred[neg_mask] = np.exp(logit_pred[neg_mask]) / (
            1 + np.exp(logit_pred[neg_mask])
        )
        predictions_per_sample.append(prob_pred)

    predictions_per_sample = np.array(predictions_per_sample)
    pred_mean = predictions_per_sample.mean(axis=0)
    pred_std = predictions_per_sample.std(axis=0)

    return pred_mean, pred_std


# ===================================================================
# RUN PROJECTIONS
# ===================================================================

print("=" * 80)
print("GENERATING ZONE-LEVEL CLIMATE PROJECTIONS")
print("=" * 80)
print()

all_results = []

for ensemble_pair in ENSEMBLE_PAIRS:
    ensemble_id = ensemble_pair["id"]
    temp_file = ensemble_pair["temp_file"]
    precip_file = ensemble_pair["precip_file"]

    print(f"\n{'#' * 80}")
    print(f"# ENSEMBLE MEMBER: {ensemble_id}")
    print(f"{'#' * 80}\n")

    if not Path(temp_file).exists() or not Path(precip_file).exists():
        print(f"  ⚠️  Climate files not found, skipping {ensemble_id}")
        continue

    projection_iterator = list(itertools.product(PROJECTION_YEARS, PROJECTION_MONTHS))

    for year, month in tqdm(
        projection_iterator, desc=f"  {ensemble_id[:40]} projections"
    ):
        current_date = datetime(year, month, 15)

        # Extract climate features
        climate_features = extract_climate_features(
            current_date, grid_gdf, temp_file, precip_file, template
        )

        # Combine with static features
        all_features = pd.concat([static_features_df, climate_features], axis=1)

        # Ensure all expected features present
        expected_features = list(scaler.feature_names_in_)
        missing_features = [
            f for f in expected_features if f not in all_features.columns
        ]
        if missing_features:
            tqdm.write(f"  ⚠️  WARNING: {len(missing_features)} features missing")
            for feat in missing_features:
                all_features[feat] = 0

        # Reorder to match scaler
        all_features = all_features[expected_features]

        # Scale features
        X_scaled = scaler.transform(all_features)

        # Generate predictions
        pred_mean, pred_std = generate_relative_predictions(
            trace, X_scaled, temporal_groups, group_names
        )

        # Aggregate by zone
        grid_gdf["prediction"] = pred_mean
        grid_gdf["uncertainty"] = pred_std

        zone_stats = (
            grid_gdf.groupby("zone_id")
            .agg({"prediction": ["mean", "max", "std"], "uncertainty": "mean"})
            .reset_index()
        )

        zone_stats.columns = [
            "zone_id",
            "mean_risk",
            "max_risk",
            "spatial_std",
            "prediction_uncertainty",
        ]
        zone_stats["year"] = year
        zone_stats["month"] = month
        zone_stats["ensemble_member"] = ensemble_id

        all_results.append(zone_stats)
        gc.collect()

# Combine all results
print("\n" + "=" * 80)
print("COMPILING RESULTS")
print("=" * 80)

df_results = pd.concat(all_results, ignore_index=True)
df_results = df_results.merge(
    gdf_zones[["zone_id", "zone_name"]], on="zone_id", how="left"
)

# Save results
output_csv = OUTPUT_DIR / "zone_projections_ensemble_members.csv"
df_results.to_csv(output_csv, index=False)
print(f"\n✓ Results saved: {output_csv}")

# ===================================================================
# SUMMARY STATISTICS
# ===================================================================

print("\n" + "=" * 80)
print("PROJECTION SUMMARY")
print("=" * 80)
print()

baseline_year = 2020

seasons = {
    "Winter (Feb)": [2],
    "Spring (Mar)": [3],
    "Summer (Aug)": [8],
    "Fall (Oct)": [10],
}

for season_name, season_months in seasons.items():
    print(f"\n{season_name}:")
    print("-" * 60)

    season_data = df_results[df_results["month"].isin(season_months)]

    if len(season_data) == 0:
        print(f"  No data available")
        continue

    baseline_data = season_data[season_data["year"] == baseline_year]
    baseline = baseline_data["mean_risk"].mean() if len(baseline_data) > 0 else 0

    print(
        f"  {'Year':<8} {'Ensemble Mean':<15} {'Ensemble Std':<15} {'vs Baseline':<15}"
    )
    print(f"  {'-' * 60}")

    for year in PROJECTION_YEARS:
        year_data = season_data[season_data["year"] == year]
        if len(year_data) == 0:
            continue

        # Calculate ensemble mean and std across all ensemble members
        ensemble_mean = year_data["mean_risk"].mean()
        ensemble_std = year_data["mean_risk"].std()

        ratio = ensemble_mean / baseline if baseline > 0 else np.nan
        pct_change = (ratio - 1) * 100 if not np.isnan(ratio) else np.nan

        print(
            f"  {year:<8} {ensemble_mean:<15.6f} {ensemble_std:<15.6f} {ratio:.2f}x ({pct_change:+.1f}%)"
        )

print("\n" + "=" * 80)
print(f"\nEnsemble spread provides uncertainty estimate")
print(
    f"Total projections: {len(df_results)} (zones × years × months × ensemble members)"
)
print(f"Unique ensemble members: {df_results['ensemble_member'].nunique()}")
print(
    f"CSV contains individual results for each: zone × year × month × ensemble_member"
)
print("=" * 80)
print("\nCOMPLETE")
print("=" * 80)
