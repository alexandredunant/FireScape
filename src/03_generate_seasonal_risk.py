#!/usr/bin/env python
"""
GENERATE SEASONAL FIRE RISK MAPS FROM DAILY PREDICTIONS - 250m Resolution

This script implements an alternative approach to seasonal risk assessment:
1. Load multi-year meteorological data using xr.open_mfdataset() (2015-2024)
2. Apply the trained EBM model to every available day
3. Calculate seasonal means FROM daily predictions

Key difference from climatology approach:
- Climatology: Calculate seasonal means → Make 1 prediction per season
- This script: Make daily predictions → Calculate seasonal means from predictions

Outputs match the format from generate_seasonal_risk_maps_simplified.py
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import joblib
import xarray as xr
import rioxarray as rxr
from pathlib import Path
import warnings
import geopandas as gpd
from exactextract import exact_extract
import rasterio
from rasterio.io import MemoryFile
from affine import Affine
from tqdm import tqdm
import gc

warnings.filterwarnings("ignore")

# ===================================================================
# CONFIGURATION
# ===================================================================

BASE_DIR = Path(os.environ.get("FIRESCAPE_ROOT", Path(__file__).resolve().parents[1]))
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output/03_Seasonal_Risk"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_RASTER_DIR = BASE_DIR / "Data/STATIC_INPUT_250m"
TEMP_DIR = BASE_DIR / "Data/05_Meteorological_Data/Temperature"
PRECIP_DIR = BASE_DIR / "Data/05_Meteorological_Data/Precipitation"
SPEI30_DIR = BASE_DIR / "Data/05_Meteorological_Data/SPEI30_Standardized"
SPEI90_DIR = BASE_DIR / "Data/05_Meteorological_Data/SPEI90_Standardized"
LIGHTNING_DIR = BASE_DIR / "Data/05_Meteorological_Data/Lightning_Standardized"
FIREBRIGADE_PATH = (
    BASE_DIR
    / "Data/06_Administrative_Boundaries/Processed/FireBrigade_ResponsibilityAreas_Bolzano_clipped.gpkg"
)

# Model paths (using WITHOUT lightning model - baseline)
# Model paths (models stored in centralized location)
MODEL_DIR = BASE_DIR / "output/02_Model_Training/EBM_SPEI"
MODEL_PATH = MODEL_DIR / "ebm_model_without_lightning_250m.joblib"
FEATURES_PATH = MODEL_DIR / "feature_names_without_lightning_250m.joblib"
OUTPUT_SUFFIX = "_daily_baseline_250m"
YEARS = range(2015, 2025)  # Report prediction period: 2015-2024

# Parallelization settings
N_JOBS = 1  # Use all available cores (-1), or set to specific number (e.g., 4, 8)
BATCH_SIZE = 100  # Process timesteps in batches to manage memory

# Season definitions
SEASONS = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11],
}

print("=" * 80)
print("DAILY PREDICTION-BASED SEASONAL FIRE RISK ASSESSMENT")
print("=" * 80)
print(f"Model: {MODEL_PATH.name}")
print(f"Period: {min(YEARS)}-{max(YEARS)}")
print(f"Output: {OUTPUT_DIR}")
print()

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================


def get_season(month):
    """Get season name from month number."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:  # 9, 10, 11
        return "Autumn"


def calculate_sin_doy(time_coord):
    """Calculate sine of day-of-year for seasonality encoding."""
    doy = time_coord.dt.dayofyear.values
    return np.sin(2 * np.pi * doy / 365.25)


def calculate_cos_doy(time_coord):
    """Calculate cosine of day-of-year for seasonality encoding."""
    doy = time_coord.dt.dayofyear.values
    return np.cos(2 * np.pi * doy / 365.25)


def load_static_feature(feature_name, template):
    """
    Load a single static feature raster and align to template.

    Args:
        feature_name: Name of the feature file (without extension)
        template: Template raster for spatial alignment

    Returns:
        2D numpy array aligned to template
    """
    raster_path = STATIC_RASTER_DIR / f"{feature_name}.tif"

    if not raster_path.exists():
        print(f"Warning: {raster_path} not found, using zeros")
        return np.zeros_like(template)

    # Load and reproject to match template
    raster = rxr.open_rasterio(raster_path, masked=True).squeeze()

    # Ensure CRS is set
    if raster.rio.crs is None:
        raster = raster.rio.write_crs(template.rio.crs)

    # Reproject to match template
    raster_aligned = raster.rio.reproject_match(template)

    # Fill NaNs with nearest neighbor
    data = raster_aligned.values
    if np.isnan(data).any():
        from scipy.interpolate import griddata

        valid_mask = ~np.isnan(data)
        if valid_mask.any():
            y, x = np.where(valid_mask)
            values = data[valid_mask]
            yi, xi = np.where(np.isnan(data))
            data[yi, xi] = griddata((y, x), values, (yi, xi), method="nearest")

    return data


def load_multiyear_variable(file_pattern, var_name, years=YEARS):
    """
    Load all years of a variable using open_mfdataset.

    Args:
        file_pattern: Pattern for files (e.g., "tmean_{year}.nc")
        var_name: Variable name in the NetCDF files
        years: Range of years to load

    Returns:
        xarray.DataArray with all years concatenated
    """
    # Build list of file paths
    files = [str(Path(file_pattern.format(year=year))) for year in years]

    # Filter to existing files
    existing_files = [f for f in files if Path(f).exists()]

    if not existing_files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

    print(f"  Loading {len(existing_files)} files for {var_name}...")

    # Load with dask (lazy loading)
    ds = xr.open_mfdataset(
        existing_files,
        combine="by_coords",
        decode_times=True,
        chunks={"time": 30},  # 30-day chunks for memory efficiency
        parallel=False,  # Avoid issues with parallel loading
    )

    return ds[var_name]


# ===================================================================
# MAIN PROCESSING
# ===================================================================

print("Step 1: Loading EBM model and feature names...")
ebm = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
print(f"  Model loaded: {len(feature_names)} features")
print(f"  Features: {', '.join(feature_names)}")
print()

print("Step 2: Loading static features (11 rasters)...")
# Use DEM as template for spatial alignment
template_path = STATIC_RASTER_DIR / "nasadem.tif"
template = rxr.open_rasterio(template_path, masked=True).squeeze()
print(f"  Template shape: {template.shape}")
print(f"  Template CRS: {template.rio.crs}")

# Load all static features
static_features = {}
static_feature_names = [
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

for feat in static_feature_names:
    static_features[feat] = load_static_feature(feat, template)
    print(f"  ✓ Loaded {feat}")

print()

print("Step 3: Loading multi-year dynamic data...")
# Load all dynamic variables across all years (lazy with dask)
temp_da = load_multiyear_variable(str(TEMP_DIR / "tmean_{year}.nc"), "tmean")
precip_da = load_multiyear_variable(str(PRECIP_DIR / "prec_{year}.nc"), "prec")
spei30_da = load_multiyear_variable(str(SPEI30_DIR / "spei30_{year}.nc"), "spei30")
spei90_da = load_multiyear_variable(str(SPEI90_DIR / "spei90_{year}.nc"), "spei90")

# Only load lightning if it's in the feature names
if "lightning_density" in feature_names:
    lightning_da = load_multiyear_variable(
        str(LIGHTNING_DIR / "lightning_density_{year}.nc"), "lightning_density"
    )
    print(f"  ✓ Lightning: {len(lightning_da.time)} timesteps")
else:
    lightning_da = None
    print(f"  ✓ Lightning: Not used (baseline model)")

print(f"  ✓ Temperature: {len(temp_da.time)} timesteps")
print(f"  ✓ Precipitation: {len(precip_da.time)} timesteps")
print(f"  ✓ SPEI-30: {len(spei30_da.time)} timesteps")
print(f"  ✓ SPEI-90: {len(spei90_da.time)} timesteps")
print()

print("Step 4: Applying model to daily data...")
n_times = len(temp_da.time)
ny, nx = temp_da.shape[1:]

print(f"  Grid dimensions: {ny} × {nx}")
print(f"  Number of timesteps: {n_times}")
print(f"  Total predictions: {ny * nx * n_times:,}")
print()

# ===================================================================
# PARALLEL PREDICTION FUNCTION
# ===================================================================


def predict_timestep(
    t,
    temp_da,
    precip_da,
    spei30_da,
    spei90_da,
    lightning_da,
    static_features,
    feature_names,
    ebm_model,
    ny,
    nx,
):
    """
    Process a single timestep and return predictions.
    This function is called in parallel for each timestep.
    """
    try:
        # Extract all features for this timestep
        # Get day of year for seasonality encoding
        time_val = temp_da.time[t].values
        time_pd = pd.to_datetime(time_val)
        doy = time_pd.dayofyear
        sin_doy_val = np.sin(2 * np.pi * doy / 365.25)
        cos_doy_val = np.cos(2 * np.pi * doy / 365.25)

        # Build feature matrix
        feature_grids = []

        for feat in feature_names:
            if feat in static_features:
                # Static feature - same for all timesteps
                feature_grids.append(static_features[feat])
            elif feat == "T_daily":
                feature_grids.append(temp_da.isel(time=t).values)
            elif feat == "P_daily":
                feature_grids.append(precip_da.isel(time=t).values)
            elif feat == "lightning_density":
                if lightning_da is not None:
                    feature_grids.append(lightning_da.isel(time=t).values)
                else:
                    feature_grids.append(np.zeros((ny, nx)))
            elif feat == "SPEI_30d":
                feature_grids.append(spei30_da.isel(time=t).values)
            elif feat == "SPEI_90d":
                feature_grids.append(spei90_da.isel(time=t).values)
            elif feat == "sin_doy":
                feature_grids.append(np.full((ny, nx), sin_doy_val))
            elif feat == "cos_doy":
                feature_grids.append(np.full((ny, nx), cos_doy_val))
            else:
                # Unknown feature
                feature_grids.append(np.zeros((ny, nx)))

        # Stack into (ny, nx, n_features)
        X_2d = np.stack(feature_grids, axis=-1)

        # Reshape to (n_pixels, n_features)
        X_flat = X_2d.reshape(-1, len(feature_names))

        # Check for NaNs
        if np.isnan(X_flat).any():
            X_flat = np.nan_to_num(X_flat, nan=0.0)

        # Predict
        risk_flat = ebm_model.predict_proba(X_flat)[:, 1]

        # Reshape back to (ny, nx)
        return risk_flat.reshape(ny, nx)

    except Exception as e:
        print(f"\n  Warning: Error at timestep {t}: {e}")
        return np.full((ny, nx), np.nan)


# ===================================================================
# PARALLEL PROCESSING
# ===================================================================

print("  Processing timesteps in parallel...")
print(f"  Using {N_JOBS if N_JOBS > 0 else 'all available'} cores")

from joblib import Parallel, delayed

# Process all timesteps in parallel
predictions_list = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(predict_timestep)(
        t,
        temp_da,
        precip_da,
        spei30_da,
        spei90_da,
        lightning_da,
        static_features,
        feature_names,
        ebm,
        ny,
        nx,
    )
    for t in range(n_times)
)

# Stack results into 3D array
predictions = np.stack(predictions_list, axis=0)

print(f"  ✓ Completed {n_times} daily predictions")
print()

# Old sequential code removed - now using parallel processing above

print("Step 5: Creating xarray DataArray with predictions...")
predictions_da = xr.DataArray(
    predictions,
    coords={
        "time": temp_da.time,
        "y": temp_da.y,
        "x": temp_da.x,
    },
    dims=["time", "y", "x"],
    name="fire_susceptibility",
    attrs={
        "long_name": "Daily relative wildfire susceptibility score",
        "units": "dimensionless score",
        "model": "EBM baseline without lightning (250m)",
        "description": (
            "Relative EBM score from 0 to 1; not an absolute calibrated "
            "probability because the model was fitted to sampled pseudo-absences"
        ),
    },
)

print(f"  Predictions shape: {predictions_da.shape}")
print(
    f"  Predictions range: [{float(predictions_da.min()):.4f}, {float(predictions_da.max()):.4f}]"
)
print()

# Optional: Save daily predictions to NetCDF for reuse
save_daily = False  # Set to True to save daily predictions
if save_daily:
    print("  Saving daily predictions to NetCDF...")
    daily_output_path = OUTPUT_DIR / "daily_predictions_2015_2024.nc"
    predictions_da.to_netcdf(daily_output_path)
    print(f"  ✓ Saved to {daily_output_path}")
    print()

print("Step 6: Seasonal aggregation...")
# Add season coordinate
seasons_list = [get_season(pd.to_datetime(t).month) for t in predictions_da.time.values]
predictions_da = predictions_da.assign_coords(season=("time", seasons_list))

# Group by season and calculate mean
seasonal_risk = predictions_da.groupby("season").mean(dim="time")
seasonal_std = predictions_da.groupby("season").std(dim="time")

print(f"  ✓ Seasonal risk shape: {seasonal_risk.shape}")
print(f"  ✓ Seasons: {list(seasonal_risk.season.values)}")
print()

# Print seasonal statistics
for season in ["Winter", "Spring", "Summer", "Autumn"]:
    if season in seasonal_risk.season.values:
        risk_vals = seasonal_risk.sel(season=season).values
        std_vals = seasonal_std.sel(season=season).values
        print(
            f"  {season:8s}: Mean={np.nanmean(risk_vals):.4f}, Std={np.nanmean(std_vals):.4f}"
        )

print()

print("Step 7: Creating seasonal risk maps...")

# Load Bolzano region boundary for extent clipping
BOLZANO_BOUNDARY_PATH = BASE_DIR / "Data/00_QGIS/ADMIN/BOLZANO_REGION_UTM32.gpkg"
bolzano_boundary = gpd.read_file(BOLZANO_BOUNDARY_PATH)

# Ensure CRS match
if bolzano_boundary.crs != template.rio.crs:
    bolzano_boundary = bolzano_boundary.to_crs(template.rio.crs)

print(f"  Loaded Bolzano boundary (CRS: {bolzano_boundary.crs})")

# Get extent from boundary
bounds = bolzano_boundary.total_bounds  # [minx, miny, maxx, maxy]
print(f"  Bolzano extent: {bounds}")

# Create mask from boundary using rasterio features
from rasterio import features
import numpy.ma as ma

# Create transform for the raster
transform = template.rio.transform()

# Rasterize the boundary to create a mask
mask = features.geometry_mask(
    bolzano_boundary.geometry,
    out_shape=(ny, nx),
    transform=transform,
    invert=True,  # True inside boundary, False outside
)

print(f"  Created boundary mask: {mask.sum()} pixels inside region")

# Create 4-panel figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

season_order = ["Winter", "Spring", "Summer", "Autumn"]
vmin, vmax = 0, float(seasonal_risk.max())

for idx, season in enumerate(season_order):
    ax = axes[idx]

    if season in seasonal_risk.season.values:
        risk_grid = seasonal_risk.sel(season=season).values

        # Apply mask - set values outside Bolzano to NaN
        risk_grid_masked = np.where(mask, risk_grid, np.nan)

        # Calculate extent in data coordinates for imshow
        # Get pixel coordinates corresponding to Bolzano bounds
        from rasterio.transform import rowcol

        row_min, col_min = rowcol(transform, bounds[0], bounds[3])  # top-left
        row_max, col_max = rowcol(transform, bounds[2], bounds[1])  # bottom-right

        # Clip to raster bounds
        row_min = max(0, row_min)
        col_min = max(0, col_min)
        row_max = min(ny, row_max)
        col_max = min(nx, col_max)

        # Extract subset
        risk_subset = risk_grid_masked[row_min:row_max, col_min:col_max]

        # Plot with correct extent
        im = ax.imshow(
            risk_subset,
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            extent=[
                bounds[0],
                bounds[2],
                bounds[1],
                bounds[3],
            ],  # [left, right, bottom, top]
        )

        # Add boundary outline
        bolzano_boundary.boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=10)

        ax.set_title(f"{season}", fontsize=14, fontweight="bold")
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect("equal")
        ax.axis("off")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Fire Risk Probability", fontsize=10)

plt.suptitle(
    f"Seasonal Fire Risk Maps (Daily Prediction Approach)\n"
    f"Mean of {n_times} daily predictions ({min(YEARS)}-{max(YEARS)})",
    fontsize=16,
    fontweight="bold",
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_map_path = OUTPUT_DIR / f"seasonal_risk_maps{OUTPUT_SUFFIX}.png"
plt.savefig(output_map_path, dpi=300, bbox_inches="tight")
print(f"  ✓ Saved map to {output_map_path}")
plt.close()
print()

print("Step 8: Extracting zonal statistics per fire brigade...")
# Load fire brigade zones
fire_brigades = gpd.read_file(FIREBRIGADE_PATH)
print(f"  Loaded {len(fire_brigades)} fire brigade zones")

# Ensure CRS match
if fire_brigades.crs != template.rio.crs:
    fire_brigades = fire_brigades.to_crs(template.rio.crs)

# Extract zonal statistics for each season
zone_stats = []

for season in season_order:
    if season not in seasonal_risk.season.values:
        continue

    risk_grid = seasonal_risk.sel(season=season).values

    # Create in-memory raster
    transform = template.rio.transform()

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=ny,
            width=nx,
            count=1,
            dtype=risk_grid.dtype,
            crs=template.rio.crs,
            transform=transform,
        ) as dataset:
            dataset.write(risk_grid, 1)

        with memfile.open() as dataset:
            # Extract zonal statistics - pass GeoDataFrame directly
            stats = exact_extract(
                dataset,
                fire_brigades,  # Pass GeoDataFrame, not just geometry
                ["mean"],
                output="pandas",
            )

            # Add season and zone info
            stats["season"] = season
            stats["zone_id"] = fire_brigades.index.values

            # Add zone name if available
            if "name" in fire_brigades.columns:
                stats["name"] = fire_brigades["name"].values

            zone_stats.append(stats)

# Combine all seasons
zone_stats_df = pd.concat(zone_stats, ignore_index=True)
zone_stats_df = zone_stats_df.rename(columns={"mean": "mean_risk"})

# Save to CSV
output_csv_path = OUTPUT_DIR / f"seasonal_risk_by_firebrigade{OUTPUT_SUFFIX}.csv"
zone_stats_df.to_csv(output_csv_path, index=False)
print(f"  ✓ Saved zonal statistics to {output_csv_path}")
print()

print("=" * 80)
print("Step 8: Create GeoPackage with Seasonal Risk Columns...")
print("=" * 80)

# Load fire brigade zones
FIREBRIGADE_GPKG = (
    BASE_DIR
    / "Data/06_Administrative_Boundaries/Processed/FireBrigade_ResponsibilityAreas_Bolzano_clipped.gpkg"
)
gd = gpd.read_file(FIREBRIGADE_GPKG)

# Create zone_id if not present
if "zone_id" not in gd.columns:
    gd["zone_id"] = gd.index

# Pivot the risk data (Long to Wide)
d_pivot = zone_stats_df.pivot(index="zone_id", columns="season", values="mean_risk")
d_pivot.columns = [f"risk_{col.lower()}" for col in d_pivot.columns]

# Merge with GeoDataFrame
ngd = gd.merge(d_pivot, left_on="zone_id", right_index=True, how="left")

# Calculate regional means
regional_means = d_pivot.mean()

# Area-weighted mean
weighted_means = {}
for col in d_pivot.columns:
    valid_mask = ngd[col].notna() & ngd["AREA"].notna()
    if valid_mask.sum() > 0:
        subset = ngd[valid_mask]
        weighted_means[col] = (subset[col] * subset["AREA"]).sum() / subset[
            "AREA"
        ].sum()

print("\n--- Regional Risk Means (Simple) ---")
print(regional_means)
print("\n--- Regional Risk Means (Area-Weighted) ---")
for season, val in weighted_means.items():
    print(f"{season}: {val:.6f}")

# Export to GeoPackage
output_gpkg = OUTPUT_DIR / f"FireBrigade_Risk_Seasonal{OUTPUT_SUFFIX}.gpkg"
ngd.to_file(output_gpkg, driver="GPKG")
print(f"\n✓ Exported spatial dataset to: {output_gpkg}")
print(f"  Zones: {len(ngd)}")
print()

print("=" * 80)
print("Step 9: Validation - Predicted vs Actual Fires...")
print("=" * 80)

# Load actual fire data
try:
    FIRE_TRAINING_DATA = (
        BASE_DIR / "output/01_Training_Data/training_features_ebm_250m.parquet"
    )

    if FIRE_TRAINING_DATA.exists():
        print(f"Loading fire training data from: {FIRE_TRAINING_DATA}")
        fire_df = pd.read_parquet(FIRE_TRAINING_DATA)

        # Extract year, month, and fire occurrence
        if "fire" in fire_df.columns and "date" in fire_df.columns:
            fire_df["date"] = pd.to_datetime(fire_df["date"])
            fire_df["year"] = fire_df["date"].dt.year
            fire_df["month"] = fire_df["date"].dt.month

            # Map months to seasons
            def map_season(month):
                if month in [12, 1, 2]:
                    return "Winter"
                elif month in [3, 4, 5]:
                    return "Spring"
                elif month in [6, 7, 8]:
                    return "Summer"
                else:
                    return "Autumn"

            fire_df["season"] = fire_df["month"].apply(map_season)

            # Calculate actual fire frequency by season
            actual_fires = (
                fire_df.groupby("season")["fire"]
                .agg(["sum", "count", "mean"])
                .reset_index()
            )
            actual_fires.columns = [
                "season",
                "total_fires",
                "total_obs",
                "fire_frequency",
            ]

            # Get predicted risk by season (simple mean across zones)
            predicted_risk = (
                zone_stats_df.groupby("season")["mean_risk"].mean().reset_index()
            )
            predicted_risk.columns = ["season", "predicted_risk"]

            # Merge
            validation_df = actual_fires.merge(predicted_risk, on="season")

            print(
                "\n--- Seasonal Validation: Predicted Risk vs Actual Fire Frequency ---"
            )
            print(validation_df.to_string(index=False))

            # Calculate correlation
            correlation = (
                validation_df[["predicted_risk", "fire_frequency"]].corr().iloc[0, 1]
            )
            print(
                f"\nCorrelation (predicted risk vs actual fire frequency): {correlation:.3f}"
            )

            # Create validation plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Timeline plot
            ax = axes[0]
            x = range(len(validation_df))
            ax.plot(
                x,
                validation_df["predicted_risk"],
                "o-",
                label="Predicted Risk",
                linewidth=2,
                markersize=8,
            )
            ax.plot(
                x,
                validation_df["fire_frequency"],
                "s-",
                label="Actual Fire Frequency",
                linewidth=2,
                markersize=8,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(validation_df["season"], rotation=45)
            ax.set_xlabel("Season", fontsize=12)
            ax.set_ylabel("Risk / Frequency", fontsize=12)
            ax.set_title(
                "Predicted Risk vs Actual Fire Frequency by Season",
                fontsize=14,
                fontweight="bold",
            )
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Scatter plot
            ax = axes[1]
            ax.scatter(
                validation_df["fire_frequency"],
                validation_df["predicted_risk"],
                s=200,
                alpha=0.6,
            )

            # Add season labels
            for _, row in validation_df.iterrows():
                ax.annotate(
                    row["season"],
                    (row["fire_frequency"], row["predicted_risk"]),
                    fontsize=10,
                    ha="center",
                    va="bottom",
                )

            # Add 1:1 reference line (scaled)
            min_val = min(
                validation_df["fire_frequency"].min(),
                validation_df["predicted_risk"].min(),
            )
            max_val = max(
                validation_df["fire_frequency"].max(),
                validation_df["predicted_risk"].max(),
            )

            ax.set_xlabel("Actual Fire Frequency", fontsize=12)
            ax.set_ylabel("Predicted Risk", fontsize=12)
            ax.set_title(
                f"Predicted vs Actual (r={correlation:.3f})",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save
            validation_plot = (
                OUTPUT_DIR / f"validation_predicted_vs_actual{OUTPUT_SUFFIX}.png"
            )
            plt.savefig(validation_plot, dpi=300, bbox_inches="tight")
            print(f"\n✓ Saved validation plot to: {validation_plot}")

            # Save validation data
            validation_csv = (
                OUTPUT_DIR / f"validation_predicted_vs_actual{OUTPUT_SUFFIX}.csv"
            )
            validation_df.to_csv(validation_csv, index=False)
            print(f"✓ Saved validation data to: {validation_csv}")

        else:
            print("  Fire training data does not have expected columns (fire, date)")
    else:
        print(f"  Fire training data not found at: {FIRE_TRAINING_DATA}")

except Exception as e:
    print(f"  Warning: Could not perform validation: {e}")

print()

print("=" * 80)
print("PROCESSING COMPLETE")
print("=" * 80)
print(f"Total timesteps processed: {n_times}")
print(f"Total predictions: {ny * nx * n_times:,}")
print(f"Outputs:")
print(f"  - Maps: {output_map_path}")
print(f"  - Zone stats CSV: {output_csv_path}")
print(f"  - GeoPackage: {output_gpkg}")
if "validation_plot" in locals():
    print(f"  - Validation plot: {validation_plot}")
    print(f"  - Validation data: {validation_csv}")
print()
