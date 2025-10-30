
# LANDCOVER FIRE RISK MAPPING:
# This script uses pre-transformed landcover_fire_risk.tif (ordinal values 0-5)
# The transformation is done ONCE by:
#   Scripts/01_Data_Preparation/create_transformed_landcover.py
#
# For the canonical Corine → Fire Risk mapping dictionary, see that script.
# Values: 0=No risk, 1=Very low, 2=Low, 3=Moderate, 4=High, 5=Very high

"""
Feature Extraction for Climate Projections

Extracts features for future climate scenarios, properly handling:
1. Mid-month dates with temporal context (not single-day snapshots)
2. Categorical landcover data
3. Multiple spatial locations across Bolzano Province
"""

import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
from datetime import datetime, timedelta
from scipy import stats
from typing import Dict, List, Tuple
import glob
from tqdm import tqdm

# Import scenario configuration
from config_scenarios import (
    ClimateScenario, SCENARIOS, REGIONS, TIME_AGGREGATION_WINDOWS,
    get_projection_dates, get_scenario_output_dir
)


# Static raster directory
STATIC_RASTER_DIR = "/mnt/CEPH_PROJECTS/Firescape/Data/STATIC_INPUT/"

# Feature definitions
STATIC_VARS = [
    'tri', 'northness', 'slope', 'aspect', 'nasadem',
    'treecoverdensity', 'landcover_fire_risk', 'distroads',
    'eastness', 'flammability', 'walking_time_to_bldg',
    'walking_time_to_elec_infra'
]
DYNAMIC_VARS = ['T', 'P']
TIME_STEPS = 60
TARGET_CRS = "EPSG:32632"


def create_spatial_grid(bounds=None, resolution=1000):
    """
    Create a regular grid of prediction points across Bolzano Province.

    Args:
        bounds: (xmin, ymin, xmax, ymax) in EPSG:4326, or None for full province
        resolution: Grid spacing in meters (UTM coordinates)

    Returns:
        GeoDataFrame with grid points
    """
    print(f"\n📍 Creating spatial grid (resolution: {resolution}m)...")

    # Load province boundary (assuming you have this)
    # If not available, use bounding box from static rasters
    template_path = os.path.join(STATIC_RASTER_DIR, "nasadem.tif")

    with rioxarray.open_rasterio(template_path) as template:
        if bounds is None:
            # Use full raster extent
            xmin, ymin, xmax, ymax = template.rio.bounds()
        else:
            # Convert lat/lon bounds to UTM
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
            xmin, ymin = transformer.transform(bounds[0], bounds[1])
            xmax, ymax = transformer.transform(bounds[2], bounds[3])

    # Create grid
    x_coords = np.arange(xmin, xmax, resolution)
    y_coords = np.arange(ymin, ymax, resolution)
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Flatten to points
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()),
        crs=TARGET_CRS
    )
    points['grid_id'] = np.arange(len(points))

    print(f"   ✓ Generated {len(points)} grid points")
    return points


def extract_static_features_at_point(point_geom, window_size=4):
    """
    Extract static features at a single point.

    Args:
        point_geom: Shapely Point geometry (in UTM)
        window_size: Spatial window size (pixels)

    Returns:
        Dictionary of static feature values
    """
    features = {}

    # Load template to get coordinates
    template_path = os.path.join(STATIC_RASTER_DIR, "nasadem.tif")

    with rioxarray.open_rasterio(template_path) as template_raster:
        # Find pixel coordinates
        iy = np.abs(template_raster.y - point_geom.y).argmin().item()
        ix = np.abs(template_raster.x - point_geom.x).argmin().item()

        half_window = window_size // 2
        y_slice = slice(max(0, iy - half_window), min(template_raster.y.size, iy + half_window))
        x_slice = slice(max(0, ix - half_window), min(template_raster.x.size, ix + half_window))

    # Extract each static variable
    for var_name in STATIC_VARS:
        raster_path = os.path.join(STATIC_RASTER_DIR, f"{var_name}.tif")

        try:
            with rioxarray.open_rasterio(raster_path) as rds:
                window_data = rds.isel(y=y_slice, x=x_slice, band=0)

                # Categorical handling for landcover
                if var_name == 'landcover_fire_risk':
                    values = window_data.values.flatten()
                    values = values[~np.isnan(values)]
                    if len(values) > 0:
                        mode_result = stats.mode(values, keepdims=False)
                        # landcover_fire_risk.tif already contains ordinal values (0-5)
                        # No transformation needed - use mode directly
                        features['landcover_fire_risk'] = float(mode_result.mode)
                    else:
                        features[var_name] = 0
                else:
                    # Continuous variables: use mean
                    features[var_name] = float(window_data.mean().item())

        except Exception as e:
            print(f"   ⚠️  Error extracting {var_name}: {e}")
            features[var_name] = np.nan

    return features


def load_dynamic_data_for_period(
    scenario: ClimateScenario,
    center_date: datetime,
    n_days_back: int = 60
) -> Dict[str, xr.DataArray]:
    """
    Load temperature and precipitation data for a time window.

    Args:
        scenario: Climate scenario configuration
        center_date: Center date for extraction (e.g., 2050-07-15)
        n_days_back: Number of days to load before center_date

    Returns:
        Dictionary with 'T' and 'P' DataArrays (time x y x x)
    """
    data = {}

    for var_prefix, data_dir in [('T', scenario.temp_dir), ('P', scenario.precip_dir)]:
        # Determine date range
        start_date = center_date - timedelta(days=n_days_back - 1)
        end_date = center_date

        # Find all relevant files (monthly files)
        year_months = set()
        current = start_date
        while current <= end_date:
            year_months.add((current.year, current.month))
            current += timedelta(days=30)  # Approximate monthly step

        # Load and concatenate
        monthly_arrays = []
        for year, month in sorted(year_months):
            search_pattern = os.path.join(data_dir, str(year), f"*{year}{month:02d}.nc")
            file_matches = glob.glob(search_pattern)

            if file_matches:
                filepath = file_matches[0]
                try:
                    with xr.open_dataset(filepath) as ds:
                        var_name = list(ds.data_vars)[0]
                        # Select date range
                        ds_subset = ds[var_name].sel(DATE=slice(start_date, end_date))
                        monthly_arrays.append(ds_subset)
                except Exception as e:
                    print(f"   ⚠️  Error loading {filepath}: {e}")

        if monthly_arrays:
            # Combine all months
            combined = xr.concat(monthly_arrays, dim='DATE')
            data[var_prefix] = combined
        else:
            print(f"   ❌ No data found for {var_prefix} in period {start_date} to {end_date}")
            data[var_prefix] = None

    return data


def extract_dynamic_features_at_point(
    point_geom,
    dynamic_data: Dict[str, xr.DataArray],
    window_size: int = 4
) -> Dict[str, float]:
    """
    Extract dynamic (climate) features at a single point.

    Computes temporal aggregations (cumulative means and maxs) for various time windows.

    Args:
        point_geom: Shapely Point geometry (in UTM)
        dynamic_data: Dictionary with 'T' and 'P' DataArrays
        window_size: Spatial window size (pixels)

    Returns:
        Dictionary of dynamic feature values
    """
    features = {}

    # Load template to get pixel coordinates
    template_path = os.path.join(STATIC_RASTER_DIR, "nasadem.tif")

    with rioxarray.open_rasterio(template_path) as template_raster:
        template_crs = template_raster.rio.crs

        for var_prefix, da in dynamic_data.items():
            if da is None:
                # Missing data - fill with NaN
                for day_window in TIME_AGGREGATION_WINDOWS:
                    features[f"{var_prefix}_cumulative_mean_{day_window}d"] = np.nan
                    features[f"{var_prefix}_cumulative_max_{day_window}d"] = np.nan
                continue

            # Reproject and extract spatial window
            # Set CRS if missing
            if not da.rio.crs:
                da = da.rio.write_crs(template_crs)

            # Get pixel coordinates
            iy = np.abs(template_raster.y - point_geom.y).argmin().item()
            ix = np.abs(template_raster.x - point_geom.x).argmin().item()

            half_window = window_size // 2
            y_slice = slice(max(0, iy - half_window), min(template_raster.y.size, iy + half_window))
            x_slice = slice(max(0, ix - half_window), min(template_raster.x.size, ix + half_window))

            # Reproject to template
            try:
                template_chip = template_raster.isel(y=y_slice, x=x_slice, band=0)
                da_reprojected = da.rio.reproject_match(template_chip)

                # Compute spatial mean over window (time series)
                spatial_mean_series = da_reprojected.mean(dim=['y', 'x'])

                # Reverse time (most recent first)
                spatial_mean_series = spatial_mean_series.isel(DATE=slice(None, None, -1))

                # Compute cumulative statistics
                cumulative_sum = spatial_mean_series.cumsum(dim='DATE')
                divisor = xr.DataArray(np.arange(1, len(spatial_mean_series.DATE) + 1), dims='DATE')
                cumulative_means = cumulative_sum / divisor

                cumulative_maxs = xr.DataArray(
                    np.maximum.accumulate(spatial_mean_series.values),
                    coords=spatial_mean_series.coords
                )

                # Extract values for specific time windows
                for day_window in TIME_AGGREGATION_WINDOWS:
                    i = day_window - 1
                    if i < len(cumulative_means):
                        features[f"{var_prefix}_cumulative_mean_{day_window}d"] = float(cumulative_means.isel(DATE=i).item())
                        features[f"{var_prefix}_cumulative_max_{day_window}d"] = float(cumulative_maxs.isel(DATE=i).item())
                    else:
                        features[f"{var_prefix}_cumulative_mean_{day_window}d"] = np.nan
                        features[f"{var_prefix}_cumulative_max_{day_window}d"] = np.nan

            except Exception as e:
                print(f"   ⚠️  Error processing {var_prefix}: {e}")
                for day_window in TIME_AGGREGATION_WINDOWS:
                    features[f"{var_prefix}_cumulative_mean_{day_window}d"] = np.nan
                    features[f"{var_prefix}_cumulative_max_{day_window}d"] = np.nan

    return features


def extract_features_for_scenario(
    scenario: ClimateScenario,
    spatial_grid: gpd.GeoDataFrame,
    target_dates: List[Tuple[int, int, int]],
    output_path: str
):
    """
    Extract features for all spatial points and dates in a scenario.

    Args:
        scenario: Climate scenario configuration
        spatial_grid: GeoDataFrame with grid points
        target_dates: List of (year, month, day) tuples
        output_path: Path to save output CSV
    """
    print(f"\n{'='*80}")
    print(f"EXTRACTING FEATURES FOR SCENARIO: {scenario.name}")
    print(f"{'='*80}")
    print(f"Period: {scenario.period[0]}-{scenario.period[1]}")
    print(f"Spatial points: {len(spatial_grid)}")
    print(f"Target dates: {len(target_dates)}")
    print(f"Total extractions: {len(spatial_grid) * len(target_dates)}")

    all_features = []

    # Loop over dates
    for year, month, day in tqdm(target_dates, desc="Processing dates"):
        center_date = datetime(year, month, day)

        # Load dynamic data for this date (60-day window)
        dynamic_data = load_dynamic_data_for_period(scenario, center_date, n_days_back=TIME_STEPS)

        # Check if data was loaded
        if dynamic_data['T'] is None or dynamic_data['P'] is None:
            print(f"   ⚠️  Skipping {center_date.date()} - missing climate data")
            continue

        # Loop over spatial points
        for idx, row in spatial_grid.iterrows():
            point_geom = row.geometry

            # Extract static features
            static_features = extract_static_features_at_point(point_geom)

            # Extract dynamic features
            dynamic_features = extract_dynamic_features_at_point(point_geom, dynamic_data)

            # Combine all features
            feature_dict = {
                'scenario': scenario.name,
                'year': year,
                'month': month,
                'day': day,
                'date': center_date,
                'grid_id': row['grid_id'],
                'x': point_geom.x,
                'y': point_geom.y,
                **static_features,
                **dynamic_features
            }

            all_features.append(feature_dict)

    # Convert to DataFrame and save
    df = pd.DataFrame(all_features)
    df.to_csv(output_path, index=False)

    print(f"\n✓ Features extracted and saved to: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")

    return df


def main():
    """Main execution function."""
    print("="*80)
    print("CLIMATE PROJECTION FEATURE EXTRACTION")
    print("="*80)

    # Create spatial grid (full province, 1km resolution)
    spatial_grid = create_spatial_grid(bounds=None, resolution=1000)

    # Process each scenario
    for scenario in SCENARIOS:
        # Skip historical (already have training data)
        if scenario.name == "historical":
            continue

        print(f"\n\n{'#'*80}")
        print(f"# SCENARIO: {scenario.name}")
        print(f"{'#'*80}")

        # Get projection dates (fire season only)
        target_dates = get_projection_dates(scenario, months=[6, 7, 8, 9])

        # Output path
        output_dir = get_scenario_output_dir(scenario.name)
        output_path = os.path.join(output_dir, f"features_{scenario.name}.csv")

        # Check if already exists
        if os.path.exists(output_path):
            print(f"\n⚠️  Output file already exists: {output_path}")
            print("   Skipping. Delete file to reprocess.")
            continue

        # Extract features
        try:
            extract_features_for_scenario(scenario, spatial_grid, target_dates, output_path)
        except Exception as e:
            print(f"\n❌ ERROR processing scenario {scenario.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
