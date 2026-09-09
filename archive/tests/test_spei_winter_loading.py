#!/usr/bin/env python
"""
Quick test to verify SPEI loading works correctly for Winter season
"""
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

SPEI_DIR = Path("/mnt/CEPH_PROJECTS/Firescape/Data/05_Meteorological_Data")
YEARS = range(2012, 2025)

def load_seasonal_climatology(
    file_pattern, var_name, months, data_var_name, statistic="mean", years=YEARS
):
    """
    Load standardized NetCDF files and calculate seasonal statistic using open_mfdataset
    for lazy loading. This function handles the creation of a 'season_year' coordinate
    for correct grouping, especially for cross-year seasons like Winter.

    Args:
        statistic: 'mean', 'p10' (10th percentile), or 'p90' (90th percentile)
    """
    # Collect all files
    files = [
        file_pattern.format(year=year)
        for year in years
        if Path(file_pattern.format(year=year)).exists()
    ]

    if not files:
        print(f"    No files found for {var_name}")
        return None

    try:
        # Open all datasets lazily using open_mfdataset.
        ds_combined = xr.open_mfdataset(
            files, concat_dim="time", combine="nested", decode_times=True, lock=False
        )

        # Select only the relevant data variable and months
        ds_seasonal = ds_combined[data_var_name].sel(
            time=ds_combined["time"].dt.month.isin(months)
        )

        if ds_seasonal.sizes["time"] == 0:
            print(f"    No seasonal data found for {var_name} in selected months.")
            return None

        # Determine if it's a cross-year season (e.g., Winter: Dec, Jan, Feb)
        is_cross_year = (12 in months) and (1 in months)

        # Create 'season_year' coordinate based on the season type
        if is_cross_year:
            # For cross-year seasons, Decembers are grouped with the following year's Jan/Feb
            season_year = xr.where(
                ds_seasonal["time"].dt.month == 12,
                ds_seasonal["time"].dt.year + 1,
                ds_seasonal["time"].dt.year,
            )
        else:
            # For seasons fully within a calendar year, the season_year is simply the year of the month
            season_year = ds_seasonal["time"].dt.year

        # Calculate statistic based on requested method
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")

            if statistic == "mean":
                # Assign the new 'season_year' coordinate to the dataset
                ds_with_season_year = ds_seasonal.assign_coords(
                    season_year=("time", season_year.data)
                )

                # Calculate the mean for each 'season_year' group.
                mean_per_season_year = ds_with_season_year.groupby("season_year").mean("time")

                # Finally, calculate the mean across all 'season_year' entries to get the climatology.
                final_climatology = mean_per_season_year.mean("season_year")

            elif statistic == "p90":
                # 90th percentile across all seasonal days (extreme conditions)
                final_climatology = ds_seasonal.quantile(0.9, dim="time")

            elif statistic == "p10":
                # 10th percentile across all seasonal days (mild conditions)
                final_climatology = ds_seasonal.quantile(0.1, dim="time")

            else:
                raise ValueError(f"Unknown statistic: {statistic}")

        # Check if the result contains valid data
        values = final_climatology.values
        if np.isnan(values).all():
            print(f"    WARNING: {var_name} ({statistic}) resulted in ALL NaNs. Check data coverage.")

        # Return the computed climatology as a NumPy array. This triggers Dask computation if lazy.
        return values

    except Exception as e:
        print(
            f"    Error in open_mfdataset or climatology calculation for {var_name}: {e}"
        )
        return None


# Test Winter season (cross-year)
print("=" * 80)
print("TESTING WINTER SEASON SPEI LOADING")
print("=" * 80)
print()

months = [12, 1, 2]  # Winter

for scale in [30, 90]:
    print(f"Testing SPEI{scale}...")

    spei_grid = load_seasonal_climatology(
        str(SPEI_DIR / f"SPEI{scale}_Standardized" / f"spei{scale}_{{year}}.nc"),
        f"SPEI{scale}",
        months,
        f"spei{scale}",
        statistic="mean",
    )

    if spei_grid is not None:
        print(f"  ✓ Successfully loaded SPEI{scale}")
        print(f"    Shape: {spei_grid.shape}")
        print(f"    Min: {np.nanmin(spei_grid):.4f}")
        print(f"    Max: {np.nanmax(spei_grid):.4f}")
        print(f"    Mean: {np.nanmean(spei_grid):.4f}")
        print(f"    NaN count: {np.isnan(spei_grid).sum()}")
    else:
        print(f"  ✗ Failed to load SPEI{scale}")

    print()

print("=" * 80)
print("✓ TEST COMPLETE")
print("=" * 80)
