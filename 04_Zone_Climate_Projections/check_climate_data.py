#!/usr/bin/env python
"""
Check climate data extraction for August precipitation
"""

import xarray as xr
import numpy as np
from pathlib import Path

# Check RCP4.5 median precipitation
nc_file = Path("/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/pr/rcp45/pr_EUR-11_pctl50_rcp45.nc")

print("="*80)
print("CHECKING PRECIPITATION DATA FOR AUGUST")
print("="*80)
print()

ds = xr.open_dataset(nc_file)
print("Dataset structure:")
print(ds)
print()

print("Variable 'pr' attributes:")
print(ds['pr'].attrs)
print()

# Check historical period (around 2020)
year = 2020
print(f"Checking August {year}:")
august_data = ds['pr'].sel(time=slice(f'{year}-08-01', f'{year}-08-31'))

print(f"  Shape: {august_data.shape}")
print(f"  Time range: {august_data.time.values[0]} to {august_data.time.values[-1]}")
print(f"  Number of days: {len(august_data.time)}")
print()

# Calculate statistics
spatial_mean_per_day = august_data.mean(dim=['x', 'y']).values
monthly_total_per_pixel = august_data.sum(dim='time')
spatial_mean_monthly = monthly_total_per_pixel.mean().values

print(f"  Daily spatial means (mm):")
for i, val in enumerate(spatial_mean_per_day, 1):
    print(f"    Day {i:2d}: {val:.2f} mm")

print()
print(f"  Mean daily precipitation: {spatial_mean_per_day.mean():.2f} mm")
print(f"  Monthly total (averaged over space): {spatial_mean_monthly:.2f} mm")
print()

# Check units
print("Units check:")
print(f"  pr units attribute: {ds['pr'].attrs.get('units', 'NOT SPECIFIED')}")
print(f"  WARNING: Units are 'mm', NOT 'mm/day' - these are DAILY totals!")
print()

# Check another year
year = 2050
print(f"Checking August {year}:")
august_data = ds['pr'].sel(time=slice(f'{year}-08-01', f'{year}-08-31'))
spatial_mean_per_day = august_data.mean(dim=['x', 'y']).values
monthly_total_per_pixel = august_data.sum(dim='time')
spatial_mean_monthly = monthly_total_per_pixel.mean().values

print(f"  Number of days: {len(august_data.time)}")
print(f"  Mean daily precipitation: {spatial_mean_per_day.mean():.2f} mm")
print(f"  Monthly total (averaged over space): {spatial_mean_monthly:.2f} mm")
print()

ds.close()

print("="*80)
print("RECOMMENDATION:")
print("="*80)
print("The climate projection should show MONTHLY TOTAL, not daily average.")
print("For comparison with 92mm historical, we need to sum over the month.")
print("="*80)
