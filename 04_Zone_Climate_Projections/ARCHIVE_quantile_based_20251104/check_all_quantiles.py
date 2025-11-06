#!/usr/bin/env python
"""
Check all precipitation quantiles for August
"""

import xarray as xr
import numpy as np
from pathlib import Path

print("="*80)
print("CHECKING ALL PRECIPITATION QUANTILES FOR AUGUST 2020")
print("="*80)
print()

base_dir = Path("/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/pr/rcp45")
year = 2020

for quantile in ['pctl25', 'pctl50', 'pctl75', 'pctl99']:
    nc_file = base_dir / f"pr_EUR-11_{quantile}_rcp45.nc"

    print(f"{quantile.upper()}:")
    ds = xr.open_dataset(nc_file)

    august_data = ds['pr'].sel(time=slice(f'{year}-08-01', f'{year}-08-31'))

    # Calculate statistics
    spatial_mean_per_day = august_data.mean(dim=['x', 'y']).values
    monthly_total = august_data.sum(dim='time').mean().values

    print(f"  Daily average: {spatial_mean_per_day.mean():.2f} mm/day")
    print(f"  Monthly total: {monthly_total:.2f} mm")
    print()

    ds.close()

print("="*80)
print("HISTORICAL COMPARISON")
print("="*80)
print(f"Historical August precipitation: 92 mm")
print()
print("ISSUE: The median (pctl50) shows only 26mm, which is far below historical.")
print("This suggests the climate model ensemble quantiles might represent dry/wet")
print("scenarios rather than precipitation amounts directly.")
print("="*80)
