#!/usr/bin/env python
"""
Filter existing baseline raster stack to 2012+ only
(when lightning data became available)

This is much faster than re-processing all rasters!
"""

import xarray as xr
import pandas as pd
import os

print("="*80)
print("FILTERING BASELINE STACK TO 2012+ DATES")
print("="*80)
print()

# Paths
INPUT_STACK = "/mnt/CEPH_PROJECTS/Firescape/Scripts/OUTPUT/01_Training_Data/spacetime_stacks.nc"
OUTPUT_DIR = "/mnt/CEPH_PROJECTS/Firescape/Scripts/OUTPUT/01_Training_Data_Baseline_2012plus/"
OUTPUT_STACK = os.path.join(OUTPUT_DIR, "spacetime_stacks_baseline_2012plus.nc")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Input: {INPUT_STACK}")
print(f"Output: {OUTPUT_STACK}")
print()

# Load the full stack
print("Loading full baseline stack...")
ds = xr.open_dataset(INPUT_STACK)

print(f"Original dataset:")
print(f"  Observations: {len(ds.id_obs)}")
print(f"  Variables: {list(ds.data_vars)}")
print(f"  Channels: {list(ds.channel.values)}")

# Get dates and filter
dates = pd.to_datetime(ds['event_date'].values)
print(f"  Date range: {dates.min()} to {dates.max()}")
print(f"  Fire rate: {ds['label'].values.mean():.3f}")
print()

# Filter to 2012+
print("Filtering to 2012 onwards...")
mask_2012plus = dates.year >= 2012
ds_filtered = ds.isel(id_obs=mask_2012plus)

dates_filtered = pd.to_datetime(ds_filtered['event_date'].values)
print(f"Filtered dataset:")
print(f"  Observations: {len(ds_filtered.id_obs)} ({len(ds_filtered.id_obs)/len(ds.id_obs)*100:.1f}% of original)")
print(f"  Date range: {dates_filtered.min()} to {dates_filtered.max()}")
print(f"  Fire rate: {ds_filtered['label'].values.mean():.3f}")
print()

# Check seasonal distribution
print("Seasonal distribution:")
months = dates_filtered.month
seasons = pd.cut(months, bins=[0, 2, 5, 8, 11, 12],
                 labels=['Winter', 'Spring', 'Summer', 'Fall', 'Winter2'])
seasons = seasons.astype(str)
seasons = ['Winter' if s=='Winter2' else s for s in seasons]

for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    season_mask = [s == season for s in seasons]
    n_obs = sum(season_mask)
    n_fires = ds_filtered['label'].values[season_mask].sum()
    print(f"  {season}: {n_obs} obs, {n_fires} fires")
print()

# Save filtered stack
print("Saving filtered stack...")
ds_filtered.to_netcdf(OUTPUT_STACK)
print(f"✓ Saved: {OUTPUT_STACK}")

# Close datasets
ds.close()
ds_filtered.close()

print()
print("="*80)
print("FILTERING COMPLETE!")
print("="*80)
print()
print(f"Summary:")
print(f"  Input observations: {len(ds.id_obs)}")
print(f"  Output observations: {len(ds_filtered.id_obs)}")
print(f"  Reduction: {len(ds.id_obs) - len(ds_filtered.id_obs)} observations removed")
print(f"  Time period: 2012-2024 only")
print(f"  Output file: {OUTPUT_STACK}")
print()
print("Next steps:")
print("  1. Create filtered lightning stack (same dates)")
print("  2. Train both models on 2012+ data")
print("  3. Compare results")
