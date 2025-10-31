#!/usr/bin/env python
"""
Filter existing lightning raster stack to 2012+ only
(when lightning data became available)

This removes observations with NaN lightning data.
"""

import xarray as xr
import pandas as pd
import numpy as np
import os

print("="*80)
print("FILTERING LIGHTNING STACK TO 2012+ DATES")
print("="*80)
print()

# Paths
INPUT_STACK = "/mnt/CEPH_PROJECTS/Firescape/Scripts/OUTPUT/01_Training_Data_Lightning/spacetime_stacks_lightning.nc"
OUTPUT_DIR = "/mnt/CEPH_PROJECTS/Firescape/Scripts/OUTPUT/01_Training_Data_Lightning_2012plus/"
OUTPUT_STACK = os.path.join(OUTPUT_DIR, "spacetime_stacks_lightning_2012plus.nc")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Input: {INPUT_STACK}")
print(f"Output: {OUTPUT_STACK}")
print()

# Load the full stack
print("Loading full lightning stack...")
ds = xr.open_dataset(INPUT_STACK)

print(f"Original dataset:")
print(f"  Observations: {len(ds.id_obs)}")
print(f"  Variables: {list(ds.data_vars)}")
print(f"  Channels: {list(ds.channel.values)}")

# Get dates
dates = pd.to_datetime(ds['event_date'].values)
print(f"  Date range: {dates.min()} to {dates.max()}")
print(f"  Fire rate: {ds['label'].values.mean():.3f}")
print()

# Check for NaN lightning values
main_var = list(ds.data_vars)[0]
lightning_data = ds[main_var].sel(channel='L')
lightning_event_day = lightning_data.isel(time=0).mean(dim=['x', 'y']).values
has_nan = np.isnan(lightning_event_day)

print(f"Lightning data quality:")
print(f"  Valid values: {(~has_nan).sum()} ({(~has_nan).sum()/len(has_nan)*100:.1f}%)")
print(f"  NaN values: {has_nan.sum()} ({has_nan.sum()/len(has_nan)*100:.1f}%)")
print()

# Filter to 2012+ (should remove all NaN values)
print("Filtering to 2012 onwards...")
mask_2012plus = dates.year >= 2012
ds_filtered = ds.isel(id_obs=mask_2012plus)

dates_filtered = pd.to_datetime(ds_filtered['event_date'].values)
lightning_filtered = ds_filtered[main_var].sel(channel='L').isel(time=0).mean(dim=['x', 'y']).values
has_nan_filtered = np.isnan(lightning_filtered)

print(f"Filtered dataset:")
print(f"  Observations: {len(ds_filtered.id_obs)} ({len(ds_filtered.id_obs)/len(ds.id_obs)*100:.1f}% of original)")
print(f"  Date range: {dates_filtered.min()} to {dates_filtered.max()}")
print(f"  Fire rate: {ds_filtered['label'].values.mean():.3f}")
print(f"  Lightning NaN values: {has_nan_filtered.sum()} (should be 0!)")

if has_nan_filtered.sum() > 0:
    print(f"  ⚠️ WARNING: Still have {has_nan_filtered.sum()} NaN values!")
else:
    print(f"  ✓ No NaN values - all observations have valid lightning data!")
print()

# Lightning statistics
valid_lightning = lightning_filtered[~has_nan_filtered]
labels_valid = ds_filtered['label'].values[~has_nan_filtered]

print("Lightning statistics (valid data only):")
print(f"  Mean: {valid_lightning.mean():.6f}")
print(f"  Std: {valid_lightning.std():.6f}")
print(f"  Median: {np.median(valid_lightning):.6f}")
print(f"  Max: {valid_lightning.max():.6f}")
print(f"  % with lightning (>0): {(valid_lightning > 0).sum() / len(valid_lightning) * 100:.1f}%")
print()

print("Fire vs Non-fire:")
fire_mask = labels_valid == 1
nonfire_mask = labels_valid == 0
print(f"  Fire days:")
print(f"    Count: {fire_mask.sum()}")
print(f"    Mean lightning: {valid_lightning[fire_mask].mean():.6f}")
print(f"    % with lightning: {(valid_lightning[fire_mask] > 0).sum() / fire_mask.sum() * 100:.1f}%")
print(f"  Non-fire days:")
print(f"    Count: {nonfire_mask.sum()}")
print(f"    Mean lightning: {valid_lightning[nonfire_mask].mean():.6f}")
print(f"    % with lightning: {(valid_lightning[nonfire_mask] > 0).sum() / nonfire_mask.sum() * 100:.1f}%")
print()

# Check seasonal distribution
print("Seasonal distribution:")
months = dates_filtered.month
seasons = pd.cut(months, bins=[0, 2, 5, 8, 11, 12],
                 labels=['Winter', 'Spring', 'Summer', 'Fall', 'Winter2'])
seasons = seasons.astype(str)
seasons = ['Winter' if s=='Winter2' else s for s in seasons]

for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    season_mask = np.array([s == season for s in seasons])
    n_obs = season_mask.sum()
    n_fires = ds_filtered['label'].values[season_mask].sum()
    if n_obs > 0 and (~has_nan_filtered[season_mask]).sum() > 0:
        season_lightning = valid_lightning[season_mask[~has_nan_filtered]]
        pct_with_lightning = (season_lightning > 0).sum() / len(season_lightning) * 100
        print(f"  {season}: {n_obs} obs, {n_fires} fires, {pct_with_lightning:.1f}% with lightning")
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
print(f"  Removed: {len(ds.id_obs) - len(ds_filtered.id_obs)} observations (no lightning data)")
print(f"  Time period: 2012-2024 only")
print(f"  Output file: {OUTPUT_STACK}")
print(f"  Lightning signal: Fire days have {valid_lightning[fire_mask].mean() / valid_lightning[nonfire_mask].mean():.1f}x more lightning!")
print()
print("Next steps:")
print("  1. Train baseline model on filtered baseline stack")
print("  2. Train lightning model on filtered lightning stack")
print("  3. Compare results - lightning should now show predictive value!")
