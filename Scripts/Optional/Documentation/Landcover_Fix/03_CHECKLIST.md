# Landcover Encoding Fix - Implementation Checklist

## Overview

This checklist ensures the `landcoverfull` categorical variable is consistently encoded as ordinal fire risk across ALL scripts.

## Quick Start

### Option 1: Automated Fix (Recommended)

```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts
python apply_landcover_fix.py
```

This will:
- Create backups of all files
- Apply ordinal encoding consistently
- Generate a summary report

### Option 2: Manual Fix

Follow the checklist below to update each file individually.

## Files Requiring Updates

### ✅ Core Training Pipeline

#### 1. [01_Data_Preparation/create_raster_stacks.py](01_Data_Preparation/create_raster_stacks.py)

**Status**: ⚠️ Needs fixing

**Current behavior** (line 95-104):
- Uses `mode` to aggregate (✓ correct)
- Stores as `features[var_name]` with raw class ID (✗ incorrect)

**Required changes**:

```python
# Add after line 54
LANDCOVER_FIRE_RISK_ORDINAL = {
    9: 0, 10: 0,   # Water, Snow/Ice
    1: 1, 8: 1,    # Urban, Bare rock
    2: 2,          # Agriculture
    3: 3, 5: 3,    # Grassland, Broadleaf
    7: 4, 4: 4,    # Mixed forest, Shrubland
    6: 5           # Coniferous
}

# Update line 52-55 (STATIC_VARS list)
STATIC_VARS = [
    'tri', 'northness', 'slope', 'aspect', 'nasadem',
    'treecoverdensity', 'landcover_fire_risk', 'distroads',  # ← changed
    'eastness', 'flammability', 'walking_time_to_bldg',
    'walking_time_to_elec_infra'
]

# Update line 95-104 (in extract_stack_for_point function)
if var_name == 'landcoverfull':
    # Get mode (most common value) for categorical variables
    values = window_data.values.flatten()
    values = values[~np.isnan(values)]
    if len(values) > 0:
        from scipy import stats
        mode_result = stats.mode(values, keepdims=False)
        landcover_class = int(mode_result.mode)

        # Map to ordinal fire risk
        features['landcover_fire_risk'] = float(
            LANDCOVER_FIRE_RISK_ORDINAL.get(landcover_class, 2)
        )
    else:
        features['landcover_fire_risk'] = 0.0
```

#### 2. [02_Model_Training/train_relative_probability_model.py](02_Model_Training/train_relative_probability_model.py)

**Status**: ⚠️ Needs fixing

**Current behavior** (line 60-67, 95-104):
- STATIC_VARS includes `'landcoverfull'`
- `create_cumulative_features()` uses mode but treats as continuous

**Required changes**:

```python
# Add after line 50 (in CONFIGURATION section)
LANDCOVER_FIRE_RISK_ORDINAL = {
    9: 0, 10: 0, 1: 1, 8: 1, 2: 2,
    3: 3, 5: 3, 7: 4, 4: 4, 6: 5
}

# Update line 61-67 (STATIC_VARS)
STATIC_VARS = [
    'tri', 'northness', 'slope', 'aspect', 'nasadem',
    'treecoverdensity', 'landcover_fire_risk', 'distroads',  # ← changed
    'eastness', 'flammability', 'walking_time_to_bldg',
    'walking_time_to_elec_infra'
]

# Remove or comment out CATEGORICAL_VARS (line 85)
# CATEGORICAL_VARS = ['landcoverfull']  # No longer needed

# Update line 95-104 (in create_cumulative_features function)
if var_name == 'landcoverfull':
    # Get mode (most common value)
    values = window_data.values.flatten()
    values = values[~np.isnan(values)]
    if len(values) > 0:
        from scipy import stats
        mode_result = stats.mode(values, keepdims=False)
        landcover_class = int(mode_result.mode)

        # Map to ordinal fire risk
        features['landcover_fire_risk'] = float(
            LANDCOVER_FIRE_RISK_ORDINAL.get(landcover_class, 2)
        )
    else:
        features['landcover_fire_risk'] = 0.0
```

#### 3. [02_Model_Training/train_Dask_PyMC_timeseries.py](02_Model_Training/train_Dask_PyMC_timeseries.py)

**Status**: ⚠️ Needs fixing

**Current behavior** (line 77-82, 108-122):
- Uses `compute_mode_robust()` (✓ correct)
- Treats result as continuous (✗ incorrect)

**Required changes**:

```python
# Add after line 66 (in CONFIGURATION section)
LANDCOVER_FIRE_RISK_ORDINAL = {
    9: 0, 10: 0, 1: 1, 8: 1, 2: 2,
    3: 3, 5: 3, 7: 4, 4: 4, 6: 5
}

# Update line 77-82 (STATIC_VARS)
STATIC_VARS = [
    'tri', 'northness', 'slope', 'aspect', 'nasadem',
    'treecoverdensity', 'landcover_fire_risk', 'distroads',  # ← changed
    'eastness', 'flammability', 'walking_time_to_bldg',
    'walking_time_to_elec_infra'
]

# Update line 108-122 (landcover handling)
landcover_data_full = ds[main_data_var].sel(channel='landcoverfull').isel(time=0)

# Get mode values
landcover_mode = landcover_data_full.groupby('id_obs').apply(
    lambda x: xr.DataArray(compute_mode_robust(x.data),
                          coords={'id_obs': x['id_obs'].item()}, dims=[])
)

# Map to ordinal fire risk
def map_landcover_to_fire_risk(landcover_value):
    return LANDCOVER_FIRE_RISK_ORDINAL.get(int(landcover_value), 2)

landcover_fire_risk = xr.apply_ufunc(
    map_landcover_to_fire_risk,
    landcover_mode,
    vectorize=True
)
landcover_fire_risk = landcover_fire_risk.expand_dims(channel=1)
landcover_fire_risk = landcover_fire_risk.assign_coords(channel=['landcover_fire_risk'])

# Combine with other static data
static_data = xr.concat([static_cont_data, landcover_fire_risk], dim='channel')
```

### ✅ Climate Projection Scripts

#### 4. [03_Climate_Projections/extract_projection_features.py](03_Climate_Projections/extract_projection_features.py)

**Status**: ⚠️ Needs fixing

**Current behavior** (line 42-52, 131-138):
- STATIC_VARS includes `'landcoverfull'`
- Uses mode but stores raw class ID

**Required changes**:

```python
# Add after line 38 (after imports)
LANDCOVER_FIRE_RISK_ORDINAL = {
    9: 0, 10: 0, 1: 1, 8: 1, 2: 2,
    3: 3, 5: 3, 7: 4, 4: 4, 6: 5
}

# Update line 42-52 (STATIC_VARS)
STATIC_VARS = [
    'tri', 'northness', 'slope', 'aspect', 'nasadem',
    'treecoverdensity', 'landcover_fire_risk', 'distroads',  # ← changed
    'eastness', 'flammability', 'walking_time_to_bldg',
    'walking_time_to_elec_infra'
]

# Update line 131-138 (in extract_static_features_at_point)
# Categorical handling for landcover → ordinal fire risk
if var_name == 'landcoverfull':
    values = window_data.values.flatten()
    values = values[~np.isnan(values)]
    if len(values) > 0:
        mode_result = stats.mode(values, keepdims=False)
        landcover_class = int(mode_result.mode)

        # Map to fire risk ordinal
        features['landcover_fire_risk'] = LANDCOVER_FIRE_RISK_ORDINAL.get(
            landcover_class, 2
        )
    else:
        features['landcover_fire_risk'] = 0
```

#### 5. [04_Zone_Climate_Projections/project_zone_fire_risk.py](04_Zone_Climate_Projections/project_zone_fire_risk.py)

**Status**: ⚠️ Needs fixing

**Current behavior** (line 89-93, 206-228):
- STATIC_VARS includes `'landcoverfull'`
- Extracts raw pixel values (no mode aggregation)
- Treats as continuous (✗ incorrect)

**Required changes**:

```python
# Add after line 95 (after DAY_WINDOWS_TO_KEEP)
import pandas as pd  # Ensure pandas is imported

LANDCOVER_FIRE_RISK_ORDINAL = {
    9: 0, 10: 0, 1: 1, 8: 1, 2: 2,
    3: 3, 5: 3, 7: 4, 4: 4, 6: 5
}

# Update line 89-93 (STATIC_VARS)
STATIC_VARS = [
    'tri', 'northness', 'slope', 'aspect', 'nasadem',
    'treecoverdensity', 'landcover_fire_risk', 'distroads', 'eastness',  # ← changed
    'flammability', 'walking_time_to_bldg', 'walking_time_to_elec_infra'
]

# Add after line 228 (after static features extraction loop)
# Post-process landcover: convert to fire risk ordinal
if 'landcoverfull' in static_features_df.columns:
    static_features_df['landcover_fire_risk'] = static_features_df['landcoverfull'].apply(
        lambda x: LANDCOVER_FIRE_RISK_ORDINAL.get(int(x) if not pd.isna(x) else 0, 2)
    )
    static_features_df = static_features_df.drop(columns=['landcoverfull'])
```

### ✅ Lightning Comparison Scripts

#### 6. [05_Lightning_Comparison/01_Data_Preparation/create_raster_stacks_with_lightning.py](05_Lightning_Comparison/01_Data_Preparation/create_raster_stacks_with_lightning.py)

**Status**: ⚠️ Needs fixing (same as #1)

**Required changes**: Same as `create_raster_stacks.py` above.

#### 7. [05_Lightning_Comparison/02_Model_Training/train_relative_probability_model_with_lightning.py](05_Lightning_Comparison/02_Model_Training/train_relative_probability_model_with_lightning.py)

**Status**: ⚠️ Needs fixing (same as #2)

**Required changes**: Same as `train_relative_probability_model.py` above.

## Verification Steps

After applying fixes to all files:

### 1. Verify Consistency

```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts

# Check that all STATIC_VARS lists use 'landcover_fire_risk'
grep -n "STATIC_VARS" 01_Data_Preparation/create_raster_stacks.py \
                        02_Model_Training/train_relative_probability_model.py \
                        03_Climate_Projections/extract_projection_features.py \
                        04_Zone_Climate_Projections/project_zone_fire_risk.py

# Check that ordinal mapping is present
grep -n "LANDCOVER_FIRE_RISK_ORDINAL" 01_Data_Preparation/create_raster_stacks.py \
                                       02_Model_Training/train_relative_probability_model.py \
                                       03_Climate_Projections/extract_projection_features.py \
                                       04_Zone_Climate_Projections/project_zone_fire_risk.py
```

### 2. Test Training Pipeline

```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts

# Regenerate training data (small sample for testing)
cd 01_Data_Preparation
python create_raster_stacks.py  # Should see 'landcover_fire_risk' in output

# Check output NetCDF
python -c "
import xarray as xr
ds = xr.open_dataset('OUTPUT/01_Training_Data/spacetime_stacks.nc')
print('Channels:', ds.channel.values)
# Should contain 'landcover_fire_risk', NOT 'landcoverfull'
"

# Retrain model
cd ../02_Model_Training
python train_relative_probability_model.py

# Check feature names in scaler
python -c "
import joblib
scaler = joblib.load('OUTPUT/02_Model_RelativeProbability/scaler_relative.joblib')
print('Feature names:', scaler.feature_names_in_)
# Should contain 'landcover_fire_risk'
"
```

### 3. Test Climate Projections

```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts/03_Climate_Projections

# Run single scenario test
python run_all_scenarios.py  # Interrupt after first scenario

# Check output features
python -c "
import pandas as pd
df = pd.read_csv('OUTPUT/03_Climate_Projections/rcp45_2030/features_rcp45_2030.csv', nrows=5)
print('Feature columns:', df.columns.tolist())
# Should contain 'landcover_fire_risk'
"
```

### 4. Test Zone Projections

```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts/04_Zone_Climate_Projections

# Run with single year/month for testing
# Edit project_zone_fire_risk.py temporarily:
#   PROJECTION_YEARS = [2020]
#   PROJECTION_MONTHS = [8]

python project_zone_fire_risk.py

# Check that no errors occur
```

## Troubleshooting

### Issue: "KeyError: 'landcoverfull'" in scaler

**Cause**: Model was trained with old feature names, but trying to predict with new names.

**Solution**: Retrain the model after updating training scripts.

```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts/02_Model_Training
python train_relative_probability_model.py
```

### Issue: "ValueError: Feature names mismatch"

**Cause**: Mismatch between feature names in training vs prediction.

**Solution**: Ensure ALL scripts use 'landcover_fire_risk' consistently.

```bash
# Check all files
grep -r "landcoverfull" --include="*.py" | grep -v "Archive" | grep -v "backup" | grep -v ".md"
# Should only appear in comments or string literals
```

### Issue: NaN values in landcover_fire_risk

**Cause**: Landcover class not in mapping dictionary.

**Solution**: Check your landcover raster for unexpected class values.

```python
import rioxarray
import numpy as np

# Check unique values
with rioxarray.open_rasterio("Data/STATIC_INPUT/landcoverfull.tif") as rds:
    unique_values = np.unique(rds.values[~np.isnan(rds.values)])
    print(f"Landcover classes found: {unique_values}")

# If unexpected classes found, update LANDCOVER_FIRE_RISK_ORDINAL mapping
```

## Rollback Instructions

If issues occur after applying fixes:

### 1. Restore from backups

```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts

# List backup files
find . -name "*_backup_*.py" -not -path "*/Archive/*"

# Restore individual file
cp 01_Data_Preparation/create_raster_stacks_backup_*.py 01_Data_Preparation/create_raster_stacks.py
```

### 2. Or revert via git (if using version control)

```bash
cd /mnt/CEPH_PROJECTS/Firescape
git checkout HEAD -- Scripts/
```

## Completion Checklist

- [ ] Backed up all files (automatically done by apply_landcover_fix.py)
- [ ] Applied fixes to all 7 files
- [ ] Verified STATIC_VARS lists updated
- [ ] Verified ordinal mapping present in all files
- [ ] Regenerated training data
- [ ] Retrained model successfully
- [ ] Tested climate projection on single scenario
- [ ] Tested zone projection on single year/month
- [ ] No "landcoverfull" in error messages
- [ ] Feature names consistent across pipeline
- [ ] Deleted backup files after successful testing

## Summary

**Before**:
```python
STATIC_VARS = [..., 'landcoverfull', ...]  # ✗ Categorical treated as continuous
features['landcoverfull'] = int(mode_result.mode)  # ✗ Raw class ID
```

**After**:
```python
STATIC_VARS = [..., 'landcover_fire_risk', ...]  # ✓ Ordinal fire risk
LANDCOVER_FIRE_RISK_ORDINAL = {6: 5, 4: 4, ...}  # ✓ Mapping defined
features['landcover_fire_risk'] = LANDCOVER_FIRE_RISK_ORDINAL.get(
    int(mode_result.mode), 2
)  # ✓ Ordinal encoding
```

**Result**: Consistent, mathematically correct handling of categorical landcover data across entire pipeline.
