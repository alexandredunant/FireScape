# Firescape Model Consistency Fixes - Summary

**Date:** 2025-10-29
**Purpose:** Fix critical inconsistencies between training and projection scripts to ensure valid climate projections

---

## Problems Identified

### 1. **Landcover Data Inconsistency**
- **Issue:** Scripts had hardcoded `LANDCOVER_FIRE_RISK_ORDINAL` mapping but were using different source files
- **Root Cause:** `landcoverfull.tif` contained pre-classified values (0-28), not raw 3-digit Corine codes
- **Impact:** High - Landcover transformation was inconsistent or missing across scripts

### 2. **Static Feature Spatial Aggregation Mismatch**
- **Issue:** Training used 4x4 window mean, projection used point extraction
- **Location:**
  - Training: `train_relative_probability_model.py` lines 84-97
  - Projection: `project_zone_fire_risk.py` lines 248-251
- **Impact:** High - Feature distributions didn't match between training and projection

### 3. **Redundant Transformation Logic**
- **Issue:** Landcover transformation code duplicated across multiple scripts
- **Impact:** Medium - Maintenance burden and risk of drift

---

## Solutions Implemented

### Solution 1: Created Pre-Transformed Landcover Raster

**New Script:** `Scripts/01_Data_Preparation/create_transformed_landcover.py`

**What it does:**
- Loads `CorineLandCover_polygon.tif` (actual 3-digit Corine codes)
- Applies `LANDCOVER_FIRE_RISK_ORDINAL` mapping (0-5 ordinal values)
- Creates `landcover_fire_risk.tif` in `Data/STATIC_INPUT/`
- Validates output (CRS, bounds, value range, distribution)

**Output:** `Data/STATIC_INPUT/landcover_fire_risk.tif`

**Fire Risk Distribution:**
```
0 (No risk):      153,947 pixels (0.9%)
1 (Very low):   1,091,868 pixels (6.2%)
2 (Low):       10,697,248 pixels (60.4%)
3 (Moderate):   1,833,858 pixels (10.4%)
4 (High):         878,421 pixels (5.0%)
5 (Very high):  3,056,703 pixels (17.3%)
```

**When to run:** Once after updating landcover data or mapping dictionary

### Solution 2: Updated Training Data Preparation

**Modified:** `Scripts/01_Data_Preparation/create_raster_stacks.py`

**Changes:**
- Removed special case handling for `landcoverfull.tif`
- Removed inline Corine→ordinal transformation
- Now loads `landcover_fire_risk.tif` directly
- Simplified code by ~10 lines

**Key lines changed:** 130-150

**Modified:** `Scripts/01_Data_Preparation/create_spacetime_dataset.py`

**Changes:**
- Updated trivial mask creation (lines 201-210)
- Changed from: `grids['landcoverfull'] != [10, 17, 23]` (simplified codes)
- Changed to: `grids['landcover_fire_risk'] > 0` (exclude fire_risk = 0)
- More robust: Excludes all water/ice/wetland areas (fire_risk = 0)
- Includes proper Corine codes: 335 (glaciers), 511 (rivers), 512 (lakes), 411/412 (wetlands)
- Better documentation of what's being excluded

**Key lines changed:** 201-210

### Solution 3: Updated Projection Script

**Modified:** `Scripts/04_Zone_Climate_Projections/project_zone_fire_risk.py`

**Changes:**

1. **Added 4x4 Window Averaging (lines 232-260)**
   ```python
   SPATIAL_WINDOW_SIZE = 4  # Matches training
   rds_windowed = rds_downsampled.rolling(
       x=SPATIAL_WINDOW_SIZE,
       y=SPATIAL_WINDOW_SIZE,
       center=True
   ).mean()
   ```

2. **Removed Landcover Post-Processing (lines 267-269)**
   - Eliminated lines 254-258 (old transformation code)
   - Now uses pre-transformed `landcover_fire_risk.tif`

3. **Added Validation Checks (lines 273-307)**
   - Validates landcover range [0-5]
   - Shows landcover distribution
   - Checks for missing values >10%
   - Reports feature consistency with trained model

4. **Enhanced Feature Validation (lines 504-520)**
   - Reports missing features
   - Validates feature order matches scaler
   - One-time validation message for debugging

---

## Files Modified

1. ✅ **Created:** `Scripts/01_Data_Preparation/create_transformed_landcover.py`
2. ✅ **Modified:** `Scripts/01_Data_Preparation/create_raster_stacks.py`
3. ✅ **Modified:** `Scripts/01_Data_Preparation/create_spacetime_dataset.py`
4. ✅ **Modified:** `Scripts/04_Zone_Climate_Projections/project_zone_fire_risk.py`
5. ✅ **Modified:** `Scripts/05_Lightning_Comparison/01_Data_Preparation/create_raster_stacks_with_lightning.py`
6. ✅ **Modified:** `Scripts/05_Lightning_Comparison/02_Model_Training/train_relative_probability_model_with_lightning.py`
7. ✅ **Generated:** `Data/STATIC_INPUT/landcover_fire_risk.tif`

---

## Consistency Verification

### ✅ Landcover Mapping
- **Status:** CONSISTENT
- All scripts now use identical `LANDCOVER_FIRE_RISK_ORDINAL` mapping (verified lines 25-43)
- Single source raster: `landcover_fire_risk.tif`

### ✅ Static Feature Spatial Aggregation
- **Status:** CONSISTENT
- Both training and projection use 4x4 window mean
- Training: lines 84-87 (create_cumulative_features)
- Projection: lines 254-260 (rolling window)

### ✅ Feature Lists
- **Status:** CONSISTENT
- STATIC_VARS: 12 variables (identical)
- DYNAMIC_VARS: ['T', 'P'] (identical)
- DAY_WINDOWS_TO_KEEP: [1,3,5,10,15,30,60] (identical)
- TIME_STEPS: 60 (identical)

### ✅ Prediction Logic
- **Status:** CONSISTENT
- Attention mechanism: Identical
- Sigmoid transformation: Compatible (projection uses numerically stable version)
- Minor difference: 100 vs 300 posterior samples (acceptable)

---

## Testing Recommendations

### 1. Verify Training Data Needs Update
If you've already trained your model using the old `landcoverfull.tif` or inline transformation:

```bash
# Check what data was used for training
python -c "
import xarray as xr
with xr.open_dataset('Scripts/OUTPUT/01_Training_Data/spacetime_stacks.nc') as ds:
    lc_data = ds['spacetime_stack'].sel(channel='landcover_fire_risk').isel(time=0, id_obs=0)
    print('Unique landcover values:', sorted(set(lc_data.values.flatten())))
"
```

**Expected output:** Values in range [0-5]

**If different:** Re-run `create_raster_stacks.py` to regenerate training data

### 2. Test Projection Script
```bash
cd /mnt/CEPH_PROJECTS/Firescape
python Scripts/04_Zone_Climate_Projections/project_zone_fire_risk.py
```

**Look for:**
- ✓ landcover_fire_risk in valid range: [0, 5]
- ✓ Feature validation messages
- ✓ No warnings about missing features
- ✓ Predictions complete successfully

### 3. Compare Feature Distributions
Create a test script to compare feature statistics between training and projection:

```python
# Compare mean/std of features
import joblib
import pandas as pd

# Load training scaler
scaler = joblib.load('Scripts/OUTPUT/02_Model_RelativeProbability/scaler_relative.joblib')
train_means = scaler.mean_
train_stds = scaler.scale_

# Compare with projection features
# (extract from projection run)
```

---

## Next Steps

### Immediate
1. ✅ Run `create_transformed_landcover.py` (DONE)
2. ✅ Update `create_raster_stacks.py` (DONE)
3. ✅ Update `project_zone_fire_risk.py` (DONE)
4. ⏳ Test projection script with new changes
5. ⏳ Verify predictions are reasonable

### Lightning Comparison Scripts (COMPLETED)
1. ✅ **Updated:** `Scripts/05_Lightning_Comparison/01_Data_Preparation/create_raster_stacks_with_lightning.py`
   - Removed inline landcover transformation
   - Now uses pre-transformed `landcover_fire_risk.tif`
   - Added documentation comments
   - **Key lines changed:** 143-163

2. ✅ **Updated:** `Scripts/05_Lightning_Comparison/02_Model_Training/train_relative_probability_model_with_lightning.py`
   - Removed special handling for 'landcoverfull' channel
   - Removed inline LANDCOVER_FIRE_RISK_ORDINAL transformation
   - Now expects 'landcover_fire_risk' channel with pre-transformed values (0-5)
   - Simplified: All static features use mean aggregation (consistent with main training)
   - **Key lines changed:** 121-133

### Optional

2. Re-train model if needed
   - Only necessary if old training data used wrong landcover values
   - Check using test in section above

3. Create feature distribution comparison script
   - Quantitatively verify training/projection consistency

---

## Key Learnings

1. **Source Data Verification Critical:**
   - `landcoverfull.tif` was pre-classified (0-28), not raw Corine codes
   - Always verify raster contents match expected schema

2. **Window Aggregation Matters:**
   - 4x4 window mean vs point extraction = different feature distributions
   - Spatial aggregation must match between training and inference

3. **Single Source of Truth:**
   - Pre-transforming rasters eliminates redundancy
   - Reduces code complexity and maintenance burden
   - Ensures perfect consistency

4. **Validation is Essential:**
   - Added multiple validation checks
   - Catches inconsistencies early
   - Makes debugging easier

---

## Contact / Questions

For questions about these changes, refer to:
- Consistency analysis report (generated during planning)
- This summary document
- Code comments in modified files

**Critical Files:**
- `Scripts/01_Data_Preparation/create_transformed_landcover.py` - Master transformation
- `Scripts/CONSISTENCY_FIX_SUMMARY.md` - This document
- `Data/STATIC_INPUT/landcover_fire_risk.tif` - Pre-transformed raster

---

**End of Summary**
