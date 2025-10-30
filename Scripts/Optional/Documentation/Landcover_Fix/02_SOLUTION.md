# FINAL Landcover Encoding Fix - Using Actual Corine Codes

## Discovery

Your `landcoverfull_onehotencoder.csv` contains **Corine Land Cover Level 3 codes** (28 different 3-digit codes, not 1-10).

The previous documentation incorrectly assumed simplified 1-10 class IDs. This document provides the **correct** mapping based on your actual data.

## Actual Land Cover Codes in Bolzano Province

Your dataset contains these **Corine Land Cover** codes:

| Code | Description | Count in Dataset | Fire Risk (0-5) |
|------|-------------|------------------|-----------------|
| **312** | Coniferous forest | 148 | **5** (Very high) |
| **324** | Transitional woodland-shrub | 392 | **4** (High) |
| **322** | Moors and heathland | 193 | **4** (High) |
| **313** | Mixed forest | 75 | **4** (High) |
| **321** | Natural grasslands | 261 | **3** (Moderate) |
| **311** | Broad-leaved forest | 66 | **3** (Moderate) |
| **333** | Sparsely vegetated areas | 260 | **3** (Moderate) |
| **231** | Pastures | 476 | **2** (Low) |
| **243** | Agriculture with natural veg | 195 | **2** (Low) |
| **242** | Complex cultivation | 53 | **2** (Low) |
| **221** | Vineyards | 32 | **2** (Low) |
| **222** | Fruit trees/berries | 17 | **2** (Low) |
| **211** | Non-irrigated arable | 21 | **2** (Low) |
| **112** | Discontinuous urban | 139 | **1** (Very low) |
| **332** | Bare rocks | 116 | **1** (Very low) |
| **331** | Sand/gravel | 3 | **1** (Very low) |
| **335** | Glaciers/snow | 45 | **0** (No risk) |
| **511** | Water courses | 3 | **0** (No risk) |
| **512** | Water bodies | 16 | **0** (No risk) |
| ... | (and 9 more urban/facility codes) | ... | **0-1** |

## Correct Ordinal Mapping

Use this mapping in **all scripts**:

```python
# Corine Land Cover → Fire Risk Ordinal Mapping
LANDCOVER_FIRE_RISK_ORDINAL = {
    # 0 = No fire risk (water, snow/ice, wetlands)
    335: 0,  # Glaciers and perpetual snow
    511: 0,  # Water courses
    512: 0,  # Water bodies
    411: 0,  # Inland marshes
    412: 0,  # Peat bogs

    # 1 = Very low fire risk (urban, bare soil)
    111: 1,  # Continuous urban fabric
    112: 1,  # Discontinuous urban fabric
    121: 1,  # Industrial or commercial units
    122: 1,  # Road and rail networks
    124: 1,  # Airports
    131: 1,  # Mineral extraction sites
    133: 1,  # Construction sites
    142: 1,  # Sport and leisure facilities
    331: 1,  # Beaches, dunes, sands
    332: 1,  # Bare rocks

    # 2 = Low fire risk (agriculture, managed land)
    211: 2,  # Non-irrigated arable land
    221: 2,  # Vineyards
    222: 2,  # Fruit trees and berry plantations
    231: 2,  # Pastures (most common in dataset!)
    242: 2,  # Complex cultivation patterns
    243: 2,  # Agriculture with natural vegetation

    # 3 = Moderate fire risk (grassland, broadleaf forest)
    311: 3,  # Broad-leaved forest
    321: 3,  # Natural grasslands
    333: 3,  # Sparsely vegetated areas

    # 4 = High fire risk (shrubland, mixed forest)
    313: 4,  # Mixed forest
    322: 4,  # Moors and heathland
    324: 4,  # Transitional woodland-shrub (2nd most common!)

    # 5 = Very high fire risk (coniferous forest)
    312: 5,  # Coniferous forest (Alpine conifers - very flammable)
}
```

## Implementation

### Quick Fix Script

I've created a complete mapping module:
- **[CORINE_LANDCOVER_FIRE_RISK_MAPPING.py](01_Data_Preparation/CORINE_LANDCOVER_FIRE_RISK_MAPPING.py)** - Import this in your scripts

### Usage in Scripts

**Option 1: Import the mapping** (recommended)

```python
# At top of script
from CORINE_LANDCOVER_FIRE_RISK_MAPPING import LANDCOVER_FIRE_RISK_ORDINAL, get_fire_risk

# In feature extraction
if var_name == 'landcoverfull':
    # Get mode (most common Corine code in window)
    values = window_data.values.flatten()
    values = values[~np.isnan(values)]
    if len(values) > 0:
        from scipy import stats
        mode_result = stats.mode(values, keepdims=False)
        corine_code = int(mode_result.mode)

        # Map to fire risk ordinal using imported function
        features['landcover_fire_risk'] = get_fire_risk(corine_code)
    else:
        features['landcover_fire_risk'] = 0.0
```

**Option 2: Copy the dictionary** (standalone)

```python
# Copy full LANDCOVER_FIRE_RISK_ORDINAL dictionary above
# Then use:
corine_code = int(mode_result.mode)
features['landcover_fire_risk'] = LANDCOVER_FIRE_RISK_ORDINAL.get(corine_code, 2)
```

## Files to Update

All the files mentioned in the original checklist, but with the **Corine mapping** instead:

1. ✅ [01_Data_Preparation/create_raster_stacks.py](01_Data_Preparation/create_raster_stacks.py)
2. ✅ [02_Model_Training/train_relative_probability_model.py](02_Model_Training/train_relative_probability_model.py)
3. ✅ [02_Model_Training/train_Dask_PyMC_timeseries.py](02_Model_Training/train_Dask_PyMC_timeseries.py)
4. ✅ [03_Climate_Projections/extract_projection_features.py](03_Climate_Projections/extract_projection_features.py)
5. ✅ [04_Zone_Climate_Projections/project_zone_fire_risk.py](04_Zone_Climate_Projections/project_zone_fire_risk.py)
6. ✅ [05_Lightning_Comparison/...](05_Lightning_Comparison/)

## Updated Automated Fix Script

I need to update `apply_landcover_fix.py` to use the Corine mapping. Let me create a new version:

```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts
python apply_landcover_fix_CORINE.py  # New script with correct mapping
```

## Key Differences from Previous Documentation

| Aspect | Previous (Wrong) | Corrected |
|--------|------------------|-----------|
| Code range | 1-10 | 111-512 (Corine 3-digit) |
| Number of classes | 10 | 28 |
| Coniferous code | 6 | **312** |
| Most common class | Unknown | **231** (Pastures) |
| 2nd most common | Unknown | **324** (Woodland-shrub) |
| Mapping complexity | Simple | More detailed, domain-specific |

## Why This Matters

The **Corine codes** are much more detailed than the simplified 1-10 system I previously assumed:

- **312 (Coniferous)** is correctly mapped to highest risk (5)
- **231 (Pastures)** - most common in your area - correctly low risk (2)
- **324 (Transitional woodland-shrub)** - 2nd most common - high risk (4)
- Alpine-specific classes (glaciers, high-altitude grassland) properly handled

## Verification

After applying the fix, verify that Corine codes are preserved then mapped:

```python
# In your raster stack output, check a sample
import xarray as xr
import numpy as np

ds = xr.open_dataset("OUTPUT/01_Training_Data/spacetime_stacks.nc")

# Check landcover_fire_risk values
lc_values = ds['spacetime_stack'].sel(channel='landcover_fire_risk')
print("Landcover fire risk values (0-5):")
print(np.unique(lc_values.values[~np.isnan(lc_values.values)]))
# Should show: [0. 1. 2. 3. 4. 5.]

# Compare to original (if you kept it)
if 'landcoverfull' in ds.channel.values:
    lc_orig = ds['spacetime_stack'].sel(channel='landcoverfull')
    print("\nOriginal Corine codes (3-digit):")
    print(np.unique(lc_orig.values[~np.isnan(lc_orig.values)]))
    # Should show codes like: 112, 231, 312, 324, etc.
```

## Summary

✅ **Correct approach**: Use Corine Land Cover codes (111-512) → map to fire risk ordinal (0-5)

❌ **Previous error**: Assumed simple 1-10 classification

🎯 **Key insight**: Your data uses **Corine Level 3** classification with 28 classes specific to Alpine/South Tyrol region

📊 **Most important codes for fire modeling**:
- **312** (Coniferous - 5th most common) → Risk = 5 (very high)
- **324** (Transitional woodland - 2nd most common) → Risk = 4 (high)
- **231** (Pastures - most common) → Risk = 2 (low, managed)

All previous documentation about the fix still applies, just use the **Corine mapping** instead of the simplified 1-10 mapping.
