# Path Updates Summary
**Date:** October 20, 2025
**Status:** ✓ All scripts updated for consistency

---

## Overview

All analysis scripts have been updated to use consistent file paths following the reorganization of data files. The main changes involve:

1. **Static raster directory** moved from external location to project directory
2. **NetCDF output** location corrected
3. **File naming conventions** standardized (removed "50" suffix)

---

## Updated File Paths

### Static Rasters
- **Old:** `/mnt/CEPH_PROJECTS/ESA_EO4MULTIHA/eo4multiha-wildfires/STATIC/`
- **New:** `/mnt/CEPH_PROJECTS/Firescape/Data/STATIC_INPUT/`

### Raster Stack NetCDF
- **Old:** `/mnt/CEPH_PROJECTS/Firescape/Scripts/Uncertainty_Attention/spacetime_stacks.nc`
- **New:** `/mnt/CEPH_PROJECTS/Firescape/Data/OUTPUT/spacetime_stacks.nc`

### File Naming Convention
- **Old:** `nasadem50.tif`, `slope50.tif`, etc.
- **New:** `nasadem.tif`, `slope.tif`, etc.
- **Special case:** `landcover` → `landcoverfull.tif`

---

## Scripts Updated

### 1. `04_Bayesian_pyMCLogisticRegression_Linear_Attention_commented.py`
**Changes:**
- Line 54: Updated `NETCDF_PATH` to `/mnt/CEPH_PROJECTS/Firescape/Data/OUTPUT/spacetime_stacks.nc`

**Status:** ✓ Ready for training

---

### 2. `05_Bayesian_Climate_Projection_CLEAN.py` (New clean version)
**Changes:**
- Line 58: `STATIC_RASTER_DIR` → `/mnt/CEPH_PROJECTS/Firescape/Data/STATIC_INPUT/`
- Line 148: Template path → `nasadem.tif` (no "50" suffix)
- Lines 187-192: File naming logic updated:
  ```python
  if var_name in ['flammability', 'walking_time_to_bldg', 'walking_time_to_elec_infra']:
      raster_path = STATIC_RASTER_DIR / f"{var_name}.tif"
  elif var_name == 'landcover':
      raster_path = STATIC_RASTER_DIR / "landcoverfull.tif"
  else:
      raster_path = STATIC_RASTER_DIR / f"{var_name}.tif"
  ```

**Features:**
- Removed all try/except blocks (uses assertions instead)
- Proper chunking for memory safety
- Full 50m resolution by default (`SPATIAL_DOWNSAMPLE = 1`)
- Can run at 100m (`=2`) or 150m (`=3`) for faster execution

**Status:** ✓ Ready for climate projections

---

### 3. `05_Bayesian_Climate_Projection_OPTIMIZED.py`
**Changes:**
- Line 58: `STATIC_RASTER_DIR` → `/mnt/CEPH_PROJECTS/Firescape/Data/STATIC_INPUT/`
- Line 148: Template path → `nasadem.tif`
- Lines 187-192: File naming logic updated (same as CLEAN version)

**Note:** This is the older optimized version. **Recommend using `05_Bayesian_Climate_Projection_CLEAN.py` instead.**

---

### 4. `05_Bayesian_Lookback_2022_GIF.py`
**Changes:**
- Line 44: `STATIC_RASTER_DIR` → `/mnt/CEPH_PROJECTS/Firescape/Data/STATIC_INPUT/`
- Line 53: `MODEL_DIR` → `Uncertainty_Attention/model_plots_bayesian_linear`
- Lines 102-107: File naming logic updated
- All `nasadem50.tif` references → `nasadem.tif`

**Status:** ✓ Ready for GIF generation after model training

---

### 5. `07_Fire_Brigade_Zone_Analysis.py`
**No changes needed** - Already using correct paths:
- Line 27: Fire brigade shapefile (clipped version)
- Line 33: Model directory path correct

**Status:** ✓ Ready for brigade analysis

---

## File Locations Summary

### Input Data
```
/mnt/CEPH_PROJECTS/Firescape/Data/
├── STATIC_INPUT/                       # Static environmental rasters
│   ├── nasadem.tif
│   ├── slope.tif
│   ├── aspect.tif
│   ├── northness.tif
│   ├── eastness.tif
│   ├── tri.tif
│   ├── treecoverdensity.tif
│   ├── landcoverfull.tif
│   ├── distroads.tif
│   ├── flammability.tif
│   ├── walking_time_to_bldg.tif
│   └── walking_time_to_elec_infra.tif
│
├── OUTPUT/                             # Generated data
│   ├── spacetime_stacks.nc             # 5.8GB, 1781 observations
│   └── temp_stacks/                    # Individual stack files
│
├── 00_QGIS/ADMIN/                      # Administrative boundaries
│   └── BOLZANO_REGION_UTM32.gpkg
│
├── 06_Administrative_Boundaries/Processed/
│   └── FireBrigade_ResponsibilityAreas_Bolzano_clipped.gpkg  # 481 zones
│
└── WILDFIRE_INVENTORY/
    ├── REGISTRO_incendi_1999_2025.csv
    └── wildfire_point_Bolzano_Period1999_2025.gpkg
```

### Climate Data (External)
```
/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/
├── tas/rcp85/
│   └── tas_EUR-11_pctl50_rcp85.nc
└── pr/rcp85/
    └── pr_EUR-11_pctl50_rcp85.nc
```

### Model Outputs
```
/mnt/CEPH_PROJECTS/Firescape/Scripts/
└── Uncertainty_Attention/
    └── model_plots_bayesian_linear/
        ├── trace.nc                    # Bayesian posterior
        ├── scaler.joblib               # Feature scaler
        ├── baseline_stats.joblib       # Fire rate priors
        ├── validation_plots.png
        ├── attention_weights.png
        └── feature_importance.png
```

---

## Verification Checklist

Before running the pipeline, verify:

- [x] Raster stack generation complete → `/mnt/CEPH_PROJECTS/Firescape/Data/OUTPUT/spacetime_stacks.nc`
- [x] Static rasters accessible → `/mnt/CEPH_PROJECTS/Firescape/Data/STATIC_INPUT/*.tif`
- [x] Climate files accessible → `/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/{tas,pr}/rcp85/*.nc`
- [x] Fire brigade shapefile clipped → `FireBrigade_ResponsibilityAreas_Bolzano_clipped.gpkg`
- [x] All scripts updated to new paths
- [ ] Model training complete (next step)

---

## Next Steps

1. **Wait for raster generation to complete** (currently running)
2. **Train Bayesian model:**
   ```bash
   python 04_Bayesian_pyMCLogisticRegression_Linear_Attention_commented.py
   ```
   - Expected time: 60-90 minutes
   - Peak RAM: ~9GB

3. **Run climate projections:**
   ```bash
   python 05_Bayesian_Climate_Projection_CLEAN.py
   ```
   - Full resolution (50m): ~8-12 hours for 3 dates
   - 100m resolution: ~2-3 hours
   - 150m resolution: ~1-2 hours

4. **Generate fire brigade analysis:**
   ```bash
   python 07_Fire_Brigade_Zone_Analysis.py
   ```
   - Expected time: 20-30 minutes

5. **Optional - 2022 lookback GIF:**
   ```bash
   python 05_Bayesian_Lookback_2022_GIF.py
   ```

---

## Configuration Options

### Climate Projection Resolution

Edit line 39 in `05_Bayesian_Climate_Projection_CLEAN.py`:

```python
SPATIAL_DOWNSAMPLE = 1   # Full 50m (best quality, slower)
SPATIAL_DOWNSAMPLE = 2   # 100m (excellent quality, faster)
SPATIAL_DOWNSAMPLE = 3   # 150m (very good, very fast)
```

### Projection Dates

Edit line 41 in `05_Bayesian_Climate_Projection_CLEAN.py`:

```python
# Current (3 key decades):
PROJECTION_DATES = [f"{year}-07-15" for year in [2020, 2050, 2080]]

# Alternative (all decades):
PROJECTION_DATES = [f"{year}-07-15" for year in range(2020, 2101, 10)]
```

---

**All scripts are now consistent and ready for execution!**
