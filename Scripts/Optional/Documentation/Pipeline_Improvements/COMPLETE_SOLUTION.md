# Complete Solution Summary - Firescape Wildfire Risk Modeling

This document provides a complete overview of all improvements made to address your modeling questions.

## 🎯 Questions Addressed

### 1. ✅ **Mid-month date issue**
> "CANT BE A SINGLE DATE - need to run and aggregate at least 10 days"

**Solution**: 60-day temporal windows with multiple aggregation periods
- [extract_projection_features.py](03_Climate_Projections/extract_projection_features.py) - Feature extraction with temporal context
- Function: `load_dynamic_data_for_period()` (line 115-163)
- Computes cumulative statistics for 1d, 3d, 5d, 10d, 15d, 30d, 60d windows

### 2. ✅ **Landcover categorical handling**
> "How is landcoverfull managed as it is a categorical data?"

**Discovery**: Your data uses **Corine Land Cover Level 3** codes (28 classes, 3-digit codes 111-512)

**Solution**: Ordinal encoding based on fire risk (0-5 scale)
- [CORINE_LANDCOVER_FIRE_RISK_MAPPING.py](01_Data_Preparation/CORINE_LANDCOVER_FIRE_RISK_MAPPING.py) - Complete mapping
- [FINAL_LANDCOVER_FIX.md](FINAL_LANDCOVER_FIX.md) - Detailed fix instructions
- [apply_landcover_fix.py](apply_landcover_fix.py) - Automated fix script (UPDATED with Corine codes)

**Key mappings**:
- 312 (Coniferous) → 5 (very high risk)
- 324 (Transitional woodland-shrub) → 4 (high risk)
- 231 (Pastures - most common) → 2 (low risk)
- 335 (Glaciers) → 0 (no risk)

### 3. ✅ **Spatial and temporal visualization**
> "It would be nice at the end to use the output to draw some temporal and spatial plot"

**Solution**: Comprehensive visualization suite
- [visualize_risk_evolution.py](03_Climate_Projections/visualize_risk_evolution.py)
- Temporal evolution plots (all scenarios over time)
- Spatial risk maps (geographic distribution)
- Regional comparisons (different zones)
- Scenario comparison heatmaps

### 4. ✅ **Iterate different TARGET_SCENARIO**
> "It would be nice also to iterate the different TARGET_SCENARIO"

**Solution**: Automated scenario iteration framework
- [config_scenarios.py](03_Climate_Projections/config_scenarios.py) - Centralized configuration
- [run_all_scenarios.py](03_Climate_Projections/run_all_scenarios.py) - Orchestration script
- Processes all RCP 4.5 and RCP 8.5 scenarios (2030, 2050, 2070)

## 📁 New Files Created

### Climate Projection Pipeline
```
03_Climate_Projections/
├── config_scenarios.py                 # Scenario definitions (RCP, time periods, regions)
├── extract_projection_features.py      # Feature extraction (60-day windows)
├── run_all_scenarios.py                # Automated processing
├── visualize_risk_evolution.py         # Visualization suite
├── README.md                           # Full pipeline documentation
└── WORKFLOW_DIAGRAM.txt                # Visual workflow guide
```

### Landcover Encoding Fix
```
01_Data_Preparation/
├── CORINE_LANDCOVER_FIRE_RISK_MAPPING.py   # Mapping module (import this!)
└── CORINE_fire_risk_mapping.csv            # Reference table

Root/Scripts/
├── apply_landcover_fix.py              # Automated fix (UPDATED for Corine)
├── FINAL_LANDCOVER_FIX.md              # Corrected instructions (Corine codes)
├── LANDCOVER_ENCODING_ISSUE.md         # Original problem description
├── LANDCOVER_FIX_CHECKLIST.md          # Step-by-step checklist
└── fix_landcover_encoding.py           # Utilities (in 02_Model_Training/)
```

### Documentation
```
Scripts/
├── IMPROVEMENTS_SUMMARY.md             # High-level summary
├── README_COMPLETE_SOLUTION.md         # This file
└── WORKFLOW_DIAGRAM.txt                # Visual pipeline (duplicated in 03_)
```

## 🔧 Files Requiring Updates

**7 files** need the landcover fix applied:

1. `01_Data_Preparation/create_raster_stacks.py`
2. `02_Model_Training/train_relative_probability_model.py`
3. `02_Model_Training/train_Dask_PyMC_timeseries.py`
4. `03_Climate_Projections/extract_projection_features.py`
5. `04_Zone_Climate_Projections/project_zone_fire_risk.py`
6. `05_Lightning_Comparison/01_Data_Preparation/create_raster_stacks_with_lightning.py`
7. `05_Lightning_Comparison/02_Model_Training/train_relative_probability_model_with_lightning.py`

## 🚀 Quick Start Guide

### Step 1: Apply Landcover Fix (CRITICAL)

**Option A: Automated** (recommended)
```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts
python apply_landcover_fix.py
```

**Option B: Manual** (if you want more control)
Follow instructions in [LANDCOVER_FIX_CHECKLIST.md](LANDCOVER_FIX_CHECKLIST.md)

**Option C: Import mapping** (cleanest for new code)
```python
# In your script
from CORINE_LANDCOVER_FIRE_RISK_MAPPING import LANDCOVER_FIRE_RISK_ORDINAL, get_fire_risk

# Use in feature extraction
features['landcover_fire_risk'] = get_fire_risk(corine_code)
```

### Step 2: Retrain Model

```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts/02_Model_Training
python train_relative_probability_model.py
```

This will:
- Re-extract features with `landcover_fire_risk` instead of `landcoverfull`
- Train new model with correct categorical encoding
- Save updated model artifacts

### Step 3: Run Climate Projections

```bash
cd ../03_Climate_Projections

# Process all scenarios (RCP 4.5 and 8.5, all time periods)
python run_all_scenarios.py

# This will take 2-4 hours for all 6 scenarios
# Output: features_*.csv and predictions_*.csv for each scenario
```

### Step 4: Create Visualizations

```bash
# Generate all plots
python visualize_risk_evolution.py

# Output directory: OUTPUT/03_Climate_Projections/Visualizations/
# - temporal_evolution_all_scenarios.png
# - spatial_risk_maps_rcp85_2070.png
# - regional_comparison_rcp85_2050.png
# - scenario_comparison_heatmap.png
# - summary_statistics.csv
```

## 📊 Key Improvements Summary

| Issue | Before | After | Impact |
|-------|--------|-------|--------|
| **Temporal context** | Single day (day 15) | 60-day windows with 1d-60d aggregations | Matches training data structure |
| **Landcover encoding** | Categorical treated as continuous (wrong!) | Ordinal 0-5 based on Corine fire risk | Mathematically correct |
| **Scenario processing** | Manual, error-prone | Automated iteration (6 scenarios) | Consistent, reproducible |
| **Visualization** | None | 5 types of plots + summary table | Easy interpretation |
| **Code consistency** | Landcover handled differently across files | Single mapping used everywhere | Maintainable |

## 🔍 Understanding Your Landcover Data

Your dataset uses **Corine Land Cover Level 3** classification:

### Most Common Classes in Bolzano Province

1. **231 - Pastures** (476 occurrences) → Fire Risk: 2 (Low)
   - Managed, grazed, low fuel load

2. **324 - Transitional woodland-shrub** (392 occurrences) → Fire Risk: 4 (High)
   - Dense understory, transition zones, high fuel

3. **321 - Natural grasslands** (261 occurrences) → Fire Risk: 3 (Moderate)
   - Moderate fuel load, Alpine meadows

4. **333 - Sparsely vegetated areas** (260 occurrences) → Fire Risk: 3 (Moderate)
   - Low fuel but dry conditions

5. **243 - Agriculture with natural vegetation** (195 occurrences) → Fire Risk: 2 (Low)
   - Mixed use, partially managed

### Highest Risk Classes

- **312 - Coniferous forest** (148 occurrences) → Fire Risk: **5 (Very high)**
  - Alpine conifers (spruce, pine, larch), resinous, highly flammable

- **322 - Moors and heathland** (193 occurrences) → Fire Risk: 4 (High)
  - Dense shrubs, dry conditions

- **313 - Mixed forest** (75 occurrences) → Fire Risk: 4 (High)
  - Combination of fuel types

### Zero Risk Classes

- **335 - Glaciers/snow** (45 occurrences) → Fire Risk: 0
- **512 - Water bodies** (16 occurrences) → Fire Risk: 0
- **511 - Water courses** (3 occurrences) → Fire Risk: 0

**Total**: 28 unique Corine classes mapped to 6 fire risk levels (0-5)

## 📖 Documentation Reference

### Primary Documentation
- **[FINAL_LANDCOVER_FIX.md](FINAL_LANDCOVER_FIX.md)** - START HERE for landcover fix
- **[03_Climate_Projections/README.md](03_Climate_Projections/README.md)** - Complete climate projection pipeline
- **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)** - High-level overview

### Detailed Guides
- **[LANDCOVER_FIX_CHECKLIST.md](LANDCOVER_FIX_CHECKLIST.md)** - Step-by-step manual fix
- **[LANDCOVER_ENCODING_ISSUE.md](LANDCOVER_ENCODING_ISSUE.md)** - Problem explanation
- **[WORKFLOW_DIAGRAM.txt](WORKFLOW_DIAGRAM.txt)** - Visual pipeline

### Reference Files
- **[CORINE_LANDCOVER_FIRE_RISK_MAPPING.py](01_Data_Preparation/CORINE_LANDCOVER_FIRE_RISK_MAPPING.py)** - Import this for mapping
- **[CORINE_fire_risk_mapping.csv](01_Data_Preparation/CORINE_fire_risk_mapping.csv)** - Human-readable table

## ⚠️ Important Notes

### 1. Model Output Interpretation
- **Output**: Relative probability scores (0-1 range)
- **NOT**: Absolute fire counts or absolute probability
- **Use for**: Ranking, comparison, trend analysis
- **Don't use for**: Predicting exact number of fires

### 2. Corine Code Consistency
After applying the fix, ensure:
- All `STATIC_VARS` lists use `'landcover_fire_risk'` (not `'landcoverfull'`)
- All scripts import or copy the same `LANDCOVER_FIRE_RISK_ORDINAL` dictionary
- Feature names match between training and prediction

### 3. Data Requirements
Climate projection files must have:
- NetCDF format with `DATE`, `y`, `x` dimensions
- Daily temporal resolution
- Consistent CRS (EPSG:32632 or Lambert Azimuthal Equal Area)
- File naming: `{var}_{scenario}_{year}{month}.nc` or similar

### 4. Performance
- **Feature extraction**: ~10-30 min per scenario (depends on spatial resolution)
- **Predictions**: ~5-10 min per scenario
- **Total for 6 scenarios**: ~2-4 hours
- **Reduce time**: Increase spatial resolution (e.g., 2000m), reduce posterior samples

## 🐛 Troubleshooting

### "KeyError: 'landcoverfull'" or "KeyError: 'landcover_fire_risk'"
**Cause**: Mismatch between feature names in training vs prediction.

**Solution**: Ensure all scripts use the same feature name after fix. Retrain model if needed.

### "ValueError: Feature names mismatch"
**Cause**: Scaler trained with different features than being used for prediction.

**Solution**: Retrain model after applying landcover fix to all scripts.

### Unexpected Corine codes in raster
**Cause**: Your raster contains codes not in the mapping.

**Solution**:
```bash
# Check unique values in raster
python -c "
import rioxarray as rxr
import numpy as np
with rxr.open_rasterio('Data/STATIC_INPUT/landcoverfull.tif') as rds:
    unique_vals = np.unique(rds.values[~np.isnan(rds.values)])
    print('Corine codes in raster:', sorted(unique_vals.astype(int)))
"
```
Add missing codes to `LANDCOVER_FIRE_RISK_ORDINAL` dictionary.

### Missing climate data files
**Cause**: File paths in `config_scenarios.py` don't match actual data structure.

**Solution**: Edit `config_scenarios.py` to match your file organization:
```python
ClimateScenario(
    ...,
    temp_dir="/your/actual/path/to/temperature/data",
    precip_dir="/your/actual/path/to/precipitation/data"
)
```

## ✅ Verification Checklist

After completing all steps, verify:

- [ ] Landcover fix applied to all 7 files
- [ ] `LANDCOVER_FIRE_RISK_ORDINAL` uses Corine codes (111-512), not 1-10
- [ ] All `STATIC_VARS` lists have `'landcover_fire_risk'`
- [ ] Model retrained successfully
- [ ] Scaler contains `'landcover_fire_risk'` in `feature_names_in_`
- [ ] Climate projections run without errors
- [ ] Output CSVs contain `landcover_fire_risk` column (not `landcoverfull`)
- [ ] Visualizations generated successfully
- [ ] No Corine codes missing from mapping (check raster unique values)

## 📧 Support

For issues or questions:
1. Check relevant documentation files (listed above)
2. Review error messages carefully
3. Verify file paths and data availability
4. Check that all scripts use consistent feature names

## 🎓 Technical Details

### Corine Land Cover
- **Level 3**: 44 classes (28 present in Bolzano)
- **3-digit codes**: 1st digit = major category, 2nd = sub-category, 3rd = specific class
- **Examples**:
  - 3xx = Forests and semi-natural areas
  - 31x = Forests
  - 312 = Coniferous forest

### Fire Risk Ordinal Scale
- **0**: No fire possible (water, ice)
- **1**: Very low (urban, rock)
- **2**: Low (managed agriculture)
- **3**: Moderate (grassland, deciduous)
- **4**: High (shrubland, mixed forest)
- **5**: Very high (coniferous, Alpine conditions)

### Temporal Aggregation Windows
- **1d**: Immediate conditions
- **3d, 5d**: Short-term accumulation
- **10d, 15d**: Medium-term drying
- **30d, 60d**: Long-term drought/wetness

## 🏆 Final Result

A complete, consistent wildfire risk modeling pipeline that:
- ✅ Correctly handles categorical landcover data (Corine codes → ordinal fire risk)
- ✅ Uses proper temporal context (60-day windows, not single days)
- ✅ Automates scenario processing (all RCP scenarios)
- ✅ Provides comprehensive visualizations (spatial + temporal)
- ✅ Maintains consistency across all scripts
- ✅ Is well-documented and reproducible

---

**Created**: 2025-10-28
**Version**: 1.0 (Corrected with actual Corine codes)
**Status**: Ready for implementation
