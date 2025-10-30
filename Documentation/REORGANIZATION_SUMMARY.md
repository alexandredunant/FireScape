# Project Reorganization Summary
**Date:** October 20, 2025
**Purpose:** Clean, well-documented structure for sharing and reproducibility

---

## Changes Implemented

### 1. Directory Structure Created

**New Data Organization:**
```
Data/OUTPUT/
├── 01_Training_Data/          # Spatial-temporal training dataset
│   ├── spacetime_stacks.nc (5.8 GB)
│   └── spacetime_dataset.parquet
├── 02_Model/                  # Trained Bayesian model (will be created)
├── 03_Model_Validation/       # Diagnostic plots (will be created)
├── 04_Climate_Projections/    # Future fire risk maps
│   └── rcp85/
│       ├── pctl25/            # 25th percentile (conservative)
│       ├── pctl50/            # 50th percentile (median)
│       ├── pctl75/            # 75th percentile
│       └── pctl99/            # 99th percentile (extreme)
├── 05_Fire_Brigade_Analysis/  # Operational planning outputs
├── 06_Figures/                # Publication-ready figures
└── 07_Historical_Analysis/    # Optional retrospective analysis
```

**New Scripts Organization:**
```
Scripts/
├── 01_Data_Preparation/
│   ├── 01_create_raster_stacks.py
│   └── clip_fire_brigade_to_bolzano.py
├── 02_Model_Training/
│   ├── 04_Bayesian_pyMCLogisticRegression_Linear_Attention_commented.py
│   └── test_prior_validation.py
├── 03_Climate_Projections/
│   ├── 05_Bayesian_Climate_Projection_MultiQuantile_Seasonal.py (MAIN)
│   ├── 05_Bayesian_Climate_Projection_CLEAN.py (alternative)
│   └── 05_Bayesian_Lookback_2022_GIF.py (historical)
├── 04_Fire_Brigade_Analysis/
│   └── 07_Fire_Brigade_Zone_Analysis.py
└── 05_Utilities/
    ├── monitor_progress.py
    └── estimate_climate_projection_time.py
```

**New Documentation Structure:**
```
Documentation/
├── REORGANIZATION_SUMMARY.md (this file)
├── PROJECT_ORGANIZATION_PLAN.md (detailed plan)
├── PATH_UPDATES_SUMMARY.md (path changes)
├── EXECUTION_PLAN.md (original workflow)
├── RAM_OPTIMIZATION_GUIDE.md (memory optimization)
└── CLIMATE_PROJECTION_RUNTIME_REPORT.md (performance analysis)
```

**Archive:**
```
Archive/
├── deprecated_scripts/
│   ├── 05_Bayesian_Climate_Projection.py (old version)
│   ├── 05_Bayesian_Climate_Projection_OPTIMIZED.py (old version)
│   ├── 00_balanced_pts_dataset.py
│   ├── access_lightning_data.py
│   └── master_task_executor.py
└── old_outputs_uncertainty_attention/
    └── (previous run outputs)
```

---

### 2. Files Moved

**Training Data:**
- ✓ `spacetime_stacks.nc` → `Data/OUTPUT/01_Training_Data/`
- ✓ `spacetime_dataset.parquet` → `Data/OUTPUT/01_Training_Data/`

**Scripts:**
- ✓ Data preparation scripts → `Scripts/01_Data_Preparation/`
- ✓ Model training scripts → `Scripts/02_Model_Training/`
- ✓ Climate projection scripts → `Scripts/03_Climate_Projections/`
- ✓ Fire brigade script → `Scripts/04_Fire_Brigade_Analysis/`
- ✓ Utility scripts → `Scripts/05_Utilities/`

**Documentation:**
- ✓ All .md files → `Documentation/`

**Deprecated:**
- ✓ Old scripts → `Archive/deprecated_scripts/`
- ✓ Old outputs → `Archive/old_outputs_uncertainty_attention/`

---

### 3. Script Paths Updated

All scripts now use consistent path variables pointing to new locations:

**Model Training Script:**
- Input: `Data/OUTPUT/01_Training_Data/spacetime_stacks.nc`
- Output: `Data/OUTPUT/02_Model/`

**Climate Projection Scripts:**
- Model Input: `Data/OUTPUT/02_Model/`
- Output: `Data/OUTPUT/04_Climate_Projections/rcp85/{quantile}/`

**Lookback Script:**
- Model Input: `Data/OUTPUT/02_Model/`
- Output: `Data/OUTPUT/07_Historical_Analysis/lookback_2022_monthly/`

**Fire Brigade Script:**
- Model Input: `Data/OUTPUT/02_Model/`
- Output: `Data/OUTPUT/05_Fire_Brigade_Analysis/`

---

### 4. Documentation Created

**Main Project README:**
- Location: `/mnt/CEPH_PROJECTS/Firescape/README.md`
- Contents: Project overview, quick start, structure, key features

**Scripts README:**
- Location: `/mnt/CEPH_PROJECTS/Firescape/Scripts/README.md`
- Contents: Detailed execution order, configuration options, troubleshooting

**Static Input README:**
- Location: `/mnt/CEPH_PROJECTS/Firescape/Data/STATIC_INPUT/README.md`
- Contents: Data sources, processing notes, why each feature matters

---

## Benefits of New Structure

✓ **Clear Workflow**: Numbered phases show execution order
✓ **Easy to Share**: Well-documented, self-explanatory structure
✓ **Reproducible**: Everything needed is in one place
✓ **Maintainable**: Old versions archived, not mixed with current
✓ **Publishable**: Ready for data/code repository (Zenodo, GitHub)

---

## Next Steps

### Ready to Execute:

1. **Model Training** (60-90 minutes):
   ```bash
   cd Scripts/02_Model_Training
   python 04_Bayesian_pyMCLogisticRegression_Linear_Attention_commented.py
   ```

2. **Climate Projections** (5.7 days):
   ```bash
   cd Scripts/03_Climate_Projections
   python 05_Bayesian_Climate_Projection_MultiQuantile_Seasonal.py
   ```

3. **Fire Brigade Analysis** (20-30 minutes):
   ```bash
   cd Scripts/04_Fire_Brigade_Analysis
   python 07_Fire_Brigade_Zone_Analysis.py
   ```

### Optional:

4. **Historical Lookback** (2-3 hours):
   ```bash
   cd Scripts/03_Climate_Projections
   python 05_Bayesian_Lookback_2022_GIF.py
   ```

---

## Verification Checklist

- [x] Directory structure created
- [x] Training data moved to new location
- [x] All scripts organized into numbered subdirectories
- [x] Deprecated scripts archived
- [x] Script paths updated to new OUTPUT structure
- [x] README files created for key directories
- [x] Documentation consolidated
- [x] Old output directories archived
- [ ] Model training complete (next step)
- [ ] Climate projections complete
- [ ] Fire brigade analysis complete

---

## Files Summary

**Total Active Scripts:** 10
- Data Preparation: 2
- Model Training: 2
- Climate Projections: 3
- Fire Brigade: 1
- Utilities: 2

**Total Archived Scripts:** 5

**Documentation Files:** 7

**Total Project Size:**
- Training data: 5.8 GB
- Expected projection outputs: ~40-50 GB (684 dates × 4 quantiles)

---

**Reorganization completed:** October 20, 2025
**Status:** ✓ Ready for model training and projections
