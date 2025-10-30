# Firescape Project Structure - Absolute Probability Model

## Overview

This document describes the cleaned-up project structure after migrating to the **absolute probability model**.

**Last Updated**: 2025-10-20
**Status**: Production Ready

---

## 📁 Directory Structure

```
Firescape/
├── Scripts/                              # All executable scripts
│   ├── 01_Data_Preparation/
│   │   └── 01_create_raster_stacks.py   # Feature extraction (unchanged)
│   │
│   ├── 02_Model_Training/
│   │   └── 05_Bayesian_AbsoluteProbability_Regional.py  # ✅ ACTIVE MODEL
│   │
│   ├── 03_Climate_Projections/
│   │   ├── 05_Bayesian_Climate_Projection_ABSOLUTE.py   # ✅ Main projections
│   │   └── 06_Fire_Brigade_Climate_Projections_ABSOLUTE.py  # ✅ Zone analysis
│   │
│   └── 06_Validation/
│       └── 01_Absolute_Probability_Deep_Validation.py   # Validation analysis
│
├── Data/
│   ├── STATIC_INPUT/                     # Static rasters (elevation, land cover, etc.)
│   ├── WILDFIRE_INVENTORY/               # Historical fire data
│   │   └── wildfire_point_Bolzano_Period1999_2025.gpkg
│   │
│   └── OUTPUT/
│       ├── 01_Training_Data/
│       │   ├── spacetime_dataset.parquet
│       │   └── spacetime_stacks.nc
│       │
│       ├── 02_Model_AbsoluteProbability/           # ✅ ACTIVE MODEL
│       │   ├── trace_absolute.nc                   # Posterior samples
│       │   ├── scaler_absolute.joblib              # Feature scaler
│       │   ├── true_fire_stats.joblib              # Historical baseline
│       │   ├── temporal_groups.joblib              # Feature groups
│       │   ├── group_names.joblib                  # Group labels
│       │   ├── model_results.joblib                # Validation metrics
│       │   ├── absolute_probability_validation.png # ROC, PR, Calibration, Lift
│       │   └── temporal_validation.png             # Monthly/seasonal validation
│       │
│       ├── 04_Climate_Projections_Absolute/        # ✅ Climate projections
│       │   └── rcp85/
│       │       ├── fire_probability_20200715.tif
│       │       ├── fire_probability_20500715.tif
│       │       ├── fire_probability_20800715.tif
│       │       └── projection_summary_absolute.csv
│       │
│       ├── 06_Fire_Brigade_Projections_Absolute/   # ✅ Zone-level projections
│       │   ├── zone_projections_2020.csv
│       │   ├── zone_projections_2050.csv
│       │   ├── zone_projections_2080.csv
│       │   ├── provincial_summary.csv
│       │   └── provincial_summary.png
│       │
│       └── 06_Validation_Analysis/                 # Comprehensive validation
│           ├── posterior_distributions.png
│           ├── historical_fire_patterns.png
│           ├── temporal_validation_detailed.png
│           └── validation_summary.joblib
│
├── Documentation/                        # All documentation
│   ├── Absolute_Probability_Model_README.md        # 📖 START HERE
│   ├── QUICK_REFERENCE.md                          # Quick lookup
│   ├── Technical_Deep_Dive_Absolute_Probability.md # Deep dive (28 pages)
│   ├── Climate_Projections_AbsoluteProb_Update_Guide.md
│   ├── Climate_Scripts_Update_Summary.md
│   ├── Implementation_Summary_And_Ideas.md
│   └── PROJECT_STRUCTURE.md                        # This file
│
└── Archive/                              # Old/deprecated files
    └── Relative_Probability_Model_OLD/   # ⚠️ OLD - Reference only
        ├── README_ARCHIVE.md             # Why archived
        ├── Scripts/
        │   ├── 02_Model_Training/
        │   │   ├── 04_Bayesian_pyMCLogisticRegression_Linear_Attention_commented.py
        │   │   └── test_prior_validation.py
        │   └── 03_Climate_Projections/
        │       ├── 05_Bayesian_Climate_Projection_CLEAN.py
        │       ├── 06_Fire_Brigade_Climate_Projections.py
        │       ├── 05_Bayesian_Climate_Projection_MultiQuantile_Seasonal.py
        │       └── 05_Bayesian_Lookback_2022_GIF.py
        └── Data/OUTPUT/
            ├── 02_Model/                 # Old relative probability model
            ├── 04_Climate_Projections/   # Old projections
            └── 06_Fire_Brigade_Projections/
```

---

## 🎯 Active Scripts (Use These!)

### 1. Data Preparation
**File**: `Scripts/01_Data_Preparation/01_create_raster_stacks.py`
- Creates balanced case-control dataset
- Extracts spatiotemporal features
- **Status**: Unchanged (works with absolute probability model)

### 2. Model Training
**File**: `Scripts/02_Model_Training/05_Bayesian_AbsoluteProbability_Regional.py`
- Trains absolute probability model
- Comprehensive validation
- Saves all model artifacts
- **Runtime**: ~4 minutes
- **Output**: `Data/OUTPUT/02_Model_AbsoluteProbability/`

### 3. Climate Projections
**File**: `Scripts/03_Climate_Projections/05_Bayesian_Climate_Projection_ABSOLUTE.py`
- Main climate projection script
- Projects for 2020, 2050, 2080
- RCP scenarios (2.6, 4.5, 8.5)
- **Output**: `Data/OUTPUT/04_Climate_Projections_Absolute/`

### 4. Fire Brigade Analysis
**File**: `Scripts/03_Climate_Projections/06_Fire_Brigade_Climate_Projections_ABSOLUTE.py`
- Zone-specific projections
- Expected fires per brigade area
- **Output**: `Data/OUTPUT/06_Fire_Brigade_Projections_Absolute/`

### 5. Validation Analysis
**File**: `Scripts/06_Validation/01_Absolute_Probability_Deep_Validation.py`
- Deep dive validation
- Posterior analysis
- Historical pattern comparison
- **Output**: `Data/OUTPUT/06_Validation_Analysis/`

---

## 📊 Model Artifacts

**Location**: `Data/OUTPUT/02_Model_AbsoluteProbability/`

| File | Size | Description |
|------|------|-------------|
| `trace_absolute.nc` | 3.9 MB | Posterior samples (8000 draws) |
| `scaler_absolute.joblib` | 2.6 KB | Feature standardization |
| `true_fire_stats.joblib` | 498 B | Historical baseline (8.5 fires/year) |
| `temporal_groups.joblib` | 318 B | Feature group indices |
| `group_names.joblib` | 186 B | Group labels |
| `model_results.joblib` | 3.0 KB | Validation metrics |
| `absolute_probability_validation.png` | 593 KB | Performance plots |
| `temporal_validation.png` | 140 KB | Temporal validation |

**Total**: 4.7 MB

---

## 📖 Documentation (Read These!)

### Start Here
1. **`QUICK_REFERENCE.md`** - Quick lookup (5 min read)
2. **`Absolute_Probability_Model_README.md`** - User guide (15 min read)

### Deep Dive
3. **`Technical_Deep_Dive_Absolute_Probability.md`** - Full technical details (28 pages)

### Updating Scripts
4. **`Climate_Projections_AbsoluteProb_Update_Guide.md`** - How to update scripts
5. **`Climate_Scripts_Update_Summary.md`** - Current status

### Planning
6. **`Implementation_Summary_And_Ideas.md`** - Future enhancements

---

## 🗄️ Archive (Don't Use!)

**Location**: `Archive/Relative_Probability_Model_OLD/`

**Contains**:
- Old relative probability scripts
- Old model outputs
- Scripts to be updated

**Status**: Reference only - DO NOT USE for new work

**See**: `Archive/Relative_Probability_Model_OLD/README_ARCHIVE.md`

---

## 🚀 Typical Workflow

### Step 1: Train Model (Once)
```bash
cd /mnt/CEPH_PROJECTS/Firescape
python Scripts/02_Model_Training/05_Bayesian_AbsoluteProbability_Regional.py
```

### Step 2: Generate Climate Projections
```bash
# Main projections
python Scripts/03_Climate_Projections/05_Bayesian_Climate_Projection_ABSOLUTE.py

# Fire brigade zones
python Scripts/03_Climate_Projections/06_Fire_Brigade_Climate_Projections_ABSOLUTE.py
```

### Step 3: Validate (Optional)
```bash
python Scripts/06_Validation/01_Absolute_Probability_Deep_Validation.py
```

### Step 4: View Results
```bash
# Model validation
eog Data/OUTPUT/02_Model_AbsoluteProbability/*.png

# Climate projections
ls -lh Data/OUTPUT/04_Climate_Projections_Absolute/rcp85/

# Zone analysis
cat Data/OUTPUT/06_Fire_Brigade_Projections_Absolute/provincial_summary.csv
```

---

## 📝 Quick Commands

### List active scripts
```bash
find Scripts -name "*ABSOLUTE*.py" -o -name "05_Bayesian_AbsoluteProbability*.py"
```

### Check model trained
```bash
ls -lh Data/OUTPUT/02_Model_AbsoluteProbability/
```

### View latest projections
```bash
ls -lt Data/OUTPUT/04_Climate_Projections_Absolute/rcp85/*.tif | head -5
```

### Check documentation
```bash
ls -1 Documentation/*.md
```

---

## ⚠️ Important Notes

### DO Use
- ✅ Scripts with `_ABSOLUTE` suffix
- ✅ `05_Bayesian_AbsoluteProbability_Regional.py`
- ✅ Model artifacts in `02_Model_AbsoluteProbability/`
- ✅ Documentation in `Documentation/`

### DON'T Use
- ❌ Scripts in `Archive/` directory
- ❌ Old `02_Model/` output directory
- ❌ Scripts without `_ABSOLUTE` suffix (except data preparation)

---

## 🔄 Migration Status

| Component | Status | Location |
|-----------|--------|----------|
| Model Training | ✅ Complete | `02_Model_Training/05_Bayesian_AbsoluteProbability_Regional.py` |
| Main Climate Projection | ✅ Complete | `03_Climate_Projections/05_Bayesian_Climate_Projection_ABSOLUTE.py` |
| Fire Brigade Analysis | ✅ Complete | `03_Climate_Projections/06_Fire_Brigade_Climate_Projections_ABSOLUTE.py` |
| Deep Validation | ✅ Complete | `06_Validation/01_Absolute_Probability_Deep_Validation.py` |
| Multi-Quantile Seasonal | ⏳ To Update | Archived - needs update |
| Historical Lookback GIF | ⏳ To Update | Archived - needs update |

---

## 📞 Need Help?

1. **Quick lookup**: `Documentation/QUICK_REFERENCE.md`
2. **User guide**: `Documentation/Absolute_Probability_Model_README.md`
3. **Technical details**: `Documentation/Technical_Deep_Dive_Absolute_Probability.md`
4. **Update scripts**: `Documentation/Climate_Scripts_Update_Summary.md`

---

## 🎯 Key Metrics

**Model Performance**:
- ROC-AUC: 0.766
- Monthly correlation: **0.942**
- Seasonal correlation: **0.998**

**Baseline**:
- 8.5 fires/year (historical)
- 0.0233 fires/day
- Based on 227 fires (1999-2025)

**Production Status**: ✅ Ready

---

**Last Updated**: 2025-10-20
**Version**: 1.0 (Absolute Probability)
**Maintainer**: See Documentation
