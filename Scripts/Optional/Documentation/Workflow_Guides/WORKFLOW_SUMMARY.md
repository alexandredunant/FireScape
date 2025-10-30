# Firescape Workflow - Refactored (October 2025)

## Overview

This directory contains the refactored Firescape wildfire prediction workflow, cleaned up on 2025-10-28 to ensure **NO fudge factors** are used in climate projections.

## Key Principle

**All climate projections use PURE RELATIVE RISK scores from the Bayesian model.**
- ✅ NO post-hoc scaling factors applied to predictions
- ✅ NO conversion to "absolute" probabilities
- ✅ Results are interpretable as relative fire risk comparisons only

## Workflow Structure

### 01_Data_Preparation/
Prepare spatial-temporal input data for model training

- `clip_fire_brigade_to_bolzano.py` - Clip fire brigade responsibility zones to Bolzano
- `create_raster_stacks.py` - Stack multi-temporal rasters
- `create_spacetime_dataset.py` - Create space-time training dataset

### 02_Model_Training/
Train Bayesian attention-based wildfire model

- `train_relative_probability_model.py` - Train relative probability model with:
  - **NEW**: Prior distribution visualization before training
  - Bayesian inference with uncertainty quantification
  - Attention mechanism for temporal feature weighting
  - Model outputs relative risk scores (0-1 scale)

**Key output**: Trained model produces relative fire risk, NOT absolute probabilities

### 03_Temporal_Validation (Built into Training)

**Temporal validation is already performed during model training** and displayed in the training output.

No separate script needed - the training script outputs:
- **Monthly fire counts**: Actual vs Predicted (12 months)
- **Seasonal fire counts**: Actual vs Predicted (4 seasons)
- **Correlation metrics**: Pearson R, R², RMSE, MAE
- **Visualization**: Comparison plots showing temporal fit

**Key metrics from your training run:**
```
MONTHLY:
  Pearson R (trend): 0.915
  R² (magnitude fit): 0.577

SEASONAL:
  Pearson R (trend): 0.960
  R² (magnitude fit): 0.576
```

This temporal validation is **sufficient** for assessing whether the model captures fire seasonality. A separate zone-level lookback validation would be too computationally expensive (hours of runtime) for marginal additional insight.

### 04_Zone_Climate_Projections/
Generate climate-driven fire risk projections by zone

- `project_zone_fire_risk.py` - Project future fire risk under climate change:
  - **NO SCALING FACTORS** - pure relative risk only
  - Multiple climate quantiles (25th, 50th, 99th percentile)
  - Decadal projections (2020-2080, every 10 years)
  - Year-round coverage: Winter (Jan-Feb), Spring (Mar-Apr), Summer (Jul-Aug), Fall (Oct-Nov)
  - Zone-level risk aggregation
  - Uncertainty quantification

**Output interpretation**:
- Relative risk scores for comparing zones
- Temporal trends showing risk changes
- Seasonal patterns including winter fire risk
- Spatial patterns of high-risk zones
- **NOT** absolute fire count predictions

**Note**: Includes winter months because South Tyrol experiences dry winter conditions that create fire risk

## Archive/

Previous versions of scripts archived on 2025-10-28 including:
- `20251028_PreRefactor/`
  - `03_Climate_Projections/` - Old projection scripts (some used scaling factors)
  - `04_Fire_Brigade_Analysis/` - Old zone analysis scripts
  - `05_Utilities/` - Old utility scripts
  - `06_Validation/` - Old validation scripts
  - `03_Zone_Lookback_Validation_TooSlow/` - Zone-level validation (too computationally expensive)
  - `train_relative_probability_model_backup.py` - Pre-refactor training script
  - Reference documents (README.md, text files)

## Critical Change: NO Fudge Factors

### What Changed

**Before (archived scripts)**:
```python
# Old approach - analyze_fire_brigade_zones.py:223
SCALING_FACTOR = model_results['temporal_validation']['monthly_scale']
predictions_mean_absolute = predictions_mean * SCALING_FACTOR  # ❌ FUDGE FACTOR
```

**After (current scripts)**:
```python
# New approach - project_zone_fire_risk.py
prob_pred = 1 / (1 + np.exp(-logit_pred))  # ✅ PURE RELATIVE RISK
# NO multiplication by scaling factors!
```

### Why This Matters

1. **Scientific integrity**: Scaling factors don't generalize to future climate
2. **Transparency**: Clear that outputs are relative risk, not absolute counts
3. **Consistency**: All projections use same model output interpretation
4. **Uncertainty honesty**: We acknowledge limitations rather than hide them

## How to Use

### Training
```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts/02_Model_Training
python train_relative_probability_model.py
```

**Output**:
- Trained Bayesian model with uncertainty
- Prior distribution visualization (NEW!)
- Validation metrics showing temporal patterns

### Validation
```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts/03_Zone_Lookback_Validation
python validate_zone_predictions.py
```

**Output**:
- Zone-level correlation between predicted risk and actual fires
- Spatial maps showing model performance by zone
- Identifies where model works well vs. poorly

### Climate Projections
```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts/04_Zone_Climate_Projections
python project_zone_fire_risk.py
```

**Output**:
- CSV with zone-level relative risk by year/month/quantile
- Pure relative risk scores (NO scaling factors)
- Uncertainty estimates for all projections
- Trend analysis showing risk changes over time

## Interpreting Results

### ✅ Valid Interpretations

- "Zone A has 2x higher risk than Zone B in 2050"
- "August shows 30% higher risk than March"
- "Risk increases 50% from 2020 to 2080 under RCP8.5"
- "High-quantile climate scenarios show 3x baseline risk"

### ❌ Invalid Interpretations

- "Zone A will have exactly 15 fires in 2050" (NO - we don't predict counts)
- "The absolute probability of fire is 0.034" (NO - this is relative, not absolute)
- "We can calculate expected fire frequency" (NO - no absolute calibration)

## Data Requirements

### Climate Data
- Temperature: `/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/tas/rcp85/`
- Precipitation: `/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/pr/rcp85/`
- Format: NetCDF with `tas` and `pr` variables
- Quantiles: pctl25, pctl50, pctl99

### Static Rasters
- Location: `/mnt/CEPH_PROJECTS/Firescape/Data/STATIC_INPUT/`
- Required: DEM, slope, aspect, land cover, infrastructure, etc.
- Format: GeoTIFF in UTM32N

### Fire Brigade Zones
- Location: `/mnt/CEPH_PROJECTS/Firescape/Data/06_Administrative_Boundaries/Processed/`
- File: `FireBrigade_ResponsibilityAreas_Bolzano_clipped.gpkg`
- Must include zone ID and name fields

## Technical Details

### Model
- **Type**: Bayesian logistic regression with attention mechanism
- **Framework**: PyMC (MCMC inference)
- **Output**: Relative fire risk probabilities (0-1 scale)
- **Uncertainty**: Quantified via posterior samples

### Features
- **Static**: Topography, vegetation, infrastructure (12 variables)
- **Dynamic**: Temperature and precipitation (14 temporal aggregations)
- **Temporal**: 60-day lookback with multiple windows (1, 3, 5, 10, 15, 30, 60 days)

### Spatial Resolution
- **Training**: 50m native resolution
- **Projections**: 200m (for computational efficiency)
- **Aggregation**: Zone-level summaries

## References

- Training script: [train_relative_probability_model.py](02_Model_Training/train_relative_probability_model.py)
- Zone validation: [validate_zone_predictions.py](03_Zone_Lookback_Validation/validate_zone_predictions.py)
- Climate projections: [project_zone_fire_risk.py](04_Zone_Climate_Projections/project_zone_fire_risk.py)

## Changelog

### 2025-10-28: Major Refactor
- ✅ Added prior visualization to training script
- ✅ Archived old scripts with scaling factors
- ✅ Created zone-level lookback validation
- ✅ Created clean zone-level climate projections (NO scaling factors)
- ✅ Verified NO fudge factors in prediction pipeline
- ✅ Cleaned up directory structure
- ✅ Updated all paths and documentation

---

**Questions?** See `01_Data_Preparation/IWouldLikeTo.md` for workflow requirements.
