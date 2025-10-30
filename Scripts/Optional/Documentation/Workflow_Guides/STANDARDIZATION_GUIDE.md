# Firescape Model Prediction Standardization Guide

## Overview

All scripts that apply the trained Bayesian wildfire model now use a **shared prediction framework** to ensure consistency across:
- Historical validation (hindcasting)
- Future climate projections
- Warning level analysis
- Model comparisons

## Key Principle

**The prediction methodology must match the training data structure:**
- Training used 32×32 pixel chips with spatial window averaging
- Predictions use the same feature extraction approach
- Zone-level predictions aggregate from grid-level predictions

## Shared Module: `shared_prediction_utils.py`

Located at: `Scripts/shared_prediction_utils.py`

### Core Components:

1. **FirescapeModel** - Wrapper for trained model
   - Loads trace, scaler, temporal groups
   - Handles hierarchical beta structure
   - Pre-computes posterior means for speed

2. **load_static_rasters()** - Standardized raster loading
   - Aligns all rasters to common template
   - Handles resampling to target resolution
   - Ensures consistent grid dimensions

3. **load_climate_timeseries()** - Climate data loading
   - Loads temperature and precipitation
   - Aligns to template grid
   - Supports caching for performance

4. **extract_features_grid()** - Feature extraction
   - **Matches training methodology**
   - Static features: Grid cell values
   - Dynamic features: Cumulative mean/max over time windows
   - Returns DataFrame with all 40 features

5. **predict_grid_to_zones()** - Complete workflow
   - Load data → Extract features → Predict → Aggregate to zones
   - Single function call for convenience

6. **ClimateCache** - Simple LRU cache
   - Reduces redundant file I/O
   - Especially helpful for overlapping time windows

## Updated Scripts

### 1. Historical Lookback Validation (v2)
**File**: `Scripts/03_Threshold_Optimization/historical_lookback_validation_v2.py`

**Changes:**
- Uses `FirescapeModel` for standardized prediction
- Uses shared feature extraction
- Grid-level predictions (200m) → zone aggregation
- ~9 minute runtime for 160 sampled days

**Usage:**
```bash
python Scripts/03_Threshold_Optimization/historical_lookback_validation_v2.py
```

### 2. Zone Climate Projections (TODO)
**File**: `Scripts/04_Zone_Climate_Projections/project_zone_fire_risk.py`

**Needs update to:**
- Import from `shared_prediction_utils`
- Use `predict_grid_to_zones()`
- Ensure feature extraction matches training

### 3. Warning Level Evolution (TODO)
**File**: `Scripts/04_Zone_Climate_Projections/analyze_warning_level_evolution.py`

**Needs update to:**
- Import from `shared_prediction_utils`
- Use standardized prediction workflow
- Apply warning thresholds from optimization

## Feature Extraction Details

### Static Features (12 total):
From grid cell values (or small windows):
- Topography: tri, northness, slope, aspect, nasadem, eastness
- Vegetation: treecoverdensity, landcoverfull, flammability
- Human: distroads, walking_time_to_bldg, walking_time_to_elec_infra

### Dynamic Features (28 total):
Cumulative statistics over time windows (1, 3, 5, 10, 15, 30, 60 days):
- Temperature (T): 14 features (7 windows × 2 stats: mean, max)
- Precipitation (P): 14 features (7 windows × 2 stats: mean, max)

**Total**: 40 features (matches training exactly)

## Scale Considerations

### Training Scale:
- **Point-based**: Each sample is a specific location + date
- **Spatial context**: 32×32 pixel chip (~3.2 km × 3.2 km at 100m resolution)
- **Feature window**: 4×4 pixel average (~400m)
- **Purpose**: Learn local fire risk patterns

### Prediction Scale:
- **Grid-based**: Predictions at 200m resolution
- **Zone aggregation**: Average across ~15 km² zones
- **Purpose**: Operational zone-level risk assessment

### Scale Mismatch:
Fire brigade zones (15 km²) are **100× larger** than training windows (0.16 km²).

**Solution**:
1. Predict at grid level (200m cells)
2. Aggregate to zones: `zone_risk = mean(cell_predictions)`
3. Interpret as: "Average risk across zone area"

This is appropriate for operational use where fire brigades need zone-level summaries.

## Performance Optimization

### Sampling Strategies:
- `all_days`: Process every day (slow, ~4 hours for 10 years)
- `weekly`: Sample weekly (moderate, ~10 minutes)
- `fire_season_monthly`: Sample 2×/month in fire season (fast, ~9 minutes) ⭐ Recommended
- `monthly`: Sample 2×/month all year (very fast, ~7 minutes)

### Caching:
- Climate data cached (up to 120 files)
- Reduces redundant I/O for overlapping time windows
- Especially beneficial for consecutive dates

### Checkpointing:
- Progress saved every 20 dates
- Can resume if interrupted
- Checkpoint file automatically removed on completion

## Migration Checklist

To update an existing prediction script:

- [ ] Import from `shared_prediction_utils`
- [ ] Replace model loading with `FirescapeModel(MODEL_DIR)`
- [ ] Replace static raster loading with `load_static_rasters()`
- [ ] Replace prediction code with `predict_grid_to_zones()`
- [ ] Test output format matches expectations
- [ ] Verify predictions are reasonable (0-1 range, spatial patterns make sense)
- [ ] Update documentation

## Validation

To verify predictions are correct:

1. **Feature check**: Ensure 40 features extracted
2. **Value check**: Predictions in [0, 1] range
3. **Spatial check**: Higher risk in expected areas (steep slopes, dry conditions)
4. **Temporal check**: Risk varies with seasons (higher in summer/dry periods)
5. **Consistency check**: Similar inputs → similar predictions

## Support

For questions or issues with standardized predictions:
1. Check this guide first
2. Review `shared_prediction_utils.py` docstrings
3. Compare with training script: `Scripts/02_Model_Training/train_relative_probability_model.py`
4. Check that feature names match exactly: `scaler.feature_names_in_`

## Future Improvements

Potential enhancements (not yet implemented):
- [ ] Point-based sampling within zones (more accurate but slower)
- [ ] Uncertainty quantification (full posterior predictions, not just means)
- [ ] Parallel processing for faster predictions
- [ ] GPU acceleration for large grids
- [ ] Online prediction API
