# Firescape Model Improvements Summary

This document summarizes the improvements made to address your questions about the wildfire risk modeling pipeline.

## Questions Addressed

### 1. ✅ Mid-month date issue ("CANT BE A SINGLE DATE")

**Problem**: Climate projections used single-day snapshots (day 15 of each month), but the model was trained on 60-day temporal windows.

**Solution**: [extract_projection_features.py](03_Climate_Projections/extract_projection_features.py)
- For each mid-month date (e.g., 2050-07-15), load **60 days** of climate data
- Compute temporal aggregations (cumulative means/maxs) for 1d, 3d, 5d, 10d, 15d, 30d, 60d windows
- This matches the training data structure

**Key function**: `load_dynamic_data_for_period()` (line 115-163)

```python
# Example: Extract features for July 15, 2050
center_date = datetime(2050, 7, 15)
dynamic_data = load_dynamic_data_for_period(
    scenario,
    center_date,
    n_days_back=60  # Load June 16 to July 15
)
# Then compute temporal features from this 60-day window
```

### 2. ✅ Landcover categorical handling

**Problem**: `landcoverfull` is categorical (forest, grassland, etc.) but was treated as continuous numeric.

**Solution**: [fix_landcover_encoding.py](02_Model_Training/fix_landcover_encoding.py) + [LANDCOVER_ENCODING_ISSUE.md](LANDCOVER_ENCODING_ISSUE.md)
- **Ordinal encoding** based on fire risk (0-5 scale)
- Maps vegetation types to flammability levels
- Preserves single-feature simplicity

**How it's currently handled** (needs fixing):
- [create_raster_stacks.py](01_Data_Preparation/create_raster_stacks.py:95): Uses mode (correct for aggregation)
- [train_relative_probability_model.py](02_Model_Training/train_relative_probability_model.py:95): Uses mode but treats as continuous (incorrect)

**Fix required**: See [LANDCOVER_ENCODING_ISSUE.md](LANDCOVER_ENCODING_ISSUE.md) for detailed implementation steps.

### 3. ✅ Spatial and temporal visualization

**Solution**: [visualize_risk_evolution.py](03_Climate_Projections/visualize_risk_evolution.py)

Creates comprehensive visualizations:

1. **Temporal evolution plot**: Risk trends over time across all scenarios
   - Line plots with confidence intervals
   - Comparison of RCP 4.5 vs RCP 8.5
   - High-risk day frequency

2. **Spatial risk maps**: Geographic distribution of risk
   - Scatter plots colored by risk level
   - Side-by-side comparison of different time periods
   - Region-specific views

3. **Regional comparison**: Risk evolution by region
   - Time series for different geographic areas
   - Alta Val Venosta, Val Badia, Oltradige, Alta Pusteria
   - Spatial variability shown as error bands

4. **Scenario comparison heatmap**: Monthly risk by scenario
   - Matrix view of all scenarios × months
   - Easy identification of highest risk periods
   - Quantitative values displayed

### 4. ✅ Iterate different TARGET_SCENARIO

**Solution**: [config_scenarios.py](03_Climate_Projections/config_scenarios.py) + [run_all_scenarios.py](03_Climate_Projections/run_all_scenarios.py)

**Scenario configuration** (centralized):
```python
SCENARIOS = [
    ClimateScenario(name="historical", rcp="historical", period=(1999, 2025), ...),
    ClimateScenario(name="rcp45_2030", rcp="rcp45", period=(2021, 2040), ...),
    ClimateScenario(name="rcp45_2050", rcp="rcp45", period=(2041, 2060), ...),
    ClimateScenario(name="rcp45_2070", rcp="rcp45", period=(2061, 2080), ...),
    ClimateScenario(name="rcp85_2030", rcp="rcp85", period=(2021, 2040), ...),
    ClimateScenario(name="rcp85_2050", rcp="rcp85", period=(2041, 2060), ...),
    ClimateScenario(name="rcp85_2070", rcp="rcp85", period=(2061, 2080), ...)
]
```

**Automated iteration**:
```python
# Process all scenarios automatically
for scenario in SCENARIOS:
    if scenario.name != 'historical':
        process_scenario(scenario, ...)
```

**Regional subsets** also supported:
```python
REGIONS = {
    'full_province': {...},
    'alta_val_venosta': {...},
    'val_badia': {...},
    'oltradige': {...},
    'alta_pusteria': {...}
}
```

## New File Structure

```
Scripts/
├── 01_Data_Preparation/
│   └── create_raster_stacks.py (existing, needs landcover fix)
│
├── 02_Model_Training/
│   ├── train_relative_probability_model.py (existing, needs landcover fix)
│   ├── train_Dask_PyMC_timeseries.py (existing, needs landcover fix)
│   └── fix_landcover_encoding.py (NEW - landcover encoding utilities)
│
├── 03_Climate_Projections/ (NEW DIRECTORY)
│   ├── config_scenarios.py (NEW - scenario definitions)
│   ├── extract_projection_features.py (NEW - feature extraction with temporal aggregation)
│   ├── run_all_scenarios.py (NEW - orchestration script)
│   ├── visualize_risk_evolution.py (NEW - comprehensive visualizations)
│   └── README.md (NEW - documentation)
│
├── LANDCOVER_ENCODING_ISSUE.md (NEW - detailed fix instructions)
└── IMPROVEMENTS_SUMMARY.md (THIS FILE)
```

## Usage Workflow

### Step 1: Fix Landcover Encoding (Recommended)

```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts/02_Model_Training

# Review the fix
cat ../LANDCOVER_ENCODING_ISSUE.md

# Apply changes to:
# - train_relative_probability_model.py (lines 95-104)
# - train_Dask_PyMC_timeseries.py (lines 108-122)

# Retrain model
python train_relative_probability_model.py
```

### Step 2: Configure Climate Scenarios

```bash
cd ../03_Climate_Projections

# Review/edit scenario configuration
nano config_scenarios.py

# Check data availability
python config_scenarios.py
```

### Step 3: Run All Scenarios

```bash
# Process all scenarios (extracts features + generates predictions)
python run_all_scenarios.py

# This will:
# - Load trained model
# - For each scenario:
#     - Extract features with 60-day temporal windows
#     - Generate predictions with uncertainty
#     - Save results
```

**Expected runtime**:
- Feature extraction: ~10-30 min per scenario (depends on spatial resolution)
- Predictions: ~5-10 min per scenario
- Total: ~2-4 hours for all 6 scenarios

### Step 4: Create Visualizations

```bash
# Generate all plots
python visualize_risk_evolution.py

# Output: OUTPUT/03_Climate_Projections/Visualizations/
```

## Key Features

### 1. Proper Temporal Context
- ✅ Loads 60-day windows (not single days)
- ✅ Computes temporal aggregations matching training data
- ✅ Handles monthly file structure automatically

### 2. Categorical Encoding
- ⚠️ Currently broken (treats categorical as continuous)
- ✅ Fix provided with ordinal fire risk encoding
- ✅ Alternative one-hot encoding also available

### 3. Multi-Scenario Support
- ✅ Automated processing of 6 scenarios (RCP 4.5 and 8.5)
- ✅ Centralized configuration
- ✅ Consistent output structure
- ✅ Regional subsets supported

### 4. Comprehensive Visualization
- ✅ Temporal evolution plots
- ✅ Spatial risk maps
- ✅ Regional comparisons
- ✅ Scenario comparison heatmaps
- ✅ Summary statistics tables

### 5. Uncertainty Quantification
- ✅ Posterior sampling (300 samples by default)
- ✅ Mean and std predictions
- ✅ Confidence intervals in visualizations

## Output Files

### For Each Scenario
- `features_<scenario>.csv`: Extracted features (~1-5 GB depending on spatial resolution)
- `predictions_<scenario>.csv`: Risk predictions with uncertainty (~100-500 MB)

### Visualizations
- `temporal_evolution_all_scenarios.png`: Temporal trends
- `spatial_risk_maps_<scenario>.png`: Geographic risk distribution
- `regional_comparison_<scenario>.png`: Regional time series
- `scenario_comparison_heatmap.png`: Matrix view
- `summary_statistics.csv`: Quantitative comparison

## Important Notes

### 1. Model Interpretation
- **Output**: Relative probability scores (0-1)
- **NOT absolute fire counts**: Calibration was unreliable (see training script header)
- **Use for**: Ranking, comparison, trend analysis

### 2. Data Requirements
Climate projection files must follow structure:
```
/path/to/climate/data/
├── <scenario>/
│   ├── <year>/
│   │   ├── <variable>_<scenario>_<year><month>.nc
│   │   └── ...
```

NetCDF files must have:
- Variable name (auto-detected)
- Dimensions: `DATE`, `y`, `x`
- Daily temporal resolution

### 3. Performance Tuning
**Memory**: If running out of memory:
- Increase spatial grid resolution (e.g., 2000m instead of 1000m)
- Process fewer scenarios at once
- Reduce posterior samples

**Speed**: To speed up:
- Reduce posterior samples (default: 300 → try 100)
- Use coarser spatial grid
- Process only fire season months

### 4. Validation
After running, check:
- Prediction ranges (should be 0-1)
- Temporal trends (should show climate change signal)
- Spatial patterns (should match known high-risk areas)
- Uncertainty values (higher in extrapolation regions)

## Next Actions

### Immediate (Required)
1. ✅ **Fix landcover encoding** (see [LANDCOVER_ENCODING_ISSUE.md](LANDCOVER_ENCODING_ISSUE.md))
2. ✅ **Retrain model** with corrected encoding
3. ✅ **Test on single scenario** before full run

### Then (Full Pipeline)
4. ✅ **Run all scenarios** (`run_all_scenarios.py`)
5. ✅ **Generate visualizations** (`visualize_risk_evolution.py`)
6. ✅ **Interpret results** (compare RCP 4.5 vs 8.5, temporal trends)

### Optional (Advanced)
7. ⚠️ **Regional analysis**: Focus on specific geographic areas
8. ⚠️ **Sensitivity analysis**: Test different temporal aggregation windows
9. ⚠️ **Ensemble predictions**: If multiple climate models available

## Questions & Troubleshooting

### Q: Why are predictions so different from training data fire rates?
**A**: This is expected! The model was trained on case-control data (50% fire, 50% non-fire), so it learned **relative** probabilities, not absolute fire occurrence rates. See temporal validation plots in training output for scaling factors.

### Q: Can I convert relative probabilities to fire counts?
**A**: Not reliably. The training script tried calibration, but it degraded performance. Use relative probabilities for ranking and comparison only.

### Q: How do I focus on a specific region?
**A**: Edit `create_spatial_grid()` call in [run_all_scenarios.py](03_Climate_Projections/run_all_scenarios.py:94) to pass regional bounds from `REGIONS` dictionary.

### Q: The model is very slow. How can I speed it up?
**A**:
1. Reduce posterior samples: Line 450 in [run_all_scenarios.py](03_Climate_Projections/run_all_scenarios.py:450) (change `n_samples = 300` to `n_samples = 100`)
2. Coarser spatial grid: Line 94 (change `resolution=1000` to `resolution=2000`)
3. Fewer months: Edit `get_projection_dates()` call to only include high-risk months

### Q: My climate data has different file naming/structure
**A**: Edit `load_dynamic_data_for_period()` in [extract_projection_features.py](03_Climate_Projections/extract_projection_features.py:135) to match your file pattern.

## Documentation

- **Full documentation**: [03_Climate_Projections/README.md](03_Climate_Projections/README.md)
- **Landcover fix**: [LANDCOVER_ENCODING_ISSUE.md](LANDCOVER_ENCODING_ISSUE.md)
- **Model training**: [02_Model_Training/train_relative_probability_model.py](02_Model_Training/train_relative_probability_model.py) (header comments)

## References

Training data preparation:
- [create_raster_stacks.py](01_Data_Preparation/create_raster_stacks.py)

Model training:
- [train_relative_probability_model.py](02_Model_Training/train_relative_probability_model.py)
- [train_Dask_PyMC_timeseries.py](02_Model_Training/train_Dask_PyMC_timeseries.py)

Climate projections (NEW):
- [config_scenarios.py](03_Climate_Projections/config_scenarios.py)
- [extract_projection_features.py](03_Climate_Projections/extract_projection_features.py)
- [run_all_scenarios.py](03_Climate_Projections/run_all_scenarios.py)
- [visualize_risk_evolution.py](03_Climate_Projections/visualize_risk_evolution.py)
