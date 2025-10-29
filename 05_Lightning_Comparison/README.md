# Lightning vs No-Lightning Model Comparison

## Overview

This workflow tests whether including **lightning data** improves wildfire prediction performance compared to using only temperature and precipitation.

### Models Compared

1. **Baseline Model (T+P)**
   - Temperature + Precipitation only
   - Location: `Data/OUTPUT/02_Model_RelativeProbability/`

2. **Lightning Model (T+P+L)**
   - Temperature + Precipitation + Lightning flash density
   - Location: `Data/OUTPUT/02_Model_RelativeProbability_Lightning/`

## Research Question

**Does including lightning flash density data significantly improve:**
- Temporal correlation (monthly/seasonal fit)?
- Discrimination performance (ROC-AUC, PR-AUC)?
- Feature importance and interpretability?

## Data Requirements

### Lightning Data
- **Source**: South Tyrolean Civil Protection Agency
- **Format**: Daily GeoTIFF (`Flash_Dens_YYYYMMDD.tif`)
- **Location**: `/mnt/CEPH_PROJECTS/Firescape/Data/05_Meteorological_Data/Lightning/`
- **Temporal coverage**: 2012 onwards
- **Spatial resolution**: ~50m (matches other rasters)
- **Units**: Lightning flash density per grid cell

### Training Data Subset
Since lightning data starts in 2012, the training dataset will be filtered to:
- **Years**: 2012-2025 (instead of 1999-2025 for baseline)
- **Fire events**: Only those from 2012 onwards
- **Implications**: Smaller training set, but consistent data availability

## Workflow

### Step 1: Create Lightning-Enabled Training Data

```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts/05_Lightning_Comparison/01_Data_Preparation
python create_raster_stacks_with_lightning.py
```

**What it does:**
- Reads same point dataset as baseline (from `01_Training_Data/spacetime_dataset.parquet`)
- Extracts all variables including lightning flash density (L)
- Creates 4D tensors: `(time=60, y=32, x=32, channels=15)`
  - 12 static variables
  - 3 dynamic variables: T, P, **L**
- Saves to: `Data/OUTPUT/01_Training_Data_Lightning/spacetime_stacks_lightning.nc`

**Key difference from baseline:**
- Adds `DYNAMIC_VARS = ['T', 'P', 'L']` instead of `['T', 'P']`
- Lightning files: `Flash_Dens_YYYYMMDD.tif` (daily GeoTIFFs)
- Missing lightning values filled with 0 (no lightning = 0 flashes)

### Step 2: Train Lightning-Enabled Model

```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts/05_Lightning_Comparison/02_Model_Training
python train_relative_probability_model_with_lightning.py
```

**What it does:**
- Same Bayesian attention model architecture as baseline
- Processes lightning features through temporal windows (1, 3, 5, 10, 15, 30, 60 days)
- Creates lightning feature groups (e.g., `light_1d`, `light_short`, etc.)
- Trains with PyMC MCMC (4 chains, 2000 samples)
- Saves model to: `Data/OUTPUT/02_Model_RelativeProbability_Lightning/`

**Expected runtime:** ~60-90 minutes (similar to baseline)

**Outputs:**
- `trace_relative.nc` - Posterior samples
- `scaler_relative.joblib` - Feature scaling
- `temporal_groups.joblib` - Feature groupings (includes lightning groups)
- `model_results.joblib` - Validation metrics
- `prior_distributions.png` - Prior visualization

### Step 3: Compare Models

```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts/05_Lightning_Comparison/03_Model_Comparison
python compare_models.py
```

**What it does:**
- Loads both baseline and lightning model results
- Compares metrics:
  - **Temporal fit**: Monthly R², Seasonal R²
  - **Discrimination**: ROC-AUC, PR-AUC
  - **Feature importance**: Attention weights
- Generates comparison plots
- Provides recommendation on whether to use lightning

**Outputs:**
- `Results/model_comparison.png` - Visual comparison
- `Results/comparison_summary.csv` - Metric comparison table
- Console recommendation: Use lightning or not?

## Expected Outcomes

### Scenario A: Lightning Improves Performance
If lightning flash density is a strong fire ignition source:
- ✅ Monthly R² increases (e.g., 0.577 → 0.65+)
- ✅ ROC-AUC increases slightly
- ✅ Lightning feature groups have high attention weights
- **Recommendation**: Use lightning model operationally

### Scenario B: Lightning Marginal Benefit
If lightning helps but only slightly:
- ~ Monthly R² increases <5% (e.g., 0.577 → 0.60)
- ~ Small improvement in metrics
- ~ Lightning groups have moderate attention
- **Recommendation**: Evaluate cost vs. benefit (lightning data availability, processing time)

### Scenario C: Lightning No Benefit
If lightning doesn't improve predictions (possible in South Tyrol if most fires are human-caused):
- ✗ No improvement or performance decline
- ✗ Lightning groups have low attention weights
- **Recommendation**: Continue with baseline (T+P) model

## Feature Engineering

### Lightning Temporal Windows

Lightning is processed the same as T and P:

```python
# Cumulative lightning features (same as T, P)
for day_window in [1, 3, 5, 10, 15, 30, 60]:
    L_cumulative_mean_{day_window}d  # Average flash density
    L_cumulative_max_{day_window}d   # Maximum flash density
```

### Lightning Feature Groups

The attention mechanism groups lightning features:
- `light_1d`: 1-day lightning
- `light_short`: 3-5 day cumulative
- `light_medium`: 10-15 day cumulative
- `light_30d`: 30-day cumulative
- `light_60d`: 60-day cumulative

## Technical Notes

### Data Handling

**Lightning = 0 when missing:**
```python
if prefix == 'L':
    chip_reprojected = chip_reprojected.fillna(0)  # No lightning = 0 flashes
```

This is appropriate because:
- Lightning sensors detect all flashes in coverage area
- Missing data = no lightning activity
- Different from T/P where missing = unknown conditions

### Model Architecture

Both models use identical architecture:
- Bayesian logistic regression
- Linear attention mechanism
- PyMC MCMC inference
- Weakly informative priors

**Only difference**: Lightning model has more feature groups (adds lightning groups)

### Temporal Coverage

**Baseline (T+P)**: 1999-2025
**Lightning (T+P+L)**: 2012-2025 (limited by lightning data availability)

To ensure fair comparison, you may want to retrain baseline model on 2012-2025 subset.

## Interpretation

### If Lightning is Important

Lightning-ignited fires tend to:
- Occur in remote areas (low human activity)
- Follow convective storms (short temporal lag)
- Cluster spatially near strike locations
- Peak in summer thunderstorm season

### If Lightning is Not Important

Most fires may be:
- Human-caused (agriculture, arson, accidents)
- Not well-correlated with lightning density
- Better predicted by human infrastructure variables
- Already captured by temperature/dryness patterns

## Files Created

```
05_Lightning_Comparison/
├── 01_Data_Preparation/
│   └── create_raster_stacks_with_lightning.py
├── 02_Model_Training/
│   └── train_relative_probability_model_with_lightning.py
├── 03_Model_Comparison/
│   ├── compare_models.py
│   └── Results/
│       ├── model_comparison.png
│       └── comparison_summary.csv
└── README.md (this file)
```

## Next Steps

1. **Run data preparation** (Step 1)
2. **Train lightning model** (Step 2)
3. **Compare models** (Step 3)
4. **Interpret results**: Check if lightning groups are important
5. **Decision**: Use lightning model or stick with baseline

## Cost-Benefit Considerations

### Benefits of Including Lightning
- ✅ Better temporal correlation (if significant)
- ✅ Physical interpretability (lightning = ignition source)
- ✅ Captures convective storm patterns

### Costs of Including Lightning
- ❌ Data availability: Lightning starts 2012 (vs 1999 for T/P)
- ❌ Processing time: Additional variable to extract
- ❌ Operational complexity: Need lightning data for predictions
- ❌ Smaller training dataset (2012-2025 vs 1999-2025)

## References

- Lightning data: South Tyrolean Civil Protection Agency
- Azure portal link: In `/mnt/CEPH_PROJECTS/Firescape/Scripts/Archive/20251028_PreRefactor/link_lightning_data_province.txt`
- Original lightning analysis: Archive scripts (e.g., `04_LSTM_attention_f2_lightnolight.py`)

---

**Contact**: For questions about this workflow, see `../WORKFLOW_SUMMARY.md`
