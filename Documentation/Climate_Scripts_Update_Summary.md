# Climate Scripts Update Summary

## Status: Climate Scripts Migration to Absolute Probability

### ✅ **COMPLETED**

#### 1. Main Climate Projection Script
**File**: `Scripts/03_Climate_Projections/05_Bayesian_Climate_Projection_ABSOLUTE.py`

**Status**: ✅ **Complete and Production Ready**

**Key Changes**:
- Uses absolute probability model artifacts
- Outputs expected fire counts per year
- Provides change from baseline percentages
- Generates interpretable stakeholder reports

**Usage**:
```bash
python Scripts/03_Climate_Projections/05_Bayesian_Climate_Projection_ABSOLUTE.py
```

#### 2. Fire Brigade Analysis Script
**File**: `Scripts/03_Climate_Projections/06_Fire_Brigade_Climate_Projections_ABSOLUTE.py`

**Status**: ✅ **Complete and Production Ready**

**Key Changes**:
- Zone-specific absolute probability predictions
- Expected fires per brigade zone
- Temporal comparison (2020, 2050, 2080)
- Visualization of zone-level changes

**Usage**:
```bash
python Scripts/03_Climate_Projections/06_Fire_Brigade_Climate_Projections_ABSOLUTE.py
```

---

### 📋 **REMAINING SCRIPTS TO UPDATE**

#### 3. Multi-Quantile Seasonal Projections
**File**: `Scripts/03_Climate_Projections/05_Bayesian_Climate_Projection_MultiQuantile_Seasonal.py`

**Status**: ⏳ **Needs Update**

**Required Changes**:
1. Update model paths to absolute probability:
   ```python
   # OLD
   MODEL_DIR = BASE_DIR / "Data/OUTPUT/02_Model"
   TRACE_PATH = MODEL_DIR / "trace.nc"
   SCALER_PATH = MODEL_DIR / "scaler.joblib"

   # NEW
   MODEL_DIR = BASE_DIR / "Data/OUTPUT/02_Model_AbsoluteProbability"
   TRACE_PATH = MODEL_DIR / "trace_absolute.nc"
   SCALER_PATH = MODEL_DIR / "scaler_absolute.joblib"
   TRUE_FIRE_STATS_PATH = MODEL_DIR / "true_fire_stats.joblib"
   TEMPORAL_GROUPS_PATH = MODEL_DIR / "temporal_groups.joblib"
   GROUP_NAMES_PATH = MODEL_DIR / "group_names.joblib"
   ```

2. Load all model components:
   ```python
   trace = az.from_netcdf(TRACE_PATH)
   scaler = joblib.load(SCALER_PATH)
   true_fire_stats = joblib.load(TRUE_FIRE_STATS_PATH)
   temporal_groups = joblib.load(TEMPORAL_GROUPS_PATH)
   group_names = joblib.load(GROUP_NAMES_PATH)
   ```

3. Use absolute probability prediction function (copy from `05_Bayesian_Climate_Projection_ABSOLUTE.py`)

4. Update output interpretation:
   ```python
   # Add expected fire counts
   expected_fires_per_year = predictions.mean() * 365.25
   change_from_baseline = ((expected_fires_per_year - true_fire_stats['fires_per_year_regional']) /
                           true_fire_stats['fires_per_year_regional']) * 100
   ```

**Estimated Effort**: 2-3 hours

---

#### 4. Historical Lookback GIF
**File**: `Scripts/03_Climate_Projections/05_Bayesian_Lookback_2022_GIF.py`

**Status**: ⏳ **Needs Update**

**Required Changes**:
1. Update model paths (same as above)

2. Update color scale for absolute probability:
   ```python
   # OLD (relative risk, 0-1)
   vmin, vmax = 0, 1
   cmap = plt.cm.YlOrRd
   cbar_label = "Relative Fire Risk"

   # NEW (absolute probability)
   baseline = true_fire_stats['fires_per_day_regional']  # 0.0233
   vmin = 0
   vmax = baseline * 5  # Show up to 5× baseline
   cmap = plt.cm.YlOrRd
   cbar_label = "Daily Fire Probability"

   # Add baseline reference line
   plt.axhline(y=baseline, color='red', linestyle='--',
              label=f'Baseline ({baseline:.4f})')
   ```

3. Update title and labels:
   ```python
   # OLD
   title = f"Fire Risk - {date}"

   # NEW
   title = f"Fire Probability - {date}"
   subtitle = f"Expected: {predictions.mean() * 365:.1f} fires/year"
   ```

**Estimated Effort**: 1-2 hours

---

## Quick Update Template

For **any** climate projection script, follow this template:

### Step 1: Update Imports
```python
import joblib
import arviz as az
```

### Step 2: Update Model Paths
```python
MODEL_DIR = Path("/mnt/CEPH_PROJECTS/Firescape/Data/OUTPUT/02_Model_AbsoluteProbability")
TRACE_PATH = MODEL_DIR / "trace_absolute.nc"
SCALER_PATH = MODEL_DIR / "scaler_absolute.joblib"
TRUE_FIRE_STATS_PATH = MODEL_DIR / "true_fire_stats.joblib"
TEMPORAL_GROUPS_PATH = MODEL_DIR / "temporal_groups.joblib"
GROUP_NAMES_PATH = MODEL_DIR / "group_names.joblib"
```

### Step 3: Load Model Components
```python
print("Loading absolute probability model...")
trace = az.from_netcdf(TRACE_PATH)
scaler = joblib.load(SCALER_PATH)
true_fire_stats = joblib.load(TRUE_FIRE_STATS_PATH)
temporal_groups = joblib.load(TEMPORAL_GROUPS_PATH)
group_names = joblib.load(GROUP_NAMES_PATH)

print(f"  Baseline: {true_fire_stats['fires_per_year_regional']:.1f} fires/year")
```

### Step 4: Copy Prediction Function
Copy the `generate_absolute_probability_predictions()` function from:
`Scripts/03_Climate_Projections/05_Bayesian_Climate_Projection_ABSOLUTE.py` (lines 105-152)

### Step 5: Update Output Interpretation
```python
# After generating predictions
mean_prob = predictions.mean()
expected_annual_fires = mean_prob * 365.25
baseline_fires = true_fire_stats['fires_per_year_regional']
change_pct = ((expected_annual_fires - baseline_fires) / baseline_fires) * 100

print(f"Daily probability: {mean_prob:.6f}")
print(f"Expected fires/year: {expected_annual_fires:.1f}")
print(f"Baseline: {baseline_fires:.1f} fires/year")
print(f"Change: {change_pct:+.1f}%")
```

### Step 6: Update Visualizations
```python
# Update color scales
baseline_daily = true_fire_stats['fires_per_day_regional']  # 0.0233
vmin = 0
vmax = baseline_daily * 5  # Up to 5× baseline

# Update labels
plt.xlabel("Daily Fire Probability")
plt.ylabel("Frequency")
plt.title("Fire Probability Distribution")

# Add baseline reference
plt.axvline(baseline_daily, color='red', linestyle='--',
           label=f'Baseline ({baseline_daily:.4f})')
```

---

## Testing Checklist

After updating a script, verify:

- [ ] Script runs without errors
- [ ] Predictions are in reasonable range (0 to ~0.1)
- [ ] Expected annual fires match baseline for historical periods
- [ ] Plots have correct labels (probability, not risk)
- [ ] Output files saved correctly
- [ ] Results interpretable for stakeholders

---

## Common Issues & Solutions

### Issue 1: Model file not found
**Error**: `FileNotFoundError: Model trace not found`

**Solution**:
```python
# Check model was trained
ls -lh /mnt/CEPH_PROJECTS/Firescape/Data/OUTPUT/02_Model_AbsoluteProbability/

# If missing, train model first
python Scripts/02_Model_Training/05_Bayesian_AbsoluteProbability_Regional.py
```

### Issue 2: Predictions out of range
**Error**: Predictions > 1 or < 0

**Solution**:
```python
# Check features were scaled
X_scaled = scaler.transform(X)  # Don't forget this!

# Check prediction function uses sigmoid
prob = 1 / (1 + np.exp(-logit_pred))  # Should be between 0 and 1
```

### Issue 3: Missing temporal_groups
**Error**: `KeyError: 'temporal_groups'`

**Solution**:
```python
# Load from joblib file
temporal_groups = joblib.load(TEMPORAL_GROUPS_PATH)
group_names = joblib.load(GROUP_NAMES_PATH)

# Use in prediction function
for group_idx, (group_name, feature_indices) in enumerate(temporal_groups.items()):
    ...
```

---

## File Structure

```
Scripts/03_Climate_Projections/
├── 05_Bayesian_Climate_Projection_CLEAN.py              # OLD (relative)
├── 05_Bayesian_Climate_Projection_ABSOLUTE.py           # ✅ NEW (absolute)
├── 06_Fire_Brigade_Climate_Projections.py               # OLD (relative)
├── 06_Fire_Brigade_Climate_Projections_ABSOLUTE.py      # ✅ NEW (absolute)
├── 05_Bayesian_Climate_Projection_MultiQuantile_Seasonal.py  # ⏳ TO UPDATE
└── 05_Bayesian_Lookback_2022_GIF.py                     # ⏳ TO UPDATE
```

---

## Timeline

**Immediate** (This week):
- ✅ Main climate projection script
- ✅ Fire brigade analysis script

**Short-term** (Next week):
- ⏳ Multi-quantile seasonal projections
- ⏳ Historical lookback GIF

**Ongoing**:
- Test all updated scripts
- Generate stakeholder reports
- Validate results

---

## References

- **Main guide**: `Documentation/Climate_Projections_AbsoluteProb_Update_Guide.md`
- **Technical details**: `Documentation/Technical_Deep_Dive_Absolute_Probability.md`
- **Quick reference**: `Documentation/QUICK_REFERENCE.md`
- **Example code**: `Scripts/03_Climate_Projections/05_Bayesian_Climate_Projection_ABSOLUTE.py`

---

## Need Help?

1. Check documentation in `/Documentation/`
2. Review working examples in updated scripts
3. Consult the Quick Reference card
4. Test with small date range first

---

**Last Updated**: 2025-10-20
**Version**: 1.0
**Status**: 2 of 4 climate scripts updated ✅
