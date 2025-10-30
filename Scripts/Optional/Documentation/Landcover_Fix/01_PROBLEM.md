# Landcover Categorical Encoding Issue

## Current Problem

The `landcoverfull` variable is currently treated as a **continuous numeric variable** throughout the pipeline, which is **mathematically incorrect**.

### Why This Is Wrong

Landcover classes represent **discrete categories** (e.g., forest, grassland, urban), not continuous measurements.

**Example Problem**:
- Class 3 (Grassland) is not "halfway between" Class 1 (Urban) and Class 5 (Broadleaf forest)
- A coefficient of +0.5 on landcover doesn't mean "higher class numbers increase fire risk"
- Interpolation between classes is meaningless

### Where This Occurs

1. **[create_raster_stacks.py](01_Data_Preparation/create_raster_stacks.py:95-104)**
   - Correctly uses **mode** (most common value) to aggregate
   - But then stores as numeric in NetCDF

2. **[train_relative_probability_model.py](02_Model_Training/train_relative_probability_model.py:95-104)**
   - Extracts mode value (correct)
   - Treats as continuous in model (incorrect)

3. **[train_Dask_PyMC_timeseries.py](02_Model_Training/train_Dask_PyMC_timeseries.py:108-122)**
   - Uses `compute_mode_robust()` (correct)
   - Treats as continuous in model (incorrect)

### Current Model Behavior

The Bayesian model learns a single coefficient for `landcoverfull`:
```python
beta_landcover * landcover_value
```

This assumes:
- Linear relationship between class number and fire risk
- Classes are equally spaced
- **Both assumptions are wrong!**

## Solution: Ordinal Encoding

### What Is Ordinal Encoding?

Map categorical classes to **meaningful ordered values** based on fire risk:

| Landcover Class | Name | Fire Risk (0-5) |
|----------------|------|-----------------|
| 9 | Water | 0 (no fire) |
| 10 | Snow/Ice | 0 (no fire) |
| 1 | Urban | 1 (low) |
| 8 | Bare rock | 1 (low) |
| 2 | Agriculture | 2 (moderate-low) |
| 3 | Grassland | 3 (moderate-high) |
| 5 | Broadleaf forest | 3 (moderate-high) |
| 7 | Mixed forest | 4 (high) |
| 4 | Shrubland | 4 (high) |
| 6 | Coniferous forest | 5 (very high) |

### Why This Works

1. **Preserves single-feature simplicity**: No explosion of features (unlike one-hot encoding)
2. **Encodes domain knowledge**: Higher values = more flammable
3. **Interpretable**: Coefficient shows "how much does fire risk increase per unit increase in vegetation flammability"
4. **Compatible**: Works with existing Bayesian model structure

## Implementation Steps

### Step 1: Update Feature Extraction (Training Data)

Edit [train_relative_probability_model.py](02_Model_Training/train_relative_probability_model.py) at line 95-104:

**Current code**:
```python
if var_name in CATEGORICAL_VARS:
    # Get mode (most common value) for categorical variables
    values = window_data.values.flatten()
    values = values[~np.isnan(values)]
    if len(values) > 0:
        from scipy import stats
        mode_result = stats.mode(values, keepdims=False)
        features[var_name] = float(mode_result.mode)  # ← PROBLEM: Treats as continuous
    else:
        features[var_name] = 0.0
```

**Updated code**:
```python
# Ordinal mapping based on fire risk
LANDCOVER_FIRE_RISK_ORDINAL = {
    9: 0, 10: 0,  # Water, Snow/Ice
    1: 1, 8: 1,   # Urban, Bare rock
    2: 2,         # Agriculture
    3: 3, 5: 3,   # Grassland, Broadleaf
    7: 4, 4: 4,   # Mixed forest, Shrubland
    6: 5          # Coniferous (highest risk)
}

if var_name in CATEGORICAL_VARS:
    # Get mode (most common value)
    values = window_data.values.flatten()
    values = values[~np.isnan(values)]
    if len(values) > 0:
        from scipy import stats
        mode_result = stats.mode(values, keepdims=False)
        landcover_class = int(mode_result.mode)

        # Map to ordinal fire risk value
        features['landcover_fire_risk'] = float(
            LANDCOVER_FIRE_RISK_ORDINAL.get(landcover_class, 2)  # Default: moderate
        )
    else:
        features['landcover_fire_risk'] = 0.0
```

**Changes**:
- Replace `features[var_name]` with `features['landcover_fire_risk']`
- Map landcover class to fire risk ordinal value
- This creates a new feature: `landcover_fire_risk` (0-5 scale)

### Step 2: Update Feature List

In the same file, update the `STATIC_VARS` list:

**Current**:
```python
STATIC_VARS = [
    'tri', 'northness', 'slope', 'aspect', 'nasadem',
    'treecoverdensity', 'landcoverfull', 'distroads',  # ← landcoverfull
    'eastness', 'flammability', 'walking_time_to_bldg',
    'walking_time_to_elec_infra'
]
```

**Updated**:
```python
STATIC_VARS = [
    'tri', 'northness', 'slope', 'aspect', 'nasadem',
    'treecoverdensity', 'landcover_fire_risk', 'distroads',  # ← changed
    'eastness', 'flammability', 'walking_time_to_bldg',
    'walking_time_to_elec_infra'
]

CATEGORICAL_VARS = []  # No longer needed, we're encoding directly
```

### Step 3: Update Dask Script

Make the same changes to [train_Dask_PyMC_timeseries.py](02_Model_Training/train_Dask_PyMC_timeseries.py):

**At line 108-122**, replace the landcover handling:

```python
# Current: Uses mode but treats as continuous
landcover_data_full = ds[main_data_var].sel(channel='landcoverfull').isel(time=0)
landcover_mode = landcover_data_full.groupby('id_obs').apply(
    lambda x: xr.DataArray(compute_mode_robust(x.data), ...)
)
```

**Updated**:
```python
# Get landcover mode values
landcover_data_full = ds[main_data_var].sel(channel='landcoverfull').isel(time=0)
landcover_mode = landcover_data_full.groupby('id_obs').apply(
    lambda x: xr.DataArray(compute_mode_robust(x.data), ...)
)

# Map to ordinal fire risk
LANDCOVER_FIRE_RISK_ORDINAL = {
    9: 0, 10: 0, 1: 1, 8: 1, 2: 2,
    3: 3, 5: 3, 7: 4, 4: 4, 6: 5
}

def map_landcover_to_fire_risk(landcover_value):
    return LANDCOVER_FIRE_RISK_ORDINAL.get(int(landcover_value), 2)

# Vectorized mapping
landcover_fire_risk = xr.apply_ufunc(
    map_landcover_to_fire_risk,
    landcover_mode,
    vectorize=True
)
landcover_fire_risk = landcover_fire_risk.expand_dims(channel=1)
landcover_fire_risk = landcover_fire_risk.assign_coords(channel=['landcover_fire_risk'])
```

### Step 4: Update Climate Projection Scripts

The climate projection scripts inherit the same issue. Update [extract_projection_features.py](03_Climate_Projections/extract_projection_features.py) at line 131-138:

**Current**:
```python
# Categorical handling for landcover
if var_name == 'landcoverfull':
    values = window_data.values.flatten()
    values = values[~np.isnan(values)]
    if len(values) > 0:
        mode_result = stats.mode(values, keepdims=False)
        features[var_name] = int(mode_result.mode)  # ← Problem
    else:
        features[var_name] = 0
```

**Updated**:
```python
# Categorical handling for landcover → ordinal fire risk
if var_name == 'landcoverfull':
    values = window_data.values.flatten()
    values = values[~np.isnan(values)]
    if len(values) > 0:
        mode_result = stats.mode(values, keepdims=False)
        landcover_class = int(mode_result.mode)

        # Map to fire risk ordinal
        features['landcover_fire_risk'] = LANDCOVER_FIRE_RISK_ORDINAL.get(
            landcover_class, 2
        )
    else:
        features['landcover_fire_risk'] = 0
```

**Also update** the `STATIC_VARS` list in the same file (line 42-52).

### Step 5: Retrain Model

After making all changes:

```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts/02_Model_Training

# Retrain with corrected encoding
python train_relative_probability_model.py
```

This will:
1. Re-extract features with ordinal encoding
2. Train a new model with `landcover_fire_risk` instead of `landcoverfull`
3. Save updated model artifacts

### Step 6: Re-run Climate Projections

```bash
cd ../03_Climate_Projections

# Extract features and generate predictions
python run_all_scenarios.py

# Create visualizations
python visualize_risk_evolution.py
```

## Alternative: One-Hot Encoding

If you prefer one-hot encoding (less recommended):

### Pros
- No assumptions about ordering
- Model learns separate coefficients for each class
- Captures non-linear relationships

### Cons
- Creates 8-10 additional features (dimensionality explosion)
- Requires more training data
- Coefficients less interpretable
- Reference class ambiguity

### Implementation

Use [fix_landcover_encoding.py](02_Model_Training/fix_landcover_encoding.py):

```python
from fix_landcover_encoding import encode_landcover_onehot

# In feature extraction function:
if var_name == 'landcoverfull':
    # Extract mode as before
    landcover_class = int(mode_result.mode)

    # Store temporarily
    features_temp['landcover_class'] = landcover_class

# After loop, one-hot encode all landcover classes
landcover_series = pd.Series([f['landcover_class'] for f in all_features])
landcover_onehot = encode_landcover_onehot(landcover_series)

# Merge with other features
X = pd.concat([X, landcover_onehot], axis=1)
```

## Verification

After retraining, check that the model learned meaningful coefficients:

```python
# Load trace
import arviz as az
trace = az.from_netcdf("OUTPUT/02_Model_RelativeProbability/trace_relative.nc")

# Find landcover coefficient
# If using temporal groups, it might be in a specific group (e.g., 'static_veg')
landcover_beta = trace.posterior['beta_static_veg'].mean(dim=['chain', 'draw'])

print(f"Landcover fire risk coefficient: {landcover_beta[<index>]:.4f}")
```

**Expected**:
- **Positive coefficient**: Higher fire risk classes → higher wildfire probability
- **Magnitude**: Should be comparable to other static variables

## Summary

### Current Status
❌ `landcoverfull` treated as continuous (incorrect)

### After Fix
✅ `landcover_fire_risk` encoded as ordinal (0-5 scale based on flammability)

### Impact
- **Model accuracy**: Improved (correct mathematical representation)
- **Interpretability**: Better (coefficients show effect of vegetation flammability)
- **Prediction reliability**: Higher confidence in high-flammability areas

### Action Required
1. Update 3 training/extraction scripts (lines identified above)
2. Retrain model
3. Re-run climate projections
4. Compare results with previous version

## Questions?

Refer to:
- [fix_landcover_encoding.py](02_Model_Training/fix_landcover_encoding.py): Example implementation
- [README.md](03_Climate_Projections/README.md): Full documentation
- Research literature on categorical encoding in wildfire models
