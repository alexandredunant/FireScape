# Climate Projections Update Guide: Absolute Probability

## Overview

This guide explains how to update climate projection scripts to use the new **absolute probability model** instead of the relative probability model.

## Key Changes Required

### 1. Model File Paths

**OLD (Relative Probability)**:
```python
MODEL_DIR = BASE_PROJECT_DIR / "Data/OUTPUT/02_Model"
TRACE_PATH = MODEL_DIR / "trace.nc"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
BASELINE_STATS_PATH = MODEL_DIR / "baseline_stats.joblib"
```

**NEW (Absolute Probability)**:
```python
MODEL_DIR = BASE_PROJECT_DIR / "Data/OUTPUT/02_Model_AbsoluteProbability"
TRACE_PATH = MODEL_DIR / "trace_absolute.nc"
SCALER_PATH = MODEL_DIR / "scaler_absolute.joblib"
TRUE_FIRE_STATS_PATH = MODEL_DIR / "true_fire_stats.joblib"
TEMPORAL_GROUPS_PATH = MODEL_DIR / "temporal_groups.joblib"
GROUP_NAMES_PATH = MODEL_DIR / "group_names.joblib"
```

### 2. Load Additional Model Components

```python
# Load all model components
trace = az.from_netcdf(TRACE_PATH)
scaler = joblib.load(SCALER_PATH)
true_fire_stats = joblib.load(TRUE_FIRE_STATS_PATH)
temporal_groups = joblib.load(TEMPORAL_GROUPS_PATH)
group_names = joblib.load(GROUP_NAMES_PATH)

print(f"Model baseline: {true_fire_stats['fires_per_year_regional']:.1f} fires/year")
```

### 3. Update Prediction Function

Import the prediction function from the absolute probability model:

```python
import sys
sys.path.append('/mnt/CEPH_PROJECTS/Firescape/Scripts/02_Model_Training')
from 05_Bayesian_AbsoluteProbability_Regional import generate_predictions
```

Or use it directly:

```python
def generate_absolute_predictions(trace, temporal_groups, group_names, X_scaled, n_samples=300):
    """
    Generate absolute probability predictions.
    Returns mean and std of predicted probabilities.
    """
    alpha_samples = trace.posterior['alpha'].values.reshape(-1)[:n_samples]
    attention_samples = trace.posterior['attention_weights'].values.reshape(-1, len(group_names))[:n_samples]

    group_betas = {}
    for group_name in temporal_groups.keys():
        beta_key = f'beta_{group_name}'
        if beta_key in trace.posterior:
            beta_samples = trace.posterior[beta_key].values
            beta_flat = beta_samples.reshape(-1, beta_samples.shape[-1])[:n_samples]
            group_betas[group_name] = beta_flat

    n_test = X_scaled.shape[0]
    prob_predictions = []

    for sample_idx in range(n_samples):
        logit_pred = np.full(n_test, alpha_samples[sample_idx])

        for group_idx, (group_name, feature_indices) in enumerate(temporal_groups.items()):
            if group_name in group_betas:
                beta_sample = group_betas[group_name][sample_idx]
                group_features_data = X_scaled[:, feature_indices]
                group_contrib = np.dot(group_features_data, beta_sample)
                attention_weight = attention_samples[sample_idx, group_idx]
                weighted_contrib = attention_weight * group_contrib
                logit_pred += weighted_contrib

        prob_pred = 1 / (1 + np.exp(-logit_pred))
        prob_predictions.append(prob_pred)

    prob_predictions = np.array(prob_predictions)
    return prob_predictions.mean(axis=0), prob_predictions.std(axis=0)
```

### 4. Update Output Interpretation

**CRITICAL**: Predictions now represent actual fire probability!

**OLD Interpretation (Relative)**:
```python
# Relative risk scores (0-1)
print(f"Mean risk score: {predictions.mean():.4f}")
# Comparison only - no absolute meaning
```

**NEW Interpretation (Absolute)**:
```python
# Actual daily fire probability
daily_prob = predictions.mean()
print(f"Daily fire probability: {daily_prob:.6f}")
print(f"Expected fires per day: {daily_prob:.4f}")
print(f"Expected fires per year: {daily_prob * 365:.1f}")
print(f"Expected fires this projection: {predictions.sum():.1f}")
```

### 5. Aggregating Predictions

For climate scenarios, **sum daily probabilities** to get expected fire counts:

```python
# For a specific time period
start_date = pd.to_datetime('2050-01-01')
end_date = pd.to_datetime('2050-12-31')
period_mask = (dates >= start_date) & (dates <= end_date)

# Expected fires in this period
expected_fires = predictions[period_mask].sum()
expected_fires_per_year = expected_fires / ((end_date - start_date).days / 365.25)

print(f"Expected fires in 2050: {expected_fires_per_year:.1f} fires/year")
```

### 6. Comparison Across Scenarios

```python
# Compare different RCP scenarios
scenarios = {
    'Historical (baseline)': 8.5,  # From true_fire_stats
    'RCP 2.6 (2050)': predictions_rcp26.sum() / n_years,
    'RCP 4.5 (2050)': predictions_rcp45.sum() / n_years,
    'RCP 8.5 (2050)': predictions_rcp85.sum() / n_years
}

for scenario, fires_per_year in scenarios.items():
    change_pct = ((fires_per_year - scenarios['Historical (baseline)']) /
                  scenarios['Historical (baseline)']) * 100
    print(f"{scenario}: {fires_per_year:.1f} fires/year ({change_pct:+.1f}%)")
```

### 7. Uncertainty Quantification

Use posterior samples to get credible intervals:

```python
# Generate predictions for multiple posterior samples
n_posterior_samples = 100
predictions_ensemble = []

for i in range(n_posterior_samples):
    # Use subset of posterior
    trace_subset = trace.posterior.isel(draw=slice(i*10, (i+1)*10))
    mean_pred, _ = generate_absolute_predictions(
        trace_subset, temporal_groups, group_names, X_scaled, n_samples=10
    )
    predictions_ensemble.append(mean_pred.sum() / n_years)  # Fires per year

predictions_ensemble = np.array(predictions_ensemble)
mean_fires = predictions_ensemble.mean()
lower_ci = np.percentile(predictions_ensemble, 2.5)
upper_ci = np.percentile(predictions_ensemble, 97.5)

print(f"Expected fires: {mean_fires:.1f} [95% CI: {lower_ci:.1f}-{upper_ci:.1f}] per year")
```

### 8. Spatial Aggregation

For regional summaries:

```python
# Add region ID to predictions
gdf['fire_probability'] = predictions
gdf['expected_fires_per_year'] = predictions * 365

# Aggregate by administrative region
regional_summary = gdf.groupby('region_id').agg({
    'fire_probability': 'mean',
    'expected_fires_per_year': 'sum'
})

print("\nRegional Fire Expectations:")
print(regional_summary.round(2))
```

### 9. Temporal Aggregation (Monthly/Seasonal)

```python
# Monthly expected fires
monthly_predictions = pd.DataFrame({
    'date': dates,
    'probability': predictions
})
monthly_predictions['month'] = monthly_predictions['date'].dt.month
monthly_summary = monthly_predictions.groupby('month').agg({
    'probability': 'sum'  # Expected fires per month
})

# Seasonal
seasonal_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'}
monthly_predictions['season'] = monthly_predictions['month'].map(seasonal_map)
seasonal_summary = monthly_predictions.groupby('season').agg({
    'probability': 'sum'
})

print("\nExpected fires by season:")
print(seasonal_summary.round(2))
```

### 10. Visualization Updates

**Color scale interpretation**:

```python
# OLD: Relative risk (0-1 scale, arbitrary)
vmin, vmax = 0, 1
label = "Relative Fire Risk"

# NEW: Absolute probability
vmin = 0
vmax = true_fire_stats['fires_per_day_regional'] * 5  # Up to 5x baseline
label = "Daily Fire Probability"

# Add baseline reference line
ax.axhline(y=true_fire_stats['fires_per_day_regional'],
           color='red', linestyle='--',
           label=f'Baseline ({true_fire_stats["fires_per_year_regional"]:.1f} fires/year)')
```

## Example: Complete Climate Projection Script Update

Here's a template for the main changes:

```python
#!/usr/bin/env python
"""
Climate Projection with Absolute Probability Model
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import joblib
import arviz as az
from pathlib import Path

# === CONFIGURATION ===
BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape")
MODEL_DIR = BASE_DIR / "Data/OUTPUT/02_Model_AbsoluteProbability"
OUTPUT_DIR = BASE_DIR / "Data/OUTPUT/04_Climate_Projections_Absolute"

# Load absolute probability model
print("Loading absolute probability model...")
trace = az.from_netcdf(MODEL_DIR / "trace_absolute.nc")
scaler = joblib.load(MODEL_DIR / "scaler_absolute.joblib")
true_fire_stats = joblib.load(MODEL_DIR / "true_fire_stats.joblib")
temporal_groups = joblib.load(MODEL_DIR / "temporal_groups.joblib")
group_names = joblib.load(MODEL_DIR / "group_names.joblib")

print(f"✓ Model loaded")
print(f"  Baseline: {true_fire_stats['fires_per_year_regional']:.1f} fires/year")
print(f"  Regional daily probability: {true_fire_stats['fires_per_day_regional']:.6f}")

# === GENERATE PREDICTIONS ===
# ... extract features X_scaled ...

# Generate absolute probabilities
mean_prob, std_prob = generate_absolute_predictions(
    trace, temporal_groups, group_names, X_scaled
)

# === INTERPRET RESULTS ===
total_days = len(dates)
n_years = total_days / 365.25

print(f"\n🔥 FIRE EXPECTATIONS:")
print(f"  Projection period: {n_years:.1f} years")
print(f"  Total expected fires: {mean_prob.sum():.1f}")
print(f"  Expected fires per year: {mean_prob.sum() / n_years:.1f}")
print(f"  Baseline (historical): {true_fire_stats['fires_per_year_regional']:.1f}")

change_pct = ((mean_prob.sum() / n_years - true_fire_stats['fires_per_year_regional']) /
              true_fire_stats['fires_per_year_regional']) * 100
print(f"  Change from baseline: {change_pct:+.1f}%")

# === SAVE RESULTS ===
results = {
    'predictions': mean_prob,
    'uncertainty': std_prob,
    'expected_fires_per_year': mean_prob.sum() / n_years,
    'baseline_fires_per_year': true_fire_stats['fires_per_year_regional'],
    'change_percent': change_pct
}
joblib.dump(results, OUTPUT_DIR / "projection_results.joblib")
```

## Files to Update

1. **`05_Bayesian_Climate_Projection_CLEAN.py`**
   - Main climate projection script
   - Update model paths, prediction function, and interpretation

2. **`06_Fire_Brigade_Climate_Projections.py`**
   - Fire brigade operational projections
   - Add expected fire count summaries

3. **`05_Bayesian_Climate_Projection_MultiQuantile_Seasonal.py`**
   - Multi-quantile seasonal projections
   - Update for absolute probability across quantiles

4. **`05_Bayesian_Lookback_2022_GIF.py`**
   - Historical lookback animations
   - Update color scales and labels

## Validation Checklist

After updating scripts, verify:

- [ ] Model loads successfully with all components
- [ ] Predictions are in reasonable range (0 to ~0.1)
- [ ] Expected annual fires match baseline for historical period
- [ ] Seasonal patterns are preserved (summer peaks)
- [ ] Spatial patterns are reasonable
- [ ] Uncertainty estimates are included
- [ ] Outputs are labeled as "probability" not "risk score"
- [ ] Documentation mentions absolute probability

## Common Pitfalls

1. **Forgetting to update model paths** → Script will load old relative model
2. **Not updating interpretation** → Users think scores are relative
3. **Incorrect aggregation** → Sum probabilities for counts, don't average
4. **Missing uncertainty** → Always include credible intervals
5. **Wrong color scales** → Update visualizations for absolute values

## Questions?

- See: `Documentation/Absolute_Probability_Model_README.md`
- Model code: `Scripts/02_Model_Training/05_Bayesian_AbsoluteProbability_Regional.py`
- Example usage: See validation sections in model training script

---

**Last Updated**: 2025-10-20
**Version**: 1.0
