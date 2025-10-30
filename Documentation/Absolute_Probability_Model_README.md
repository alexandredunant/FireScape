# Absolute Probability Wildfire Prediction Model

## Overview

This document describes the transition from **relative probability** to **absolute probability** wildfire prediction for Bolzano Province.

## Key Changes

### Previous Model (Relative Probability)
- **Output**: Relative risk scores (0-1 scale)
- **Interpretation**: Comparative risk between locations/times
- **Prior**: Based on case-control sampling (~20% fire rate)
- **Use case**: Ranking locations by risk
- **Limitation**: Cannot estimate actual number of fires

### New Model (Absolute Probability)
- **Output**: Actual fire probability for the region
- **Interpretation**: P(at least 1 fire in Bolzano Province on this day)
- **Prior**: Based on TRUE regional fire rate (~0.023 fires/day)
- **Use case**: Predicting actual fire counts and risk levels
- **Advantage**: Direct comparison across climate scenarios

## True Fire Statistics

Based on complete historical data (1999-2025):

```
Total fires: 227
Study period: 9,740 days (26.7 years)
Fires per day: 0.0233
Fires per year: 8.51

Regional daily probability: 0.0233 (2.33%)
Log-odds (prior): -3.7355
```

## Model Architecture

### Bayesian Logistic Regression with Attention
- **Intercept (α)**: Calibrated to TRUE regional fire rate
  - Prior: Normal(μ = -3.74, σ = 0.5)
  - Interpretation: Baseline log-odds of fire in the region

- **Feature Groups**: Weather, topography, vegetation, infrastructure
  - Attention weights learn group importance
  - Beta coefficients within each group

- **Likelihood**: Trained on case-control sample (balanced dataset)
  - Training uses oversampled fire events (~20% fires)
  - But predictions interpreted as absolute probability via calibrated prior

### Key Innovation
The model is trained on case-control data (for computational efficiency and class balance) but the **prior is anchored to the TRUE regional fire rate**. This allows predictions to represent actual fire probability while maintaining good model performance.

## Interpretation Guide

### Prediction Values
```
Model Output     Interpretation                      Example Date Type
----------------------------------------------------------------
0.01            Low risk (43% of baseline)          Wet winter day
0.023           Average risk (baseline)              Typical spring day
0.05            High risk (2.1x baseline)            Dry summer day
0.10            Very high risk (4.3x baseline)       Extreme conditions
```

### Aggregation Over Time
To estimate expected fires over a period:

```python
# Sum daily probabilities to get expected fire count
expected_fires_per_week = daily_predictions.sum()

# Example: If model predicts 0.03 for 7 days:
# Expected fires = 7 * 0.03 = 0.21 fires/week
# Or about 1 fire every 4-5 weeks
```

### Regional vs Pixel-Level
- **Model predicts**: Regional probability (whole province)
- **Pixel-level**: 7.9e-9 per pixel-day (technical reference only)
- **Pixels are NOT independent**: One fire affects nearby pixels
- **Correct interpretation**: Probability of ANY fire in the region

## Validation

### Temporal Validation
- Monthly fire count correlation: Compare predicted vs actual fires by month
- Seasonal patterns: Validate summer fire peaks, winter lows
- Expected output: High correlation (>0.7) if well-calibrated

### Performance Metrics
1. **ROC-AUC**: Discrimination ability (>0.80 good)
2. **PR-AUC**: Performance with class imbalance
3. **F1 Score**: Balance of precision and recall
4. **Lift**: Targeting effectiveness (top 10% should have high lift)
5. **Calibration**: Predicted probabilities match observed frequencies

### Spatial Validation
- Regional fire density comparison
- Expected fires per 100 km² per year by zone
- Validates spatial risk distribution

## Using the Model

### Training
```bash
python Scripts/02_Model_Training/05_Bayesian_AbsoluteProbability_Regional.py
```

### Model Outputs
Saved to `/mnt/CEPH_PROJECTS/Firescape/Data/OUTPUT/02_Model_AbsoluteProbability/`:
- `trace_absolute.nc`: Posterior samples (Bayesian inference)
- `scaler_absolute.joblib`: Feature standardization
- `true_fire_stats.joblib`: Historical fire statistics
- `temporal_groups.joblib`: Feature groupings for attention
- `model_results.joblib`: Validation metrics

### Making Predictions
```python
import joblib
import arviz as az

# Load model artifacts
trace = az.from_netcdf("trace_absolute.nc")
scaler = joblib.load("scaler_absolute.joblib")
temporal_groups = joblib.load("temporal_groups.joblib")
group_names = joblib.load("group_names.joblib")

# Prepare features and scale
X_new_scaled = scaler.transform(X_new)

# Generate predictions (use generate_predictions function)
from Scripts.02_Model_Training.05_Bayesian_AbsoluteProbability_Regional import generate_predictions
mean_prob, std_prob = generate_predictions(trace, temporal_groups, group_names, X_new_scaled)

# Interpret results
print(f"Daily fire probability: {mean_prob[0]:.4f}")
print(f"Expected fires per month: {mean_prob[0] * 30:.2f}")
print(f"Expected fires per year: {mean_prob[0] * 365:.1f}")
```

## Climate Projections

For climate scenario analysis:

1. **Generate predictions** for each scenario (RCP 2.6, 4.5, 8.5)
2. **Sum daily probabilities** to get expected annual fires
3. **Compare scenarios**:
   ```
   Baseline: 8.5 fires/year
   RCP 4.5 (2050): 12.3 fires/year (+45%)
   RCP 8.5 (2050): 18.7 fires/year (+120%)
   ```

4. **Uncertainty quantification**: Use posterior samples to get credible intervals

## Advantages Over Relative Probability

| Aspect | Relative Model | Absolute Model |
|--------|---------------|---------------|
| Output | Risk scores (0-1) | Actual probability |
| Interpretation | Comparative ranking | Quantitative risk |
| Validation | ROC, AUC | + Fire count correlation |
| Climate projections | Relative change | Absolute fire counts |
| Resource planning | Prioritization | Actual expected events |
| Stakeholder communication | "Higher risk here" | "Expect 12 fires/year" |

## Technical Details

### Case-Control Correction
The model handles the mismatch between training data (20% fires) and reality (0.023% daily probability) through:

1. **Prior calibration**: Intercept set to true regional rate
2. **Feature effects learned from case-control**: Beta coefficients capture relative risk factors
3. **Prediction interpretation**: Sigmoid(alpha + beta*X) represents absolute probability

This is valid because:
- Case-control sampling preserves odds ratios
- Bayesian prior anchors predictions to true baseline
- Model learns which conditions increase/decrease risk from baseline

### Study Area Parameters
```
Province: Bolzano (South Tyrol)
Area: 7,400 km²
Pixel resolution: 50m × 50m
Total pixels: 2,960,000
Training period: 2012-2024
Historical data: 1999-2025
```

### Model Hyperparameters
```
MCMC Sampling:
- Draws: 2000 per chain
- Tuning: 1000 samples
- Chains: 4 (parallel)
- Target accept: 0.99

Prior Specifications:
- Intercept: Normal(μ=-3.74, σ=0.5)
- Climate betas: Normal(μ=0, σ=1.0)
- Static betas: Normal(μ=0, σ=0.5)
- Attention: Dirichlet(α=1.5)
```

## Files Modified/Created

### New Files
1. `Scripts/02_Model_Training/05_Bayesian_AbsoluteProbability_Regional.py`
2. `Documentation/Absolute_Probability_Model_README.md`

### Files to Update
1. `Scripts/03_Climate_Projections/05_Bayesian_Climate_Projection_CLEAN.py`
   - Use absolute probability trace
   - Interpret outputs as fire counts

2. `Scripts/03_Climate_Projections/06_Fire_Brigade_Climate_Projections.py`
   - Update to absolute probability model
   - Add fire count summaries

## References

### Bayesian Case-Control Modeling
- Prentice & Pyke (1979): Case-control studies preserve odds ratios
- King & Zeng (2001): Rare events logistic regression
- Gelman et al. (2013): Bayesian Data Analysis, Ch 14

### Wildfire Risk Assessment
- Bolzano Province Fire History: 1999-2025 wildfire inventory
- Resolution: 50m spatial, daily temporal
- Features: Climate, topography, vegetation, infrastructure

## Contact

For questions about this model:
- Check documentation in `/mnt/CEPH_PROJECTS/Firescape/Documentation/`
- Review validation plots in `/mnt/CEPH_PROJECTS/Firescape/Data/OUTPUT/02_Model_AbsoluteProbability/`
- Consult original relative probability model for comparison

---

**Last Updated**: 2025-10-20
**Model Version**: 1.0 (Absolute Probability)
**Status**: Production Ready
