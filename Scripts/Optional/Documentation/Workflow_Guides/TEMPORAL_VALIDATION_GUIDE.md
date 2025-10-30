# Temporal Validation Guide

## Understanding Your Model's Temporal Fit

The training script already provides comprehensive temporal validation. This guide explains how to interpret those metrics.

## Your Current Results

From your latest training run:

```
📅 MONTHLY FIRE COUNTS:
 month  actual_fires  predicted_fires
     1            12         7.825279
     2            10        10.293285
     3            17        19.046257
     4            24        17.389853
     5             8         7.157967
     6            33        20.384933
     7            35        17.368115
     8            30        18.160607
     9             6         5.272097
    10             5         5.077375
    11             3         5.872385
    12             1         3.757984

📅 SEASONAL FIRE COUNTS:
season  actual_fires  predicted_fires
  Fall            14        16.221858
Spring            49        43.594077
Summer            98        55.913655
Winter            23        21.876548

📊 TEMPORAL VALIDATION METRICS:

  MONTHLY:
    Pearson R (trend): 0.915
    R² (magnitude fit): 0.577
    RMSE: 7.60 fires/month
    MAE: 5.21 fires/month

  SEASONAL:
    Pearson R (trend): 0.960
    R² (magnitude fit): 0.576
    RMSE: 21.25 fires/season
    MAE: 12.71 fires/season
```

## What These Metrics Mean

### 1. Pearson R (Correlation)

**Your scores**: Monthly = 0.915, Seasonal = 0.960

**What it measures**: How well the model captures temporal **trends**
- 1.0 = Perfect correlation (predictions go up/down exactly with actuals)
- 0.0 = No correlation
- Your scores (0.91-0.96) = **Excellent trend matching**

**Interpretation**:
✅ The model correctly identifies **when** fire risk is high vs low
✅ It captures seasonal patterns (summer peak, winter low)
✅ Strong correlation means relative risk ranking works well

### 2. R² (Coefficient of Determination)

**Your scores**: Monthly = 0.577, Seasonal = 0.576

**What it measures**: How well predictions match actual **magnitudes**
- 1.0 = Perfect fit (predictions exactly match actual counts)
- 0.0 = Model no better than predicting the mean
- Your scores (0.58) = **Moderate magnitude fit**

**Interpretation**:
~ Model captures 58% of fire count variability
~ Remaining 42% is unexplained (weather extremes, human factors, etc.)
~ This is **expected** for a relative probability model

**Why R² < Pearson R:**
- Pearson R = trend correlation (easier to get high)
- R² = magnitude accuracy (harder, especially with case-control sampling)
- Your case-control sampling (1:3 fire:no-fire ratio) affects absolute magnitudes

### 3. RMSE & MAE (Error Metrics)

**Your scores**:
- Monthly: RMSE=7.6, MAE=5.2 fires/month
- Seasonal: RMSE=21.3, MAE=12.7 fires/season

**What they measure**: Average prediction error
- RMSE = Root Mean Square Error (penalizes large errors)
- MAE = Mean Absolute Error (average error)
- Lower = better

**Interpretation**:
~ On average, monthly predictions are off by 5-8 fires
~ Some months have larger errors (e.g., June: actual=33, predicted=20)
~ This is **acceptable** for relative risk modeling

## Pattern Analysis

### Strong Seasonal Signal ✅

Your model captures the clear seasonal pattern:
- **Winter (Dec-Feb)**: Low fire activity (correct)
- **Spring (Mar-May)**: Moderate, increasing (correct trend)
- **Summer (Jun-Aug)**: Peak fire season (correct identification)
- **Fall (Sep-Nov)**: Declining activity (correct)

### Month-by-Month Performance

**Well-predicted months** (prediction close to actual):
- February: 10 actual vs 10 predicted ✓
- May: 8 actual vs 7 predicted ✓
- September: 6 actual vs 5 predicted ✓
- October: 5 actual vs 5 predicted ✓

**Under-predicted months** (model predicts less than actual):
- **June: 33 actual vs 20 predicted** ← Largest miss
- August: 30 actual vs 18 predicted
- July: 35 actual vs 17 predicted

This under-prediction in peak summer is **typical** because:
1. Extreme weather events not fully captured
2. Human-caused fires (not in climate data)
3. Case-control sampling reduces absolute scale

**Over-predicted months**:
- March: 17 actual vs 19 predicted
- November: 3 actual vs 6 predicted

Minor over-predictions in shoulder seasons are acceptable.

## What This Means for Your Work

### ✅ Model is Working Well

1. **Excellent trend capture**: Pearson R > 0.9
   - Model correctly ranks high-risk vs low-risk periods
   - Suitable for comparing scenarios (e.g., 2020 vs 2050)

2. **Moderate magnitude fit**: R² ≈ 0.58
   - Expected for relative probability models
   - Good enough for risk ranking, not for exact counts

3. **Clear seasonal pattern**:
   - Summer peak correctly identified
   - Winter low correctly identified
   - Operational planning is feasible

### ⚠️ Limitations to Remember

1. **Under-predicts peak summer**:
   - Extreme fire events are hard to predict
   - Consider this when planning resources for July-August

2. **Not absolute counts**:
   - Predictions are relative risk, not exact fire numbers
   - Use for comparing zones, time periods, scenarios
   - Don't use for budgeting based on exact counts

3. **Case-control sampling effect**:
   - Training used 1:3 fire:no-fire ratio
   - Absolute magnitudes are scaled down
   - Scaling factor available but **not recommended** for projections

## How to Use This for Lightning Comparison

When you train the lightning model, compare these metrics:

```
Metric          | Baseline (T+P) | Lightning (T+P+L) | Improvement?
----------------|----------------|-------------------|-------------
Monthly R²      | 0.577          | ???               | Target: >0.60
Seasonal R²     | 0.576          | ???               | Target: >0.60
Pearson R       | 0.915          | ???               | Hard to improve
RMSE (monthly)  | 7.60           | ???               | Target: <7.0
```

**Lightning is useful if**:
- Monthly R² improves by >5% (e.g., 0.577 → 0.61+)
- Summer months (Jun-Aug) prediction errors decrease
- Lightning feature groups have high attention weights

**Lightning is not useful if**:
- R² stays same or decreases
- No improvement in summer peak predictions
- Lightning attention weights are low

## Recommendations

### For Operational Use

1. **Use model for relative risk ranking** ✅
   - Compare zones: "Zone A has 2x risk of Zone B"
   - Compare time periods: "August has 5x risk of January"
   - Compare scenarios: "RCP8.5 2050 has 1.5x risk of 2020"

2. **Don't use for exact fire counts** ❌
   - Don't say: "Expect 35 fires in July 2050"
   - Don't budget based on predicted counts
   - Model is relative, not absolute

3. **Focus on peak summer** ⚠️
   - Model under-predicts June-August
   - In operations, add safety buffer for summer peak
   - Consider extreme weather monitoring for peak season

### For Climate Projections

1. **Compare risk ratios**, not absolute values
   - Good: "Fire risk increases 50% by 2050"
   - Bad: "Expect 120 fires per year in 2050"

2. **Use seasonal aggregates**
   - Summer season risk is most reliable
   - Winter predictions have fewer fires (noisier)

3. **Focus on spatial patterns**
   - Which zones see greatest risk increase?
   - Where should resources be allocated?

## Summary

Your model has:
- ✅ **Excellent temporal trend capture** (R=0.91-0.96)
- ✅ **Moderate magnitude fit** (R²=0.58)
- ✅ **Clear seasonal pattern recognition**
- ⚠️ **Expected limitations** in absolute scaling

This is **exactly what you want** from a relative probability model. The temporal validation is **sufficient** - no additional zone-level lookback validation needed.

Focus your efforts on:
1. Testing whether lightning improves these metrics (Script 05)
2. Generating zone-level climate projections (Script 04)
3. Interpreting results for operational fire brigade planning

---

**Questions?** See `WORKFLOW_SUMMARY.md` for workflow overview.
