# Province-Wide Baseline Implementation Summary

**Date:** October 20, 2025
**Update:** Model interpretation shifted from pixel-level probabilities to province-wide fire frequency

---

## What Changed

### Core Concept Shift

**Before:** Model outputs interpreted as pixel-level daily fire probabilities (~10⁻⁷)

**After:** Model outputs interpreted as relative risk scores predicting where ~28 annual fires will occur

---

## Why This Change

The previous pixel-level probability approach (1 in 10 million pixel-days) was:
- ❌ Technically correct but practically confusing
- ❌ Not operationally meaningful for fire brigades
- ❌ Obscured the actual fire frequency (28 fires/year)

The new province-wide approach is:
- ✅ Intuitive: "The province experiences ~28 fires per year"
- ✅ Operational: "Model predicts WHERE those fires will occur"
- ✅ Actionable: "High-risk zones will see more of those 28 fires"

---

## Implementation

### 1. Model Training Script Updated

**File:** `Scripts/02_Model_Training/04_Bayesian_pyMCLogisticRegression_Linear_Attention_commented.py`

**Lines 1466-1547:** Added true baseline calculation and display

**Key statistics saved:**
```python
true_baseline = {
    'fires_per_year': 28.2,              # Province-wide annual rate
    'fires_per_100km2_per_year': 0.38,   # Fire density
    'case_control_fire_rate': 0.2055,    # Training data (artifact)
    'pixel_day_probability': 1.04e-7     # Technical reference only
}
```

**Console output shows:**
- ⚠ Case-control rate: 20.55% (NOT the true rate)
- ✓ True baseline: ~28 fires/year
- 📊 Model interpretation: Relative risk scores
- 💡 Examples: How to interpret scores
- 🔬 Technical detail: Pixel probability (for reference)

### 2. Documentation Updated

#### BASELINE_FIRE_RATE_ANALYSIS.md (Major rewrite)

**Key sections updated:**
- **Province-Wide Calculation:** Emphasizes 28 fires/year baseline
- **Model Interpretation:** Scores predict WHERE fires occur, not absolute probabilities
- **Fire Brigade Guidance:** Operational thresholds based on risk scores
- **Code Implementation:** Shows how baseline is calculated and used

#### README.md

**Additions:**
- Province-wide baseline section (lines 154-158)
- Model interpretation section (lines 166-169)
- Recent updates: True baseline calculation (line 183)
- Recent updates: Model interpretation category (lines 190-193)

#### README_FIRE_BRIGADE_WORKFLOW.md

**Updates:**
- **Scientific Interpretation section (lines 389-437):**
  - Province-wide baseline context
  - Risk score interpretation (not probabilities)
  - Operational implications with fire frequency examples
- **Citation section:** Updated to include baseline context

#### FIRE_BRIGADE_INTEGRATION.md

**Updates:**
- **Key Features section:** Added province-wide baseline subsection
- **Expected Results section:** Province-wide fire frequency context
- Changed "fire probability" to "fire risk score" throughout

---

## Key Messages

### For Scientists

1. **Model outputs are relative risk scores** (0-1 scale)
2. Trained on case-control data (20.55% fire rate is sampling artifact)
3. True baseline: 28 fires/year in province
4. Scores rank locations/times by fire likelihood
5. Climate projections show changes in fire frequency and distribution

### For Fire Brigades

1. **Province experiences ~28 fires per year** (current baseline)
2. Model predicts **where** those fires will occur
3. **Risk score 0.20-0.35:** High risk - fires concentrate here
4. **Risk score >0.35:** Very high risk - top priority
5. Climate change may increase to ~40-45 fires/year by 2080
6. Use score increases to prioritize resource allocation

### For Publications

**Recommended wording:**

> "The Bayesian model outputs fire risk scores on a 0-1 scale, representing relative
> risk compared to the study baseline. Model was trained on case-control data
> (366 fires, 1:4 sampling ratio) to maximize statistical power. The true baseline
> fire rate in Bolzano Province is approximately 28 fires per year (0.38 fires per
> 100 km² per year, 2012-2024). Model scores indicate spatial and temporal variation
> in fire likelihood, helping predict where these fires will occur. Climate projections
> show relative changes in fire risk, with potential 40-60% increase in fire frequency
> by 2080 under RCP 8.5."

---

## Files Modified

### Code
1. `Scripts/02_Model_Training/04_Bayesian_pyMCLogisticRegression_Linear_Attention_commented.py`
   - Lines 1466-1547: True baseline calculation and output

### Documentation
1. `Documentation/BASELINE_FIRE_RATE_ANALYSIS.md` - Comprehensive rewrite
2. `README.md` - Added baseline context, updated interpretation
3. `Scripts/04_Fire_Brigade_Analysis/README_FIRE_BRIGADE_WORKFLOW.md` - Updated scientific interpretation
4. `Documentation/FIRE_BRIGADE_INTEGRATION.md` - Added baseline context

### New Files
1. `Documentation/BASELINE_UPDATE_SUMMARY.md` - This document

---

## Output Artifacts

When model training runs, it now creates:
- `Data/OUTPUT/02_Model/true_baseline_stats.joblib` - Contains province-wide statistics

**Contents:**
- total_fires: 366
- fires_per_year: 28.2
- fires_per_100km2_per_year: 0.38
- case_control_fire_rate: 0.2055
- pixel_day_probability: 1.04e-7 (for reference)

---

## Benefits

### Clarity
- ✅ Removes confusion about "20% fire rate"
- ✅ Provides intuitive province-wide context
- ✅ Clear distinction between relative scores and absolute frequencies

### Operational Utility
- ✅ Fire brigades understand "28 fires/year" baseline
- ✅ Can plan resources based on expected fire frequency changes
- ✅ Risk scores provide spatial prioritization

### Scientific Rigor
- ✅ Correctly represents case-control study design
- ✅ Maintains technical accuracy (pixel probabilities still calculated)
- ✅ Emphasizes appropriate use of relative risk scores

---

## Next Steps (Optional)

1. **Use baseline in projection outputs:**
   - Add province-wide fire frequency estimates to climate projection scripts
   - Example: "Current: 28 fires/year → Projected: 42 fires/year by 2080"

2. **Incorporate into fire brigade analysis:**
   - Calculate expected fires per zone based on risk scores
   - Example: "Zone A: 2 fires/year → 3.5 fires/year by 2080"

3. **Validation studies:**
   - Compare model predictions against actual fire locations
   - Verify that fires concentrate in high-scoring areas

---

## Technical Notes

### Relationship Between Scores and Frequencies

Model scores don't directly convert to fire counts, but indicate **relative likelihood**:

- If all locations had equal risk (score = 0.20), fires would be uniformly distributed
- Higher scores (e.g., 0.40) indicate 2× higher likelihood
- The ~28 annual fires will preferentially occur in highest-scoring locations

### Seasonal Variation

The 28 fires/year is an annual average. Actual risk varies by season:
- Summer (Jul-Aug): ~60% of fires → ~17 fires in 2 months
- Winter (Dec-Feb): ~10% of fires → ~3 fires in 3 months

Model scores capture this temporal variation.

---

**Last Updated:** October 20, 2025
