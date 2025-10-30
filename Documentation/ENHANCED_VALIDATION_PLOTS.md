# Enhanced Validation Plots - Model Training

**Date:** October 20, 2025
**Status:** ✓ Implemented
**Requested by:** User feedback on validation metrics

---

## Enhancements Added

### 1. Sample Counts on PR Curve ✓

**Purpose:** Contextualize class imbalance for better interpretation

**Implementation:**
```python
# Calculate class counts
n_positive = int(np.sum(y_test))
n_negative = len(y_test) - n_positive

# Add text box with sample counts
sample_info = f'Samples:\n  Fires: {n_positive}\n  Non-fires: {n_negative}\n  Imbalance: 1:{n_negative/n_positive:.1f}'
axes[1].text(0.98, 0.02, sample_info, ha='right', va='bottom',
            transform=axes[1].transAxes,
            bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.8))
```

**What it shows:**
- Total number of fire vs non-fire samples
- Imbalance ratio (e.g., 1:3.9 means 3.9× more non-fires than fires)
- Helps interpret why PR-AUC might be lower than ROC-AUC (imbalance effect)

---

### 2. Cross-Validation Confidence Bands on ROC Curve ✓

**Purpose:** Show robustness and variability across CV folds

**Implementation:**
```python
# Add CV confidence bands if available
if cv_results and 'fold_results' in cv_results:
    valid_folds = [f for f in cv_results['fold_results']
                  if 'error' not in f and not np.isnan(f.get('roc_auc', float('nan')))]
    if len(valid_folds) > 1:
        # Interpolate all fold ROC curves to common FPR points
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        for fold in valid_folds:
            if 'y_pred' in fold and 'y_true' in fold:
                fold_fpr, fold_tpr, _ = roc_curve(fold['y_true'], fold['y_pred'])
                tprs.append(np.interp(mean_fpr, fold_fpr, fold_tpr))

        tprs = np.array(tprs)
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        axes[0].fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                            alpha=0.2, color='blue', label=f'±1 std (n={len(valid_folds)} folds)')
```

**What it shows:**
- Shaded area representing ±1 standard deviation across CV folds
- Number of folds used to calculate confidence band
- Wider band = more variability across folds (less robust)
- Narrow band = consistent performance (more robust)

---

### 3. Calibration Plot (Reliability Diagram) ✓

**Purpose:** Show if predicted probabilities match observed frequencies

**Replaces:** Cumulative Lift Curve (less informative for fire risk)

**Implementation:**
```python
# PLOT 3: CALIBRATION PLOT
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)

observed_freq = []
predicted_freq = []
bin_counts = []

for i in range(n_bins):
    mask = (mean_prob >= bin_edges[i]) & (mean_prob < bin_edges[i+1])
    if np.sum(mask) > 0:
        observed_freq.append(np.mean(y_test[mask]))
        predicted_freq.append(np.mean(mean_prob[mask]))
        bin_counts.append(np.sum(mask))

# Plot calibration curve
axes[2].plot(predicted_freq, observed_freq, marker='o', color='darkgreen', lw=2)
axes[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Perfect Calibration')

# Add histogram showing distribution of predictions
ax2_twin = axes[2].twinx()
ax2_twin.bar(bin_centers, bin_counts, width=1/n_bins, alpha=0.3, color='gray')
```

**What it shows:**
- **X-axis:** Predicted fire probability (model output)
- **Y-axis:** Observed fire frequency (actual proportion of fires)
- **Perfect line:** If model is perfectly calibrated, points lie on diagonal
- **Above diagonal:** Model underestimates risk (predicts lower than actual)
- **Below diagonal:** Model overestimates risk (predicts higher than actual)
- **Gray bars:** Distribution of predictions (helps identify sparse bins)

**Why calibration matters for fire risk:**
- Operational decisions depend on probability magnitudes, not just rankings
- If model says "20% probability", it should mean fires occur ~20% of the time
- Important for resource allocation and risk communication

---

## Updated Validation Plot Layout

```
┌─────────────────────────────────────────────────────────────┐
│            Model Validation Metrics                          │
├──────────────┬──────────────────┬──────────────────────────┤
│              │                  │                           │
│   ROC Curve  │   PR Curve       │   Calibration Plot       │
│              │                  │                           │
│  • Main line │ • Sample counts  │  • Predicted vs observed │
│  • CV bands  │ • Optimal F1     │  • Perfect line          │
│  • Random    │ • Baseline       │  • Sample distribution   │
│              │                  │                           │
└──────────────┴──────────────────┴──────────────────────────┘
```

---

## Example Interpretation

### Good Calibration:
```
Predicted: 0.1  → Observed: 0.11  ✓ (close to diagonal)
Predicted: 0.3  → Observed: 0.28  ✓
Predicted: 0.6  → Observed: 0.62  ✓
```
→ Model probabilities are reliable for decision-making

### Poor Calibration:
```
Predicted: 0.1  → Observed: 0.25  ✗ (above diagonal)
Predicted: 0.3  → Observed: 0.50  ✗
Predicted: 0.6  → Observed: 0.75  ✗
```
→ Model systematically underestimates risk, needs recalibration

---

## Benefits for Fire Risk Modeling

1. **Sample Counts**: Essential for understanding why PR-AUC differs from ROC-AUC
   - Fire data is typically imbalanced (more non-fire days than fire days)
   - PR curve more sensitive to this imbalance
   - Seeing the ratio (e.g., 1:3.9) helps contextualize the metrics

2. **CV Confidence Bands**: Shows robustness across temporal/spatial folds
   - Narrow bands = model generalizes well across different time periods and locations
   - Wide bands = performance varies, may need more data or better features
   - Critical for operational deployment confidence

3. **Calibration Plot**: Validates probability estimates for operational use
   - Fire brigades need accurate risk probabilities for resource planning
   - Policy decisions (e.g., fire warnings) depend on threshold calibration
   - Identifies if post-processing calibration is needed (e.g., Platt scaling)

---

## Files Modified

**Script:** `04_Bayesian_pyMCLogisticRegression_Linear_Attention_commented.py`

**Function:** `create_validation_plots(y_test, mean_prob, cv_results=None)`
- Lines 472-606: Complete rewrite with enhancements
- Lines 1455: Updated function call to pass cv_results

**Output:** `Data/OUTPUT/02_Model/validation_plots.png`
- Updated figure with all three enhancements
- Saved at 300 DPI for publication quality

---

## Usage

The enhanced plots are generated automatically during model training:

```python
# In run_linear_analysis():
roc_auc, pr_auc, max_f1, opt_thresh = create_validation_plots(
    y_test, mean_prob,
    cv_results=cv_results  # Passes CV results for confidence bands
)
```

No additional user action required - plots are saved to:
`/mnt/CEPH_PROJECTS/Firescape/Data/OUTPUT/02_Model/validation_plots.png`

---

## Next Steps (Optional Future Enhancements)

1. **Spatial Validation Maps** (suggested by user):
   - Plot predicted vs observed fire frequency by geographic region
   - Identify areas where model performs better/worse
   - Could use fire brigade zones as spatial units

2. **Temporal Validation:**
   - Show performance across different months/seasons
   - Identify if model is better for summer vs winter fires

3. **Reliability Index:**
   - Calculate Expected Calibration Error (ECE)
   - Quantify how far predictions deviate from perfect calibration

---

**Status:** ✓ All requested enhancements implemented and ready for next training run

**Last Updated:** October 20, 2025
