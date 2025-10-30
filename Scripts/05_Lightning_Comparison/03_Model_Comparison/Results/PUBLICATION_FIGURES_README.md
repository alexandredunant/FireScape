# Publication-Quality Figures

All figures have been updated to publication standards using the `scienceplots` library with the following specifications:

## Style Settings
- **Style**: `['science', 'no-latex']` - Professional scientific style without LaTeX dependencies
- **No titles**: All figure titles removed (titles should be in figure captions in manuscript)
- **High DPI**: All figures saved at 600 DPI for print quality
- **Dual format**: Both PNG (600 dpi) and PDF versions saved
- **Consistent colors**:
  - Observed data: `#2C3E50` (dark gray)
  - Baseline model: `#3498DB` (blue)
  - Lightning model: `#E67E22` (orange)

## Generated Figures

### Comparison Figures (from `compare_models.py`)

1. **fig1_monthly_temporal_fit.png** (.pdf)
   - Monthly fire counts: Observed vs Baseline vs Lightning
   - Bar chart showing temporal fit improvement
   - R² values annotated in text box

2. **fig2_seasonal_temporal_fit.png** (.pdf)
   - Seasonal fire counts: Observed vs Baseline vs Lightning
   - Clear visualization of summer peak improvement
   - R² values annotated in text box

3. **fig3_roc_curve.png** (.pdf)
   - ROC curves for both models
   - Shows discrimination performance
   - AUC values in legend

4. **fig4_precision_recall_curve.png** (.pdf)
   - Precision-Recall curves for both models
   - Shows performance with imbalanced classes
   - Baseline random performance shown

5. **fig5_feature_importance.png** (.pdf)
   - Top 10 feature groups by attention weight
   - Horizontal bar chart comparing models
   - Shows lightning feature contributions

6. **fig6_tradeoff_summary.png** (.pdf)
   - Two-panel figure summarizing trade-off
   - Panel A: Bar comparison of all metrics
   - Panel B: Change visualization with percent improvements

### Training Figures (from training scripts)

Both training scripts now generate publication-quality figures:
- **Prior distributions** - No titles, clean styling
- **Temporal validation** - Monthly and seasonal fits
- **Model performance** - ROC, PR curves, calibration, lift

All titles removed and `scienceplots` styling applied.

## Updated Files

1. **Scripts/05_Lightning_Comparison/03_Model_Comparison/compare_models.py**
   - Added `scienceplots` import and style
   - Replaced 4-panel combined plot with 6 separate figures
   - Removed all titles
   - Increased DPI to 600
   - Added PDF output

2. **Scripts/05_Lightning_Comparison/02_Model_Training/train_relative_probability_model_with_lightning.py**
   - Added `scienceplots` import and style
   - Commented out all `set_title()` calls
   - Figures now ready for publication

3. **Scripts/02_Model_Training/train_relative_probability_model.py**
   - Added `scienceplots` import and style
   - Commented out all `set_title()` calls
   - Figures now ready for publication

## Usage in Manuscript

Suggested figure organization:

- **Main Text**:
  - Figure 1: `fig1_monthly_temporal_fit.png` - Shows improved temporal accuracy
  - Figure 2: `fig2_seasonal_temporal_fit.png` - Emphasizes summer fire season
  - Figure 3: `fig6_tradeoff_summary.png` - Main result showing trade-off

- **Supplementary Material**:
  - Figure S1: `fig3_roc_curve.png` - Discrimination performance
  - Figure S2: `fig4_precision_recall_curve.png` - Imbalanced class performance
  - Figure S3: `fig5_feature_importance.png` - Model interpretability

## Figure Captions (Suggested)

**Figure 1. Monthly temporal fit comparison.**
Comparison of observed fire counts (dark gray) with Baseline (Temperature + Precipitation, blue) and Lightning (Temperature + Precipitation + Lightning, orange) model predictions for each month during 2012-2024. The Lightning model shows substantially improved temporal fit (R² = 0.767) compared to the Baseline model (R² = 0.556), particularly capturing the summer fire peak (months 7-8).

**Figure 2. Seasonal temporal fit comparison.**
Seasonal aggregation of fire counts showing Observed, Baseline, and Lightning model predictions. The Lightning model demonstrates exceptional seasonal accuracy (R² = 0.814 vs Baseline R² = 0.579), with marked improvement in capturing the summer fire season peak.

**Figure 3. Performance trade-off summary.**
(A) Comparison of model performance across temporal and discrimination metrics. The Lightning model shows substantial gains in temporal metrics (Monthly R²: +38.0%, Seasonal R²: +40.7%) at the cost of moderate declines in discrimination metrics (ROC-AUC: -3.8%, PR-AUC: -21.1%). (B) Absolute and relative changes highlighting the trade-off between temporal accuracy and event discrimination.

## Notes

- All figures use consistent styling and color schemes
- No titles on figures (as per publication standards)
- High resolution (600 DPI) suitable for print
- Both raster (PNG) and vector (PDF) formats provided
- All code updated to maintain this standard going forward
