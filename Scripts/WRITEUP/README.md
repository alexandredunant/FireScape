# WriteUp Directory

LaTeX manuscript and supporting materials for the FireScape publication.

## Files

- **main.tex** - Main manuscript file
- **instruction_tex_file.md** - Writing instructions and objectives
- **references.bib** - Bibliography (if separate from main.tex)
- **figures/** - Publication figures (linked from `../05_Lightning_Comparison/03_Model_Comparison/Results/`)

## Manuscript Status

**Current Title:** "Lightning, Fire, and Climate: A Bayesian Analysis of Alpine Regime Shift in the Bolzano Mountains"

**Structure:**
- Abstract ✓
- Introduction ✓
- Methods ✓
- Results ⚠️ (needs updating with corrected metrics)
- Discussion ⚠️ (needs updating)
- Conclusion ✓

## Key Updates Needed

### Results Section
Update with corrected lightning comparison metrics:
- **Temporal improvement:** +38-41% (not +5% as currently stated)
- **Discrimination change:** -3.8% to -21.1% ROC/PR-AUC
- **Trade-off interpretation:** Better temporal fit vs. lower discrimination

### Figures to Include

From `Scripts/05_Lightning_Comparison/03_Model_Comparison/Results/`:

**Main Text:**
1. **Figure 1:** `fig1_monthly_temporal_fit.png` - Monthly temporal accuracy
2. **Figure 2:** `fig2_seasonal_temporal_fit.png` - Seasonal patterns
3. **Figure 3:** `fig6_tradeoff_summary.png` - Main result (trade-off)

**Supplementary:**
- **Figure S1:** `fig3_roc_curve.png` - Discrimination performance
- **Figure S2:** `fig4_precision_recall_curve.png` - Precision-recall curves
- **Figure S3:** `fig5_feature_importance.png` - Feature attention weights

## Compiling

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use your preferred LaTeX editor (Overleaf, TeXShop, etc.)

## Target Journal

[To be determined - suggest Nature Communications, Science Advances, or domain-specific like Fire Ecology]

## Co-authors

Update author list in `main.tex` lines 59-67

## Questions

See `instruction_tex_file.md` for original writing objectives and goals.
