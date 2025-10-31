# LaTeX Manuscript Update Summary

## Overview

Updated `main.tex` to reflect the actual analyses performed and figures generated. Created `main_updated.tex` with accurate results, proper figure references, and workflow diagram integration.

## Major Changes

### 1. **Corrected Abstract and Objectives**
- **Before**: Claimed 83% regime shift probability, which was speculative
- **After**: Focus on model comparison (T+P vs T+P+L) with actual results
- **Key finding**: Lightning improves temporal fit (+56% seasonal R²) with small trade-off in discrimination (-4% ROC-AUC)

### 2. **Updated Results to Match Actual Data**

From `output/figures/comparison_summary.csv`:

| Metric | Baseline (T+P) | Lightning (T+P+L) | Change |
|--------|----------------|-------------------|---------|
| Monthly R² | 0.769 | 0.927 | +20.5% |
| Seasonal R² | 0.640 | 1.000 | +56.3% |
| ROC-AUC | 0.836 | 0.805 | -3.7% |
| PR-AUC | 0.654 | 0.516 | -21.1% |

### 3. **Integrated Actual Figures**

All figures now reference actual generated files in `../output/figures/`:

- `validation_temporal_baseline.png` and `validation_temporal_lightning.png`
- `validation_performance_baseline.png` and `validation_performance_lightning.png`
- `temporal_correlations_comparison.png`
- `model_comparison.png`
- `climate_projection_plot_march.png` and `climate_projection_plot_august.png`

### 4. **Added Workflow Diagram**

Integrated the workflow diagram from `diagram.html` into the appendix, adapted to show:
- Data inputs (meteorological, lightning, topographic, wildfire inventory)
- Processing steps (datacube construction, feature engineering)
- Bayesian model with attention mechanism
- Outputs (relative probability, uncertainty, feature importance)

### 5. **Corrected Methods Section**

- **Data period**: Baseline 1999-2025 (911 samples), Lightning 2012-2025 (467 samples)
- **Feature engineering**: Accurate description of temporal windows (1, 3, 5, 10, 15, 30, 60 days)
- **Attention mechanism**: Proper description of feature grouping
- **Model implementation**: PyMC v5.10 with actual parameters (2000 draws, 1000 tuning, 4 chains)

### 6. **Revised Discussion**

Key additions:
- **Complementary strengths**: Lightning excels at temporal patterns, baseline at spatial discrimination
- **Use-case dependent selection**: Choose model based on whether "when" (temporal) or "where" (spatial) is priority
- **Operational implications**: Two-model approach suggested
- **Removed speculative content**: No regime shift claims, no forestry activity analysis (not performed)

### 7. **Climate Projections Section**

Added brief results showing RCP4.5 and RCP8.5 projections for March and August, referencing actual figures.

### 8. **Removed Speculative Content**

Deleted sections on:
- Regime shift probability (83%) - not calculated in actual analysis
- Forestry department response metrics - data not available/analyzed
- Lightning elevation gradients - detailed breakdown not in results
- Absolute fire count predictions - models output relative probability only

## File Structure

```
WRITEUP/
├── main.tex              # Original version (preserved)
├── main_updated.tex      # Updated version with actual results
├── diagram.html          # Workflow diagram (TikZ code)
└── UPDATE_SUMMARY.md     # This file
```

## How to Use

### Compile the Updated Manuscript

```bash
cd WRITEUP
pdflatex main_updated.tex
bibtex main_updated
pdflatex main_updated.tex
pdflatex main_updated.tex
```

### Figure Paths

All figures use relative paths from WRITEUP directory:
```latex
\includegraphics[width=...]{../output/figures/filename.png}
```

Ensure figures exist before compiling:
```bash
ls -la ../output/figures/*.png
```

### Customization Needed

Before submission, update:

1. **Author information** (lines 30-35):
   - Replace "Author Name" with actual authors
   - Update affiliations
   - Add corresponding author email

2. **Acknowledgments** (line ~870):
   - Add funding sources
   - Specify individuals to thank

3. **Data availability** (line ~875):
   - Update GitHub URL if repository moved
   - Add DOI if archived (Zenodo, etc.)

4. **References**:
   - Expand bibliography with relevant Alpine fire literature
   - Add citations for PyMC, Bayesian methods
   - Include climate scenario references

## Key Differences from Original

| Aspect | Original main.tex | Updated main_updated.tex |
|--------|-------------------|--------------------------|
| **Focus** | Regime shift detection | Model comparison (T+P vs T+P+L) |
| **Main result** | 83% regime shift probability | +56% seasonal R² with lightning |
| **Figures** | Placeholders/boxes | Actual figure files referenced |
| **Methods** | Generic description | Accurate implementation details |
| **Results** | Hypothetical values | Actual computed metrics |
| **Discussion** | Broad speculation | Evidence-based interpretation |
| **Length** | ~690 lines | ~890 lines (more detailed methods/results) |

## Scientific Contributions Emphasized

1. **Bayesian uncertainty quantification** for wildfire prediction
2. **Lightning integration value** quantified through rigorous comparison
3. **Trade-off documentation** between temporal and spatial performance
4. **Operational decision framework** based on complementary model strengths
5. **Climate projections** under RCP scenarios

## Next Steps

1. Review updated manuscript for scientific accuracy
2. Add co-authors and update affiliations
3. Expand references section
4. Consider submission to:
   - *International Journal of Wildland Fire*
   - *Environmental Modelling & Software*
   - *Natural Hazards and Earth System Sciences*
   - *Agricultural and Forest Meteorology*

5. Prepare supplementary materials:
   - High-resolution figures
   - Feature importance tables (extended)
   - Sensitivity analyses
   - Code repository documentation

## Notes

- All results are based on actual computed values from the model comparison
- Figures use scienceplots styling (publication-ready)
- Uncertainty is properly quantified throughout
- Trade-offs are honestly reported (not just positive results)
- Operational implications are grounded in actual performance metrics
