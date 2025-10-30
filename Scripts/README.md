# Scripts Directory

Analysis pipeline for the FireScape wildfire risk modeling system.

## Quick Start

```bash
# Train lightning model (2012-2024, T+P+L)
cd 05_Lightning_Comparison/02_Model_Training
python train_relative_probability_model_with_lightning.py

# Generate comparison figures
cd ../03_Model_Comparison
python compare_models.py
```

## Directory Structure

```
Scripts/
├── 01_Data_Preparation/              # Feature extraction
├── 02_Model_Training/                # Baseline model (T+P, 1999-2024)
├── 04_Zone_Climate_Projections/      # Future projections
├── 05_Lightning_Comparison/          # Lightning analysis (NEW!)
│   ├── 01_Data_Preparation/          # Filter to 2012+
│   ├── 02_Model_Training/            # Lightning model (T+P+L)
│   └── 03_Model_Comparison/          # Publication figures
├── Bash/                             # Shell scripts
├── Optional/                         # Experimental
├── WRITEUP/                          # Manuscript
└── OUTPUT/                           # Model results
```

## Key Results

### Lightning Model Performance (2012-2024 comparison):

**✅ Temporal Fit - DRAMATICALLY IMPROVED:**
- Monthly R²: 0.556 → 0.767 **(+38.0%)**
- Seasonal R²: 0.579 → 0.814 **(+40.7%)**

**⚠️ Discrimination - Declined:**
- ROC-AUC: 0.835 → 0.804 (-3.8%)
- PR-AUC: 0.654 → 0.516 (-21.1%)

**Interpretation:** Lightning model excels at predicting **WHEN** fires occur (temporal patterns) but is slightly worse at predicting **WHICH specific days** have fires (discrimination).

**Recommendation:**
- **Climate projections** → Use Lightning model
- **Operational alerts** → Use Baseline model
- **Research** → Ensemble both

## Publication Figures

All figures are publication-ready (600 DPI, no titles):

### From `05_Lightning_Comparison/03_Model_Comparison/Results/`:
1. **fig1_monthly_temporal_fit.png** - Monthly comparison
2. **fig2_seasonal_temporal_fit.png** - Seasonal comparison
3. **fig3_roc_curve.png** - ROC curves
4. **fig4_precision_recall_curve.png** - PR curves
5. **fig5_feature_importance.png** - Attention weights
6. **fig6_tradeoff_summary.png** - Trade-off visualization

## Model Specifications

### Baseline Model (`02_Model_Training/`)
- **Features:** Temperature + Precipitation (1999-2024)
- **Observations:** 3,035 (case-control sampled)
- **Architecture:** Bayesian hierarchical with attention
- **Performance:** ROC-AUC 0.835, Monthly R² 0.556 (on 2012-2024 test)

### Lightning Model (`05_Lightning_Comparison/02_Model_Training/`)
- **Features:** Temperature + Precipitation + Lightning (2012-2024)
- **Observations:** 1,556 (2012+ only, when lightning data available)
- **Architecture:** Same as baseline + 5 lightning attention groups
- **Performance:** ROC-AUC 0.804, Monthly R² 0.767

## Dependencies

```bash
conda create -n firescape python=3.10
conda activate firescape
pip install pymc arviz xarray pandas numpy scipy matplotlib seaborn scienceplots
```

See `requirements.txt` for complete list.

## Documentation

- **Data descriptions:** `Data/README.md`
- **Analysis details:** `05_Lightning_Comparison/03_Model_Comparison/Results/`
- **Figure guide:** `PUBLICATION_FIGURES_README.md`

## Citation

[Your paper citation once published]

## Questions?

GitHub Issues: https://github.com/alexandredunant/FireScape/issues
