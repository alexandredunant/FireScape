# FireScape: Bayesian Wildfire Risk Modeling for the Alps

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyMC 5.x](https://img.shields.io/badge/PyMC-5.x-orange.svg)](https://www.pymc.io/)

> Hierarchical Bayesian framework for wildfire risk assessment in Alpine environments, with integrated lightning ignition modeling and uncertainty quantification.

## Overview

FireScape is a probabilistic wildfire risk modeling system developed for the Bolzano/Alto Adige region (Italian Alps). The project addresses three key questions:

1. **Can we quantify wildfire risk with full uncertainty?**
2. **Does lightning data improve predictions?**
3. **What does climate change mean for fire services?**

## Repository Structure

```
FireScape/
├── Data/                    # Data descriptions (actual files not in repo - see Data/README.md)
│   └── README.md           # Detailed data documentation
├── Scripts/                # Analysis pipeline
│   ├── 01_Data_Preparation/
│   ├── 02_Model_Training/                    # Baseline model (T+P, 1999-2024)
│   ├── 04_Zone_Climate_Projections/
│   ├── 05_Lightning_Comparison/              # Lightning analysis
│   │   ├── 01_Data_Preparation/
│   │   ├── 02_Model_Training/                # Lightning model (T+P+L, 2012-2024)
│   │   └── 03_Model_Comparison/              # Publication figures
│   └── WRITEUP/                              # Manuscript
├── Archive/                # Deprecated code (not in GitHub)
├── .gitignore
├── README.md               # This file
├── requirements.txt
└── LICENSE
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/alexandredunant/FireScape.git
cd FireScape

# Create environment
conda create -n firescape python=3.10
conda activate firescape

# Install dependencies
pip install -r requirements.txt
conda install -c conda-forge pymc  # Better performance than pip
```

### 2. Data Access

**Large data files (~18 GB) are not included in this repository.**

To reproduce the analysis:
1. See `Data/README.md` for data sources and descriptions
2. Contact the authors for processed datasets (available upon request)
3. Or obtain raw data from sources listed in `Data/README.md`

### 3. Run Analysis

```bash
# Train lightning model (requires data in Data/ML/)
cd Scripts/05_Lightning_Comparison/02_Model_Training
python train_relative_probability_model_with_lightning.py

# Generate comparison figures
cd ../03_Model_Comparison
python compare_models.py
```

**Runtime:** ~30-60 minutes for training, <1 minute for comparisons

---

## Publication Figures

**From `Scripts/05_Lightning_Comparison/03_Model_Comparison/Results/`:**

1. **fig1_monthly_temporal_fit.png** - Monthly fire count comparison
2. **fig2_seasonal_temporal_fit.png** - Seasonal patterns
3. **fig3_roc_curve.png** - ROC curves
4. **fig4_precision_recall_curve.png** - Precision-Recall curves
5. **fig5_feature_importance.png** - Attention weights
6. **fig6_tradeoff_summary.png** - Trade-off visualization

See `Scripts/05_Lightning_Comparison/03_Model_Comparison/Results/PUBLICATION_FIGURES_README.md` for figure captions and usage guidelines.

---



## Citation

If you use this code or methodology, please cite:

```bibtex
@article{dunant2025firescape,
  title={Lightning, Fire, and Climate: A Bayesian Analysis of Alpine Regime Shift in the Bolzano Mountains},
  author={Dunant, Alexandre and [Co-authors]},
  journal={[Journal Name]},
  year={2025},
  note={In preparation}
}
```

Preprint: [Link when available]

---

## Dependencies

**Core:**
- Python 3.10+
- PyMC 5.x (Bayesian inference)
- ArviZ 0.17+ (MCMC diagnostics)
- xarray 2023.x (NetCDF handling)

**Spatial:**
- GDAL 3.x
- rasterio 1.3+
- geopandas 0.14+

**Visualization:**
- matplotlib 3.8+
- seaborn 0.13+
- scienceplots 2.x

See `requirements.txt` for complete list.

---

## Computing Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 16 GB
- Storage: 50 GB
- Runtime: ~4-6 hours (full pipeline)

**Recommended:**
- CPU: 8+ cores
- RAM: 32 GB
- Storage: 100 GB
- Runtime: ~2-3 hours

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Data:** Remains subject to original provider licenses (see `Data/README.md`)

---

## Contact

- **GitHub Issues:** [https://github.com/alexandredunant/FireScape/issues](https://github.com/alexandredunant/FireScape/issues)
- **Email:** [Your email address]
- **Institution:** [Your institution]

---

## Acknowledgments

- **Data Providers:**
  - Bolzano Province Forest Fire Service (wildfire inventory)
  - EURAC CCT

- **Funding:**
  - EURAC CCT internal funding 2025

- **Collaborators:**
  - [List key collaborators]

---

## Related Publications

1. [M.Moreno 2024]
2. [Related paper 2 if applicable]

---

*Last updated: 2025-01-30*
