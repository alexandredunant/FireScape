# Firescape Project Organization Plan
**Date:** October 20, 2025
**Purpose:** Clean, well-documented structure for sharing and reproducibility

---

## Proposed Directory Structure

```
/mnt/CEPH_PROJECTS/Firescape/
│
├── README.md                           # Project overview and quick start
├── REQUIREMENTS.txt                    # Python dependencies
├── ENVIRONMENT.yml                     # Conda environment specification
├── LICENSE                             # License information
│
├── Data/                               # All input and processed data
│   ├── 00_QGIS/                       # GIS data
│   │   └── ADMIN/
│   │       └── BOLZANO_REGION_UTM32.gpkg
│   │
│   ├── STATIC_INPUT/                  # Static environmental rasters (50m)
│   │   ├── README.md                  # Data sources and descriptions
│   │   ├── nasadem.tif
│   │   ├── slope.tif
│   │   ├── aspect.tif
│   │   ├── northness.tif
│   │   ├── eastness.tif
│   │   ├── tri.tif
│   │   ├── treecoverdensity.tif
│   │   ├── landcoverfull.tif
│   │   ├── distroads.tif
│   │   ├── flammability.tif
│   │   ├── walking_time_to_bldg.tif
│   │   └── walking_time_to_elec_infra.tif
│   │
│   ├── WILDFIRE_INVENTORY/            # Historical fire data
│   │   ├── README.md
│   │   ├── REGISTRO_incendi_1999_2025.csv
│   │   └── wildfire_point_Bolzano_Period1999_2025.gpkg
│   │
│   ├── FIRE_BRIGADE/                  # Fire brigade zones
│   │   ├── README.md
│   │   ├── FireBrigade_ResponsibilityAreas_Bolzano_clipped.gpkg
│   │   └── processing_notes.md
│   │
│   └── OUTPUT/                        # All generated outputs
│       │
│       ├── 01_Training_Data/          # Prepared training dataset
│       │   ├── spacetime_stacks.nc    # Training data (5.8 GB)
│       │   ├── spacetime_dataset.parquet
│       │   └── temp_stacks/           # Individual stack files (can delete after training)
│       │
│       ├── 02_Model/                  # Trained Bayesian model
│       │   ├── trace.nc               # Posterior samples
│       │   ├── scaler.joblib          # Feature scaler
│       │   ├── baseline_stats.joblib  # Fire rate priors
│       │   └── model_metadata.json    # Model configuration
│       │
│       ├── 03_Model_Validation/       # Model diagnostics and validation
│       │   ├── validation_plots.png
│       │   ├── attention_weights.png
│       │   ├── feature_importance.png
│       │   ├── corner_plot.png
│       │   ├── roc_curves.png
│       │   └── cross_validation_results.csv
│       │
│       ├── 04_Climate_Projections/    # Future fire risk maps
│       │   ├── rcp85/
│       │   │   ├── pctl25/            # 25th percentile (conservative)
│       │   │   │   ├── fire_risk_20200201.tif
│       │   │   │   ├── fire_risk_20200204.tif
│       │   │   │   └── ...
│       │   │   ├── pctl50/            # 50th percentile (median)
│       │   │   ├── pctl75/            # 75th percentile
│       │   │   └── pctl99/            # 99th percentile (extreme)
│       │   └── summary_statistics/
│       │       ├── projection_summary_pctl25.csv
│       │       ├── projection_summary_pctl50.csv
│       │       ├── projection_summary_pctl75.csv
│       │       └── projection_summary_pctl99.csv
│       │
│       ├── 05_Fire_Brigade_Analysis/  # Operational planning outputs
│       │   ├── current_risk_by_zone_2020.png
│       │   ├── projected_risk_2050.png
│       │   ├── projected_risk_2080.png
│       │   ├── risk_increase_2020_to_2050.png
│       │   ├── risk_increase_2020_to_2080.png
│       │   ├── top_20_zones_risk_increase.csv
│       │   ├── zone_statistics_all_scenarios.csv
│       │   └── summary_report.pdf
│       │
│       ├── 06_Figures/                # Publication-ready figures
│       │   ├── figure1_study_area.png
│       │   ├── figure2_model_validation.png
│       │   ├── figure3_climate_projections.png
│       │   ├── figure4_uncertainty_analysis.png
│       │   └── figure5_fire_brigade_zones.png
│       │
│       └── 07_Historical_Analysis/    # Optional retrospective analysis
│           ├── lookback_2022_monthly/
│           └── historical_validation/
│
├── Scripts/                           # Analysis pipeline
│   ├── README.md                      # Pipeline overview
│   │
│   ├── 01_Data_Preparation/          # Data preprocessing
│   │   ├── 01_create_raster_stacks.py
│   │   └── clip_fire_brigade_to_bolzano.py
│   │
│   ├── 02_Model_Training/            # Bayesian model
│   │   ├── 04_Bayesian_pyMCLogisticRegression_Linear_Attention_commented.py
│   │   └── test_prior_validation.py
│   │
│   ├── 03_Climate_Projections/       # Future scenarios
│   │   ├── 05_Bayesian_Climate_Projection_MultiQuantile_Seasonal.py  # MAIN
│   │   ├── 05_Bayesian_Climate_Projection_CLEAN.py                   # Alternative
│   │   └── 05_Bayesian_Lookback_2022_GIF.py                          # Historical
│   │
│   ├── 04_Fire_Brigade_Analysis/     # Operational planning
│   │   └── 07_Fire_Brigade_Zone_Analysis.py
│   │
│   └── 05_Utilities/                 # Helper scripts
│       ├── monitor_progress.py
│       └── estimate_climate_projection_time.py
│
├── Documentation/                    # Project documentation
│   ├── 01_Data_Sources.md
│   ├── 02_Methodology.md
│   ├── 03_Model_Description.md
│   ├── 04_Configuration_Guide.md
│   ├── 05_Results_Interpretation.md
│   └── PATH_UPDATES_SUMMARY.md
│
└── Archive/                          # Old versions and experiments
    ├── deprecated_scripts/
    └── old_configurations/
```

---

## Files to Create/Update

### 1. Main README.md

```markdown
# Firescape: Bayesian Fire Risk Modeling for Alto Adige/South Tyrol

Climate change-driven fire risk projections for the Bolzano/Bozen province using
Bayesian logistic regression with attention mechanisms.

## Overview
- **Study Area:** Bolzano Province, Italy (7,397 km²)
- **Temporal Coverage:** 2020-2100 (RCP 8.5)
- **Spatial Resolution:** 100m
- **Uncertainty:** Ensemble quantiles (25th, 50th, 75th, 99th percentile)

## Quick Start
1. Install dependencies: `conda env create -f ENVIRONMENT.yml`
2. Activate environment: `conda activate firescape`
3. Train model: `python Scripts/02_Model_Training/04_Bayesian_*.py`
4. Generate projections: `python Scripts/03_Climate_Projections/05_Bayesian_*.py`

## Citation
[To be added]

## License
[To be added]
```

### 2. Data/STATIC_INPUT/README.md

```markdown
# Static Environmental Rasters

## Source Data
All rasters are 50m resolution, UTM32N projection, clipped to Bolzano Province.

### Topography
- **nasadem.tif**: NASA DEM elevation (source: NASA)
- **slope.tif**: Terrain slope in degrees (derived from nasadem)
- **aspect.tif**: Terrain aspect in degrees (derived from nasadem)
- **northness.tif**: North-facing component (cos(aspect))
- **eastness.tif**: East-facing component (sin(aspect))
- **tri.tif**: Terrain Ruggedness Index (derived from nasadem)

### Vegetation
- **treecoverdensity.tif**: Forest cover percentage (source: Copernicus)
- **landcoverfull.tif**: Land cover classification (source: Copernicus)
- **flammability.tif**: Vegetation flammability index (custom)

### Infrastructure
- **distroads.tif**: Distance to nearest road in meters (source: OSM)
- **walking_time_to_bldg.tif**: Walking time to buildings (minutes)
- **walking_time_to_elec_infra.tif**: Walking time to electrical infrastructure

## Processing
Rasters generated from original sources, resampled to 50m, and clipped to AOI.

## Last Updated
October 2025
```

### 3. Scripts/README.md

```markdown
# Firescape Analysis Pipeline

## Execution Order

### Phase 1: Data Preparation
1. `01_Data_Preparation/01_create_raster_stacks.py`
   - Creates training dataset from fire observations
   - Runtime: ~30 minutes
   - Output: Data/OUTPUT/spacetime_stacks.nc

### Phase 2: Model Training
2. `02_Model_Training/04_Bayesian_pyMCLogisticRegression_Linear_Attention_commented.py`
   - Trains Bayesian attention model
   - Runtime: 60-90 minutes
   - Output: Uncertainty_Attention/model_plots_bayesian_linear/

### Phase 3: Climate Projections
3. `03_Climate_Projections/05_Bayesian_Climate_Projection_MultiQuantile_Seasonal.py`
   - Generates future fire risk maps
   - Runtime: ~5.7 days (684 dates × 4 quantiles)
   - Output: climate_projections_rcp85_multiquantile/

### Phase 4: Operational Analysis
4. `04_Fire_Brigade_Analysis/07_Fire_Brigade_Zone_Analysis.py`
   - Analyzes fire brigade zone statistics
   - Runtime: 20-30 minutes
   - Output: Uncertainty_Attention/fire_brigade_analysis/

## Configuration

All scripts use consistent path variables defined at the top.
Climate data location: `/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/`

## Requirements

See REQUIREMENTS.txt for Python packages.
Minimum RAM: 10 GB (30 GB recommended)
```

---

## Cleanup Actions

### Files to Move to Archive/

```bash
# Old/deprecated scripts
Scripts/05_Bayesian_Climate_Projection.py  # → Archive/
Scripts/05_Bayesian_Climate_Projection_OPTIMIZED.py  # → Archive/
Scripts/*LSTM* Scripts/*Transformer*  # → Archive/old_experiments/

# Old documentation
Scripts/EXECUTION_PLAN.md  # → Documentation/
Scripts/RAM_OPTIMIZATION_GUIDE.md  # → Documentation/
Scripts/CLIMATE_PROJECTION_RUNTIME_REPORT.md  # → Documentation/
```

### Files to Rename for Clarity

```bash
# Current naming is confusing with numbers 04, 05, 07
# Suggest organizing by phase instead:

01_Data_Preparation/
  - 01_create_raster_stacks.py
  - 01b_clip_fire_brigade.py

02_Model_Training/
  - 02_train_bayesian_model.py (renamed from 04_*)
  - 02_validate_priors.py (renamed from test_prior_validation.py)

03_Climate_Projections/
  - 03_multiquantile_seasonal.py (renamed from 05_*)
  - 03_single_quantile.py (alternative version)
  - 03_historical_2022_gif.py (lookback analysis)

04_Fire_Brigade_Analysis/
  - 04_zone_analysis.py (renamed from 07_*)
```

### Documentation to Add

```markdown
1. Documentation/01_Data_Sources.md
   - Where each dataset came from
   - Processing steps applied
   - Licensing information

2. Documentation/02_Methodology.md
   - Bayesian attention mechanism explanation
   - Why these features matter for fire risk
   - Cross-validation approach

3. Documentation/03_Model_Description.md
   - Model architecture
   - Prior selection rationale
   - Validation metrics

4. Documentation/04_Configuration_Guide.md
   - How to adjust resolution
   - How to change temporal sampling
   - How to add new quantiles

5. Documentation/05_Results_Interpretation.md
   - How to read fire risk maps
   - Understanding uncertainty bounds
   - Fire brigade zone statistics
```

---

## Annotation Standards

### Required Header for Each Script

```python
#!/usr/bin/env python
"""
[Script Name]
===============================================================================

PURPOSE:
[Clear 1-2 sentence description of what this script does]

INPUTS:
- [List of input files/data]

OUTPUTS:
- [List of output files/results]

CONFIGURATION:
- [Key parameters that users might want to adjust]

RUNTIME:
[Estimated execution time]

AUTHOR: [Name]
DATE: [Date]
LAST MODIFIED: [Date]

NOTES:
[Any important caveats or dependencies]
"""
```

### Inline Comments Standard

```python
# BAD: x = data[::2]
# GOOD:
# Downsample to 100m resolution (every 2nd pixel)
x = data[::2]

# BAD: complicated logic with no explanation
# GOOD:
# Extract climate features for 60-day lookback window
# This captures the cumulative fire weather conditions
# that influence ignition probability
```

---

## Implementation Steps

### Step 1: Create New Structure (Don't delete yet)
```bash
# Script organization
mkdir -p Scripts/{01_Data_Preparation,02_Model_Training,03_Climate_Projections,04_Fire_Brigade_Analysis,05_Utilities}

# Output organization within Data/OUTPUT/
mkdir -p Data/OUTPUT/{01_Training_Data,02_Model,03_Model_Validation,04_Climate_Projections/rcp85/{pctl25,pctl50,pctl75,pctl99},04_Climate_Projections/summary_statistics,05_Fire_Brigade_Analysis,06_Figures,07_Historical_Analysis}

# Documentation and archive
mkdir -p Documentation
mkdir -p Archive/{deprecated_scripts,old_configurations}
```

### Step 2: Move Existing Outputs to New Structure
```bash
# Move training data
mv Data/OUTPUT/spacetime_stacks.nc Data/OUTPUT/01_Training_Data/
mv Data/OUTPUT/spacetime_dataset.parquet Data/OUTPUT/01_Training_Data/
mv Data/OUTPUT/temp_stacks Data/OUTPUT/01_Training_Data/

# Move model outputs (when they exist)
mv Scripts/Uncertainty_Attention/model_plots_bayesian_linear/* Data/OUTPUT/02_Model/ 2>/dev/null || true

# Split model outputs into model and validation
mv Data/OUTPUT/02_Model/*.png Data/OUTPUT/03_Model_Validation/ 2>/dev/null || true
mv Data/OUTPUT/02_Model/*.csv Data/OUTPUT/03_Model_Validation/ 2>/dev/null || true
```

### Step 3: Update Script Output Paths
All scripts need to point to new output locations:
- Training script → Data/OUTPUT/02_Model/
- Validation plots → Data/OUTPUT/03_Model_Validation/
- Climate projections → Data/OUTPUT/04_Climate_Projections/rcp85/{quantile}/
- Fire brigade → Data/OUTPUT/05_Fire_Brigade_Analysis/
```

### Step 3: Create Documentation
```bash
# Create all README files
# Add headers to all scripts
# Create main project README
```

### Step 4: Move Old Files to Archive
```bash
# After verifying new structure works
mv Scripts/05_Bayesian_Climate_Projection.py Archive/deprecated_scripts/
```

### Step 5: Update All Import Paths
```bash
# If scripts reference each other, update paths
```

---

## Benefits of This Organization

✓ **Clear workflow**: Numbered phases show execution order
✓ **Easy to share**: Well-documented, self-explanatory structure
✓ **Reproducible**: Everything needed is in one place
✓ **Maintainable**: Old versions archived, not mixed with current
✓ **Publishable**: Ready for data/code repository (Zenodo, GitHub)

---

## Next Steps

1. Review this structure - does it make sense?
2. Create main README.md files
3. Add headers to all active scripts
4. Test that everything still runs
5. Move deprecated files to Archive/

Would you like me to proceed with implementing this structure?
