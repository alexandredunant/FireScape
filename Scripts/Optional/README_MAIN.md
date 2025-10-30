# Firescape Wildfire Risk Modeling Pipeline

Bayesian wildfire risk modeling for Bolzano Province (South Tyrol) with climate change projections.

## 📁 Directory Structure

```
Scripts/
├── 00_Documentation/           # All project documentation
│   ├── Landcover_Fix/          # Landcover encoding issue & solution
│   ├── Pipeline_Improvements/  # Project improvements summary
│   └── Workflow_Guides/        # How-to guides
├── 00_Utilities/               # Shared utility scripts
├── 01_Data_Preparation/        # Training data generation
├── 02_Model_Training/          # Bayesian model training
├── 03_Climate_Projections/     # Future risk projections
├── 04_Zone_Climate_Projections/# Fire brigade zone analysis
├── 05_Lightning_Comparison/    # Lightning data integration
├── Archive/                    # Historical versions
└── Bash/                       # Shell scripts
```

## 🚀 Quick Start

### 1. Fix Landcover Encoding (CRITICAL - Do First!)

Landcover data uses Corine codes (111-512) but was treated as continuous. Must fix before training:

```bash
cd 00_Utilities
python apply_landcover_fix.py
```

**Read first**: [00_Documentation/Landcover_Fix/02_SOLUTION.md](00_Documentation/Landcover_Fix/02_SOLUTION.md)

### 2. Train Model

```bash
cd 01_Data_Preparation
python create_raster_stacks.py     # Generate training data

cd ../02_Model_Training
python train_relative_probability_model.py  # Train Bayesian model
```

### 3. Run Climate Projections

```bash
cd ../03_Climate_Projections
python run_all_scenarios.py        # Process all RCP scenarios
python visualize_risk_evolution.py # Create visualizations
```

## 📖 Documentation

### Start Here
- **[00_Documentation/Pipeline_Improvements/COMPLETE_SOLUTION.md](00_Documentation/Pipeline_Improvements/COMPLETE_SOLUTION.md)** - Complete overview of all improvements

### Landcover Encoding Fix
- **[01_PROBLEM.md](00_Documentation/Landcover_Fix/01_PROBLEM.md)** - Why landcover encoding was wrong
- **[02_SOLUTION.md](00_Documentation/Landcover_Fix/02_SOLUTION.md)** - How to fix it (Corine codes)
- **[03_CHECKLIST.md](00_Documentation/Landcover_Fix/03_CHECKLIST.md)** - Step-by-step implementation

### Pipeline Documentation
- **[03_Climate_Projections/README.md](03_Climate_Projections/README.md)** - Climate projection pipeline
- **[00_Documentation/Workflow_Guides/STANDARDIZATION_GUIDE.md](00_Documentation/Workflow_Guides/STANDARDIZATION_GUIDE.md)** - Code standards
- **[00_Documentation/Workflow_Guides/TEMPORAL_VALIDATION_GUIDE.md](00_Documentation/Workflow_Guides/TEMPORAL_VALIDATION_GUIDE.md)** - Validation methods

## 🔑 Key Features

### Landcover Encoding
- **Corine Land Cover** classification (28 classes specific to Alpine region)
- **Ordinal fire risk encoding** (0-5 scale)
  - 0 = No risk (water, glaciers)
  - 5 = Very high (coniferous forest)
- Most common: Pastures (231) → Risk 2, Transitional woodland (324) → Risk 4

### Temporal Aggregation
- **60-day windows** (not single-day snapshots)
- Multiple aggregation periods: 1d, 3d, 5d, 10d, 15d, 30d, 60d
- Cumulative means and maximums for temperature and precipitation

### Climate Scenarios
- **RCP 4.5** (moderate emissions): 2030, 2050, 2070
- **RCP 8.5** (high emissions): 2030, 2050, 2070
- Automated iteration through all scenarios

### Visualization
- Temporal evolution plots (risk trends over time)
- Spatial risk maps (geographic distribution)
- Regional comparisons (different zones)
- Scenario comparison heatmaps

## 📂 Pipeline Components

### 01_Data_Preparation
- `create_raster_stacks.py` - Generate training data
- `CORINE_LANDCOVER_FIRE_RISK_MAPPING.py` - Landcover fire risk mapping

**Input**: Wildfire points + static/dynamic rasters
**Output**: `spacetime_stacks.nc` (training data)

### 02_Model_Training
- `train_relative_probability_model.py` - Train Bayesian model
- `train_Dask_PyMC_timeseries.py` - Alternative Dask implementation

**Input**: Training data (NetCDF)
**Output**: Trained model, scaler, validation plots

### 03_Climate_Projections
- `config_scenarios.py` - Scenario definitions
- `extract_projection_features.py` - Feature extraction
- `run_all_scenarios.py` - Orchestration
- `visualize_risk_evolution.py` - Visualization

**Input**: Climate projection NetCDFs (RCP 4.5/8.5)
**Output**: Risk predictions + visualizations

### 04_Zone_Climate_Projections
- `project_zone_fire_risk.py` - Fire brigade zone risk
- `analyze_warning_level_evolution.py` - Warning level analysis

**Input**: Fire brigade zone boundaries + projections
**Output**: Zone-level risk scores

## 🛠️ Utilities

### 00_Utilities/
- `apply_landcover_fix.py` - Automated landcover encoding fix for all scripts
- `shared_prediction_utils.py` - Shared prediction functions

## ⚠️ Important Notes

### Model Output Interpretation
- **Output**: Relative probability scores (0-1)
- **NOT**: Absolute fire counts or probabilities
- **Use for**: Ranking, comparison, trend analysis
- **Don't use for**: Predicting exact number of fires

### Landcover Data
Your dataset uses **Corine Land Cover Level 3** codes:
- 28 classes (3-digit codes: 111-512)
- **NOT** simple 1-10 classification
- See [CORINE mapping](01_Data_Preparation/CORINE_LANDCOVER_FIRE_RISK_MAPPING.py) for details

### Climate Data Requirements
- NetCDF format with `DATE`, `y`, `x` dimensions
- Daily temporal resolution
- Consistent CRS (EPSG:32632 or Lambert Azimuthal)
- File structure: `{var}_{scenario}_{year}{month}.nc`

## 🐛 Troubleshooting

### "KeyError: 'landcoverfull'" or "KeyError: 'landcover_fire_risk'"
**Solution**: Retrain model after applying landcover fix to ensure feature name consistency.

### Missing climate data files
**Solution**: Edit `03_Climate_Projections/config_scenarios.py` to match your data paths.

### Slow predictions
**Solution**:
- Increase spatial grid resolution (e.g., 2000m instead of 1000m)
- Reduce posterior samples in prediction functions
- Process fewer scenarios at once

## 📊 Example Workflow

```bash
# 1. Fix landcover encoding
cd /mnt/CEPH_PROJECTS/Firescape/Scripts/00_Utilities
python apply_landcover_fix.py

# 2. Generate training data
cd ../01_Data_Preparation
python create_raster_stacks.py

# 3. Train model
cd ../02_Model_Training
python train_relative_probability_model.py

# 4. Run climate projections
cd ../03_Climate_Projections
python run_all_scenarios.py
python visualize_risk_evolution.py

# 5. Analyze fire brigade zones
cd ../04_Zone_Climate_Projections
python project_zone_fire_risk.py
```

## 📈 Output Locations

```
Scripts/OUTPUT/
├── 01_Training_Data/
│   └── spacetime_stacks.nc                # Training data
├── 02_Model_RelativeProbability/
│   ├── trace_relative.nc                  # Trained model
│   ├── scaler_relative.joblib             # Feature scaler
│   └── *.png                              # Validation plots
└── 03_Climate_Projections/
    ├── rcp45_2030/
    │   ├── features_rcp45_2030.csv        # Extracted features
    │   └── predictions_rcp45_2030.csv     # Risk predictions
    ├── ... (other scenarios)
    └── Visualizations/
        ├── temporal_evolution_*.png       # Trend plots
        ├── spatial_risk_maps_*.png        # Geographic maps
        └── summary_statistics.csv         # Summary table
```

## 🤝 Contributing

When adding new features or documentation:
- Place documentation in `00_Documentation/` subdirectories
- Place reusable utilities in `00_Utilities/`
- Follow existing naming conventions
- Update this README if adding new pipeline components

## 📧 Support

For issues or questions:
1. Check relevant documentation in `00_Documentation/`
2. Review error messages and troubleshooting section
3. Verify file paths and data availability
4. Ensure all scripts use consistent feature names (after landcover fix)

---

**Last Updated**: 2025-10-28
**Version**: 1.0
**Pipeline Status**: ✅ Ready for production (after landcover fix applied)
