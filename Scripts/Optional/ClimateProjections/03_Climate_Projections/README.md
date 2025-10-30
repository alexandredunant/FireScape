# Climate Projection Analysis for Wildfire Risk

This directory contains scripts for projecting wildfire risk under different climate scenarios (RCP 4.5 and RCP 8.5).

## Overview

The pipeline addresses three key improvements to the original workflow:

1. **Proper temporal aggregation**: Uses 60-day windows (not single-day snapshots) for mid-month projections
2. **Categorical landcover handling**: Provides ordinal encoding based on fire risk
3. **Multi-scenario iteration**: Automated processing of all RCP scenarios
4. **Comprehensive visualization**: Spatial and temporal evolution of wildfire risk

## Files

### Configuration
- **[config_scenarios.py](config_scenarios.py)**: Defines all climate scenarios (RCP 4.5, RCP 8.5) and regional subsets

### Core Pipeline
- **[extract_projection_features.py](extract_projection_features.py)**: Extracts features for future climate scenarios with proper temporal aggregation
- **[run_all_scenarios.py](run_all_scenarios.py)**: Orchestrates the complete pipeline across all scenarios
- **[visualize_risk_evolution.py](visualize_risk_evolution.py)**: Creates spatial and temporal visualizations

### Utilities
- **[../02_Model_Training/fix_landcover_encoding.py](../02_Model_Training/fix_landcover_encoding.py)**: Fixes categorical landcover encoding issue

## Climate Scenarios

### Historical Baseline
- **Period**: 1999-2025
- **Purpose**: Training data (already processed)

### RCP 4.5 (Moderate Emissions)
- **rcp45_2030**: 2021-2040 (near-term)
- **rcp45_2050**: 2041-2060 (mid-century)
- **rcp45_2070**: 2061-2080 (late-century)

### RCP 8.5 (High Emissions)
- **rcp85_2030**: 2021-2040 (near-term)
- **rcp85_2050**: 2041-2060 (mid-century)
- **rcp85_2070**: 2061-2080 (late-century)

## Regional Subsets

The analysis can be run for specific regions within Bolzano Province:
- **Full Province**: Entire study area
- **Alta Val Venosta**: Upper Venosta Valley (high elevation forests)
- **Val Badia**: Badia Valley (Dolomites region)
- **Oltradige**: Lower Adige Valley (wine-growing region)
- **Alta Pusteria**: Upper Puster Valley (Alpine forests)

## Usage

### 1. Configure Scenarios

Edit [config_scenarios.py](config_scenarios.py) to adjust:
- Scenario definitions (RCP, time periods)
- Regional boundaries
- Temporal aggregation windows
- Output directories

```python
# Example: Add a new scenario
SCENARIOS.append(
    ClimateScenario(
        name="rcp45_2090",
        rcp="rcp45",
        period=(2081, 2100),
        description="RCP 4.5: End of century",
        temp_dir="/path/to/temperature/data",
        precip_dir="/path/to/precipitation/data"
    )
)
```

### 2. Run Feature Extraction (Optional)

Extract features for a specific scenario:

```bash
python extract_projection_features.py
```

This will:
- Create a 1km spatial grid across Bolzano Province
- Extract features for mid-month dates (fire season: Jun-Sep)
- Compute 60-day temporal aggregations for each date
- Save features to CSV

**Note**: This is automatically called by `run_all_scenarios.py`, so manual execution is optional.

### 3. Run Complete Pipeline

Process all scenarios and generate predictions:

```bash
python run_all_scenarios.py
```

This will:
1. Load the trained Bayesian model
2. For each scenario:
   - Extract features (if not already done)
   - Generate predictions with uncertainty
   - Save results to CSV
3. Print summary statistics

**Output**: For each scenario, creates:
- `features_<scenario>.csv`: Extracted features
- `predictions_<scenario>.csv`: Risk predictions with uncertainty

### 4. Create Visualizations

Generate spatial and temporal plots:

```bash
python visualize_risk_evolution.py
```

This creates:
- **Temporal evolution plot**: Risk trends over time across all scenarios
- **Spatial risk maps**: Geographic distribution of risk
- **Regional comparison**: Risk evolution by region
- **Scenario comparison heatmap**: Monthly risk by scenario
- **Summary statistics table**: Key metrics for each scenario

**Output directory**: `OUTPUT/03_Climate_Projections/Visualizations/`

## Key Improvements

### 1. Temporal Aggregation (Mid-Month Issue)

**Problem**: Original approach used single-day snapshots (day 15) for projections, missing the temporal context that the model was trained on.

**Solution**:
- For each mid-month date (e.g., 2050-07-15), load 60 days of climate data (back to 2050-05-16)
- Compute temporal features: cumulative means and maximums for 1d, 3d, 5d, 10d, 15d, 30d, 60d windows
- This matches the training data structure

**Implementation**: See `load_dynamic_data_for_period()` in [extract_projection_features.py](extract_projection_features.py:115-163)

```python
# Load 60-day window for mid-month date
center_date = datetime(2050, 7, 15)
dynamic_data = load_dynamic_data_for_period(scenario, center_date, n_days_back=60)

# Compute temporal aggregations
for day_window in [1, 3, 5, 10, 15, 30, 60]:
    cumulative_mean = ...  # Mean over window
    cumulative_max = ...   # Maximum over window
```

### 2. Landcover Categorical Encoding

**Problem**: `landcoverfull` is a categorical variable (land cover classes) but was treated as continuous numeric.

**Solution**: Use **ordinal encoding** based on fire risk:

| Land Cover Class | Fire Risk (0-5) | Reasoning |
|-----------------|-----------------|-----------|
| Water, Snow/Ice | 0 | No fire risk |
| Urban, Bare rock | 1 | Low risk |
| Agriculture | 2 | Moderate (seasonal) |
| Grassland, Broadleaf | 3 | Moderate-high |
| Shrubland, Mixed forest | 4 | High |
| Coniferous forest | 5 | Very high (resinous) |

**Implementation**: See [fix_landcover_encoding.py](../02_Model_Training/fix_landcover_encoding.py)

**To apply**:
1. Update feature extraction in training script
2. Replace `landcoverfull` with `landcover_fire_risk`
3. Retrain model

### 3. Scenario Iteration

**Problem**: Manual processing of each scenario was error-prone and time-consuming.

**Solution**:
- Centralized scenario configuration in `config_scenarios.py`
- Automated iteration through all scenarios in `run_all_scenarios.py`
- Consistent naming and output structure

**Usage**:
```python
# Process all scenarios
for scenario in SCENARIOS:
    if scenario.name != 'historical':
        process_scenario(scenario, ...)
```

### 4. Spatial and Temporal Visualization

**Problem**: Original workflow lacked visualization of risk evolution across space and time.

**Solution**: Comprehensive visualization module with:

#### Temporal Evolution Plot
Shows how wildfire risk changes over time for different scenarios:
- Line plots with confidence intervals
- Comparison of RCP 4.5 vs RCP 8.5
- High-risk day frequency

#### Spatial Risk Maps
Geographic distribution of risk at specific time points:
- Scatter plots colored by risk
- Side-by-side comparison of different months
- Regional focus options

#### Regional Comparison
Risk evolution for different geographic regions:
- Time series for each region
- Spatial variability (error bands)
- Seasonal patterns

#### Scenario Comparison Heatmap
Matrix view of risk by scenario and month:
- Easy comparison across scenarios
- Seasonal patterns visible
- Quantitative values displayed

## Output Structure

```
OUTPUT/03_Climate_Projections/
├── rcp45_2030/
│   ├── features_rcp45_2030.csv
│   └── predictions_rcp45_2030.csv
├── rcp45_2050/
│   ├── features_rcp45_2050.csv
│   └── predictions_rcp45_2050.csv
├── ...
├── rcp85_2070/
│   ├── features_rcp85_2070.csv
│   └── predictions_rcp85_2070.csv
└── Visualizations/
    ├── temporal_evolution_all_scenarios.png
    ├── spatial_risk_maps_rcp85_2070.png
    ├── regional_comparison_rcp85_2050.png
    ├── scenario_comparison_heatmap.png
    └── summary_statistics.csv
```

## Data Requirements

### Climate Projection Data

Expected directory structure:

```
/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/
├── tas/  (temperature)
│   ├── rcp45/
│   │   ├── 2021/
│   │   │   ├── tas_rcp45_202101.nc
│   │   │   ├── tas_rcp45_202102.nc
│   │   │   └── ...
│   │   └── 2022/
│   │       └── ...
│   └── rcp85/
│       └── ...
└── pr/  (precipitation)
    ├── rcp45/
    │   └── ...
    └── rcp85/
        └── ...
```

**NetCDF format requirements**:
- Variable name: Any (auto-detected as first data variable)
- Dimensions: `DATE`, `y`, `x`
- Coordinate system: Same as training data (EPSG:32632)
- Temporal resolution: Daily

### Static Rasters

Required static raster files in `/mnt/CEPH_PROJECTS/Firescape/Data/STATIC_INPUT/`:
- `nasadem.tif` (elevation)
- `tri.tif` (terrain ruggedness)
- `slope.tif`, `aspect.tif`
- `northness.tif`, `eastness.tif`
- `treecoverdensity.tif`
- `landcoverfull.tif` (categorical - needs encoding fix)
- `flammability.tif`
- `distroads.tif`
- `walking_time_to_bldg.tif`
- `walking_time_to_elec_infra.tif`

## Interpretation Guidelines

### Risk Scores
- **Output**: Relative probability scores (0-1 range)
- **Meaning**: Higher scores = higher relative risk compared to baseline
- **NOT**: Absolute probability of fire occurrence

### Use Cases
✅ **Appropriate uses**:
- Ranking days/locations by relative risk
- Comparing scenarios (e.g., RCP 4.5 vs RCP 8.5)
- Identifying high-risk periods and locations
- Trend analysis over time

❌ **Inappropriate uses**:
- Predicting exact number of fires
- Converting to absolute fire probability
- Decision-making without considering uncertainty

### Uncertainty
- **std_risk**: Standard deviation across posterior samples
- **Higher values**: Model is less confident
- **Use**: Filter high-uncertainty predictions or report confidence intervals

## Troubleshooting

### Missing Climate Data
If climate projection files are missing:
1. Check file paths in [config_scenarios.py](config_scenarios.py)
2. Verify NetCDF file naming convention matches pattern
3. Ensure DATE coordinate is properly formatted

### Memory Issues
If running out of memory:
1. Reduce spatial grid resolution (e.g., 2000m instead of 1000m)
2. Process fewer scenarios at once
3. Reduce number of posterior samples (default: 300)

### Slow Predictions
To speed up prediction generation:
1. Reduce posterior samples in `generate_predictions()` (line 450 in [run_all_scenarios.py](run_all_scenarios.py:450))
2. Use fewer spatial points (increase grid resolution)
3. Process only fire season months (default)

## Next Steps

### 1. Retrain Model with Fixed Landcover Encoding
```bash
# First, update training script with ordinal encoding
cd ../02_Model_Training
python fix_landcover_encoding.py  # See example usage

# Then retrain model
python train_relative_probability_model.py
```

### 2. Run Climate Projections
```bash
cd ../03_Climate_Projections
python run_all_scenarios.py
```

### 3. Generate Visualizations
```bash
python visualize_risk_evolution.py
```

### 4. Regional Analysis
Edit [visualize_risk_evolution.py](visualize_risk_evolution.py) to focus on specific regions:
```python
# Example: Focus on Alta Val Venosta
from config_scenarios import REGIONS

region_bounds_utm = transform_bounds_to_utm(REGIONS['alta_val_venosta']['bounds'])

plot_regional_comparison(
    scenario_name='rcp85_2050',
    region_bounds={'Alta Val Venosta': region_bounds_utm},
    output_path='regional_alta_val_venosta.png'
)
```

## References

- Trained model: `OUTPUT/02_Model_RelativeProbability/`
- Training data: `OUTPUT/01_Training_Data/spacetime_stacks.nc`
- Model documentation: See [train_relative_probability_model.py](../02_Model_Training/train_relative_probability_model.py) header

## Contact

For questions or issues, refer to the main project documentation or contact the project maintainer.
