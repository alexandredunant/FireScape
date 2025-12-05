# Temporal Generalisability Analysis

This directory contains scripts to analyze the temporal distribution of fire observations in the FireScape dataset. These analyses help assess whether your model can generalize across different time periods (years, seasons, months).

## Available Scripts

### 1. `check_temporal_generalisability.py` (NetCDF Version)
Analyzes the stacked netCDF dataset directly.

**Input:** `spacetime_stacks.nc` (the 5D netCDF stack)
**Output:**
- Comprehensive visualization: `temporal_generalisability_analysis.png`
- Statistics CSV: `temporal_generalisability_stats.csv`
- Console output with detailed statistics

### 2. `check_temporal_generalisability_parquet.py` (Parquet Version)
Lightweight version that reads from the parquet file instead of the full netCDF stack.

**Input:** `spacetime_dataset.parquet` (the observation points dataset)
**Output:** Same as above, but faster execution

### 3. `check_temporal_generalisability.ipynb` (Jupyter Notebook)
Interactive notebook version for exploratory analysis.

**Input:** `spacetime_stacks.nc`
**Output:** Interactive visualizations and statistics

## How to Run

### Option 1: Python Script (Parquet - Recommended for speed)

```bash
cd /home/user/FireScape/01_Data_Preparation
python check_temporal_generalisability_parquet.py
```

### Option 2: Python Script (NetCDF - Full dataset)

```bash
cd /home/user/FireScape/01_Data_Preparation
python check_temporal_generalisability.py
```

### Option 3: Jupyter Notebook (Interactive)

```bash
cd /home/user/FireScape/01_Data_Preparation
jupyter notebook check_temporal_generalisability.ipynb
```

## What the Analysis Provides

### 1. Distribution Visualizations
- **By Year**: Bar chart showing observation counts per year (fire vs non-fire)
- **By Month**: Distribution across all 12 months
- **By Season**: Distribution across Winter, Spring, Summer, Fall
- **Day of Year**: Histogram showing which days of the year have most fire events
- **Timeline**: Scatter plot showing when observations occurred
- **Heatmap**: Year × Month grid showing temporal coverage

### 2. Fire Rate Analysis
Shows the proportion of fire events (vs non-fire) across:
- Different years
- Different months
- Different seasons

Helps identify if certain periods are over/under-represented with fire events.

### 3. Balance Metrics

#### Coefficient of Variation (CV)
Measures how balanced the data is across different temporal periods:
- **CV < 0.3**: Well balanced ✓
- **CV 0.3-0.5**: Moderate imbalance ⚠
- **CV > 0.5**: Significant imbalance ✗

Lower values indicate better temporal generalisability.

#### Fire Rate Consistency
Standard deviation of fire rates across years:
- **Std < 0.1**: Consistent fire rate ✓
- **Std 0.1-0.2**: Moderate variation ⚠
- **Std > 0.2**: High variation ✗

#### Temporal Coverage
- Percentage of possible months covered by data
- Number of years represented
- Gaps in temporal coverage

### 4. Key Statistics Saved to CSV

The script saves detailed statistics including:
- Total observations
- Fire vs non-fire counts
- Date range
- Number of years/months covered
- Coefficient of variation for years, months, seasons
- Mean and standard deviation of fire rates

## Interpreting Results

### Good Temporal Generalisability

Your dataset has good temporal generalisability if:

1. **Balanced across years**: CV < 0.3
2. **Balanced across seasons**: CV < 0.3
3. **High temporal coverage**: >70% of months covered
4. **Consistent fire rates**: Std < 0.1
5. **Multi-year coverage**: Data spans ≥3 years

### Poor Temporal Generalisability

Warning signs of poor generalisability:

1. **Data clustered in specific years**: High year CV (>0.5)
2. **Strong seasonal bias**: Only summer months, or only one season
3. **Low temporal coverage**: <50% of months covered
4. **Inconsistent fire rates**: Fire rate varies widely between years
5. **Short time span**: Data from <2 years

### Implications for Model Training

- **Training/Validation Split**: Should maintain temporal distribution
- **Temporal Cross-Validation**: Consider using temporal holdout sets
- **Model Generalisability**: Model may not generalize well to under-represented periods
- **Data Collection**: Identify gaps that need more data

## Customization

To modify the analysis, edit the configuration section:

```python
# Define seasons for your region
SEASONS = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Fall': [9, 10, 11]
}

# Change input/output paths
OUTPUT_DIR = "/your/custom/path/"
NETCDF_PATH = os.path.join(OUTPUT_DIR, "spacetime_stacks.nc")
```

## Requirements

### Python Packages
- numpy
- pandas
- xarray (for netCDF version)
- geopandas (for parquet version)
- matplotlib
- seaborn
- rioxarray (for netCDF version)

### Install with:
```bash
pip install numpy pandas xarray geopandas matplotlib seaborn rioxarray
```

Or with conda:
```bash
conda install numpy pandas xarray geopandas matplotlib seaborn rioxarray
```

## Troubleshooting

### "ModuleNotFoundError"
Install the required packages (see Requirements above).

### "File not found"
Update the paths in the configuration section to point to your data files.

### "Dataset not accessible"
Ensure the remote filesystem is mounted, or copy the data to a local directory.

### Memory issues with large datasets
Use the parquet version instead of the netCDF version for faster, more memory-efficient analysis.

## Output Files

After running the analysis, you'll find:

```
/mnt/CEPH_PROJECTS/Firescape/output/01_Training_Data/
├── temporal_generalisability_analysis.png  # Comprehensive 9-panel visualization
└── temporal_generalisability_stats.csv     # Detailed statistics table
```

## Related Documentation

- For information about the netCDF stack structure, see `create_raster_stacks.py`
- For information about the observation points, see `create_spacetime_dataset.py`
- For model training implications, see `/home/user/FireScape/02_Model_Training/`
