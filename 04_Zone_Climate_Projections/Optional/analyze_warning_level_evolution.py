#!/usr/bin/env python
"""
Warning Level Evolution Under Climate Change

Analyzes how the distribution of fire warning levels evolves over time
under different climate scenarios for each fire brigade zone.

Purpose:
- Apply optimal thresholds from threshold optimization to climate projections
- Track changes in warning level distribution (Low → Extreme) over decades
- Identify zones experiencing greatest shifts toward higher warning levels
- Support long-term resource planning and climate adaptation strategies

Key Features:
- Uses warning thresholds from threshold optimization analysis
- Tracks 5 warning levels: Low, Moderate, High, Very High, Extreme
- Analyzes trends by zone, month, and climate quantile
- Calculates "warning level shift index" to quantify risk increases
- NO SCALING FACTORS - pure relative risk categorization

Output:
- CSV with warning level distributions by zone/year/month/quantile
- Trend plots showing shift toward higher warning levels over time
- Zone comparison maps showing differential climate impacts
- Summary statistics for resource planning
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import rioxarray
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xvec
import gc
import itertools
from tqdm import tqdm
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

print("="*80)
print("WARNING LEVEL EVOLUTION UNDER CLIMATE CHANGE")
print("="*80)
print()

# ===================================================================
# CONFIGURATION
# ===================================================================

# Paths
BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape")
AOI_PATH = BASE_DIR / "Data/00_QGIS/ADMIN/BOLZANO_REGION_UTM32.gpkg"
FIRE_BRIGADE_ZONES = BASE_DIR / "Data/06_Administrative_Boundaries/Processed/FireBrigade_ResponsibilityAreas_Bolzano_clipped.gpkg"
STATIC_RASTER_DIR = BASE_DIR / "Data/STATIC_INPUT"

# Climate data
TARGET_SCENARIO = "rcp85"
TEMP_BASE_DIR = Path("/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/tas") / TARGET_SCENARIO
PRECIP_BASE_DIR = Path("/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/pr") / TARGET_SCENARIO

# Model files
MODEL_DIR = BASE_DIR / "Scripts/OUTPUT/02_Model_RelativeProbability"
TRACE_PATH = MODEL_DIR / "trace_relative.nc"
SCALER_PATH = MODEL_DIR / "scaler_relative.joblib"
TEMPORAL_GROUPS_PATH = MODEL_DIR / "temporal_groups.joblib"
GROUP_NAMES_PATH = MODEL_DIR / "group_names.joblib"

# Warning level thresholds
THRESHOLD_FILE = BASE_DIR / "Scripts/03_Threshold_Optimization/Results/warning_levels.csv"

# Output
OUTPUT_DIR = BASE_DIR / f"Scripts/OUTPUT/04_Zone_Climate_Projections/{TARGET_SCENARIO}/WarningLevelEvolution"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Projection parameters
PROJECTION_YEARS = list(range(2020, 2090, 10))  # Every 10 years
# South Tyrol has year-round fire risk including winter dry periods
PROJECTION_MONTHS = [1, 2, 3, 4, 7, 8, 10, 11]  # All high-risk periods
CLIMATE_QUANTILES = ["pctl25", "pctl50", "pctl99"]
SPATIAL_RESOLUTION = 200  # Meters (200m for computational efficiency)

# Model parameters (must match training)
TIME_STEPS = 60
STATIC_VARS = [
    'tri', 'northness', 'slope', 'aspect', 'nasadem',
    'treecoverdensity', 'landcoverfull', 'distroads', 'eastness',
    'flammability', 'walking_time_to_bldg', 'walking_time_to_elec_infra'
]
DYNAMIC_VARS = ['T', 'P']
DAY_WINDOWS_TO_KEEP = [1, 3, 5, 10, 15, 30, 60]

month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 7: 'Jul', 8: 'Aug', 10: 'Oct', 11: 'Nov'}
season_mapping = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring',
                  7: 'Summer', 8: 'Summer', 10: 'Fall', 11: 'Fall'}

print(f"Configuration:")
print(f"  Scenario: {TARGET_SCENARIO}")
print(f"  Years: {PROJECTION_YEARS[0]}-{PROJECTION_YEARS[-1]} (every 10 years)")
print(f"  Months: {', '.join([month_names[m] for m in PROJECTION_MONTHS])}")
print(f"  Climate quantiles: {CLIMATE_QUANTILES}")
print(f"  Spatial resolution: {SPATIAL_RESOLUTION}m")
print(f"  Output: {OUTPUT_DIR}")
print()

# ===================================================================
# LOAD WARNING LEVEL THRESHOLDS
# ===================================================================

print("Loading warning level thresholds...")
warning_levels_df = pd.read_csv(THRESHOLD_FILE)
print("✓ Warning levels loaded:")
print()
for _, row in warning_levels_df.iterrows():
    print(f"  {row['level']:12s}: [{row['min_threshold']:.3f}, {row['max_threshold']:.3f}) - {row['action']}")
print()

# Create threshold array for categorization
warning_thresholds = warning_levels_df.sort_values('min_threshold')['min_threshold'].values
warning_labels = warning_levels_df.sort_values('min_threshold')['level'].values

def categorize_risk(prob_array):
    """Categorize probability values into warning levels"""
    categories = np.digitize(prob_array, bins=warning_thresholds[1:])  # Skip first (0.0)
    return warning_labels[categories]

# ===================================================================
# LOAD MODEL
# ===================================================================

print("Loading trained Bayesian model...")
trace = az.from_netcdf(TRACE_PATH)
scaler = joblib.load(SCALER_PATH)
temporal_groups = joblib.load(TEMPORAL_GROUPS_PATH)
group_names = joblib.load(GROUP_NAMES_PATH)
print("✓ Model loaded")
print(f"  Model type: Relative probability (Bayesian)")
print(f"  Output interpretation: Warning levels based on optimal thresholds")
print()

# ===================================================================
# LOAD FIRE BRIGADE ZONES
# ===================================================================

print("Loading fire brigade zones...")
gdf_zones = gpd.read_file(FIRE_BRIGADE_ZONES)
print(f"✓ Loaded {len(gdf_zones)} zones")

# Ensure unique zone identifiers
if 'ID' not in gdf_zones.columns:
    gdf_zones['zone_id'] = range(len(gdf_zones))
else:
    gdf_zones['zone_id'] = gdf_zones['ID']

# Get zone names if available
zone_name_col = None
for col in ['PLACE_IT', 'NAME', 'name', 'NAME_IT']:
    if col in gdf_zones.columns:
        zone_name_col = col
        gdf_zones['zone_name'] = gdf_zones[col]
        break

if zone_name_col is None:
    gdf_zones['zone_name'] = 'Zone_' + gdf_zones['zone_id'].astype(str)

print(f"  Zone ID column: zone_id")
print(f"  Zone name column: zone_name")
print()

# ===================================================================
# LOAD STATIC RASTER STACK
# ===================================================================

print("Loading static variables...")
static_stack = None
for var in STATIC_VARS:
    file_path = STATIC_RASTER_DIR / f"{var}.tif"
    if not file_path.exists():
        print(f"✗ Missing: {var}.tif")
        continue

    raster = rioxarray.open_rasterio(file_path, masked=True)
    raster = raster.squeeze(drop=True)  # Remove band dimension

    # Resample if needed
    if SPATIAL_RESOLUTION != 30:
        scale_factor = SPATIAL_RESOLUTION / 30
        new_width = int(raster.rio.width / scale_factor)
        new_height = int(raster.rio.height / scale_factor)
        raster = raster.rio.reproject(
            raster.rio.crs,
            shape=(new_height, new_width),
            resampling=5  # Average
        )

    if static_stack is None:
        static_stack = raster.to_dataset(name=var)
    else:
        static_stack[var] = raster

print(f"✓ Loaded {len(STATIC_VARS)} static variables")
print(f"  Resolution: {SPATIAL_RESOLUTION}m")
print(f"  Shape: {static_stack[STATIC_VARS[0]].shape}")
print()

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def load_climate_data(year, month, quantile, variable):
    """Load climate data for specific year/month/quantile"""
    if variable == 'T':
        base_dir = TEMP_BASE_DIR
    elif variable == 'P':
        base_dir = PRECIP_BASE_DIR
    else:
        raise ValueError(f"Unknown variable: {variable}")

    file_path = base_dir / f"{variable}_{quantile}_{year}_{month:02d}.tif"

    if not file_path.exists():
        return None

    raster = rioxarray.open_rasterio(file_path, masked=True)
    raster = raster.squeeze(drop=True)

    # Resample if needed
    if SPATIAL_RESOLUTION != 30:
        scale_factor = SPATIAL_RESOLUTION / 30
        new_width = int(raster.rio.width / scale_factor)
        new_height = int(raster.rio.height / scale_factor)
        raster = raster.rio.reproject(
            raster.rio.crs,
            shape=(new_height, new_width),
            resampling=5
        )

    # Match static stack extent
    raster = raster.rio.reproject_match(static_stack[STATIC_VARS[0]])

    return raster

def create_feature_stack(static_stack, climate_data_dict):
    """Create feature stack combining static and dynamic variables"""
    # Start with static variables
    feature_stack = static_stack.copy()

    # Add dynamic variables (replicate across time windows)
    for var in DYNAMIC_VARS:
        for window in DAY_WINDOWS_TO_KEEP:
            feature_name = f"{var}_{window:02d}"
            feature_stack[feature_name] = climate_data_dict[var]

    return feature_stack

def predict_risk(feature_stack, trace, scaler, temporal_groups, month):
    """Generate risk predictions with uncertainty"""
    # Extract feature arrays
    feature_names = []
    for var in STATIC_VARS:
        feature_names.append(var)
    for var in DYNAMIC_VARS:
        for window in DAY_WINDOWS_TO_KEEP:
            feature_names.append(f"{var}_{window:02d}")

    # Stack features
    feature_arrays = []
    for fname in feature_names:
        arr = feature_stack[fname].values
        feature_arrays.append(arr.flatten())

    X = np.column_stack(feature_arrays)

    # Remove NaN rows
    valid_mask = ~np.isnan(X).any(axis=1)
    X_valid = X[valid_mask]

    if len(X_valid) == 0:
        return None, None

    # Standardize
    X_scaled = scaler.transform(X_valid)

    # Get temporal group
    month_group = temporal_groups[month - 1]

    # Extract posterior samples
    intercepts = trace.posterior['intercept'].values.flatten()
    alphas = trace.posterior['alpha'].values  # shape: (chains*draws, n_groups, n_features)
    betas = trace.posterior['beta'].values    # shape: (chains*draws, n_groups, n_features)

    # Reshape
    n_samples = len(intercepts)
    n_features = X_scaled.shape[1]

    alphas_group = alphas[:, month_group, :]
    betas_group = betas[:, month_group, :]

    # Compute attention weights
    attention_weights = np.exp(alphas_group)  # shape: (n_samples, n_features)
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)

    # Compute predictions for each posterior sample
    predictions = []
    for i in range(n_samples):
        weighted_features = X_scaled * attention_weights[i]
        logit = intercepts[i] + np.dot(weighted_features, betas_group[i])
        prob = 1 / (1 + np.exp(-logit))
        predictions.append(prob)

    predictions = np.array(predictions)  # shape: (n_samples, n_pixels)

    # Compute mean and std
    mean_prob = predictions.mean(axis=0)
    std_prob = predictions.std(axis=0)

    # Reconstruct spatial arrays
    mean_prob_full = np.full(len(valid_mask), np.nan)
    std_prob_full = np.full(len(valid_mask), np.nan)
    mean_prob_full[valid_mask] = mean_prob
    std_prob_full[valid_mask] = std_prob

    original_shape = feature_stack[STATIC_VARS[0]].shape
    mean_prob_2d = mean_prob_full.reshape(original_shape)
    std_prob_2d = std_prob_full.reshape(original_shape)

    return mean_prob_2d, std_prob_2d

def compute_zone_statistics(prob_array, zone_geometry, transform, crs):
    """Compute warning level statistics for a zone"""
    # Create xarray DataArray for vectorized operations
    y_coords = np.arange(prob_array.shape[0]) * transform[4] + transform[5]
    x_coords = np.arange(prob_array.shape[1]) * transform[0] + transform[2]

    da = xr.DataArray(
        prob_array,
        dims=['y', 'x'],
        coords={'y': y_coords, 'x': x_coords}
    )
    da.rio.write_crs(crs, inplace=True)
    da.rio.write_transform(transform, inplace=True)

    # Extract zone values
    try:
        zone_gdf = gpd.GeoDataFrame([{'geometry': zone_geometry}], crs=crs)
        clipped = da.rio.clip(zone_gdf.geometry, all_touched=True)
        zone_values = clipped.values.flatten()
        zone_values = zone_values[~np.isnan(zone_values)]

        if len(zone_values) == 0:
            return None

        # Categorize into warning levels
        categories = categorize_risk(zone_values)

        # Count each level
        level_counts = {}
        for level in warning_labels:
            level_counts[level] = np.sum(categories == level)

        total_pixels = len(zone_values)
        level_proportions = {level: count / total_pixels for level, count in level_counts.items()}

        # Additional statistics
        stats = {
            'mean_prob': float(np.mean(zone_values)),
            'median_prob': float(np.median(zone_values)),
            'p90_prob': float(np.percentile(zone_values, 90)),
            'p95_prob': float(np.percentile(zone_values, 95)),
            'p99_prob': float(np.percentile(zone_values, 99)),
            'total_pixels': total_pixels
        }

        # Add level proportions
        for level, prop in level_proportions.items():
            stats[f'prop_{level}'] = prop

        return stats

    except Exception as e:
        print(f"    Warning: Could not process zone: {e}")
        return None

# ===================================================================
# MAIN ANALYSIS LOOP
# ===================================================================

print("="*80)
print("ANALYZING WARNING LEVEL EVOLUTION")
print("="*80)
print()

results = []

# Iterate through all combinations
total_iterations = len(PROJECTION_YEARS) * len(PROJECTION_MONTHS) * len(CLIMATE_QUANTILES) * len(gdf_zones)
pbar = tqdm(total=total_iterations, desc="Processing")

for year in PROJECTION_YEARS:
    for month in PROJECTION_MONTHS:
        for quantile in CLIMATE_QUANTILES:

            # Load climate data
            climate_data = {}
            all_loaded = True
            for var in DYNAMIC_VARS:
                data = load_climate_data(year, month, quantile, var)
                if data is None:
                    all_loaded = False
                    break
                climate_data[var] = data

            if not all_loaded:
                print(f"✗ Missing climate data: {year}-{month:02d} {quantile}")
                pbar.update(len(gdf_zones))
                continue

            # Create feature stack
            feature_stack = create_feature_stack(static_stack, climate_data)

            # Generate predictions
            mean_prob, std_prob = predict_risk(feature_stack, trace, scaler, temporal_groups, month)

            if mean_prob is None:
                print(f"✗ Prediction failed: {year}-{month:02d} {quantile}")
                pbar.update(len(gdf_zones))
                continue

            # Get transform and CRS from static stack
            transform = static_stack[STATIC_VARS[0]].rio.transform()
            crs = static_stack[STATIC_VARS[0]].rio.crs

            # Compute statistics for each zone
            for _, zone in gdf_zones.iterrows():
                zone_stats = compute_zone_statistics(
                    mean_prob,
                    zone.geometry,
                    transform,
                    crs
                )

                if zone_stats is not None:
                    result = {
                        'zone_id': zone['zone_id'],
                        'zone_name': zone['zone_name'],
                        'year': year,
                        'month': month,
                        'month_name': month_names[month],
                        'season': season_mapping[month],
                        'quantile': quantile,
                        **zone_stats
                    }
                    results.append(result)

                pbar.update(1)

            # Clean up
            del feature_stack, mean_prob, std_prob
            gc.collect()

pbar.close()

# ===================================================================
# SAVE RESULTS
# ===================================================================

print()
print("Saving results...")

if len(results) == 0:
    print("✗ No results to save")
    exit(1)

df_results = pd.DataFrame(results)

# Save detailed results
output_file = OUTPUT_DIR / "warning_level_evolution.csv"
df_results.to_csv(output_file, index=False)
print(f"✓ Detailed results saved: {output_file}")

# ===================================================================
# COMPUTE SUMMARY STATISTICS
# ===================================================================

print()
print("Computing summary statistics...")

# 1. Trend analysis: Change in high-risk proportion over time
trend_stats = []

for zone_id in df_results['zone_id'].unique():
    zone_data = df_results[df_results['zone_id'] == zone_id]
    zone_name = zone_data['zone_name'].iloc[0]

    for quantile in CLIMATE_QUANTILES:
        for month in PROJECTION_MONTHS:
            subset = zone_data[(zone_data['quantile'] == quantile) & (zone_data['month'] == month)]

            if len(subset) < 2:
                continue

            # Compute "high risk proportion" = Very High + Extreme
            subset = subset.sort_values('year')
            high_risk_prop = subset['prop_Very High'] + subset['prop_Extreme']
            years = subset['year'].values

            # Linear trend
            if len(years) > 1:
                slope, intercept, r_value, p_value, stderr = stats.linregress(years, high_risk_prop)

                # Project change from 2020 to 2080
                change_2020_2080 = slope * (2080 - 2020)

                trend_stats.append({
                    'zone_id': zone_id,
                    'zone_name': zone_name,
                    'month': month,
                    'month_name': month_names[month],
                    'season': season_mapping[month],
                    'quantile': quantile,
                    'trend_slope': slope,
                    'trend_r2': r_value**2,
                    'trend_pvalue': p_value,
                    'high_risk_2020': high_risk_prop.iloc[0] if len(high_risk_prop) > 0 else np.nan,
                    'high_risk_2080': high_risk_prop.iloc[-1] if len(high_risk_prop) > 0 else np.nan,
                    'change_2020_2080': change_2020_2080
                })

df_trends = pd.DataFrame(trend_stats)
trends_file = OUTPUT_DIR / "warning_level_trends.csv"
df_trends.to_csv(trends_file, index=False)
print(f"✓ Trend analysis saved: {trends_file}")

# 2. Zone comparison: Rank zones by climate vulnerability
zone_summary = []

for zone_id in df_results['zone_id'].unique():
    zone_data = df_results[df_results['zone_id'] == zone_id]
    zone_name = zone_data['zone_name'].iloc[0]

    # Focus on median climate (pctl50) and summer months (Jul, Aug)
    summer_median = zone_data[(zone_data['quantile'] == 'pctl50') &
                               (zone_data['month'].isin([7, 8]))]

    if len(summer_median) == 0:
        continue

    # Baseline (2020) vs future (2080)
    baseline = summer_median[summer_median['year'] == 2020]
    future = summer_median[summer_median['year'] == 2080]

    if len(baseline) == 0 or len(future) == 0:
        continue

    baseline_high_risk = (baseline['prop_Very High'] + baseline['prop_Extreme']).mean()
    future_high_risk = (future['prop_Very High'] + future['prop_Extreme']).mean()

    zone_summary.append({
        'zone_id': zone_id,
        'zone_name': zone_name,
        'baseline_high_risk_prop': baseline_high_risk,
        'future_high_risk_prop': future_high_risk,
        'absolute_increase': future_high_risk - baseline_high_risk,
        'relative_increase': (future_high_risk - baseline_high_risk) / (baseline_high_risk + 0.001) * 100
    })

df_zone_summary = pd.DataFrame(zone_summary)
df_zone_summary = df_zone_summary.sort_values('absolute_increase', ascending=False)
zone_summary_file = OUTPUT_DIR / "zone_climate_vulnerability.csv"
df_zone_summary.to_csv(zone_summary_file, index=False)
print(f"✓ Zone vulnerability summary saved: {zone_summary_file}")

# ===================================================================
# GENERATE VISUALIZATIONS
# ===================================================================

print()
print("Generating visualizations...")

# 1. Time series plot: Warning level evolution for top vulnerable zones
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Warning Level Evolution - Most Vulnerable Zones\n(Median Climate, Summer Months)',
             fontsize=16, fontweight='bold')

top_zones = df_zone_summary.nlargest(4, 'absolute_increase')['zone_id'].values

for idx, zone_id in enumerate(top_zones):
    ax = axes[idx // 2, idx % 2]

    zone_data = df_results[(df_results['zone_id'] == zone_id) &
                           (df_results['quantile'] == 'pctl50') &
                           (df_results['month'].isin([7, 8]))]

    zone_name = zone_data['zone_name'].iloc[0]

    # Aggregate by year
    yearly_data = zone_data.groupby('year').agg({
        'prop_Low': 'mean',
        'prop_Moderate': 'mean',
        'prop_High': 'mean',
        'prop_Very High': 'mean',
        'prop_Extreme': 'mean'
    }).reset_index()

    # Stacked area plot
    years = yearly_data['year'].values
    ax.fill_between(years, 0, yearly_data['prop_Low'], label='Low', color='#2ecc71', alpha=0.7)
    ax.fill_between(years, yearly_data['prop_Low'],
                    yearly_data['prop_Low'] + yearly_data['prop_Moderate'],
                    label='Moderate', color='#f39c12', alpha=0.7)
    ax.fill_between(years, yearly_data['prop_Low'] + yearly_data['prop_Moderate'],
                    yearly_data['prop_Low'] + yearly_data['prop_Moderate'] + yearly_data['prop_High'],
                    label='High', color='#e67e22', alpha=0.7)
    ax.fill_between(years, yearly_data['prop_Low'] + yearly_data['prop_Moderate'] + yearly_data['prop_High'],
                    yearly_data['prop_Low'] + yearly_data['prop_Moderate'] + yearly_data['prop_High'] + yearly_data['prop_Very High'],
                    label='Very High', color='#e74c3c', alpha=0.7)
    ax.fill_between(years, yearly_data['prop_Low'] + yearly_data['prop_Moderate'] + yearly_data['prop_High'] + yearly_data['prop_Very High'],
                    1.0, label='Extreme', color='#c0392b', alpha=0.7)

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Proportion', fontsize=11)
    ax.set_title(f'{zone_name}', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plot_file = OUTPUT_DIR / "warning_level_evolution_top_zones.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Evolution plot saved: {plot_file}")

# 2. Heatmap: Zone vulnerability by season
fig, ax = plt.subplots(figsize=(14, max(8, len(df_zone_summary) * 0.3)))

# Compute seasonal vulnerability
seasonal_vuln = []
for zone_id in df_zone_summary['zone_id'].values:
    zone_name = df_zone_summary[df_zone_summary['zone_id'] == zone_id]['zone_name'].iloc[0]
    row_data = {'zone_name': zone_name}

    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_data = df_results[(df_results['zone_id'] == zone_id) &
                                 (df_results['season'] == season) &
                                 (df_results['quantile'] == 'pctl50')]

        if len(season_data) == 0:
            row_data[season] = 0
            continue

        # Compare 2020 vs 2080
        baseline = season_data[season_data['year'] == 2020]
        future = season_data[season_data['year'] == 2080]

        if len(baseline) > 0 and len(future) > 0:
            baseline_risk = (baseline['prop_Very High'] + baseline['prop_Extreme']).mean()
            future_risk = (future['prop_Very High'] + future['prop_Extreme']).mean()
            row_data[season] = (future_risk - baseline_risk) * 100  # Percentage point increase
        else:
            row_data[season] = 0

    seasonal_vuln.append(row_data)

df_seasonal = pd.DataFrame(seasonal_vuln)
df_seasonal = df_seasonal.sort_values('Summer', ascending=False)

# Create heatmap
heatmap_data = df_seasonal[['Winter', 'Spring', 'Summer', 'Fall']].values
sns.heatmap(heatmap_data,
            xticklabels=['Winter', 'Spring', 'Summer', 'Fall'],
            yticklabels=df_seasonal['zone_name'].values,
            cmap='RdYlGn_r', center=0, vmin=-5, vmax=30,
            cbar_kws={'label': 'Change in High-Risk Proportion\n(percentage points, 2020→2080)'},
            ax=ax)

ax.set_title('Seasonal Climate Vulnerability by Fire Brigade Zone\n(Median Climate Scenario)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Season', fontsize=12, fontweight='bold')
ax.set_ylabel('Fire Brigade Zone', fontsize=12, fontweight='bold')

plt.tight_layout()
heatmap_file = OUTPUT_DIR / "zone_seasonal_vulnerability_heatmap.png"
plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Vulnerability heatmap saved: {heatmap_file}")

# 3. Bar chart: Top 10 most vulnerable zones
fig, ax = plt.subplots(figsize=(12, 8))

top10 = df_zone_summary.nlargest(10, 'absolute_increase')

colors = ['#c0392b' if x > 0.2 else '#e74c3c' if x > 0.1 else '#e67e22'
          for x in top10['absolute_increase']]

bars = ax.barh(range(len(top10)), top10['absolute_increase'] * 100, color=colors, alpha=0.8)

ax.set_yticks(range(len(top10)))
ax.set_yticklabels(top10['zone_name'])
ax.set_xlabel('Increase in High-Risk Proportion (percentage points)', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Most Climate-Vulnerable Fire Brigade Zones\n(Summer, Median Climate, 2020→2080)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top10['absolute_increase'] * 100)):
    ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=10)

plt.tight_layout()
bar_file = OUTPUT_DIR / "top10_vulnerable_zones.png"
plt.savefig(bar_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Vulnerability ranking saved: {bar_file}")

# ===================================================================
# PRINT SUMMARY
# ===================================================================

print()
print("="*80)
print("SUMMARY")
print("="*80)
print()

print(f"Total projections generated: {len(df_results):,}")
print(f"Zones analyzed: {len(df_results['zone_id'].unique())}")
print(f"Time period: {PROJECTION_YEARS[0]}-{PROJECTION_YEARS[-1]}")
print()

print("TOP 5 MOST VULNERABLE ZONES (Summer, Median Climate):")
print()
for i, row in df_zone_summary.nlargest(5, 'absolute_increase').iterrows():
    print(f"{row['zone_name']:30s}")
    print(f"  2020 high-risk: {row['baseline_high_risk_prop']*100:5.1f}%")
    print(f"  2080 high-risk: {row['future_high_risk_prop']*100:5.1f}%")
    print(f"  Absolute change: +{row['absolute_increase']*100:5.1f} percentage points")
    print(f"  Relative change: +{row['relative_increase']:5.1f}%")
    print()

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
