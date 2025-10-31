#!/usr/bin/env python
"""
Plot Climate Change Projections for Fire Risk

Generates plots from pre-computed climate projection data.

Purpose:
- Visualize future fire risk trends by zone under different climate scenarios.
- Compare RCP4.5 and RCP8.5 scenarios.
- Focus on plotting the data from zone_projections_relative_risk.csv.
"""

import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Use publication-quality settings
plt.style.use(['science', 'no-latex'])
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

print("="*80)
print("PLOTTING FIRE RISK CLIMATE PROJECTIONS")
print("="*80)
print()

# ===================================================================
# CONFIGURATION
# ===================================================================

BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape")
OUTPUT_DIR = BASE_DIR / "output/figures"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Input data paths
RCP45_CSV = BASE_DIR / "output/04_Zone_Climate_Projections/rcp45/zone_projections_relative_risk.csv"
RCP85_CSV = BASE_DIR / "output/04_Zone_Climate_Projections/rcp85/zone_projections_relative_risk.csv"

# Plotting parameters
MONTHS_TO_PLOT = [3, 8]  # March and August
month_names = {3: 'March', 8: 'August'}
ZONES_TO_PLOT = ['Bolzano', 'Merano', 'Bressanone', 'Silandro', 'Brunico'] # Some major zones

print("Configuration:")
print(f"  Input RCP4.5: {RCP45_CSV}")
print(f"  Input RCP8.5: {RCP85_CSV}")
print(f"  Output directory: {OUTPUT_DIR}")
print()

# ===================================================================
# LOAD DATA
# ===================================================================

print("Loading data...")
if RCP45_CSV.exists():
    df_rcp45 = pd.read_csv(RCP45_CSV)
    df_rcp45['scenario'] = 'RCP4.5'
    print("✓ RCP4.5 data loaded")
else:
    df_rcp45 = pd.DataFrame()
    print("✗ RCP4.5 data not found")

if RCP85_CSV.exists():
    df_rcp85 = pd.read_csv(RCP85_CSV)
    df_rcp85['scenario'] = 'RCP8.5'
    print("✓ RCP8.5 data loaded")
else:
    df_rcp85 = pd.DataFrame()
    print("✗ RCP8.5 data not found")

df_full = pd.concat([df_rcp45, df_rcp85], ignore_index=True)

if df_full.empty:
    print("No data to plot. Exiting.")
    exit()

print(f"✓ Data loaded and combined. Total records: {len(df_full)}")
print()

# ===================================================================
# GENERATE PLOTS
# ===================================================================

print("Generating plots...")

# Filter for zones and months of interest
df_plot = df_full[df_full['zone_name'].isin(ZONES_TO_PLOT) & df_full['month'].isin(MONTHS_TO_PLOT)]
df_plot['month_name'] = df_plot['month'].map(month_names)


# Create a plot for each month
for month_val, month_name in month_names.items():
    if month_val not in df_plot['month'].unique():
        continue

    print(f"  Generating plot for {month_name}...")

    g = sns.relplot(
        data=df_plot[df_plot['month'] == month_val],
        x='year',
        y='mean_risk',
        hue='climate_quantile',
        style='scenario',
        col='zone_name',
        kind='line',
        col_wrap=3,
        facet_kws={'sharey': False}
    )

    g.set_axis_labels("Year", "Mean Relative Risk")
    g.set_titles("Zone: {col_name}")
    # Add a single colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=-0.1, vmax=0.1))
    sm._A = []
    # Create a new axis for the colorbar
    cbar_ax = g.fig.add_axes([0.15, 0.02, 0.7, 0.02]) # [left, bottom, width, height]
    cbar = g.fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Change in Mean Relative Risk')


    plt.tight_layout(rect=[0, 0.08, 1, 1.0])

    plot_filename = OUTPUT_DIR / f"climate_projection_plot_{month_name.lower()}.png"
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Plot saved to {plot_filename}")

print()
print("="*80)
print("PLOTTING COMPLETE")
print("="*80)
