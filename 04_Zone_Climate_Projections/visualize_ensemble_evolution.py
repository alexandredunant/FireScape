#!/usr/bin/env python
"""
Visualize Ensemble Member Evolution - Maps and Time Series

Creates comprehensive visualizations showing:
1. Spatial maps: How wildfire risk evolves over time for each ensemble member
2. Time series: Trend plots showing risk evolution by zone and ensemble member
3. Ensemble spread: Uncertainty visualization across all members

Outputs:
- Multi-panel spatial maps showing risk evolution
- Time series plots with ensemble spread
- Per-season analysis
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path
import seaborn as sns

# Use SciencePlots style for publication-quality figures
import scienceplots

plt.style.use(["science", "no-latex"])

# ===================================================================
# CONFIGURATION
# ===================================================================

BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape")
FIRE_BRIGADE_ZONES = (
    BASE_DIR
    / "Data/06_Administrative_Boundaries/Processed/FireBrigade_ResponsibilityAreas_Bolzano_clipped.gpkg"
)

# Choose scenario
TARGET_SCENARIO = "rcp85"  # Primary scenario to visualize

INPUT_CSV_RCP85 = (
    BASE_DIR
    / "output/04_Zone_Climate_Projections/rcp85_ensemble_members/zone_projections_ensemble_members.csv"
)
INPUT_CSV_RCP45 = (
    BASE_DIR
    / "output/04_Zone_Climate_Projections/rcp45_ensemble_members/zone_projections_ensemble_members.csv"
)
OUTPUT_DIR = (
    BASE_DIR / f"output/04_Zone_Climate_Projections/{TARGET_SCENARIO}_ensemble_members"
)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Visualization parameters
SELECTED_YEARS = [2020, 2040, 2060, 2080, 2100]  # Years to show in maps
MONTH_NAMES = {2: "Feb", 3: "Mar", 8: "Aug", 10: "Oct"}
SEASON_NAMES = {2: "Winter", 3: "Spring", 8: "Summer", 10: "Fall"}

print("=" * 80)
print(f"ENSEMBLE MEMBER EVOLUTION VISUALIZATION - {TARGET_SCENARIO.upper()}")
print("=" * 80)
print()

# ===================================================================
# LOAD DATA
# ===================================================================

print("Loading data...")
df_rcp85 = pd.read_csv(INPUT_CSV_RCP85)
df_rcp45 = pd.read_csv(INPUT_CSV_RCP45)
gdf_zones = gpd.read_file(FIRE_BRIGADE_ZONES)

# Ensure zone_id consistency
if "ID" in gdf_zones.columns:
    gdf_zones["zone_id"] = gdf_zones["ID"]
elif "zone_id" not in gdf_zones.columns:
    gdf_zones["zone_id"] = range(len(gdf_zones))

print(f"✓ Loaded RCP8.5: {len(df_rcp85):,} projection records")
print(f"  - {df_rcp85['ensemble_member'].nunique()} ensemble members")
print(f"  - {df_rcp85['year'].nunique()} years")
print(f"  - {df_rcp85['month'].nunique()} months")
print(f"  - {df_rcp85['zone_id'].nunique()} zones")
print()
print(f"✓ Loaded RCP4.5: {len(df_rcp45):,} projection records")
print(f"  - {df_rcp45['ensemble_member'].nunique()} ensemble members")
print(f"  - {df_rcp45['year'].nunique()} years")
print(f"  - {df_rcp45['month'].nunique()} months")
print(f"  - {df_rcp45['zone_id'].nunique()} zones")
print()

# Use RCP8.5 as primary data for individual ensemble member visualization
df = df_rcp85.copy()

# Get ensemble member list
ensemble_members = sorted(df["ensemble_member"].unique())
print(f"RCP8.5 Ensemble members ({len(ensemble_members)}):")
for em in ensemble_members:
    print(f"  - {em}")
print()

# ===================================================================
# CUSTOM COLORMAP FOR FIRE RISK
# ===================================================================


def create_fire_risk_colormap():
    """Create a perceptually uniform colormap for fire risk."""
    colors = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
    return LinearSegmentedColormap.from_list("fire_risk", colors, N=256)


fire_cmap = create_fire_risk_colormap()

# ===================================================================
# 1. SPATIAL MAPS: EVOLUTION OVER TIME BY ENSEMBLE MEMBER
# ===================================================================

print("Creating spatial evolution maps...")

for month in sorted(df["month"].unique()):
    season_name = SEASON_NAMES.get(month, f"Month{month}")
    month_name = MONTH_NAMES.get(month, f"M{month}")

    print(f"\n  Processing {season_name} ({month_name})...")

    # Filter data for this month
    df_month = df[df["month"] == month].copy()

    # Calculate ensemble statistics for color scale limits
    all_risks = df_month.groupby(["zone_id", "year"])["mean_risk"].mean().values
    all_risks = all_risks[~np.isnan(all_risks)]
    vmin = np.percentile(all_risks, 1)
    vmax = np.percentile(all_risks, 99)

    # Create figure for each ensemble member
    for ens_idx, ensemble_member in enumerate(ensemble_members):
        print(f"    - {ensemble_member[:50]}...")

        df_ens = df_month[df_month["ensemble_member"] == ensemble_member].copy()

        # Create figure with maps for selected years
        n_years = len(SELECTED_YEARS)
        fig, axes = plt.subplots(1, n_years, figsize=(4 * n_years, 5))
        if n_years == 1:
            axes = [axes]

        for ax_idx, year in enumerate(SELECTED_YEARS):
            ax = axes[ax_idx]

            # Get data for this year
            df_year = df_ens[df_ens["year"] == year].copy()

            if len(df_year) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
                continue

            # Merge with geometries
            gdf_plot = gdf_zones.merge(
                df_year[["zone_id", "mean_risk"]], on="zone_id", how="left"
            )

            # Plot with consistent 0-0.6 scale
            gdf_plot.plot(
                column="mean_risk",
                ax=ax,
                cmap=fire_cmap,
                vmin=0,
                vmax=0.6,
                edgecolor="white",
                linewidth=0.3,
                legend=False,
                missing_kwds={
                    "color": "lightgrey",
                    "edgecolor": "white",
                    "linewidth": 0.3,
                },
            )

            # Subplot label
            ax.text(
                0.05,
                0.95,
                f"{year}",
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
            ax.axis("off")

        # Add colorbar with proper spacing
        sm = plt.cm.ScalarMappable(cmap=fire_cmap, norm=plt.Normalize(vmin=0, vmax=0.6))
        sm.set_array([])
        plt.tight_layout()
        cbar = fig.colorbar(
            sm, ax=axes, orientation="horizontal", pad=0.08, fraction=0.04, aspect=30
        )
        cbar.set_label("Mean Fire Risk (Relative Probability)", fontsize=12)
        cbar.ax.tick_params(labelsize=11)

        # Save
        safe_ens_name = ensemble_member.replace("/", "_").replace(" ", "_")[:80]
        output_file = OUTPUT_DIR / f"spatial_evolution_{month_name}_{safe_ens_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

print("\n✓ Spatial evolution maps created")

# ===================================================================
# 1B. WORST-CASE SCENARIO SUMMARY (HIGHEST RISK ENSEMBLE PER MONTH)
# ===================================================================

print("\nCreating worst-case scenario summary maps...")

# For each month, find the ensemble member with highest average risk
worst_case_data = []

for month in sorted(df["month"].unique()):
    df_month = df[df["month"] == month].copy()

    # Calculate average risk per ensemble member across all zones and years
    ensemble_avg_risk = df_month.groupby("ensemble_member")["mean_risk"].mean()
    worst_case_ensemble = ensemble_avg_risk.idxmax()

    worst_case_data.append(
        {
            "month": month,
            "worst_case_ensemble": worst_case_ensemble,
            "avg_risk": ensemble_avg_risk.max(),
        }
    )

    print(
        f"  {SEASON_NAMES.get(month, f'Month{month}')} ({MONTH_NAMES.get(month, f'M{month}')}): {worst_case_ensemble[:60]}..."
    )

# Create summary figure showing worst-case ensemble for each month
selected_months = [2, 3, 8, 10]  # Feb, Mar, Aug, Oct
n_months = len(selected_months)
n_years = len(SELECTED_YEARS)

fig, axes = plt.subplots(
    n_months,
    n_years,
    figsize=(4 * n_years, 3.5 * n_months),
    gridspec_kw={"hspace": 0.05, "wspace": 0.05},
)  # Reduce vertical and horizontal spacing

# Calculate global color scale across all months and years for worst-case scenarios
all_worst_risks = []
for month_data in worst_case_data:
    if month_data["month"] in selected_months:
        month = month_data["month"]
        worst_ensemble = month_data["worst_case_ensemble"]
        df_worst = df[
            (df["month"] == month) & (df["ensemble_member"] == worst_ensemble)
        ]
        all_worst_risks.extend(df_worst["mean_risk"].values)

all_worst_risks = np.array(all_worst_risks)
all_worst_risks = all_worst_risks[~np.isnan(all_worst_risks)]
vmin = np.percentile(all_worst_risks, 1)
vmax = np.percentile(all_worst_risks, 99)

for month_idx, month in enumerate(selected_months):
    season_name = SEASON_NAMES.get(month, f"Month{month}")
    month_name = MONTH_NAMES.get(month, f"M{month}")

    # Get worst-case ensemble for this month
    month_data = [m for m in worst_case_data if m["month"] == month][0]
    worst_ensemble = month_data["worst_case_ensemble"]

    df_worst = df[
        (df["month"] == month) & (df["ensemble_member"] == worst_ensemble)
    ].copy()

    for year_idx, year in enumerate(SELECTED_YEARS):
        if n_months == 1:
            ax = axes[year_idx]
        else:
            ax = axes[month_idx, year_idx]

        # Get data for this year
        df_year = df_worst[df_worst["year"] == year].copy()

        if len(df_year) == 0:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.axis("off")
            continue

        # Merge with geometries
        gdf_plot = gdf_zones.merge(
            df_year[["zone_id", "mean_risk"]], on="zone_id", how="left"
        )

        # Plot with consistent 0-0.6 scale
        gdf_plot.plot(
            column="mean_risk",
            ax=ax,
            cmap=fire_cmap,
            vmin=0,
            vmax=0.6,
            edgecolor="white",
            linewidth=0.3,
            legend=False,
            missing_kwds={"color": "lightgrey", "edgecolor": "white", "linewidth": 0.3},
        )

        # Add subplot label with doubled font size
        if year_idx == 0:
            ax.text(
                0.05,
                0.95,
                f"{season_name}\n{year}",
                transform=ax.transAxes,
                fontsize=20,
                fontweight="bold",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
        else:
            ax.text(
                0.05,
                0.95,
                f"{year}",
                transform=ax.transAxes,
                fontsize=20,
                fontweight="bold",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
        ax.axis("off")

# Add colorbar - reduced to 1/3 size with consistent 0-0.6 scale
sm = plt.cm.ScalarMappable(cmap=fire_cmap, norm=plt.Normalize(vmin=0, vmax=0.6))
sm.set_array([])
plt.tight_layout()
cbar = fig.colorbar(
    sm, ax=axes, orientation="horizontal", pad=0.02, fraction=0.01, aspect=50
)
cbar.set_label("Mean Fire Risk (Relative Probability)", fontsize=20)
cbar.ax.tick_params(labelsize=16)

output_file = OUTPUT_DIR / f"spatial_evolution_worst_case_summary.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n✓ Saved: {output_file}")

# ===================================================================
# 2. TIME SERIES: ENSEMBLE SPREAD BY SEASON
# ===================================================================

print("\nCreating ensemble time series plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for ax_idx, month in enumerate(sorted(df["month"].unique())):
    ax = axes[ax_idx]
    season_name = SEASON_NAMES.get(month, f"Month{month}")
    month_name = MONTH_NAMES.get(month, f"M{month}")

    df_month_rcp85 = df_rcp85[df_rcp85["month"] == month].copy()
    df_month_rcp45 = df_rcp45[df_rcp45["month"] == month].copy()

    # Calculate province-wide average for each ensemble member and year (RCP8.5)
    ensemble_timeseries_rcp85 = (
        df_month_rcp85.groupby(["ensemble_member", "year"])["mean_risk"]
        .mean()
        .reset_index()
    )

    # Plot each RCP8.5 ensemble member
    for ensemble_member in ensemble_members:
        df_ens = ensemble_timeseries_rcp85[
            ensemble_timeseries_rcp85["ensemble_member"] == ensemble_member
        ]
        ax.plot(
            df_ens["year"],
            df_ens["mean_risk"],
            alpha=0.4,
            linewidth=1.5,
            label=None,
            color="tab:blue",
        )

    # Calculate and plot RCP8.5 ensemble mean
    ensemble_mean_rcp85 = (
        ensemble_timeseries_rcp85.groupby("year")["mean_risk"].mean().reset_index()
    )
    ensemble_std_rcp85 = (
        ensemble_timeseries_rcp85.groupby("year")["mean_risk"].std().reset_index()
    )

    ax.plot(
        ensemble_mean_rcp85["year"],
        ensemble_mean_rcp85["mean_risk"],
        color="darkblue",
        linewidth=3,
        label="RCP8.5 Ensemble Mean",
        zorder=10,
        linestyle="-",
    )

    # Add uncertainty band for RCP8.5
    ax.fill_between(
        ensemble_mean_rcp85["year"],
        ensemble_mean_rcp85["mean_risk"] - ensemble_std_rcp85["mean_risk"],
        ensemble_mean_rcp85["mean_risk"] + ensemble_std_rcp85["mean_risk"],
        alpha=0.2,
        color="blue",
        label="RCP8.5 ± 1 Std Dev",
    )

    # Calculate and plot RCP4.5 ensemble mean as dotted line
    ensemble_timeseries_rcp45 = (
        df_month_rcp45.groupby(["ensemble_member", "year"])["mean_risk"]
        .mean()
        .reset_index()
    )
    ensemble_mean_rcp45 = (
        ensemble_timeseries_rcp45.groupby("year")["mean_risk"].mean().reset_index()
    )

    ax.plot(
        ensemble_mean_rcp45["year"],
        ensemble_mean_rcp45["mean_risk"],
        color="darkgreen",
        linewidth=3,
        label="RCP4.5 Ensemble Mean",
        zorder=10,
        linestyle=":",
    )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Mean Fire Risk", fontsize=12)
    ax.set_ylim(0, 0.6)  # Consistent scale across all plots
    # Add season label as text annotation (upper-left)
    ax.text(
        0.02,
        0.98,
        f"({chr(97 + ax_idx)}) {season_name} ({month_name})",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.tick_params(axis="both", labelsize=11)

    # Add baseline reference (2020)
    baseline = ensemble_mean_rcp85[ensemble_mean_rcp85["year"] == 2020][
        "mean_risk"
    ].values
    if len(baseline) > 0:
        ax.axhline(
            baseline[0],
            color="red",
            linestyle="--",
            alpha=0.5,
            linewidth=1.5,
            label="2020 Baseline",
        )

plt.tight_layout()

output_file = OUTPUT_DIR / f"timeseries_ensemble_spread_all_seasons.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"✓ Saved: {output_file}")

# ===================================================================
# 3. INDIVIDUAL ZONE TIME SERIES (TOP 5 HIGHEST RISK ZONES)
# ===================================================================

print("\nCreating individual zone time series...")

# Define specific zones to plot by name (PLACE_IT column)
target_zone_names = ["Predoi", "Braies", "Nova Ponente", "Mazia", "San Martino"]

# Get zone names - check for different possible column names, prioritizing PLACE_IT
zone_name_col = None
for col in ["PLACE_IT", "zone_name", "ZONE_NAME", "name", "NAME", "Name"]:
    if col in gdf_zones.columns:
        zone_name_col = col
        break

if zone_name_col:
    zone_names = gdf_zones.set_index("zone_id")[zone_name_col].to_dict()
    # Find zone IDs for the target zone names
    top_zones = []
    for zone_id, zone_name in zone_names.items():
        if zone_name in target_zone_names:
            top_zones.append(zone_id)

    if len(top_zones) != 5:
        print(f"Warning: Found {len(top_zones)} matching zones instead of 5")
        print(f"  Available zone names: {sorted(set(zone_names.values()))}")
else:
    # If no name column, fall back to highest risk zones
    print("Warning: No PLACE_IT or zone name column found, using highest-risk zones")
    zone_avg_risk = (
        df.groupby("zone_id")["mean_risk"].mean().sort_values(ascending=False).head(5)
    )
    top_zones = zone_avg_risk.index.tolist()
    zone_names = {zone_id: f"Zone {zone_id}" for zone_id in gdf_zones["zone_id"]}

# Create single combined figure with all months color-coded
fig, axes = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={"hspace": 0.25})
axes = axes.flatten()

# Define colors for each month/season
month_colors = {
    2: "#3182bd",  # Feb (Winter) - Blue
    3: "#31a354",  # Mar (Spring) - Green
    8: "#de2d26",  # Aug (Summer) - Red
    10: "#ff7f00",  # Oct (Fall) - Orange
}

for ax_idx, zone_id in enumerate(top_zones):
    ax = axes[ax_idx]
    zone_name = zone_names.get(zone_id, f"Zone {zone_id}")

    # Plot each month with different colors
    for month in sorted(df["month"].unique()):
        season_name = SEASON_NAMES.get(month, f"Month{month}")
        month_name = MONTH_NAMES.get(month, f"M{month}")
        color = month_colors.get(month, "gray")

        df_month_rcp85 = df_rcp85[df_rcp85["month"] == month].copy()
        df_month_rcp45 = df_rcp45[df_rcp45["month"] == month].copy()

        df_zone_rcp85 = df_month_rcp85[df_month_rcp85["zone_id"] == zone_id].copy()
        df_zone_rcp45 = df_month_rcp45[df_month_rcp45["zone_id"] == zone_id].copy()

        # Calculate RCP8.5 ensemble mean and std
        ensemble_mean_rcp85 = (
            df_zone_rcp85.groupby("year")["mean_risk"].mean().reset_index()
        )
        ensemble_std_rcp85 = (
            df_zone_rcp85.groupby("year")["mean_risk"].std().reset_index()
        )

        # Plot RCP8.5 ensemble mean as solid line
        ax.plot(
            ensemble_mean_rcp85["year"],
            ensemble_mean_rcp85["mean_risk"],
            color=color,
            linewidth=2.5,
            label=f"{season_name} RCP8.5",
            zorder=10,
            linestyle="-",
        )

        # Add uncertainty band with more transparency
        ax.fill_between(
            ensemble_mean_rcp85["year"],
            ensemble_mean_rcp85["mean_risk"] - ensemble_std_rcp85["mean_risk"],
            ensemble_mean_rcp85["mean_risk"] + ensemble_std_rcp85["mean_risk"],
            alpha=0.15,
            color=color,
        )

        # Calculate and plot RCP4.5 ensemble mean as dotted line
        ensemble_mean_rcp45 = (
            df_zone_rcp45.groupby(["ensemble_member", "year"])["mean_risk"]
            .mean()
            .groupby("year")
            .mean()
            .reset_index()
        )

        ax.plot(
            ensemble_mean_rcp45["year"],
            ensemble_mean_rcp45["mean_risk"],
            color=color,
            linewidth=2.5,
            label=f"{season_name} RCP4.5",
            zorder=10,
            linestyle=":",
        )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Mean Fire Risk", fontsize=12)
    ax.set_ylim(0, 0.6)  # Consistent scale across all plots
    # Add zone label as text annotation with zone name (center-left)
    ax.text(
        0.02,
        0.5,
        f"({chr(97 + ax_idx)}) {zone_name}",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="center",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9, ncol=2)
    ax.tick_params(axis="both", labelsize=11)

# Add minimap in the last subplot (bottom right, position 5)
ax_map = axes[-1]
ax_map.axis("on")

# Plot all zones in light gray
gdf_zones.plot(ax=ax_map, color="lightgray", edgecolor="black", linewidth=0.5)

# Highlight the top 5 zones with different colors
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]  # Distinct colors
for idx, zone_id in enumerate(top_zones):
    zone_geom = gdf_zones[gdf_zones["zone_id"] == zone_id]
    zone_geom.plot(
        ax=ax_map, color=colors[idx], edgecolor="black", linewidth=0.8, alpha=0.7
    )

    # Add zone label on map
    centroid = zone_geom.geometry.centroid.iloc[0]
    ax_map.text(
        centroid.x,
        centroid.y,
        chr(97 + idx),
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="circle", facecolor="white", alpha=0.9, edgecolor="black"),
    )

ax_map.set_title("Zone Locations", fontsize=12, fontweight="bold")
ax_map.axis("off")

plt.tight_layout()

output_file = OUTPUT_DIR / f"timeseries_top_zones_combined.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"✓ Saved combined time series: {output_file}")

# ===================================================================
# 3B. ENSEMBLE MIN/MEAN/MAX ACROSS ALL ZONES
# ===================================================================

print("\nCreating ensemble min/mean/max time series for extreme zones...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for ax_idx, month in enumerate(sorted(df["month"].unique())):
    ax = axes[ax_idx]
    season_name = SEASON_NAMES.get(month, f"Month{month}")
    month_name = MONTH_NAMES.get(month, f"M{month}")

    df_month_rcp85 = df_rcp85[df_rcp85["month"] == month].copy()
    df_month_rcp45 = df_rcp45[df_rcp45["month"] == month].copy()

    # Calculate risk change from 2020 to 2100 for each zone and ensemble member (RCP8.5)
    risk_2020_rcp85 = (
        df_month_rcp85[df_month_rcp85["year"] == 2020]
        .groupby(["zone_id", "ensemble_member"])["mean_risk"]
        .mean()
    )
    risk_2100_rcp85 = (
        df_month_rcp85[df_month_rcp85["year"] == 2100]
        .groupby(["zone_id", "ensemble_member"])["mean_risk"]
        .mean()
    )
    risk_change_rcp85 = (risk_2100_rcp85 - risk_2020_rcp85).reset_index()
    risk_change_rcp85.columns = ["zone_id", "ensemble_member", "risk_change"]

    # Find zones with maximum and minimum average risk increase
    zone_avg_change_rcp85 = risk_change_rcp85.groupby("zone_id")["risk_change"].mean()
    max_zone_id = zone_avg_change_rcp85.idxmax()
    min_zone_id = zone_avg_change_rcp85.idxmin()

    # Get zone names
    max_zone_name = zone_names.get(max_zone_id, f"Zone {max_zone_id}")
    min_zone_name = zone_names.get(min_zone_id, f"Zone {min_zone_id}")

    # Get time series for max risk increase zone (RCP8.5 and RCP4.5)
    df_max_zone_rcp85 = df_month_rcp85[df_month_rcp85["zone_id"] == max_zone_id]
    df_max_zone_rcp45 = df_month_rcp45[df_month_rcp45["zone_id"] == max_zone_id]
    max_zone_mean_rcp85 = (
        df_max_zone_rcp85.groupby("year")["mean_risk"].mean().reset_index()
    )
    max_zone_mean_rcp45 = (
        df_max_zone_rcp45.groupby(["ensemble_member", "year"])["mean_risk"]
        .mean()
        .groupby("year")
        .mean()
        .reset_index()
    )

    # Get time series for min risk increase zone (RCP8.5 and RCP4.5)
    df_min_zone_rcp85 = df_month_rcp85[df_month_rcp85["zone_id"] == min_zone_id]
    df_min_zone_rcp45 = df_month_rcp45[df_month_rcp45["zone_id"] == min_zone_id]
    min_zone_mean_rcp85 = (
        df_min_zone_rcp85.groupby("year")["mean_risk"].mean().reset_index()
    )
    min_zone_mean_rcp45 = (
        df_min_zone_rcp45.groupby(["ensemble_member", "year"])["mean_risk"]
        .mean()
        .groupby("year")
        .mean()
        .reset_index()
    )

    # Debug: Print zone statistics for all seasons to investigate anomalies
    print(f"\n  {season_name} ({month_name}) - Zone Statistics:")
    print(f"    Max zone: {max_zone_name} (ID: {max_zone_id})")
    print(f"      - Risk change 2020→2100: {zone_avg_change_rcp85[max_zone_id]:.4f}")
    print(
        f"      - Risk 2020: {df_max_zone_rcp85[df_max_zone_rcp85['year'] == 2020]['mean_risk'].mean():.4f}"
    )
    print(
        f"      - Risk 2100: {df_max_zone_rcp85[df_max_zone_rcp85['year'] == 2100]['mean_risk'].mean():.4f}"
    )
    print(f"    Min zone: {min_zone_name} (ID: {min_zone_id})")
    print(f"      - Risk change 2020→2100: {zone_avg_change_rcp85[min_zone_id]:.4f}")
    print(
        f"      - Risk 2020: {df_min_zone_rcp85[df_min_zone_rcp85['year'] == 2020]['mean_risk'].mean():.4f}"
    )
    print(
        f"      - Risk 2100: {df_min_zone_rcp85[df_min_zone_rcp85['year'] == 2100]['mean_risk'].mean():.4f}"
    )
    print(f"    Province mean:")
    print(
        f"      - Risk 2020: {df_month_rcp85[df_month_rcp85['year'] == 2020]['mean_risk'].mean():.4f}"
    )
    print(
        f"      - Risk 2100: {df_month_rcp85[df_month_rcp85['year'] == 2100]['mean_risk'].mean():.4f}"
    )
    print(
        f"      - Change: {df_month_rcp85[df_month_rcp85['year'] == 2100]['mean_risk'].mean() - df_month_rcp85[df_month_rcp85['year'] == 2020]['mean_risk'].mean():.4f}"
    )

    # Get province-wide ensemble mean (RCP8.5 and RCP4.5)
    province_mean_rcp85 = (
        df_month_rcp85.groupby(["ensemble_member", "year"])["mean_risk"]
        .mean()
        .groupby("year")
        .mean()
        .reset_index()
    )
    province_mean_rcp45 = (
        df_month_rcp45.groupby(["ensemble_member", "year"])["mean_risk"]
        .mean()
        .groupby("year")
        .mean()
        .reset_index()
    )

    # Find the worst-case ensemble member (highest average risk) for this month
    ensemble_avg_risk_rcp85 = df_month_rcp85.groupby("ensemble_member")[
        "mean_risk"
    ].mean()
    worst_case_ensemble_rcp85 = ensemble_avg_risk_rcp85.idxmax()
    ensemble_avg_risk_rcp45 = df_month_rcp45.groupby("ensemble_member")[
        "mean_risk"
    ].mean()
    worst_case_ensemble_rcp45 = ensemble_avg_risk_rcp45.idxmax()

    # Get time series for worst-case ensemble members
    df_worst_rcp85 = df_month_rcp85[
        df_month_rcp85["ensemble_member"] == worst_case_ensemble_rcp85
    ]
    worst_ensemble_mean_rcp85 = (
        df_worst_rcp85.groupby("year")["mean_risk"].mean().reset_index()
    )
    df_worst_rcp45 = df_month_rcp45[
        df_month_rcp45["ensemble_member"] == worst_case_ensemble_rcp45
    ]
    worst_ensemble_mean_rcp45 = (
        df_worst_rcp45.groupby("year")["mean_risk"].mean().reset_index()
    )

    # Plot RCP8.5 (solid lines, thicker)
    ax.plot(
        province_mean_rcp85["year"],
        province_mean_rcp85["mean_risk"],
        color="darkblue",
        linewidth=3.5,
        label="RCP8.5 Mean",
        zorder=10,
        linestyle="-",
    )
    ax.plot(
        max_zone_mean_rcp85["year"],
        max_zone_mean_rcp85["mean_risk"],
        color="red",
        linewidth=3.5,
        label=f"RCP8.5 Max ({max_zone_name})",
        zorder=9,
        linestyle="-",
    )
    ax.plot(
        min_zone_mean_rcp85["year"],
        min_zone_mean_rcp85["mean_risk"],
        color="lightblue",
        linewidth=3.5,
        label=f"RCP8.5 Min ({min_zone_name})",
        zorder=9,
        linestyle="-",
    )
    ax.plot(
        worst_ensemble_mean_rcp85["year"],
        worst_ensemble_mean_rcp85["mean_risk"],
        color="orange",
        linewidth=3.5,
        label="RCP8.5 Worst Ensemble",
        zorder=9,
        linestyle="-",
    )

    # Plot RCP4.5 (dotted lines, same colors as RCP8.5)
    ax.plot(
        province_mean_rcp45["year"],
        province_mean_rcp45["mean_risk"],
        color="darkblue",
        linewidth=3.5,
        label="RCP4.5 Mean",
        zorder=10,
        linestyle=":",
    )
    ax.plot(
        max_zone_mean_rcp45["year"],
        max_zone_mean_rcp45["mean_risk"],
        color="red",
        linewidth=3.5,
        label=f"RCP4.5 Max ({max_zone_name})",
        zorder=9,
        linestyle=":",
    )
    ax.plot(
        min_zone_mean_rcp45["year"],
        min_zone_mean_rcp45["mean_risk"],
        color="lightblue",
        linewidth=3.5,
        label=f"RCP4.5 Min ({min_zone_name})",
        zorder=9,
        linestyle=":",
    )
    ax.plot(
        worst_ensemble_mean_rcp45["year"],
        worst_ensemble_mean_rcp45["mean_risk"],
        color="orange",
        linewidth=3.5,
        label="RCP4.5 Worst Ensemble",
        zorder=9,
        linestyle=":",
    )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Mean Fire Risk", fontsize=12)
    ax.set_ylim(0, 0.5)
    ax.text(
        0.02,
        0.98,
        f"({chr(97 + ax_idx)}) {season_name} ({month_name})",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9, ncol=2)
    ax.tick_params(axis="both", labelsize=11)

plt.tight_layout()

output_file = OUTPUT_DIR / f"timeseries_ensemble_min_mean_max.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"✓ Saved: {output_file}")

# ===================================================================
# 4. ENSEMBLE SPREAD ANALYSIS
# ===================================================================

print("\nCreating ensemble spread analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for ax_idx, month in enumerate(sorted(df["month"].unique())):
    ax = axes[ax_idx]
    season_name = SEASON_NAMES.get(month, f"Month{month}")
    month_name = MONTH_NAMES.get(month, f"M{month}")

    df_month = df[df["month"] == month].copy()

    # Calculate ensemble spread (coefficient of variation) for each year
    ensemble_stats = (
        df_month.groupby(["year", "zone_id"])["mean_risk"]
        .agg(["mean", "std"])
        .reset_index()
    )
    ensemble_stats["cv"] = (
        ensemble_stats["std"] / ensemble_stats["mean"]
    )  # Coefficient of variation

    # Average across zones
    yearly_cv = ensemble_stats.groupby("year")["cv"].mean().reset_index()
    yearly_std = ensemble_stats.groupby("year")["std"].mean().reset_index()

    ax2 = ax.twinx()

    # Plot both absolute spread (std) and relative spread (CV)
    line1 = ax.plot(
        yearly_std["year"],
        yearly_std["std"],
        color="blue",
        linewidth=2.5,
        marker="o",
        label="Absolute Spread (Std Dev)",
    )
    line2 = ax2.plot(
        yearly_cv["year"],
        yearly_cv["cv"],
        color="red",
        linewidth=2.5,
        marker="s",
        label="Relative Spread (CV)",
    )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Standard Deviation", fontsize=12, color="blue")
    ax2.set_ylabel("Coefficient of Variation", fontsize=12, color="red")
    # Add season label as text annotation
    ax.text(
        0.02,
        0.98,
        f"({chr(97 + ax_idx)}) {season_name} ({month_name})",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(axis="y", labelcolor="blue", labelsize=11)
    ax2.tick_params(axis="y", labelcolor="red", labelsize=11)
    ax.tick_params(axis="x", labelsize=11)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper right", fontsize=10, framealpha=0.9)

plt.tight_layout()

output_file = OUTPUT_DIR / f"ensemble_spread_analysis.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"✓ Saved: {output_file}")

# ===================================================================
# 5. SUMMARY STATISTICS TABLE
# ===================================================================

print("\nGenerating summary statistics...")

summary_data = []

for month in sorted(df["month"].unique()):
    season_name = SEASON_NAMES.get(month, f"Month{month}")
    df_month = df[df["month"] == month].copy()

    for year in sorted(df_month["year"].unique()):
        df_year = df_month[df_month["year"] == year]

        # Province-wide statistics
        province_mean = df_year["mean_risk"].mean()
        province_std = df_year["mean_risk"].std()

        # Ensemble spread
        ensemble_spread = df_year.groupby("ensemble_member")["mean_risk"].mean().std()

        summary_data.append(
            {
                "Season": season_name,
                "Year": year,
                "Province_Mean_Risk": province_mean,
                "Province_Std": province_std,
                "Ensemble_Spread": ensemble_spread,
                "Relative_Uncertainty": ensemble_spread / province_mean
                if province_mean > 0
                else np.nan,
            }
        )

df_summary = pd.DataFrame(summary_data)
summary_file = OUTPUT_DIR / "ensemble_summary_statistics.csv"
df_summary.to_csv(summary_file, index=False)

print(f"✓ Saved: {summary_file}")
print()

# ===================================================================
# COMPLETION
# ===================================================================

print("==" * 40)
print("VISUALIZATION COMPLETE")
print("==" * 40)
print()
print("Generated outputs:")
print(
    f"  1. Spatial evolution maps (per ensemble member): {OUTPUT_DIR}/spatial_evolution_*.png"
)
print(
    f"  2. Worst-case scenario summary map: {OUTPUT_DIR}/spatial_evolution_worst_case_summary.png"
)
print(
    f"  3. Time series with ensemble spread: {OUTPUT_DIR}/timeseries_ensemble_spread_all_seasons.png"
)
print(f"  4. Top zones time series: {OUTPUT_DIR}/timeseries_top_zones_*.png")
print(f"  5. Ensemble spread analysis: {OUTPUT_DIR}/ensemble_spread_analysis.png")
print(f"  6. Summary statistics: {summary_file}")
print()
print("==" * 40)
