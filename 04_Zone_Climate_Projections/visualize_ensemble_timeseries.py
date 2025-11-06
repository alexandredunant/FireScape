#!/usr/bin/env python
"""
Visualize Ensemble Member Fire Risk Projections as Time Series

Creates time series plots showing fire risk evolution over time with
ensemble spread for different zones and scenarios.

Features:
- Time series for both RCP4.5 and RCP8.5
- Ensemble mean with uncertainty bands (ensemble spread)
- Individual ensemble member trajectories
- Separate plots for different seasons
- Zone-specific and region-wide plots
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scienceplots

plt.style.use(["science", "no-latex"])
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

print("=" * 80)
print("ENSEMBLE FIRE RISK PROJECTION TIME SERIES")
print("=" * 80)
print()

# ===================================================================
# CONFIGURATION
# ==================================================================

BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape")
FIRE_BRIGADE_ZONES = (
    BASE_DIR
    / "Data/06_Administrative_Boundaries/Processed/FireBrigade_ResponsibilityAreas_Bolzano_clipped.gpkg"
)

# Input data directories
RCP45_DIR = BASE_DIR / "output/04_Zone_Climate_Projections/rcp45_ensemble_members"
RCP85_DIR = BASE_DIR / "output/04_Zone_Climate_Projections/rcp85_ensemble_members"

# Output directory
OUTPUT_DIR = BASE_DIR / "output/04_Zone_Climate_Projections/ensemble_timeseries"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Visualization parameters
SCENARIOS = {
    "rcp45": {"dir": RCP45_DIR, "label": "RCP4.5", "color": "steelblue"},
    "rcp85": {"dir": RCP85_DIR, "label": "RCP8.5", "color": "orangered"},
}

MONTHS = {2: "February", 3: "March", 8: "August", 10: "October"}

# Options
SHOW_INDIVIDUAL_MEMBERS = True  # Show individual ensemble member lines
TOP_N_ZONES = 5  # Number of highest-risk zones to plot individually

print("Configuration:")
print(f"  Scenarios: {', '.join([v['label'] for v in SCENARIOS.values()])}")
print(f"  Months: {', '.join(MONTHS.values())}")
print(f"  Show individual members: {SHOW_INDIVIDUAL_MEMBERS}")
print(f"  Output: {OUTPUT_DIR}")
print()

# ===================================================================
# LOAD DATA
# ===================================================================

print("Loading fire brigade zones...")
gdf_zones = gpd.read_file(FIRE_BRIGADE_ZONES)
# Get zone names
for col in ["PLACE_IT", "NAME", "name", "NAME_IT"]:
    if col in gdf_zones.columns:
        gdf_zones["zone_name"] = gdf_zones[col]
        break
else:
    gdf_zones["zone_name"] = (
        "Zone_" + gdf_zones["ID"].astype(str)
        if "ID" in gdf_zones.columns
        else "Zone_" + gdf_zones.index.astype(str)
    )

print(f"✓ Loaded {len(gdf_zones)} zones")
print()

# Load ensemble results
ensemble_data = {}

for scenario_key, scenario_info in SCENARIOS.items():
    csv_path = scenario_info["dir"] / "zone_projections_ensemble_members.csv"

    if csv_path.exists():
        print(f"Loading {scenario_info['label']} data...")
        df = pd.read_csv(csv_path)

        # Merge zone names
        df = df.merge(
            gdf_zones[["ID", "zone_name"]].rename(columns={"ID": "zone_id"}),
            on="zone_id",
            how="left",
        )

        ensemble_data[scenario_key] = df
        print(f"  ✓ {len(df)} records")
        print(f"  Ensemble members: {df['ensemble_member'].nunique()}")
    else:
        print(f"⚠️  Warning: {csv_path} not found, skipping {scenario_info['label']}")

print()

if not ensemble_data:
    print("ERROR: No data files found!")
    exit(1)

# ===================================================================
# FUNCTION: PLOT TIME SERIES
# ===================================================================


def plot_timeseries_with_ensemble(
    data_dict, month, title_suffix, output_path, zone_filter=None, show_members=True
):
    """
    Plot time series with ensemble statistics.

    Parameters:
    - data_dict: dict of {scenario_key: dataframe}
    - month: month to filter
    - title_suffix: additional title text
    - output_path: where to save
    - zone_filter: list of zone_ids to include (None = all zones aggregated)
    - show_members: whether to show individual ensemble member lines
    """

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    for scenario_key, df in data_dict.items():
        scenario_info = SCENARIOS[scenario_key]
        color = scenario_info["color"]
        label = scenario_info["label"]

        # Filter data
        df_month = df[df["month"] == month].copy()

        if zone_filter is not None:
            df_month = df_month[df_month["zone_id"].isin(zone_filter)]

        if len(df_month) == 0:
            continue

        # Calculate ensemble statistics per year
        ensemble_stats = (
            df_month.groupby("year")
            .agg({"mean_risk": ["mean", "std", "min", "max"]})
            .reset_index()
        )
        ensemble_stats.columns = [
            "year",
            "ensemble_mean",
            "ensemble_std",
            "ensemble_min",
            "ensemble_max",
        ]

        years = ensemble_stats["year"].values
        mean = ensemble_stats["ensemble_mean"].values
        std = ensemble_stats["ensemble_std"].values

        # Plot ensemble mean
        ax.plot(
            years,
            mean,
            color=color,
            linewidth=2.5,
            label=f"{label} (Ensemble Mean)",
            marker="o",
            markersize=6,
            zorder=10,
        )

        # Plot uncertainty band (mean ± std)
        ax.fill_between(
            years,
            mean - std,
            mean + std,
            color=color,
            alpha=0.2,
            label=f"{label} (±1 Std)",
            zorder=5,
        )

        # Plot individual ensemble members (optional)
        if show_members:
            for member in df_month["ensemble_member"].unique():
                member_data = (
                    df_month[df_month["ensemble_member"] == member]
                    .groupby("year")["mean_risk"]
                    .mean()
                )
                ax.plot(
                    member_data.index,
                    member_data.values,
                    color=color,
                    linewidth=0.5,
                    alpha=0.3,
                    zorder=3,
                )

    ax.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax.set_ylabel("Relative Fire Risk", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{MONTHS[month]} - {title_suffix}", fontsize=14, fontweight="bold", pad=15
    )
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"  ✓ Saved: {output_path.name}")


# ===================================================================
# GENERATE TIME SERIES PLOTS
# ===================================================================

print("=" * 80)
print("GENERATING TIME SERIES PLOTS")
print("=" * 80)
print()

# 1. REGION-WIDE TIME SERIES (all zones aggregated)
print("Region-wide time series:")
print("-" * 60)

for month in MONTHS.keys():
    output_path = OUTPUT_DIR / f"timeseries_regionwide_{month:02d}_{MONTHS[month]}.png"
    plot_timeseries_with_ensemble(
        ensemble_data,
        month,
        title_suffix="Region-Wide Average",
        output_path=output_path,
        zone_filter=None,
        show_members=SHOW_INDIVIDUAL_MEMBERS,
    )

print()

# 2. TOP RISK ZONES TIME SERIES
print(f"Top {TOP_N_ZONES} highest-risk zones:")
print("-" * 60)

# Identify top risk zones based on RCP8.5 August 2080
if "rcp85" in ensemble_data:
    df_ranking = ensemble_data["rcp85"]
    df_ranking = df_ranking[(df_ranking["month"] == 8) & (df_ranking["year"] == 2080)]

    top_zones = (
        df_ranking.groupby("zone_id")
        .agg({"mean_risk": "mean", "zone_name": "first"})
        .reset_index()
    )
    top_zones = top_zones.nlargest(TOP_N_ZONES, "mean_risk")

    print(f"\nTop {TOP_N_ZONES} zones (based on RCP8.5 August 2080):")
    for idx, row in top_zones.iterrows():
        print(f"  {row['zone_name']}: {row['mean_risk']:.6f}")
    print()

    # Plot each top zone
    for month in MONTHS.keys():
        for _, zone_row in top_zones.iterrows():
            zone_id = zone_row["zone_id"]
            zone_name = zone_row["zone_name"]

            output_path = (
                OUTPUT_DIR
                / f"timeseries_zone_{zone_id}_{month:02d}_{MONTHS[month]}.png"
            )
            plot_timeseries_with_ensemble(
                ensemble_data,
                month,
                title_suffix=f"Zone: {zone_name}",
                output_path=output_path,
                zone_filter=[zone_id],
                show_members=SHOW_INDIVIDUAL_MEMBERS,
            )

print()

# 3. SEASONAL COMPARISON (all scenarios on one plot per season)
print("Seasonal comparison plots:")
print("-" * 60)

for month in MONTHS.keys():
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    for scenario_key, df in ensemble_data.items():
        scenario_info = SCENARIOS[scenario_key]
        color = scenario_info["color"]
        label = scenario_info["label"]

        df_month = df[df["month"] == month]

        if len(df_month) == 0:
            continue

        # Ensemble statistics
        ensemble_stats = (
            df_month.groupby("year").agg({"mean_risk": ["mean", "std"]}).reset_index()
        )
        ensemble_stats.columns = ["year", "ensemble_mean", "ensemble_std"]

        years = ensemble_stats["year"].values
        mean = ensemble_stats["ensemble_mean"].values
        std = ensemble_stats["ensemble_std"].values

        # Plot
        ax.plot(
            years,
            mean,
            color=color,
            linewidth=2.5,
            label=label,
            marker="o",
            markersize=6,
            zorder=10,
        )
        ax.fill_between(years, mean - std, mean + std, color=color, alpha=0.2, zorder=5)

    ax.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax.set_ylabel("Relative Fire Risk", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{MONTHS[month]} - Scenario Comparison (Region-Wide)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc="best", fontsize=11, framealpha=0.9)

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"timeseries_comparison_{month:02d}_{MONTHS[month]}.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"  ✓ Saved: {output_path.name}")

print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)
print(f"Time series plots saved to: {OUTPUT_DIR}")
print("=" * 80)
