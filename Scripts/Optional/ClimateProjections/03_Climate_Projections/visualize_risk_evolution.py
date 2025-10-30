"""
Visualization Module for Wildfire Risk Evolution

Creates spatial and temporal plots showing:
1. Risk evolution over time (temporal trends)
2. Spatial risk maps for different scenarios
3. Regional comparisons
4. Scenario comparisons (RCP 4.5 vs RCP 8.5)
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from datetime import datetime
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from config_scenarios import SCENARIOS, REGIONS, get_scenario_output_dir


# Custom colormap for wildfire risk (white -> yellow -> orange -> red)
FIRE_RISK_CMAP = LinearSegmentedColormap.from_list(
    'fire_risk',
    ['#f7fbff', '#fee391', '#fc8d59', '#d7301f', '#7f0000']
)


def load_predictions(scenario_name: str) -> pd.DataFrame:
    """Load predictions for a scenario."""
    output_dir = get_scenario_output_dir(scenario_name)
    pred_path = os.path.join(output_dir, f"predictions_{scenario_name}.csv")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    df = pd.read_csv(pred_path, parse_dates=['date'])
    return df


def plot_temporal_evolution_by_scenario(
    scenarios: List[str],
    output_path: str,
    aggregate_by: str = 'month'
):
    """
    Plot temporal evolution of wildfire risk across multiple scenarios.

    Args:
        scenarios: List of scenario names
        output_path: Path to save plot
        aggregate_by: 'month' or 'year'
    """
    print(f"\n📈 Creating temporal evolution plot ({aggregate_by})...")

    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    # Color scheme
    colors = {
        'historical': 'black',
        'rcp45_2030': '#3498db',
        'rcp45_2050': '#2874a6',
        'rcp45_2070': '#1a5490',
        'rcp85_2030': '#e67e22',
        'rcp85_2050': '#d35400',
        'rcp85_2070': '#a04000'
    }

    # --- PLOT 1: Mean risk over time ---
    ax1 = axes[0]

    for scenario_name in scenarios:
        try:
            df = load_predictions(scenario_name)

            if aggregate_by == 'month':
                # Group by year-month
                df['year_month'] = df['date'].dt.to_period('M')
                temporal_agg = df.groupby('year_month').agg({
                    'mean_risk': 'mean',
                    'std_risk': 'mean'
                }).reset_index()
                temporal_agg['date'] = temporal_agg['year_month'].dt.to_timestamp()

            elif aggregate_by == 'year':
                # Group by year
                df['year'] = df['date'].dt.year
                temporal_agg = df.groupby('year').agg({
                    'mean_risk': 'mean',
                    'std_risk': 'mean'
                }).reset_index()
                temporal_agg['date'] = pd.to_datetime(temporal_agg['year'], format='%Y')

            # Plot line with confidence interval
            color = colors.get(scenario_name, 'gray')
            ax1.plot(temporal_agg['date'], temporal_agg['mean_risk'],
                    label=scenario_name, color=color, linewidth=2.5, alpha=0.9)

            # Add confidence band (mean ± std)
            ax1.fill_between(
                temporal_agg['date'],
                temporal_agg['mean_risk'] - temporal_agg['std_risk'],
                temporal_agg['mean_risk'] + temporal_agg['std_risk'],
                color=color, alpha=0.2
            )

        except FileNotFoundError:
            print(f"   ⚠️  Skipping {scenario_name} - predictions not found")

    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Mean Wildfire Risk (Relative Probability)', fontsize=12)
    ax1.set_title('Temporal Evolution of Wildfire Risk Across Climate Scenarios', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Format x-axis
    if aggregate_by == 'month':
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.YearLocator(5))
    else:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_major_locator(mdates.YearLocator(10))

    # --- PLOT 2: High-risk days (threshold exceedance) ---
    ax2 = axes[1]
    threshold = 0.7  # Define "high risk" threshold

    for scenario_name in scenarios:
        try:
            df = load_predictions(scenario_name)

            # Count high-risk days per year
            df['year'] = df['date'].dt.year
            df['high_risk'] = df['mean_risk'] > threshold

            yearly_high_risk = df.groupby('year')['high_risk'].sum().reset_index()
            yearly_high_risk.columns = ['year', 'high_risk_days']
            yearly_high_risk['date'] = pd.to_datetime(yearly_high_risk['year'], format='%Y')

            # Plot
            color = colors.get(scenario_name, 'gray')
            ax2.plot(yearly_high_risk['date'], yearly_high_risk['high_risk_days'],
                    label=scenario_name, color=color, linewidth=2.5, marker='o', alpha=0.9)

        except FileNotFoundError:
            pass

    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel(f'Number of High-Risk Days (risk > {threshold})', fontsize=12)
    ax2.set_title(f'Frequency of High-Risk Days Across Scenarios', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved: {output_path}")


def plot_spatial_risk_maps(
    scenario_name: str,
    time_slices: List[Tuple[int, int]],  # [(year, month), ...]
    output_path: str
):
    """
    Create spatial risk maps for specific time slices.

    Args:
        scenario_name: Scenario name
        time_slices: List of (year, month) tuples to visualize
        output_path: Path to save plot
    """
    print(f"\n🗺️  Creating spatial risk maps for {scenario_name}...")

    df = load_predictions(scenario_name)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    n_slices = len(time_slices)
    fig, axes = plt.subplots(1, n_slices, figsize=(6 * n_slices, 6))

    if n_slices == 1:
        axes = [axes]

    for ax, (year, month) in zip(axes, time_slices):
        # Filter data
        df_slice = df[(df['year'] == year) & (df['month'] == month)]

        if len(df_slice) == 0:
            print(f"   ⚠️  No data for {year}-{month:02d}")
            ax.text(0.5, 0.5, f'No data\n{year}-{month:02d}',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            continue

        # Create scatter plot (x, y, risk)
        scatter = ax.scatter(
            df_slice['x'], df_slice['y'], c=df_slice['mean_risk'],
            cmap=FIRE_RISK_CMAP, s=20, vmin=0, vmax=1, alpha=0.8
        )

        ax.set_xlabel('Easting (m)', fontsize=10)
        ax.set_ylabel('Northing (m)', fontsize=10)
        ax.set_title(f'{year}-{month:02d}', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Wildfire Risk', fontsize=10)

    plt.suptitle(f'Spatial Wildfire Risk Maps - {scenario_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved: {output_path}")


def plot_regional_comparison(
    scenario_name: str,
    region_bounds: Dict[str, tuple],  # {region_name: (xmin, ymin, xmax, ymax)}
    output_path: str
):
    """
    Compare risk evolution across different regions.

    Args:
        scenario_name: Scenario name
        region_bounds: Dictionary of region bounds in UTM
        output_path: Path to save plot
    """
    print(f"\n📊 Creating regional comparison for {scenario_name}...")

    df = load_predictions(scenario_name)
    df['year_month'] = df['date'].dt.to_period('M')

    fig, axes = plt.subplots(len(region_bounds), 1, figsize=(16, 4 * len(region_bounds)))

    if len(region_bounds) == 1:
        axes = [axes]

    for ax, (region_name, bounds) in zip(axes, region_bounds.items()):
        # Filter by region
        xmin, ymin, xmax, ymax = bounds
        df_region = df[
            (df['x'] >= xmin) & (df['x'] <= xmax) &
            (df['y'] >= ymin) & (df['y'] <= ymax)
        ]

        if len(df_region) == 0:
            print(f"   ⚠️  No data for region {region_name}")
            ax.text(0.5, 0.5, f'No data for {region_name}',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            continue

        # Aggregate by month
        monthly_agg = df_region.groupby('year_month').agg({
            'mean_risk': ['mean', 'std'],
            'std_risk': 'mean'
        }).reset_index()
        monthly_agg.columns = ['year_month', 'mean_risk', 'std_risk', 'uncertainty']
        monthly_agg['date'] = monthly_agg['year_month'].dt.to_timestamp()

        # Plot
        ax.plot(monthly_agg['date'], monthly_agg['mean_risk'],
               color='#e74c3c', linewidth=2.5, label='Mean Risk')
        ax.fill_between(
            monthly_agg['date'],
            monthly_agg['mean_risk'] - monthly_agg['std_risk'],
            monthly_agg['mean_risk'] + monthly_agg['std_risk'],
            color='#e74c3c', alpha=0.2, label='Spatial variability'
        )

        ax.set_ylabel('Wildfire Risk', fontsize=11)
        ax.set_title(f'{region_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    axes[-1].set_xlabel('Date', fontsize=12)
    plt.suptitle(f'Regional Wildfire Risk Evolution - {scenario_name}',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved: {output_path}")


def plot_scenario_comparison_heatmap(
    scenarios: List[str],
    output_path: str
):
    """
    Create a heatmap comparing mean risk across scenarios and months.

    Args:
        scenarios: List of scenario names
        output_path: Path to save plot
    """
    print(f"\n🔥 Creating scenario comparison heatmap...")

    # Collect data
    data = []
    for scenario_name in scenarios:
        try:
            df = load_predictions(scenario_name)
            df['month'] = df['date'].dt.month

            monthly_mean = df.groupby('month')['mean_risk'].mean()

            for month, risk in monthly_mean.items():
                data.append({
                    'scenario': scenario_name,
                    'month': month,
                    'mean_risk': risk
                })

        except FileNotFoundError:
            print(f"   ⚠️  Skipping {scenario_name}")

    df_pivot = pd.DataFrame(data).pivot(index='scenario', columns='month', values='mean_risk')

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(df_pivot, cmap=FIRE_RISK_CMAP, annot=True, fmt='.3f',
               linewidths=1, cbar_kws={'label': 'Mean Wildfire Risk'},
               vmin=0, vmax=1, ax=ax)

    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Scenario', fontsize=12)
    ax.set_title('Wildfire Risk by Scenario and Month', fontsize=14, fontweight='bold')

    # Month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticklabels([month_labels[int(m)-1] for m in df_pivot.columns], rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved: {output_path}")


def create_summary_statistics_table(
    scenarios: List[str],
    output_path: str
):
    """
    Create a summary table comparing key statistics across scenarios.

    Args:
        scenarios: List of scenario names
        output_path: Path to save CSV
    """
    print(f"\n📋 Creating summary statistics table...")

    summary_data = []

    for scenario_name in scenarios:
        try:
            df = load_predictions(scenario_name)

            stats = {
                'scenario': scenario_name,
                'mean_risk': df['mean_risk'].mean(),
                'median_risk': df['mean_risk'].median(),
                'std_risk': df['mean_risk'].std(),
                'max_risk': df['mean_risk'].max(),
                'min_risk': df['mean_risk'].min(),
                'pct_high_risk': (df['mean_risk'] > 0.7).mean() * 100,  # % above threshold
                'mean_uncertainty': df['std_risk'].mean(),
                'n_observations': len(df)
            }

            summary_data.append(stats)

        except FileNotFoundError:
            print(f"   ⚠️  Skipping {scenario_name}")

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_path, index=False)

    print(f"   ✓ Saved: {output_path}")
    print("\nSummary Statistics:")
    print(df_summary.to_string(index=False))


def main():
    """Main execution function."""
    print("="*80)
    print("WILDFIRE RISK EVOLUTION VISUALIZATION")
    print("="*80)

    # Define output directory
    viz_output_dir = "/mnt/CEPH_PROJECTS/Firescape/Scripts/OUTPUT/03_Climate_Projections/Visualizations"
    os.makedirs(viz_output_dir, exist_ok=True)

    # Get all non-historical scenarios
    scenario_names = [s.name for s in SCENARIOS if s.name != 'historical']

    # 1. Temporal evolution across all scenarios
    plot_temporal_evolution_by_scenario(
        scenarios=scenario_names,
        output_path=os.path.join(viz_output_dir, "temporal_evolution_all_scenarios.png"),
        aggregate_by='year'
    )

    # 2. Spatial risk maps for RCP 8.5 2070 (worst case)
    plot_spatial_risk_maps(
        scenario_name='rcp85_2070',
        time_slices=[(2070, 7), (2070, 8)],  # July and August 2070
        output_path=os.path.join(viz_output_dir, "spatial_risk_maps_rcp85_2070.png")
    )

    # 3. Regional comparison for RCP 8.5 2050
    # Convert REGIONS to UTM bounds (simplified - you may need actual conversion)
    # For now, using full province
    plot_regional_comparison(
        scenario_name='rcp85_2050',
        region_bounds={'Full Province': (600000, 5100000, 750000, 5250000)},  # Approximate UTM bounds
        output_path=os.path.join(viz_output_dir, "regional_comparison_rcp85_2050.png")
    )

    # 4. Scenario comparison heatmap
    plot_scenario_comparison_heatmap(
        scenarios=scenario_names,
        output_path=os.path.join(viz_output_dir, "scenario_comparison_heatmap.png")
    )

    # 5. Summary statistics table
    create_summary_statistics_table(
        scenarios=scenario_names,
        output_path=os.path.join(viz_output_dir, "summary_statistics.csv")
    )

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print(f"Output directory: {viz_output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
