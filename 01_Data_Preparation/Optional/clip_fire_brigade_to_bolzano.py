#!/usr/bin/env python
"""
Clip Fire Brigade Responsibility Areas to Bolzano Province Boundary

This script clips the fire brigade shapefile to only include zones within
the Bolzano province boundary.
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime

print("="*80)
print("CLIPPING FIRE BRIGADE ZONES TO BOLZANO PROVINCE")
print("="*80)

# ===================================================================
# FILE PATHS
# ===================================================================

# Input files
BOLZANO_BOUNDARY = "/mnt/CEPH_PROJECTS/Firescape/Data/06_Administrative_Boundaries/BOLZANO_REGION_UTM32.gpkg"
FIRE_BRIGADE_SHP = "/mnt/CEPH_PROJECTS/Firescape/Data/06_Administrative_Boundaries/FME_12060556_1748518737281_26580/DownloadService/FireBrigade-ResponsibilityAreas_polygon.shp"

# Output files
OUTPUT_DIR = "/mnt/CEPH_PROJECTS/Firescape/Data/06_Administrative_Boundaries/Processed/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CLIPPED_SHP = os.path.join(OUTPUT_DIR, "FireBrigade_ResponsibilityAreas_Bolzano_clipped.shp")
OUTPUT_CLIPPED_GPKG = os.path.join(OUTPUT_DIR, "FireBrigade_ResponsibilityAreas_Bolzano_clipped.gpkg")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "fire_brigade_clipping_comparison.png")

# ===================================================================
# LOAD DATA
# ===================================================================

print("\nLoading Bolzano province boundary...")
gdf_bolzano = gpd.read_file(BOLZANO_BOUNDARY)
print(f"✓ Loaded Bolzano boundary")
print(f"  Records: {len(gdf_bolzano)}")
print(f"  CRS: {gdf_bolzano.crs}")
print(f"  Area: {gdf_bolzano.geometry.area.sum() / 1e6:.2f} km²")

print("\nLoading fire brigade responsibility areas...")
gdf_fire_brigade = gpd.read_file(FIRE_BRIGADE_SHP)
print(f"✓ Loaded fire brigade zones")
print(f"  Records: {len(gdf_fire_brigade)}")
print(f"  CRS: {gdf_fire_brigade.crs}")
print(f"  Total area: {gdf_fire_brigade.geometry.area.sum() / 1e6:.2f} km²")

# ===================================================================
# REPROJECT IF NECESSARY
# ===================================================================

# Check if CRS match
if gdf_bolzano.crs != gdf_fire_brigade.crs:
    print(f"\nReprojecting fire brigade zones from {gdf_fire_brigade.crs} to {gdf_bolzano.crs}...")
    gdf_fire_brigade = gdf_fire_brigade.to_crs(gdf_bolzano.crs)
    print("✓ Reprojection complete")
else:
    print("\n✓ CRS already match - no reprojection needed")

# ===================================================================
# PERFORM CLIPPING
# ===================================================================

print("\nPerforming spatial clip operation...")
print("(This may take a minute...)")

# Method 1: Using gpd.clip (recommended)
gdf_clipped = gpd.clip(gdf_fire_brigade, gdf_bolzano)

print(f"\n✓ Clipping complete!")
print(f"\nResults:")
print(f"  Original zones: {len(gdf_fire_brigade)}")
print(f"  Clipped zones: {len(gdf_clipped)}")
print(f"  Zones removed: {len(gdf_fire_brigade) - len(gdf_clipped)}")
print(f"  Original total area: {gdf_fire_brigade.geometry.area.sum() / 1e6:.2f} km²")
print(f"  Clipped total area: {gdf_clipped.geometry.area.sum() / 1e6:.2f} km²")
print(f"  Area reduction: {(1 - gdf_clipped.geometry.area.sum() / gdf_fire_brigade.geometry.area.sum()) * 100:.1f}%")

# ===================================================================
# QUALITY CHECK
# ===================================================================

print("\nQuality check:")

# Check for empty geometries
empty_geoms = gdf_clipped.geometry.is_empty.sum()
print(f"  Empty geometries: {empty_geoms}")

# Check for invalid geometries
invalid_geoms = (~gdf_clipped.geometry.is_valid).sum()
print(f"  Invalid geometries: {invalid_geoms}")

# Fix invalid geometries if any
if invalid_geoms > 0:
    print("  Fixing invalid geometries...")
    gdf_clipped['geometry'] = gdf_clipped.geometry.buffer(0)
    invalid_after_fix = (~gdf_clipped.geometry.is_valid).sum()
    print(f"  Invalid geometries after fix: {invalid_after_fix}")

# Remove empty geometries if any
if empty_geoms > 0:
    print("  Removing empty geometries...")
    gdf_clipped = gdf_clipped[~gdf_clipped.geometry.is_empty].copy()
    print(f"  Final zone count: {len(gdf_clipped)}")

# Check that clipped zones are within Bolzano boundary
print("\nVerifying all clipped zones are within Bolzano boundary...")
bolzano_geom = gdf_bolzano.geometry.union_all()
zones_within = gdf_clipped.geometry.within(bolzano_geom).sum()
zones_intersect = gdf_clipped.geometry.intersects(bolzano_geom).sum()
print(f"  Zones completely within boundary: {zones_within}")
print(f"  Zones intersecting boundary: {zones_intersect}")
print(f"  Coverage: {zones_intersect / len(gdf_clipped) * 100:.1f}%")

# ===================================================================
# SAVE OUTPUTS
# ===================================================================

print(f"\nSaving clipped shapefile...")
gdf_clipped.to_file(OUTPUT_CLIPPED_SHP)
print(f"✓ Saved: {OUTPUT_CLIPPED_SHP}")

print(f"\nSaving clipped geopackage...")
gdf_clipped.to_file(OUTPUT_CLIPPED_GPKG, driver='GPKG')
print(f"✓ Saved: {OUTPUT_CLIPPED_GPKG}")

# ===================================================================
# CREATE COMPARISON VISUALIZATION
# ===================================================================

print(f"\nCreating comparison visualization...")

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Plot 1: Original fire brigade zones + Bolzano boundary
ax1 = axes[0]
gdf_fire_brigade.plot(ax=ax1, color='lightblue', edgecolor='black', linewidth=0.3, alpha=0.5)
gdf_bolzano.boundary.plot(ax=ax1, color='red', linewidth=3, label='Bolzano Province')
ax1.set_title('Original Fire Brigade Zones\n(All zones + Bolzano boundary in red)',
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Easting (m)', fontsize=12)
ax1.set_ylabel('Northing (m)', fontsize=12)
ax1.legend(loc='upper right', fontsize=12)
ax1.grid(True, alpha=0.3)

# Add text box with statistics
textstr_orig = f'Total zones: {len(gdf_fire_brigade)}\nTotal area: {gdf_fire_brigade.geometry.area.sum() / 1e6:.0f} km²'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.02, 0.98, textstr_orig, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

# Plot 2: Clipped fire brigade zones
ax2 = axes[1]
gdf_clipped.plot(ax=ax2, color='lightgreen', edgecolor='black', linewidth=0.5, alpha=0.7)
gdf_bolzano.boundary.plot(ax=ax2, color='red', linewidth=3, label='Bolzano Province')
ax2.set_title('Clipped Fire Brigade Zones\n(Only zones within Bolzano)',
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Easting (m)', fontsize=12)
ax2.set_ylabel('Northing (m)', fontsize=12)
ax2.legend(loc='upper right', fontsize=12)
ax2.grid(True, alpha=0.3)

# Add text box with statistics
textstr_clip = f'Clipped zones: {len(gdf_clipped)}\nClipped area: {gdf_clipped.geometry.area.sum() / 1e6:.0f} km²'
ax2.text(0.02, 0.98, textstr_clip, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization: {OUTPUT_PLOT}")
plt.close()

# ===================================================================
# CREATE DETAILED MAP OF CLIPPED ZONES
# ===================================================================

print(f"\nCreating detailed map of clipped zones...")

fig, ax = plt.subplots(1, 1, figsize=(16, 14))

# Plot clipped zones with zone IDs
gdf_clipped.plot(ax=ax, column='ID', cmap='tab20', edgecolor='black',
                 linewidth=0.5, alpha=0.6, legend=False)

# Overlay Bolzano boundary
gdf_bolzano.boundary.plot(ax=ax, color='red', linewidth=2.5, label='Bolzano Province')

# Add zone labels for zones with names
if 'PLACE_IT' in gdf_clipped.columns:
    # Label only larger zones to avoid clutter
    gdf_clipped['area_km2'] = gdf_clipped.geometry.area / 1e6
    large_zones = gdf_clipped.nlargest(30, 'area_km2')  # Top 30 largest zones

    for idx, row in large_zones.iterrows():
        if row['PLACE_IT'] and str(row['PLACE_IT']) != 'nan':
            centroid = row.geometry.centroid
            ax.annotate(text=row['PLACE_IT'], xy=(centroid.x, centroid.y),
                       ha='center', fontsize=7, color='black',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax.set_title(f'Fire Brigade Responsibility Zones - Bolzano Province\n({len(gdf_clipped)} zones)',
             fontsize=16, fontweight='bold')
ax.set_xlabel('Easting (m)', fontsize=12)
ax.set_ylabel('Northing (m)', fontsize=12)
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3)

# Add north arrow and scale (simple version)
ax.text(0.95, 0.05, 'N ↑', transform=ax.transAxes, fontsize=16,
        fontweight='bold', ha='right')

plt.tight_layout()
OUTPUT_DETAILED = os.path.join(OUTPUT_DIR, "fire_brigade_zones_bolzano_detailed.png")
plt.savefig(OUTPUT_DETAILED, dpi=300, bbox_inches='tight')
print(f"✓ Saved detailed map: {OUTPUT_DETAILED}")
plt.close()

# ===================================================================
# EXPORT SUMMARY STATISTICS
# ===================================================================

print(f"\nExporting summary statistics...")

# Create summary dataframe
summary_stats = {
    'Metric': [
        'Original zone count',
        'Clipped zone count',
        'Zones removed',
        'Removal percentage',
        'Original total area (km²)',
        'Clipped total area (km²)',
        'Area reduction (%)',
        'Bolzano area (km²)',
        'Coverage of Bolzano (%)',
        'Processing date'
    ],
    'Value': [
        len(gdf_fire_brigade),
        len(gdf_clipped),
        len(gdf_fire_brigade) - len(gdf_clipped),
        f"{(len(gdf_fire_brigade) - len(gdf_clipped)) / len(gdf_fire_brigade) * 100:.1f}",
        f"{gdf_fire_brigade.geometry.area.sum() / 1e6:.2f}",
        f"{gdf_clipped.geometry.area.sum() / 1e6:.2f}",
        f"{(1 - gdf_clipped.geometry.area.sum() / gdf_fire_brigade.geometry.area.sum()) * 100:.1f}",
        f"{gdf_bolzano.geometry.area.sum() / 1e6:.2f}",
        f"{gdf_clipped.geometry.area.sum() / gdf_bolzano.geometry.area.sum() * 100:.1f}",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ]
}

import pandas as pd
df_summary = pd.DataFrame(summary_stats)
summary_csv = os.path.join(OUTPUT_DIR, "clipping_summary_statistics.csv")
df_summary.to_csv(summary_csv, index=False)
print(f"✓ Saved summary: {summary_csv}")

print(f"\nSummary Statistics:")
print(df_summary.to_string(index=False))

# ===================================================================
# FINAL SUMMARY
# ===================================================================

print("\n" + "="*80)
print("CLIPPING COMPLETE!")
print("="*80)

print(f"\nInput files:")
print(f"  • Bolzano boundary: {BOLZANO_BOUNDARY}")
print(f"  • Fire brigade zones: {FIRE_BRIGADE_SHP}")

print(f"\nOutput files created:")
print(f"  • Shapefile: {OUTPUT_CLIPPED_SHP}")
print(f"  • GeoPackage: {OUTPUT_CLIPPED_GPKG}")
print(f"  • Comparison plot: {OUTPUT_PLOT}")
print(f"  • Detailed map: {OUTPUT_DETAILED}")
print(f"  • Statistics CSV: {summary_csv}")

print(f"\nKey results:")
print(f"  • Original zones: {len(gdf_fire_brigade)}")
print(f"  • Clipped zones: {len(gdf_clipped)}")
print(f"  • Zones kept: {len(gdf_clipped) / len(gdf_fire_brigade) * 100:.1f}%")

print("\n" + "="*80)
print("You can now use the clipped shapefile in your analysis:")
print(f"  gdf = gpd.read_file('{OUTPUT_CLIPPED_GPKG}')")
print("="*80)
