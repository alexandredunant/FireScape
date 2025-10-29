#!/usr/bin/env python
"""
Create Pre-Transformed Landcover Fire Risk Raster

This script creates a landcover_fire_risk.tif file by transforming the raw
Corine Land Cover codes in landcoverfull.tif to ordinal fire risk values (0-5).

Purpose:
- Eliminate redundant transformation in multiple scripts
- Ensure consistency across all analyses (training, projection, etc.)
- Improve performance by doing transformation once
- Single source of truth for landcover fire risk mapping

Input:  landcoverfull.tif (raw 3-digit Corine codes)
Output: landcover_fire_risk.tif (ordinal fire risk values 0-5)

Run this script ONCE after updating the landcover data or mapping dictionary.
"""

import rioxarray
import numpy as np
from pathlib import Path

print("="*80)
print("CREATE PRE-TRANSFORMED LANDCOVER FIRE RISK RASTER")
print("="*80)
print()

# ===================================================================
# CONFIGURATION
# ===================================================================

# Corine Land Cover → Fire Risk Ordinal Mapping
# Based on actual 3-digit Corine codes in Bolzano Province dataset
# This mapping is the canonical reference used across all scripts
LANDCOVER_FIRE_RISK_ORDINAL = {
    # 0 = No fire risk (water, snow/ice, wetlands)
    335: 0, 511: 0, 512: 0, 411: 0, 412: 0,

    # 1 = Very low fire risk (urban, bare soil)
    111: 1, 112: 1, 121: 1, 122: 1, 124: 1, 131: 1, 133: 1, 142: 1, 331: 1, 332: 1,

    # 2 = Low fire risk (agriculture, managed land)
    211: 2, 221: 2, 222: 2, 231: 2, 242: 2, 243: 2,

    # 3 = Moderate fire risk (grassland, broadleaf forest)
    311: 3, 321: 3, 333: 3,

    # 4 = High fire risk (shrubland, mixed forest)
    313: 4, 322: 4, 324: 4,

    # 5 = Very high fire risk (coniferous forest)
    312: 5
}

# Paths
BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape")
LANDCOVER_DIR = BASE_DIR / "Data/02_Land_Cover"
STATIC_RASTER_DIR = BASE_DIR / "Data/STATIC_INPUT"

# Use the raster with actual 3-digit Corine codes (not the simplified landcoverfull.tif)
INPUT_RASTER = LANDCOVER_DIR / "CorineLandCover_polygon.tif"
OUTPUT_RASTER = STATIC_RASTER_DIR / "landcover_fire_risk.tif"

# ===================================================================
# VALIDATION
# ===================================================================

print("Validating configuration...")
print(f"  Input:  {INPUT_RASTER}")
print(f"  Output: {OUTPUT_RASTER}")
print()

if not INPUT_RASTER.exists():
    raise FileNotFoundError(f"Input raster not found: {INPUT_RASTER}")

if OUTPUT_RASTER.exists():
    print(f"⚠️  WARNING: Output file already exists: {OUTPUT_RASTER}")
    response = input("    Overwrite? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Aborted.")
        exit(0)
    print()

# ===================================================================
# LOAD AND TRANSFORM
# ===================================================================

print("Loading landcover raster...")
with rioxarray.open_rasterio(INPUT_RASTER) as rds:
    print(f"✓ Loaded raster")
    print(f"  Shape: {rds.shape}")
    print(f"  CRS: {rds.rio.crs}")
    print(f"  Bounds: {rds.rio.bounds()}")
    print()

    # Get the data array (squeeze band dimension if present)
    if 'band' in rds.dims:
        data = rds.squeeze('band', drop=True)
    else:
        data = rds

    # Get unique Corine codes present in the raster
    print("Analyzing input data...")
    unique_codes = np.unique(data.values[~np.isnan(data.values)])
    print(f"✓ Found {len(unique_codes)} unique Corine codes")
    print(f"  Codes: {sorted([int(c) for c in unique_codes])}")
    print()

    # Check which codes are not in the mapping
    unmapped_codes = [int(c) for c in unique_codes if int(c) not in LANDCOVER_FIRE_RISK_ORDINAL]
    if unmapped_codes:
        print(f"⚠️  WARNING: {len(unmapped_codes)} codes not in mapping (will use default value 2):")
        print(f"  Unmapped codes: {unmapped_codes}")
        print()

    # Transform to fire risk ordinal values
    print("Transforming Corine codes to fire risk ordinal values...")
    print("  Mapping: Corine codes → Fire risk (0-5)")
    print("  Default value for unmapped/NaN: 2 (moderate risk)")
    print()

    # Vectorized transformation
    def map_to_fire_risk(x):
        """Map Corine code to fire risk ordinal (0-5)."""
        if np.isnan(x):
            return 0  # NaN → no risk
        return LANDCOVER_FIRE_RISK_ORDINAL.get(int(x), 2)  # Default: moderate risk

    transformed_values = np.vectorize(map_to_fire_risk)(data.values)

    # Create new data array with transformed values
    transformed = data.copy(deep=True)
    transformed.values = transformed_values

    # Verify transformation
    print("Verifying transformation...")
    unique_risk_values = np.unique(transformed.values[~np.isnan(transformed.values)])
    print(f"✓ Output contains {len(unique_risk_values)} unique fire risk values:")
    print(f"  Risk values: {sorted([int(v) for v in unique_risk_values])}")

    # Count pixels per risk level
    print()
    print("  Fire risk distribution:")
    for risk_val in sorted(unique_risk_values):
        count = np.sum(transformed.values == risk_val)
        percentage = (count / np.sum(~np.isnan(transformed.values))) * 100
        risk_labels = {
            0: "No risk",
            1: "Very low",
            2: "Low",
            3: "Moderate",
            4: "High",
            5: "Very high"
        }
        label = risk_labels.get(int(risk_val), "Unknown")
        print(f"    {int(risk_val)} ({label:10s}): {count:,} pixels ({percentage:.1f}%)")
    print()

    # Check for values outside expected range
    invalid_mask = (transformed.values < 0) | (transformed.values > 5)
    invalid_count = np.sum(invalid_mask & ~np.isnan(transformed.values))
    if invalid_count > 0:
        print(f"❌ ERROR: Found {invalid_count} pixels with values outside [0-5] range!")
        print("   Transformation failed. Check mapping dictionary.")
        exit(1)

    # Save transformed raster
    print(f"Saving transformed raster to: {OUTPUT_RASTER}")
    transformed.rio.to_raster(OUTPUT_RASTER, dtype='int16', compress='lzw')
    print(f"✓ Saved successfully")
    print()

# ===================================================================
# VALIDATION OF OUTPUT FILE
# ===================================================================

print("Validating output file...")
with rioxarray.open_rasterio(OUTPUT_RASTER) as validation:
    val_data = validation.squeeze('band', drop=True) if 'band' in validation.dims else validation

    # Check CRS preserved
    if val_data.rio.crs != data.rio.crs:
        print("❌ ERROR: CRS mismatch!")
    else:
        print(f"✓ CRS preserved: {val_data.rio.crs}")

    # Check bounds preserved
    input_bounds = data.rio.bounds()
    output_bounds = val_data.rio.bounds()
    if not np.allclose([input_bounds], [output_bounds], rtol=1e-5):
        print("❌ ERROR: Bounds changed!")
    else:
        print(f"✓ Bounds preserved")

    # Check shape preserved
    if val_data.shape != data.shape:
        print(f"❌ ERROR: Shape changed! {data.shape} → {val_data.shape}")
    else:
        print(f"✓ Shape preserved: {val_data.shape}")

    # Check value range
    val_unique = np.unique(val_data.values[~np.isnan(val_data.values)])
    if np.all((val_unique >= 0) & (val_unique <= 5)):
        print(f"✓ All values in valid range [0-5]")
    else:
        print(f"❌ ERROR: Invalid values found: {val_unique}")

print()
print("="*80)
print("TRANSFORMATION COMPLETE")
print("="*80)
print()
print("✓ landcover_fire_risk.tif created successfully")
print()
print("Next steps:")
print("  1. Update scripts to load landcover_fire_risk.tif instead of landcoverfull.tif")
print("  2. Remove transformation code from individual scripts")
print("  3. Re-run training data preparation if needed")
print()
print("Files to update:")
print("  - Scripts/01_Data_Preparation/create_raster_stacks.py")
print("  - Scripts/04_Zone_Climate_Projections/project_zone_fire_risk.py")
print("  - Scripts/05_Lightning_Comparison/01_Data_Preparation/create_raster_stacks_with_lightning.py")
print()
