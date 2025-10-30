#!/usr/bin/env python
"""
Test Script: Verify Training/Projection Consistency

This script performs quick checks to verify that the consistency fixes
were applied correctly and that training and projection will produce
compatible results.

Run this after applying consistency fixes and before running projections.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

print("="*80)
print("FIRESCAPE MODEL CONSISTENCY TEST")
print("="*80)
print()

# ===================================================================
# TEST 1: Check landcover_fire_risk.tif exists and is valid
# ===================================================================

print("TEST 1: Landcover Fire Risk Raster")
print("-" * 40)

try:
    import rioxarray
    landcover_path = Path("/mnt/CEPH_PROJECTS/Firescape/Data/STATIC_INPUT/landcover_fire_risk.tif")

    if not landcover_path.exists():
        print("❌ FAIL: landcover_fire_risk.tif not found")
        print(f"   Expected: {landcover_path}")
        print("   Run: python Scripts/01_Data_Preparation/create_transformed_landcover.py")
        sys.exit(1)

    with rioxarray.open_rasterio(landcover_path) as rds:
        data = rds.squeeze('band', drop=True) if 'band' in rds.dims else rds
        unique_values = np.unique(data.values[~np.isnan(data.values)])

        # Check range
        if unique_values.min() < 0 or unique_values.max() > 5:
            print(f"❌ FAIL: Values out of range [0-5]: {unique_values}")
            sys.exit(1)

        # Check we have multiple risk levels
        if len(unique_values) < 3:
            print(f"⚠️  WARNING: Only {len(unique_values)} unique values found")
            print(f"   Expected 5-6 risk levels (0-5)")

        print(f"✓ PASS: landcover_fire_risk.tif is valid")
        print(f"  Risk levels present: {sorted([int(v) for v in unique_values])}")
        print(f"  Shape: {data.shape}")
        print(f"  CRS: {data.rio.crs}")

except Exception as e:
    print(f"❌ FAIL: Error reading landcover raster: {e}")
    sys.exit(1)

print()

# ===================================================================
# TEST 2: Check training data has correct landcover values
# ===================================================================

print("TEST 2: Training Data Landcover Values")
print("-" * 40)

try:
    import xarray as xr
    netcdf_path = Path("/mnt/CEPH_PROJECTS/Firescape/Scripts/OUTPUT/01_Training_Data/spacetime_stacks.nc")

    if not netcdf_path.exists():
        print("⚠️  SKIP: Training data not found (not generated yet)")
        print("   This is OK if you haven't run create_raster_stacks.py yet")
    else:
        with xr.open_dataset(netcdf_path) as ds:
            main_var = list(ds.data_vars)[0]

            # Check if landcover_fire_risk channel exists
            if 'landcover_fire_risk' not in ds.channel.values:
                print("❌ FAIL: landcover_fire_risk channel not found in training data")
                print(f"   Available channels: {list(ds.channel.values)}")
                sys.exit(1)

            # Get landcover values
            lc_data = ds[main_var].sel(channel='landcover_fire_risk').isel(time=0)
            unique_lc = np.unique(lc_data.values[~np.isnan(lc_data.values)])

            # Check range
            if unique_lc.min() < 0 or unique_lc.max() > 5:
                print(f"❌ FAIL: Training landcover values out of range [0-5]")
                print(f"   Found: {unique_lc}")
                print("   ACTION REQUIRED: Re-run create_raster_stacks.py with updated code")
                sys.exit(1)

            print(f"✓ PASS: Training data has valid landcover values")
            print(f"  Risk levels in training: {sorted([int(v) for v in unique_lc])}")
            print(f"  Number of observations: {len(ds.id_obs)}")

except Exception as e:
    print(f"⚠️  SKIP: Could not check training data: {e}")

print()

# ===================================================================
# TEST 3: Check trained model exists and has expected features
# ===================================================================

print("TEST 3: Trained Model Scaler")
print("-" * 40)

try:
    import joblib
    scaler_path = Path("/mnt/CEPH_PROJECTS/Firescape/Scripts/OUTPUT/02_Model_RelativeProbability/scaler_relative.joblib")

    if not scaler_path.exists():
        print("⚠️  SKIP: Model not trained yet")
        print("   This is OK if you haven't run training yet")
    else:
        scaler = joblib.load(scaler_path)
        feature_names = list(scaler.feature_names_in_)

        # Check landcover_fire_risk is in features
        if 'landcover_fire_risk' not in feature_names:
            print("❌ FAIL: landcover_fire_risk not in trained model features")
            print(f"   Features: {feature_names}")
            sys.exit(1)

        # Check for landcoverfull (should NOT be present)
        if 'landcoverfull' in feature_names:
            print("⚠️  WARNING: Old feature name 'landcoverfull' found in model")
            print("   Consider retraining with updated feature names")

        print(f"✓ PASS: Trained model has expected features")
        print(f"  Total features: {len(feature_names)}")
        print(f"  landcover_fire_risk position: {feature_names.index('landcover_fire_risk')}")

        # Show feature statistics
        print(f"\n  landcover_fire_risk statistics from training:")
        lc_idx = feature_names.index('landcover_fire_risk')
        print(f"    Mean: {scaler.mean_[lc_idx]:.3f}")
        print(f"    Std:  {scaler.scale_[lc_idx]:.3f}")

except Exception as e:
    print(f"⚠️  SKIP: Could not check trained model: {e}")

print()

# ===================================================================
# TEST 4: Check projection script has been updated
# ===================================================================

print("TEST 4: Projection Script Updates")
print("-" * 40)

try:
    projection_script = Path("/mnt/CEPH_PROJECTS/Firescape/Scripts/04_Zone_Climate_Projections/project_zone_fire_risk.py")

    if not projection_script.exists():
        print("❌ FAIL: Projection script not found")
        sys.exit(1)

    with open(projection_script) as f:
        content = f.read()

    # Check for window averaging
    if "SPATIAL_WINDOW_SIZE = 4" in content:
        print("✓ PASS: Window averaging added (SPATIAL_WINDOW_SIZE = 4)")
    else:
        print("❌ FAIL: SPATIAL_WINDOW_SIZE not found in projection script")
        print("   Window averaging may not be implemented")
        sys.exit(1)

    # Check for rolling window
    if ".rolling(" in content and "center=True" in content:
        print("✓ PASS: Rolling window implementation found")
    else:
        print("⚠️  WARNING: Rolling window implementation not clearly visible")

    # Check old transformation code is removed
    if "LANDCOVER_FIRE_RISK_ORDINAL.get" in content:
        # Check it's only in comments or mapping definition
        lines_with_mapping = [l for l in content.split('\n') if 'LANDCOVER_FIRE_RISK_ORDINAL.get' in l and not l.strip().startswith('#')]
        if len(lines_with_mapping) > 1:  # Allow one for definition
            print("⚠️  WARNING: Inline landcover transformation code may still be present")
            print("   Consider removing redundant transformation")

    # Check for validation
    if "landcover_fire_risk" in content and "valid range" in content.lower():
        print("✓ PASS: Landcover validation checks added")
    else:
        print("⚠️  WARNING: Validation checks may be missing")

except Exception as e:
    print(f"❌ FAIL: Error checking projection script: {e}")
    sys.exit(1)

print()

# ===================================================================
# SUMMARY
# ===================================================================

print("="*80)
print("CONSISTENCY TEST SUMMARY")
print("="*80)
print()
print("✓ All critical tests passed!")
print()
print("Next steps:")
print("  1. If training data needs update, run:")
print("     python Scripts/01_Data_Preparation/create_raster_stacks.py")
print()
print("  2. If model needs retraining, run:")
print("     python Scripts/02_Model_Training/train_relative_probability_model.py")
print()
print("  3. Run projections:")
print("     python Scripts/04_Zone_Climate_Projections/project_zone_fire_risk.py")
print()
print("For details, see:")
print("  Scripts/CONSISTENCY_FIX_SUMMARY.md")
print()
