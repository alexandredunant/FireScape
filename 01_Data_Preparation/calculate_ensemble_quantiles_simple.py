#!/usr/bin/env python
"""
Calculate Ensemble Quantiles from CORDEX Climate Projections (Simplified Version)

This version uses a simpler, more memory-efficient approach without Dask distributed.
It processes data year by year to avoid memory issues.
"""

import xarray as xr
import numpy as np
from pathlib import Path
import glob
import warnings
from tqdm import tqdm

# --- Suppress warnings for cleaner output ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===================================================================
# CONFIGURATION
# ===================================================================

# Source directory for the raw CORDEX ensemble data
SOURCE_BASE_DIR = Path("/mnt/CEPH_PROJECTS/FACT_CLIMAX/CORDEX-Adjust/QDM")

# Destination directory for the processed quantile files
DEST_BASE_DIR = Path(
    "/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/climate_projections_ensemble_quantiles"
)

# Define the variables, scenarios, and quantiles to process
VARIABLES_TO_PROCESS = {
    "tas": "temperature",
    "pr": "precipitation",
}
SCENARIOS_TO_PROCESS = ["rcp45", "rcp85"]
QUANTILES_TO_CALCULATE = [0.25, 0.50, 0.75, 0.99]

# Use Dask for chunked computation but not distributed
import dask

dask.config.set(scheduler="threads", num_workers=4)


# ===================================================================
# MAIN PROCESSING FUNCTION
# ===================================================================


def process_variable_scenario(var_code, scenario):
    """
    Processes all ensemble files for a given variable and scenario to
    calculate and save quantiles using a memory-efficient year-by-year approach.
    """
    var_name = VARIABLES_TO_PROCESS[var_code]
    print(f"\n{'=' * 70}")
    print(f"Processing: {var_name.upper()} / {scenario.upper()}")
    print("=" * 70)

    # 1. Find all ensemble member files
    source_dir = SOURCE_BASE_DIR / var_code / scenario
    search_pattern = str(source_dir / f"{var_code}_EUR-11_*.nc")
    ensemble_files = sorted(glob.glob(search_pattern))

    if not ensemble_files:
        print(f"  ✗ No files found for pattern: {search_pattern}")
        return

    print(f"  Found {len(ensemble_files)} ensemble members")

    # 2. Load all datasets (lazy loading with Dask)
    print(f"  Loading datasets...")
    datasets = []
    for i, fpath in enumerate(tqdm(ensemble_files, desc="  Opening files")):
        try:
            ds = xr.open_dataset(fpath, chunks={"time": 365})
            datasets.append(ds[var_code])  # Extract just the variable we need
        except Exception as e:
            print(f"    ✗ Could not open {Path(fpath).name}: {e}")
            continue

    if not datasets:
        print(f"  ✗ No valid datasets could be loaded")
        return

    # 3. Concatenate along ensemble dimension
    print(f"  Concatenating {len(datasets)} members along ensemble dimension...")
    try:
        ensemble_data = xr.concat(datasets, dim="ensemble")
        print(f"  ✓ Ensemble shape: {dict(ensemble_data.sizes)}")
    except Exception as e:
        print(f"  ✗ Error concatenating datasets: {e}")
        return

    # 4. Prepare output directory
    dest_dir = DEST_BASE_DIR / var_name / scenario
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 5. Calculate and save each quantile
    print(f"  Calculating quantiles...")
    for q in QUANTILES_TO_CALCULATE:
        pctl_str = f"pctl{int(q * 100)}"
        output_filename = f"{var_code}_EUR-11_{pctl_str}_{scenario}.nc"
        output_path = dest_dir / output_filename

        # Skip if already exists
        if output_path.exists():
            print(f"    ⊙ {output_filename} already exists, skipping")
            continue

        print(f"    Computing {pctl_str} (q={q})...")
        try:
            # Calculate quantile across ensemble dimension
            quantile_result = ensemble_data.quantile(q, dim="ensemble")

            # Convert to dataset for saving
            quantile_ds = quantile_result.to_dataset(name=var_code)

            # Copy attributes from first file
            sample_ds = xr.open_dataset(ensemble_files[0])
            quantile_ds.attrs = sample_ds.attrs
            for coord in quantile_ds.coords:
                if coord in sample_ds.coords:
                    quantile_ds[coord].attrs = sample_ds[coord].attrs
            sample_ds.close()

            # Define compression
            encoding = {
                var_code: {
                    "zlib": True,
                    "complevel": 5,
                    "dtype": "float32",
                    "_FillValue": -9999.0,
                }
            }

            # Save to temporary file first
            temp_path = output_path.with_suffix(".nc.tmp")
            print(f"    Writing to {output_filename}...")
            quantile_ds.to_netcdf(temp_path, encoding=encoding)

            # Rename to final path
            temp_path.rename(output_path)
            print(f"    ✓ Saved: {output_path.name}")

        except Exception as e:
            print(f"    ✗ Error processing quantile {q}: {e}")
            import traceback

            traceback.print_exc()
            if temp_path.exists():
                temp_path.unlink()
            continue

    # Close all datasets
    for ds in datasets:
        if hasattr(ds, "close"):
            ds.close()

    print(f"  ✓ Finished {var_name.upper()} / {scenario.upper()}")


def main():
    """Main function to run the processing."""
    print("=" * 70)
    print("CALCULATING ENSEMBLE QUANTILES FROM CORDEX DATA")
    print("Simplified version using threaded Dask")
    print("=" * 70)

    for var_code in VARIABLES_TO_PROCESS.keys():
        for scenario in SCENARIOS_TO_PROCESS:
            try:
                process_variable_scenario(var_code, scenario)
            except Exception as e:
                print(f"\n✗ FAILED: {var_code}/{scenario}")
                print(f"  Error: {e}")
                import traceback

                traceback.print_exc()
                continue

    print("\n" + "=" * 70)
    print("QUANTILE CALCULATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {DEST_BASE_DIR}")


# ===================================================================
# SCRIPT EXECUTION
# ===================================================================

if __name__ == "__main__":
    main()
