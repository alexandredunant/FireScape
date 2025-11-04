#!/usr/bin/env python
"""
Calculate Ensemble Quantiles from CORDEX Climate Projections (Parallel Version)

This version processes multiple variable/scenario combinations in parallel
to take advantage of multiple CPUs.
"""

import xarray as xr
import numpy as np
from pathlib import Path
import glob
import warnings
from tqdm import tqdm
import sys

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===================================================================
# CONFIGURATION
# ===================================================================

SOURCE_BASE_DIR = Path("/mnt/CEPH_PROJECTS/FACT_CLIMAX/CORDEX-Adjust/QDM")
DEST_BASE_DIR = Path(
    "/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/climate_projections_ensemble_quantiles"
)

VARIABLES_TO_PROCESS = {
    "tas": "temperature",
    "pr": "precipitation",
}
SCENARIOS_TO_PROCESS = ["rcp45", "rcp85"]
QUANTILES_TO_CALCULATE = [0.25, 0.50, 0.75, 0.99]

# Use Dask for chunked computation with more workers
import dask
dask.config.set(scheduler="threads", num_workers=8)


# ===================================================================
# PROCESSING FUNCTIONS
# ===================================================================


def process_single_quantile(var_code, scenario, quantile, ensemble_files):
    """
    Process a single quantile for a given variable and scenario.
    This function is designed to be called in parallel.
    """
    var_name = VARIABLES_TO_PROCESS[var_code]
    pctl_str = f"pctl{int(quantile * 100)}"

    # Prepare output path
    dest_dir = DEST_BASE_DIR / var_name / scenario
    dest_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{var_code}_EUR-11_{pctl_str}_{scenario}.nc"
    output_path = dest_dir / output_filename

    # Skip if already exists
    if output_path.exists():
        return f"SKIP: {output_filename} already exists"

    try:
        # Load datasets
        datasets = []
        for fpath in ensemble_files:
            try:
                ds = xr.open_dataset(fpath, chunks={"time": 365})
                datasets.append(ds[var_code])
            except Exception as e:
                print(f"    ✗ Could not open {Path(fpath).name}: {e}")
                continue

        if not datasets:
            return f"ERROR: No valid datasets for {var_code}/{scenario}/{pctl_str}"

        # Concatenate along ensemble dimension
        ensemble_data = xr.concat(datasets, dim="ensemble")

        # Calculate quantile
        quantile_result = ensemble_data.quantile(quantile, dim="ensemble")

        # Convert to dataset
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
        quantile_ds.to_netcdf(temp_path, encoding=encoding)

        # Rename to final path
        temp_path.rename(output_path)

        # Clean up
        for ds in datasets:
            if hasattr(ds, "close"):
                ds.close()

        file_size_gb = output_path.stat().st_size / (1024**3)
        return f"SUCCESS: {output_filename} ({file_size_gb:.2f} GB)"

    except Exception as e:
        import traceback
        error_msg = f"ERROR: {var_code}/{scenario}/{pctl_str}: {e}\n{traceback.format_exc()}"
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        return error_msg


def process_variable_scenario(args):
    """
    Process all quantiles for a given variable and scenario.
    This wrapper function is designed to be called by multiprocessing.
    """
    var_code, scenario = args
    var_name = VARIABLES_TO_PROCESS[var_code]

    print(f"\n[{var_code}/{scenario}] Starting...")

    # Find ensemble files
    source_dir = SOURCE_BASE_DIR / var_code / scenario
    search_pattern = str(source_dir / f"{var_code}_EUR-11_*.nc")
    ensemble_files = sorted(glob.glob(search_pattern))

    if not ensemble_files:
        return f"[{var_code}/{scenario}] No files found"

    print(f"[{var_code}/{scenario}] Found {len(ensemble_files)} ensemble members")

    # Process each quantile sequentially within this worker
    results = []
    for q in QUANTILES_TO_CALCULATE:
        pctl_str = f"pctl{int(q * 100)}"
        print(f"[{var_code}/{scenario}] Processing {pctl_str}...")
        result = process_single_quantile(var_code, scenario, q, ensemble_files)
        results.append(result)
        print(f"[{var_code}/{scenario}] {result}")

    return f"[{var_code}/{scenario}] Completed all quantiles"


def main():
    """Main function to run parallel processing."""
    from multiprocessing import Pool

    print("=" * 70)
    print("CALCULATING ENSEMBLE QUANTILES FROM CORDEX DATA")
    print("Parallel version using multiprocessing")
    print("=" * 70)

    # Create list of all variable/scenario combinations
    tasks = [
        (var_code, scenario)
        for var_code in VARIABLES_TO_PROCESS.keys()
        for scenario in SCENARIOS_TO_PROCESS
    ]

    print(f"\nProcessing {len(tasks)} variable/scenario combinations:")
    for var_code, scenario in tasks:
        print(f"  - {var_code} / {scenario}")

    print(f"\nUsing {min(4, len(tasks))} parallel workers...\n")

    # Process in parallel - use 4 workers (tas/rcp45, tas/rcp85, pr/rcp45, pr/rcp85)
    # Each worker will process 4 quantiles sequentially
    with Pool(processes=min(4, len(tasks))) as pool:
        results = pool.map(process_variable_scenario, tasks)

    print("\n" + "=" * 70)
    print("RESULTS:")
    for result in results:
        print(result)

    print("\n" + "=" * 70)
    print("QUANTILE CALCULATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {DEST_BASE_DIR}")


# ===================================================================
# SCRIPT EXECUTION
# ===================================================================

if __name__ == "__main__":
    main()
