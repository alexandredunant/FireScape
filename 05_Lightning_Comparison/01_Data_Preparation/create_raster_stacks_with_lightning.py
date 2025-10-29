# %%
# ===================================================================
# LIBRARIES AND GLOBAL CONFIGURATION
# ===================================================================
import os
import pandas as pd
import geopandas as gpd
import rioxarray
import xarray as xr
import numpy as np
import glob
from datetime import timedelta
from joblib import Parallel, delayed
from shapely.wkb import loads
import traceback
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# --- Define File Paths ---
# Path to the input point data

# Corine Land Cover → Fire Risk Ordinal Mapping
# Based on actual 3-digit Corine codes in Bolzano Province dataset
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

OUTPUT_DIR = "/mnt/CEPH_PROJECTS/Firescape/Scripts/OUTPUT/01_Training_Data_Lightning/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
INPUT_PARQUET_PATH = "/mnt/CEPH_PROJECTS/Firescape/Scripts/OUTPUT/01_Training_Data/spacetime_dataset.parquet"

# Paths to raster data directories
STATIC_RASTER_DIR = "/mnt/CEPH_PROJECTS/Firescape/Data/STATIC_INPUT/"
TEMP_DIR = "/mnt/CEPH_PROJECTS/CLIMATE/GRIDS/TEMPERATURE/TIME_SERIES/UPLOAD/"
PRECIP_DIR = "/mnt/CEPH_PROJECTS/CLIMATE/GRIDS/PRECIPITATION/TIME_SERIES/UPLOAD/"
LIGHTNING_DIR = "/mnt/CEPH_PROJECTS/Firescape/Data/05_Meteorological_Data/Lightning/"

# Path for the final output NetCDF file
OUTPUT_NETCDF_PATH = os.path.join(OUTPUT_DIR, "spacetime_stacks_lightning.nc")

TEMP_STACK_DIR = os.path.join(OUTPUT_DIR, "temp_stacks/")
os.makedirs(TEMP_STACK_DIR, exist_ok=True)

# --- Define Stack Parameters ---
CHIP_SIZE = 32
TIME_STEPS = 60
TARGET_CRS = "EPSG:32632" # ED50 / UTM zone 32N

# Define the variable names for static and dynamic data
STATIC_VARS = [
    'tri',
    'northness',
    'slope',
    'aspect',
    'nasadem',
    'treecoverdensity',
    'landcover_fire_risk',
    'distroads',
    'eastness',
    'flammability',
    'walking_time_to_bldg',
    'walking_time_to_elec_infra'
] # 'landcoverclass',
DYNAMIC_VARS = ['T', 'P', 'L']  # Added Lightning (L)

print("="*80)
print("RASTER STACK CREATION - WITH LIGHTNING DATA")
print("="*80)
print(f"\nConfiguration:")
print(f"  Static variables: {len(STATIC_VARS)}")
print(f"  Dynamic variables: {DYNAMIC_VARS}")
print(f"  Lightning data: {LIGHTNING_DIR}")
print(f"  Output: {OUTPUT_NETCDF_PATH}")
print(f"  Note: Lightning data available from 2012 onwards")
print()


# %%
# ===================================================================
# DATA LOADING AND PREPARATION
# ===================================================================
print(f"Reading point data from: {INPUT_PARQUET_PATH}")

# Read the Parquet file into a pandas DataFrame
gdf = gpd.read_parquet(INPUT_PARQUET_PATH)

# Prepare the list of rows for parallel processing
rows_to_process = [row for _, row in gdf.iterrows()]
print(f"Loaded {len(rows_to_process)} points to process.")


# %%
# ===================================================================
# RASTER STACK EXTRACTION FUNCTION
# ===================================================================


def extract_stack_for_point(row, static_vars, dynamic_vars, chip_size, time_steps):
    """
    Extracts a 4D spatio-temporal stack for a single point, saves it as a
    NetCDF tensor, and returns the file path.
    """
    point_geom = row.geometry
    point_date = row.date
    point_id = row.id_obs
    variable_stacks = {}
    print(f"Processing ID: {point_id} for Date: {point_date.date()}")

    # --- 1. Load Template and Define Chip Area ---
    template_path = os.path.join(STATIC_RASTER_DIR, "nasadem.tif")
    try:
        with rioxarray.open_rasterio(template_path) as template_raster:
            iy = np.abs(template_raster.y - point_geom.y).argmin().item()
            ix = np.abs(template_raster.x - point_geom.x).argmin().item()

            half_chip = chip_size // 2
            y_slice = slice(max(0, iy - half_chip), min(template_raster.y.size, iy + half_chip))
            x_slice = slice(max(0, ix - half_chip), min(template_raster.x.size, ix + half_chip))

            template_chip = template_raster.isel(y=y_slice, x=x_slice).squeeze('band', drop=True)

    except Exception as e:
        print(f"Error loading template raster for ID {point_id}: {e}")
        return None

    # --- 2. Extract Static Variable Chips ---
    for var_name in static_vars:
        # All static variables load from {var_name}.tif
        # Note: landcover_fire_risk.tif is pre-transformed (0-5 ordinal values)
        # Created by Scripts/01_Data_Preparation/create_transformed_landcover.py
        raster_path = os.path.join(STATIC_RASTER_DIR, f"{var_name}.tif")

        try:
            with rioxarray.open_rasterio(raster_path) as rds:
                chip = rds.rio.reproject_match(template_chip).squeeze('band', drop=True)
                if "spatial_ref" in chip.coords:
                    chip = chip.drop_vars("spatial_ref")

                # landcover_fire_risk.tif already contains ordinal values (0-5)
                # No transformation needed here - it was done during raster creation

                static_stack = chip.expand_dims(time=time_steps).assign_coords(time=range(time_steps))
                variable_stacks[var_name] = static_stack
        except Exception as e:
            print(f"Could not process static var {var_name} for ID {point_id}: {e}")
            continue

    # --- 3. Extract Dynamic Variable Stacks ---
    for prefix in dynamic_vars:
        dynamic_stack_np = np.full((time_steps, chip_size, chip_size), np.nan, dtype=np.float32)

        # Determine data directory and file type
        if prefix == 'T':
            data_dir, ftype = TEMP_DIR, 'nc'
        elif prefix == 'P':
            data_dir, ftype = PRECIP_DIR, 'nc'
        elif prefix == 'L':
            data_dir, ftype = LIGHTNING_DIR, 'tif'
        else:
            print(f"Unknown dynamic variable: {prefix}")
            continue

        for day_offset in range(time_steps):
            current_date = point_date - timedelta(days=day_offset)

            # Different search patterns for NetCDF vs TIF
            if ftype == 'nc':
                search_pattern = os.path.join(data_dir, str(current_date.year), f"*{current_date.year}{current_date.month:02d}.nc")
            else:  # tif for lightning
                search_pattern = os.path.join(data_dir, f"Flash_Dens_{current_date.strftime('%Y%m%d')}.tif")

            file_matches = glob.glob(search_pattern)

            if file_matches:
                filepath = file_matches[0]

                try:
                    if ftype == 'tif':
                        # Lightning data is already a GeoTIFF
                        with rioxarray.open_rasterio(filepath, masked=True) as rda_raw:
                            rda = rda_raw.squeeze('band', drop=True)
                    else:
                        # Temperature/Precipitation are NetCDF
                        with xr.open_dataset(filepath) as rds:
                            var = list(rds.data_vars)[0]
                            rda = rds[var].sel(DATE=pd.to_datetime(current_date, format="%Y%m%d"), method="nearest").squeeze()
                            rda = rda.rio.write_crs(template_raster.rio.crs)

                    chip_reprojected = rda.rio.reproject_match(template_chip)

                    # For lightning, fill NaN with 0 (no lightning = 0 flashes)
                    if prefix == 'L':
                        chip_reprojected = chip_reprojected.fillna(0)

                    chip_np = chip_reprojected.values

                    if chip_np.shape == (chip_size, chip_size):
                        dynamic_stack_np[day_offset, :, :] = chip_np

                except Exception as e:
                    print(f"issue with the extraction of dynamic data \n {filepath}. Error: {e}")
                    # For lightning, fill with zeros if extraction fails
                    if prefix == 'L':
                        dynamic_stack_np[day_offset, :, :] = np.zeros((chip_size, chip_size), dtype=np.float32)
                    continue

        y_coords = template_chip.y
        x_coords = template_chip.x
        dynamic_da = xr.DataArray(dynamic_stack_np, 
                                  coords=[range(time_steps), y_coords, x_coords], 
                                  dims=['time', 'y', 'x'])
        variable_stacks[prefix] = dynamic_da

    # --- 4. Combine all variables into a single DataArray ---
    all_vars_ordered = static_vars + dynamic_vars
    if len(variable_stacks) != len(all_vars_ordered):
        print(f"Skipping ID {point_id} due to missing variable data.")
        return None

    stacked_arrays = [variable_stacks[var] for var in all_vars_ordered]
    
    final_stack = xr.concat(stacked_arrays, dim='channel').assign_coords(channel=all_vars_ordered)
    final_stack = final_stack.transpose('time', 'y', 'x', 'channel')
    final_stack = final_stack.assign_coords(id_obs=point_id)

    # --- 5. Convert to a simple tensor and save ---
    final_stack = final_stack.assign_coords(
        y=range(final_stack.y.size),
        x=range(final_stack.x.size)
    )
    
    output_path = os.path.join(TEMP_STACK_DIR, f"stack_{point_id}.nc")
    
    final_stack.to_netcdf(output_path)
    
    return output_path



# %% ===================================================================
# DEBUGGING AND VISUALIZATION
# ===================================================================

# 1. Select a few random rows to test
num_to_debug = 2
random_indices = np.random.choice(len(rows_to_process), num_to_debug, replace=False)
print(f"Randomly selected row indices to debug: {random_indices}\n")

# 2. Loop through the selected rows, run the function, and visualize
for index in random_indices:
    row_to_test = rows_to_process[index]
    point_id = row_to_test.id_obs

    print(f"==================== Testing Row Index: {index} (ID: {point_id}) ====================")

    output_path = None  # Initialize to None
    try:
        # Call the function which now returns a file path
        output_path = extract_stack_for_point(
            row_to_test,
            static_vars=STATIC_VARS,
            dynamic_vars=DYNAMIC_VARS,
            chip_size=CHIP_SIZE,
            time_steps=TIME_STEPS
        )

        # Check the result and print feedback
        if output_path and os.path.exists(output_path):
            print(f"\n✅ SUCCESS: Stack saved for ID: {point_id}")
            print(f"   - File at: {output_path}")

            # --- Open the saved file as a DataArray to inspect it ---
            # This is the key change to fix the error
            loaded_dataarray = xr.open_dataarray(output_path)

            # Perform checks on the loaded data
            print(f"   - Output Shape: {loaded_dataarray.dims}")
            nan_percentage = np.isnan(loaded_dataarray.values).mean() * 100
            print(f"   - NaN Percentage: {nan_percentage:.2f}%")

            # --- VISUALIZATION QC FOR THIS STACK ---
            print(f"\n--- Generating QC Plot for ID: {point_id} ---")
            event_date = row_to_test.date

            fig, axes = plt.subplots(1, len(DYNAMIC_VARS) + 1, figsize=(24, 5))
            fig.suptitle(f"Quality Control for Stack ID: {point_id}", fontsize=16)

            # 1. Spatial QC Plot (using .sel() on the channel coordinate)
            ax_map = axes[0]
            nasadem_chip = loaded_dataarray.sel(channel='nasadem').isel(time=0)
            nasadem_chip.plot.imshow(ax=ax_map, cmap='terrain', add_colorbar=False)
            center_coord = CHIP_SIZE // 2
            ax_map.scatter(center_coord, center_coord, color='red', marker='+', s=150, label='Center Point')
            ax_map.set_title("Spatial Chip (nasadem)")
            ax_map.legend()

            # 2. Temporal QC Plots
            for i, var_name in enumerate(DYNAMIC_VARS):
                ax_ts = axes[i+1]
                time_series_data = loaded_dataarray.sel(channel=var_name).mean(dim=['x', 'y'])
                date_axis = [event_date - timedelta(days=int(t)) for t in time_series_data['time'].values]

                ax_ts.plot(date_axis, time_series_data.values, marker='.', linestyle='-')
                ax_ts.set_title(f"Time Series for {var_name}")
                ax_ts.set_ylabel("Mean Value")
                ax_ts.grid(True, linestyle='--', alpha=0.6)
                ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax_ts.xaxis.set_major_locator(mdates.DayLocator(interval=10))

            fig.autofmt_xdate()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            plot_save_path = os.path.join(os.path.dirname(output_path), f"QC_plot_{point_id}.png")
            print(f"   - Saving plot to: {plot_save_path}")
            plt.savefig(plot_save_path, dpi=200)
            plt.show()

            # Close the file handle
            loaded_dataarray.close()

        else:
            print(f"\n❌ FAILED: Function returned None or file not found for ID: {point_id}")

    except Exception:
        print(f"\n💥 CRITICAL FAILURE on row index: {index}, ID: {point_id} 💥")
        traceback.print_exc()

    finally:
        # Optional: Clean up the temporary file after inspection
        if output_path and os.path.exists(output_path):
            os.remove(output_path)





# %%
# ===================================================================
# PARALLEL EXECUTION
# ===================================================================
import shutil

if __name__ == "__main__":

    print("Cleaning up temporary file...")

    if os.path.exists(TEMP_STACK_DIR):
        shutil.rmtree(TEMP_STACK_DIR)
    os.makedirs(TEMP_STACK_DIR, exist_ok=True)

    print("Starting parallel stack extraction...")

    # The parallel function now returns a list of file paths
    list_of_filepaths = Parallel(n_jobs=5, verbose=10)(
        delayed(extract_stack_for_point)(
            row,
            static_vars=STATIC_VARS,
            dynamic_vars=DYNAMIC_VARS,
            chip_size=CHIP_SIZE,
            time_steps=TIME_STEPS
            ) for row in rows_to_process[:]
    )

    # Filter out None results from failed points
    successful_files = [path for path in list_of_filepaths if path is not None]

    print(f"\nExtraction complete. Generated {len(successful_files)} temporary stack files.")

    # --- Combine files from disk and save the final dataset ---
    if successful_files:
        print("Combining all temporary stacks into a final dataset...")

        # Use open_mfdataset to efficiently combine files from disk
        final_dataset = xr.open_mfdataset(
                                successful_files,
                                combine='nested',
                                concat_dim="id_obs"
                            )

        # Add the target variable ('bin') and date as coordinates
        labels_df = gdf.set_index('id_obs')
        final_dataset = final_dataset.assign_coords(
            label=('id_obs', labels_df.loc[final_dataset.id_obs.values, 'bin']),
            event_date=('id_obs', labels_df.loc[final_dataset.id_obs.values, 'date'])
        )

        # Save the final combined dataset to a single file
        print(f"Saving final dataset to: {OUTPUT_NETCDF_PATH}")
        final_dataset.to_netcdf(OUTPUT_NETCDF_PATH)
        print("Dataset successfully saved.")

    else:
        print("No successful stacks were generated. Nothing to save.")
# %%
