# %%
# ===================================================================
# INITIAL SETTINGS & ENVIRONMENT SETUP
# ===================================================================
# GOAL: Import libraries and define all necessary file paths.

import os
import pandas as pd
import geopandas as gpd
import rioxarray
import xarray as xr
import numpy as np
import warnings
import glob
import calendar
import pyarrow
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from datetime import timedelta

# --- Suppress warnings for cleaner output ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Define absolute paths to input data ---
AOI_PATH = "/mnt/CEPH_PROJECTS/Firescape/Data/00_QGIS/ADMIN/BOLZANO_REGION_UTM32.gpkg"
STATIC_RASTER_DIR = "/mnt/CEPH_PROJECTS/Firescape/Data/STATIC_INPUT/"
TEMP_DIR = "/mnt/CEPH_PROJECTS/CLIMATE/GRIDS/TEMPERATURE/TIME_SERIES/UPLOAD/"
PRECIP_DIR = "/mnt/CEPH_PROJECTS/CLIMATE/GRIDS/PRECIPITATION/TIME_SERIES/UPLOAD/"
OUTPUT_DIR = "/mnt/CEPH_PROJECTS/Firescape/Scripts/OUTPUT/01_Training_Data/"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "spacetime_dataset.parquet")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Libraries imported and environment set up.")

# %%
# ===================================================================
# RASTER PROCESSING & LOADING SCRIPT
# ===================================================================
print("PHASE 1: Loading, clipping, and aligning static rasters...")

# --- Load AOI and initial static raster files ---
aoi_gdf = gpd.read_file(AOI_PATH)
static_files = [f for f in os.listdir(STATIC_RASTER_DIR) if f.endswith('.tif')]
grids = {}

# --- Clip each static raster ---
print(f"...found {len(static_files)} static rasters to process.")
for f in static_files:
    grid_name = f.replace('.tif', '')
    raster_path = os.path.join(STATIC_RASTER_DIR, f)
    # --- Add .squeeze() here to remove the band dimension ---
    grid = rioxarray.open_rasterio(raster_path, masked=True).squeeze('band', drop=True)

    # Reproject AOI to match raster CRS if they don't match
    aoi_reprojected = aoi_gdf.to_crs(grid.rio.crs) if aoi_gdf.crs != grid.rio.crs else aoi_gdf

    # Clip the raster to the AOI
    clipped_grid = grid.rio.clip(aoi_reprojected.geometry.values, all_touched=True, drop=True)
    grids[grid_name] = clipped_grid

# --- Define template and align all static rasters to it ---
template_raster = grids['nasadem']
print("...aligning all static grids to the 'nasadem' template.")
for name, grid in grids.items():
    grids[name] = grid.rio.reproject_match(template_raster)

# ===================================================================
# COMPLETION
# ===================================================================
print("\nAll static grids have been clipped and aligned.")
print(f"Final available grids: {list(grids.keys())}")


# %%
# ===================================================================
# 3. Absence Data Generation (Phases 1 & 2)
# ===================================================================
# This section generates the absence data points, which are crucial for 
# training a binary classification model.

# First, load the actual wildfire locations (presences).
d = gpd.read_file('/mnt/CEPH_PROJECTS/Firescape/Data/WILDFIRE_INVENTORY/wildfire_point_Bolzano_Period1999_2025.gpkg')

# Define the columns to check for duplicates
columns_to_check = ['Data Inc', 'UTM 32 N Lat.N.', 'Long.E.']
# Group by the columns and count the size of each group
duplicate_counts = d.groupby(columns_to_check).size().reset_index(name='Count')
# Filter to see only the records that appear more than once
actual_duplicates = duplicate_counts[duplicate_counts['Count'] > 1]
print(f"Number of duplicates : {len(actual_duplicates)}")

# Find rows where all columns are NA
empty_rows = d[d.isnull().all(axis=1)]
print(f"Number of empty rows: {len(empty_rows)}")

# %%
# Prepare and filter dataset
d['date'] = pd.to_datetime(d['Data Inc'])
d.dropna(subset=['date'], inplace=True)

# Filter dates requiring sufficient historical climate data
# Climate data available from 1980, so we can use fires from 1999 onwards
earliest_data_available = pd.to_datetime('1999-01-01')
TIME_STEPS = 60
first_valid_event_date = earliest_data_available + pd.Timedelta(days=TIME_STEPS)

# Filter out any presence points that are too early or too recent
d = d[d['date'] >= first_valid_event_date]  # Need 60 days of history
d = d[d['date'].dt.year <= 2024]  # Climate data through 2024
d = d[~d.is_empty]
d.reset_index(drop=True, inplace=True)

print(f"Dataset prepared and filtered. It now contains {len(d)} records.")
print(f"Date range: {d['date'].min().date()} to {d['date'].max().date()}")
print(f"Years covered: {d['date'].dt.year.nunique()}")

d = d[["Nr.", "date", "geometry"]]
ax = d.plot()
aoi_gdf.plot(ax=ax, 
             facecolor="none", 
              edgecolor='red', 
              lw=0.7)
plt.title("Filtered Presence Points")
plt.show()

# %%
# Get the index of the point with the minimum x-coordinate
outlier_index = d.geometry.x.idxmin()
# Use the index to select and display the outlier row
outlier_point = d.loc[outlier_index]
print("Outlier found:")
print(outlier_point)

# Get rid of the outlier
index_to_drop = d.geometry.x.idxmin()
d = d.drop(index_to_drop)

print("\nOutlier removed. The new dataframe has", len(d), "rows.")

ax = d.plot()
aoi_gdf.plot(ax=ax, facecolor="none", 
              edgecolor='red', lw=0.7)
plt.title("Presence Points After Outlier Removal")
plt.show()

# %%
# ===================================================================
# PHASE 1: ABSENCE DATA GENERATION (AT PRESENCE LOCATIONS)
# ===================================================================

# --- Replicate presence points to create absence candidates ---
n_replicates = 20
d_replicated = d.loc[d.index.repeat(n_replicates)].reset_index(drop=True)
d_replicated['bin'] = np.where(d_replicated.duplicated(subset=['Nr.']), 0, 1)

# --- Assign random dates to absence candidates ---
start_date = d['date'].min()
end_date = d['date'].max()
print(f'Setting date range between {start_date} and {end_date}')
date_range = pd.to_datetime(pd.date_range(start=start_date, end=end_date))
is_absence = d_replicated['bin'] == 0
d_replicated.loc[is_absence, 'date'] = np.random.choice(date_range, size=is_absence.sum())

# --- Filter absences at presence locations for temporal independence ---
d_presences = d_replicated[d_replicated['bin'] == 1].copy()
d_absences = d_replicated[d_replicated['bin'] == 0].copy()

absences_at_presences_list = []
for fire_id, group in d_absences.groupby('Nr.'):
    presence_date = d_presences[d_presences['Nr.'] == fire_id]['date'].iloc[0]
    time_buffer_before = presence_date - pd.Timedelta(days=60)
    time_buffer_after = presence_date + pd.Timedelta(days=1825) # 5 years
    valid_absences = group[(group['date'] < time_buffer_before) | (group['date'] > time_buffer_after)]
    absences_at_presences_list.append(valid_absences)

absences_at_presences_filtered = pd.concat(absences_at_presences_list)
# Sample 2 absences for each original presence location
absences_at_presences_final = absences_at_presences_filtered.groupby('Nr.').sample(n=2, replace=False, random_state=7)

print(f"Generated {len(absences_at_presences_final)} absences at presence locations.")

# %%
# ===================================================================
# PHASE 2: ABSENCE DATA GENERATION (AT NEW LOCATIONS)
# ===================================================================
print("PHASE 2: Generating absences at new locations...")

# Create AOI mask: rasterize the AOI boundary to ensure sampling only within the province
aoi_mask = rasterize(
    shapes=aoi_gdf.geometry,
    out_shape=(template_raster.rio.height, template_raster.rio.width),
    transform=template_raster.rio.transform(),
    fill=0,          # Pixels outside AOI are 0
    default_value=1  # Pixels inside AOI are 1
)
aoi_mask = xr.DataArray(aoi_mask, coords=template_raster.coords, dims=template_raster.dims)

# Create trivial mask: fires cannot happen on water, ice, or wetlands
# Using landcover_fire_risk (0-5 ordinal): exclude areas with fire_risk = 0
# fire_risk = 0 includes: glaciers (335), water courses (511), water bodies (512),
# inland marshes (411), and peatbogs (412)
# Note: landcover_fire_risk.tif is pre-transformed from Corine codes
trivial_mask = xr.where(
    grids['landcover_fire_risk'] > 0,  # Fire risk > 0 (excludes water/ice/wetlands)
    1,  # Valid areas for fire (risk levels 1-5)
    0   # Invalid areas (risk level 0: water, ice, wetlands)
).astype(float)

burned_gdf = d.copy()

# --- Create a buffer around each point to generate a circular polygon ---
BUFFER_DISTANCE = 500
print(f"Creating a {BUFFER_DISTANCE}-meter buffer around each wildfire point...")
burned_gdf['geometry'] = burned_gdf.geometry.buffer(BUFFER_DISTANCE)

# Now, rasterize the newly created buffered polygons
burned_mask = rasterize(
    shapes=burned_gdf.geometry,
    out_shape=(template_raster.rio.height, template_raster.rio.width),
    transform=template_raster.rio.transform(),
    fill=0,          # Pixels outside the buffer are 0
    default_value=1  # Pixels inside the buffer are 1
)

burned_mask = xr.DataArray(burned_mask, coords=template_raster.coords, dims=template_raster.dims)

# Combine all masks: must be inside AOI, valid terrain, and outside burned buffers
sampling_area = xr.where((aoi_mask == 1) & (trivial_mask == 1) & (burned_mask == 0), 1, np.nan).squeeze()
valid_pixels = np.argwhere(~np.isnan(sampling_area.values))

# %%
# ===================================================================
# VISUALIZE VALID AND INVALID SAMPLING MASKS
# ===================================================================
print("Visualizing sampling masks...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: AOI mask
ax1 = axes[0, 0]
im1 = ax1.imshow(aoi_mask.values, cmap='Blues', vmin=0, vmax=1,
                 extent=[aoi_mask.x.min(), aoi_mask.x.max(),
                        aoi_mask.y.min(), aoi_mask.y.max()],
                 origin='upper', aspect='equal')
aoi_gdf.plot(ax=ax1, facecolor="none", edgecolor='darkblue', lw=2)
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Inside AOI (1) / Outside (0)', fontsize=10)
ax1.set_title('AOI Mask\n(Bolzano Province Boundary)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Easting (m)', fontsize=10)
ax1.set_ylabel('Northing (m)', fontsize=10)
ax1.ticklabel_format(style='plain', useOffset=False)
ax1.grid(True, alpha=0.3, linestyle='--')

# Plot 2: Trivial mask (landcover exclusions)
ax2 = axes[0, 1]
im2 = ax2.imshow(trivial_mask.values, cmap='RdYlGn', vmin=0, vmax=1,
                 extent=[trivial_mask.x.min(), trivial_mask.x.max(),
                        trivial_mask.y.min(), trivial_mask.y.max()],
                 origin='upper', aspect='equal')
aoi_gdf.plot(ax=ax2, facecolor="none", edgecolor='black', lw=2)
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Valid (1) / Invalid (0)', fontsize=10)
ax2.set_title('Trivial Mask\n(Excludes: Ice, Lakes, Rivers)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Easting (m)', fontsize=10)
ax2.set_ylabel('Northing (m)', fontsize=10)
ax2.ticklabel_format(style='plain', useOffset=False)
ax2.grid(True, alpha=0.3, linestyle='--')

# Plot 3: Burned area mask
ax3 = axes[1, 0]
im3 = ax3.imshow(burned_mask.values, cmap='RdYlGn_r', vmin=0, vmax=1,
                 extent=[burned_mask.x.min(), burned_mask.x.max(),
                        burned_mask.y.min(), burned_mask.y.max()],
                 origin='upper', aspect='equal')
aoi_gdf.plot(ax=ax3, facecolor="none", edgecolor='black', lw=2)
cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.set_label('Buffered (1) / Not Buffered (0)', fontsize=10)
ax3.set_title(f'Burned Area Buffer Mask\n({BUFFER_DISTANCE}m around each fire)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Easting (m)', fontsize=10)
ax3.set_ylabel('Northing (m)', fontsize=10)
ax3.ticklabel_format(style='plain', useOffset=False)
ax3.grid(True, alpha=0.3, linestyle='--')

# Plot 4: Final sampling area (valid for absence generation)
ax4 = axes[1, 1]
# Create a custom visualization showing valid areas and fires
im4 = ax4.imshow(sampling_area.values, cmap='Greens',
                 extent=[sampling_area.x.min(), sampling_area.x.max(),
                        sampling_area.y.min(), sampling_area.y.max()],
                 origin='upper', aspect='equal')
aoi_gdf.plot(ax=ax4, facecolor="none", edgecolor='black', lw=2)
d.plot(ax=ax4, color='red', markersize=15, alpha=0.7, edgecolor='darkred', linewidth=0.5, label='Fire Locations')
cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
cbar4.set_label('Valid Sampling Area', fontsize=10)
ax4.set_title('Final Valid Sampling Area\n(Inside AOI + Valid terrain + Outside fire buffers)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Easting (m)', fontsize=10)
ax4.set_ylabel('Northing (m)', fontsize=10)
ax4.ticklabel_format(style='plain', useOffset=False)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(loc='upper right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sampling_masks_visualization.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"\nSampling Area Statistics:")
print(f"  Valid sampling pixels: {len(valid_pixels):,}")
print(f"  Total pixels in AOI: {trivial_mask.size:,}")
print(f"  Percentage valid for sampling: {(len(valid_pixels) / trivial_mask.size * 100):.2f}%")

# %%
# --- Create new absence locations ---
n_new_locations = 2 * len(d_presences)
np.random.seed(42)
sample_indices = valid_pixels[np.random.choice(len(valid_pixels), n_new_locations, replace=False)]
ys, xs = sample_indices[:, 0], sample_indices[:, 1]
coords_x, coords_y = template_raster.x[xs].values, template_raster.y[ys].values

absences_new_locs = gpd.GeoDataFrame(geometry=gpd.points_from_xy(coords_x, coords_y), crs=template_raster.rio.crs)
absences_new_locs['bin'] = 0

max_existing_id = int(d['Nr.'].max())
absences_new_locs['Nr.'] = range(max_existing_id + 1, max_existing_id + 1 + len(absences_new_locs)) # creates and assigns new IDs

# filtering dates for new locations
n_samples_per_new_loc = 2 # 2 absences per new location for balance
absences_new_locs_replicated = absences_new_locs.loc[absences_new_locs.index.repeat(n_samples_per_new_loc)].reset_index(drop=True)
absences_new_locs_replicated['date'] = np.random.choice(date_range, size=len(absences_new_locs_replicated))
absences_new_locs_final = absences_new_locs_replicated.copy()


print(f"Generated {len(absences_new_locs_final)} absences at new random locations.")

# %%
# ===================================================================
# PHASE 3: BALANCE FINAL DATASET (WITH STRATIFIED SAMPLING)
# ===================================================================
print("PHASE 3: Balancing final dataset with temporal stratification...")

# --- Ensure all data has a consistent CRS before merging ---
target_crs = d_presences.crs
print(f"Standardizing all data to target CRS: {target_crs}")

if absences_at_presences_final.crs != target_crs:
    absences_at_presences_final = absences_at_presences_final.to_crs(target_crs)

if absences_new_locs_final.crs != target_crs:
    absences_new_locs_final = absences_new_locs_final.to_crs(target_crs)

# --- Combine all available absence points into one pool ---
all_absences = pd.concat([absences_at_presences_final, absences_new_locs_final], ignore_index=True)
all_absences['month'] = all_absences['date'].dt.month

# --- 1. Define the total number of absence points needed ---
target_n_absences = len(d_presences) * 5

# --- 2. Calculate the monthly distribution of PRESENCE points ---
d_presences['month'] = d_presences['date'].dt.month
presence_monthly_distribution = d_presences['month'].value_counts(normalize=True)
print("\n Monthly presence distribution:")
print(presence_monthly_distribution)

# --- 3. Determine how many absences to sample from each month ---
samples_per_month = (presence_monthly_distribution * target_n_absences).round().astype(int)
print("\nTarget number of absence samples per month:")
print(samples_per_month)

# --- 4. Perform the stratified sampling ---
balanced_absences_list = []
for month, n_to_sample in samples_per_month.items():
    absences_in_month = all_absences[all_absences['month'] == month]
    if len(absences_in_month) > n_to_sample:
        sampled_absences = absences_in_month.sample(n=n_to_sample, random_state=42)
    else:
        sampled_absences = absences_in_month
    balanced_absences_list.append(sampled_absences)

balanced_absences = pd.concat(balanced_absences_list)


# --- 5. Assemble the final dataframe ---
final_df = pd.concat([d_presences, balanced_absences], ignore_index=True)
final_df.reset_index(drop=True, inplace=True)
final_df['id_obs'] = final_df.index
print(f"\nFinal dataset assembled. Shape: {final_df.shape}")

print("\nFinal class distribution:")
print(final_df['bin'].value_counts())

print("\nSample of the final dataframe before data extraction:")
print(final_df.head())

# %%
# ===================================================================
# VISUALIZE DATASET BALANCE
# ===================================================================
# 1. Count the number of presence (1) and absence (0) points for each month.
monthly_counts = pd.crosstab(final_df['month'], final_df['bin'])

# Rename columns for a clearer legend
monthly_counts.rename(columns={0: 'Absence', 1: 'Presence'}, inplace=True)

# 2. Create the grouped bar plot
ax = monthly_counts.plot(kind='bar', figsize=(10, 5), rot=0,
                         color={'Presence': 'orangered', 'Absence': 'steelblue'})

# 3. Format the plot for readability
ax.set_title('Final Distribution of Presence and Absence Points per Month', fontsize=16)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Number of Points', fontsize=12)
ax.legend(title='Point Type')

# Use month abbreviations for x-axis labels
ax.set_xticklabels([calendar.month_abbr[i] for i in monthly_counts.index])

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 5. Plot
ax = aoi_gdf.plot(facecolor="none", 
              edgecolor='red', lw=0.7)
# trivial_mask.plot(ax=ax)
# burned_mask.plot(ax=ax, alpha=0.2)
# final_df[final_df.bin == 0].plot(c='b', ax=ax, label='absence')
final_df[final_df.bin == 1].plot(c='r', markersize=20, ax=ax, label='presence')
absences_at_presences_final.plot(c='g', markersize=2, ax=ax, label='absence at presence')
absences_new_locs_final.plot(c='b', markersize=2, ax=ax, label='absence at absence')

plt.legend()
plt.show()


# %%
# ===================================================================
# SAVE THE BALANCED GEODATAFRAME TO PARQUET
# ===================================================================

# Create a copy to avoid changing the original final_df
df_for_parquet = final_df.copy()

# 1. Convert the 'Nr.' column to an integer type
df_for_parquet['Nr.'] = df_for_parquet['Nr.'].astype(int)

# # 2. Convert the 'geometry' column to Well-Known Binary (WKB) format
# # This is a standard way to store geometry in non-spatial formats like Parquet
# df_for_parquet['geometry'] = df_for_parquet['geometry'].apply(lambda geom: geom.wkb)

print(f"Saving final_df to Parquet file at: {OUTPUT_PATH}")

try:
    # Save the modified copy to the Parquet file
    df_for_parquet.to_parquet(OUTPUT_PATH, engine='pyarrow')
    print("Successfully saved the DataFrame.")
except Exception as e:
    print(f"An error occurred while saving to Parquet: {e}")
# %%
