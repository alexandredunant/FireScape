# Data Directory

**Note:** Due to large file sizes (~18 GB total), actual data files are **not included** in this GitHub repository. This README describes the data structure and sources.

## Overview

This directory contains all input data for the FireScape wildfire risk modeling system. Data spans 1999-2024 and covers the Bolzano/Alto Adige province in the Italian Alps (7,400 km²).

## Data Structure

```
Data/
├── 00_QGIS/                          # QGIS project files (~1.4 GB)
├── 01_Topographic_Features/          # DEM, slope, aspect (~991 MB)
├── 02_Land_Cover/                    # Vegetation and land use (~449 MB)
├── 03_Human_Infrastructure/          # Roads, buildings (~2.2 GB)
├── 04_Electrical_Infrastructure/     # Power lines (~203 MB)
├── 05_Meteorological_Data/           # Temperature, precipitation (~2.5 GB)
├── 06_Administrative_Boundaries/     # Province boundaries (~423 MB)
├── 07_Demographics/                  # Population data (~924 MB)
├── 08_Climate_Data/                  # Future climate projections
├── ML/                               # Processed ML-ready datasets (~8.8 GB)
├── STATIC_INPUT/                     # Static features (~375 MB)
├── WILDFIRE_INVENTORY/               # Historical fire records (~315 MB)
└── README.md                         # This file

Total size: ~18.3 GB
```

## Dataset Descriptions

### 00_QGIS/
**Purpose:** Geographic visualization and preprocessing
**Contents:**
- QGIS project files (.qgz)
- Layer styling configurations
- Spatial analysis workflows

**Source:** Custom-built project files

---

### 01_Topographic_Features/
**Purpose:** Terrain characteristics affecting fire behavior
**Contents:**
- Digital Elevation Model (DEM) at 30m resolution
- Slope (degrees and percent)
- Aspect (cardinal and continuous)
- Terrain Ruggedness Index (TRI)
- Topographic Position Index (TPI)

**Source:**
- EU-DEM v1.1 (European Environment Agency)
- Derived using GDAL/GRASS GIS

**Spatial Resolution:** 30m
**Temporal Coverage:** Static (2010 baseline)

---

### 02_Land_Cover/
**Purpose:** Vegetation types and fuel characteristics
**Contents:**
- CORINE Land Cover classifications
- Forest type maps (conifer, broadleaf, mixed)
- Vegetation density indices
- Fire risk categories by land cover type

**Source:**
- CORINE Land Cover 2018 (Copernicus)
- Provincial forest inventory (Bolzano Forestry Service)

**Spatial Resolution:** 100m (CORINE), 10m (provincial)
**Temporal Coverage:** 2018 baseline, updated 2021

---

### 03_Human_Infrastructure/
**Purpose:** Human access and ignition sources
**Contents:**
- Road network (highways, local roads, trails)
- Building footprints and density
- Distance to nearest road/building rasters
- Urban-wildland interface zones

**Source:**
- OpenStreetMap (OSM)
- Provincial cadastral data

**Spatial Resolution:** 30m
**Temporal Coverage:** Updated annually 2015-2024

---

### 04_Electrical_Infrastructure/
**Purpose:** Lightning-ignition interaction with power lines
**Contents:**
- High-voltage transmission lines
- Medium-voltage distribution lines
- Substations and transformers
- Distance to power infrastructure rasters

**Source:**
- Provincial infrastructure database
- Energy provider (Alperia) records

**Spatial Resolution:** 30m
**Temporal Coverage:** 2020 baseline

---

### 05_Meteorological_Data/
**Purpose:** Climate drivers of fire risk
**Contents:**
- Daily temperature (min, max, mean)
- Daily precipitation
- Relative humidity
- Wind speed and direction
- Derived indices:
  - Temperature accumulations (3d, 5d, 7d, 14d, 30d, 60d)
  - Precipitation deficits (same windows)
  - Vapor Pressure Deficit (VPD)

**Source:**
- Provincial weather station network (34 stations)
- ARPAV (Regional Environmental Agency)
- Gridded interpolations via kriging

**Spatial Resolution:** 1km grid
**Temporal Coverage:** Daily, 1999-01-01 to 2024-12-31
**Missing Data:** < 2% (imputed via spatial interpolation)

---

### 06_Administrative_Boundaries/
**Purpose:** Spatial aggregation and reporting zones
**Contents:**
- Province boundary (Bolzano/Alto Adige)
- Municipality boundaries (116 units)
- Forest districts (8 zones)
- Protected areas (national parks, nature reserves)

**Source:**
- ISTAT (Italian National Statistics Institute)
- Provincial geographic service

**Spatial Resolution:** Vector polygons
**Temporal Coverage:** 2024 boundaries

---

### 07_Demographics/
**Purpose:** Population exposure and fire management capacity
**Contents:**
- Population counts by municipality
- Population density rasters
- Temporal population dynamics (tourism, seasonal)
- Forestry department staffing records

**Source:**
- ISTAT census data
- Provincial tourism statistics

**Spatial Resolution:** 100m
**Temporal Coverage:** Annual, 2002-2024

---

### 08_Climate_Data/
**Purpose:** Future climate projections for scenario analysis
**Contents:**
- CMIP6 downscaled projections
- RCP 4.5 and RCP 8.5 scenarios
- Temperature and precipitation (2025-2100)
- Derived fire weather indices

**Source:**
- EURO-CORDEX regional climate models
- 5-model ensemble mean

**Spatial Resolution:** 12km
**Temporal Coverage:** Daily, 2025-2100

---

### ML/
**Purpose:** Pre-processed machine learning datasets
**Contents:**
- `spacetime_stacks_baseline.nc` - T+P features (1999-2024)
- `spacetime_stacks_lightning_2012plus.nc` - T+P+L features (2012-2024)
- Case-control sampled training datasets
- Feature engineering outputs:
  - Temporal aggregations (3d, 5d, 7d, 14d, 30d, 60d windows)
  - Spatial buffers (100m, 500m, 1km)
  - Interaction terms

**Format:** NetCDF-4 (xarray-compatible)
**Structure:**
- Dimensions: `(id_obs, features)`
- Coordinates: `fire_date`, `fire_x`, `fire_y`
- Variables: Temperature, precipitation, lightning, static features

**Sampling Strategy:**
- Fire events: 998 confirmed wildfires
- Non-fire controls: 2 per fire event (stratified by month, elevation)
- Total observations: 2,994 (1999-2024 baseline)
- Total observations: 1,556 (2012-2024 lightning)

---

### STATIC_INPUT/
**Purpose:** Time-invariant features for model training
**Contents:**
- Compiled static feature rasters (topography, land cover, infrastructure)
- Preprocessed for direct model input
- Normalized and scaled versions

**Format:** GeoTIFF and NetCDF
**Spatial Resolution:** 30m

---

### WILDFIRE_INVENTORY/
**Purpose:** Historical wildfire records (response variable)
**Contents:**
- 998 wildfire events (1999-2024)
- Attributes:
  - Date and time of ignition
  - Location (coordinates)
  - Burned area (hectares)
  - Cause (lightning, human, unknown)
  - Forest type affected
  - Suppression resources deployed

**Source:**
- Bolzano Province Forest Fire Service
- Fire incident reports
- Satellite validation (Sentinel-2, Landsat)

**Temporal Coverage:** 1999-01-01 to 2024-12-31
**Spatial Accuracy:** ±50m (GPS coordinates)

**Quality Control:**
- Manual verification of all events > 1 ha
- Satellite confirmation for events > 5 ha
- Cross-validation with newspaper reports

---

## Lightning Data (Critical for Analysis)

**Period:** 2012-present
**Source:** EUCLID (European Cooperation for Lightning Detection)
**Coverage:** Central Europe
**Variables:**
- Lightning flash density (flashes/km²/day)
- Cloud-to-ground (CG) polarity
- Peak current (kA)
- Temporal patterns (3d, 5d, 7d, 14d, 30d, 60d accumulations)

**Key Findings:**
- 6.5% of days have lightning activity
- Fire days have 7.6× more lightning than non-fire days (2012-2024)
- Lightning contributes to 28±5% of summer fire risk

**Limitation:** Data only available from 2012 onwards, requiring separate analysis periods:
- **Baseline model**: 1999-2024 (Temperature + Precipitation only)
- **Lightning model**: 2012-2024 (Temperature + Precipitation + Lightning)

---

## Data Access

### For Reproducibility:

Due to file sizes and licensing restrictions, data is **not hosted on GitHub**. To reproduce this analysis:

1. **Topographic data:** Download EU-DEM from [Copernicus](https://land.copernicus.eu/imagery-in-situ/eu-dem)
2. **Land cover:** Download CORINE 2018 from [Copernicus](https://land.copernicus.eu/pan-european/corine-land-cover)
3. **Meteorological data:** Request from ARPAV (Regional Environmental Agency of Veneto)
4. **Lightning data:** Contact EUCLID for research access: [www.euclid.org](https://www.euclid.org)
5. **Wildfire inventory:** Request from Bolzano Province Forest Service

### Contact for Data Sharing:

For **processed datasets** ready for model training:
- Contact: [Your email]
- Datasets available upon reasonable request
- Collaboration opportunities welcome

---

## Data Processing Scripts

All data preprocessing is documented in:
- `Scripts/01_Data_Preparation/` - Feature extraction pipelines
- `Scripts/05_Lightning_Comparison/01_Data_Preparation/` - Lightning-specific processing

---

## Citation

If you use these data descriptions or processed datasets, please cite:

```
[Your paper citation here once published]
```

---

## License

- **Wildfire inventory:** © Bolzano Province, shared under research agreement
- **Meteorological data:** © ARPAV, research use permitted
- **Topographic/Land cover:** Copernicus data (free and open access)
- **Lightning data:** © EUCLID, research license required
- **Infrastructure:** © Bolzano Province, research use only

Please respect data provider licenses when reproducing this work.
