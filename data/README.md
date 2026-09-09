# Input data manifest

The analysis scripts expect the following layout beneath the project root. If the files already live elsewhere, set `FIRESCAPE_ROOT` to that directory instead of copying them.

```text
Data/
├── 00_QGIS/ADMIN/BOLZANO_REGION_UTM32.gpkg
├── 05_Meteorological_Data/
│   ├── Temperature/tmean_<year>.nc
│   ├── Precipitation/prec_<year>.nc
│   ├── SPEI30_Standardized/spei30_<year>.nc
│   └── SPEI90_Standardized/spei90_<year>.nc
├── 06_Administrative_Boundaries/Processed/
│   └── FireBrigade_ResponsibilityAreas_Bolzano_clipped.gpkg
├── 08_SPEI/Firescape/
└── STATIC_INPUT_250m/
    ├── nasadem.tif
    ├── slope.tif
    ├── aspect.tif
    ├── northness.tif
    ├── eastness.tif
    ├── tri.tif
    ├── treecoverdensity.tif
    ├── flammability.tif
    ├── distroads.tif
    ├── walking_time_to_bldg.tif
    └── walking_time_to_elec_infra.tif

output/
└── 01_Training_Data/spacetime_dataset.parquet
```

The preparation stage creates `output/01_Training_Data/training_features_ebm_250m.parquet`; model training writes the fitted EBM and validation products under `output/02_Model_Training/EBM_SPEI/`; daily prediction writes seasonal products under `output/03_Seasonal_Risk/`.

## Access and licensing

Raw wildfire occurrence records, provincial administrative layers, and some meteorological inputs are not redistributed here because access and licensing must be confirmed with their providers. Public users can inspect the complete method and released report artifacts, but full numerical reproduction requires authorized copies of those inputs.

Before redistributing any input, record its provider, version/date, license, spatial/temporal coverage, processing steps, and checksum in this file.
