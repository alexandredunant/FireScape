# CORDEX Climate Ensemble Members Summary

This document describes the ensemble members available for each climate variable and scenario in the QDM-adjusted CORDEX dataset used for the FireScape project.

## Overview

| Variable | Scenario | Ensemble Members | Notes |
|----------|----------|------------------|-------|
| **tas** (temperature) | RCP4.5 | 15 | Standard ensemble |
| **tas** (temperature) | RCP8.5 | 15 | Standard ensemble |
| **pr** (precipitation) | RCP4.5 | 11 | Missing SMHI-RCA4 variants |
| **pr** (precipitation) | RCP8.5 | 17 | Expanded ensemble with additional RCMs |

---

## Temperature (tas) - 15 Members (Both Scenarios)

The temperature ensemble is consistent across both RCP4.5 and RCP8.5 scenarios:

### Global Climate Models (GCMs):
1. CNRM-CERFACS-CNRM-CM5 (4 RCMs)
2. ICHEC-EC-EARTH (4 RCMs)
3. IPSL-IPSL-CM5A-MR (2 RCMs)
4. MPI-M-MPI-ESM-LR (3 RCMs)
5. NCC-NorESM1-M (2 RCMs)

### Regional Climate Models (RCMs):
- CLMcom-CCLM4-8-17
- CNRM-ALADIN63
- KNMI-RACMO22E
- SMHI-RCA4
- DMI-HIRHAM5
- IPSL-WRF381P
- MPI-CSC-REMO2009
- GERICS-REMO2015

### Complete List:
```
tas_EUR-11_CNRM-CERFACS-CNRM-CM5_CLMcom-CCLM4-8-17_r1i1p1_v1_day_19702100_rcp{45,85}.nc
tas_EUR-11_CNRM-CERFACS-CNRM-CM5_CNRM-ALADIN63_r1i1p1_v2_day_19702100_rcp{45,85}.nc
tas_EUR-11_CNRM-CERFACS-CNRM-CM5_KNMI-RACMO22E_r1i1p1_v2_day_19702100_rcp{45,85}.nc
tas_EUR-11_CNRM-CERFACS-CNRM-CM5_SMHI-RCA4_r1i1p1_v1_day_19702100_rcp{45,85}.nc
tas_EUR-11_ICHEC-EC-EARTH_CLMcom-CCLM4-8-17_r12i1p1_v1_day_19702100_rcp{45,85}.nc
tas_EUR-11_ICHEC-EC-EARTH_DMI-HIRHAM5_r3i1p1_v2_day_19702100_rcp{45,85}.nc
tas_EUR-11_ICHEC-EC-EARTH_KNMI-RACMO22E_r12i1p1_v1_day_19702100_rcp{45,85}.nc
tas_EUR-11_ICHEC-EC-EARTH_SMHI-RCA4_r12i1p1_v1_day_19702100_rcp{45,85}.nc
tas_EUR-11_IPSL-IPSL-CM5A-MR_IPSL-WRF381P_r1i1p1_v1_day_19702100_rcp{45,85}.nc
tas_EUR-11_IPSL-IPSL-CM5A-MR_SMHI-RCA4_r1i1p1_v1_day_19702100_rcp{45,85}.nc
tas_EUR-11_MPI-M-MPI-ESM-LR_CLMcom-CCLM4-8-17_r1i1p1_v1_day_19702100_rcp{45,85}.nc
tas_EUR-11_MPI-M-MPI-ESM-LR_MPI-CSC-REMO2009_r1i1p1_v1_day_19702100_rcp{45,85}.nc
tas_EUR-11_MPI-M-MPI-ESM-LR_SMHI-RCA4_r1i1p1_v1a_day_19702100_rcp{45,85}.nc
tas_EUR-11_NCC-NorESM1-M_GERICS-REMO2015_r1i1p1_v1_day_19702100_rcp{45,85}.nc
tas_EUR-11_NCC-NorESM1-M_SMHI-RCA4_r1i1p1_v1_day_19702100_rcp{45,85}.nc
```

---

## Precipitation (pr) - Variable Ensemble Size

### RCP4.5 - 11 Members

**Missing compared to temperature ensemble:**
- All 5 SMHI-RCA4 variants (CNRM-CM5, EC-EARTH, IPSL-CM5A-MR, MPI-ESM-LR, NCC-NorESM1-M)
- ICHEC-EC-EARTH with DMI-HIRHAM5
- NCC-NorESM1-M with GERICS-REMO2015

**Additional models not in temperature ensemble:**
- CNRM-CERFACS-CNRM-CM5 with RMIB-UGent-ALARO-0
- MOHC-HadGEM2-ES with CLMcom-CCLM4-8-17
- MOHC-HadGEM2-ES with KNMI-RACMO22E

**Complete List:**
```
pr_EUR-11_CNRM-CERFACS-CNRM-CM5_CLMcom-CCLM4-8-17_r1i1p1_v1_day_19702100_rcp45.nc
pr_EUR-11_CNRM-CERFACS-CNRM-CM5_CNRM-ALADIN63_r1i1p1_v2_day_19702100_rcp45.nc
pr_EUR-11_CNRM-CERFACS-CNRM-CM5_KNMI-RACMO22E_r1i1p1_v2_day_19702100_rcp45.nc
pr_EUR-11_CNRM-CERFACS-CNRM-CM5_RMIB-UGent-ALARO-0_r1i1p1_v1_day_19702100_rcp45.nc
pr_EUR-11_ICHEC-EC-EARTH_CLMcom-CCLM4-8-17_r12i1p1_v1_day_19702100_rcp45.nc
pr_EUR-11_ICHEC-EC-EARTH_KNMI-RACMO22E_r12i1p1_v1_day_19702100_rcp45.nc
pr_EUR-11_IPSL-IPSL-CM5A-MR_IPSL-WRF381P_r1i1p1_v1_day_19702100_rcp45.nc
pr_EUR-11_MOHC-HadGEM2-ES_CLMcom-CCLM4-8-17_r1i1p1_v1_day_19702100_rcp45.nc
pr_EUR-11_MOHC-HadGEM2-ES_KNMI-RACMO22E_r1i1p1_v2_day_19702100_rcp45.nc
pr_EUR-11_MPI-M-MPI-ESM-LR_CLMcom-CCLM4-8-17_r1i1p1_v1_day_19702100_rcp45.nc
pr_EUR-11_MPI-M-MPI-ESM-LR_MPI-CSC-REMO2009_r1i1p1_v1_day_19702100_rcp45.nc
```

### RCP8.5 - 17 Members

**Additional models compared to RCP4.5:**
- 3 models with CLMcom-ETH-COSMO-crCLIM-v1-1 (CNRM-CM5, EC-EARTH, MPI-ESM-LR)
- 2 models with ICTP-RegCM4-6 (EC-EARTH, NorESM1-M)
- ICHEC-EC-EARTH with IPSL-WRF381P

**Complete List:**
```
pr_EUR-11_CNRM-CERFACS-CNRM-CM5_CLMcom-CCLM4-8-17_r1i1p1_v1_day_19702100_rcp85.nc
pr_EUR-11_CNRM-CERFACS-CNRM-CM5_CLMcom-ETH-COSMO-crCLIM-v1-1_r1i1p1_v1_day_19702100_rcp85.nc
pr_EUR-11_CNRM-CERFACS-CNRM-CM5_CNRM-ALADIN63_r1i1p1_v2_day_19702100_rcp85.nc
pr_EUR-11_CNRM-CERFACS-CNRM-CM5_KNMI-RACMO22E_r1i1p1_v2_day_19702100_rcp85.nc
pr_EUR-11_CNRM-CERFACS-CNRM-CM5_RMIB-UGent-ALARO-0_r1i1p1_v1_day_19702100_rcp85.nc
pr_EUR-11_ICHEC-EC-EARTH_CLMcom-CCLM4-8-17_r12i1p1_v1_day_19702100_rcp85.nc
pr_EUR-11_ICHEC-EC-EARTH_CLMcom-ETH-COSMO-crCLIM-v1-1_r12i1p1_v1_day_19702100_rcp85.nc
pr_EUR-11_ICHEC-EC-EARTH_ICTP-RegCM4-6_r12i1p1_v1_day_19702100_rcp85.nc
pr_EUR-11_ICHEC-EC-EARTH_IPSL-WRF381P_r12i1p1_v1_day_19702100_rcp85.nc
pr_EUR-11_ICHEC-EC-EARTH_KNMI-RACMO22E_r12i1p1_v1_day_19702100_rcp85.nc
pr_EUR-11_IPSL-IPSL-CM5A-MR_IPSL-WRF381P_r1i1p1_v1_day_19702100_rcp85.nc
pr_EUR-11_MOHC-HadGEM2-ES_CLMcom-CCLM4-8-17_r1i1p1_v1_day_19702100_rcp85.nc
pr_EUR-11_MOHC-HadGEM2-ES_KNMI-RACMO22E_r1i1p1_v2_day_19702100_rcp85.nc
pr_EUR-11_MPI-M-MPI-ESM-LR_CLMcom-CCLM4-8-17_r1i1p1_v1_day_19702100_rcp85.nc
pr_EUR-11_MPI-M-MPI-ESM-LR_CLMcom-ETH-COSMO-crCLIM-v1-1_r1i1p1_v1_day_19702100_rcp85.nc
pr_EUR-11_MPI-M-MPI-ESM-LR_MPI-CSC-REMO2009_r1i1p1_v1_day_19702100_rcp85.nc
pr_EUR-11_NCC-NorESM1-M_ICTP-RegCM4-6_r1i1p1_v1_day_19702100_rcp85.nc
```

---

## Key Differences Summary

### Temperature vs Precipitation (RCP4.5)
- **7 models have TAS but not PR**: Primarily SMHI-RCA4 variants (5), plus DMI-HIRHAM5 and GERICS-REMO2015
- **3 models have PR but not TAS**: RMIB-UGent-ALARO-0 and 2 MOHC-HadGEM2-ES variants

### RCP4.5 vs RCP8.5 (Precipitation)
- **RCP8.5 has 6 additional members**: Mainly ETH-COSMO-crCLIM variants and ICTP-RegCM4-6
- RCP8.5 generally has broader model participation in CORDEX

---

## Implications for FireScape Project

1. **Ensemble Size**: All combinations have sufficient members (11-17) for robust quantile estimates
2. **Consistency**: Temperature data is consistent across scenarios, simplifying interpretation
3. **Precipitation Variability**: The larger RCP8.5 precipitation ensemble (17 vs 11) provides better uncertainty characterization for high-emission scenarios
4. **Quantile Calculations**: Different ensemble sizes are appropriate - quantiles are calculated independently for each variable/scenario combination

---

## Data Source

- **Dataset**: CORDEX EUR-11 (0.11° resolution, ~12.5 km)
- **Bias Correction**: QDM (Quantile Delta Mapping)
- **Location**: `/mnt/CEPH_PROJECTS/FACT_CLIMAX/CORDEX-Adjust/QDM/`
- **Time Period**: 1970-2100 (daily data, 47,847 time steps)
- **Spatial Domain**: European domain, 173 × 173 grid points

---

## Processing Scripts

- `calculate_ensemble_quantiles_simple.py`: Sequential processing
- `calculate_ensemble_quantiles_parallel.py`: Parallel processing (4 workers)
- Output location: `/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/climate_projections_ensemble_quantiles/`

---

*Last updated: 2025-11-03*
