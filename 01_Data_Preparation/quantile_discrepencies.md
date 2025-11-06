Files with discrepencies:

================================================================================
ENSEMBLE QUANTILE FILES COMPARISON REPORT
================================================================================

Examples
old path : /mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/pr/rcp45/pr_EUR-11_pctl25_rcp45.nc
new path : /mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/climate_projections_ensemble_quantiles/precipitation/rcp45/pr_EUR-11_pctl25_rcp45.nc

TEMPERATURE (tas)
--------------------------------------------------------------------------------

  Scenario: RCP45

    pctl25: NEW_MISSING
      ✗ New file not found: /mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/climate_projections_ensemble_quantiles/temperature/rcp45/tas_EUR-11_pctl25_rcp45.nc

    pctl50: NEW_MISSING
      ✗ New file not found: /mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/climate_projections_ensemble_quantiles/temperature/rcp45/tas_EUR-11_pctl50_rcp45.nc

    pctl75: NEW_MISSING
      ✗ New file not found: /mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/climate_projections_ensemble_quantiles/temperature/rcp45/tas_EUR-11_pctl75_rcp45.nc

    pctl99: NEW_MISSING
      ✗ New file not found: /mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/climate_projections_ensemble_quantiles/temperature/rcp45/tas_EUR-11_pctl99_rcp45.nc

  Scenario: RCP85

    pctl25: DIFFERENT
      Old size: 5.34 GB
      New size: 2.34 GB
      Shape: {'time': 47847, 'y': 173, 'x': 173}
      Time range (old): 1970-01-01 to 2100-12-31
      Time range (new): 1970-01-01 to 2100-12-31
      Mean absolute difference: 0.295038
      Max absolute difference: 3.760000
      Mean relative difference: 7232772.5000%
      RMSE: 0.416821
      Correlation: 0.999348

    pctl50: IDENTICAL
      Old size: 5.34 GB
      New size: 2.25 GB
      Shape: {'time': 47847, 'y': 173, 'x': 173}
      Time range (old): 1970-01-01 to 2100-12-31
      Time range (new): 1970-01-01 to 2100-12-31
      Mean absolute difference: 0.000000
      Max absolute difference: 0.000000
      Mean relative difference: 0.0000%
      RMSE: 0.000000
      Correlation: 1.000000

    pctl75: DIFFERENT
      Old size: 5.34 GB
      New size: 2.27 GB
      Shape: {'time': 47847, 'y': 173, 'x': 173}
      Time range (old): 1970-01-01 to 2100-12-31
      Time range (new): 1970-01-01 to 2100-12-31
      Mean absolute difference: 0.282012
      Max absolute difference: 3.134001
      Mean relative difference: 4363551.0000%
      RMSE: 0.399560
      Correlation: 0.999342

    pctl99: DIFFERENT
      Old size: 5.34 GB
      New size: 2.39 GB
      Shape: {'time': 47847, 'y': 173, 'x': 173}
      Time range (old): 1970-01-01 to 2100-12-31
      Time range (new): 1970-01-01 to 2100-12-31
      Mean absolute difference: 0.149454
      Max absolute difference: 1.161510
      Mean relative difference: 1338037.8750%
      RMSE: 0.201680
      Correlation: 0.999851

PRECIPITATION (pr)
--------------------------------------------------------------------------------

  Scenario: RCP45

    pctl25: DIFFERENT
      Old size: 5.34 GB
      New size: 0.37 GB
      Shape: {'time': 47847, 'y': 173, 'x': 173}
      Time range (old): 1970-01-01 to 2100-12-31
      Time range (new): 1970-01-01 to 2100-12-31
      Mean absolute difference: 0.035610
      Max absolute difference: 5.671500
      Mean relative difference: 14876628992.0000%
      RMSE: 0.167453
      Correlation: 0.924713

    pctl50: IDENTICAL
      Old size: 5.34 GB
      New size: 0.96 GB
      Shape: {'time': 47847, 'y': 173, 'x': 173}
      Time range (old): 1970-01-01 to 2100-12-31
      Time range (new): 1970-01-01 to 2100-12-31
      Mean absolute difference: 0.000000
      Max absolute difference: 0.000000
      Mean relative difference: 0.0000%
      RMSE: 0.000000
      Correlation: 1.000000

    pctl75: DIFFERENT
      Old size: 5.34 GB
      New size: 1.88 GB
      Shape: {'time': 47847, 'y': 173, 'x': 173}
      Time range (old): 1970-01-01 to 2100-12-31
      Time range (new): 1970-01-01 to 2100-12-31
      Mean absolute difference: 0.923310
      Max absolute difference: 30.772499
      Mean relative difference: 22.3164%
      RMSE: 1.811699
      Correlation: 0.974657

    pctl99: DIFFERENT
      Old size: 5.34 GB
      New size: 2.22 GB
      Shape: {'time': 47847, 'y': 173, 'x': 173}
      Time range (old): 1970-01-01 to 2100-12-31
      Time range (new): 1970-01-01 to 2100-12-31
      Mean absolute difference: 0.774777
      Max absolute difference: 34.350586
      Mean relative difference: 4.9101%
      RMSE: 1.331806
      Correlation: 0.999226

  Scenario: RCP85

    pctl25: IDENTICAL
      Old size: 5.34 GB
      New size: 0.20 GB
      Shape: {'time': 47847, 'y': 173, 'x': 173}
      Time range (old): 1970-01-01 to 2100-12-31
      Time range (new): 1970-01-01 to 2100-12-31
      Mean absolute difference: 0.000000
      Max absolute difference: 0.000000
      Mean relative difference: 0.0000%
      RMSE: 0.000000
      Correlation: 1.000000

    pctl50: IDENTICAL
      Old size: 5.34 GB
      New size: 0.88 GB
      Shape: {'time': 47847, 'y': 173, 'x': 173}
      Time range (old): 1970-01-01 to 2100-12-31
      Time range (new): 1970-01-01 to 2100-12-31
      Mean absolute difference: 0.000000
      Max absolute difference: 0.000000
      Mean relative difference: 0.0000%
      RMSE: 0.000000
      Correlation: 1.000000

    pctl75: IDENTICAL
      Old size: 5.34 GB
      New size: 1.68 GB
      Shape: {'time': 47847, 'y': 173, 'x': 173}
      Time range (old): 1970-01-01 to 2100-12-31
      Time range (new): 1970-01-01 to 2100-12-31
      Mean absolute difference: 0.000000
      Max absolute difference: 0.000000
      Mean relative difference: 0.0000%
      RMSE: 0.000000
      Correlation: 1.000000

    pctl99: DIFFERENT
      Old size: 5.34 GB
      New size: 2.20 GB
      Shape: {'time': 47847, 'y': 173, 'x': 173}
      Time range (old): 1970-01-01 to 2100-12-31
      Time range (new): 1970-01-01 to 2100-12-31
      Mean absolute difference: 1.358311
      Max absolute difference: 56.316315
      Mean relative difference: 6.9298%
      RMSE: 2.352046
      Correlation: 0.997692

Comparison summary saved to: /mnt/CEPH_PROJECTS/Firescape/output/01_Data_Preparation/quantile_comparison_summary.txt
