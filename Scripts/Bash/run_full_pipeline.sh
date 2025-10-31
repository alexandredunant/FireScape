#!/bin/bash
################################################################################
# FIRESCAPE COMPLETE PIPELINE
################################################################################
# This script runs the complete Firescape workflow from data preparation
# through model training, validation, and climate projections.
#
# Pipeline steps:
# 1. Create spacetime dataset (parquet) - ~30-60 min
# 2. Create raster stacks (NetCDF) - ~15-30 min
# 3. Validate training data - ~5 min
# 4. Train relative probability model - ~5-10 min
# 5. Temporal lookback validation - ~30-40 hours (5 years)
# 6. Climate scenario projections - ~varies by configuration
#
# All output is visible and logged to files for later review.
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PYTHON_ENV="/home/adunant/miniconda3/envs/dask-geo/bin/python"
PROJECT_DIR="/mnt/CEPH_PROJECTS/Firescape"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory
mkdir -p "${LOG_DIR}"

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}                     FIRESCAPE COMPLETE PIPELINE                               ${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo -e "Pipeline start time: $(date)"
echo -e "Python: ${PYTHON_ENV}"
echo -e "Working directory: ${PROJECT_DIR}"
echo -e "Logs: ${LOG_DIR}"
echo ""

cd "${PROJECT_DIR}"

PIPELINE_START=$(date +%s)

################################################################################
# STEP 1: Create Spacetime Dataset (Parquet)
################################################################################

echo -e "${YELLOW}================================================================================${NC}"
echo -e "${YELLOW}STEP 1/6: Creating Spacetime Dataset (Parquet)${NC}"
echo -e "${YELLOW}================================================================================${NC}"
echo -e "Script: Scripts/01_Data_Preparation/create_spacetime_dataset.py"
echo -e "Expected output: Data/OUTPUT/01_Training_Data/spacetime_dataset.parquet"
echo -e "Expected fires: ~614 (1999-2024, 26 years)"
echo -e "Expected runtime: ~30-60 minutes"
echo ""

STEP1_LOG="${LOG_DIR}/step1_spacetime_dataset_${TIMESTAMP}.log"
echo -e "Logging to: ${STEP1_LOG}"
echo ""

STEP1_START=$(date +%s)

if ${PYTHON_ENV} Scripts/01_Data_Preparation/create_spacetime_dataset.py 2>&1 | tee "${STEP1_LOG}"; then
    STEP1_END=$(date +%s)
    STEP1_DURATION=$((STEP1_END - STEP1_START))
    echo ""
    echo -e "${GREEN}✓ STEP 1 COMPLETE${NC} (Runtime: $((STEP1_DURATION/60))m $((STEP1_DURATION%60))s)"
    echo ""
else
    echo ""
    echo -e "${RED}✗ STEP 1 FAILED${NC}"
    echo -e "${RED}Check log: ${STEP1_LOG}${NC}"
    exit 1
fi

################################################################################
# STEP 2: Create Raster Stacks (NetCDF)
################################################################################

echo -e "${YELLOW}================================================================================${NC}"
echo -e "${YELLOW}STEP 2/6: Creating Raster Stacks (NetCDF)${NC}"
echo -e "${YELLOW}================================================================================${NC}"
echo -e "Script: Scripts/01_Data_Preparation/create_raster_stacks.py"
echo -e "Input: Data/OUTPUT/01_Training_Data/spacetime_dataset.parquet"
echo -e "Output: Data/OUTPUT/01_Training_Data/spacetime_stacks.nc"
echo -e "Expected runtime: ~15-30 minutes"
echo ""

STEP2_LOG="${LOG_DIR}/step2_raster_stacks_${TIMESTAMP}.log"
echo -e "Logging to: ${STEP2_LOG}"
echo ""

STEP2_START=$(date +%s)

if ${PYTHON_ENV} Scripts/01_Data_Preparation/create_raster_stacks.py 2>&1 | tee "${STEP2_LOG}"; then
    STEP2_END=$(date +%s)
    STEP2_DURATION=$((STEP2_END - STEP2_START))
    echo ""
    echo -e "${GREEN}✓ STEP 2 COMPLETE${NC} (Runtime: $((STEP2_DURATION/60))m $((STEP2_DURATION%60))s)"
    echo ""
else
    echo ""
    echo -e "${RED}✗ STEP 2 FAILED${NC}"
    echo -e "${RED}Check log: ${STEP2_LOG}${NC}"
    exit 1
fi

################################################################################
# STEP 3: Validate Training Data
################################################################################

echo -e "${YELLOW}================================================================================${NC}"
echo -e "${YELLOW}STEP 3/6: Validating Training Data${NC}"
echo -e "${YELLOW}================================================================================${NC}"
echo -e "Script: Scripts/06_Validation/validate_training_data.py"
echo -e "Validates: Balance, temporal coverage, spatial distribution, completeness"
echo -e "Output: Data/OUTPUT/06_Validation_Analysis/training_data_validation/"
echo -e "Expected runtime: ~5 minutes"
echo ""

STEP3_LOG="${LOG_DIR}/step3_validate_data_${TIMESTAMP}.log"
echo -e "Logging to: ${STEP3_LOG}"
echo ""

STEP3_START=$(date +%s)

if ${PYTHON_ENV} Scripts/06_Validation/validate_training_data.py 2>&1 | tee "${STEP3_LOG}"; then
    STEP3_END=$(date +%s)
    STEP3_DURATION=$((STEP3_END - STEP3_START))
    echo ""
    echo -e "${GREEN}✓ STEP 3 COMPLETE${NC} (Runtime: $((STEP3_DURATION/60))m $((STEP3_DURATION%60))s)"
    echo ""
else
    echo ""
    echo -e "${RED}✗ STEP 3 FAILED${NC}"
    echo -e "${RED}Check log: ${STEP3_LOG}${NC}"
    exit 1
fi

################################################################################
# STEP 4: Train Relative Probability Model
################################################################################

echo -e "${YELLOW}================================================================================${NC}"
echo -e "${YELLOW}STEP 4/6: Training Relative Probability Model${NC}"
echo -e "${YELLOW}================================================================================${NC}"
echo -e "Script: Scripts/02_Model_Training/train_relative_probability_model.py"
echo -e "Method: Bayesian attention mechanism with MCMC sampling"
echo -e "Output: Data/OUTPUT/02_Model_RelativeProbability/"
echo -e "Key outputs: trace_relative.nc, scaler_relative.joblib, model_results.joblib"
echo -e "Note: Outputs RELATIVE risk only (absolute calibration removed - was unreliable)"
echo -e "Expected runtime: ~5-10 minutes (MCMC sampling)"
echo ""

STEP4_LOG="${LOG_DIR}/step4_train_model_${TIMESTAMP}.log"
echo -e "Logging to: ${STEP4_LOG}"
echo ""

STEP4_START=$(date +%s)

if ${PYTHON_ENV} Scripts/02_Model_Training/train_relative_probability_model.py 2>&1 | tee "${STEP4_LOG}"; then
    STEP4_END=$(date +%s)
    STEP4_DURATION=$((STEP4_END - STEP4_START))
    echo ""
    echo -e "${GREEN}✓ STEP 4 COMPLETE${NC} (Runtime: $((STEP4_DURATION/60))m $((STEP4_DURATION%60))s)"
    echo -e "${CYAN}  Model outputs: Relative fire risk probabilities${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}✗ STEP 4 FAILED${NC}"
    echo -e "${RED}Check log: ${STEP4_LOG}${NC}"
    exit 1
fi

################################################################################
# STEP 5: Temporal Lookback Validation (Multi-Year)
################################################################################

echo -e "${YELLOW}================================================================================${NC}"
echo -e "${YELLOW}STEP 5/6: Temporal Lookback Validation (Multi-Year)${NC}"
echo -e "${YELLOW}================================================================================${NC}"
echo -e "Script: Scripts/06_Validation/temporal_lookback_validation.py"
echo -e "Validation years: 2003, 2007, 2012, 2017, 2022 (5 years)"
echo -e "Total fires: ~220 across 5 years"
echo -e "Analysis: TP/FP/TN/FN, ROC curves, confusion matrices, GIFs"
echo -e "Output: Data/OUTPUT/08_Temporal_Lookback_Validation/"
echo -e "Expected runtime: ~30-40 hours (6-8 hours per year)"
echo ""
echo -e "${CYAN}NOTE: This is the longest step. Consider running overnight.${NC}"
echo ""

read -p "Continue with temporal lookback validation? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}⏸  Skipping Step 5 (Temporal Lookback Validation)${NC}"
    echo -e "${YELLOW}   You can run it later with:${NC}"
    echo -e "${YELLOW}   ${PYTHON_ENV} Scripts/06_Validation/temporal_lookback_validation.py${NC}"
    echo ""
    STEP5_DURATION=0
else
    STEP5_LOG="${LOG_DIR}/step5_lookback_validation_${TIMESTAMP}.log"
    echo -e "Logging to: ${STEP5_LOG}"
    echo ""

    STEP5_START=$(date +%s)

    if ${PYTHON_ENV} Scripts/06_Validation/temporal_lookback_validation.py 2>&1 | tee "${STEP5_LOG}"; then
        STEP5_END=$(date +%s)
        STEP5_DURATION=$((STEP5_END - STEP5_START))
        echo ""
        echo -e "${GREEN}✓ STEP 5 COMPLETE${NC} (Runtime: $((STEP5_DURATION/3600))h $((STEP5_DURATION%3600/60))m)"
        echo ""
    else
        echo ""
        echo -e "${RED}✗ STEP 5 FAILED${NC}"
        echo -e "${RED}Check log: ${STEP5_LOG}${NC}"
        exit 1
    fi
fi

################################################################################
# STEP 6: Climate Scenario Projections
################################################################################

echo -e "${YELLOW}================================================================================${NC}"
echo -e "${YELLOW}STEP 6/6: Climate Scenario Projections${NC}"
echo -e "${YELLOW}================================================================================${NC}"
echo -e "Script: Scripts/03_Climate_Projections/project_climate_scenarios.py"
echo -e "Scenario: RCP8.5"
echo -e "Years: 2020, 2050, 2080 (July monthly aggregates)"
echo -e "Quantiles: 25th, 50th, 99th percentile"
echo -e "Output: Data/OUTPUT/04_Climate_Projections_Relative/rcp85/"
echo -e "Expected runtime: ~varies by configuration"
echo ""

read -p "Continue with climate projections? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}⏸  Skipping Step 6 (Climate Projections)${NC}"
    echo -e "${YELLOW}   You can run it later with:${NC}"
    echo -e "${YELLOW}   ${PYTHON_ENV} Scripts/03_Climate_Projections/project_climate_scenarios.py${NC}"
    echo ""
    STEP6_DURATION=0
else
    STEP6_LOG="${LOG_DIR}/step6_climate_projections_${TIMESTAMP}.log"
    echo -e "Logging to: ${STEP6_LOG}"
    echo ""

    STEP6_START=$(date +%s)

    if ${PYTHON_ENV} Scripts/03_Climate_Projections/project_climate_scenarios.py 2>&1 | tee "${STEP6_LOG}"; then
        STEP6_END=$(date +%s)
        STEP6_DURATION=$((STEP6_END - STEP6_START))
        echo ""
        echo -e "${GREEN}✓ STEP 6 COMPLETE${NC} (Runtime: $((STEP6_DURATION/60))m $((STEP6_DURATION%60))s)"
        echo ""
    else
        echo ""
        echo -e "${RED}✗ STEP 6 FAILED${NC}"
        echo -e "${RED}Check log: ${STEP6_LOG}${NC}"
        exit 1
    fi
fi

################################################################################
# FINAL SUMMARY
################################################################################

PIPELINE_END=$(date +%s)
TOTAL_DURATION=$((PIPELINE_END - PIPELINE_START))

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}                   PIPELINE COMPLETE - SUCCESS!                                ${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""
echo -e "Pipeline end time: $(date)"
echo -e "Total runtime: $((TOTAL_DURATION/3600))h $((TOTAL_DURATION%3600/60))m $((TOTAL_DURATION%60))s"
echo ""
echo -e "${BLUE}Step Runtimes:${NC}"
echo -e "  Step 1 (Create dataset):       $((STEP1_DURATION/60))m $((STEP1_DURATION%60))s"
echo -e "  Step 2 (Create stacks):        $((STEP2_DURATION/60))m $((STEP2_DURATION%60))s"
echo -e "  Step 3 (Validate data):        $((STEP3_DURATION/60))m $((STEP3_DURATION%60))s"
echo -e "  Step 4 (Train model):          $((STEP4_DURATION/60))m $((STEP4_DURATION%60))s"
if [ $STEP5_DURATION -gt 0 ]; then
    echo -e "  Step 5 (Lookback validation):  $((STEP5_DURATION/3600))h $((STEP5_DURATION%3600/60))m"
else
    echo -e "  Step 5 (Lookback validation):  SKIPPED"
fi
if [ $STEP6_DURATION -gt 0 ]; then
    echo -e "  Step 6 (Climate projections):  $((STEP6_DURATION/60))m $((STEP6_DURATION%60))s"
else
    echo -e "  Step 6 (Climate projections):  SKIPPED"
fi
echo ""
echo -e "${BLUE}Output Directories:${NC}"
echo -e "  Training data:    Data/OUTPUT/01_Training_Data/"
echo -e "  Model artifacts:  Data/OUTPUT/02_Model_RelativeProbability/"
echo -e "  Validation:       Data/OUTPUT/06_Validation_Analysis/"
echo -e "  Lookback:         Data/OUTPUT/08_Temporal_Lookback_Validation/"
echo -e "  Projections:      Data/OUTPUT/04_Climate_Projections_Relative/"
echo ""
echo -e "${BLUE}Logs:${NC}"
echo -e "  Step 1: ${STEP1_LOG}"
echo -e "  Step 2: ${STEP2_LOG}"
echo -e "  Step 3: ${STEP3_LOG}"
echo -e "  Step 4: ${STEP4_LOG}"
if [ $STEP5_DURATION -gt 0 ]; then
    echo -e "  Step 5: ${STEP5_LOG}"
fi
if [ $STEP6_DURATION -gt 0 ]; then
    echo -e "  Step 6: ${STEP6_LOG}"
fi
echo ""
echo -e "${CYAN}Key Model Parameters:${NC}"
echo -e "  Model type: Relative probability (Bayesian attention)"
echo -e "  Training fires: ~614 (1999-2024)"
echo -e "  Categorical handling: Mode (landcoverfull)"
echo -e "  Output: Relative fire risk scores (0-1 scale)"
echo -e "  Note: Absolute calibration removed (unreliable)"
echo ""
echo -e "${GREEN}================================================================================${NC}"
