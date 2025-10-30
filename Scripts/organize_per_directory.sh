#!/bin/bash
# Move optional scripts within each directory to Optional/ subfolders

set -e
cd /mnt/CEPH_PROJECTS/Firescape/Scripts

echo "================================================================================"
echo "ORGANIZING OPTIONAL SCRIPTS WITHIN EACH DIRECTORY"
echo "================================================================================"
echo ""

# 01_Data_Preparation
echo "01_Data_Preparation/..."
cd 01_Data_Preparation
mkdir -p Optional
# Keep only main script
# Move optional/reference files
[ -f "CORINE_LANDCOVER_FIRE_RISK_MAPPING.py" ] && mv CORINE_LANDCOVER_FIRE_RISK_MAPPING.py Optional/
[ -f "CORINE_fire_risk_mapping.csv" ] && mv CORINE_fire_risk_mapping.csv Optional/
[ -f "clip_fire_brigade_to_bolzano.py" ] && mv clip_fire_brigade_to_bolzano.py Optional/
[ -f "create_spacetime_dataset.py" ] && mv create_spacetime_dataset.py Optional/
echo "  ✓ Main: create_raster_stacks.py"
echo "  ✓ Optional: $(ls Optional/ 2>/dev/null | wc -l) files moved"
cd ..

# 02_Model_Training
echo "02_Model_Training/..."
cd 02_Model_Training
mkdir -p Optional
# Keep only main training script
[ -f "train_Dask_PyMC_timeseries.py" ] && mv train_Dask_PyMC_timeseries.py Optional/
[ -f "fix_landcover_encoding.py" ] && mv fix_landcover_encoding.py Optional/
echo "  ✓ Main: train_relative_probability_model.py"
echo "  ✓ Optional: $(ls Optional/ 2>/dev/null | wc -l) files moved"
cd ..

# 04_Zone_Climate_Projections
echo "04_Zone_Climate_Projections/..."
cd 04_Zone_Climate_Projections
mkdir -p Optional
# Keep main projection script, move optional
[ -f "analyze_warning_level_evolution.py" ] && mv analyze_warning_level_evolution.py Optional/
[ -f "PERCENTILE_INTERPRETATION_ISSUE.md" ] && mv PERCENTILE_INTERPRETATION_ISSUE.md Optional/
echo "  ✓ Main: project_zone_fire_risk.py"
echo "  ✓ Optional: $(ls Optional/ 2>/dev/null | wc -l) files moved"
cd ..

# 05_Lightning_Comparison (all optional - experimental)
echo "05_Lightning_Comparison/..."
cd 05_Lightning_Comparison
if [ -d "01_Data_Preparation" ]; then
    echo "  ℹ️  Entire directory is experimental/optional"
fi
cd ..

echo ""
echo "================================================================================"
echo "ORGANIZATION COMPLETE!"
echo "================================================================================"
echo ""
echo "Structure now:"
echo ""
echo "01_Data_Preparation/"
echo "├── create_raster_stacks.py              ← YOUR MAIN SCRIPT"
echo "└── Optional/                            ← Reference files"
echo ""
echo "02_Model_Training/"
echo "├── train_relative_probability_model.py  ← YOUR MAIN SCRIPT"
echo "└── Optional/                            ← Alternative implementations"
echo ""
echo "04_Zone_Climate_Projections/"
echo "├── project_zone_fire_risk.py            ← YOUR MAIN SCRIPT"
echo "└── Optional/                            ← Analysis tools"
echo ""
echo "Each directory now has ONE main script + Optional/ subfolder!"
echo ""
