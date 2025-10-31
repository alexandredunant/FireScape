#!/bin/bash
# Move optional scripts and documentation to Optional/ folders

set -e
cd /mnt/CEPH_PROJECTS/Firescape/Scripts

echo "================================================================================"
echo "ORGANIZING OPTIONAL SCRIPTS AND DOCUMENTATION"
echo "================================================================================"
echo ""

# Create Optional folders
echo "Creating Optional/ directories..."
mkdir -p Optional/Documentation
mkdir -p Optional/Utilities
mkdir -p Optional/ClimateProjections
mkdir -p Optional/Scripts

# Move documentation
echo "Moving documentation..."
mv 00_Documentation/* Optional/Documentation/ 2>/dev/null || true
mv 00_Utilities/* Optional/Utilities/ 2>/dev/null || true

# Move climate projections (entire new directory)
echo "Moving optional climate projection pipeline..."
mv 03_Climate_Projections Optional/ClimateProjections/ 2>/dev/null || true

# Move root documentation files
echo "Moving root documentation..."
mv README.md Optional/README_MAIN.md 2>/dev/null || true
mv START_HERE.md Optional/ 2>/dev/null || true
mv cleanup_directory.sh Optional/Scripts/ 2>/dev/null || true

# Remove empty directories
rmdir 00_Documentation 2>/dev/null || true
rmdir 00_Utilities 2>/dev/null || true

echo ""
echo "================================================================================"
echo "ORGANIZATION COMPLETE!"
echo "================================================================================"
echo ""
echo "New structure:"
echo "  Scripts/"
echo "    ├── Optional/                    ← All optional/reference material"
echo "    │   ├── Documentation/           ← All .md docs"
echo "    │   ├── Utilities/               ← Helper scripts"
echo "    │   ├── ClimateProjections/      ← Multi-scenario pipeline"
echo "    │   ├── Scripts/                 ← Cleanup scripts"
echo "    │   ├── README_MAIN.md           ← Main reference"
echo "    │   └── START_HERE.md            ← Simple guide"
echo "    ├── 01_Data_Preparation/         ← YOUR CORE SCRIPTS"
echo "    ├── 02_Model_Training/"
echo "    ├── 04_Zone_Climate_Projections/"
echo "    └── ..."
echo ""
echo "Your core pipeline is now clearly separated from optional material!"
echo ""
