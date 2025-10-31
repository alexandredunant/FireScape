#!/bin/bash
# Directory Cleanup Script for Firescape/Scripts
# Organizes documentation and utilities into proper subdirectories

set -e  # Exit on error

SCRIPTS_DIR="/mnt/CEPH_PROJECTS/Firescape/Scripts"
cd "$SCRIPTS_DIR"

echo "=============================================================================="
echo "FIRESCAPE SCRIPTS DIRECTORY CLEANUP"
echo "=============================================================================="
echo ""

# Confirm with user
echo "This script will:"
echo "  1. Create 00_Documentation/ and 00_Utilities/ directories"
echo "  2. Move 8 markdown files into organized subdirectories"
echo "  3. Move 2 Python utility scripts to 00_Utilities/"
echo ""
echo "Current root directory has:"
ls -1 *.md *.py 2>/dev/null | wc -l
echo " files (should be 10)"
echo ""
read -p "Continue with cleanup? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Step 1: Creating directory structure..."
mkdir -p 00_Documentation/Landcover_Fix
mkdir -p 00_Documentation/Pipeline_Improvements
mkdir -p 00_Documentation/Workflow_Guides
mkdir -p 00_Utilities
echo "✓ Directories created"

echo ""
echo "Step 2: Moving landcover documentation..."
if [ -f "LANDCOVER_ENCODING_ISSUE.md" ]; then
    mv LANDCOVER_ENCODING_ISSUE.md 00_Documentation/Landcover_Fix/01_PROBLEM.md
    echo "  ✓ Moved LANDCOVER_ENCODING_ISSUE.md → 01_PROBLEM.md"
fi

if [ -f "FINAL_LANDCOVER_FIX.md" ]; then
    mv FINAL_LANDCOVER_FIX.md 00_Documentation/Landcover_Fix/02_SOLUTION.md
    echo "  ✓ Moved FINAL_LANDCOVER_FIX.md → 02_SOLUTION.md"
fi

if [ -f "LANDCOVER_FIX_CHECKLIST.md" ]; then
    mv LANDCOVER_FIX_CHECKLIST.md 00_Documentation/Landcover_Fix/03_CHECKLIST.md
    echo "  ✓ Moved LANDCOVER_FIX_CHECKLIST.md → 03_CHECKLIST.md"
fi

echo ""
echo "Step 3: Moving pipeline improvement documentation..."
if [ -f "IMPROVEMENTS_SUMMARY.md" ]; then
    mv IMPROVEMENTS_SUMMARY.md 00_Documentation/Pipeline_Improvements/
    echo "  ✓ Moved IMPROVEMENTS_SUMMARY.md"
fi

if [ -f "README_COMPLETE_SOLUTION.md" ]; then
    mv README_COMPLETE_SOLUTION.md 00_Documentation/Pipeline_Improvements/COMPLETE_SOLUTION.md
    echo "  ✓ Moved README_COMPLETE_SOLUTION.md → COMPLETE_SOLUTION.md"
fi

echo ""
echo "Step 4: Moving workflow guides..."
if [ -f "STANDARDIZATION_GUIDE.md" ]; then
    mv STANDARDIZATION_GUIDE.md 00_Documentation/Workflow_Guides/
    echo "  ✓ Moved STANDARDIZATION_GUIDE.md"
fi

if [ -f "TEMPORAL_VALIDATION_GUIDE.md" ]; then
    mv TEMPORAL_VALIDATION_GUIDE.md 00_Documentation/Workflow_Guides/
    echo "  ✓ Moved TEMPORAL_VALIDATION_GUIDE.md"
fi

if [ -f "WORKFLOW_SUMMARY.md" ]; then
    mv WORKFLOW_SUMMARY.md 00_Documentation/Workflow_Guides/
    echo "  ✓ Moved WORKFLOW_SUMMARY.md"
fi

echo ""
echo "Step 5: Moving utility scripts..."
if [ -f "apply_landcover_fix.py" ]; then
    mv apply_landcover_fix.py 00_Utilities/
    echo "  ✓ Moved apply_landcover_fix.py"
fi

if [ -f "shared_prediction_utils.py" ]; then
    mv shared_prediction_utils.py 00_Utilities/
    echo "  ✓ Moved shared_prediction_utils.py"
fi

echo ""
echo "Step 6: Moving cleanup files..."
if [ -f "DIRECTORY_CLEANUP_PLAN.md" ]; then
    mv DIRECTORY_CLEANUP_PLAN.md 00_Documentation/
    echo "  ✓ Moved DIRECTORY_CLEANUP_PLAN.md"
fi

echo ""
echo "=============================================================================="
echo "CLEANUP COMPLETE!"
echo "=============================================================================="
echo ""
echo "New structure:"
echo "  00_Documentation/"
echo "    ├── Landcover_Fix/           (3 files)"
echo "    ├── Pipeline_Improvements/   (2 files)"
echo "    ├── Workflow_Guides/         (3 files)"
echo "    └── DIRECTORY_CLEANUP_PLAN.md"
echo "  00_Utilities/                  (2 scripts)"
echo ""

echo "Verification:"
echo "  Root .md/.py files: $(ls -1 *.md *.py 2>/dev/null | wc -l) (should be 0-1)"
echo "  Documentation files: $(find 00_Documentation -name '*.md' | wc -l) (should be 9)"
echo "  Utility scripts: $(find 00_Utilities -name '*.py' | wc -l) (should be 2)"
echo ""

echo "Next steps:"
echo "  1. Review the new structure: ls -lR 00_*"
echo "  2. Test utilities: python 00_Utilities/apply_landcover_fix.py --help"
echo "  3. Create main README.md (see next script)"
echo ""
