#!/bin/bash
# Cleanup Script: Remove Quantile-Based Files (Now Replaced by Ensemble Members)
#
# This script moves old quantile-based analysis files to an archive directory
# instead of deleting them permanently.
#
# Run this script from: /mnt/CEPH_PROJECTS/Firescape/04_Zone_Climate_Projections/

echo "=========================================="
echo "Cleanup: Old Quantile-Based Files"
echo "=========================================="
echo

# Create archive directory
ARCHIVE_DIR="./ARCHIVE_quantile_based_$(date +%Y%m%d)"
OUTPUT_ARCHIVE_DIR="../output/04_Zone_Climate_Projections/ARCHIVE_quantile_based_$(date +%Y%m%d)"

echo "Creating archive directories..."
mkdir -p "$ARCHIVE_DIR"
mkdir -p "$OUTPUT_ARCHIVE_DIR"
echo "  ✓ $ARCHIVE_DIR"
echo "  ✓ $OUTPUT_ARCHIVE_DIR"
echo

# ===================================================================
# SCRIPTS TO ARCHIVE
# ===================================================================

echo "Archiving old quantile-based scripts..."

FILES_TO_ARCHIVE=(
    "project_zone_fire_risk.py"
    "project_zone_fire_risk_SIMPLIFIED.py"
    "create_publication_climate_figures.py"
    "create_publication_maps_and_plots.py"
    "create_final_climate_figures.py"
    "plot_climate_projections.py"
    "check_all_quantiles.py"
)

for file in "${FILES_TO_ARCHIVE[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" "$ARCHIVE_DIR/"
        echo "  ✓ Archived: $file"
    else
        echo "  - Not found: $file"
    fi
done

echo

# ===================================================================
# BACKUP FILES TO REMOVE
# ===================================================================

echo "Removing backup files..."

BACKUP_FILES=(
    "project_zone_fire_risk_ENSEMBLE_MEMBERS.py.broken"
)

for file in "${BACKUP_FILES[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "  ✓ Removed: $file"
    else
        echo "  - Not found: $file"
    fi
done

echo

# ===================================================================
# OUTPUT DATA FILES TO ARCHIVE
# ===================================================================

echo "Archiving old quantile-based output data..."

OUTPUT_FILES=(
    "../output/04_Zone_Climate_Projections/climate_drivers_data.csv"
    "../output/04_Zone_Climate_Projections/climate_projection_data.csv"
)

for file in "${OUTPUT_FILES[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" "$OUTPUT_ARCHIVE_DIR/"
        echo "  ✓ Archived: $file"
    else
        echo "  - Not found: $file"
    fi
done

echo
echo "=========================================="
echo "CLEANUP COMPLETE"
echo "=========================================="
echo
echo "Archived files can be found in:"
echo "  - $ARCHIVE_DIR"
echo "  - $OUTPUT_ARCHIVE_DIR"
echo
echo "Current files:"
ls -lh *.py 2>/dev/null | grep -v ARCHIVE | awk '{print "  " $9 " (" $5 ")"}'
echo
echo "You can safely delete the archive directories if you no longer need them."
echo "=========================================="
