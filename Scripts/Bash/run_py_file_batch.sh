#!/bin/bash

# Script to run Dask export for multiple years
# Usage: ./run_export_by_years.sh [start_year] [end_year]

# Set default years if not provided
START_YEAR=${1:-2010}
END_YEAR=${2:-2025}

echo "===== Running export for years $START_YEAR to $END_YEAR ====="

# Loop through each year and run the Python script
for YEAR in $(seq $START_YEAR $END_YEAR); do
    echo ""
    echo "===== Starting export for year $YEAR ====="
    echo "$(date)"
    
    # Run the Python script for this year using the absolute path
    python ~/Documents/CEPH_PROJECTS/Firescape/Scripts/Python/00_export_dynamics_parallel_Dask.py $YEAR
    
    # Check if the script executed successfully
    if [ $? -eq 0 ]; then
        echo "✓ Export for year $YEAR completed successfully"
    else
        echo "✗ Export for year $YEAR failed with exit code $?"
        # Uncomment the line below if you want to stop on errors
        # exit 1
    fi
    
    echo "Finished at: $(date)"
    echo "---------------------------------------------------"
    
    # Optional: Add a short pause between years to allow for system recovery
    sleep 5
done

echo ""
echo "===== All exports completed ====="
echo "Processed years from $START_YEAR to $END_YEAR"