#!/bin/bash

# Script to clip raster files to Bolzano boundary
# Parameters
SOURCE_DIR=~/Documents/CEPH_PROJECTS/Firescape/Data
BOUNDARY_FILE=~/Documents/CEPH_PROJECTS/Firescape/Data/00_QGIS/ADMIN/BOLZANO_UTM32.gpkg
OUTPUT_DIR=~/Documents/CEPH_PROJECTS/Firescape/Data_Clipped

# Check if boundary file exists
if [ ! -f "$BOUNDARY_FILE" ]; then
  echo "ERROR: Boundary file does not exist: $BOUNDARY_FILE"
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to process a single TIF file
clip_raster() {
  input_file="$1"
  
  # Create relative path for output file
  rel_path=$(realpath --relative-to="$SOURCE_DIR" "$input_file")
  output_file="$OUTPUT_DIR/$rel_path"
  
  # Create output directory if it doesn't exist
  output_dir=$(dirname "$output_file")
  mkdir -p "$output_dir"
  
  echo "Processing: $rel_path"
  echo "  Output: $output_file"
  
  # Get the raster resolution for -tr parameter
  pixel_size=$(gdalinfo "$input_file" | grep "Pixel Size" | sed -E 's/.*\(([0-9.-]+),([0-9.-]+)\).*/\1 \2/' | tr -d '-')
  
  # Clip raster using gdalwarp
  gdalwarp -overwrite -cutline "$BOUNDARY_FILE" -crop_to_cutline -tr $pixel_size -of GTiff "$input_file" "$output_file"
  
  # Check if clipping was successful
  if [ $? -eq 0 ]; then
    echo "  ✅ Successfully clipped"
  else
    echo "  ❌ Failed to clip"
  fi
  
  echo "---------------------------"
}

echo "Starting to clip rasters to Bolzano boundary"
echo "Boundary file: $BOUNDARY_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "====================================="

# Find all TIF files and process them
find "$SOURCE_DIR" -type f \( -name "*.tif" -o -name "*.tiff" \) | while read -r file; do
  clip_raster "$file"
done

echo "====================================="
echo "Process completed."
echo "Check the clipped files in: $OUTPUT_DIR"