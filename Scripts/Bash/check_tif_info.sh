#!/bin/bash

# Script to check resolution, CRS, and statistics of TIFF files
find ~/Documents/CEPH_PROJECTS/Firescape/Data -type f \( -name "*.tif" -o -name "*.tiff" \) -exec bash -c '
  file="$1"
  echo -e "\n===== $file ====="
  
  # Get basic file info
  file_size=$(du -h "$file" | cut -f1)
  echo "File size: $file_size"
  
  # Get raster dimensions
  dimensions=$(gdalinfo "$file" | grep -E "Size is" | head -1)
  echo "$dimensions"
  
  # Get pixel size (resolution)
  pixel_size=$(gdalinfo "$file" | grep "Pixel Size")
  echo "$pixel_size"
  
  # Get number of bands
  bands=$(gdalinfo "$file" | grep "Band " | wc -l)
  echo "Number of bands: $bands"
  
  # Get corner coordinates
  echo "Corner Coordinates:"
  gdalinfo "$file" | grep -A 5 "Corner Coordinates:" | tail -5
  
  # Get min/max values if available
  min_max=$(gdalinfo -stats "$file" | grep -E "Min=|Max=" | head -1)
  if [ -n "$min_max" ]; then
    echo "Statistics: $min_max"
  fi
  
  # Get data type
  data_type=$(gdalinfo "$file" | grep "Type=" | head -1)
  echo "$data_type"
  
  # Get CRS information
  crs_info=$(gdalinfo -proj4 "$file")
  
  # Extract PROJ.4 string
  proj4=$(echo "$crs_info" | grep -A 1 "PROJ.4 string" | tail -1 | tr -d " ")
  expected_proj="+proj=utm+zone=32+datum=WGS84+units=m+no_defs"
  
  # Display CRS and check if it matches expected projection
  if [ -n "$proj4" ]; then
    echo "PROJ.4: $proj4"
    
    # Remove spaces to standardize comparison
    clean_proj4=$(echo "$proj4" | tr -d " ")
    
    if [[ "$clean_proj4" == *"$expected_proj"* ]]; then
      echo "✅ CRS MATCHES EXPECTED UTM ZONE 32 WGS84"
    else
      echo "⚠️ WARNING: CRS DOES NOT MATCH EXPECTED UTM ZONE 32 WGS84"
    fi
  else
    echo "CRS info:"
    wkt_info=$(gdalinfo "$file" | grep -A 3 "Coordinate System is" | tail -3)
    echo "$wkt_info"
    echo "⚠️ WARNING: Could not extract PROJ.4 string to verify CRS"
  fi
  
  # Check for no data value
  nodata=$(gdalinfo "$file" | grep "NoData Value" | head -1)
  if [ -n "$nodata" ]; then
    echo "$nodata"
  fi
  
  # Check if statistics are computed
  has_stats=$(gdalinfo "$file" | grep -c "STATISTICS_")
  if [ "$has_stats" -gt 0 ]; then
    echo "Additional statistics available"
  else
    echo "No pre-computed statistics found"
  fi
' _ {} \;