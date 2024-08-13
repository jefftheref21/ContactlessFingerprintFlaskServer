#!/bin/bash

# Source directory containing BMP files
src="/home/spandey8/ridgebase_ICPR/FM_bhavin/hkpoly_data/cl"

# Destination directory for PNG files
dest="/home/spandey8/ridgebase_ICPR/FM_bhavin/hkpoly_data/cl_png"

# Create the destination directory if it doesn't exist
mkdir -p "$dest"

# Find all BMP files in the source directory and convert them to PNG
find "$src" -type f -name "*.bmp" -exec bash -c '
  for img do
    # Replace the source path with destination in the file path
    newpath="${img/$src/$dest}"

    # Create directory structure in the destination
    mkdir -p "$(dirname "$newpath")"

    # Convert BMP to PNG (change extension to .png)
    convert "$img" "${newpath%.bmp}.png"
  done
' bash {} +
