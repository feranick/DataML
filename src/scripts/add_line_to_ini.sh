#!/bin/bash

# Check if a target directory was provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <target_directory>"
  exit 1
fi

TARGET_DIR="$1"

# Check if the provided argument is a valid directory
if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Directory '$TARGET_DIR' does not exist."
  exit 1
fi

# Find all matching files and process them
find "$TARGET_DIR" -type f -name "DataML_DAE.ini" | while read -r file; do
  
  # Check if the loss_metric line already exists to prevent duplicate entries
  if grep -q "^lossMetric[ \t]*=" "$file"; then
    echo "Skipping (already contains lossMetric): $file"
  else
    # Cross-platform solution using awk to avoid BSD vs GNU sed conflicts
    awk '/^activation[ \t]*=/ {print; print "lossMetric = mean_squared_error"; next} 1' "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
    echo "Updated: $file"
  fi
  
done

echo "Process complete."
