#!/bin/bash

# Target directory (defaults to current directory '.' if no argument is given)
TARGET_DIR="${1:-.}"

# Find matching files recursively and safely handle spaces in paths
find "$TARGET_DIR" -type f \( -name "*_Random_plots.pdf" -o -name "*_Random_noSpur_plots.pdf" \) -print0 | while IFS= read -r -d '' file; do
    dir=$(dirname "$file")
    target="$dir/train.pdf"
    
    # Make the copy
    cp "$file" "$target"
    echo "Copied: $file -> $target"
done
