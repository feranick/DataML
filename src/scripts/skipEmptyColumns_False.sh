#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <folder_path> <filename> "
    exit 1
fi

folder_path="$1"
filename="$2"

# Check if the provided folder path is a valid directory
if [ ! -d "$folder_path" ]; then
    echo "Error: Folder '$folder_path' not found or is not a directory."
    exit 1
fi

echo "Searching for files named '$filename' in '$folder_path' and its subfolders..."

# Find files with the specific name recursively
find "$folder_path" -type f -name "$filename" | while read -r file; do
    echo "Processing file: $file"
    
    # Check if the file contains the line to be replaced
    if grep -q "skipEmptyColumns = True" "$file"; then
        echo "  Found 'skipEmptyColumns = True'. Replacing it with 'skipEmptyColumns = False'."
        # Use sed to replace the line in-place.
        # -i option edits files in place.
        # 's/original_string/new_string/g' is the substitution command.
        sed -i '' 's/skipEmptyColumns = True/skipEmptyColumns = False/g' "$file"
        echo "  Replacement complete for $file"
    else
        echo "  'skipEmptyColumns = True' not found in $file. Skipping replacement."
    fi
done

echo "Script finished."
