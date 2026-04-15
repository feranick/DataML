#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <value> <folder_path> <ini file>"
    exit 1
fi

value="$1"
folder_path="$2"
#filename="DataML_VAE.ini"
inifile="$3"

metric="epochs"

if [[ $inifile == "DataML_DAE.ini" ]]; then
    filename="DataML_DAE.ini"
elif [[ $inifile="DataML_VAE.ini" ]]; then
    filename="DataML_VAE.ini"
else
    echo "Error: inifile has to be either DataML_DAE.ini or DataML_VAE.ini"
fi

# Check if the provided folder path is a valid directory
if [ ! -d "$folder_path" ]; then
    echo "Error: Folder '$folder_path' not found or is not a directory."
    exit 1
fi

echo "Searching for files named '$filename' in '$folder_path' and its subfolders..."

# Find files with the specific name recursively
find "$folder_path" -type f -name "$filename" | while read -r file; do
    echo "Processing file: $file"
    
    # Check if the file contains the variable
    if grep -q "$metric =" "$file"; then
        echo "  Found '$metric'. Replacing its value with '$value'"
        # Use sed to replace the rest of the line with the new value
        sed -i '' "s/$metric = .*/$metric = $value/g" "$file"
        echo "  Replacement complete for $file"
    else
        echo "  '$metric =' not found in $file. Skipping replacement."
    fi
done

echo "Script finished."
