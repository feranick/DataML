#!/bin/bash

# This script recursively finds all files named "CorrAnalysis.ini"
# and replaces the line "columnSpecColors = 89" with "columnSpecColors = 97".
# It creates a backup of each modified file with a .bak extension.

# Set the directory to start the search from.
# "." means the current directory.
SEARCH_DIR="."

# The filename to search for.
FILENAME="CorrAnalysis.ini"

# The original line to find.
OLD_LINE="columnSpecColors = 89"

# The new line to replace it with.
NEW_LINE="columnSpecColors = 97"

# Use 'find' to locate the files and pipe the results to 'sed' for in-place editing.
# -print0 and xargs -0 are used to handle filenames that may contain spaces or special characters.
find "$SEARCH_DIR" -type f -name "$FILENAME" -print0 | while IFS= read -r -d $'\0' file; do
    echo "Processing file: $file"
    # Use sed to perform the replacement.
    # The '-i.bak' option creates a backup of the original file.
    sed -i.bak "s/$OLD_LINE/$NEW_LINE/g" "$file"
done

echo "Script finished. All matching files have been updated."
echo "Backup files have been created with the .bak extension."
