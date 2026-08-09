#!/usr/bin/env bash

# Safety flag: Exit on errors
set -euo pipefail

# Ensure a target directory argument was provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 /path/to/target/directory"
    exit 1
fi

TARGET_DIR="$1"

# Check if the path exists and is a directory
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

# Function to compress a single file with terminal output
compress_file() {
    local file="$1"
    echo "Compressing: $file"
    xz -z -f "$file"
}

export -f compress_file

# Detect Operating System and stream matching files to the compression handler
OS="$(uname -s)"

case "$OS" in
    Darwin)
        # macOS (BSD find)
        find -E "$TARGET_DIR" -type f -regex '.*/[^/]+\.o[0-9]{4}$' -print0 | while IFS= read -r -d '' file; do
            compress_file "$file"
        done
        ;;
    Linux|*)
        # Linux / GNU find
        find "$TARGET_DIR" -type f -regextype posix-extended -regex '.*/[^/]+\.o[0-9]{4}$' -print0 | while IFS= read -r -d '' file; do
            compress_file "$file"
        done
        ;;
esac

echo "Done!"
