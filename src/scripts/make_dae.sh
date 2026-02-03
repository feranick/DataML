#!/bin/bash

###############################################################
# Launch slurm jobs for DAE training recursively across folders
###############################################################

# 1. Check if an argument was provided
if [ -z "$1" ]; then
    echo "Error: No argument provided."
    echo "Usage: ./make_dae.sh <initial learning file>"
    exit 1
fi

INPUT_ARG="$1"
BASE_DIR=$(pwd)

echo "Starting submission loop with argument: $INPUT_ARG"
echo "------------------------------------------------"

# 2. Loop through all directories in the current location
for dir in */; do
    # Remove the trailing slash from directory name for cleaner output
    dirname=${dir%/}
    
    # 3. Check if the 'dae' subdirectory exists
    if [ -d "$dirname/dae" ]; then
        echo "Entering $dirname/dae..."
        
        # Enter the directory
        cd "$dirname/dae" || { echo "Failed to enter $dirname/dae"; continue; }
        
        # 4. Run the sbatch command
        # Using "$INPUT_ARG" in quotes handles arguments with spaces correctly
        sbatch ../../sub_DAE.sh "$INPUT_ARG"
        
        # Return to the main folder
        cd "$BASE_DIR" || exit
    else
        # Optional: verify if we should warn about missing dae folders
        echo "Skipping $dirname: No 'dae' folder found."
    fi
done

echo "------------------------------------------------"
echo "All submissions complete."
