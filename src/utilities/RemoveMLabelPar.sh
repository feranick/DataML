#!/bin/bash

# Define the acceptable filenames
filename_numeric="config_numeric.txt"
filename_config="config.txt"

# Check if an argument was provided
if [ -n "$1" ]; then
    # Check if the argument is one of the acceptable filenames
    if [ "$1" == "$filename_numeric" ] || [ "$1" == "$filename_config" ]; then
        # Check if the file exists
        if [ ! -f "$1" ]; then
            echo "Error: File '$1' not found!"
            exit 1
        fi
        read -r line < "$1"
    else
        # If the argument is not a file, treat it as a string
        line="$1"
    fi
else
    # Display usage instructions if no argument is provided
    echo
    echo " Usage: $0 <config_numeric.txt | config.txt> or <sequence>"
    echo " Example: $0 config_numeric.txt"
    echo " Example: $0 m1,m2,m3"
    echo
    exit 1
fi

# Use sed to remove all 'm' characters globally
transformed_line=$(echo "$line" | sed 's/m//g')

echo
echo " Original string: $line"
echo
echo " Transformed string: $transformed_line"
echo
