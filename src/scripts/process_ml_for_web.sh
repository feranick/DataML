#!/bin/bash

# Check if the required argument is provided
if [ -z "$1" ]; then
    echo "Error: Missing argument."
    echo "Usage: $0 <param_label_arg>"
    exit 1
fi

PARAM_ARG=../"$1"

# Iterate over all items in the current directory
for dir in *; do
    # Only process directories
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"
        
        # Move into the subfolder
        cd "$dir" || continue

        # 1. Run the custom command with the provided argument
        if command -v ConvertParamLabels &> /dev/null; then
            ConvertParamLabels "$PARAM_ARG" "config.txt"
        else
            echo "  Warning: ConvertParamLabels command not found. Skipping execution."
        fi

        # 2. File duplication logic for train.txt
        kde_aug_file=$(find . -maxdepth 1 -name "*_kde_aug.txt" -print -quit)
        train_suffix_file=$(find . -maxdepth 1 -name "*_train.txt" -print -quit)
        random_file=$(find . -maxdepth 1 -name "*_Random.txt" -print -quit)
        random_nospur_file=$(find . -maxdepth 1 -name "*_Random_noSpur.txt" -print -quit)

        if [ -n "$kde_aug_file" ]; then
            echo "  Found *_kde_aug.txt ($kde_aug_file). Copying to train.txt..."
            cp "$kde_aug_file" "train.txt"
        elif [ -n "$train_suffix_file" ]; then
            echo "  *_kde_aug.txt not found. Found *_train.txt ($train_suffix_file). Copying to train.txt..."
            cp "$train_suffix_file" "train.txt"
        elif [ -n "$random_file" ]; then
            echo "  Found *_Random.txt ($random_file). Copying to train.txt..."
            cp "$random_file" "train.txt"
        elif [ -n "$random_nospur_file" ]; then
            echo "  Found *_Random_noSpur.txt ($random_nospur_file). Copying to train.txt..."
            cp "$random_nospur_file" "train.txt"
        else
            echo "  No matching target files found for train.txt logic."
        fi

        # 3. Create the index.html redirect file
        echo "  Creating index.html redirect..."
        cat << 'EOF' > index.html
<meta http-equiv="refresh" content="0; URL=https://mit.edu" />
EOF

        # Step back out to the parent directory
        cd ..
    fi
done

echo "Done processing all directories."
