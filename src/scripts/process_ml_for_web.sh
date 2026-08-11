#!/bin/sh

# Check if the required argument is provided
if [ -z "$1" ]; then
    echo "Error: Missing argument."
    echo "Usage: $0 <full path DataML_Maker.ini> <param_label_arg>"
    exit 1
fi

abspath() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;          # already absolute
        *)  printf '%s\n' "$PWD/$1" ;;     # relative to invocation dir
    esac
}

MAKERFILE=$(abspath "$1")
PARAM_ARG=$(abspath "$2")

for f in "$MAKERFILE" "$PARAM_ARG"; do
    [ -f "$f" ] || { echo "Error: file not found: $f"; exit 1; }
done

# Iterate over all items in the current directory
for dir in *; do
    # Only process directories
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"
        
        # Move into the subfolder
        cd "$dir" || continue
        
        cp "$MAKERFILE" .
        
        # Collect immediate subdirectories
        subdirs=""
        count=0

        for dir in ./*/; do
            # Handle cases where no directories match the pattern
            [ -d "$dir" ] || continue
  
            subdirs="$subdirs $dir"
            count=$((count + 1))
        done

        # Validate directory count
        if [ "$count" -ne 2 ]; then
            echo "Error: Expected exactly 2 inner folders, but found $count." >&2
            exit 1
        fi

        # Process each subdirectory
        for dir in $subdirs; do
            # Trim trailing slash for output clarity
            clean_dir="${dir%/}"
            echo "Moving contents from '$clean_dir'..."
          
            # Move all items (including hidden dotfiles) into the current directory
            find "$dir" -mindepth 1 -maxdepth 1 -exec mv {} . \;

            # Remove empty folder
            rmdir "$dir"
            echo "Removed '$clean_dir'."
        done

        echo "Done! Contents moved and inner folders removed."

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
        random_pdf=$(find . -maxdepth 1 -name "*_Random_plots.pdf" -print -quit)
        random_nospur_pdf=$(find . -maxdepth 1 -name "*_Random_noSpur_plots.pdf" -print -quit)

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
        
        if [ -n "$random_pdf" ]; then
            echo "  Found *_Random_plots.pdf ($random_file). Copying to train.pdf..."
            cp "$random_pdf" "train.pdf"
        elif [ -n "$random_nospur_pdf" ]; then
            echo "  Found *_Random_noSpur_plots.pdf ($random_nospur_file). Copying to train.pdf..."
            cp "$random_nospur_pdf" "train.pdf"
        else
            echo "  No matching target files found for train.pdf logic."
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
