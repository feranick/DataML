#!/usr/bin/env bash
#
# flatten_dae_ml.sh
#
# For every "a" folder inside a container "b", move the contents of the
# "dae" and "ML" subfolders up into "a", then remove the now-empty
# "dae" and "ML" folders.
#
# Usage:
#   ./flatten_dae_ml.sh [-n] [-p PREFIX] <b>
#
#   <b>          Path to the container folder that holds the "a" folders.
#   -n           Dry run: show what would happen without changing anything.
#   -p PREFIX    Prefix to prepend to each "a" folder's name.
#
set -euo pipefail

DRY_RUN=0
PREFIX=""

usage() {
    echo "Usage: $0 [-n] [-p PREFIX] <container_folder>" >&2
    echo "  -n           Dry run (show actions without performing them)" >&2
    echo "  -p PREFIX    Prefix to prepend to each 'a' folder's name" >&2
    exit 1
}

# --- parse arguments -------------------------------------------------------
while getopts ":np:" opt; do
    case "$opt" in
        n) DRY_RUN=1 ;;
        p) PREFIX="$OPTARG" ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

[ $# -eq 1 ] || usage
B="$1"

if [ ! -d "$B" ]; then
    echo "Error: '$B' is not a directory." >&2
    exit 1
fi

# --- helper: move all contents (incl. hidden files) of $1 into $2 ----------
move_contents() {
    local src="$1" dst="$2"
    # nullglob + dotglob so the loop also catches hidden entries and
    # simply does nothing when the folder is empty.
    shopt -s nullglob dotglob
    local moved=0
    for item in "$src"/*; do
        local base
        base="$(basename "$item")"
        # never carry macOS .DS_Store junk along
        if [ "$base" = ".DS_Store" ]; then
            if [ "$DRY_RUN" -eq 1 ]; then
                echo "  would delete: $item"
            else
                rm -f -- "$item"
                echo "  deleted: $item"
            fi
            continue
        fi
        if [ -e "$dst/$base" ]; then
            echo "  ! SKIP (already exists in target): $item" >&2
            continue
        fi
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "  would move: $item -> $dst/"
        else
            mv -- "$item" "$dst/"
            echo "  moved: $item -> $dst/"
        fi
        moved=1
    done
    shopt -u nullglob dotglob
    return 0
}

# --- helper: remove a folder if empty --------------------------------------
remove_if_empty() {
    local dir="$1"
    if [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "  would remove empty: $dir"
        else
            rmdir "$dir"
            echo "  removed empty: $dir"
        fi
    else
        echo "  ! NOT removed (not empty): $dir" >&2
    fi
}

# --- remove all .DS_Store files under the container ------------------------
if [ "$DRY_RUN" -eq 1 ]; then
    while IFS= read -r -d '' f; do
        echo "would delete: $f"
    done < <(find "$B" -type f -name .DS_Store -print0)
else
    find "$B" -type f -name .DS_Store -print -delete
fi

# --- main loop: iterate over each "a" folder in "b" -------------------------
for a in "$B"/*/; do
    a="${a%/}"                       # strip trailing slash
    [ -d "$a" ] || continue

    echo "Processing: $a"

    for sub in dae ML; do
        subdir="$a/$sub"
        if [ -d "$subdir" ]; then
            echo " - $sub"
            move_contents "$subdir" "$a"
            remove_if_empty "$subdir"
        fi
    done

    # --- rename the "a" folder with the prefix, if requested ---------------
    if [ -n "$PREFIX" ]; then
        parent="$(dirname "$a")"
        base="$(basename "$a")"
        newpath="$parent/${PREFIX}${base}"
        if [ "$newpath" = "$a" ]; then
            :   # already prefixed / nothing to do
        elif [ -e "$newpath" ]; then
            echo "  ! SKIP rename (target exists): $newpath" >&2
        elif [ "$DRY_RUN" -eq 1 ]; then
            echo "  would rename: $a -> $newpath"
        else
            mv -- "$a" "$newpath"
            echo "  renamed: $a -> $newpath"
        fi
    fi
done

echo "Done."
