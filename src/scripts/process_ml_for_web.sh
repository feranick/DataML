#!/bin/sh
#
# process_ml_for_web.sh
#
# For every subdirectory of the current directory:
#   1. flatten its two inner folders into it,
#   2. copy in the DataML maker file,
#   3. run ConvertParamLabels to produce config.txt,
#   4. pick the best available source file for train.txt / train.pdf,
#   5. drop in an index.html redirect.
#
# Usage: process_ml_for_web.sh <DataML_Maker.ini> <param_label_file> [redirect_url]
#
# Both file arguments accept relative or absolute paths; they are resolved to
# absolute paths before the script descends into any subdirectory.
#
# Portability: POSIX sh only (works under dash on Linux and bash-as-sh on
# macOS). No arrays, no mapfile, no [[ ]], no &>, no pipefail. The single
# non-POSIX construct used is `local`, which dash, ash, bash and zsh all
# support.

set -u

PROGNAME=${0##*/}
DEFAULT_REDIRECT_URL="https://mit.edu"
EXPECTED_SUBDIRS=2
readonly PROGNAME DEFAULT_REDIRECT_URL EXPECTED_SUBDIRS

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

usage() {
    cat <<USAGE
Usage: $PROGNAME <DataML_Maker.ini> <param_label_file> [redirect_url]

Arguments:
  DataML_Maker.ini   maker file copied into each processed directory
  param_label_file   parameter/label file passed to ConvertParamLabels
  redirect_url       URL for the generated index.html
                     (default: $DEFAULT_REDIRECT_URL)

Both file paths may be relative or absolute.
USAGE
}

msg()  { printf '%s\n' "$*"; }
warn() { printf '%s: warning: %s\n' "$PROGNAME" "$*" >&2; }
die()  { printf '%s: error: %s\n' "$PROGNAME" "$*" >&2; exit 1; }

# Turn a possibly-relative path into an absolute one, relative to the
# directory the script was invoked from.
abspath() {
    case $1 in
        /*) printf '%s\n' "$1" ;;
        *)  printf '%s\n' "${PWD%/}/$1" ;;
    esac
}

html_escape() {
    printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' \
                           -e 's/>/\&gt;/g' -e 's/"/\&quot;/g'
}

# copy_first <target> <candidate>...
# Copies the first candidate that is a regular file to <target>.
# Unmatched globs arrive as literal patterns and are skipped by the -f test.
# Returns 1 if nothing matched.
copy_first() {
    local target src
    target=$1
    shift
    for src in "$@"; do
        [ -f "$src" ] || continue
        msg "  Copying '$src' -> $target"
        cp -- "$src" "$target" || return 1
        return 0
    done
    return 1
}

# --------------------------------------------------------------------------- #
# Flattening
# --------------------------------------------------------------------------- #

# OS metadata that turns up everywhere (macOS .DS_Store in particular).
# These are never moved up; they are deleted from the inner folders so the
# folders can be removed, and they are ignored by the collision check.
is_junk() {
    case $1 in
        .DS_Store|._*|.AppleDouble|.AppleDB|.AppleDesktop|.localized \
        |.Spotlight-V100|.TemporaryItems|.Trashes|.fseventsd|.VolumeIcon.icns \
        |Thumbs.db|ehthumbs.db|desktop.ini)
            return 0
            ;;
    esac
    return 1
}

# scan_inner <check|move> <dir>...
# check: verify nothing would be overwritten, in this directory or between the
#        inner folders.  move: actually move the contents up one level.
# `seen` is a /-delimited set of basenames; '/' is the one byte a filename
# cannot contain, so it is a safe delimiter.
scan_inner() {
    local mode sub item base
    mode=$1
    shift
    for sub in "$@"; do
        for item in "$sub"/* "$sub"/.*; do
            base=${item##*/}
            case $base in
                .|..) continue ;;
            esac
            # Skips unmatched globs, which remain literal in POSIX sh.
            [ -e "$item" ] || [ -L "$item" ] || continue

            if is_junk "$base"; then
                if [ "$mode" = move ]; then
                    rm -rf -- "$item"
                fi
                continue
            fi

            if [ "$mode" = check ]; then
                if [ -e "./$base" ] || [ -L "./$base" ]; then
                    warn "'$base' already exists in '$PWD'; refusing to overwrite"
                    return 1
                fi
                case $seen in
                    *"/$base/"*)
                        warn "'$base' is present in both inner folders; refusing to overwrite"
                        return 1
                        ;;
                esac
                seen="$seen$base/"
            else
                mv -- "$item" ./ || return 1
            fi
        done
    done
    return 0
}

flatten_subdirs() {
    local count sub1 sub2 d

    count=0
    sub1=
    sub2=
    for d in */ .*/; do
        d=${d%/}
        case $d in
            .|..) continue ;;
        esac
        [ -d "$d" ] || continue
        is_junk "$d" && continue
        count=$((count + 1))
        case $count in
            1) sub1=$d ;;
            2) sub2=$d ;;
        esac
    done

    if [ "$count" -eq 0 ]; then
        msg "  No inner folders found; assuming already flattened."
        return 0
    fi

    if [ "$count" -ne "$EXPECTED_SUBDIRS" ]; then
        warn "expected $EXPECTED_SUBDIRS inner folders, found $count"
        return 1
    fi

    # Two passes: nothing moves until the whole move is known to be safe.
    seen="/"
    scan_inner check "$sub1" "$sub2" || return 1
    scan_inner move  "$sub1" "$sub2" || return 1

    for d in "$sub1" "$sub2"; do
        rmdir -- "$d" || { warn "could not remove '$d'"; return 1; }
        msg "  Flattened and removed '$d'."
    done

    return 0
}

# --------------------------------------------------------------------------- #
# Process one directory (always called inside a subshell, so the cd is scoped)
# --------------------------------------------------------------------------- #

process_dir() {
    cd -- "$1" || return 1

    flatten_subdirs || return 1

    cp -- "$MAKERFILE" . || return 1

    if [ "$HAVE_CONVERT" -eq 1 ]; then
        ConvertParamLabels "$PARAM_ARG" "config.txt" \
            || { warn "ConvertParamLabels failed in '$1'"; return 1; }
    fi

    copy_first train.txt *_kde_aug.txt \
        || copy_first train.txt *_train.txt \
        || copy_first train.txt *_Random.txt \
        || copy_first train.txt *_Random_noSpur.txt \
        || msg "  No matching source file found for train.txt."

    copy_first train.pdf *_Random_plots.pdf \
        || copy_first train.pdf *_Random_noSpur_plots.pdf \
        || msg "  No matching source file found for train.pdf."

    msg "  Creating index.html redirect..."
    printf '<meta http-equiv="refresh" content="0; URL=%s" />\n' \
        "$(html_escape "$REDIRECT_URL")" > index.html || return 1

    return 0
}

# --------------------------------------------------------------------------- #
# Argument handling
# --------------------------------------------------------------------------- #

case ${1-} in
    -h|--help) usage; exit 0 ;;
esac

if [ "$#" -lt 2 ]; then
    usage >&2
    die "missing argument(s)"
fi
if [ "$#" -gt 3 ]; then
    usage >&2
    die "too many arguments"
fi

MAKERFILE=$(abspath "$1")
PARAM_ARG=$(abspath "$2")
REDIRECT_URL=${3:-$DEFAULT_REDIRECT_URL}
readonly MAKERFILE PARAM_ARG REDIRECT_URL

for f in "$MAKERFILE" "$PARAM_ARG"; do
    [ -e "$f" ] || die "file not found: $f"
    [ -f "$f" ] || die "not a regular file: $f"
    [ -r "$f" ] || die "file not readable: $f"
done

if command -v ConvertParamLabels >/dev/null 2>&1; then
    HAVE_CONVERT=1
else
    HAVE_CONVERT=0
    warn "ConvertParamLabels not found in PATH; label conversion will be skipped"
fi
readonly HAVE_CONVERT

# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

ok=0
failed=0

for dir in */; do
    dir=${dir%/}
    [ -d "$dir" ] || continue
    msg "Processing directory: $dir"
    if ( process_dir "$dir" ); then
        ok=$((ok + 1))
    else
        failed=$((failed + 1))
        warn "skipped '$dir' (left unchanged or partially processed)"
    fi
done

if [ "$((ok + failed))" -eq 0 ]; then
    msg "No directories found in $PWD; nothing to do."
    exit 0
fi

msg "Done. $ok directory/directories processed, $failed skipped."
[ "$failed" -eq 0 ] || exit 1
