#!/usr/bin/env bash
#
# compress_logs.sh - compress scheduler job logs named  <name>.<X><digits>  with
#                    xz, and clean up originals left behind by earlier runs.
#                    <X> defaults to o (stdout) and e (stderr).
#
# Version: 2026.08.09.2
#
# Usage: compress_logs.sh [options] DIR [DIR...]
#
#   --suffixes CHARS   Letters preceding the job number. Default: oe
#                      (e.g. --suffixes o  for stdout logs only)
#   --cleanup-only     Do not compress anything; only remove originals that
#                      already have a verified .xz archive next to them.
#   --dry-run          Report what would happen; change nothing.
#   --no-verify        Skip byte-for-byte comparison of archive vs original.
#                      (Faster, but only checks container integrity.)
#   --include-open     Also process files currently held open by a process.
#   --clear-flags      Attempt to clear immutable flags (chattr -i / chflags)
#                      when a delete is refused.
#   --min-age MINUTES  Only touch files not modified in the last N minutes.
#   -q, --quiet        Only print warnings, errors and the final summary.
#   -h, --help         This text.
#
# Exit status: 0 = everything handled, 1 = usage error, 2 = one or more files failed.

set -uo pipefail   # deliberately NOT -e: one bad file must not abort the run

VERSION="2026.08.09.2"

suffixes="oe"
cleanup_only=0
dry_run=0
verify=1
include_open=0
clear_flags=0
min_age=""
quiet=0

die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
warn() { printf 'warn:  %s\n' "$*" >&2; }
say()  { [ "$quiet" -eq 1 ] || printf '%s\n' "$*"; }

usage() { sed -n '3,30p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --suffixes)     shift; suffixes="${1:-}"
                        [[ "$suffixes" =~ ^[A-Za-z]+$ ]] || die "--suffixes needs one or more letters" ;;
        --cleanup-only) cleanup_only=1 ;;
        --dry-run)      dry_run=1 ;;
        --no-verify)    verify=0 ;;
        --include-open) include_open=1 ;;
        --clear-flags)  clear_flags=1 ;;
        --min-age)      shift; min_age="${1:-}"; [ -n "$min_age" ] || die "--min-age needs a value" ;;
        -q|--quiet)     quiet=1 ;;
        -h|--help)      usage; exit 0 ;;
        --version)      echo "$VERSION"; exit 0 ;;
        --)             shift; break ;;
        -*)             die "unknown option: $1" ;;
        *)              break ;;
    esac
    shift
done

[ $# -ge 1 ] || { usage >&2; exit 1; }
command -v xz >/dev/null 2>&1 || die "xz not found in PATH"

n_ok=0 n_skip=0 n_fail=0

# ---------------------------------------------------------------- helpers ---

# Is another process holding this file open? (best effort)
file_is_open() {
    if command -v lsof >/dev/null 2>&1; then
        lsof -- "$1" >/dev/null 2>&1 && return 0
    elif command -v fuser >/dev/null 2>&1; then
        fuser -- "$1" >/dev/null 2>&1 && return 0
    fi
    return 1
}

# Does $2 (.xz) faithfully contain $1?
archive_is_good() {
    local orig="$1" arch="$2"
    xz -t -- "$arch" >/dev/null 2>&1 || return 1
    [ "$verify" -eq 1 ] || return 0
    xz -dc -- "$arch" 2>/dev/null | cmp -s - "$orig"
}

# Explain, as far as we can, why a file could not be unlinked.
diagnose() {
    local f="$1" dir
    dir="$(dirname -- "$f")"

    printf '  diagnostics for %s\n' "$f" >&2
    ls -ld -- "$dir" >&2 2>/dev/null
    ls -l  -- "$f"   >&2 2>/dev/null

    # Deleting a file needs write+search on the PARENT DIRECTORY, not on the file.
    if [ ! -w "$dir" ] || [ ! -x "$dir" ]; then
        printf '  -> parent directory is not writable/searchable by %s\n' "$(id -un)" >&2
    fi
    if ! touch -- "$dir/.rmtest.$$" 2>/dev/null; then
        printf '  -> parent directory is read-only (read-only mount, quota, or ACL)\n' >&2
    else
        rm -f -- "$dir/.rmtest.$$"
    fi

    case "$(uname -s)" in
        Linux)
            command -v lsattr >/dev/null 2>&1 && lsattr -d -- "$f" "$dir" >&2 2>/dev/null
            ;;
        Darwin|*BSD*)
            ls -lO -- "$f" >&2 2>/dev/null
            ;;
    esac

    if file_is_open "$f"; then
        printf '  -> file is open by another process (on NFS this leaves .nfsXXXX stubs)\n' >&2
    fi
    case "$(basename -- "$f")" in
        .nfs*) printf '  -> this is an NFS silly-rename stub; it disappears when the holder exits\n' >&2 ;;
    esac
}

# Try hard to unlink, then actually confirm it is gone.
remove_original() {
    local f="$1"

    if [ "$dry_run" -eq 1 ]; then
        say "  would remove $f"
        return 0
    fi

    rm -f -- "$f" 2>/dev/null
    [ -e "$f" ] || return 0

    if [ "$clear_flags" -eq 1 ]; then
        case "$(uname -s)" in
            Linux)       command -v chattr  >/dev/null 2>&1 && chattr -i -a -- "$f" 2>/dev/null ;;
            Darwin|*BSD*) command -v chflags >/dev/null 2>&1 && chflags nouchg,noschg -- "$f" 2>/dev/null ;;
        esac
        rm -f -- "$f" 2>/dev/null
        [ -e "$f" ] || return 0
    fi

    warn "could not delete $f"
    diagnose "$f"
    return 1
}

# ------------------------------------------------------------ per-file work --

process_file() {
    local f="$1" arch="$1.xz" links

    if [ "$include_open" -eq 0 ] && file_is_open "$f"; then
        say "skip (open by another process): $f"
        n_skip=$((n_skip + 1))
        return 0
    fi

    links=$(stat -c '%h' -- "$f" 2>/dev/null || stat -f '%l' -- "$f" 2>/dev/null || echo 1)
    [ "${links:-1}" -gt 1 ] && warn "$f has $links hard links; removing this name will not free space"

    # Case 1: an archive already exists -- this is the "leftover original" case.
    if [ -e "$arch" ]; then
        if archive_is_good "$f" "$arch"; then
            say "leftover original, archive verified: $f"
            if remove_original "$f"; then n_ok=$((n_ok + 1)); else n_fail=$((n_fail + 1)); fi
            return 0
        fi
        if [ "$cleanup_only" -eq 1 ]; then
            warn "archive $arch does not match $f -- refusing to delete original"
            n_fail=$((n_fail + 1))
            return 0
        fi
        say "archive $arch is bad or stale; recompressing"
    elif [ "$cleanup_only" -eq 1 ]; then
        say "skip (no archive): $f"
        n_skip=$((n_skip + 1))
        return 0
    fi

    # Case 2: compress. -k keeps the original so WE decide when to unlink it,
    # and only after the archive has been verified. xz's own deletion is what
    # made failures silent and unrecoverable before.
    if [ "$dry_run" -eq 1 ]; then
        say "would compress $f"
        n_ok=$((n_ok + 1))
        return 0
    fi

    say "compressing $f"
    if ! xz -z -k -f -- "$f"; then
        warn "xz failed on $f (disk full? quota? unreadable?) -- original kept"
        rm -f -- "$arch"          # drop the truncated archive, do not leave a trap
        n_fail=$((n_fail + 1))
        return 0
    fi

    if ! archive_is_good "$f" "$arch"; then
        warn "archive verification FAILED for $arch -- original kept"
        n_fail=$((n_fail + 1))
        return 0
    fi

    if remove_original "$f"; then n_ok=$((n_ok + 1)); else n_fail=$((n_fail + 1)); fi
}

# ------------------------------------------------------------------- main ---

for target in "$@"; do
    [ -d "$target" ] || { warn "not a directory: $target"; n_fail=$((n_fail + 1)); continue; }

    find_args=("$target" -type f)
    [ -n "$min_age" ] && find_args+=(-mmin "+$min_age")

    # Process substitution, not a pipe: a pipe puts the loop in a subshell, so
    # the counters vanish and an early exit is invisible to the caller.
    while IFS= read -r -d '' f; do
        # Portable match instead of BSD -E / GNU -regextype. One-or-more digits:
        # the old [0-9]{4} silently ignored job.o999 and job.o123456. And any of
        # $suffixes, not just o: PBS/SGE also write the .e<jobid> stderr file.
        [[ "${f##*/}" =~ ^.+\.[$suffixes][0-9]+$ ]] || continue
        process_file "$f"
    done < <(find "${find_args[@]}" -print0 2>/dev/null)
done

printf 'Summary: %d handled, %d skipped, %d failed\n' "$n_ok" "$n_skip" "$n_fail"
[ "$n_fail" -eq 0 ] || exit 2
