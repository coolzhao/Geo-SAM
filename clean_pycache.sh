#!/usr/bin/env bash
# clean_pycache.sh — Recursively remove all __pycache__ directories in the project.
#
# Usage:
#   ./clean_pycache.sh          # remove and report each directory
#   ./clean_pycache.sh -q       # quiet — only print summary
#   ./clean_pycache.sh -n       # dry-run — list directories without deleting
#   ./clean_pycache.sh -h       # show help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUIET=false
DRY_RUN=false

usage() {
    echo "Usage: $(basename "$0") [-q] [-n] [-h]"
    echo ""
    echo "  (no flag)  Remove all __pycache__ dirs, reporting each one"
    echo "  -q         Quiet — only print summary statistics"
    echo "  -n         Dry-run — list dirs without deleting"
    echo "  -h         Show this help message"
    exit 0
}

while getopts "qnh" opt; do
    case "$opt" in
        q) QUIET=true ;;
        n) DRY_RUN=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

echo "Scanning for __pycache__ directories..."
echo "  Project root: $SCRIPT_DIR"
echo ""

# Collect all paths into a temp file so the find stream isn't truncated by pipes.
TMPFILE="$(mktemp)"
trap 'rm -f "$TMPFILE"' EXIT

find "$SCRIPT_DIR" -type d -name "__pycache__" -print0 > "$TMPFILE"

# Count entries (null-delimited).
COUNT=$(tr '\0' '\n' < "$TMPFILE" | grep -c .)

if [ "$COUNT" -eq 0 ]; then
    echo "No __pycache__ directories found — the project is already clean."
    exit 0
fi

if $DRY_RUN; then
    echo "Would delete $COUNT directories (dry-run):"
    echo "----------------------------------------"
    tr '\0' '\n' < "$TMPFILE" | while IFS= read -r dir; do
        echo "  $dir"
    done
    echo "----------------------------------------"
    echo "Run without -n to actually delete: $(basename "$0")"
    exit 0
fi

echo "Removing $COUNT __pycache__ directories..."

tr '\0' '\n' < "$TMPFILE" | while IFS= read -r dir; do
    rm -rf "$dir"
    $QUIET || echo "  Removed: ${dir#$SCRIPT_DIR/}"
done

# Re-scan to confirm.
REMAINING=$(find "$SCRIPT_DIR" -type d -name "__pycache__" 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "========================================"
echo "  Found:    $COUNT directories"
echo "  Remaining: $REMAINING directories"
echo "========================================"

if [ "$REMAINING" -eq 0 ]; then
    echo "Done — all __pycache__ directories removed."
else
    echo "Warning: $REMAINING directories could not be removed (permission issue?)."
fi
