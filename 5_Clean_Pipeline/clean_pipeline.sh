#!/usr/bin/env bash

# =============================================================================
# 5_Clean_Pipeline – Clean pipeline artefacts
# =============================================================================
# This script removes generated artefacts from previous stages while preserving
# user‑provided files and documentation. It is safe to run on a copied archive of
# the repository; the original source data remains untouched.
#
# What it does:
#   • Deletes everything inside any `output` or `references` directory of the
#     four main stages (1‑4), except files named CONTEXT.md, README.md or
#     .gitignore.
#   • In `4_Port_MOOSE/41_Export_Model/references` it removes only `*.pickle`
#     files; *.i files and the `base` sub‑directory are kept.
#   • Does **not** touch the `utils`, `examples`, or top‑level documentation
#     files.
#
# Usage:
#   ./clean_pipeline.sh            # run with default behaviour
#   ./clean_pipeline.sh --dry-run  # show what would be removed without deleting
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Determine repository root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

DRY_RUN=false
if [[ "${1-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

log() {
    echo "[clean_pipeline] $*"
}

remove_path() {
    local path="$1"
    if $DRY_RUN; then
        log "Would remove: $path"
    else
        rm -rf "$path"
        log "Removed: $path"
    fi
}

# ---------------------------------------------------------------------------
# Clean a generic directory (output or references) by removing all files except
# documentation artefacts and deleting empty sub‑directories (excluding any
# "base" folder that might be needed elsewhere).
# ---------------------------------------------------------------------------
clean_generic_dir() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        # Delete files that are not documentation artifacts
        # Exclude files inside any "base" subdirectory (e.g., MOOSE base headers) and the usual docs/files.
        find "$dir" -type f \! \( -path "*/base/*" -o -name "CONTEXT.md" -o -name "README.md" -o -name "*.pdf" -o -name ".gitignore" -o -name ".gitkeep" -o -name "*.i" -o -name "*.py" -o -name "*.sh" \) -print0 |
            while IFS= read -r -d '' file; do
                remove_path "$file"
            done
        # Remove empty directories, but preserve any "base" folder that may be needed elsewhere
        find "$dir" -depth -type d ! -path "*/base" -empty -print0 |
            while IFS= read -r -d '' d; do
                remove_path "$d"
            done
    else
        log "Directory not found (skipped): $dir"
    fi
}

log "Starting pipeline clean-up"

# Iterate over the main stage directories (excluding utils/examples)
# Each stage may contain nested sub‑folders that also have their own `output`
# or `references` directories (e.g., 3_Surrogate_Modeling/31_Surrogate_Training/output).
# We locate all such directories recursively and clean them.
for stage in 1_Data_Generation 2_Data_Processing 3_Surrogate_Modeling 4_Port_MOOSE; do
    find "${REPO_ROOT}/${stage}" -type d \( -name output -o -name references \) -print0 |
        while IFS= read -r -d '' dir; do
            clean_generic_dir "$dir"
        done
done

# Special handling for the Export_Model references folder
export_ref_dir="${REPO_ROOT}/4_Port_MOOSE/41_Export_Model/references"
if [[ -d "$export_ref_dir" ]]; then
    log "Cleaning *.pickle files in $export_ref_dir"
    find "$export_ref_dir" -type f -name "*.pickle" -print0 |
        while IFS= read -r -d '' p; do
            remove_path "$p"
        done
    # Do NOT delete .i files or the "base" subdirectory – they are left untouched.
fi

log "Pipeline clean-up complete"
