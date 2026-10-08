#!/usr/bin/env bash
# Single entry point to fetch and prepare all 5 public datasets used by
# configs/experiment/*.yaml: downloads raw data (download_datasets.sh) then
# converts each into datasets/<name>/ matching the committed split CSVs. A
# dataset already converted (datasets/<name>/ non-empty) is skipped entirely
# (no re-download, no re-convert) unless --force is passed.
#
# Usage:
#   ./prepare_datasets.sh                  # download + convert all 5 (skips already-converted ones)
#   ./prepare_datasets.sh --only papila     # just one dataset
#   ./prepare_datasets.sh --clean-raw       # delete datasets/raw/<name>/ after each dataset converts successfully
#   ./prepare_datasets.sh --force           # re-download + re-convert even if already present
#   ./prepare_datasets.sh --only idrid --official   # use the official IEEE DataPort source for idrid
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# In the Docker image, datasets/*.csv ship baked in, but bind-mounting an
# empty host directory over datasets/ shadows them - restore any missing
# ones from the build-time stash kept outside that mount point (a no-op
# outside Docker, since this stash dir doesn't exist there).
STASH_DIR="/opt/rfm-dataset-csvs"
if [[ -d "$STASH_DIR" ]]; then
    mkdir -p datasets
    for f in "$STASH_DIR"/*.csv; do
        [[ -f "datasets/$(basename "$f")" ]] || cp "$f" datasets/
    done
fi

NAMES=(aptos messidor2 idrid papila gfid)

ONLY=""
CLEAN_RAW=0
FORCE=0
DOWNLOAD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)
            ONLY="${2:?Usage: ./prepare_datasets.sh --only <aptos|messidor2|idrid|papila|gfid>}"
            shift 2 ;;
        --clean-raw)
            CLEAN_RAW=1
            shift ;;
        --force)
            FORCE=1
            DOWNLOAD_ARGS+=("$1")
            shift ;;
        --official)
            DOWNLOAD_ARGS+=("$1")
            shift ;;
        -h|--help)
            sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1 ;;
    esac
done

if [[ -n "$ONLY" ]]; then
    DATASETS=("$ONLY")
else
    DATASETS=("${NAMES[@]}")
fi

already_present() {
    local dir="$1"
    [[ -d "$dir" ]] && [[ -n "$(find "$dir" -type f -print -quit 2>/dev/null)" ]]
}

# The converters run under a second, libjpeg-turbo Pillow in .prep_pillow/
# (see turbo_pillow.py); install it once if missing.
python turbo_pillow.py

total=${#DATASETS[@]}
i=0
FAILED=()
for name in "${DATASETS[@]}"; do
    i=$((i + 1))
    echo
    echo "=== [$i/$total] $name ==="

    out_dir="datasets/$name"
    [[ "$name" == "aptos" ]] && out_dir="datasets/aptos-2019"

    if already_present "$out_dir" && [[ "$FORCE" == 0 ]]; then
        echo "[$i/$total] $name: already converted, skipping (use --force to re-download + re-convert)"
        continue
    fi

    echo "[$i/$total] $name: downloading..."
    if ! ./download_datasets.sh --only "$name" "${DOWNLOAD_ARGS[@]}"; then
        echo "[$i/$total] $name: download failed, skipping (re-run with --only $name to retry)" >&2
        FAILED+=("$name")
        continue
    fi

    echo "[$i/$total] $name: converting..."
    if ! python "convert_${name}.py"; then
        echo "[$i/$total] $name: conversion failed, skipping (re-run with --only $name to retry)" >&2
        FAILED+=("$name")
        continue
    fi

    if [[ "$CLEAN_RAW" == 1 ]]; then
        raw_dir="datasets/raw/$name"
        [[ "$name" == "aptos" ]] && raw_dir="datasets/raw/aptos-2019"
        if [[ -d "$raw_dir" ]]; then
            echo "[$i/$total] $name: removing raw download ($raw_dir)"
            rm -rf "$raw_dir"
        fi
    fi

    echo "[$i/$total] $name: done"
done

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Done, but failed: ${FAILED[*]} (re-run with --only <name> to retry each)" >&2
    exit 1
fi
echo "All datasets prepared."
