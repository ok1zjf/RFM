#!/usr/bin/env bash
# One-shot setup: download model weights (download_weights.sh), then download
# + convert the public datasets (prepare_datasets.sh), so a fresh checkout
# only needs `./init.sh` before `./train_test.sh` / `./test.sh`.
#
# Usage:
#   ./init.sh                        # weights + all 5 datasets
#   ./init.sh --only papila          # weights + just one dataset
#   ./init.sh --force                # re-download + re-convert everything even if already present
#   ./init.sh --clean-raw            # delete each dataset's raw download once it converts
#   ./init.sh --only idrid --official   # use the official IEEE DataPort source for idrid
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

WEIGHTS_ARGS=()
DATASET_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            WEIGHTS_ARGS+=("--force")
            DATASET_ARGS+=("--force")
            shift ;;
        --only)
            DATASET_ARGS+=("--only" "${2:?Usage: ./init.sh --only <aptos|messidor2|idrid|papila|gfid>}")
            shift 2 ;;
        --clean-raw|--official)
            DATASET_ARGS+=("$1")
            shift ;;
        -h|--help)
            sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1 ;;
    esac
done

# Run both steps to completion even if one fails, then summarize both at the
# end - a single flaky download (e.g. an external mirror timing out) shouldn't
# hide whether everything else succeeded.
extract_failed() {
    # Pulls the space-separated names off a "Done, but failed: a b c" line,
    # if the wrapped command's own output printed one.
    grep -m1 "^Done, but failed:" "$1" | sed 's/^Done, but failed: //'
}

WEIGHTS_LOG="$(mktemp)"
DATASETS_LOG="$(mktemp)"
trap 'rm -f "$WEIGHTS_LOG" "$DATASETS_LOG"' EXIT

echo "=== [1/2] weights ==="
if ./download_weights.sh "${WEIGHTS_ARGS[@]}" 2>&1 | tee "$WEIGHTS_LOG"; then
    WEIGHTS_STATUS="OK"
else
    WEIGHTS_STATUS="FAILED: $(extract_failed "$WEIGHTS_LOG")"
fi

echo
echo "=== [2/2] datasets ==="
if ./prepare_datasets.sh "${DATASET_ARGS[@]}" 2>&1 | tee "$DATASETS_LOG"; then
    DATASETS_STATUS="OK"
else
    DATASETS_STATUS="FAILED: $(extract_failed "$DATASETS_LOG")"
fi

echo
echo "=== Summary ==="
echo "Weights:  $WEIGHTS_STATUS"
echo "Datasets: $DATASETS_STATUS"

if [[ "$WEIGHTS_STATUS" == OK && "$DATASETS_STATUS" == OK ]]; then
    echo
    echo "Init complete. Run ./train_test.sh to train and test all experiments."
else
    echo
    echo "Init finished with failures (see above) - re-run this script, or" >&2
    echo "./download_weights.sh / ./prepare_datasets.sh --only <name> to retry" >&2
    echo "just the failed ones." >&2
    exit 1
fi
