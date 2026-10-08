#!/usr/bin/env bash
# Re-test already-trained heads against their test splits, without
# retraining, then (re)compile the results.pdf report. Wraps test_only.py +
# compile_results.py over the run directories recorded by train_test.sh in
# its manifest.
#
# Everything this script writes - including its own console log - lands
# under the manifest's directory (outputs/ by default; a --manifest in
# another directory moves the whole run there), so a run is self-contained
# to that one directory.
#
# Usage:
#   ./test.sh                        # retest every run in outputs/results_manifest.txt -> outputs/results.pdf
#   ./test.sh --dataset messidor     # retest all 6 variants for one dataset (rfm, dinov3b-9l,
#                                     #   dinov3, retfound_dinov2, retfound_mae,
#                                     #   retfound_green) from outputs/results_messidor.txt
#                                     #   -> outputs/results_messidor.pdf
#   ./test.sh --manifest my_runs.txt # retest a specific manifest -> my_runs.pdf
#
# Valid --dataset values: messidor, aptos, idrid, papila, glaucoma_fundus
#
# All stdout/stderr (from this script and test_only.py/compile_results.py) is
# both printed to the terminal and saved under <manifest dir>/logs/
# (outputs/logs/ by default).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Must match train_test.sh's DATASET_VARIANTS.
DATASET_VARIANTS=(rfm dinov3b-9l dinov3 retfound_dinov2 retfound_mae retfound_green)

DATASET=""
MANIFEST_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="${2:?Usage: ./test.sh --dataset <messidor|aptos|idrid|papila|glaucoma_fundus>}"
            shift 2 ;;
        --manifest)
            MANIFEST_OVERRIDE="${2:?Usage: ./test.sh --manifest <path.txt>}"
            shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1 ;;
    esac
done

MANIFEST="outputs/results_manifest.txt"
REPORT="outputs/results.pdf"
LOG_TAG="all"

if [[ -n "$DATASET" ]]; then
    for variant in "${DATASET_VARIANTS[@]}"; do
        cfg="${DATASET}_${variant}"
        if [[ ! -f "configs/experiment/${cfg}.yaml" ]]; then
            echo "Unknown dataset '$DATASET' (no config configs/experiment/${cfg}.yaml)" >&2
            exit 1
        fi
    done
    MANIFEST="outputs/results_${DATASET}.txt"
    REPORT="outputs/results_${DATASET}.pdf"
    LOG_TAG="$DATASET"
fi

if [[ -n "$MANIFEST_OVERRIDE" ]]; then
    MANIFEST="$MANIFEST_OVERRIDE"
    REPORT="${MANIFEST%.txt}.pdf"
    LOG_TAG="$(basename "${MANIFEST%.txt}")"
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "No manifest at $MANIFEST — run ./train_test.sh first." >&2
    exit 1
fi

LOG_DIR="$(dirname "$MANIFEST")/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/test_${LOG_TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
echo "Re-testing runs from manifest: $MANIFEST"

python test_only.py --manifest "$MANIFEST"
python compile_results.py --manifest "$MANIFEST" --out "$REPORT"
echo "Done. See $REPORT"
