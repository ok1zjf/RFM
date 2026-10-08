#!/usr/bin/env bash
# Train + auto-test all experiments (train.py runs the test/report step at
# the end of each training run), then compile a results.pdf. Run ./init.sh
# first to fetch weights + datasets. To re-test already-trained heads without
# retraining, use ./test.sh instead.
#
# Everything this script writes - including its own console log - lands
# under the manifest's directory (outputs/ by default; a --manifest in
# another directory, or hydra.run.dir=<other>/... overrides, moves the whole
# run there), so a run is self-contained to that one directory.
#
# Usage:
#   ./train_test.sh                                # all 30 experiments (5 datasets x 6 variants) -> outputs/results.pdf
#   ./train_test.sh messidor_rfm                   # just one (or a few) configs by name -> outputs/results.pdf
#   ./train_test.sh --dataset messidor             # all 6 variants (rfm, dinov3b-9l, dinov3,
#                                                #    retfound_dinov2, retfound_mae, retfound_green) for one
#                                                #    dataset -> outputs/results_messidor.pdf (separate
#                                                #    manifest, doesn't touch the combined
#                                                #    outputs/results_manifest.txt/results.pdf)
#   ./train_test.sh --manifest my_runs.txt          # override the manifest file (and derive my_runs.pdf);
#                                                 # not auto-reset between invocations, so repeated calls
#                                                 # accumulate every individual training run into it
#   ./train_test.sh --manifest outputs-LN/results_manifest.txt  # --manifest in another directory relocates
#                                                 # the whole run there: every run's hydra.run.dir is
#                                                 # automatically pointed at that same directory too (unless
#                                                 # you pass your own hydra.run.dir=... override, which wins)
#   ./train_test.sh --dataset messidor --manifest my_runs.txt   # combine both
#   ./train_test.sh trainer.max_epochs=100                       # any key=value arg is passed through
#                                                             # as a Hydra override to every train.py call
#
# Valid --dataset values: messidor, aptos, idrid, papila, glaucoma_fundus
#
# All stdout/stderr (from this script and every train.py/compile_results.py
# call) is both printed to the terminal and saved under <manifest dir>/logs/
# (outputs/logs/ by default).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# All datasets and all variants trained for each (see model.py for what each
# variant means). Both the bare invocation (ALL_EXPERIMENTS) and --dataset
# are built from these two lists, so they always stay in sync.
ALL_DATASETS=(messidor aptos idrid papila glaucoma_fundus)
DATASET_VARIANTS=(rfm dinov3b-9l dinov3 retfound_dinov2 retfound_mae retfound_green)

ALL_EXPERIMENTS=()
for dataset in "${ALL_DATASETS[@]}"; do
    for variant in "${DATASET_VARIANTS[@]}"; do
        ALL_EXPERIMENTS+=("${dataset}_${variant}")
    done
done

# Only RFM
# ALL_EXPERIMENTS=()
# for dataset in "${ALL_DATASETS[@]}"; do
#     ALL_EXPERIMENTS+=("${dataset}_rfm")
# done

DATASET=""
MANIFEST_OVERRIDE=""
POSITIONAL=()
OVERRIDES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="${2:?Usage: ./train_test.sh --dataset <messidor|aptos|idrid|papila|glaucoma_fundus>}"
            shift 2 ;;
        --manifest)
            MANIFEST_OVERRIDE="${2:?Usage: ./train_test.sh --manifest <path.txt>}"
            shift 2 ;;
        *=*)
            # Hydra override (key=value), e.g. trainer.max_epochs=100
            OVERRIDES+=("$1")
            shift ;;
        *)
            POSITIONAL+=("$1")
            shift ;;
    esac
done

MANIFEST="outputs/results_manifest.txt"
REPORT="outputs/results.pdf"
LOG_TAG="all"

if [[ -n "$DATASET" ]]; then
    EXPERIMENTS=()
    for variant in "${DATASET_VARIANTS[@]}"; do
        EXPERIMENTS+=("${DATASET}_${variant}")
    done
    for cfg in "${EXPERIMENTS[@]}"; do
        if [[ ! -f "configs/experiment/${cfg}.yaml" ]]; then
            echo "Unknown dataset '$DATASET' (no config configs/experiment/${cfg}.yaml)" >&2
            exit 1
        fi
    done
    MANIFEST="outputs/results_${DATASET}.txt"
    REPORT="outputs/results_${DATASET}.pdf"
    LOG_TAG="$DATASET"
    if [[ -z "$MANIFEST_OVERRIDE" ]]; then
        rm -f "$MANIFEST"
    fi
else
    EXPERIMENTS=("${POSITIONAL[@]:-${ALL_EXPERIMENTS[@]}}")
fi

if [[ -n "$MANIFEST_OVERRIDE" ]]; then
    MANIFEST="$MANIFEST_OVERRIDE"
    REPORT="${MANIFEST%.txt}.pdf"
    LOG_TAG="$(basename "${MANIFEST%.txt}")"
fi

# Every experiment config hardcodes hydra.run.dir under a literal "outputs/",
# independent of the manifest path, so --manifest alone would leave the
# actual run (checkpoints/test.csv/plots) behind in outputs/ while only the
# manifest/log/PDF moved. Point hydra.run.dir at the manifest's own directory
# too, unless the caller already passed their own override (which wins) -
# so one --manifest relocates the whole run, not just its bookkeeping.
HAS_RUN_DIR_OVERRIDE=false
for o in "${OVERRIDES[@]}"; do
    if [[ "$o" == hydra.run.dir=* ]]; then
        HAS_RUN_DIR_OVERRIDE=true
        break
    fi
done
if ! $HAS_RUN_DIR_OVERRIDE; then
    OUTPUTS_ROOT="$(dirname "$MANIFEST")"
    OVERRIDES+=("hydra.run.dir=${OUTPUTS_ROOT}/\${experiment_name}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}")
fi

# Logs live under the manifest's own directory, not a hardcoded "outputs/",
# so a --manifest pointed elsewhere (e.g. outputs-LN/results_manifest.txt)
# keeps its log alongside its runs instead of splitting output across two dirs.
LOG_DIR="$(dirname "$MANIFEST")/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_test_${LOG_TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
echo "Recording runs to manifest: $MANIFEST"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
    echo "Applying overrides to every run: ${OVERRIDES[*]}"
fi

for cfg in "${EXPERIMENTS[@]}"; do
    echo "=== Running $cfg ==="
    python train.py --config-name="$cfg" results_manifest="$MANIFEST" "${OVERRIDES[@]}"
done

python compile_results.py --manifest "$MANIFEST" --out "$REPORT"
echo "Done. See $REPORT"
