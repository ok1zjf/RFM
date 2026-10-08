#!/usr/bin/env bash
# Download raw source data for the 5 fundus datasets used by
# configs/experiment/*.yaml into datasets/raw/<name>/.
#
# Fully automatic (no auth): papila, gfid
# Needs a Kaggle account + API token (~/.kaggle/kaggle.json): aptos, messidor2
#   - aptos additionally needs a one-time manual "Join Competition" click at
#     https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules
# idrid defaults to an automatable CC-BY-4.0 Zenodo mirror; --official prints
#   IEEE DataPort instructions instead (that source needs a login, no CLI).
#
# Usage:
#   ./download_datasets.sh                    # download all 5, skip already-present
#   ./download_datasets.sh --only papila       # just one dataset
#   ./download_datasets.sh --force             # re-download even if present
#   ./download_datasets.sh --only idrid --official
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RAW_DIR="datasets/raw"
ALL_DATASETS=(papila gfid idrid aptos messidor2)

PAPILA_URL="https://ndownloader.figshare.com/files/35013982"
GFID_URL="https://dataverse.harvard.edu/api/access/datafile/3314942"
IDRID_ZENODO_URL="https://zenodo.org/api/records/17219542/files/B.%20Disease%20Grading.zip/content"

ONLY=""
FORCE=0
OFFICIAL=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)
            ONLY="${2:?Usage: ./download_datasets.sh --only <papila|gfid|idrid|aptos|messidor2>}"
            shift 2 ;;
        --force)
            FORCE=1
            shift ;;
        --official)
            OFFICIAL=1
            shift ;;
        -h|--help)
            sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1 ;;
    esac
done

if [[ -n "$ONLY" ]]; then
    DATASETS=("$ONLY")
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

already_present() {
    # Ignores stray .download-*.zip leftovers from an interrupted previous
    # attempt (e.g. killed mid-download) - those don't count as "downloaded".
    local dir="$1"
    [[ -d "$dir" ]] && [[ -n "$(find "$dir" -type f ! -name '.download-*.zip' -print -quit 2>/dev/null)" ]]
}

download_unzip() {
    # download_unzip <url> <dest_dir>
    local url="$1" dest="$2"
    mkdir -p "$dest"
    rm -f "$dest"/.download-*.zip   # clear any stray leftover from an interrupted previous attempt
    local tmp_zip
    tmp_zip="$(mktemp "${dest}/.download-XXXXXX.zip")"
    trap 'rm -f "$tmp_zip"' RETURN
    echo "  downloading -> $dest"
    # --retry* only kicks in once an attempt actually completes (with an
    # error); a connection that hangs with no response at all (observed
    # against a struggling Zenodo mirror - accepts the connection, then never
    # sends anything) never completes on its own, so --max-time bounds each
    # individual attempt (curl gives every retry its own fresh budget),
    # letting a truly stalled one fail fast enough for --retry to take over.
    curl -fL --progress-bar --connect-timeout 15 --max-time 180 \
        --retry 10 --retry-all-errors --retry-max-time 600 "$url" -o "$tmp_zip"
    echo "  unzipping..."
    unzip -q -o "$tmp_zip" -d "$dest"
    rm -f "$tmp_zip"
    trap - RETURN
}

download_papila() {
    local dest="$RAW_DIR/papila"
    if already_present "$dest" && [[ "$FORCE" == 0 ]]; then
        echo "[papila] already present, skipping (use --force to re-download)"
        return
    fi
    echo "[papila] downloading from figshare (CC BY 4.0, ~563MB)..."
    download_unzip "$PAPILA_URL" "$dest"
    echo "[papila] done"
}

download_gfid() {
    local dest="$RAW_DIR/gfid"
    if already_present "$dest" && [[ "$FORCE" == 0 ]]; then
        echo "[gfid] already present, skipping (use --force to re-download)"
        return
    fi
    echo "[gfid] downloading from Harvard Dataverse (CC0, ~124MB)..."
    download_unzip "$GFID_URL" "$dest"
    echo "[gfid] done"
}

download_idrid() {
    local dest="$RAW_DIR/idrid"
    if already_present "$dest" && [[ "$FORCE" == 0 ]]; then
        echo "[idrid] already present, skipping (use --force to re-download)"
        return
    fi
    if [[ "$OFFICIAL" == 1 ]]; then
        cat <<EOF
[idrid] Official source requires a free IEEE DataPort account (no CLI download):
  1. Create an account / log in: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
  2. Download the "B. Disease Grading" component.
  3. Unzip it so its contents land under: $dest/
Re-run this script (without --official) once done, or place the files and
run: python convert_idrid.py
EOF
        return
    fi
    echo "[idrid] downloading 'B. Disease Grading' from the CC-BY-4.0 Zenodo mirror (~212MB)..."
    echo "        (unofficial re-host of the IEEE DataPort release; pass --official for the canonical source)"
    download_unzip "$IDRID_ZENODO_URL" "$dest"
    echo "[idrid] done"
}

download_aptos() {
    local dest="$RAW_DIR/aptos-2019"
    if already_present "$dest" && [[ "$FORCE" == 0 ]]; then
        echo "[aptos-2019] already present, skipping (use --force to re-download)"
        return
    fi
    if ! command -v kaggle >/dev/null 2>&1; then
        cat <<EOF
[aptos-2019] Needs the Kaggle CLI: pip install kaggle
             Then place your API token at ~/.kaggle/kaggle.json
             (https://www.kaggle.com/settings -> API -> Create New Token)
EOF
        return
    fi
    mkdir -p "$dest"
    echo "[aptos-2019] downloading via Kaggle (~10GB, this will take a while)..."
    if ! kaggle competitions download -c aptos2019-blindness-detection -p "$dest"; then
        cat <<EOF
[aptos-2019] Download failed. Most likely you haven't joined the competition yet:
  1. Visit https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules
  2. Click "I Understand and Accept" (one-time, requires being logged in)
  3. Re-run this script
EOF
        return
    fi
    echo "[aptos-2019] unzipping..."
    for z in "$dest"/*.zip; do
        [[ -f "$z" ]] && unzip -q -o "$z" -d "$dest" && rm -f "$z"
    done
    echo "[aptos-2019] done"
}

download_messidor2() {
    local dest="$RAW_DIR/messidor2"
    if already_present "$dest" && [[ "$FORCE" == 0 ]]; then
        echo "[messidor2] already present, skipping (use --force to re-download)"
        return
    fi
    if ! command -v kaggle >/dev/null 2>&1; then
        cat <<EOF
[messidor2] Needs the Kaggle CLI: pip install kaggle
            Then place your API token at ~/.kaggle/kaggle.json
            (https://www.kaggle.com/settings -> API -> Create New Token)
EOF
        return
    fi
    mkdir -p "$dest"
    echo "[messidor2] downloading via Kaggle mirror mariaherrerot/messidor2preprocess..."
    if ! kaggle datasets download -d mariaherrerot/messidor2preprocess -p "$dest"; then
        echo "[messidor2] Download failed. Check your Kaggle credentials and try again." >&2
        return
    fi
    echo "[messidor2] unzipping..."
    for z in "$dest"/*.zip; do
        [[ -f "$z" ]] && unzip -q -o "$z" -d "$dest" && rm -f "$z"
    done
    echo "[messidor2] done"
}

for name in "${DATASETS[@]}"; do
    case "$name" in
        papila) download_papila ;;
        gfid) download_gfid ;;
        idrid) download_idrid ;;
        aptos) download_aptos ;;
        messidor2) download_messidor2 ;;
        *)
            echo "Unknown dataset: $name (expected one of: ${ALL_DATASETS[*]})" >&2
            exit 1 ;;
    esac
done

echo "All done."
