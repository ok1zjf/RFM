#!/usr/bin/env bash
# Download the pretrained weights needed by model.py:
#   weights/rfm-1.3.pt                        - stripped RFM/SSL backbone checkpoint (Google Drive)
#   weights/vit_base_patch16_dinov3.lvd1689m   - LVD-1689M-pretrained DINOv3 backbone (Google Drive)
#   weights/LICENSE-DINOv3.md                  - DINOv3 License, which covers both files above (Google Drive,
#                                                falling back to the repository's own copy)
#   weights/RETFound_mae_natureCFP.pth         - RETFound ViT-L/16 MAE checkpoint (HuggingFace, gated)
#   weights/RETFound_dinov2_meh.pth            - RETFound ViT-L/14 DINOv2 checkpoint (HuggingFace, gated)
#   weights/retfoundgreen_statedict.pth        - RETFound-Green ViT-S/14 checkpoint (GitHub release)
#
#
# The two HuggingFace checkpoints are gated: log in at their model page and
# request access (https://huggingface.co/YukunZhou/RETFound_mae_natureCFP,
# https://huggingface.co/YukunZhou/RETFound_dinov2_meh), create an access
# token at https://huggingface.co/settings/tokens, then export it:
#   export HF_TOKEN=hf_...
# before running this script. Without HF_TOKEN set, those two are skipped
# with an explanatory error (not a silent gap) and everything else still
# downloads.
#
# Usage:
#   ./download_weights.sh            # download all 6, skip files that already exist
#   ./download_weights.sh --force    # re-download even if the file already exists
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

OUT_DIR="weights"
HF_TOKEN="${HF_TOKEN:-}"

# url filename
FILES=(
    "https://drive.google.com/file/d/1YqWkop3vk4EajrdZ_KApsmg0QhGIhVVk/view?usp=sharing rfm-1.3.pt"
    "https://drive.google.com/file/d/18ltLPGB_WlGWBq61rmqv6tRDn_iuMwAD/view?usp=sharing vit_base_patch16_dinov3.lvd1689m"
    "https://drive.google.com/file/d/1HN_1aMnnsADu9I38TcftSsiut5dptYED/view?usp=sharing LICENSE-DINOv3.md"
)

# Plain HTTP(S) downloads - no Google Drive confirm-token dance needed. The
# HuggingFace ones require Authorization: Bearer $HF_TOKEN (see header comment);
# the GitHub release asset needs no auth.
HTTP_FILES=(
    "https://huggingface.co/YukunZhou/RETFound_mae_natureCFP/resolve/main/RETFound_mae_natureCFP.pth RETFound_mae_natureCFP.pth"
    "https://huggingface.co/YukunZhou/RETFound_dinov2_meh/resolve/main/RETFound_dinov2_meh.pth RETFound_dinov2_meh.pth"
    "https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth retfoundgreen_statedict.pth"
)

FORCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=1
            shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

extract_file_id() {
    local url="$1"
    if [[ "$url" =~ /file/d/([^/]+) ]]; then
        printf '%s' "${BASH_REMATCH[1]}"
    elif [[ "$url" =~ [\?\&]id=([^&]+) ]]; then
        printf '%s' "${BASH_REMATCH[1]}"
    else
        return 1
    fi
}

html_attr_value() {
    local text="$1" attr="$2"
    printf '%s' "$text" | sed -n "s/.*${attr}=\"\([^\"]*\)\".*/\1/p" | head -n 1
}

html_input_value() {
    local file="$1" name="$2"
    local line
    line="$(grep -m1 "name=\"$name\"" "$file" || true)"
    [[ -z "$line" ]] && return 1
    html_attr_value "$line" "value"
}

is_html_file() {
    local path="$1"
    if command -v file >/dev/null 2>&1; then
        [[ "$(file -b --mime-type "$path" 2>/dev/null || true)" == "text/html" ]]
    else
        head -c 256 "$path" | grep -qiE '<!doctype html|<html'
    fi
}

download_gdrive_one() {
    # Downloads into a fresh temp file and only replaces `out` (via `mv`) once
    # the download is verified. `out` may be an existing regular file OR a
    # symlink to a file elsewhere (e.g. weights/ symlinks into rfm-jepa/) -
    # writing straight into it via `curl -o` would silently overwrite whatever
    # it points at, so we never open `out` for writing directly.
    local url="$1" out="$2"

    local file_id
    if ! file_id="$(extract_file_id "$url")"; then
        echo "ERROR: Could not parse file id from: $url" >&2
        return 2
    fi

    local tmpdir cookie page headers dl_tmp
    tmpdir="$(mktemp -d)"
    cookie="$tmpdir/cookie.txt"
    page="$tmpdir/page.html"
    headers="$tmpdir/headers.txt"
    dl_tmp="$(dirname -- "$out")/.$(basename -- "$out").part"
    trap 'rm -rf "$tmpdir" "$dl_tmp"' RETURN

    # First request: cookies + either direct download or interstitial HTML.
    # Explicit `|| return` throughout, rather than relying on `set -e`: this
    # function is called as `if ! download_gdrive_one ...`, and bash disables
    # errexit for the whole body of a function invoked as an if/while/&&/||
    # condition, so a bare failing command here would silently fall through
    # to later commands (e.g. `mv` on a file curl never wrote) instead of
    # stopping at the real point of failure.
    curl -fsSL --connect-timeout 15 --max-time 30 -D "$headers" -c "$cookie" \
        "https://drive.google.com/uc?export=download&id=${file_id}" \
        -o "$page" || return 5

    # If interstitial contains a download-form, use its action + hidden inputs.
    local formline action confirm uuid at
    formline="$(grep -m1 'id="download-form"' "$page" || true)"
    action=""
    if [[ -n "$formline" ]]; then
        action="$(html_attr_value "$formline" "action" || true)"
    fi

    if [[ -n "$action" ]]; then
        confirm="$(html_input_value "$page" "confirm" || true)"
        uuid="$(html_input_value "$page" "uuid" || true)"
        at="$(html_input_value "$page" "at" || true)"  # sometimes present

        if [[ -z "$confirm" || -z "$uuid" ]]; then
            echo "ERROR: Could not extract confirm/uuid for: $url" >&2
            return 3
        fi

        local dl="${action}?id=${file_id}&export=download&confirm=${confirm}&uuid=${uuid}"
        [[ -n "$at" ]] && dl="${dl}&at=${at}"

        echo "Downloading -> $out"
        curl -fL --connect-timeout 15 --max-time 180 \
            --retry 10 --retry-all-errors --retry-max-time 600 -b "$cookie" -c "$cookie" \
            -H "Referer: https://drive.google.com/" \
            "$dl" -o "$dl_tmp" || return 5
    else
        # Fallback to direct endpoint
        echo "Downloading -> $out"
        curl -fL --connect-timeout 15 --max-time 180 \
            --retry 10 --retry-all-errors --retry-max-time 600 -b "$cookie" -c "$cookie" \
            "https://drive.google.com/uc?export=download&id=${file_id}" \
            -o "$dl_tmp" || return 5
    fi

    if is_html_file "$dl_tmp"; then
        echo "ERROR: Downloaded HTML instead of file for URL: $url" >&2
        echo "       Hint: file may require permission/login, or Google changed the flow." >&2
        return 4
    fi

    # Atomic replace: unlinks any existing symlink instead of writing through it.
    mv -f -- "$dl_tmp" "$out" || return 5
}

download_http_one() {
    # Same atomic-temp-file-then-mv discipline as download_gdrive_one, for a
    # plain HTTP(S) URL (GitHub release asset, or HuggingFace resolve/main).
    local url="$1" out="$2"
    local dl_tmp
    dl_tmp="$(dirname -- "$out")/.$(basename -- "$out").part"
    trap 'rm -f "$dl_tmp"' RETURN

    local auth_args=()
    if [[ "$url" == *huggingface.co* ]]; then
        if [[ -z "$HF_TOKEN" ]]; then
            echo "ERROR: $url is a gated HuggingFace repo and needs a token." >&2
            echo "       Log in, request access at the model page, create a token at" >&2
            echo "       https://huggingface.co/settings/tokens, then export HF_TOKEN=<token>." >&2
            return 6
        fi
        auth_args=(-H "Authorization: Bearer $HF_TOKEN")
    fi

    echo "Downloading -> $out"
    curl -fL --connect-timeout 15 --max-time 1800 \
        --retry 10 --retry-all-errors --retry-max-time 1800 \
        "${auth_args[@]}" "$url" -o "$dl_tmp" || return 5

    if is_html_file "$dl_tmp"; then
        echo "ERROR: Downloaded HTML instead of file for URL: $url" >&2
        echo "       Hint: the token may be missing/expired, or access wasn't granted yet." >&2
        return 4
    fi

    mv -f -- "$dl_tmp" "$out" || return 5
}

FAILED=()
for entry in "${FILES[@]}"; do
    url="${entry%% *}"
    name="${entry#* }"
    out_path="$OUT_DIR/$name"

    if [[ -e "$out_path" && "$FORCE" == 0 ]]; then
        echo "Skipping $out_path (already exists, use --force to re-download)"
        continue
    fi

    if ! download_gdrive_one "$url" "$out_path"; then
        # The DINOv3 License must accompany the weights; the repository ships
        # an identical copy, so fall back to it rather than leave weights/
        # without one.
        if [[ "$name" == "LICENSE-DINOv3.md" && -f "LICENSE-DINOv3.md" ]]; then
            echo "[$name] download failed, copying the repository's own copy instead" >&2
            cp -f -- "LICENSE-DINOv3.md" "$out_path"
            continue
        fi
        echo "[$name] download failed, skipping (re-run with --force to retry)" >&2
        FAILED+=("$name")
    fi
done

for entry in "${HTTP_FILES[@]}"; do
    url="${entry%% *}"
    name="${entry#* }"
    out_path="$OUT_DIR/$name"

    if [[ -e "$out_path" && "$FORCE" == 0 ]]; then
        echo "Skipping $out_path (already exists, use --force to re-download)"
        continue
    fi

    if ! download_http_one "$url" "$out_path"; then
        echo "[$name] download failed, skipping (re-run with --force to retry)" >&2
        FAILED+=("$name")
    fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Done, but failed: ${FAILED[*]}" >&2
    exit 1
fi
echo "All done."
