#!/usr/bin/env python
"""Convert raw APTOS 2019 images into the datasets/aptos-2019/ layout.

Matches each datasets/aptos-2019_splits.csv row (filename e.g. "images/<id_code>.jpg")
to the raw Kaggle download (train_images/<id_code>.png) by filename stem, then
scales the short side to 512 and center-crops the long side down to 512
(square="crop") before re-encoding, at the CSV's exact relative path, so
images_root=datasets/aptos-2019/ resolves every row. Center-cropping matches
the original APTOS-2019-512 convention (confirmed by visual inspection: the
fundus circle fills the frame, edges trimmed - not letterboxed). Squaring is
also required because raw images have inconsistent aspect ratios and
dataset.py's test-time transform only resizes, without cropping, so un-padded
images of different aspect ratios can't be batched together.

Usage:
    python convert_aptos.py [--raw-dir datasets/raw/aptos-2019] [--csv datasets/aptos-2019_splits.csv] [--out-dir datasets/aptos-2019]
"""
import argparse
from pathlib import Path

from turbo_pillow import use_turbo_pillow

use_turbo_pillow()  # before anything imports PIL (dataset_prep_common does)

import pandas as pd  # noqa: E402

from dataset_prep_common import iter_progress, RESIZE_PARAMS, resize_reencode, save_format, stem_lookup  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="datasets/raw/aptos-2019")
    parser.add_argument("--csv", default="datasets/aptos-2019_splits.csv")
    parser.add_argument("--out-dir", default="datasets/aptos-2019")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    raw_lookup = stem_lookup(args.raw_dir, exts=(".png", ".jpg", ".jpeg"))

    converted, missing = 0, []
    for row in iter_progress(df.itertuples(), total=len(df), label="aptos-2019"):
        stem = Path(row.filename).stem.lower()
        src = raw_lookup.get(stem)
        if src is None:
            missing.append(row.filename)
            continue
        dst = Path(args.out_dir) / row.filename
        resize_reencode(src, dst, fmt=save_format("aptos-2019", row.filename), **RESIZE_PARAMS["aptos-2019"])
        converted += 1

    print(f"aptos-2019: converted {converted}/{len(df)} images -> {args.out_dir}")
    if missing:
        print(f"  {len(missing)} missing (not found under {args.raw_dir}):")
        for m in missing[:10]:
            print(f"    {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()
