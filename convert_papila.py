#!/usr/bin/env python
"""Convert raw PAPILA images into the datasets/papila/ layout.

Matches each datasets/papila_splits.csv row (filename e.g. "images/RET028OD.jpg" - one
flat directory, since both the class and the train/val/test split live in the
CSV's own columns) to the raw figshare download's flat
FundusImages/RETnnnOD|OS.jpg files by stem (globally unique, 1:1), then
square-pads to 512x512 (fit the long side, pad the short side with black) -
matching the images-512-sq convention papila_*.yaml configs expect.

Usage:
    python convert_papila.py [--raw-dir datasets/raw/papila] [--csv datasets/papila_splits.csv] [--out-dir datasets/papila]
"""
import argparse
from pathlib import Path

from turbo_pillow import use_turbo_pillow

use_turbo_pillow()  # before anything imports PIL (dataset_prep_common does)

import pandas as pd  # noqa: E402

from dataset_prep_common import iter_progress, RESIZE_PARAMS, resize_reencode, save_format, stem_lookup  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="datasets/raw/papila")
    parser.add_argument("--csv", default="datasets/papila_splits.csv")
    parser.add_argument("--out-dir", default="datasets/papila")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    raw_lookup = stem_lookup(args.raw_dir, exts=(".jpg", ".jpeg"))

    converted, missing = 0, []
    for row in iter_progress(df.itertuples(), total=len(df), label="papila"):
        stem = Path(row.filename).stem.lower()
        src = raw_lookup.get(stem)
        if src is None:
            missing.append(row.filename)
            continue
        dst = Path(args.out_dir) / row.filename
        resize_reencode(src, dst, fmt=save_format("papila", row.filename), **RESIZE_PARAMS["papila"])
        converted += 1

    print(f"papila: converted {converted}/{len(df)} images -> {args.out_dir}")
    if missing:
        print(f"  {len(missing)} missing (not found under {args.raw_dir}):")
        for m in missing[:10]:
            print(f"    {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()
