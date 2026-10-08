#!/usr/bin/env python
"""Convert raw IDRiD images into the datasets/idrid/ layout.

The official IDRiD "Disease Grading" release ships separate Training Set and
Testing Set folders whose filenames collide (both start at IDRiD_001), so a
stem alone can't tell them apart. datasets/idrid_splits.csv's `filename` keeps that
distinction in its leading path component - "train/IDRiD_nnn.jpg" vs
"test/IDRiD_nnn.jpg" is IDRiD's own set designation, not this repo's
train/val/test split (which lives in the `split` column; its test rows are
aligned with this directory, only the train vs. val boundary is redrawn).
Rows under "test/" are matched against images found in a "test"-named raw
directory, the rest against a "train"-named one.

Images are scaled to short side 512 and center-cropped to 512x512
(square="crop"), matching the original IDIRD-retfound convention (confirmed
by visual inspection: the fundus circle fills the frame, edges trimmed - not
letterboxed). Squaring is also required because raw images are 4288x2848 (not
square) and dataset.py's test-time transform only resizes, without cropping,
so un-cropped images of different aspect ratios can't be batched together.

Usage:
    python convert_idrid.py [--raw-dir datasets/raw/idrid] [--csv datasets/idrid_splits.csv] [--out-dir datasets/idrid]
"""
import argparse
from pathlib import Path

from turbo_pillow import use_turbo_pillow

use_turbo_pillow()  # before anything imports PIL (dataset_prep_common does)

import pandas as pd  # noqa: E402

from dataset_prep_common import iter_progress, RESIZE_PARAMS, resize_reencode, save_format  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def bucket_lookup(raw_dir):
    """Split raw images into train/test lookups by directory name."""
    train_lookup, test_lookup = {}, {}
    for p in Path(raw_dir).rglob("*"):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        path_str = str(p.parent).lower()
        stem = p.stem.lower()
        if "test" in path_str:
            test_lookup[stem] = p
        elif "train" in path_str:
            train_lookup[stem] = p
    return train_lookup, test_lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="datasets/raw/idrid")
    parser.add_argument("--csv", default="datasets/idrid_splits.csv")
    parser.add_argument("--out-dir", default="datasets/idrid")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    train_lookup, test_lookup = bucket_lookup(args.raw_dir)

    converted, missing = 0, []
    for row in iter_progress(df.itertuples(), total=len(df), label="idrid"):
        stem = Path(row.filename).stem.lower()
        lookup = test_lookup if Path(row.filename).parts[0] == "test" else train_lookup
        src = lookup.get(stem)
        if src is None:
            missing.append(row.filename)
            continue
        dst = Path(args.out_dir) / row.filename
        resize_reencode(src, dst, fmt=save_format("idrid", row.filename), **RESIZE_PARAMS["idrid"])
        converted += 1

    print(f"idrid: converted {converted}/{len(df)} images -> {args.out_dir}")
    if missing:
        print(f"  {len(missing)} missing (not found under {args.raw_dir}):")
        for m in missing[:10]:
            print(f"    {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()
