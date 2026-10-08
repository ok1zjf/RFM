#!/usr/bin/env python
"""Convert raw Messidor-2 images into the datasets/messidor2/ layout.

Matches each datasets/messidor2_splits.csv row (filename e.g. "images/<name>.png") to
the raw Kaggle mirror download by filename stem (case-insensitive, since the
mirror has a mix of .jpg/.png extensions per stem), then resizes (short side
512px) and re-encodes to JPEG at the CSV's exact relative path.

Usage:
    python convert_messidor2.py [--raw-dir datasets/raw/messidor2] [--csv datasets/messidor2_splits.csv] [--out-dir datasets/messidor2]
"""
import argparse
from pathlib import Path

from turbo_pillow import use_turbo_pillow

use_turbo_pillow()  # before anything imports PIL (dataset_prep_common does)

import pandas as pd  # noqa: E402

from dataset_prep_common import iter_progress, RESIZE_PARAMS, resize_reencode, save_format, stem_lookup  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="datasets/raw/messidor2")
    parser.add_argument("--csv", default="datasets/messidor2_splits.csv")
    parser.add_argument("--out-dir", default="datasets/messidor2")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    raw_lookup = stem_lookup(args.raw_dir, exts=(".png", ".jpg", ".jpeg"))

    converted, missing = 0, []
    for row in iter_progress(df.itertuples(), total=len(df), label="messidor2"):
        stem = Path(row.filename).stem.lower()
        src = raw_lookup.get(stem)
        if src is None:
            missing.append(row.filename)
            continue
        dst = Path(args.out_dir) / row.filename
        resize_reencode(src, dst, fmt=save_format("messidor2", row.filename), **RESIZE_PARAMS["messidor2"])
        converted += 1

    print(f"messidor2: converted {converted}/{len(df)} images -> {args.out_dir}")
    if missing:
        print(f"  {len(missing)} missing (not found under {args.raw_dir}):")
        for m in missing[:10]:
            print(f"    {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()
