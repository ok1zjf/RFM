#!/usr/bin/env python
"""Convert raw GlaucomaFundus (GFID) images into the datasets/gfid/ layout.

datasets/gfid_splits.csv's class directories (normal/early/advanced) are the whole
layout - the train/val/test split lives in the CSV's `split` column, not in a
directory. The raw Harvard Dataverse download spells the same three classes
normal_control/early_glaucoma/advanced_glaucoma, with matching filenames per
class, so conversion is a straight copy/re-encode under RAW_CLASS_DIRS. Images
are already small (240x240) and pre-cropped, so no resizing is applied.

Usage:
    python convert_gfid.py [--raw-dir datasets/raw/gfid] [--csv datasets/gfid_splits.csv] [--out-dir datasets/gfid]
"""
import argparse
from pathlib import Path

from turbo_pillow import use_turbo_pillow

use_turbo_pillow()  # before anything imports PIL (dataset_prep_common does)

import pandas as pd  # noqa: E402

from dataset_prep_common import iter_progress, RESIZE_PARAMS, resize_reencode, save_format  # noqa: E402

# datasets/gfid/<class>/ -> the directory the raw download calls that class
RAW_CLASS_DIRS = {"normal": "normal_control", "early": "early_glaucoma", "advanced": "advanced_glaucoma"}


def find_class_dirs(raw_dir, class_names):
    found = {}
    for name in class_names:
        dirs = [p for p in Path(raw_dir).rglob(name) if p.is_dir()]
        if dirs:
            found[name] = dirs[0]
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="datasets/raw/gfid")
    parser.add_argument("--csv", default="datasets/gfid_splits.csv")
    parser.add_argument("--out-dir", default="datasets/gfid")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    class_names = sorted({RAW_CLASS_DIRS[Path(f).parts[-2]] for f in df["filename"]})
    class_dirs = find_class_dirs(args.raw_dir, class_names)

    converted, missing = 0, []
    for row in iter_progress(df.itertuples(), total=len(df), label="gfid"):
        parts = Path(row.filename).parts  # (class, basename)
        raw_class = RAW_CLASS_DIRS[parts[-2]]
        class_dir = class_dirs.get(raw_class)
        src = (class_dir / parts[-1]) if class_dir is not None else None
        if src is None or not src.is_file():
            missing.append(row.filename)
            continue
        dst = Path(args.out_dir) / row.filename
        resize_reencode(src, dst, fmt=save_format("gfid", dst.name), **RESIZE_PARAMS["gfid"])
        converted += 1

    print(f"gfid: converted {converted}/{len(df)} images -> {args.out_dir}")
    if missing:
        print(f"  {len(missing)} missing (not found under {args.raw_dir}):")
        for m in missing[:10]:
            print(f"    {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()
