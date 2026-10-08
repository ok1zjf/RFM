"""Predict with one or more trained heads on every image in a directory.

For each head (a run directory with config.yaml + checkpoints/{best,last}.pt,
e.g. released/papila_rfm or outputs/<experiment>/<date>/<time>), rebuilds the
model (backbone from weights/) with the head's own test transform, runs it over
every image under <image_dir> (recursively, sorted by path), and writes
<out-dir>/<experiment_name>_test_images.csv - e.g. papila_rfm_test_images.csv.
The columns are the same as a test run's test_images.csv: `id` is the image
path relative to <image_dir>, `true_label` is empty, and `pred_label` /
`prob_<class>` are computed by the same code path as the test step
(train.predict), so an image gets the same probabilities as in test_images.csv.

With `--preprocess none` (the default), images are fed to the head's test
transform as they are, so they should already be prepared the way the head's
dataset was, e.g. by convert_<dataset>.py; such images give bit-exact
results. `--preprocess auto` is for raw images: it applies the converter's
steps in memory (dataset_prep_common.prepare_image) - the dataset's
resize/pad/crop to 512, then a JPEG round trip at quality 95 where the
converter saves JPEG - with the environment's IJG Pillow rather than the
converters' libjpeg-turbo one, so its results are close to, not identical
with, converting the images first. Don't use it on already-prepared images,
which it would JPEG-compress a second time.

Usage:
    python predict.py path/to/images released/papila_rfm
    python predict.py path/to/images released/*_rfm --out-dir preds/
    python predict.py path/to/images --manifest released/results_manifest.txt
    python predict.py path/to/raw_images released/papila_rfm --preprocess auto
"""
import argparse
import functools
import os

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset import IMAGE_EXTENSIONS, ImageDirDataset, build_test_transform, skip_none_collate
from dataset_prep_common import RESIZE_PARAMS, prepare_image
from train import (build_model, head_checkpoint_path, load_head_checkpoint, predict, resolve_eval_crop_pct,
                   resolve_normalization, set_seed, write_test_images_csv)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def head_dataset(cfg) -> str:
    """The datasets/ directory name of the dataset a head was trained on, e.g.
    "papila" for `data.images_root: datasets/papila/`."""
    return os.path.basename(os.path.normpath(cfg.data.images_root))


def predict_dir(run_dir: str, image_dir: str, out_path: str, skip_errors: bool = False,
                preprocess: str = "none"):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = resolve_normalization(cfg)
    transform = build_test_transform(cfg.model.img_size, mean=mean, std=std, crop_pct=resolve_eval_crop_pct(cfg))
    prepare = None
    if preprocess == "auto":
        dataset_name = head_dataset(cfg)
        prepare = functools.partial(prepare_image, dataset=dataset_name)
        print(f"[{cfg.experiment_name}] preprocessing as {dataset_name}: {RESIZE_PARAMS[dataset_name]}")
    dataset = ImageDirDataset(image_dir, transform, skip_errors=skip_errors, preprocess=prepare)
    if len(dataset) == 0:
        raise ValueError(f"no images ({', '.join(IMAGE_EXTENSIONS)}) found under {image_dir}")

    model = build_model(cfg, REPO_ROOT, device)
    ckpt_path = head_checkpoint_path(run_dir, cfg)
    ckpt = load_head_checkpoint(ckpt_path, model, cfg, device)

    loader = DataLoader(dataset, batch_size=cfg.trainer.eval_batch_size, shuffle=False,
                        num_workers=cfg.trainer.eval_num_workers, pin_memory=True, collate_fn=skip_none_collate)

    y_prob, _, ids, _ = predict(model, loader, device)
    write_test_images_csv(out_path, ids, y_prob, cfg.data.class_names)
    skipped = len(dataset) - len(ids)
    print(f"[{cfg.experiment_name}] predicted {len(ids)} images from {image_dir} with "
          f"{ckpt_path} (epoch {ckpt['epoch']})"
          f"{f', skipped {skipped} unreadable' if skipped else ''} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_dir", help="Directory of images to predict on (searched recursively)")
    parser.add_argument("run_dirs", nargs="*", default=[],
                        help="Run directories, each containing config.yaml and checkpoints/{best,last}.pt")
    parser.add_argument("--manifest", default=None,
                        help="Manifest file (one run dir per line, e.g. released/results_manifest.txt) to add to run_dirs")
    parser.add_argument("--out-dir", default=".",
                        help="Directory for the <experiment_name>_test_images.csv files (default: current directory)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSVs")
    parser.add_argument("--skip-errors", action="store_true",
                        help="Skip (and log) images that fail to load instead of stopping")
    parser.add_argument("--preprocess", choices=["none", "auto"], default="none",
                        help="none: images are already prepared like the head's dataset (bit-exact); auto: "
                             "prepare raw images in memory with convert_<dataset>.py's steps (close, not "
                             "bit-exact)")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        parser.error(f"image_dir {args.image_dir!r} is not a directory")

    run_dirs = list(args.run_dirs)
    if args.manifest is not None:
        with open(args.manifest) as f:
            run_dirs += [line.strip() for line in f if line.strip()]
    if not run_dirs:
        parser.error("no run directories given - pass some, or --manifest <file>")

    # Resolve every output path up front, so a name collision or an existing file
    # stops the run before any model is built.
    os.makedirs(args.out_dir, exist_ok=True)
    jobs = {}
    for run_dir in run_dirs:
        run_dir = os.path.abspath(run_dir)
        cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
        experiment_name = cfg.experiment_name
        if args.preprocess == "auto" and head_dataset(cfg) not in RESIZE_PARAMS:
            parser.error(f"--preprocess auto: no preparation known for {run_dir}'s dataset "
                         f"{head_dataset(cfg)!r} (known: {', '.join(RESIZE_PARAMS)})")
        out_path = os.path.join(args.out_dir, f"{experiment_name}_test_images.csv")
        if out_path in jobs:
            parser.error(f"{jobs[out_path]} and {run_dir} would both write {out_path}")
        if os.path.exists(out_path) and not args.overwrite:
            parser.error(f"{out_path} already exists (pass --overwrite to replace it)")
        jobs[out_path] = run_dir

    image_dir = os.path.abspath(args.image_dir)
    for out_path, run_dir in jobs.items():
        predict_dir(run_dir, image_dir, out_path, skip_errors=args.skip_errors, preprocess=args.preprocess)


if __name__ == "__main__":
    main()
