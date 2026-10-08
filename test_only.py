"""Re-run just the test/report step for one or more already-trained runs, without
retraining. Useful after a metrics/report-only change (e.g. the bootstrap-CI NaN
fix) when you want corrected test.csv / test_images.csv / plots without waiting
through training again.

Loads a run's saved config.yaml + head-only checkpoint (checkpoints/best.pt, or
last.pt without `trainer.select_best`), rebuilds the model (backbone from weights/) and test
dataloader, and rewrites test.csv / test_images.csv / ROC-PR-confusion-matrix
plots in place — using the exact same code path (`train.write_test_report`) that
train.py uses right after training, so results are identical to what a fresh
training run would have produced.

Usage:
    python test_only.py outputs/idrid_dinov3b-9l/2026-07-23/10-00-00
    python test_only.py outputs/idrid_dinov3b-9l/*/*/          # shell-glob multiple runs
    python test_only.py --manifest outputs/results_manifest.txt    # every run dir listed in a manifest
    python test_only.py --manifest outputs/results_manifest.txt outputs/extra_run/2026-.../   # both sources combined

Testing on a different CSV:
    python test_only.py outputs/papila_rfm/2026-09-17/08-52-05 \
        --csv-path datasets/papila_splits.csv --out-dir outputs/retest/papila_rfm_papila_splits

--csv-path replaces the run's `data.csv_path` (the CSV must use the same
phenotype column and test split label as the run's config). --out-dir writes the
report there instead of into the run directory, together with a config.yaml
(with the overridden csv_path), so the out dir can be listed in a manifest for
compile_results.py.

Testing on a subset of rows:
    python test_only.py outputs/papila_rfm/2026-09-17/08-52-05 \
        --filter split=v --out-dir outputs/retest/papila_rfm_val

--filter COLUMN=VALUE[,VALUE...] keeps only CSV rows whose COLUMN matches one of
the values, and replaces the run's `data.test_split` selection (so `split=e` is
the default behaviour). Repeat it to require several columns to match, e.g.
`--filter split=e --filter glaucoma_grade=0,1`. The filters are recorded as
`data.test_filters` in the out dir's config.yaml.

With several runs, each goes to
<out-dir>/<experiment_name>/<date>/<time>. --csv-path requires --out-dir, so a
run's original test results are never overwritten with ones from another CSV;
--filter requires it for the same reason.
"""
import argparse
import os
from typing import Optional

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset import CFPDataset, build_test_transform
from train import build_model, head_checkpoint_path, load_head_checkpoint, resolve_eval_crop_pct, resolve_normalization, set_seed, write_environment_info, write_test_report

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def retest(run_dir: str, csv_path: Optional[str] = None, out_dir: Optional[str] = None,
           filters: Optional[dict] = None):
    run_dir = os.path.abspath(run_dir)
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    # train.py enforces cudnn.deterministic=True/benchmark=False (and seeds
    # torch/numpy/random) via set_seed() before ever building a model; retest()
    # never did, so a re-test's cudnn flags silently fell back to PyTorch's
    # own defaults instead of matching how the run was originally trained.
    set_seed(cfg.seed)
    if csv_path is not None:
        cfg.data.csv_path = os.path.abspath(csv_path)
    if filters:
        cfg.data.test_filters = filters
    if out_dir is None:
        out_dir = run_dir
    else:
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(out_dir, "config.yaml"))
    print(OmegaConf.to_yaml(cfg))
    # Overwrites environment.txt with the RE-TEST's own environment, which can
    # legitimately differ from the original training run's (e.g. re-testing
    # weeks later on a machine with different driver/library versions) - that
    # drift is exactly what this file exists to catch.
    write_environment_info(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, REPO_ROOT, device)
    # Re-test whichever checkpoint the run itself tested (see train.py's
    # `trainer.select_best`), falling back to last.pt for older runs.
    ckpt_path = head_checkpoint_path(run_dir, cfg)
    # Head-only checkpoint: checks its stored backbone name against the config,
    # while the backbone itself was already loaded from weights/ by build_model.
    ckpt = load_head_checkpoint(ckpt_path, model, cfg, device)

    # Relative csv_path/images_root are relative to the repo root, not run_dir
    # (matches build_model's pretrained_weights handling in train.py).
    csv_path = cfg.data.csv_path if os.path.isabs(cfg.data.csv_path) else os.path.join(REPO_ROOT, cfg.data.csv_path)
    images_root = cfg.data.images_root if os.path.isabs(cfg.data.images_root) else os.path.join(REPO_ROOT, cfg.data.images_root)
    mean, std = resolve_normalization(cfg)
    eval_crop_pct = resolve_eval_crop_pct(cfg)
    test_set = CFPDataset(csv_path, images_root, cfg.data.phenotype, cfg.data.test_split,
                           build_test_transform(cfg.model.img_size, mean=mean, std=std, crop_pct=eval_crop_pct),
                           filters=filters)
    if len(test_set) == 0:
        raise ValueError(f"[{cfg.experiment_name}] no test rows in {csv_path} match "
                         f"{filters or {'split': cfg.data.test_split}}")
    test_loader = DataLoader(test_set, batch_size=cfg.trainer.eval_batch_size, shuffle=False,
                              num_workers=cfg.trainer.eval_num_workers, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss()

    test_metrics = write_test_report(model, test_loader, criterion, device, cfg, out_dir, ckpt["epoch"],
                                      repo_root=REPO_ROOT)
    print(f"[{cfg.experiment_name}] re-tested epoch {ckpt['epoch']} on {csv_path} "
          f"({len(test_set)} rows, {filters or {'split': cfg.data.test_split}}): "
          f"auroc={test_metrics['auroc']:.4f} acc={test_metrics['acc']:.4f} -> {out_dir}/test.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="*", default=[],
                         help="Run directories, each containing config.yaml and checkpoints/{best,last}.pt")
    parser.add_argument("--manifest", default=None,
                         help="Manifest file (one run dir per line, e.g. outputs/results_manifest.txt) to add to run_dirs")
    parser.add_argument("--csv-path", default=None,
                         help="Test on this CSV instead of the run's data.csv_path (requires --out-dir)")
    parser.add_argument("--out-dir", default=None,
                         help="Write the report here instead of into the run directory; with several "
                              "runs, into <out-dir>/<experiment_name>/<date>/<time>")
    parser.add_argument("--filter", action="append", default=[], metavar="COLUMN=VALUE[,VALUE...]",
                         help="Test only on CSV rows whose COLUMN matches one of the values, instead of the "
                              "run's test_split; repeatable, all must match (requires --out-dir)")
    args = parser.parse_args()

    run_dirs = list(args.run_dirs)
    if args.manifest is not None:
        with open(args.manifest) as f:
            run_dirs += [line.strip() for line in f if line.strip()]

    if not run_dirs:
        parser.error("no run directories given — pass some, or --manifest <file>")

    filters = {}
    for f in args.filter:
        column, sep, values = f.partition("=")
        if not sep or not column or not values:
            parser.error(f"--filter expects COLUMN=VALUE[,VALUE...], got '{f}'")
        filters[column] = values

    if (args.csv_path is not None or filters) and args.out_dir is None:
        parser.error("--csv-path/--filter require --out-dir (otherwise the run's own test results would be overwritten)")

    for run_dir in run_dirs:
        out_dir = args.out_dir
        if out_dir is not None and len(run_dirs) > 1:
            run_path = os.path.abspath(run_dir)
            experiment_name = OmegaConf.load(os.path.join(run_path, "config.yaml")).experiment_name
            date_dir, time_dir = os.path.split(run_path)
            out_dir = os.path.join(out_dir, experiment_name, os.path.basename(date_dir), time_dir)
        retest(run_dir, csv_path=args.csv_path, out_dir=out_dir, filters=filters)


if __name__ == "__main__":
    main()
