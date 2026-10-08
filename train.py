"""Train a classification head on a frozen ViT-B/16 DINOv3 backbone, then
immediately test the last/best-epoch checkpoint and write results.csv, test.csv
(with bootstrap CIs, matching the original codebase's column layout),
test_images.csv and ROC/PR/confusion-matrix plots into the Hydra run dir.

Usage:
    python train.py --config-name=messidor_rfm
    python train.py --config-name=messidor_dinov3b-9l
"""
import csv
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime

import hydra
import numpy as np
import PIL
import PIL.features
import PIL.Image
import pandas as pd
import sklearn
import timm
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from omegaconf import open_dict
from torch.utils.data import DataLoader

from dataset import CFPDataset, build_test_transform, build_train_transform
from metrics import (compute_metrics, compute_metrics_with_ci, plot_confusion_matrix, plot_pr_curve,
                      plot_roc_curve, resolve_groups)
from model import (ClassificationModel, get_eval_crop_pct_default, get_forward_layer_default,
                   get_normalization_defaults, get_output_mode_default)

# Checkpoints save "score" (best.pt) as a numpy scalar (val_metrics[select_best]
# from sklearn's roc_auc_score etc.), which torch>=2.6's default weights_only=True
# unpickler rejects unless the numpy reconstruction types are explicitly
# allowlisted (np.dtypes.Float64DType is numpy's concrete dtype class for a
# float64 scalar's pickled state). This registers them once for every
# torch.load(weights_only=True) call in this process, including test_only.py
# (which imports this module).
torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType])

# Trades some training throughput for exact run-to-run reproducibility
# (disables cuDNN's runtime algorithm autotuning, which is the main source of
# non-determinism left once the RNGs below are seeded - GPU training is
# otherwise not bit-exact reproducible even with a fixed seed).
DETERMINISTIC = True
# DETERMINISTIC = False


def _nvidia_smi_field(field: str) -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            timeout=10, text=True,
        )
        # One line per GPU; multiple GPUs with different values would matter
        # for reproducibility, so keep them all rather than just the first.
        return "; ".join(line.strip() for line in out.strip().splitlines()) or "unavailable"
    except Exception as e:
        return f"unavailable ({e})"


def environment_info() -> dict:
    """Everything about the software/hardware environment that could plausibly
    affect whether a run is exactly reproducible later - not just library
    versions, but the actual runtime determinism-relevant flags too. Recorded
    once per run (write_environment_info) rather than assumed, after this
    session spent a long time tracing a real divergence back to a code change
    that had nothing to do with the environment - the only way to rule the
    environment in or out next time is to have it on record."""
    os_release = {}
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if "=" in line:
                    k, _, v = line.strip().partition("=")
                    os_release[k] = v.strip('"')
    except OSError:
        pass

    info = {
        "os": os_release.get("PRETTY_NAME", platform.platform()),
        "kernel": platform.release(),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_version": torch.version.cuda or "n/a (CPU build)",
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "n/a",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "timm": timm.__version__,  # forward_intermediates/DropPath internals live here - see model.py's stop_early history
        "scikit-learn": sklearn.__version__,
        # The JPEG decoder changes decoded pixels: the published results need
        # IJG libjpeg 9 (see README.md), not libjpeg-turbo.
        "pillow": PIL.__version__,
        "libjpeg": f"{PIL.Image.core.jpeglib_version} (turbo={PIL.features.check_feature('libjpeg_turbo')})",
        "nvidia_driver_version": _nvidia_smi_field("driver_version"),
        "gpu_name": _nvidia_smi_field("name"),
        "cuda_visible_devices": (os.environ.get("CUDA_VISIBLE_DEVICES", "(unset - all GPUs visible)")
                                 if torch.cuda.is_available() else "n/a (no CUDA device - running on CPU)"),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_flag": DETERMINISTIC,  # train.py's own hardcoded switch, see set_seed()
    }
    return info


def write_environment_info(run_dir: str):
    """Prints (so it's written in train.log alongside everything else) and saves
    environment.txt next to config.yaml, so a run's exact software/hardware
    context is on record rather than needing to be reconstructed after the
    fact."""
    info = environment_info()
    lines = [f"{k}: {v}" for k, v in info.items()]
    print("Environment:")
    for line in lines:
        print(f"  {line}")
    with open(os.path.join(run_dir, "environment.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def _worker_init_fn(worker_id):
    # DataLoader workers each get their own seeded torch RNG automatically,
    # but not random/np.random - both are used by build_train_transform's
    # torchvision augmentations, so without this every worker forks with the
    # same inherited random/np.random state and can apply identical
    # augmentation draws to different images.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_normalization(cfg: DictConfig):
    """`data.augmentations.normalization.mean/std` overrides the per-`model.name`
    default (each backbone's own pretraining stats; see model.get_normalization_defaults)."""
    augmentations = cfg.data.get("augmentations")
    normalization = augmentations.get("normalization") if augmentations is not None else None
    if normalization is not None:
        return tuple(normalization.mean), tuple(normalization.std)
    return get_normalization_defaults(cfg.model.get("name", "dinov3"))


def resolve_eval_crop_pct(cfg: DictConfig):
    """`trainer.eval_crop_pct` overrides the per-`model.name` default (None for
    dinov3/rfm/retfound-green - resize straight to img_size, unchanged from this
    repo's original behaviour; 0.875 for retfound-mae/retfound-dinov2 - resize
    then center-crop, matching RETFound's own eval transform). Applies to both
    the val and test transforms, matching RETFound's own build_transform, which
    uses the same eval branch for both."""
    eval_crop_pct = cfg.trainer.get("eval_crop_pct", "__unset__")
    if eval_crop_pct != "__unset__":
        return eval_crop_pct
    return get_eval_crop_pct_default(cfg.model.get("name", "dinov3"))


def build_dataloaders(cfg: DictConfig, repo_root: str):
    """Build train/val/test dataloaders, resolving relative `csv_path`/`images_root`
    against `repo_root` (needed because Hydra runs with `chdir: true`, so a
    relative path in the config is relative to the repo root, not the run
    directory)."""
    csv_path = cfg.data.csv_path if os.path.isabs(cfg.data.csv_path) else os.path.join(repo_root, cfg.data.csv_path)
    images_root = cfg.data.images_root if os.path.isabs(cfg.data.images_root) else os.path.join(repo_root, cfg.data.images_root)

    mean, std = resolve_normalization(cfg)
    eval_crop_pct = resolve_eval_crop_pct(cfg)
    train_transform = build_train_transform(cfg.model.img_size, cfg.trainer.resize_before_crop, mean=mean, std=std)
    test_transform = build_test_transform(cfg.model.img_size, mean=mean, std=std, crop_pct=eval_crop_pct)

    train_set = CFPDataset(csv_path, images_root, cfg.data.phenotype,
                            cfg.data.train_split, train_transform)
    val_set = CFPDataset(csv_path, images_root, cfg.data.phenotype,
                          cfg.data.val_split, test_transform)
    test_set = CFPDataset(csv_path, images_root, cfg.data.phenotype,
                           cfg.data.test_split, test_transform)

    worker_init_fn = _worker_init_fn if DETERMINISTIC else None
    train_generator = torch.Generator().manual_seed(cfg.seed) if DETERMINISTIC else None
    train_loader = DataLoader(train_set, batch_size=cfg.trainer.batch_size, shuffle=True,
                               num_workers=cfg.trainer.num_workers, pin_memory=True, drop_last=True,
                               worker_init_fn=worker_init_fn, generator=train_generator)
    val_loader = DataLoader(val_set, batch_size=cfg.trainer.eval_batch_size, shuffle=False,
                             num_workers=cfg.trainer.eval_num_workers, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_set, batch_size=cfg.trainer.eval_batch_size, shuffle=False,
                              num_workers=cfg.trainer.eval_num_workers, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    return train_loader, val_loader, test_loader


def retfound_lr_at(step_fraction: float, base_lr: float, warmup_epochs: float,
                    max_epochs: float, min_lr: float) -> float:
    """Linear warmup then half-cycle cosine decay, matching RETFound's own
    official linear-probe schedule (util/lr_sched.py:adjust_learning_rate)
    exactly. `step_fraction` is the continuous epoch position
    (epoch + iter_in_epoch / steps_per_epoch)."""
    if step_fraction < warmup_epochs:
        return base_lr * step_fraction / warmup_epochs
    return min_lr + (base_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * (step_fraction - warmup_epochs) / (max_epochs - warmup_epochs))
    )


def train_epoch(model, loader, optimizer, scheduler, criterion, device,
                 epoch=None, steps_per_epoch=None, lr_schedule_fn=None):
    model.train()
    total_loss = 0.0
    for step, (images, labels, _) in enumerate(loader):
        if lr_schedule_fn is not None:
            lr = lr_schedule_fn(epoch + step / steps_per_epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if lr_schedule_fn is None:
            scheduler.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device, criterion=None):
    """Run `model` over every batch of `loader` and return (y_prob, y_true, ids,
    total_loss). The single forward/softmax code path shared by evaluate() (the
    val/test passes) and predict.py (labelless inference on an image directory),
    so both produce identical probabilities for the same image. `total_loss` is
    the sum of per-sample `criterion` losses, or 0.0 when `criterion` is None.
    Empty batches (every image skipped by ImageDirDataset's skip_errors) are
    ignored."""
    model.eval()
    total_loss = 0.0
    all_probs, all_labels, all_ids = [], [], []
    for images, labels, ids in loader:
        if images.size(0) == 0:
            continue
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(images)
            if criterion is not None:
                loss = criterion(logits, labels)
        if criterion is not None:
            total_loss += loss.item() * images.size(0)
        all_probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_ids.extend(ids.tolist() if torch.is_tensor(ids) else list(ids))

    if not all_probs:
        raise ValueError("no images to predict on (every batch was empty)")
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    return y_prob, y_true, all_ids, total_loss


def evaluate(model, loader, criterion, device, num_classes):
    y_prob, y_true, all_ids, total_loss = predict(model, loader, device, criterion)
    metrics = compute_metrics(y_true, y_prob, num_classes)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, y_true, y_prob, all_ids


def selection_score(val_metrics: dict, train_loss: float, select_best: str) -> float:
    """Higher-is-better score for `trainer.select_best`, the validation metric that
    decides which epoch's weights get tested.

    `"auroc"` matches RETFound's own Nature protocol ("the model weights with the
    highest AUROC on the validation set will be saved"); `"auroc+f1"` matches the
    2025 benchmark preprint ("the checkpoint with the highest sum of AUROC and F1
    score"). `"loss"` is negated so bigger stays better.
    """
    if select_best == "loss":
        return -train_loss
    if select_best == "auroc+f1":
        return val_metrics["auroc"] + val_metrics["f1"]
    if select_best not in val_metrics:
        raise ValueError(f"trainer.select_best={select_best!r} is not one of "
                         f"{sorted(val_metrics)} (or 'auroc+f1'/'loss')")
    return val_metrics[select_best]


def append_to_manifest(manifest_path: str, run_dir: str):
    with open(manifest_path, "a") as f:
        f.write(run_dir + "\n")


def build_model(cfg: DictConfig, repo_root: str, device) -> torch.nn.Module:
    """Build the ClassificationModel from a run's config, resolving a relative
    `pretrained_weights` path against `repo_root` (needed because Hydra runs
    with `chdir: true`, so a relative path in the saved config.yaml is
    relative to the repo root, not to the run directory it's read back
    from)."""
    pretrained_weights = cfg.model.get("pretrained_weights")
    if pretrained_weights is not None and not os.path.isabs(pretrained_weights):
        pretrained_weights = os.path.join(repo_root, pretrained_weights)

    model_name = cfg.model.get("name", "dinov3")
    # output_mode/forward_layer/head_layer_norm are per-model.name constants
    # (model.MODEL_DEFAULTS), not config fields - a config cannot change which
    # published experiment model.name reproduces, even by adding these keys
    # back, since they're never read from cfg here.
    output_mode = get_output_mode_default(model_name)

    print(f"[{cfg.experiment_name}] Backbone: {model_name} <- {pretrained_weights}")

    return ClassificationModel(
        img_size=cfg.model.img_size,
        num_classes=cfg.data.num_classes,
        model_name=model_name,
        output_mode=output_mode,
        drop_path_rate=cfg.model.get("drop_path_rate"),
        pretrained_weights=pretrained_weights,
        head_layer_norm=None,
        forward_layer=get_forward_layer_default(model_name),
    ).to(device)


def backbone_filename(cfg: DictConfig) -> str:
    """Basename of the backbone weights file a run uses (`model.pretrained_weights`).
    Unlike `model.name` - which "rfm" and "dinov3" configs share - it tells the
    backbones apart, and a new weights version (a different filename) no
    longer matches old heads."""
    return os.path.basename(cfg.model.pretrained_weights)


def save_head_checkpoint(path: str, model: torch.nn.Module, cfg: DictConfig, epoch: int, score=None):
    """Save only what training changes: the head (its optional LayerNorm + Linear).
    The backbone is frozen, so it is rebuilt from `weights/` at load time instead
    of being stored; `model_name` and `backbone_file` are recorded so a checkpoint
    can't silently be loaded onto a different backbone (see load_head_checkpoint)."""
    ckpt = {"epoch": epoch, "model_name": cfg.model.get("name", "dinov3"),
            "backbone_file": backbone_filename(cfg), "head": model.head.state_dict()}
    if score is not None:
        ckpt["score"] = float(score)
    torch.save(ckpt, path)


def head_checkpoint_path(run_dir: str, cfg: DictConfig) -> str:
    """The checkpoint a run itself tested: best.pt when the run selected an epoch
    (`trainer.select_best`), else last.pt (also the fallback for older runs
    without best.pt)."""
    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    if not (cfg.trainer.get("select_best") and os.path.isfile(ckpt_path)):
        ckpt_path = os.path.join(run_dir, "checkpoints", "last.pt")
    return ckpt_path


def load_head_checkpoint(path: str, model: torch.nn.Module, cfg: DictConfig, device) -> dict:
    """Load a save_head_checkpoint() file onto `model` (whose backbone
    build_model already loaded from `weights/`), after checking the stored
    backbone name and weights filename against the run's config. Returns the checkpoint dict."""
    # weights_only=True: our own checkpoints hold only tensors and plain numbers
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "head" not in ckpt:
        raise ValueError(f"{path} has no 'head' entry - it is not a head-only checkpoint "
                         f"(a full-model *_full.pt checkpoint cannot be loaded here)")
    expected = cfg.model.get("name", "dinov3")
    if ckpt.get("model_name") != expected:
        raise ValueError(f"{path} was trained on backbone {ckpt.get('model_name')!r}, but the config's "
                         f"model.name is {expected!r}")
    if ckpt.get("backbone_file") != backbone_filename(cfg):
        raise ValueError(f"{path} was trained on backbone weights {ckpt.get('backbone_file')!r}, but the "
                         f"config uses {backbone_filename(cfg)!r} - a different weights file or version "
                         f"(if it is the same file under another name, rename it back)")
    model.head.load_state_dict(ckpt["head"])
    return ckpt


def write_test_images_csv(path: str, ids, y_prob: np.ndarray, class_names, y_true: np.ndarray = None):
    """Per-image predictions: id, true_label, pred_label, prob_<class>... .
    `y_true=None` (predict.py's labelless inference) leaves true_label empty,
    keeping the same columns as a test run's test_images.csv."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "true_label", "pred_label"] + [f"prob_{c}" for c in class_names])
        pred = y_prob.argmax(axis=1)
        for i, sample_id in enumerate(ids):
            true_label = "" if y_true is None else int(y_true[i])
            writer.writerow([sample_id, true_label, int(pred[i])] + list(y_prob[i]))


def write_test_report(model, test_loader, criterion, device, cfg: DictConfig, run_dir: str, epoch: int,
                       repo_root: str) -> dict:
    """Evaluate `model` on `test_loader` and write test.csv (with bootstrap CIs,
    matching the original codebase's column layout), test_images.csv, and the
    ROC/PR/confusion-matrix plots into `run_dir`. Shared by train.py's
    post-training test pass and test_only.py's standalone re-test.

    Returns the plain (non-CI) test metrics dict, e.g. for a summary print.
    """
    test_metrics, y_true, y_prob, ids = evaluate(model, test_loader, criterion, device, cfg.data.num_classes)

    # Patient-level bootstrap groups for datasets where one patient contributes
    # more than one correlated test image (PAPILA, Messidor-2); None (row-level)
    # for everything else. See metrics.resolve_groups.
    csv_path = cfg.data.csv_path if os.path.isabs(cfg.data.csv_path) else os.path.join(repo_root, cfg.data.csv_path)
    groups = resolve_groups(csv_path, repo_root, ids)

    # Full report with bootstrap CIs, matching the original test.csv column layout:
    # epoch, timestamp, then every "<phenotype>_<metric>[_ci_lower/_ci_upper/_std]"
    # key sorted alphabetically (acc, auprc, auroc, balanced_acc, epoch, f1, loss, mae,
    # precision, recall).
    ci_metrics = compute_metrics_with_ci(y_true, y_prob, cfg.data.num_classes, groups=groups)
    ci_metrics["loss"] = test_metrics["loss"]
    ci_metrics["epoch"] = epoch
    prefixed_metrics = {f"{cfg.data.phenotype}_{k}": v for k, v in ci_metrics.items()}

    row = {"epoch": epoch, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    row.update(prefixed_metrics)
    fieldnames = ["epoch", "timestamp"] + sorted(prefixed_metrics.keys())
    with open(os.path.join(run_dir, "test.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    write_test_images_csv(os.path.join(run_dir, "test_images.csv"), ids, y_prob, cfg.data.class_names, y_true)

    plot_roc_curve(y_true, y_prob, cfg.data.num_classes, cfg.data.class_names,
                    os.path.join(run_dir, f"{cfg.data.phenotype}_roc_curve.png"),
                    ci_band=cfg.data.get("plot_auroc_ci_band", True), groups=groups)
    plot_pr_curve(y_true, y_prob, cfg.data.num_classes, cfg.data.class_names,
                  os.path.join(run_dir, f"{cfg.data.phenotype}_pr_curve.png"),
                  ci_band=cfg.data.get("plot_auprc_ci_band", True), groups=groups)
    plot_confusion_matrix(y_true, y_prob, cfg.data.num_classes, cfg.data.class_names,
                           os.path.join(run_dir, f"{cfg.data.phenotype}_confusion_matrix.png"))

    return test_metrics


@hydra.main(config_path="configs/experiment", config_name=None, version_base="1.3")
def main(cfg: DictConfig):

    if "seed" not in cfg or cfg.seed is None:
        with open_dict(cfg):
            # cfg.seed = 42
            # cfg.seed = 123
            # cfg.seed = 10
            cfg.seed = 42

    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)
    original_cwd = hydra.utils.get_original_cwd()
    run_dir = os.getcwd()  # Hydra chdir's into the run dir by default
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))
    write_environment_info(run_dir)

    train_loader, val_loader, test_loader = build_dataloaders(cfg, original_cwd)
    model = build_model(cfg, original_cwd, device)

    # `trainer.blr` (a RETFound-style reference LR at batch size 256) overrides
    # `trainer.base_lr` when present, scaled by the actual batch size.
    blr = cfg.trainer.get("blr")
    base_lr = blr * cfg.trainer.batch_size / 256 if blr is not None else cfg.trainer.base_lr

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr, weight_decay=cfg.trainer.weight_decay,
    )

    # `trainer.warmup_epochs` opts into RETFound's own linear-warmup + half-cycle
    # cosine schedule (computed per-step in train_epoch via lr_schedule_fn);
    # absent, this falls back to the original flat-base-lr CosineAnnealingLR.
    warmup_epochs = cfg.trainer.get("warmup_epochs", 0)
    scheduler = None
    lr_schedule_fn = None
    if warmup_epochs > 0:
        min_lr = cfg.trainer.get("min_lr", 0.0)
        lr_schedule_fn = lambda step_fraction: retfound_lr_at(
            step_fraction, base_lr, warmup_epochs, cfg.trainer.max_epochs, min_lr
        )
        print(f"[{cfg.experiment_name}] LR schedule: warmup_epochs={warmup_epochs} "
              f"base_lr={base_lr:.2e} min_lr={min_lr:.2e}")
    else:
        total_steps = cfg.trainer.max_epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = torch.nn.CrossEntropyLoss()

    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    results_csv = os.path.join(run_dir, "results.csv")
    fieldnames = None

    # `trainer.select_best` (a validation metric name, e.g. "auroc") tests the
    # best-scoring epoch's weights instead of the final ones, matching RETFound's
    # own protocol. Absent/null keeps this repo's original last-epoch behaviour.
    select_best = cfg.trainer.get("select_best")
    best_score, best_epoch = None, None
    if select_best:
        print(f"[{cfg.experiment_name}] checkpoint selection: best val {select_best} "
              f"(checkpoints/best.pt); without it the final epoch would be tested")

    # `trainer.terminate_after_epochs` (optional, pass as
    # +trainer.terminate_after_epochs=N) stops training after N epochs while
    # leaving max_epochs - and therefore the LR schedule, which is laid out over
    # max_epochs - untouched, so those N epochs are bit-identical to the first
    # N of a full run. Lowering trainer.max_epochs instead would compress the
    # schedule and change every epoch. The test pass below still runs.
    terminate_after = cfg.trainer.get("terminate_after_epochs")

    start = time.time()
    for epoch in range(cfg.trainer.max_epochs):
        if terminate_after is not None and epoch >= terminate_after:
            print(f"[{cfg.experiment_name}] terminating after {terminate_after} epoch(s) "
                  f"(trainer.terminate_after_epochs; LR schedule still spans max_epochs={cfg.trainer.max_epochs})")
            break
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device,
                                  epoch=epoch, steps_per_epoch=len(train_loader), lr_schedule_fn=lr_schedule_fn)
        val_metrics, *_ = evaluate(model, val_loader, criterion, device, cfg.data.num_classes)

        row = {"epoch": epoch, "train_loss": train_loss}
        row.update({f"{cfg.data.phenotype}_{k}": v for k, v in val_metrics.items()})
        if fieldnames is None:
            fieldnames = list(row.keys())
            with open(results_csv, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        with open(results_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        save_head_checkpoint(os.path.join(checkpoints_dir, "last.pt"), model, cfg, epoch)

        if select_best:
            score = selection_score(val_metrics, train_loss, select_best)
            if best_score is None or score > best_score:
                best_score, best_epoch = score, epoch
                save_head_checkpoint(os.path.join(checkpoints_dir, "best.pt"), model, cfg, epoch, score)

        elapsed = time.time() - start
        print(f"[{cfg.experiment_name}] epoch {epoch + 1}/{cfg.trainer.max_epochs} "
              f"train_loss={train_loss:.4f} val_auroc={val_metrics['auroc']:.4f} "
              f"val_acc={val_metrics['acc']:.4f} elapsed={elapsed / 60:.1f}min")

    # Final test pass immediately after training, on whichever checkpoint
    # `trainer.select_best` chose (the last epoch's by default).
    ckpt_name = "best.pt" if select_best else "last.pt"
    ckpt = load_head_checkpoint(os.path.join(checkpoints_dir, ckpt_name), model, cfg, device)
    if select_best:
        print(f"[{cfg.experiment_name}] testing epoch {ckpt['epoch'] + 1}/{cfg.trainer.max_epochs} "
              f"(best val {select_best}={best_score:.4f})")
    test_metrics = write_test_report(model, test_loader, criterion, device, cfg, run_dir, ckpt["epoch"],
                                      repo_root=original_cwd)

    manifest_path = os.path.join(original_cwd, cfg.results_manifest)
    append_to_manifest(manifest_path, run_dir)
    print(f"Done. Test AUROC={test_metrics['auroc']:.4f}. Run dir appended to {manifest_path}")


if __name__ == "__main__":
    main()
