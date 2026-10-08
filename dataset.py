"""CSV-driven fundus image dataset for a single categorical phenotype.

Expects a CSV with columns: `id`, `filename` (path relative to `images_root`),
`split` (values used to select train vs. test rows), and one column named after
the phenotype (integer class id, e.g. `dr_grade` or `glaucoma_grade`).
"""
import os
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Normalization stats used to train every experiment in this repo (NEL DESP corpus).
IMG_MEAN = (0.4271227, 0.24723781, 0.13661437)
IMG_STD = (0.26101578, 0.16189588, 0.09271526)


def build_train_transform(img_size: int, resize_before_crop: Optional[int] = None,
                           mean=IMG_MEAN, std=IMG_STD):
    """RandomResizedCrop(0.9-1.0) + flips + rotation + blur, matching the original
    `global_crops` augmentation configs used for RFM/DINOv3 head training."""
    steps = []
    if resize_before_crop is not None:
        steps.append(transforms.Resize(resize_before_crop, interpolation=transforms.InterpolationMode.BICUBIC))
    steps += [
        transforms.RandomResizedCrop(
            img_size, ratio=(1.0, 1.0), scale=(0.9, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=7, sigma=(0.3, 0.9))], p=1.0),
        transforms.RandomRotation(degrees=[-45, 45]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(steps)


def build_test_transform(img_size: int, mean=IMG_MEAN, std=IMG_STD, crop_pct: Optional[float] = None):
    """`crop_pct=None` (default, used by dinov3/rfm): resize the shorter edge to
    `img_size`, then center-crop to `img_size`x`img_size`. The crop is a no-op
    for already-square inputs (every dataset here except a couple of stray
    non-square GFID images) - it exists so a source image that isn't exactly
    square still produces a uniform `img_size`x`img_size` tensor, instead of
    silently breaking DataLoader batch collation the one time a differently-
    shaped image lands in the same eval batch as everything else.

    `crop_pct` set (used by RETFound backbones): resize to `round(img_size /
    crop_pct)` then center-crop to `img_size`, matching RETFound's own eval
    transform (`util/datasets.py:build_transform`, `crop_pct = 224/256` for
    input_size<=224)."""
    steps = []
    if crop_pct is None:
        steps.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC))
        steps.append(transforms.CenterCrop(img_size))
    else:
        resize_size = round(img_size / crop_pct)
        steps.append(transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC))
        steps.append(transforms.CenterCrop(img_size))
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(steps)


class CFPDataset(Dataset):
    def __init__(self, csv_path: str, images_root: str, phenotype: str, split_value: str, transform,
                 filters: dict = None):
        """Rows are selected by `split_value` (comma-separated values of the `split`
        column), or, when `filters` ({column: comma-separated values}) is given, by
        those instead: a row is kept only if every filtered column matches one of its
        values. Values are compared as strings."""
        self.df = pd.read_csv(csv_path)
        if not filters:
            filters = {"split": split_value}
        for column, values in filters.items():
            if column not in self.df.columns:
                raise KeyError(f"filter column '{column}' not in {csv_path} (columns: {list(self.df.columns)})")
            self.df = self.df[self.df[column].astype(str).isin(values.split(","))]
        self.df = self.df.reset_index(drop=True)
        self.images_root = images_root
        self.phenotype = phenotype
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.images_root, row["filename"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(int(row[self.phenotype]), dtype=torch.long)
        return image, label, int(row["id"])


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


class ImageDirDataset(Dataset):
    """Every image under `root` (recursively, symlinks followed), in sorted
    relative-path order, for labelless inference (predict.py). Items mirror
    CFPDataset's (image, label, id) shape: label is -1 and id is the image's
    path relative to `root`.

    `preprocess(image, rel_path)`, if given, is applied to each RGB image
    before `transform` (predict.py's `--preprocess auto`).

    With `skip_errors`, an image that fails to load is logged and returned as
    None, for `skip_none_collate` to drop from its batch; otherwise the error
    propagates."""

    def __init__(self, root: str, transform, skip_errors: bool = False, preprocess=None):
        self.root = root
        self.paths = sorted(
            os.path.relpath(os.path.join(dirpath, name), root)
            for dirpath, _, names in os.walk(root, followlinks=True)
            for name in names
            if name.lower().endswith(IMAGE_EXTENSIONS)
        )
        self.transform = transform
        self.skip_errors = skip_errors
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        rel_path = self.paths[idx]
        try:
            image = Image.open(os.path.join(self.root, rel_path)).convert("RGB")
            if self.preprocess is not None:
                image = self.preprocess(image, rel_path)
        except Exception as e:
            if not self.skip_errors:
                raise RuntimeError(f"failed to load {os.path.join(self.root, rel_path)}: {e}") from e
            print(f"[ImageDirDataset] WARNING: skipping {rel_path}: {e}")
            return None
        return self.transform(image), torch.tensor(-1, dtype=torch.long), rel_path


def skip_none_collate(batch):
    """default_collate over the non-None items of `batch`; a batch whose every
    item is None collates to an empty batch (predict() skips those)."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.empty(0), torch.empty(0, dtype=torch.long), []
    return torch.utils.data.default_collate(batch)
