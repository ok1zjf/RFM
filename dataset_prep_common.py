"""Shared helpers for the convert_<dataset>.py scripts.

Each convert script maps rows of a committed datasets/<name>.csv onto a raw
download under datasets/raw/<name>/, and writes resized/re-encoded images to
datasets/<name>/... at the exact relative path the CSV's `filename` column
already specifies.
"""
import io
from pathlib import Path

from PIL import Image

JPEG_QUALITY = 95

# Per-dataset preparation, keyed by the dataset's directory under datasets/
# (each config's `data.images_root`): resize_image() arguments, plus the save
# format where it is fixed rather than taken from the filename's extension.
# The single source for both the convert_<dataset>.py scripts and
# predict.py's `--preprocess auto`.
RESIZE_PARAMS = {
    "aptos-2019": {"short_side": 512, "square": "crop"},
    "idrid": {"short_side": 512, "square": "crop"},
    "messidor2": {"short_side": 512, "square": False},
    "papila": {"short_side": 512, "square": "pad"},
    "gfid": {"short_side": None, "square": False},
}
SAVE_FORMAT = {"gfid": "PNG"}


def save_format(dataset, filename):
    """The format convert_<dataset>.py saves `filename` in: fixed per dataset
    (SAVE_FORMAT), else by the filename's extension."""
    return SAVE_FORMAT.get(dataset) or fmt_for_suffix(filename)


def resize_image(img, short_side=512, square=False):
    """Resize an RGB PIL image.

    short_side=None returns it unchanged.
    square="pad" fits the long side to `short_side` and pads the short side
    with black to a `short_side` x `short_side` canvas (matches PAPILA's
    images-512-sq convention: the fundus circle is fully visible, with black
    letterbox bars). square="crop" (also square=True, as an alias) scales the
    short side to `short_side` and center-crops the long side down to
    `short_side` (matches the APTOS-2019-512/IDIRD-retfound convention: the
    fundus circle fills the frame, trimming its left/right edges). square=False
    just scales the short side to `short_side`, preserving aspect ratio.
    """
    if square is True:
        square = "crop"
    if short_side is None:
        return img
    w, h = img.size
    if square == "pad":
        scale = short_side / max(w, h)
        rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((rw, rh), Image.LANCZOS)
        canvas = Image.new("RGB", (short_side, short_side), (0, 0, 0))
        canvas.paste(img, ((short_side - rw) // 2, (short_side - rh) // 2))
        return canvas
    if square == "crop":
        scale = short_side / min(w, h)
        rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((rw, rh), Image.LANCZOS)
        left, top = (rw - short_side) // 2, (rh - short_side) // 2
        return img.crop((left, top, left + short_side, top + short_side))
    if h < w:
        new_h, new_w = short_side, round(w * short_side / h)
    else:
        new_w, new_h = short_side, round(h * short_side / w)
    return img.resize((new_w, new_h), Image.LANCZOS)


def resize_reencode(src, dst, short_side=512, quality=JPEG_QUALITY, square=False, fmt="JPEG"):
    """Convert `src` to RGB, resize it (see resize_image) and save to `dst` as
    `fmt`, creating parent dirs as needed."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = resize_image(img.convert("RGB"), short_side, square)
        save_kwargs = {"quality": quality} if fmt == "JPEG" else {}
        img.save(dst, fmt, **save_kwargs)


def prepare_image(img, filename, dataset):
    """convert_<dataset>.py's steps, in memory, for one RGB image: the
    dataset's resize, then a round trip through its save format (JPEG at
    JPEG_QUALITY; PNG is lossless, so nothing to do). Used by predict.py's
    `--preprocess auto` on raw images. It runs on the caller's Pillow (the
    environment's IJG libjpeg build), while the converters use the
    libjpeg-turbo build (turbo_pillow.py), so a JPEG result is close to, but
    not bit-identical with, the converted image."""
    img = resize_image(img, **RESIZE_PARAMS[dataset])
    if save_format(dataset, filename) == "JPEG":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=JPEG_QUALITY)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def fmt_for_suffix(filename):
    """Pick a save format matching the CSV's own filename extension."""
    return "PNG" if str(filename).lower().endswith(".png") else "JPEG"


def iter_progress(iterable, total=None, label=""):
    """Yield from `iterable`, printing periodic progress (~20 updates total)."""
    try:
        from tqdm import tqdm
        yield from tqdm(iterable, total=total, desc=label)
        return
    except ImportError:
        pass

    if total is None and hasattr(iterable, "__len__"):
        total = len(iterable)
    step = max(1, (total or 0) // 20)
    for i, item in enumerate(iterable, 1):
        if total:
            if i == 1 or i % step == 0 or i == total:
                print(f"[{label}] {i}/{total}", flush=True)
        elif i % 100 == 0:
            print(f"[{label}] {i}", flush=True)
        yield item


def stem_lookup(directory, exts=(".jpg", ".jpeg", ".png")):
    """Case-insensitive stem -> path map for every matching file under `directory`."""
    lookup = {}
    for p in Path(directory).rglob("*"):
        if p.suffix.lower() in exts:
            lookup[p.stem.lower()] = p
    return lookup
