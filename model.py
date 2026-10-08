"""ViT backbone(s) + single-phenotype classification head.

This repository reproduces a fixed set of published experiments, one per
`model.name`, so every hyperparameter that identifies *which* experiment a
model name is - feature-extraction mode, intermediate-layer index, whether
the head gets its own LayerNorm, stochastic depth - is a property of the
name, resolved here in code (MODEL_DEFAULTS below), not a config field. A
config only ever supplies img_size, pretrained_weights, and
num_classes/phenotype - nothing that could silently change which published
result a run reproduces.

`model.name` selects the backbone and, with it, every default below:
  - dinov3             : ViT-B/16 DINOv3 (timm), plain LVD-1689M-pretrained
                          weights (`pretrained_weights`), cls_global_pool_9l
                          feature extraction (see below) - the *_dinov3b-9l
                          experiments, DINOv3b-9L in the paper.
  - rfm                 : same architecture as dinov3, loading an SSL
                          training checkpoint (`pretrained_weights`)
                          instead, same cls_global_pool_9l extraction.
  - dinov3-global-pool   : same architecture and weights as dinov3, but
                          global_pool feature extraction instead - a
                          distinct name (not just a different `output`)
                          because output_mode is no longer a config field.
                          The *_dinov3 experiments, DINOv3 in the paper.
  - retfound-mae        : RETFound ViT-L/16, MAE-pretrained on CFP (Nature).
  - retfound-dinov2     : RETFound ViT-L/14, DINOv2/iBOT-pretrained on CFP.
  - retfound-green      : RETFound-Green ViT-S/14 (reg4), DINOv2-pretrained.

Each name's own feature-extraction mode (internally, `output_mode`) selects
how the backbone's tokens become the head's input feature vector:
  - null (default)      : CLS token from the backbone's final block, after
                          the backbone's own (pretrained) final LayerNorm.
                          This is RETFound-DINOv2's own official linear-probe
                          recipe (models_vit.RETFound_dinov2 never forwards
                          `--global_pool` to timm, so it always uses timm's
                          default CLS-token pooling).

  - global_pool         : mean-pooled RAW patch tokens from the final block
                          (the backbone's final LayerNorm is skipped - this is
                          how timm builds a `global_pool='avg'` ViT: `norm`
                          becomes Identity and `fc_norm` - untrained unless the
                          checkpoint happens to cover it - normalizes the
                          pooled feature instead). This is RETFound-MAE's own
                          official linear-probe recipe (models_vit.VisionTransformer
                          .forward_features takes exactly this path when
                          `global_pool=True`).

  - global_pool_normed  : mean-pooled patch tokens from the final block,
                          taken AFTER the backbone's own (pretrained) final
                          LayerNorm (applied to every token, then patch tokens
                          are averaged, CLS/register tokens excluded), optionally
                          with an extra LayerNorm on the pooled result (the
                          per-name head_layer_norm default). This is RETFound-Green's
                          own official recipe: its usage snippet creates the timm
                          model with the default token pooling - leaving `norm` as
                          the real pretrained LayerNorm and `fc_norm` as Identity -
                          and only then sets `global_pool = "avg"` on the built
                          model, which changes the pooling without moving the norm
                          (verified bit-identical here, max abs diff 0.0). Note
                          that building with `global_pool="avg"` instead, as
                          `global_pool` above does, is NOT equivalent for this
                          checkpoint: timm would swap the pretrained `norm` for
                          Identity and normalize with an untrained `fc_norm`
                          (cosine 0.82 against the official feature). For
                          MAE/dinov2 this mode is not their linear-probe recipe
                          at all; there it matches only RETFound's separate
                          reference *feature-extraction* notebook
                          (latent_feature.ipynb / retfound_test.py), used for
                          embedding visualization rather than training.

  - cls_global_pool_9l  : CLS token concatenated with mean-pooled patch
                          tokens from intermediate block `forward_layer`
                          (0-based) - 8 for dinov3/rfm, the output of the 9th
                          block, hence "9l". Doubles the feature dimension.
                          Reads an intermediate block and never passes through
                          the final norm; forward_layer is ignored by every
                          other output mode, which all read the final block.

"""
import math
import os
from functools import partial
from typing import Optional

import timm
import timm.models.vision_transformer as _tvit
import torch
import torch.nn as nn

DINOV3_MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
RETFOUND_TIMM_NAMES = {
    "retfound-dinov2": "vit_large_patch14_dinov2.lvd142m",
    "retfound-green": "vit_small_patch14_reg4_dinov2",
}
FORWARD_LAYER = 8  # 0-based block read by cls_global_pool_9l: the output of the 9th block ("9l")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HALF_MEAN = (0.5, 0.5, 0.5)
HALF_STD = (0.5, 0.5, 0.5)
CRI10_MEAN = (0.4271227, 0.24723781, 0.13661437)
CRI10_STD = (0.26101578, 0.16189588, 0.09271526)

# Per `model.name` default normalization stats (matching each backbone's own
# pretraining), default head LayerNorm setting (on unless the backbone's own
# final norm already normalizes the pooled feature), default eval-time
# resize/crop behaviour (`eval_crop_pct=None` resizes straight to img_size, no
# crop margin - this repo's original dinov3/rfm behaviour; a float value
# resizes to img_size/eval_crop_pct then center-crops to img_size, matching
# RETFound's own eval transform for its MAE/dinov2 backbones - see
# util/datasets.py:build_transform, crop_pct=224/256 for input_size<=224),
# stochastic depth, and feature-extraction mode/layer. None of these are
# config fields - each published experiment is fully identified by
# model.name alone, so there is nothing here a config could override even by
# accident. Only `data.augmentations.normalization` / `trainer.eval_crop_pct`
# remain overridable, for genuinely per-run infrastructure reasons (a
# non-standard input source, a different eval-time GPU memory budget).
MODEL_DEFAULTS = {
    "dinov3": {"mean": CRI10_MEAN, "std": CRI10_STD, "head_layer_norm": True, "eval_crop_pct": None,
               "drop_path_rate": 0.2, "output_mode": "cls_global_pool_9l", "forward_layer": FORWARD_LAYER},
    "rfm": {"mean": CRI10_MEAN, "std": CRI10_STD, "head_layer_norm": True, "eval_crop_pct": None,
            "drop_path_rate": 0.2, "output_mode": "cls_global_pool_9l", "forward_layer": FORWARD_LAYER},
    # Same architecture/weights as "dinov3" (see Backbone.__init__'s dispatch) but a
    # different feature-extraction mode - a distinct name rather than a config
    # override of "dinov3", since output_mode is no longer config-settable at all.
    "dinov3-global-pool": {"mean": CRI10_MEAN, "std": CRI10_STD, "head_layer_norm": True, "eval_crop_pct": None,
                            "drop_path_rate": 0.2, "output_mode": "global_pool", "forward_layer": None},
    "retfound-mae": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD, "head_layer_norm": False, "eval_crop_pct": 0.875,
                      "drop_path_rate": 0.2, "output_mode": "global_pool", "forward_layer": None},
    "retfound-dinov2": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD, "head_layer_norm": False, "eval_crop_pct": 0.875,
                         "drop_path_rate": 0.2, "output_mode": None, "forward_layer": None},
    # retfound-green's drop_path_rate here is never actually read: _build_retfound_backbone
    # forces 0.0 for it unconditionally (not configurable), since its official probe
    # precomputes embeddings under inference_mode and never trains through the backbone.
    "retfound-green": {"mean": HALF_MEAN, "std": HALF_STD, "head_layer_norm": False, "eval_crop_pct": None,
                        "drop_path_rate": 0.0, "output_mode": "global_pool_normed", "forward_layer": None},
}


def get_normalization_defaults(model_name: str):
    defaults = MODEL_DEFAULTS.get(model_name, MODEL_DEFAULTS["dinov3"])
    return defaults["mean"], defaults["std"]


def get_head_layer_norm_default(model_name: str) -> bool:
    return MODEL_DEFAULTS.get(model_name, MODEL_DEFAULTS["dinov3"])["head_layer_norm"]


def get_eval_crop_pct_default(model_name: str):
    return MODEL_DEFAULTS.get(model_name, MODEL_DEFAULTS["dinov3"])["eval_crop_pct"]


def get_drop_path_rate_default(model_name: str) -> float:
    return MODEL_DEFAULTS.get(model_name, MODEL_DEFAULTS["dinov3"])["drop_path_rate"]


def get_output_mode_default(model_name: str) -> Optional[str]:
    return MODEL_DEFAULTS.get(model_name, MODEL_DEFAULTS["dinov3"])["output_mode"]


def get_forward_layer_default(model_name: str) -> Optional[int]:
    return MODEL_DEFAULTS.get(model_name, MODEL_DEFAULTS["dinov3"])["forward_layer"]


def _build_dinov3_backbone(img_size: int, drop_path_rate: float, timm_global_pool: str):
    """ViT-B/16 DINOv3 architecture, untrained - the caller (ClassificationModel)
    loads real weights afterwards via `load_rfm_checkpoint`, for every
    model.name that uses this architecture ("dinov3", "dinov3-global-pool",
    "rfm" alike)."""
    backbone = timm.create_model(
        DINOV3_MODEL_NAME,
        pretrained=False,
        num_classes=0,
        img_size=img_size,
        drop_path_rate=drop_path_rate,
        global_pool=timm_global_pool,
    )
    _build_rope_coords_on_buffer_device(backbone.rope)
    return backbone


def _create_rope_embed_on_buffer_device(self, feat_shape, no_aug: bool = False) -> torch.Tensor:
    """timm 1.0.24's RotaryEmbeddingDinoV3._create_embed, except that the patch
    coordinate grid is built on the device of the module's own buffers (the
    GPU, once the model has been moved there) rather than timm's default CPU.

    The paper's results were produced with timm patched exactly this way, and
    the coordinates round differently on GPU vs CPU, so the RoPE sin/cos tables
    (and with them every dinov3/rfm feature) differ in the last bits otherwise.
    Keeping this makes stock timm 1.0.24 reproduce the published numbers
    bit-exactly."""
    from timm.layers.pos_embed_sincos import make_coords_dinov3

    H, W = feat_shape
    coords = make_coords_dinov3(
        H, W,
        normalize_coords=self.normalize_coords,
        grid_indexing=self.grid_indexing,
        grid_offset=self.grid_offset,
        device=next(self.buffers()).device,
    )  # (HW, 2)
    if not no_aug:
        coords = self._apply_coord_augs(coords)
    sin, cos = self._get_pos_embed_from_coords(coords)  # 2 * (HW, dim)
    return torch.cat([sin, cos], dim=-1)  # (HW, 2*dim)


def _build_rope_coords_on_buffer_device(rope: nn.Module):
    """Swaps in `_create_rope_embed_on_buffer_device` on this backbone's RoPE
    module. It replaces a private timm method, so it's tied to the pinned timm
    version (requirements.txt) and fails loudly on any other."""
    from timm.layers.pos_embed_sincos import RotaryEmbeddingDinoV3

    if timm.__version__ != "1.0.24" or not isinstance(rope, RotaryEmbeddingDinoV3):
        raise RuntimeError(f"RoPE patch written for timm 1.0.24's RotaryEmbeddingDinoV3, got timm "
                           f"{timm.__version__} / {type(rope).__name__} - re-check it against that "
                           f"version's _create_embed before relaxing this")
    rope._create_embed = _create_rope_embed_on_buffer_device.__get__(rope)


def _build_retfound_backbone(model_name: str, img_size: int, drop_path_rate: float,
                              timm_global_pool: str):
    """RETFound backbones always load their real weights from `pretrained_weights`
    afterwards (via `load_rfm_checkpoint`, strict=False), so are always built
    untrained here."""
    if model_name == "retfound-green":
        # Not configurable: RETFound-Green's official probe precomputes embeddings
        # under inference_mode and never trains through the backbone, so it uses no
        # stochastic depth regardless of model.drop_path_rate/the default below.
        drop_path_rate = 0.0

    if model_name == "retfound-mae":
        # Built directly (not via timm.create_model) to match RETFound's exact
        # ViT-Large/16 configuration.
        return _tvit.VisionTransformer(
            img_size=img_size,
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
            global_pool=timm_global_pool,
            drop_path_rate=drop_path_rate,
        )

    return timm.create_model(
        RETFOUND_TIMM_NAMES[model_name],
        pretrained=False,
        img_size=img_size,
        num_classes=0,
        global_pool=timm_global_pool,
        drop_path_rate=drop_path_rate,
    )


class Backbone(nn.Module):
    """Wraps a timm-style ViT backbone and returns a head-ready feature vector.

    The inner submodule is deliberately named `self.backbone` (not e.g.
    `self.net`): `load_rfm_checkpoint` is called on this wrapper instance, and
    every supported checkpoint format (RFM, RETFound MAE/dinov2/green) either
    already uses a `backbone.*` key prefix or gets remapped to one by that
    function's prefix-matching heuristic.
    """

    def __init__(self, model_name: str, img_size: int, output_mode: Optional[str],
                 drop_path_rate: Optional[float] = None, forward_layer: Optional[int] = FORWARD_LAYER):
        super().__init__()
        self.output_mode = output_mode
        timm_global_pool = "avg" if output_mode == "global_pool" else "token"
        if drop_path_rate is None:
            drop_path_rate = get_drop_path_rate_default(model_name)

        # Every backbone is built untrained; ClassificationModel loads the real
        # weights (`pretrained_weights`) afterwards via load_rfm_checkpoint,
        # onto this wrapper's `self.backbone` attribute.
        if model_name in ("dinov3", "dinov3-global-pool", "rfm"):
            self.backbone = _build_dinov3_backbone(img_size, drop_path_rate, timm_global_pool)
        elif model_name in ("retfound-mae", "retfound-dinov2", "retfound-green"):
            self.backbone = _build_retfound_backbone(model_name, img_size, drop_path_rate, timm_global_pool)
        else:
            raise ValueError(f"Unknown model.name: {model_name!r}")

        self.embed_dim = self.backbone.embed_dim
        self.feature_dim = self.embed_dim * (2 if output_mode == "cls_global_pool_9l" else 1)
        depth = len(self.backbone.blocks)
        if output_mode == "cls_global_pool_9l" and not (forward_layer is not None and -depth <= forward_layer < depth):
            raise ValueError(f"model.forward_layer={forward_layer} is outside this backbone's "
                             f"{depth} blocks (0..{depth - 1})")
        self.forward_layer = forward_layer
        # stop_early (skip blocks past forward_layer) is bit-identical only when no
        # DropPath draws get skipped along with them - i.e. only when this backbone
        # has no active stochastic depth. Decided once here, from the resolved
        # drop_path_rate, so a future config override that turns drop_path_rate back
        # on for dinov3/rfm automatically disables the optimization instead of
        # silently desyncing the RNG stream again (see forward()).
        self._stop_early_safe = (drop_path_rate == 0.0)

    def forward(self, x):
        """Returns the head-ready feature vector: [B, feature_dim]."""
        if self.output_mode == "cls_global_pool_9l":
            nlc_intermediates = self.backbone.forward_intermediates(
                x,
                intermediates_only=True,
                return_prefix_tokens=True,
                indices=[self.forward_layer],
                output_fmt="NLC",
                # timm runs every block unless told otherwise, discarding the outputs
                # past `forward_layer`; skipping them is ~24% less GPU work per forward
                # pass (12 blocks -> 9 at the default layer 8). Safe to skip ONLY when
                # this backbone has no active DropPath (self._stop_early_safe, decided
                # in __init__ from drop_path_rate): DropPath draws from the RNG once per
                # block whenever the (frozen) backbone is in .train() mode (DropPath
                # activation depends on .training, not requires_grad), so skipping
                # blocks with active stochastic depth would skip their RNG draws and
                # desynchronize the shared stream for the rest of the run - verified
                # directly (same seed, same input: RNG state after this call differs
                # between stop_early=True/False whenever drop_path_rate > 0, and is
                # identical whenever it's 0.0).
                stop_early=self._stop_early_safe,
            )
            patch_tokens, prefix_tokens = nlc_intermediates[-1]
            cls_token = prefix_tokens[:, 0, :]
            pooled_patches = torch.mean(patch_tokens, dim=1)
            return torch.cat([cls_token, pooled_patches], dim=-1).float()
        if self.output_mode == "global_pool_normed":
            # Backbone was built with `global_pool="token"`, so `self.backbone.norm`
            # is the real (pretrained) final LayerNorm, applied here to every
            # token; patch tokens are then mean-pooled manually (CLS/register
            # prefix tokens excluded), skipping the CLS-only pooling that
            # "token" construction would otherwise use.
            all_tokens = self.backbone.forward_features(x)
            num_prefix = self.backbone.num_prefix_tokens
            return torch.mean(all_tokens[:, num_prefix:, :], dim=1).float()
        return self.backbone(x).float()


def load_rfm_checkpoint(backbone: nn.Module, checkpoint_path: str):
    """Load a backbone checkpoint onto `backbone`. Handles every checkpoint
    format used in this repo:
      - RFM/SSL training checkpoints (key "model_state_dict"/"model"/"teacher"/
        "student", possibly DDP/torch.compile-prefixed).
      - RETFound MAE (key "model", unprefixed keys).
      - RETFound dinov2 (key "teacher", "backbone.*"-prefixed keys).
      - RETFound-Green (raw state dict, unprefixed keys).
    Also bicubically interpolates `pos_embed` when the checkpoint's patch grid
    doesn't match the model's (e.g. different img_size at pretraining time).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "model", "teacher", "student"):
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    else:
        state_dict = checkpoint

    state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    backbone_params = set(backbone.state_dict().keys())
    direct_matches = sum(1 for k in state_dict if k in backbone_params)
    if direct_matches < len(backbone_params) * 0.5:
        from collections import Counter

        prefixes = Counter(k.split(".")[0] for k in backbone_params if "." in k)
        for prefix, _ in prefixes.most_common(3):
            remapped = {f"{prefix}.{k}": v for k, v in state_dict.items()}
            if sum(1 for k in remapped if k in backbone_params) > direct_matches:
                state_dict = remapped
                break

    model_sd = backbone.state_dict()
    for pe_key in [k for k in state_dict if k.endswith("pos_embed")]:
        if pe_key not in model_sd:
            continue
        ckpt_pe, model_pe = state_dict[pe_key], model_sd[pe_key]
        if ckpt_pe.shape == model_pe.shape:
            continue
        n_patch_ckpt, n_patch_model = ckpt_pe.shape[1] - 1, model_pe.shape[1] - 1
        d = ckpt_pe.shape[2]
        h_c = w_c = int(math.isqrt(n_patch_ckpt))
        h_m = w_m = int(math.isqrt(n_patch_model))
        if h_c * w_c != n_patch_ckpt or h_m * w_m != n_patch_model:
            del state_dict[pe_key]
            continue
        cls_pos = ckpt_pe[:, :1, :]
        patch_pos = ckpt_pe[:, 1:, :].reshape(1, h_c, w_c, d).permute(0, 3, 1, 2).float()
        patch_pos_interp = torch.nn.functional.interpolate(
            patch_pos, size=(h_m, w_m), mode="bicubic", align_corners=False
        ).permute(0, 2, 3, 1).reshape(1, n_patch_model, d).to(ckpt_pe.dtype)
        state_dict[pe_key] = torch.cat([cls_pos, patch_pos_interp], dim=1)

    result = backbone.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        # strict=False silently keeps going on a partial-coverage checkpoint,
        # which is sometimes intentional (e.g. rfm-1.3.pt is deliberately
        # truncated to blocks 0-8, without the final norm, for its block-8 readout) but can
        # also silently leave part of the backbone at its pre-checkpoint init
        # for an output mode that reads further than the checkpoint covers -
        # exactly the bug that motivated this warning. Never suppress it.
        from collections import Counter

        def _summarize(keys):
            # "blocks.N....." collapses to "blocks.N" (which block, not just how many
            # keys) - a lot more actionable than a bare count when the gap is a whole
            # missing block, as with rfm-1.3.pt's blocks 9-11.
            def group(k):
                parts = k.split(".")
                if len(parts) >= 3 and parts[1] == "blocks":
                    return f"{parts[0]}.blocks.{parts[2]}"
                return ".".join(parts[:2])

            counts = Counter(group(k) for k in keys)
            return ", ".join(f"{p} x{n}" for p, n in sorted(counts.items()))

        print(f"[load_rfm_checkpoint] WARNING: {checkpoint_path!r} does not fully match this "
              f"backbone's state dict.")
        if result.missing_keys:
            print(f"  {len(result.missing_keys)} backbone params NOT found in the checkpoint "
                  f"(kept at their pre-checkpoint init): {_summarize(result.missing_keys)}")
        if result.unexpected_keys:
            print(f"  {len(result.unexpected_keys)} checkpoint keys not used by this backbone: "
                  f"{_summarize(result.unexpected_keys)}")


class Head(nn.Module):
    """Optional LayerNorm + single Linear layer over the pooled feature."""

    def __init__(self, in_dim: int, num_classes: int, layer_norm: bool = True):
        super().__init__()
        layers = []
        if layer_norm:
            layers.append(nn.LayerNorm(in_dim))
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ClassificationModel(nn.Module):
    """Frozen ViT backbone + trainable classification head."""

    def __init__(self, img_size: int, num_classes: int, model_name: str = "dinov3",
                 output_mode: Optional[str] = None, drop_path_rate: Optional[float] = None,
                 pretrained_weights: Optional[str] = None,
                 head_layer_norm: Optional[bool] = None, forward_layer: Optional[int] = FORWARD_LAYER):
        """
        `pretrained_weights` is the single, required source of backbone
        weights for every model.name - a plain LVD-1689M state dict for
        "dinov3"/"dinov3-global-pool", an SSL/finetune checkpoint (RFM,
        RETFound) for the rest. Both are loaded the same way, via
        `load_rfm_checkpoint`'s flexible key-format handling (which also
        covers the plain-state-dict case: no wrapper key matches, so it falls
        back to the raw dict, then prefix-remaps it onto this wrapper's own
        `backbone.*`-prefixed keys).
        """
        super().__init__()
        self.backbone = Backbone(
            model_name, img_size, output_mode, drop_path_rate=drop_path_rate, forward_layer=forward_layer,
        )
        if pretrained_weights is None:
            raise ValueError("pretrained_weights is required (no model.name builds an untrained backbone)")
        load_rfm_checkpoint(self.backbone, pretrained_weights)
        for p in self.backbone.parameters():
            p.requires_grad = False

        if head_layer_norm is None:
            head_layer_norm = get_head_layer_norm_default(model_name)
        self.head = Head(in_dim=self.backbone.feature_dim, num_classes=num_classes, layer_norm=head_layer_norm)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.head(features)
