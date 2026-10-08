# RETFound baselines

The RETFound baselines are frozen-backbone linear probes on three
checkpoints, trained and tested on the same five datasets and splits as RFM.
This document lists where each one follows its official linear-probe code and
where it differs.

| `model.name`      | Backbone                                | Checkpoint | Configs |
|-------------------|-----------------------------------------|------------|---------|
| `retfound-mae`    | ViT-L/16, MAE                           | [`RETFound_mae_natureCFP`](https://huggingface.co/YukunZhou/RETFound_mae_natureCFP) | `*_retfound_mae.yaml` |
| `retfound-dinov2` | ViT-L/14, DINOv2                        | [`RETFound_dinov2_meh`](https://huggingface.co/YukunZhou/RETFound_dinov2_meh) | `*_retfound_dinov2.yaml` |
| `retfound-green`  | ViT-S/14 with 4 register tokens, DINOv2 | `retfoundgreen_statedict.pth` ([RETFound-Green](https://github.com/justinengelmann/RETFound_Green) release v0.1) | `*_retfound_green.yaml` |

## RETFound-MAE and RETFound-DINOv2

Reference: the linear-probe path (`--adaptation lp`) of the official
[RETFound](https://github.com/rmaphoh/RETFound) code (`main_finetune.py`,
`models_vit.py`, `engine_finetune.py`, `util/`).

**Same as the official code:**

- **Architecture.** `retfound-mae` uses the hyperparameters of
  `models_vit.RETFound_mae`; `retfound-dinov2` is timm's
  `vit_large_patch14_dinov2.lvd142m`, as in `models_vit.RETFound_dinov2`.
- **Checkpoints.** Every backbone parameter is loaded (`checkpoint["model"]`
  for MAE, `checkpoint["teacher"]` without the `backbone.` prefix for
  DINOv2). DINOv2's 518x518 position embeddings are bicubically interpolated
  to 224x224, as in `util/pos_embed.py`.
- **Features.** RETFound-MAE mean-pools the patch tokens through `fc_norm`
  (`global_pool=True`), bit-identical to RETFound's own model class.
  RETFound-DINOv2 uses the CLS token after the pretrained final LayerNorm:
  `main_finetune.py` passes its `--global_pool` setting only when building
  RETFound-MAE; for `models_vit.RETFound_dinov2`, timm's default pooling,
  the CLS token, applies.
- **Head.** A single `nn.Linear`.
- **Preprocessing.** ImageNet mean/std and 224x224 input. Evaluation resizes
  to 256 and centre-crops to 224.
- **Training.** Drop path rate 0.2, active during head training. AdamW with
  `weight_decay=0.05`, 10 warmup epochs, then half-cycle cosine decay to
  `min_lr=1e-6` over 50 epochs, with `lr = blr * batch_size / 256`
  (`blr=5e-3`, `batch_size=24`).

**Differences:**

- **Augmentation.** This repository's shared pipeline (random resized crop,
  flips, Gaussian blur, rotation), used for every backbone. RETFound uses
  RandAugment, colour jitter and random erasing.
- **Layer-wise LR decay** is not implemented. With a frozen backbone it has
  no effect.

## RETFound-Green

Reference: the official
[RETFound-Green](https://github.com/justinengelmann/RETFound_Green) usage
example and linear probe.

**Same as the official code:**

- **Features.** The patch tokens are averaged after the pretrained final
  LayerNorm, excluding the CLS and register tokens, bit-identical to the
  official feature. Building the timm model with `global_pool="avg"` gives a
  different feature (cosine similarity 0.82), because timm then replaces the
  pretrained norm with an untrained `fc_norm`.
- **Preprocessing.** 392x392 input, mean/std `(0.5, 0.5, 0.5)`, no
  evaluation crop.
- **Head.** A single `nn.Linear`, without LayerNorm.
- **Drop path.** None, as in the official probe, which precomputes the
  embeddings.

**Differences:**

- **Head training.** AdamW, 200 epochs, `base_lr: 0.001`,
  `weight_decay: 5e-5`, best epoch by validation AUROC, as for the other
  backbones. The official probe fits an sklearn `LogisticRegression` on
  precomputed embeddings.
- **Augmentation.** This repository's shared pipeline.

`green_sklearn_probe.py` runs the official probe (`StandardScaler` fit on
train, L2 `LogisticRegression`, `max_iter=20000`, multinomial for
multi-class) on the frozen embeddings, with this repository's metrics and
bootstrap CIs. It reports a fit on train and a fit on train+val:

```bash
python green_sklearn_probe.py papila_retfound_green aptos_retfound_green
```

## Splits

All three baselines use this repository's split CSVs, so their results are
comparable with the other backbones here. Reproducing RETFound's published
numbers requires RETFound's own splits (set `data.csv_path`).

---

Back to the [main README](README.md).
