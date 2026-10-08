"""RETFound-Green's own linear-probe protocol: sklearn LogisticRegression on
precomputed embeddings (StandardScaler fit on train, L2 default, max_iter=20000,
multinomial for the 5-way DR grading), evaluated with this repo's own metrics.
"""
import os
import sys

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataset import CFPDataset, build_test_transform
from metrics import compute_metrics_with_ci, resolve_groups
from model import get_normalization_defaults
from train import build_model

# Relative csv_path/images_root in a config are relative to the repo root, like train.py.
ROOT = os.path.dirname(os.path.abspath(__file__))


def embed(cfg, split, device):
    mean, std = get_normalization_defaults(cfg.model.get("name"))
    tf = build_test_transform(cfg.model.img_size, mean=mean, std=std,
                              crop_pct=cfg.trainer.get("eval_crop_pct"))
    ds = CFPDataset(f"{ROOT}/{cfg.data.csv_path}", f"{ROOT}/{cfg.data.images_root}",
                    cfg.data.phenotype, split, tf)
    loader = DataLoader(ds, batch_size=64, num_workers=8, shuffle=False)
    model = embed.model
    feats, labels, ids = [], [], []
    with torch.inference_mode():
        for x, y, sample_ids in loader:
            feats.append(model.backbone(x.to(device)).cpu().numpy())
            labels.append(y.numpy())
            ids.extend(sample_ids.tolist() if torch.is_tensor(sample_ids) else list(sample_ids))
    return np.concatenate(feats), np.concatenate(labels), ids


def run(config_name):
    cfg = OmegaConf.create(yaml.safe_load(open(f"{ROOT}/configs/experiment/{config_name}.yaml")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed.model = build_model(cfg, ROOT, device).eval()

    xs = {s: embed(cfg, s, device) for s in ("t", "v", "e")}
    print(f"\n=== {config_name} ({cfg.data.csv_path.split('/')[-1]}) ===")
    print(f"embeddings: train {xs['t'][0].shape}, val {xs['v'][0].shape}, test {xs['e'][0].shape}")

    for name, fit_splits in (("train only (paper protocol)", ("t",)), ("train+val", ("t", "v"))):
        X = np.concatenate([xs[s][0] for s in fit_splits])
        y = np.concatenate([xs[s][1] for s in fit_splits])
        scaler = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=20000).fit(scaler.transform(X), y)
        prob = clf.predict_proba(scaler.transform(xs["e"][0]))
        groups = resolve_groups(f"{ROOT}/{cfg.data.csv_path}", ROOT, xs["e"][2])
        m = compute_metrics_with_ci(xs["e"][1], prob, cfg.data.num_classes, groups=groups)
        print(f"  fit on {name:28s} n={len(y):5d}  "
              f"auroc={m['auroc']:.4f} [{m['auroc_ci_lower']:.4f},{m['auroc_ci_upper']:.4f}]  "
              f"acc={m['acc']:.4f}  bal_acc={m['balanced_acc']:.4f}  f1={m['f1']:.4f}  "
              f"auprc={m['auprc']:.4f}")


for cfg_name in sys.argv[1:]:
    run(cfg_name)
